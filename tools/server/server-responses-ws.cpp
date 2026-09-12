#include "server-responses-ws.h"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

#include "server-common.h"

// Limits from the official WebSocket mode guide.
static constexpr int     WS_MAX_NAMED_LANES = 32;
static constexpr int     WS_MAX_IN_FLIGHT   = 16;
static constexpr int64_t WS_LIFETIME_MS     = 60 * 60 * 1000;
static constexpr int     WS_POLL_MS         = 1000;

// Official error texts, kept verbatim so clients can match on them.
static const char * WS_MSG_INVALID_STREAM_ID =
    "The 'stream_id' field must be a non-empty string with at most 256 characters "
    "and may only contain letters, numbers, underscores, hyphens, and periods.";
static const char * WS_MSG_STREAM_LIMIT =
    "This WebSocket connection has reached its maximum number of distinct stream IDs (32). "
    "Reuse an existing stream_id or open a new WebSocket connection.";
static const char * WS_MSG_CONN_LIMIT =
    "Responses websocket connection limit reached (60 minutes). "
    "Create a new websocket connection to continue.";

// One lane: a FIFO queue of response.create payloads plus its worker thread.
// The worker runs one payload at a time, so same-lane responses never overlap.
struct ws_lane {
    std::string name;              // lane key, empty for the implicit default lane
    std::deque<std::string> queue; // pending response.create bodies, FIFO
    std::thread worker;            // started on first use, joined when the session ends
};

// Connection-level state shared by the read thread and all lane workers.
struct ws_session {
    // inputs of server_responses_ws_run, copied so that workers can outlive the call frame
    std::function<server_http_res_ptr(const server_http_req &)> create_fn;
    std::function<bool(const std::string &)> send_text;
    std::map<std::string, std::string> headers;

    std::mutex mtx;                  // guards lanes, queues and the counters below
    std::condition_variable cv;      // new work, a freed in-flight slot or session close
    std::atomic<bool> closed{false}; // set on disconnect, send failure or lifetime limit
    int n_in_flight = 0;             // running responses, at most WS_MAX_IN_FLIGHT
    int n_named_lanes = 0;           // distinct named lanes, at most WS_MAX_NAMED_LANES
    std::map<std::string, std::shared_ptr<ws_lane>> lanes; // key "" is the default lane

    std::mutex send_mtx;             // serializes send_text across the read thread and workers

    bool send(const std::string & text) {
        std::lock_guard<std::mutex> lock(send_mtx);
        return send_text(text);
    }
};

// Stops all lane workers: no new sends, generation aborts at the next should_stop poll.
static void ws_close_session(ws_session & sess) {
    {
        std::lock_guard<std::mutex> lock(sess.mtx);
        sess.closed = true;
    }
    sess.cv.notify_all();
}

// Sends one JSON event. A failed send means the connection is gone.
static bool ws_send_json(ws_session & sess, const json & ev) {
    if (sess.closed) {
        return false;
    }
    if (!sess.send(ev.dump())) {
        ws_close_session(sess);
        return false;
    }
    return true;
}

// Builds the official nested error event. Empty code/param become null, and an
// empty stream_id omits the field, which marks connection-level errors.
static json ws_make_error(
        int status,
        const std::string & type,
        const std::string & code,
        const std::string & message,
        const std::string & param,
        const std::string & stream_id) {
    json error_obj = {
        {"type",    type.empty() ? std::string("server_error") : type},
        {"code",    code.empty() ? json(nullptr) : json(code)},
        {"message", message},
        {"param",   param.empty() ? json(nullptr) : json(param)},
    };
    json ev = {
        {"type",   "error"},
        {"status", status},
        {"error",  error_obj},
    };
    if (!stream_id.empty()) {
        ev["stream_id"] = stream_id;
    }
    return ev;
}

// Maps failure details to the official error. The store reports a missing
// previous response as a plain message, which needs its own code and param.
static json ws_make_error_from_failure(
        int status,
        const std::string & type,
        const std::string & code,
        const std::string & message,
        const std::string & param,
        const std::string & stream_id) {
    if (message.rfind("previous_response_id not found or expired", 0) == 0) {
        return ws_make_error(400, "invalid_request_error",
                "previous_response_not_found", message, "previous_response_id", stream_id);
    }
    return ws_make_error(status, type, code, message, param, stream_id);
}

// Converts a non-streaming error response body to the official nested error event.
static json ws_error_from_response(const server_http_res & res, const std::string & stream_id) {
    std::string message = "internal error";
    std::string type    = "server_error";
    std::string code;
    std::string param;
    try {
        const json body = json::parse(res.data);
        const json err  = body.is_object() && body.contains("error") ? body.at("error") : body;
        if (err.is_object()) {
            message = json_value(err, "message", message);
            type    = json_value(err, "type", type);
            param   = json_value(err, "param", param);
            if (err.contains("code") && err.at("code").is_string()) {
                code = err.at("code").get<std::string>();
            }
        } else if (err.is_string()) {
            message = err.get<std::string>();
        }
    } catch (const std::exception &) {
        if (!res.data.empty()) {
            message = res.data;
        }
    }
    return ws_make_error_from_failure(res.status, type, code, message, param, stream_id);
}

// True when the client stream_id is 1-256 chars of [A-Za-z0-9_.-].
static bool ws_valid_stream_id(const std::string & id) {
    if (id.empty() || id.size() > 256) {
        return false;
    }
    for (const unsigned char c : id) {
        const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                        (c >= '0' && c <= '9') || c == '_' || c == '-' || c == '.';
        if (!ok) {
            return false;
        }
    }
    return true;
}

// Extracts the data: payload of one SSE block, empty when the block has none.
static std::string ws_sse_data(const std::string & block) {
    std::string data;
    std::istringstream iss(block);
    std::string line;
    while (std::getline(iss, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.rfind("data:", 0) == 0) {
            data = line.substr(5);
            if (!data.empty() && data[0] == ' ') {
                data.erase(0, 1);
            }
        }
    }
    return data;
}

// Converts one SSE frame to a WS frame: flat error events become the official
// nested shape and every frame of a named lane echoes its stream_id.
static json ws_prepare_event(const json & in, const std::string & stream_id) {
    json out;
    if (json_value(in, "type", std::string()) == "error") {
        json error_obj = {
            {"type",    "server_error"},
            {"code",    in.contains("code") ? in.at("code") : json(nullptr)},
            {"message", json_value(in, "message", std::string())},
            {"param",   in.contains("param") ? in.at("param") : json(nullptr)},
        };
        out = {
            {"type",  "error"},
            {"error", error_obj},
        };
        if (in.contains("sequence_number")) {
            out["sequence_number"] = in.at("sequence_number");
        }
    } else {
        out = in;
    }
    if (!stream_id.empty()) {
        out["stream_id"] = stream_id;
    }
    return out;
}

// Streams one response to the socket: splits SSE blocks, injects the lane
// stream_id and calls on_complete() exactly once.
static void ws_forward_response(ws_session & sess, const std::string & stream_id, const server_http_res_ptr & res) {
    if (!res->is_stream()) {
        // the route always streams over WebSocket, keep a fallback for non-stream results
        if (res->status == 200) {
            if (!res->data.empty()) {
                json body;
                try {
                    body = json::parse(res->data);
                } catch (const std::exception & e) {
                    SRV_WRN("failed to parse non-stream response body: %s\n", e.what());
                    body = json::object();
                }
                json ev = {
                    {"type",     "response.completed"},
                    {"response", body},
                };
                if (!stream_id.empty()) {
                    ev["stream_id"] = stream_id;
                }
                ws_send_json(sess, ev);
            }
        } else {
            ws_send_json(sess, ws_error_from_response(*res, stream_id));
        }
        res->on_complete();
        return;
    }

    std::string leftover;
    std::string chunk;
    try {
        bool sending = true;
        while (sending && res->next(chunk)) {
            leftover += chunk;
            size_t pos = 0;
            while (sending && (pos = leftover.find("\n\n")) != std::string::npos) {
                const std::string block = leftover.substr(0, pos);
                leftover.erase(0, pos + 2);
                const std::string data = ws_sse_data(block);
                if (data.empty() || data == "[DONE]") {
                    continue;
                }
                json ev;
                try {
                    ev = json::parse(data);
                } catch (const std::exception & e) {
                    SRV_WRN("skipping malformed SSE frame: %s\n", e.what());
                    continue;
                }
                sending = ws_send_json(sess, ws_prepare_event(ev, stream_id));
            }
        }
    } catch (const std::exception & e) {
        ws_send_json(sess, ws_make_error_from_failure(500, "server_error", "", e.what(), "", stream_id));
    } catch (...) {
        ws_send_json(sess, ws_make_error(500, "server_error", "", "unknown error", "", stream_id));
    }
    res->on_complete();
}

// Runs one response.create payload: calls create_fn and forwards the stream.
// should_stop is bound to the session, so a disconnect aborts the generation.
static void ws_run_one(ws_session & sess, const std::string & stream_id, const std::string & body) {
    std::function<bool()> should_stop = [&sess]() { return sess.closed.load(); };
    server_http_req req {
        {},
        sess.headers,
        "/v1/responses",
        "",
        body,
        {},
        should_stop,
    };

    server_http_res_ptr res;
    try {
        res = sess.create_fn(req);
    } catch (const std::invalid_argument & e) {
        ws_send_json(sess, ws_make_error_from_failure(400, "invalid_request_error", "", e.what(), "", stream_id));
        return;
    } catch (const std::exception & e) {
        ws_send_json(sess, ws_make_error_from_failure(500, "server_error", "", e.what(), "", stream_id));
        return;
    }
    if (!res) {
        ws_send_json(sess, ws_make_error(500, "server_error", "", "internal error: empty response", "", stream_id));
        return;
    }
    ws_forward_response(sess, stream_id, res);
}

// One worker per lane: runs queued payloads in FIFO order. A payload starts
// only when a connection-wide in-flight slot is free; a close drops the queue.
static void ws_lane_worker(const std::shared_ptr<ws_session> & sess_ptr, const std::shared_ptr<ws_lane> & lane) {
    ws_session & sess = *sess_ptr;
    try {
        for (;;) {
            std::string body;
            {
                std::unique_lock<std::mutex> lock(sess.mtx);
                for (;;) {
                    if (sess.closed) {
                        return; // drop pending payloads, the connection is gone
                    }
                    if (!lane->queue.empty() && sess.n_in_flight < WS_MAX_IN_FLIGHT) {
                        break;
                    }
                    sess.cv.wait(lock);
                }
                body = std::move(lane->queue.front());
                lane->queue.pop_front();
                sess.n_in_flight++;
            }
            try {
                ws_run_one(sess, lane->name, body);
            } catch (const std::exception & e) {
                SRV_ERR("response run failed: %s\n", e.what());
            } catch (...) {
                SRV_ERR("response run failed: %s\n", "unknown error");
            }
            {
                std::lock_guard<std::mutex> lock(sess.mtx);
                sess.n_in_flight--;
            }
            sess.cv.notify_all();
        }
    } catch (const std::exception & e) {
        SRV_ERR("lane worker stopped: %s\n", e.what());
    } catch (...) {
        SRV_ERR("lane worker stopped: %s\n", "unknown error");
    }
}

// Registers a lane on first use and starts its worker. Returns false when the
// named lane limit is reached; the default lane does not count toward the limit.
static bool ws_lane_register(const std::shared_ptr<ws_session> & sess_ptr, const std::string & name) {
    ws_session & sess = *sess_ptr;
    std::lock_guard<std::mutex> lock(sess.mtx);
    if (sess.lanes.count(name)) {
        return true;
    }
    if (!name.empty()) {
        if (sess.n_named_lanes >= WS_MAX_NAMED_LANES) {
            return false;
        }
        sess.n_named_lanes++;
    }
    auto lane = std::make_shared<ws_lane>();
    lane->name = name;
    sess.lanes[name] = lane;
    lane->worker = std::thread(ws_lane_worker, sess_ptr, lane);
    return true;
}

// Appends one payload to its lane queue and wakes the lane worker.
static void ws_lane_enqueue(ws_session & sess, const std::string & name, std::string body) {
    {
        std::lock_guard<std::mutex> lock(sess.mtx);
        sess.lanes.at(name)->queue.push_back(std::move(body));
    }
    sess.cv.notify_all();
}

// Handles one client text frame. Invalid events get an error event and do not
// stop the session.
static void ws_handle_message(const std::shared_ptr<ws_session> & sess_ptr, const std::string & msg) {
    ws_session & sess = *sess_ptr;
    json ev;
    try {
        ev = json::parse(msg);
    } catch (const std::exception & e) {
        ws_send_json(sess, ws_make_error(400, "invalid_request_error", "invalid_json",
                std::string("invalid json: ") + e.what(), "", ""));
        return;
    }
    if (!ev.is_object()) {
        ws_send_json(sess, ws_make_error(400, "invalid_request_error", "invalid_json",
                "invalid json: expected an event object", "", ""));
        return;
    }
    const std::string typ = json_value(ev, "type", std::string());
    if (typ != "response.create") {
        ws_send_json(sess, ws_make_error(400, "invalid_request_error", "unsupported_event_type",
                "unsupported event type: " + (typ.empty() ? std::string("<missing>") : typ), "type", ""));
        return;
    }

    // stream_id names the lane; it is not part of POST /v1/responses
    std::string stream_id;
    if (ev.contains("stream_id")) {
        if (!ev.at("stream_id").is_string()) {
            ws_send_json(sess, ws_make_error(400, "invalid_request_error", "invalid_stream_id",
                    WS_MSG_INVALID_STREAM_ID, "stream_id", ""));
            return;
        }
        stream_id = ev.at("stream_id").get<std::string>();
        if (!ws_valid_stream_id(stream_id)) {
            ws_send_json(sess, ws_make_error(400, "invalid_request_error", "invalid_stream_id",
                    WS_MSG_INVALID_STREAM_ID, "stream_id", ""));
            return;
        }
        ev.erase("stream_id");
    }

    // a named lane is registered when the create is accepted, whatever its outcome
    if (!ws_lane_register(sess_ptr, stream_id)) {
        ws_send_json(sess, ws_make_error(400, "invalid_request_error", "websocket_stream_limit_reached",
                WS_MSG_STREAM_LIMIT, "stream_id", stream_id));
        return;
    }

    // stream is implicit over WebSocket and background is not supported there
    ev.erase("type");
    ev.erase("stream");
    ev.erase("background");
    ev["stream"] = true;

    ws_lane_enqueue(sess, stream_id, ev.dump());
}

void server_responses_ws_run(
        const std::function<server_http_res_ptr(const server_http_req &)> & create_fn,
        const std::map<std::string, std::string> & headers,
        const std::function<int(std::string &, int)> & poll_text,
        const std::function<bool(const std::string &)> & send_text) {
    auto sess = std::make_shared<ws_session>();
    sess->create_fn = create_fn;
    sess->send_text = send_text;
    sess->headers   = headers;

    const int64_t t_start = ggml_time_ms();

    for (;;) {
        if (sess->closed) {
            break; // a worker failed to send, the connection is gone
        }
        if (ggml_time_ms() - t_start >= WS_LIFETIME_MS) {
            // connection lifetime limit reached: tell the client, then close
            ws_send_json(*sess, ws_make_error(400, "invalid_request_error",
                    "websocket_connection_limit_reached", WS_MSG_CONN_LIMIT, "", ""));
            break;
        }
        std::string msg;
        const int poll = poll_text(msg, WS_POLL_MS);
        if (poll < 0) {
            break; // peer closed the connection
        }
        if (poll == 0) {
            continue; // timeout tick, loop back to the lifetime check
        }
        try {
            ws_handle_message(sess, msg);
        } catch (const std::exception & e) {
            SRV_WRN("failed to handle client event: %s\n", e.what());
            ws_send_json(*sess, ws_make_error(500, "server_error", "", e.what(), "", ""));
        }
    }

    // stop the workers (their should_stop aborts running generations), then join them
    ws_close_session(*sess);
    std::vector<std::shared_ptr<ws_lane>> lanes;
    {
        std::lock_guard<std::mutex> lock(sess->mtx);
        for (auto & kv : sess->lanes) {
            lanes.push_back(kv.second);
        }
    }
    for (auto & lane : lanes) {
        if (lane->worker.joinable()) {
            lane->worker.join();
        }
    }
}
