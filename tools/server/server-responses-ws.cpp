#include "server-responses-ws.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "server-common.h"
#include "server-responses.h"

// Limits from the official WebSocket mode guide.
static constexpr int     WS_MAX_NAMED_LANES    = 32;
static constexpr int     WS_MAX_IN_FLIGHT      = 16;
static constexpr int     WS_MAX_PENDING_STEERS = 32; // queued steering submissions per response
static constexpr int     WS_MAX_TARGETS        = 256; // cached responses per connection
static constexpr int64_t WS_LIFETIME_MS        = 60 * 60 * 1000;
static constexpr int     WS_POLL_MS            = 1000;

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

// Steering errors (response.steer.failed).
static const char * WS_MSG_STEER_NOT_FOUND =
    "The target response is not available on this WebSocket connection.";
static const char * WS_MSG_STEER_UNSUPPORTED =
    "This response does not support steering.";
static const char * WS_MSG_STEER_TOO_MANY =
    "Too much steering input is pending for this response. "
    "Wait for the automatic continuation before submitting more.";
static const char * WS_MSG_STEER_COMPLETED =
    "The response already completed and is no longer accepting steering input.";
static const char * WS_MSG_STEER_NOT_ACTIVE =
    "The response is no longer accepting steering input.";
static const char * WS_MSG_STEER_SUCC_FAIL =
    "We couldn't start the next response. Send this steering input again with response.create.";

// One queued steering submission. QUEUED waits for a carrier, CARRIED rides a
// successor that has not sent response.created yet, COMMITTED is fully applied.
enum ws_steer_state : int {
    WS_STEER_QUEUED = 0,
    WS_STEER_CARRIED,
    WS_STEER_COMMITTED,
    WS_STEER_RETURNED,
};

struct ws_steer {
    std::string id;
    json        input;               // original client input (string or item array)
    int         state = WS_STEER_QUEUED;
    bool        pending_sent = false; // response.steer.pending emitted (at most once)
};

// One response on this connection, built when response.created is forwarded.
struct ws_target {
    std::string resp_id;
    std::string lane;                // lane that created the response ("" = default)
    json        body;                // original create body (stream_id stripped)
    std::string status = "running";  // running | completed | incomplete | failed
    bool        steerable = true;    // no conversation and no automatic compaction
    std::deque<std::shared_ptr<ws_steer>> steers; // queued / carried submissions
    json        inject_items = json::array(); // accepted injections, applied by the successor
};

// One lane queue item: the create body plus any steering submissions it carries.
struct ws_payload {
    std::string body;
    std::string lane;
    std::string prev_id;                     // continuation parent, for cache eviction on failure
    std::shared_ptr<ws_target> carry_parent; // target that owns the carried steers
    std::vector<std::shared_ptr<ws_steer>> carry; // committed on response.created
};

// One lane: a FIFO queue of payloads plus its worker thread.
// The worker runs one payload at a time, so same-lane responses never overlap.
struct ws_lane {
    std::string name;              // lane key, empty for the implicit default lane
    std::deque<std::shared_ptr<ws_payload>> queue; // pending payloads, FIFO
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

    // steering state: index built from response.created, guarded by idx_mtx
    std::mutex idx_mtx;
    std::map<std::string, std::shared_ptr<ws_target>> targets;
    std::deque<std::string> target_order; // insertion order for FIFO eviction
    std::atomic<int64_t> event_seq{0}; // connection-level sequence number shared by steer and inject events

    // connection token for the connection-local store=false cache: stamped on every
    // create this session forwards, so only this connection can continue those ids
    std::string ws_token;

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

// Appends one payload to its lane queue and wakes the lane worker.
static void ws_lane_enqueue(ws_session & sess, const std::string & name, std::shared_ptr<ws_payload> payload) {
    {
        std::lock_guard<std::mutex> lock(sess.mtx);
        sess.lanes.at(name)->queue.push_back(std::move(payload));
    }
    sess.cv.notify_all();
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
                "previous_response_not_found", server_responses_previous_not_found_message(message),
                "previous_response_id", stream_id);
    }
    return ws_make_error(status, type, code, message, param, stream_id);
}

// The official error.type uses the HTTP error vocabulary; keep a body-provided
// type only when it already belongs to it, otherwise derive it from the status.
static std::string ws_official_error_type(int status, const std::string & type) {
    if (type == "invalid_request_error" || type == "authentication_error" ||
            type == "permission_error" || type == "not_found_error" || type == "server_error") {
        return type;
    }
    switch (status) {
        case 400: return "invalid_request_error";
        case 401: return "invalid_request_error";
        case 403: return "permission_error";
        case 404: return "not_found_error";
        default:  return "server_error";
    }
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
    return ws_make_error_from_failure(res.status, ws_official_error_type(res.status, type), code, message, param, stream_id);
}

// 32 lowercase hex characters from the shared generator.
static std::string ws_random_hex32() {
    static std::mutex mtx;
    static std::mt19937_64 gen(std::random_device{}());
    static const char * hex = "0123456789abcdef";
    std::lock_guard<std::mutex> lock(mtx);
    std::string out;
    out.reserve(32);
    for (int i = 0; i < 32; i++) {
        out.push_back(hex[gen() % 16]);
    }
    return out;
}

// Official steering ids: "steer_" plus 32 lowercase hex characters.
static std::string ws_steer_id() {
    return "steer_" + ws_random_hex32();
}

// Sends one response.steer.accepted event (connection-level sequence number).
static void ws_send_steer_accepted(ws_session & sess, const std::string & steer_id,
        const std::string & prev_id, const std::string & stream_id) {
    json ev = {
        {"type",            "response.steer.accepted"},
        {"sequence_number", sess.event_seq++},
        {"steer", json {
            {"id",                  steer_id},
            {"previous_response_id", prev_id},
        }},
    };
    if (!stream_id.empty()) {
        ev["stream_id"] = stream_id;
    }
    ws_send_json(sess, ev);
}

// Sends one response.steer.pending event (after the target's response.completed).
static void ws_send_steer_pending(ws_session & sess, const std::string & steer_id,
        const std::string & prev_id, const json & required_input, const std::string & stream_id) {
    json ev = {
        {"type",            "response.steer.pending"},
        {"sequence_number", sess.event_seq++},
        {"steer", json {
            {"id",                  steer_id},
            {"previous_response_id", prev_id},
        }},
        {"reason",          "waiting_for_required_input"},
        {"required_input",  required_input},
    };
    if (!stream_id.empty()) {
        ev["stream_id"] = stream_id;
    }
    ws_send_json(sess, ev);
}

// Sends one response.steer.failed event with the original input. A failure
// before the id is allocated omits steer.id.
static void ws_send_steer_failed(ws_session & sess, const json & input,
        const std::string & prev_id, const std::string & steer_id,
        const std::string & code, const std::string & message, const std::string & stream_id) {
    json steer = {
        {"input",               input},
        {"previous_response_id", prev_id},
    };
    if (!steer_id.empty()) {
        steer["id"] = steer_id;
    }
    json ev = {
        {"type",            "response.steer.failed"},
        {"sequence_number", sess.event_seq++},
        {"steer",           std::move(steer)},
        {"error", json {
            {"type",    "invalid_request_error"},
            {"code",    code},
            {"message", message},
        }},
    };
    if (!stream_id.empty()) {
        ev["stream_id"] = stream_id;
    }
    ws_send_json(sess, ev);
}

// Sends one response.inject.created event (the input was accepted for injection).
static void ws_send_inject_created(ws_session & sess, const std::string & response_id,
        const std::string & stream_id) {
    json ev = {
        {"type",            "response.inject.created"},
        {"response_id",     response_id},
        {"sequence_number", sess.event_seq++},
    };
    if (!stream_id.empty()) {
        ev["stream_id"] = stream_id;
    }
    ws_send_json(sess, ev);
}

// Sends one response.inject.failed event with the raw input echoed back; the
// official error object carries only the code and the message.
static void ws_send_inject_failed(ws_session & sess, const std::string & response_id,
        const json & input, const std::string & code, const std::string & message,
        const std::string & stream_id) {
    json ev = {
        {"type",            "response.inject.failed"},
        {"response_id",     response_id},
        {"input",           input},
        {"sequence_number", sess.event_seq++},
        {"error", json {
            {"code",    code},
            {"message", message},
        }},
    };
    if (!stream_id.empty()) {
        ev["stream_id"] = stream_id;
    }
    ws_send_json(sess, ev);
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
        // the SSE frame carries only the Responses error code: client-input
        // failures (invalid_prompt) map onto invalid_request_error, the rest
        // are internal failures
        const std::string code = in.contains("code") && in.at("code").is_string()
            ? in.at("code").get<std::string>()
            : std::string();
        json error_obj = {
            {"type",    code == "invalid_prompt" ? "invalid_request_error" : "server_error"},
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

// True when the finished output contains a client-owned function or custom tool call.
static bool ws_has_function_call(const json & response_obj) {
    if (!response_obj.is_object() || !response_obj.contains("output") ||
            !response_obj.at("output").is_array()) {
        return false;
    }
    for (const auto & item : response_obj.at("output")) {
        if (!item.is_object()) {
            continue;
        }
        const std::string type = json_value(item, "type", std::string());
        if (type == "function_call" || type == "custom_tool_call") {
            return true;
        }
    }
    return false;
}

// Required input stubs for response.steer.pending, one per client-owned tool call.
static json ws_required_input(const json & response_obj) {
    json out = json::array();
    if (!response_obj.is_object() || !response_obj.contains("output") ||
            !response_obj.at("output").is_array()) {
        return out;
    }
    for (const auto & item : response_obj.at("output")) {
        if (!item.is_object()) {
            continue;
        }
        const std::string type = json_value(item, "type", std::string());
        if (type == "function_call") {
            out.push_back(json {
                {"type",    "function_call_output"},
                {"call_id", json_value(item, "call_id", std::string())},
                {"name",    json_value(item, "name", std::string())},
            });
        } else if (type == "custom_tool_call") {
            // custom_tool_call_output carries no name
            out.push_back(json {
                {"type",    "custom_tool_call_output"},
                {"call_id", json_value(item, "call_id", std::string())},
            });
        }
    }
    return out;
}

// Validates one response.steer event and its input shape; false writes the
// invalid_input message. Only type/previous_response_id/input are accepted.
static bool ws_steer_validate(const json & ev, std::string & msg) {
    for (const auto & field : ev.items()) {
        const std::string & key = field.key();
        if (key != "type" && key != "previous_response_id" && key != "input") {
            if (key == "stream_id") {
                msg = "Do not send 'stream_id' on response.steer; the target response determines the WebSocket lane.";
            } else {
                msg = "The response.steer event accepts only 'type', 'previous_response_id', and 'input'.";
            }
            return false;
        }
    }
    if (json_value(ev, "previous_response_id", std::string()).empty()) {
        msg = "previous_response_id is required and must be a non-empty string.";
        return false;
    }
    if (!ev.contains("input")) {
        msg = "input is required.";
        return false;
    }
    const json & input = ev.at("input");
    if (input.is_string()) {
        return true;
    }
    if (!input.is_array() || input.size() == 0) {
        msg = "Steering input must be a string or a non-empty array of user messages.";
        return false;
    }
    for (const auto & item : input) {
        if (!item.is_object()) {
            msg = "Steering input items must be user messages.";
            return false;
        }
        for (const auto & field : item.items()) {
            const std::string & key = field.key();
            if (key != "type" && key != "role" && key != "content") {
                msg = "Steering messages may contain only 'type', 'role', and 'content'.";
                return false;
            }
        }
        if (json_value(item, "role", std::string()) != "user") {
            msg = "Steering accepts only messages with the 'user' role.";
            return false;
        }
        if (item.contains("type") && !item.at("type").is_null() &&
                json_value(item, "type", std::string()) != "message") {
            msg = "Steering messages must have type 'message'.";
            return false;
        }
        if (!item.contains("content")) {
            msg = "Steering messages require 'content'.";
            return false;
        }
        const json & content = item.at("content");
        if (content.is_string()) {
            continue;
        }
        if (!content.is_array()) {
            msg = "Steering content must be a string or an array of input parts.";
            return false;
        }
        for (const auto & part : content) {
            if (!part.is_object()) {
                msg = "Steering content parts must be objects.";
                return false;
            }
            const std::string ptype = json_value(part, "type", std::string());
            if (ptype != "input_text" && ptype != "input_image" && ptype != "input_file") {
                msg = "Steering content supports only input_text, input_image, and input_file parts.";
                return false;
            }
            if (ptype == "input_text" &&
                    (!part.contains("text") || !part.at("text").is_string())) {
                msg = "input_text parts require a string 'text' field.";
                return false;
            }
        }
    }
    return true;
}

// Steering is not supported for conversation-bound responses or automatic compaction.
static bool ws_steerable(const json & body) {
    if (!body.is_object()) {
        return true;
    }
    if (body.contains("conversation") && !body.at("conversation").is_null()) {
        return false;
    }
    if (body.contains("context_management") && body.at("context_management").is_array()) {
        for (const auto & cm : body.at("context_management")) {
            if (cm.is_object() && json_value(cm, "type", std::string()) == "compaction") {
                return false;
            }
        }
    }
    return true;
}

// Drops steer entries in the given state (committed or returned to the client).
static void ws_steers_erase(const std::shared_ptr<ws_target> & target, ws_steer_state state) {
    auto & q = target->steers;
    q.erase(std::remove_if(q.begin(), q.end(),
            [state](const std::shared_ptr<ws_steer> & s) { return s->state == state; }), q.end());
}

// Builds the per-response record when response.created is forwarded.
static std::shared_ptr<ws_target> ws_register_target(
        ws_session & sess, const std::string & lane, const std::string & resp_id,
        const std::string & create_body) {
    if (resp_id.empty()) {
        return nullptr;
    }
    auto target = std::make_shared<ws_target>();
    target->resp_id = resp_id;
    target->lane    = lane;
    try {
        target->body = json::parse(create_body);
    } catch (const std::exception &) {
        target->body = json::object();
    }
    target->steerable = ws_steerable(target->body);
    std::lock_guard<std::mutex> lock(sess.idx_mtx);
    sess.targets[resp_id] = target;
    sess.target_order.push_back(resp_id);
    // FIFO evict finished responses past the cap; a running target is never evicted
    while ((int) sess.targets.size() > WS_MAX_TARGETS) {
        bool evicted = false;
        size_t i = 0;
        while (i < sess.target_order.size()) {
            auto it = sess.targets.find(sess.target_order[i]);
            if (it == sess.targets.end()) {
                sess.target_order.erase(sess.target_order.begin() + i);
                continue;
            }
            if (it->second->status == "running") {
                i++;
                continue;
            }
            sess.targets.erase(it);
            sess.target_order.erase(sess.target_order.begin() + i);
            evicted = true;
            break;
        }
        if (!evicted) {
            break; // every remaining target is running
        }
    }
    return target;
}

// The successor's response.created is the commit point for the steers it carries.
static void ws_commit_carried(ws_session & sess, const ws_payload & payload) {
    if (payload.carry.empty() || !payload.carry_parent) {
        return;
    }
    std::lock_guard<std::mutex> lock(sess.idx_mtx);
    for (const auto & s : payload.carry) {
        if (s->state == WS_STEER_CARRIED) {
            s->state = WS_STEER_COMMITTED;
        }
    }
    ws_steers_erase(payload.carry_parent, WS_STEER_COMMITTED);
}

// A successor that never produced response.created returns its input to the client.
static void ws_fail_carried(ws_session & sess, const ws_payload & payload) {
    if (payload.carry.empty() || !payload.carry_parent) {
        return;
    }
    std::vector<std::shared_ptr<ws_steer>> failed;
    std::string parent_id;
    std::string lane;
    {
        std::lock_guard<std::mutex> lock(sess.idx_mtx);
        parent_id = payload.carry_parent->resp_id;
        lane      = payload.carry_parent->lane;
        for (const auto & s : payload.carry) {
            if (s->state == WS_STEER_CARRIED) {
                s->state = WS_STEER_RETURNED;
                failed.push_back(s);
            }
        }
        ws_steers_erase(payload.carry_parent, WS_STEER_RETURNED);
    }
    if (sess.closed) {
        return;
    }
    for (const auto & s : failed) {
        ws_send_steer_failed(sess, s->input, parent_id, s->id,
                "successor_creation_failed", WS_MSG_STEER_SUCC_FAIL, lane);
    }
}

// Starts the automatic successor on the target lane: the original request
// settings, continuation from the target, input = accepted injections first,
// then the queued steers in submission order.
static void ws_start_successor(
        ws_session & sess, const std::shared_ptr<ws_target> & target,
        const json & inject_items, const std::vector<std::shared_ptr<ws_steer>> & steers) {
    if (inject_items.empty() && steers.empty()) {
        return;
    }
    json body = target->body;
    body.erase("input");
    body.erase("previous_response_id");
    body.erase("stream_id");
    json input = json::array();
    for (const auto & item : inject_items) {
        input.push_back(item);
    }
    for (const auto & s : steers) {
        json norm = server_responses_normalize_input(s->input);
        for (auto & item : norm) {
            input.push_back(item);
        }
    }
    body["previous_response_id"] = target->resp_id;
    body["input"] = std::move(input);
    body["__oai_ws_local"] = sess.ws_token;

    auto payload = std::make_shared<ws_payload>();
    payload->body         = body.dump();
    payload->lane         = target->lane;
    payload->prev_id      = target->resp_id;
    payload->carry_parent = target;
    payload->carry        = steers;
    ws_lane_enqueue(sess, target->lane, std::move(payload));
}

// Applies a terminal event of one response: echoes nothing itself, but sends
// response.steer.pending / response.steer.failed and starts the successor.
static void ws_on_terminal(ws_session & sess, const std::shared_ptr<ws_target> & target,
        const std::string & event_type, const json & response_obj) {
    if (!target) {
        return;
    }
    // collect the action for every still queued submission under the index lock
    std::vector<std::shared_ptr<ws_steer>> pending_out;
    std::vector<std::shared_ptr<ws_steer>> failed_out;
    std::vector<std::shared_ptr<ws_steer>> carry_out;
    json inject_out = json::array();
    {
        std::lock_guard<std::mutex> lock(sess.idx_mtx);
        if (event_type == "response.failed") {
            target->status = "failed";
        } else if (event_type == "response.completed") {
            target->status = "completed";
        } else {
            target->status = "incomplete";
        }
        // the interrupt flag is not needed after the terminal event
        server_responses_steer_flag_consume(target->resp_id);

        // accepted injections ride the automatic successor; a failed target drops
        // them silently (inject.created only promised that they were accepted)
        if (target->status == "failed") {
            target->inject_items = json::array();
        } else {
            inject_out = std::move(target->inject_items);
            target->inject_items = json::array();
        }

        const bool completed_with_tools =
            target->status == "completed" && ws_has_function_call(response_obj);
        // an injected tool output is itself the continuation: it replaces the
        // pending handoff, and the queued steering input rides along
        const bool tools_pending = completed_with_tools && inject_out.empty();
        for (const auto & s : target->steers) {
            if (s->state != WS_STEER_QUEUED) {
                continue;
            }
            if (target->status == "failed") {
                s->state = WS_STEER_RETURNED;
                failed_out.push_back(s);
            } else if (tools_pending) {
                if (!s->pending_sent) {
                    s->pending_sent = true;
                    pending_out.push_back(s);
                }
            } else {
                s->state = WS_STEER_CARRIED;
                carry_out.push_back(s);
            }
        }
        ws_steers_erase(target, WS_STEER_RETURNED);
    }
    if (sess.closed) {
        return;
    }
    if (!pending_out.empty()) {
        // the client must fill these stubs and continue with an explicit response.create
        const json required_input = ws_required_input(response_obj);
        for (const auto & s : pending_out) {
            ws_send_steer_pending(sess, s->id, target->resp_id, required_input, target->lane);
        }
    }
    for (const auto & s : failed_out) {
        ws_send_steer_failed(sess, s->input, target->resp_id, s->id,
                "successor_creation_failed", WS_MSG_STEER_SUCC_FAIL, target->lane);
    }
    if (!carry_out.empty() || !inject_out.empty()) {
        ws_start_successor(sess, target, inject_out, carry_out);
    }
}

// Abnormal stream end: no terminal event was forwarded, so fail the response
// and return its queued steering input.
static void ws_fail_unfinished(ws_session & sess, const std::shared_ptr<ws_target> & target) {
    if (!target || sess.closed) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(sess.idx_mtx);
        if (target->status != "running") {
            return;
        }
    }
    ws_on_terminal(sess, target, "response.failed", json::object());
}

// An explicit continuation of a target prepends its still queued steering
// input, in submission order, to the request input. The steers become CARRIED
// by this create: the commit point stays its response.created, and a create
// that fails before that returns the input to the client.
static void ws_prepend_steers(ws_session & sess, json & ev, const std::shared_ptr<ws_payload> & payload) {
    const std::string prev_id = json_value(ev, "previous_response_id", std::string());
    if (prev_id.empty()) {
        return;
    }
    if (ev.contains("input") && !ev.at("input").is_null() &&
            !ev.at("input").is_string() && !ev.at("input").is_array()) {
        return; // invalid input stays invalid; the route reports it
    }
    std::vector<std::shared_ptr<ws_steer>> carried;
    std::shared_ptr<ws_target> target;
    {
        std::lock_guard<std::mutex> lock(sess.idx_mtx);
        auto it = sess.targets.find(prev_id);
        if (it == sess.targets.end()) {
            return;
        }
        target = it->second;
        for (const auto & s : target->steers) {
            if (s->state == WS_STEER_QUEUED) {
                s->state = WS_STEER_CARRIED;
                carried.push_back(s);
            }
        }
    }
    if (carried.empty()) {
        return;
    }
    json combined = json::array();
    for (const auto & s : carried) {
        json norm = server_responses_normalize_input(s->input);
        for (auto & item : norm) {
            combined.push_back(item);
        }
    }
    if (ev.contains("input") && !ev.at("input").is_null()) {
        json cur = server_responses_normalize_input(ev.at("input"));
        for (auto & item : cur) {
            combined.push_back(item);
        }
    }
    ev["input"] = std::move(combined);
    payload->carry_parent = target;
    payload->carry        = std::move(carried);
}

// Official rule: a same-lane continuation that fails (4xx/5xx) evicts the parent
// from the connection-local cache; a failed cross-lane fork keeps the shared parent.
static void ws_evict_parent_on_failure(ws_session & sess, const ws_payload & payload, int status) {
    if (payload.prev_id.empty() || status < 400) {
        return;
    }
    bool same_lane = false;
    {
        std::lock_guard<std::mutex> lock(sess.idx_mtx);
        auto it = sess.targets.find(payload.prev_id);
        if (it == sess.targets.end()) {
            return; // parent unknown (already evicted): nothing to drop
        }
        same_lane = it->second->lane == payload.lane;
    }
    if (same_lane) {
        server_responses_ws_local_cache_evict(payload.prev_id, sess.ws_token);
    }
}

// Streams one response to the socket: splits SSE blocks, injects the lane
// stream_id and calls on_complete() exactly once.
static void ws_forward_response(ws_session & sess, const std::shared_ptr<ws_payload> & payload, const server_http_res_ptr & res) {
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
                std::shared_ptr<ws_target> target = ws_register_target(sess, payload->lane,
                        json_value(body, "id", std::string()), payload->body);
                ws_commit_carried(sess, *payload);
                json ev = {
                    {"type",     "response.completed"},
                    {"response", body},
                };
                if (!payload->lane.empty()) {
                    ev["stream_id"] = payload->lane;
                }
                ws_send_json(sess, ev);
                if (target) {
                    ws_on_terminal(sess, target, "response.completed", body);
                }
            }
        } else {
            ws_send_json(sess, ws_error_from_response(*res, payload->lane));
            ws_evict_parent_on_failure(sess, *payload, res->status);
            ws_fail_carried(sess, *payload);
        }
        res->on_complete();
        return;
    }

    std::shared_ptr<ws_target> target;
    bool created_seen = false;
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
                const std::string etype = json_value(ev, "type", std::string());
                if (etype == "response.created") {
                    const json resp_obj = ev.contains("response") ? ev.at("response") : json::object();
                    target = ws_register_target(sess, payload->lane,
                            json_value(resp_obj, "id", std::string()), payload->body);
                    if (target) {
                        created_seen = true;
                        ws_commit_carried(sess, *payload);
                    }
                    sending = ws_send_json(sess, ws_prepare_event(ev, payload->lane));
                } else if (etype == "response.completed" || etype == "response.incomplete" ||
                           etype == "response.failed") {
                    const json resp_obj = ev.contains("response") ? ev.at("response") : json::object();
                    sending = ws_send_json(sess, ws_prepare_event(ev, payload->lane));
                    // steering actions follow the terminal frame
                    ws_on_terminal(sess, target, etype, resp_obj);
                } else {
                    sending = ws_send_json(sess, ws_prepare_event(ev, payload->lane));
                    if (etype == "error" && !created_seen) {
                        // request-level error before the response started: same as a 4xx return
                        ws_evict_parent_on_failure(sess, *payload, 400);
                    }
                }
            }
        }
    } catch (const std::exception & e) {
        ws_send_json(sess, ws_make_error_from_failure(500, "server_error", "", e.what(), "", payload->lane));
        if (!created_seen) {
            // the create failed before it started: same as a 5xx return
            ws_evict_parent_on_failure(sess, *payload, 500);
        }
    } catch (...) {
        ws_send_json(sess, ws_make_error(500, "server_error", "", "unknown error", "", payload->lane));
        if (!created_seen) {
            ws_evict_parent_on_failure(sess, *payload, 500);
        }
    }

    if (!created_seen) {
        ws_fail_carried(sess, *payload);
    } else {
        ws_fail_unfinished(sess, target);
    }
    res->on_complete();
}

// Runs one response.create payload: calls create_fn and forwards the stream.
// should_stop is bound to the session, so a disconnect aborts the generation.
static void ws_run_one(ws_session & sess, const std::shared_ptr<ws_payload> & payload) {
    std::function<bool()> should_stop = [&sess]() { return sess.closed.load(); };
    server_http_req req {
        {},
        sess.headers,
        "/v1/responses",
        "",
        payload->body,
        {},
        should_stop,
    };

    server_http_res_ptr res;
    try {
        res = sess.create_fn(req);
    } catch (const std::invalid_argument & e) {
        ws_send_json(sess, ws_make_error_from_failure(400, "invalid_request_error", "", e.what(), "", payload->lane));
        ws_evict_parent_on_failure(sess, *payload, 400);
        ws_fail_carried(sess, *payload);
        return;
    } catch (const std::exception & e) {
        ws_send_json(sess, ws_make_error_from_failure(500, "server_error", "", e.what(), "", payload->lane));
        ws_evict_parent_on_failure(sess, *payload, 500);
        ws_fail_carried(sess, *payload);
        return;
    }
    if (!res) {
        ws_send_json(sess, ws_make_error(500, "server_error", "", "internal error: empty response", "", payload->lane));
        ws_evict_parent_on_failure(sess, *payload, 500);
        ws_fail_carried(sess, *payload);
        return;
    }
    ws_forward_response(sess, payload, res);
}

// One worker per lane: runs queued payloads in FIFO order. A payload starts
// only when a connection-wide in-flight slot is free; a close drops the queue.
static void ws_lane_worker(const std::shared_ptr<ws_session> & sess_ptr, const std::shared_ptr<ws_lane> & lane) {
    ws_session & sess = *sess_ptr;
    try {
        for (;;) {
            std::shared_ptr<ws_payload> payload;
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
                payload = std::move(lane->queue.front());
                lane->queue.pop_front();
                sess.n_in_flight++;
            }
            try {
                ws_run_one(sess, payload);
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

// Handles one response.steer event: validates it, then queues the input or
// sends response.steer.failed. The lane comes from the target, not the client.
static void ws_handle_steer(const std::shared_ptr<ws_session> & sess_ptr, const json & ev) {
    ws_session & sess = *sess_ptr;
    const std::string prev_id = json_value(ev, "previous_response_id", std::string());
    const json input = ev.contains("input") ? ev.at("input") : json(nullptr);

    // steer.failed echoes stream_id when the target is available and came from a
    // named lane; resolve it before validation. This lookup is not authoritative,
    // the decision below re-checks under the same lock.
    std::string lane;
    if (!prev_id.empty()) {
        std::lock_guard<std::mutex> lock(sess.idx_mtx);
        auto it = sess.targets.find(prev_id);
        if (it != sess.targets.end()) {
            lane = it->second->lane;
        }
    }

    std::string err;
    if (!ws_steer_validate(ev, err)) {
        ws_send_steer_failed(sess, input, prev_id, "", "invalid_input", err, lane);
        return;
    }

    // decide under the index lock; sends happen after it is released
    bool accepted = false;
    std::string steer_id;
    std::string fail_code;
    std::string fail_msg;
    {
        std::lock_guard<std::mutex> lock(sess.idx_mtx);
        auto it = sess.targets.find(prev_id);
        if (it == sess.targets.end()) {
            lane.clear(); // target unavailable: no stream_id echo
            fail_code = "response_not_found";
            fail_msg  = WS_MSG_STEER_NOT_FOUND;
        } else {
            const std::shared_ptr<ws_target> & tgt = it->second;
            lane = tgt->lane;
            if (!tgt->steerable) {
                fail_code = "steering_not_supported";
                fail_msg  = WS_MSG_STEER_UNSUPPORTED;
            } else if (tgt->status == "completed") {
                fail_code = "response_already_completed";
                fail_msg  = WS_MSG_STEER_COMPLETED;
            } else if (tgt->status != "running") {
                fail_code = "response_not_active";
                fail_msg  = WS_MSG_STEER_NOT_ACTIVE;
            } else {
                int n_queued = 0;
                for (const auto & s : tgt->steers) {
                    if (s->state == WS_STEER_QUEUED) {
                        n_queued++;
                    }
                }
                if (n_queued >= WS_MAX_PENDING_STEERS) {
                    fail_code = "too_many_pending_steers";
                    fail_msg  = WS_MSG_STEER_TOO_MANY;
                } else {
                    auto steer = std::make_shared<ws_steer>();
                    steer->id    = ws_steer_id();
                    steer->input = input;
                    tgt->steers.push_back(steer);
                    // ask the decode loop to stop this response at the next token
                    server_responses_steer_flag_set(prev_id);
                    steer_id = steer->id;
                    accepted = true;
                }
            }
        }
    }

    if (accepted) {
        ws_send_steer_accepted(sess, steer_id, prev_id, lane);
    } else {
        ws_send_steer_failed(sess, input, prev_id, "", fail_code, fail_msg, lane);
    }
}

// Handles one beta response.inject event: a schema violation gets a generic
// error and closes the connection, otherwise the items are accepted into the
// running target response or inject.failed is sent.
static void ws_handle_inject(const std::shared_ptr<ws_session> & sess_ptr, const json & ev) {
    ws_session & sess = *sess_ptr;

    // malformed events close the connection, as the official guide requires
    std::string schema_err;
    std::string schema_param;
    if (!ev.contains("response_id") || !ev.at("response_id").is_string() ||
            ev.at("response_id").get<std::string>().empty()) {
        schema_err   = "response_id is required and must be a non-empty string.";
        schema_param = "response_id";
    } else if (!ev.contains("input") || !ev.at("input").is_array()) {
        schema_err   = "input is required and must be an array of input items.";
        schema_param = "input";
    } else if (ev.at("input").empty()) {
        schema_err   = "input must contain at least one input item.";
        schema_param = "input";
    } else {
        for (const auto & item : ev.at("input")) {
            if (!item.is_object()) {
                schema_err   = "input items must be objects.";
                schema_param = "input";
                break;
            }
        }
    }
    if (!schema_err.empty()) {
        ws_send_json(sess, ws_make_error(400, "invalid_request_error", "invalid_input",
                schema_err, schema_param, ""));
        ws_close_session(sess);
        return;
    }

    const std::string response_id = ev.at("response_id").get<std::string>();
    const json & input = ev.at("input");

    // decide under the index lock; sends happen after it is released. The lane
    // echo follows the steer rule: set when the target is available.
    bool accepted = false;
    std::string lane;
    std::string fail_code;
    std::string fail_msg;
    {
        std::lock_guard<std::mutex> lock(sess.idx_mtx);
        auto it = sess.targets.find(response_id);
        if (it == sess.targets.end()) {
            fail_code = "response_not_found";
            fail_msg  = "Response '" + response_id + "' not found.";
        } else {
            const std::shared_ptr<ws_target> & tgt = it->second;
            lane = tgt->lane;
            if (tgt->status != "running") {
                fail_code = "response_already_completed";
                fail_msg  = "Response '" + response_id + "' has already completed.";
            } else {
                for (const auto & item : input) {
                    tgt->inject_items.push_back(item);
                }
                accepted = true;
            }
        }
    }

    if (accepted) {
        ws_send_inject_created(sess, response_id, lane);
    } else {
        ws_send_inject_failed(sess, response_id, input, fail_code, fail_msg, lane);
    }
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
    if (typ == "response.steer") {
        ws_handle_steer(sess_ptr, ev);
        return;
    }
    if (typ == "response.inject") {
        ws_handle_inject(sess_ptr, ev);
        return;
    }
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
    // connection token: this create may continue a store=false response that only
    // this connection holds (see server_responses_prepare_request). Always
    // overwritten, so clients cannot stamp or forge the marker themselves.
    ev["__oai_ws_local"] = sess.ws_token;

    auto payload = std::make_shared<ws_payload>();
    payload->lane    = stream_id;
    payload->prev_id = json_value(ev, "previous_response_id", std::string());

    // a continuation of a steered target picks up its queued input; the steers
    // are carried until this create's response.created (or returned if it fails).
    // a warmup create (generate:false) has no model output and must not consume
    // them: they stay queued for the next real successor or explicit create
    const bool is_warmup = ev.contains("generate") && ev.at("generate").is_boolean() &&
        !ev.at("generate").get<bool>();
    if (!is_warmup) {
        ws_prepend_steers(sess, ev, payload);
    }

    payload->body = ev.dump();
    ws_lane_enqueue(sess, stream_id, std::move(payload));
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

    // connection token for the store=false continuation cache: stamped on every
    // create this session forwards; dropped with its cache entries when it ends
    sess->ws_token = "wstok_" + ws_random_hex32();
    server_responses_ws_token_register(sess->ws_token);

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
    {
        // drop pending steering interrupts: no successor will run on a closed socket
        std::lock_guard<std::mutex> lock(sess->idx_mtx);
        for (const auto & kv : sess->targets) {
            if (kv.second->status == "running") {
                server_responses_steer_flag_consume(kv.first);
            }
        }
    }
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
    // no cache reads or writes can follow: the token and its entries are dropped
    server_responses_ws_token_unregister(sess->ws_token);
}
