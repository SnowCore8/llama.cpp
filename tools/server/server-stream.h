#pragma once

#include "server-http.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

// streaming buffer for one generation, survives HTTP disconnect. the producer appends SSE bytes,
// readers drain from any offset via read_from. keyed by conversation_id, one conv = one live session

struct stream_session;

using stream_session_ptr = std::shared_ptr<stream_session>;

// base of the producer/consumer pipe ends. virtual dtor so each runs its own teardown:
// the producer finalizes the session, the consumer leaves it untouched
struct stream_pipe {
    virtual ~stream_pipe() = default;

    bool is_cancelled() const;

protected:
    explicit stream_pipe(stream_session_ptr session);

    stream_session_ptr session_;
};

// producer end: writes chunks into the ring buffer and owns the session lifetime, finalizing it
// on destruction.
struct stream_pipe_producer : stream_pipe {
    ~stream_pipe_producer() override;

    bool write(const char * data, size_t len);

    static stream_pipe_producer * create(stream_session_ptr session);

private:
    explicit stream_pipe_producer(stream_session_ptr session);
};

void server_stream_session_manager_start();
void server_stream_session_manager_stop();

// route handler factories wired under /v1/stream/* by server.cpp
// child-side handlers for the resumable stream routes. the conv id travels in the conv_id
// query string because it can embed a model name containing slashes (org/repo), which the
// decoded path would split before the param is captured
server_http_context::handler_t server_stream_make_get_handler();
// POST /v1/streams/lookup with body {"conversation_ids": [...]}: only answers for ids the
// caller already owns (the WebUI passes the convs visible in its sidebar), the server never
// lists ids it has not been asked about, so a random caller cannot enumerate live sessions
server_http_context::handler_t server_stream_make_lookup_handler();
server_http_context::handler_t server_stream_make_delete_handler();

// extract the X-Conversation-Id header value (case-insensitive), empty when absent
std::string server_stream_conv_id_from_headers(const std::map<std::string, std::string> & headers);

// cancel the resumable stream attached to a response id: flips cancellation so the producer
// stops at its next poll and wakes any blocked readers. no-op when the id has no live session
void server_stream_cancel_response(const std::string & response_id);

// resume status for GET /v1/responses/{id}?stream=true[&starting_after=N]
enum server_stream_resume_status {
    SERVER_STREAM_RESUME_OK = 0,
    SERVER_STREAM_RESUME_NOT_FOUND,
    SERVER_STREAM_RESUME_OFFSET_LOST,
};

// build an SSE producer that replays a stored response stream and follows live bytes,
// forwarding only events with sequence_number > starting_after (negative forwards all).
// without a cursor a partial replay is fine: it starts at the oldest buffered event.
// next_seq stamps the terminal error event sent when a following reader falls behind.
// returns NOT_FOUND when no session exists, OFFSET_LOST when the replay prefix was dropped
server_stream_resume_status server_stream_make_response_resume(
        const std::string & response_id,
        int64_t starting_after,
        const std::function<bool()> & should_stop,
        const std::function<int64_t()> & next_seq,
        std::function<bool(std::string &)> & next);

// implement tee-style pipe (spipe) for "stream replay" functionality
struct server_res_spipe : server_http_res {
private:
    // if set, the stream survives a client disconnect:
    // connection kept alive, output is forwarded to spipe and reuse later
    std::unique_ptr<stream_pipe_producer> spipe;
    // if spipe is set, use this next_orig to implement tee-style pipe
    std::function<bool(std::string &)> next_orig;
    const server_http_req * req = nullptr;
    // session key once attached (X-Conversation-Id header or an explicit id)
    std::string conv_id;
    // set once next_orig reports no more data, so on_complete() doesn't re-drain a finished stream
    bool next_finished = false;

public:
    void set_req(const server_http_req * req);
    // attach a resumable session keyed by an explicit id: background Responses use the
    // response id in place of a conversation header so clients can reattach by cursor
    void attach_conv_id(const std::string & conv_id);
    bool conn_alive();
    bool should_stop();
    void on_complete() override;
    void set_next(std::function<bool(std::string &)> next_fn);
};
