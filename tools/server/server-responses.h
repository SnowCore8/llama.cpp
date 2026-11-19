#pragma once

#include "server-common.h"
#include "server-task.h"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// OpenAI Responses API request, parsed without going through chat completions
struct server_responses_request {
    std::string model;
    json instructions = nullptr;
    json input = nullptr;
    json input_items = json::array();
    json effective_input = json::array();
    json metadata = nullptr;
    json reasoning = nullptr;
    json text = nullptr;
    json include = nullptr;
    json tools = json::array();
    json tool_choice = "auto";
    std::vector<common_chat_tool> tools_parsed;
    common_chat_tool_choice tool_choice_parsed = COMMON_CHAT_TOOL_CHOICE_AUTO;
    std::string tool_choice_name;
    bool stream = false;
    bool store = true;
    bool timings_per_token = false;
    bool return_progress = false;
    int32_t id_slot = -1;
    bool background = false;
    bool parallel_tool_calls = true;
    std::string previous_response_id;
    std::string conversation_id;
    std::string truncation = "disabled";
    std::string user;
    std::string safety_identifier;
    std::string service_tier;
    int32_t max_output_tokens = -1;
    int32_t top_logprobs = 0;
    bool include_logprobs = false;
    double temperature = -1.0;
    double top_p = -1.0;
};

struct server_responses_stored {
    json response = nullptr;
    json input_items = json::array();
    int64_t created_at = 0;

    json to_json() const;
    static bool from_json(const json & data, server_responses_stored & out);
};

// in-memory store for response objects and their input items
struct server_responses_store {
    mutable std::mutex mutex;
    std::unordered_map<std::string, server_responses_stored> items;
    std::string path;
    size_t max_items = 1024;
    int64_t ttl_seconds = 0;

    server_responses_store() = default;
    server_responses_store(const std::string & base_path, size_t max_items, int64_t ttl_seconds);

    bool get(const std::string & id, server_responses_stored & out);
    void put(const std::string & id, server_responses_stored item);
    bool erase(const std::string & id);

private:
    std::string file_path(const std::string & id) const;
    void prune_locked();
};

// throws std::invalid_argument on invalid request
server_responses_request server_responses_parse_request(const json & body);

struct server_conversations_store;

// resolve previous_response_id / conversation and build effective_input, throws on unknown id
void server_responses_resolve_input(server_responses_request & req, server_responses_store & store, server_conversations_store * conversations = nullptr);

// build prompt and sampling parameters for the engine
json server_responses_build_params(
        const llama_vocab * vocab,
        const server_chat_params & opt,
        const server_responses_request & req,
        common_chat_session & session,
        std::vector<raw_buffer> & files);

// rendered SSE events of one background response, kept so that a new connection can resume
// the stream from a sequence number. the generation thread is the only writer
struct server_responses_event_log {
    mutable std::mutex mutex;
    std::condition_variable cv;
    // sequence_number -> rendered SSE bytes, sequence numbers are contiguous
    std::deque<std::pair<int64_t, std::string>> events;
    int64_t first_seq = 0; // sequence number of the oldest retained event, next expected when empty
    bool done = false;
    size_t n_bytes = 0;
    size_t max_bytes = 4 * 1024 * 1024;

    // append one event, drop the oldest ones once the log grows past max_bytes
    void append(int64_t seq, std::string event);
    // no more events will be appended
    void close();
    bool is_done() const;
    // true if a reader starting at next_seq has not missed any retained event
    bool validate(int64_t next_seq) const;
    // block until an event at or after next_seq exists, copy everything available to out and
    // advance next_seq. returns false when the log is done and drained, when the reader fell
    // behind the retained window, or when should_stop() fires
    bool read(int64_t & next_seq, std::string & out, const std::function<bool()> & should_stop);
};

// per-request response state, shared by the non-stream and stream paths
struct server_responses_context {
    server_responses_context(
            server_responses_store & store,
            server_responses_request req,
            std::string model_name,
            const llama_vocab * vocab,
            server_conversations_store * conversations = nullptr);

    const std::string & get_id() const { return id; }

    // resumable SSE event log, only set for background responses
    std::shared_ptr<server_responses_event_log> event_log;

    // SSE events for one partial result
    json consume(const server_task_result_cmpl_partial & result);
    // SSE events for the final result, including the terminal event
    json finish(const server_task_result_cmpl_final & result);
    // non-stream response object
    json final_response(const server_task_result_cmpl_final & result);
    // SSE error event
    json failed_event(const std::string & message, const std::string & code);

    // background lifecycle
    json begin_background();
    json cancel();
    void fail(const std::string & message);
    // emit the terminal event for a cancelled background response and close the event log
    void finish_cancelled();
    // close the event log, no-op when there is none
    void close_events() {
        if (event_log) {
            event_log->close();
        }
    }

private:
    struct text_item {
        std::string id;
        size_t output_index = 0;
        bool added = false;
        bool part_added = false;
        bool done = false;
        std::string text;
    };

    struct tool_item {
        std::string item_id;
        std::string call_id;
        std::string name;
        std::string arguments;
        size_t output_index = 0;
        bool added = false;
        bool done = false;
    };

    server_responses_store & store;
    server_responses_request req;
    std::string model_name;
    const llama_vocab * vocab;
    server_conversations_store * conversations;
    std::string id;
    int64_t created_at = 0;
    mutable size_t seq = 0;
    size_t next_output_index = 0;
    bool started = false;
    bool reasoning_summary = false;
    std::vector<json> text_logprobs;
    json message_logprobs = json::array();
    text_item reasoning;
    text_item message;
    std::vector<tool_item> tools;
    std::atomic<bool> cancelled{false};
    std::atomic<bool> stored{false};
    bool conversation_appended = false;
    std::mutex cancel_mutex;
    json cancelled_response = nullptr;

    json push(json & events, const std::string & type, json data) const;
    // render events into the resumable log, no-op when the log is not enabled
    void publish(const json & events) const;
    void ensure_started(json & events);
    void apply_diffs(json & events, const std::vector<common_chat_msg_diff> & diffs);
    void add_reasoning_delta(json & events, const std::string & delta);
    void add_text_delta(json & events, const std::string & delta);
    void add_tool_delta(json & events, size_t index, const common_chat_tool_call & delta);
    void finalize_items(json & events);

    json make_response(
            const std::string & status,
            const json & output,
            const json & usage,
            const json & error,
            const json & incomplete_details) const;
    json usage_from(const server_task_result_cmpl_final & result, const std::string & reasoning_text) const;
    json output_from_msg(const common_chat_msg & msg) const;
    json output_from_stream(size_t * n_reasoning = nullptr) const;
    json message_item(const std::string & item_id, const std::string & text, const std::string & status) const;
    json reasoning_item(const std::string & item_id, const std::string & text, const std::string & status) const;
    json function_call_item(
            const std::string & item_id,
            const std::string & call_id,
            const std::string & name,
            const std::string & arguments,
            const std::string & status) const;
    void store_response(const json & response);
};
