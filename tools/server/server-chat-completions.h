#pragma once

#include "server-common.h"

#include "json.h"

#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// OpenAI Chat Completions request fields the native response layer needs
struct server_chat_completions_request {
    std::string model;
    json messages = json::array();
    json metadata = nullptr;
    std::string user;
    std::string safety_identifier;
    std::string service_tier;
    bool store = false;
    bool parallel_tool_calls = true;
    int n = 1;
    std::vector<std::string> custom_tools; // names of the requested type=custom tools
};

struct server_chat_completions_stored {
    json response = nullptr;
    json request_messages = json::array();
    json messages = json::array();
    json metadata = nullptr;
    std::string user;
    std::string safety_identifier;
    int64_t created_at = 0;

    json to_json() const;
    static bool from_json(const json & data, server_chat_completions_stored & out);
};

// in-memory store for stored chat completions
struct server_chat_completions_store {
    mutable std::mutex mutex;
    std::unordered_map<std::string, server_chat_completions_stored> items;
    std::string path;
    size_t max_items = 1024;
    int64_t ttl_seconds = 0;

    server_chat_completions_store() = default;
    server_chat_completions_store(const std::string & base_path, size_t max_items, int64_t ttl_seconds);

    bool get(const std::string & id, server_chat_completions_stored & out);
    void put(const std::string & id, server_chat_completions_stored item);
    bool erase(const std::string & id);
    bool update_metadata(const std::string & id, const json & metadata);
    std::vector<std::pair<std::string, server_chat_completions_stored>> list();

private:
    std::string file_path(const std::string & id) const;
    bool load_one_locked(const std::string & id, server_chat_completions_stored & out);
    void prune_locked();
};

server_chat_completions_request server_chat_completions_parse_request(const json & body);

// rewrite named / allowed tool choices into the shape understood by the shared parser
void server_chat_completions_normalize_request(json & body);

struct server_chat_completions_context {
    server_chat_completions_context(
            server_chat_completions_store & store,
            server_chat_completions_request req);

    // non-stream response: decorate fields and store if requested
    void finish(json & response);
    // streaming: one final choice at a time, store once all choices are present
    void finish_stream(json response);
    // rewrite custom tool calls and enforce parallel_tool_calls=false on streamed chunks
    json filter_chunks(json chunks);

private:
    struct pending_stream {
        json response = nullptr;
        std::unordered_map<int, json> choices;
        int64_t prompt_tokens = 0;
        int64_t completion_tokens = 0;
    };

    // streamed custom tool call, keyed by (choice index, tool call index)
    struct pending_custom_tool_call {
        std::string id;
        std::string name;
        std::string arguments;
    };

    server_chat_completions_store & store;
    server_chat_completions_request req;
    std::unordered_map<std::string, pending_stream> pending;
    std::map<std::pair<int, int>, pending_custom_tool_call> pending_custom_tool_calls;

    bool is_custom_tool(const std::string & name) const;
    void decorate(json & response) const;
    void store_response(const json & response);
    json make_messages(const json & response) const;
};
