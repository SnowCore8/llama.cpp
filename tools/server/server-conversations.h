// OpenAI Conversations API (conversation objects + items), memory + optional disk
#pragma once

#include "common.h"

#include "server-common.h"

#include <cstdint>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>


struct server_conversation_entry {
    std::string id;
    int64_t     created_at = 0;
    int64_t     expires_at = 0;
    json        metadata   = json::object();
    json        items      = json::array(); // oldest first
};

class server_conversations_store {
  public:
    static server_conversations_store & instance();

    // root_path empty => memory-only; otherwise JSON files under root_path
    void configure(int32_t max_entries, int32_t ttl_seconds, const std::string & root_path = "");

    void put(server_conversation_entry entry);
    std::optional<server_conversation_entry> get(const std::string & id);
    bool erase(const std::string & id);
    void clear();

    size_t size();

  private:
    server_conversations_store() = default;

    void evict_expired_unlocked(int64_t now);
    void evict_lru_unlocked();
    json entry_to_json(const server_conversation_entry & e) const;
    bool entry_from_json(const json & j, server_conversation_entry & e) const;
    void persist_unlocked(const server_conversation_entry & e);
    void erase_disk_unlocked(const std::string & id);
    void load_from_disk_unlocked();
    bool try_load_id_unlocked(const std::string & id);
    bool reload_id_from_disk_unlocked(const std::string & id);

    mutable std::mutex mutex;
    int32_t max_entries_ = 1024;
    int32_t ttl_seconds_ = 7 * 24 * 3600;
    uint64_t lru_clock_  = 1;
    std::string root_path_;
    std::unordered_map<std::string, server_conversation_entry> entries;
    std::unordered_map<std::string, uint64_t> lru_stamp;
};

// conversation endpoints report bad input (400) and missing objects (404) separately
struct server_conversations_error : public std::runtime_error {
    bool not_found;
    server_conversations_error(const std::string & msg, bool not_found_) :
        std::runtime_error(msg), not_found(not_found_) {}
};

// Conversation object { id, created_at, metadata, object }
json server_conversation_to_json(const server_conversation_entry & e);

// Assign missing id/type/status, normalize message content; throws on invalid items.
void server_conversation_item_normalize(json & item);

// Drop fields gated behind `include` (logprobs, search results, ...).
json server_conversation_item_filter(const json & item, const json & include);

// Parse the `include` query parameter (repeated or comma separated values).
json server_conversations_include_from_param(const std::string & raw);

// Handlers backend; throw server_conversations_error on failure.
json server_conversations_create(const json & body);
json server_conversations_get(const std::string & id);
json server_conversations_update(const std::string & id, const json & body);
json server_conversations_delete(const std::string & id);
json server_conversations_add_items(const std::string & id, const json & body);
json server_conversations_list_items(const std::string & id, const std::string & after,
                                     const std::string & order, int64_t limit, const json & include);
json server_conversations_get_item(const std::string & id, const std::string & item_id, const json & include);
json server_conversations_delete_item(const std::string & id, const std::string & item_id);

// Append a finished turn: the request's own input items, then the response output items.
// Returns false when the conversation no longer exists.
bool server_conversations_append_turn(const std::string & id, const json & input_items, const json & output_items);
