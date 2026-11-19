#pragma once

#include "server-common.h"

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>

struct server_conversation {
    std::string id;
    int64_t created_at = 0;
    json metadata = nullptr;
    json items = json::array();

    json to_json() const;
    static bool from_json(const json & data, server_conversation & out);
};

json server_conversation_normalize_item(const json & item);

struct server_conversations_store {
    mutable std::mutex mutex;
    std::unordered_map<std::string, server_conversation> items;
    std::string path;
    size_t max_items = 1024;
    int64_t ttl_seconds = 0;

    server_conversations_store() = default;
    server_conversations_store(const std::string & base_path, size_t max_items, int64_t ttl_seconds);

    server_conversation create(const json & metadata, const json & items);
    bool get(const std::string & id, server_conversation & out);
    bool update_metadata(const std::string & id, const json & metadata);
    bool erase(const std::string & id);
    bool add_items(const std::string & id, const json & items, json * added = nullptr);
    bool get_item(const std::string & id, const std::string & item_id, json & out);
    bool delete_item(const std::string & id, const std::string & item_id, server_conversation & out);
    bool list_items(
            const std::string & id,
            const std::string & after,
            int limit,
            const std::string & order,
            json & data,
            bool & has_more,
            json & first_id,
            json & last_id);

private:
    std::string file_path(const std::string & id) const;
    bool load_one_locked(const std::string & id, server_conversation & out);
    void save_locked(const server_conversation & conversation);
    void prune_locked();
};
