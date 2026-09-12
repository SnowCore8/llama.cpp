// OpenAI Chat Completions store — memory + optional disk
#pragma once

#include "common.h"

#include "server-common.h"

#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>


struct server_chat_completions_store_entry {
    std::string id;
    int64_t     created   = 0;
    int64_t     expires_at = 0;
    std::string model;
    json        metadata  = nullptr;
    json        completion = json::object();
};

class server_chat_completions_store {
  public:
    static server_chat_completions_store & instance();

    void configure(int32_t max_entries, int32_t ttl_seconds, const std::string & root_path = "");

    void put(server_chat_completions_store_entry entry);
    std::optional<server_chat_completions_store_entry> get(const std::string & id);
    std::optional<json> update_metadata(const std::string & id, const json & metadata);
    bool erase(const std::string & id);
    json list(const std::string & after, int limit, bool order_desc,
              const std::string & model_filter,
              const std::vector<std::pair<std::string, std::string>> & metadata_filter);
    void clear();

    int32_t max_entries() const;
    size_t  size();

  private:
    server_chat_completions_store() = default;

    void evict_expired_unlocked(int64_t now);
    void evict_lru_unlocked();
    json entry_to_json(const server_chat_completions_store_entry & e) const;
    bool entry_from_json(const json & j, server_chat_completions_store_entry & e) const;
    void persist_unlocked(const server_chat_completions_store_entry & e);
    void erase_disk_unlocked(const std::string & id);
    void load_from_disk_unlocked();
    bool try_load_id_unlocked(const std::string & id);
    void sync_new_from_disk_unlocked();

    mutable std::mutex mutex;
    int32_t max_entries_ = 1024;
    int32_t ttl_seconds_ = 7 * 24 * 3600;
    uint64_t lru_clock_  = 1;
    std::string root_path_;
    std::unordered_map<std::string, server_chat_completions_store_entry> entries;
    std::unordered_map<std::string, uint64_t> lru_stamp;
};

void server_chat_completions_remember(const json & completion_obj);
