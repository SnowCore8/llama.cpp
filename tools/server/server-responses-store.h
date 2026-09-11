// OpenAI Responses API store (previous_response_id) — memory + optional disk
#pragma once

#include "common.h"

#include "server-common.h"

#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>


struct server_responses_store_entry {
    std::string id;
    int64_t     created_at  = 0;
    int64_t     expires_at  = 0;
    std::string model;
    json        instructions = nullptr; // string or null
    json        input        = json::array();
    json        output       = json::array();
    json        usage        = json::object();
    json        response     = json::object(); // full response object
};

class server_responses_store {
  public:
    static server_responses_store & instance();

    // root_path empty => memory-only; otherwise JSON files under root_path
    void configure(int32_t max_entries, int32_t ttl_seconds, const std::string & root_path = "");

    void put(server_responses_store_entry entry);
    std::optional<server_responses_store_entry> get(const std::string & id);
    bool erase(const std::string & id);
    void clear();

    int32_t max_entries() const;
    size_t  size();

  private:
    server_responses_store() = default;

    void evict_expired_unlocked(int64_t now);
    void evict_lru_unlocked();
    json entry_to_json(const server_responses_store_entry & e) const;
    bool entry_from_json(const json & j, server_responses_store_entry & e) const;
    void persist_unlocked(const server_responses_store_entry & e);
    void erase_disk_unlocked(const std::string & id);
    void load_from_disk_unlocked();
    // load one id from disk into memory (router parent vs child share path)
    bool try_load_id_unlocked(const std::string & id);
    // Re-read disk; do not treat in_progress as restart interrupt (other process may own it).
    bool reload_id_from_disk_unlocked(const std::string & id);
    void finalize_interrupted_unlocked(server_responses_store_entry & e);

    mutable std::mutex mutex;
    int32_t max_entries_ = 1024;
    int32_t ttl_seconds_ = 7 * 24 * 3600;
    uint64_t lru_clock_  = 1;
    std::string root_path_;
    std::unordered_map<std::string, server_responses_store_entry> entries;
    std::unordered_map<std::string, uint64_t> lru_stamp;
};
