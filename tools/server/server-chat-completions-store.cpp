#include "server-chat-completions-store.h"
#include "server-openai-persist.h"

#include "server-common.h"
#include "log.h"

#include <algorithm>
#include <chrono>

static int64_t chat_cmpl_now_unix() {
    return (int64_t) std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

server_chat_completions_store & server_chat_completions_store::instance() {
    static server_chat_completions_store inst;
    return inst;
}

json server_chat_completions_store::entry_to_json(const server_chat_completions_store_entry & e) const {
    return json{
        {"id",         e.id},
        {"created",    e.created},
        {"expires_at", e.expires_at},
        {"model",      e.model},
        {"metadata",   e.metadata},
        {"completion", e.completion},
    };
}

bool server_chat_completions_store::entry_from_json(const json & j, server_chat_completions_store_entry & e) const {
    if (!j.contains("id") || !j.at("id").is_string()) {
        return false;
    }
    e.id         = j.at("id").get<std::string>();
    e.created    = j.value("created", (int64_t) 0);
    e.expires_at = j.value("expires_at", (int64_t) 0);
    e.model      = j.value("model", std::string());
    e.metadata   = j.contains("metadata") ? j.at("metadata") : json(nullptr);
    e.completion = j.contains("completion") ? j.at("completion") : json::object();
    return true;
}

void server_chat_completions_store::persist_unlocked(const server_chat_completions_store_entry & e) {
    if (root_path_.empty()) {
        return;
    }
    if (!openai_persist::write_json_file(openai_persist::json_path(root_path_, e.id), entry_to_json(e))) {
        LOG_WRN("%s: failed to persist chat completion %s\n", __func__, e.id.c_str());
    }
}

void server_chat_completions_store::erase_disk_unlocked(const std::string & id) {
    if (root_path_.empty()) {
        return;
    }
    openai_persist::erase_file(openai_persist::json_path(root_path_, id));
}

bool server_chat_completions_store::try_load_id_unlocked(const std::string & id) {
    if (root_path_.empty() || max_entries_ <= 0 || id.empty()) {
        return false;
    }
    if (entries.find(id) != entries.end()) {
        return true;
    }
    json j;
    if (!openai_persist::read_json_file(openai_persist::json_path(root_path_, id), j)) {
        const std::string legacy = root_path_ + openai_persist::safe_id_legacy(id) + ".json";
        if (!openai_persist::read_json_file(legacy, j)) {
            return false;
        }
    }
    server_chat_completions_store_entry e;
    if (!entry_from_json(j, e) || e.id != id) {
        return false;
    }
    const int64_t now = chat_cmpl_now_unix();
    if (e.expires_at > 0 && e.expires_at <= now) {
        erase_disk_unlocked(e.id);
        return false;
    }
    entries[e.id] = std::move(e);
    lru_stamp[id] = ++lru_clock_;
    evict_lru_unlocked();
    return true;
}

void server_chat_completions_store::sync_new_from_disk_unlocked() {
    if (root_path_.empty() || max_entries_ <= 0) {
        return;
    }
    const int64_t now = chat_cmpl_now_unix();
    for (const auto & name : openai_persist::list_json_basenames(root_path_)) {
        json j;
        if (!openai_persist::read_json_file(root_path_ + name, j)) {
            continue;
        }
        server_chat_completions_store_entry e;
        if (!entry_from_json(j, e)) {
            continue;
        }
        if (entries.find(e.id) != entries.end()) {
            continue;
        }
        if (e.expires_at > 0 && e.expires_at <= now) {
            erase_disk_unlocked(e.id);
            continue;
        }
        const std::string eid = e.id;
        entries[eid] = std::move(e);
        lru_stamp[eid] = ++lru_clock_;
    }
    evict_lru_unlocked();
}

void server_chat_completions_store::load_from_disk_unlocked() {
    if (root_path_.empty()) {
        return;
    }
    const int64_t now = chat_cmpl_now_unix();
    for (const auto & name : openai_persist::list_json_basenames(root_path_)) {
        json j;
        if (!openai_persist::read_json_file(root_path_ + name, j)) {
            continue;
        }
        server_chat_completions_store_entry e;
        if (!entry_from_json(j, e)) {
            continue;
        }
        if (e.expires_at > 0 && e.expires_at <= now) {
            erase_disk_unlocked(e.id);
            continue;
        }
        entries[e.id] = e;
        lru_stamp[e.id] = ++lru_clock_;
    }
    LOG_INF("%s: loaded %zu chat completions from %s\n", __func__, entries.size(), root_path_.c_str());
}

void server_chat_completions_store::configure(int32_t max_entries, int32_t ttl_seconds, const std::string & root_path) {
    std::lock_guard<std::mutex> lock(mutex);
    max_entries_ = std::max(0, max_entries);
    ttl_seconds_ = std::max(0, ttl_seconds);
    entries.clear();
    lru_stamp.clear();
    root_path_.clear();
    if (!root_path.empty() && max_entries_ > 0) {
        root_path_ = root_path;
        if (root_path_.back() != DIRECTORY_SEPARATOR) {
            root_path_ += DIRECTORY_SEPARATOR;
        }
        if (!openai_persist::ensure_dir(root_path_)) {
            LOG_ERR("%s: cannot use chat completions path '%s'\n", __func__, root_path_.c_str());
            root_path_.clear();
        } else {
            load_from_disk_unlocked();
        }
    }
}

int32_t server_chat_completions_store::max_entries() const {
    return max_entries_;
}

size_t server_chat_completions_store::size() {
    std::lock_guard<std::mutex> lock(mutex);
    return entries.size();
}

void server_chat_completions_store::clear() {
    std::lock_guard<std::mutex> lock(mutex);
    for (const auto & kv : entries) {
        erase_disk_unlocked(kv.first);
    }
    entries.clear();
    lru_stamp.clear();
}

void server_chat_completions_store::evict_expired_unlocked(int64_t now) {
    for (auto it = entries.begin(); it != entries.end();) {
        if (it->second.expires_at > 0 && it->second.expires_at <= now) {
            erase_disk_unlocked(it->first);
            lru_stamp.erase(it->first);
            it = entries.erase(it);
        } else {
            ++it;
        }
    }
}

void server_chat_completions_store::evict_lru_unlocked() {
    if (max_entries_ <= 0) {
        for (const auto & kv : entries) {
            erase_disk_unlocked(kv.first);
        }
        entries.clear();
        lru_stamp.clear();
        return;
    }
    while ((int32_t) entries.size() > max_entries_) {
        auto oldest = lru_stamp.begin();
        for (auto it = lru_stamp.begin(); it != lru_stamp.end(); ++it) {
            if (it->second < oldest->second) {
                oldest = it;
            }
        }
        if (oldest == lru_stamp.end()) {
            break;
        }
        erase_disk_unlocked(oldest->first);
        entries.erase(oldest->first);
        lru_stamp.erase(oldest);
    }
}

void server_chat_completions_store::put(server_chat_completions_store_entry entry) {
    std::lock_guard<std::mutex> lock(mutex);
    if (max_entries_ <= 0) {
        return;
    }
    const int64_t now = chat_cmpl_now_unix();
    if (entry.created <= 0) {
        entry.created = now;
    }
    if (ttl_seconds_ > 0) {
        entry.expires_at = now + ttl_seconds_;
    } else {
        entry.expires_at = 0;
    }
    evict_expired_unlocked(now);
    persist_unlocked(entry);
    entries[entry.id] = std::move(entry);
    lru_stamp[entry.id] = ++lru_clock_;
    evict_lru_unlocked();
}

std::optional<server_chat_completions_store_entry> server_chat_completions_store::get(const std::string & id) {
    std::lock_guard<std::mutex> lock(mutex);
    const int64_t now = chat_cmpl_now_unix();
    evict_expired_unlocked(now);
    try_load_id_unlocked(id);
    auto it = entries.find(id);
    if (it == entries.end()) {
        return std::nullopt;
    }
    if (it->second.expires_at > 0 && it->second.expires_at <= now) {
        erase_disk_unlocked(it->first);
        lru_stamp.erase(it->first);
        entries.erase(it);
        return std::nullopt;
    }
    lru_stamp[id] = ++lru_clock_;
    return it->second;
}

std::optional<json> server_chat_completions_store::update_metadata(const std::string & id, const json & metadata) {
    std::lock_guard<std::mutex> lock(mutex);
    const int64_t now = chat_cmpl_now_unix();
    evict_expired_unlocked(now);
    try_load_id_unlocked(id);
    auto it = entries.find(id);
    if (it == entries.end()) {
        return std::nullopt;
    }
    if (it->second.expires_at > 0 && it->second.expires_at <= now) {
        erase_disk_unlocked(it->first);
        lru_stamp.erase(it->first);
        entries.erase(it);
        return std::nullopt;
    }
    it->second.metadata = metadata;
    it->second.completion["metadata"] = metadata;
    persist_unlocked(it->second);
    lru_stamp[id] = ++lru_clock_;
    return it->second.completion;
}

bool server_chat_completions_store::erase(const std::string & id) {
    std::lock_guard<std::mutex> lock(mutex);
    try_load_id_unlocked(id);
    auto it = entries.find(id);
    if (it == entries.end()) {
        return false;
    }
    erase_disk_unlocked(it->first);
    lru_stamp.erase(it->first);
    entries.erase(it);
    return true;
}

json server_chat_completions_store::list(
    const std::string & after,
    int limit,
    bool order_desc,
    const std::string & model_filter) {
    std::lock_guard<std::mutex> lock(mutex);
    sync_new_from_disk_unlocked();
    std::vector<server_chat_completions_store_entry> rows;
    rows.reserve(entries.size());
    const int64_t now = chat_cmpl_now_unix();
    for (const auto & kv : entries) {
        if (kv.second.expires_at > 0 && kv.second.expires_at <= now) {
            continue;
        }
        if (!model_filter.empty() && kv.second.model != model_filter) {
            continue;
        }
        rows.push_back(kv.second);
    }

    std::sort(rows.begin(), rows.end(), [order_desc](const auto & a, const auto & b) {
        if (a.created != b.created) {
            return order_desc ? (a.created > b.created) : (a.created < b.created);
        }
        return order_desc ? (a.id > b.id) : (a.id < b.id);
    });

    size_t start = 0;
    if (!after.empty()) {
        for (size_t i = 0; i < rows.size(); ++i) {
            if (rows[i].id == after) {
                start = i + 1;
                break;
            }
        }
    }

    if (limit <= 0) {
        limit = 20;
    }
    if (limit > 100) {
        limit = 100;
    }

    json data = json::array();
    for (size_t i = start; i < rows.size() && (int) data.size() < limit; ++i) {
        data.push_back(rows[i].completion);
    }

    const bool has_more = (start + (size_t) data.size()) < rows.size();
    json out = {
        {"object",   "list"},
        {"data",     data},
        {"has_more", has_more},
    };
    if (!data.empty()) {
        out["first_id"] = data.front().at("id");
        out["last_id"]  = data.back().at("id");
    }
    return out;
}

void server_chat_completions_remember(const json & completion_obj) {
    if (server_chat_completions_store::instance().max_entries() <= 0) {
        return;
    }
    if (!completion_obj.contains("id") || !completion_obj.at("id").is_string()) {
        return;
    }
    const bool should_store = json_value(completion_obj, "store", false);
    if (!should_store) {
        return;
    }

    server_chat_completions_store_entry entry;
    entry.id         = completion_obj.at("id").get<std::string>();
    entry.created    = json_value(completion_obj, "created", (int64_t) 0);
    entry.model      = json_value(completion_obj, "model", std::string());
    entry.metadata   = completion_obj.contains("metadata") ? completion_obj.at("metadata") : json(nullptr);
    entry.completion = completion_obj;

    server_chat_completions_store::instance().put(std::move(entry));
}
