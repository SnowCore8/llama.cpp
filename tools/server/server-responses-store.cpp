#include "server-responses-store.h"
#include "server-openai-persist.h"

#include "log.h"

#include <algorithm>
#include <chrono>

static int64_t now_unix() {
    return (int64_t) std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

server_responses_store & server_responses_store::instance() {
    static server_responses_store inst;
    return inst;
}

json server_responses_store::entry_to_json(const server_responses_store_entry & e) const {
    return json{
        {"id",            e.id},
        {"created_at",    e.created_at},
        {"expires_at",    e.expires_at},
        {"model",         e.model},
        {"request_model", e.request_model},
        {"instructions",  e.instructions},
        {"input",         e.input},
        {"output",        e.output},
        {"usage",         e.usage},
        {"response",      e.response},
    };
}

bool server_responses_store::entry_from_json(const json & j, server_responses_store_entry & e) const {
    if (!j.contains("id") || !j.at("id").is_string()) {
        return false;
    }
    e.id            = j.at("id").get<std::string>();
    e.created_at    = j.value("created_at", (int64_t) 0);
    e.expires_at    = j.value("expires_at", (int64_t) 0);
    e.model         = j.value("model", std::string());
    e.request_model = j.value("request_model", std::string());
    e.instructions  = j.contains("instructions") ? j.at("instructions") : json(nullptr);
    e.input         = j.contains("input") ? j.at("input") : json::array();
    e.output        = j.contains("output") ? j.at("output") : json::array();
    e.usage         = j.contains("usage") ? j.at("usage") : json::object();
    e.response      = j.contains("response") ? j.at("response") : json::object();
    return true;
}

void server_responses_store::persist_unlocked(const server_responses_store_entry & e) {
    if (root_path_.empty()) {
        return;
    }
    if (!openai_persist::write_json_file(openai_persist::json_path(root_path_, e.id), entry_to_json(e))) {
        LOG_WRN("%s: failed to persist response %s\n", __func__, e.id.c_str());
    }
}

void server_responses_store::erase_disk_unlocked(const std::string & id) {
    if (root_path_.empty()) {
        return;
    }
    openai_persist::erase_file(openai_persist::json_path(root_path_, id));
}

void server_responses_store::finalize_interrupted_unlocked(server_responses_store_entry & e) {
    if (!e.response.is_object()) {
        return;
    }
    const std::string status = e.response.value("status", std::string());
    if (status != "in_progress" && status != "queued" && status != "cancelling") {
        return;
    }
    e.response["status"] = "failed";
    e.response["error"] = json{
        {"code",    "server_restart"},
        {"message", "Response interrupted by server restart"},
    };
    e.response["incomplete_details"] = nullptr;
    if (e.response.contains("background")) {
        e.response["background"] = true;
    }
    if (!e.output.is_array()) {
        e.output = json::array();
    }
    // Keep response.output aligned
    e.response["output"] = e.output;
}

bool server_responses_store::try_load_id_unlocked(const std::string & id) {
    if (root_path_.empty() || max_entries_ <= 0 || id.empty()) {
        return false;
    }
    if (entries.find(id) != entries.end()) {
        return true;
    }
    return reload_id_from_disk_unlocked(id);
}

bool server_responses_store::reload_id_from_disk_unlocked(const std::string & id) {
    if (root_path_.empty() || max_entries_ <= 0 || id.empty()) {
        return false;
    }
    json j;
    if (!openai_persist::read_json_file(openai_persist::json_path(root_path_, id), j)) {
        const std::string legacy = root_path_ + openai_persist::safe_id_legacy(id) + ".json";
        if (!openai_persist::read_json_file(legacy, j)) {
            return false;
        }
    }
    server_responses_store_entry e;
    if (!entry_from_json(j, e) || e.id != id) {
        return false;
    }
    const int64_t now = now_unix();
    if (e.expires_at > 0 && e.expires_at <= now) {
        erase_disk_unlocked(e.id);
        entries.erase(e.id);
        lru_stamp.erase(e.id);
        return false;
    }
    // Do not finalize_interrupted here: disk may be owned by a live child worker.
    entries[e.id] = std::move(e);
    lru_stamp[id] = ++lru_clock_;
    evict_lru_unlocked();
    return true;
}

void server_responses_store::load_from_disk_unlocked() {
    if (root_path_.empty()) {
        return;
    }
    const int64_t now = now_unix();
    size_t interrupted = 0;
    for (const auto & name : openai_persist::list_json_basenames(root_path_)) {
        json j;
        if (!openai_persist::read_json_file(root_path_ + name, j)) {
            continue;
        }
        server_responses_store_entry e;
        if (!entry_from_json(j, e)) {
            continue;
        }
        if (e.expires_at > 0 && e.expires_at <= now) {
            erase_disk_unlocked(e.id);
            continue;
        }
        const std::string before = e.response.is_object() ? e.response.value("status", std::string()) : "";
        finalize_interrupted_unlocked(e);
        if (e.response.is_object() && e.response.value("status", std::string()) == "failed" &&
                (before == "in_progress" || before == "queued" || before == "cancelling")) {
            ++interrupted;
            persist_unlocked(e);
        }
        entries[e.id] = e;
        lru_stamp[e.id] = ++lru_clock_;
    }
    LOG_INF("%s: loaded %zu responses from %s (interrupted=%zu)\n",
            __func__, entries.size(), root_path_.c_str(), interrupted);
}

void server_responses_store::configure(int32_t max_entries, int32_t ttl_seconds, const std::string & root_path) {
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
            LOG_ERR("%s: cannot use responses path '%s'\n", __func__, root_path_.c_str());
            root_path_.clear();
        } else {
            load_from_disk_unlocked();
        }
    }
    if (max_entries_ == 0) {
        entries.clear();
        lru_stamp.clear();
    }
}

int32_t server_responses_store::max_entries() const {
    return max_entries_;
}

size_t server_responses_store::size() {
    std::lock_guard<std::mutex> lock(mutex);
    return entries.size();
}

void server_responses_store::clear() {
    std::lock_guard<std::mutex> lock(mutex);
    for (const auto & kv : entries) {
        erase_disk_unlocked(kv.first);
    }
    entries.clear();
    lru_stamp.clear();
}

void server_responses_store::evict_expired_unlocked(int64_t now) {
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

void server_responses_store::evict_lru_unlocked() {
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

void server_responses_store::put(server_responses_store_entry entry) {
    std::lock_guard<std::mutex> lock(mutex);
    if (max_entries_ <= 0) {
        return;
    }
    const int64_t now = now_unix();
    if (entry.created_at <= 0) {
        entry.created_at = now;
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

std::optional<server_responses_store_entry> server_responses_store::get(const std::string & id) {
    std::lock_guard<std::mutex> lock(mutex);
    const int64_t now = now_unix();
    evict_expired_unlocked(now);
    if (!root_path_.empty() && max_entries_ > 0 && !id.empty()) {
        // Shared disk: refresh or drop if another process erased the file.
        if (!reload_id_from_disk_unlocked(id)) {
            entries.erase(id);
            lru_stamp.erase(id);
            return std::nullopt;
        }
    } else {
        try_load_id_unlocked(id);
    }
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

bool server_responses_store::erase(const std::string & id) {
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
