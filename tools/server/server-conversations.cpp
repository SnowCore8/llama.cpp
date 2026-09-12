#include "server-conversations.h"
#include "server-openai-persist.h"

#include "log.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <stdexcept>
#include <vector>

static int64_t now_unix() {
    return (int64_t) std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

// last write time of path in filesystem ticks; 0 when the file is missing
static int64_t file_mtime(const std::string & path) {
    std::error_code ec;
    const auto t = std::filesystem::last_write_time(path, ec);
    return ec ? 0 : (int64_t) t.time_since_epoch().count();
}

// backing file mtime for id: current encoding first, then the legacy name
static int64_t conversation_file_mtime(const std::string & root_path, const std::string & id) {
    int64_t mt = file_mtime(openai_persist::json_path(root_path, id));
    if (mt == 0) {
        mt = file_mtime(root_path + openai_persist::safe_id_legacy(id) + ".json");
    }
    return mt;
}

server_conversations_store & server_conversations_store::instance() {
    static server_conversations_store inst;
    return inst;
}

json server_conversations_store::entry_to_json(const server_conversation_entry & e) const {
    return json{
        {"id",         e.id},
        {"created_at", e.created_at},
        {"expires_at", e.expires_at},
        {"last_used",  e.last_used},
        {"metadata",   e.metadata},
        {"items",      e.items},
    };
}

bool server_conversations_store::entry_from_json(const json & j, server_conversation_entry & e) const {
    if (!j.contains("id") || !j.at("id").is_string()) {
        return false;
    }
    e.id         = j.at("id").get<std::string>();
    e.created_at = j.value("created_at", (int64_t) 0);
    e.expires_at = j.value("expires_at", (int64_t) 0);
    e.last_used  = j.value("last_used",  (int64_t) 0);
    e.metadata   = j.contains("metadata") ? j.at("metadata") : json::object();
    e.items      = json::array();
    if (j.contains("items") && j.at("items").is_array()) {
        size_t dropped = 0;
        for (const auto & item : j.at("items")) {
            if (item.is_object()) {
                e.items.push_back(item);
            } else {
                ++dropped;
            }
        }
        if (dropped > 0) {
            LOG_WRN("%s: dropped %zu non-object items of conversation %s\n", __func__, dropped, e.id.c_str());
        }
    }
    return true;
}

bool server_conversations_store::persist_unlocked(const server_conversation_entry & e) {
    if (root_path_.empty()) {
        return true;
    }
    if (!openai_persist::write_json_file(openai_persist::json_path(root_path_, e.id), entry_to_json(e))) {
        LOG_WRN("%s: failed to persist conversation %s\n", __func__, e.id.c_str());
        return false;
    }
    return true;
}

void server_conversations_store::erase_disk_unlocked(const std::string & id) {
    if (root_path_.empty()) {
        return;
    }
    openai_persist::erase_file(openai_persist::json_path(root_path_, id));
}

bool server_conversations_store::try_load_id_unlocked(const std::string & id) {
    if (root_path_.empty() || max_entries_ <= 0 || id.empty()) {
        return false;
    }
    if (entries.find(id) != entries.end()) {
        return true;
    }
    return reload_id_from_disk_unlocked(id);
}

bool server_conversations_store::reload_id_from_disk_unlocked(const std::string & id) {
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
    server_conversation_entry e;
    if (!entry_from_json(j, e) || e.id != id) {
        return false;
    }
    const int64_t now = now_unix();
    if (e.expires_at > 0 && e.expires_at <= now) {
        erase_disk_unlocked(e.id);
        drop_unlocked(e.id);
        return false;
    }
    entries[e.id] = std::move(e);
    lru_stamp[id] = ++lru_clock_;
    mtime_stamp[id] = conversation_file_mtime(root_path_, id);
    evict_lru_unlocked();
    return true;
}

void server_conversations_store::load_from_disk_unlocked() {
    if (root_path_.empty()) {
        return;
    }
    const int64_t now = now_unix();
    std::vector<std::pair<int64_t, std::string>> order; // by last_used, so LRU survives restarts
    for (const auto & name : openai_persist::list_json_basenames(root_path_)) {
        json j;
        if (!openai_persist::read_json_file(root_path_ + name, j)) {
            continue;
        }
        server_conversation_entry e;
        if (!entry_from_json(j, e)) {
            continue;
        }
        if (e.expires_at > 0 && e.expires_at <= now) {
            erase_disk_unlocked(e.id);
            continue;
        }
        order.emplace_back(e.last_used, e.id);
        entries[e.id] = std::move(e);
    }
    std::sort(order.begin(), order.end());
    for (const auto & kv : order) {
        lru_stamp[kv.second] = ++lru_clock_;
    }
    LOG_INF("%s: loaded %zu conversations from %s\n", __func__, entries.size(), root_path_.c_str());
}

void server_conversations_store::configure(int32_t max_entries, int32_t ttl_seconds, const std::string & root_path) {
    std::lock_guard<std::mutex> lock(mutex);
    max_entries_ = std::max(0, max_entries);
    ttl_seconds_ = std::max(0, ttl_seconds);
    entries.clear();
    lru_stamp.clear();
    mtime_stamp.clear();
    root_path_.clear();
    if (!root_path.empty() && max_entries_ > 0) {
        root_path_ = root_path;
        if (root_path_.back() != DIRECTORY_SEPARATOR) {
            root_path_ += DIRECTORY_SEPARATOR;
        }
        if (!openai_persist::ensure_dir(root_path_)) {
            LOG_ERR("%s: cannot use conversations path '%s'\n", __func__, root_path_.c_str());
            root_path_.clear();
        } else {
            load_from_disk_unlocked();
        }
    }
}

void server_conversations_store::evict_expired_unlocked(int64_t now) {
    for (auto it = entries.begin(); it != entries.end();) {
        if (it->second.expires_at > 0 && it->second.expires_at <= now) {
            erase_disk_unlocked(it->first);
            lru_stamp.erase(it->first);
            mtime_stamp.erase(it->first);
            it = entries.erase(it);
        } else {
            ++it;
        }
    }
}

void server_conversations_store::evict_lru_unlocked() {
    if (max_entries_ <= 0) {
        // put() and update() refuse to run while the store is disabled
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
        mtime_stamp.erase(oldest->first);
        lru_stamp.erase(oldest);
    }
}

void server_conversations_store::drop_unlocked(const std::string & id) {
    entries.erase(id);
    lru_stamp.erase(id);
    mtime_stamp.erase(id);
}

bool server_conversations_store::enabled() {
    std::lock_guard<std::mutex> lock(mutex);
    return max_entries_ > 0;
}

bool server_conversations_store::put(server_conversation_entry entry) {
    std::lock_guard<std::mutex> lock(mutex);
    if (max_entries_ <= 0) {
        return false;
    }
    const int64_t now = now_unix();
    if (entry.created_at <= 0) {
        entry.created_at = now;
    }
    entry.last_used = now;
    if (ttl_seconds_ > 0) {
        entry.expires_at = now + ttl_seconds_;
    } else {
        entry.expires_at = 0;
    }
    evict_expired_unlocked(now);
    if (!persist_unlocked(entry)) {
        return false; // keep memory in sync with disk: nothing was written
    }
    const std::string id = entry.id;
    entries[id] = std::move(entry);
    lru_stamp[id] = ++lru_clock_;
    if (!root_path_.empty()) {
        mtime_stamp[id] = conversation_file_mtime(root_path_, id);
    }
    evict_lru_unlocked();
    return true;
}

// Load entries[id] (refreshing from shared disk when the backing file changed,
// dropping it when the file is gone) and bump its LRU stamp. Caller holds the lock.
server_conversation_entry * server_conversations_store::load_unlocked(const std::string & id) {
    const int64_t now = now_unix();
    evict_expired_unlocked(now);
    if (root_path_.empty() || max_entries_ <= 0 || id.empty()) {
        try_load_id_unlocked(id);
    } else {
        const int64_t mt     = conversation_file_mtime(root_path_, id);
        const auto    mit    = mtime_stamp.find(id);
        const bool    cached = entries.find(id) != entries.end() &&
                               mit != mtime_stamp.end() && mit->second == mt;
        if (!cached && !reload_id_from_disk_unlocked(id)) {
            drop_unlocked(id);
            return nullptr;
        }
    }
    auto it = entries.find(id);
    if (it == entries.end()) {
        return nullptr;
    }
    if (it->second.expires_at > 0 && it->second.expires_at <= now) {
        erase_disk_unlocked(it->first);
        drop_unlocked(it->first);
        return nullptr;
    }
    lru_stamp[id] = ++lru_clock_;
    return &it->second;
}

std::optional<server_conversation_entry> server_conversations_store::get(const std::string & id) {
    std::lock_guard<std::mutex> lock(mutex);
    const server_conversation_entry * e = load_unlocked(id);
    if (e == nullptr) {
        return std::nullopt;
    }
    return *e;
}

bool server_conversations_store::update(const std::string & id,
        const std::function<void(server_conversation_entry &)> & fn) {
    std::lock_guard<std::mutex> lock(mutex);
    const server_conversation_entry * cur = load_unlocked(id);
    if (cur == nullptr) {
        return false;
    }
    // mutate a copy: a failing fn or persist leaves memory and disk untouched
    server_conversation_entry next = *cur;
    fn(next);
    const int64_t now = now_unix();
    next.last_used = now;
    if (ttl_seconds_ > 0) {
        next.expires_at = now + ttl_seconds_;
    } else {
        next.expires_at = 0;
    }
    if (!persist_unlocked(next)) {
        throw server_conversations_error("failed to persist conversation: " + id, ERROR_TYPE_SERVER);
    }
    entries[id] = std::move(next);
    lru_stamp[id] = ++lru_clock_;
    if (!root_path_.empty()) {
        mtime_stamp[id] = conversation_file_mtime(root_path_, id);
    }
    evict_lru_unlocked();
    return true;
}

bool server_conversations_store::erase(const std::string & id) {
    std::lock_guard<std::mutex> lock(mutex);
    try_load_id_unlocked(id);
    auto it = entries.find(id);
    if (it == entries.end()) {
        return false;
    }
    erase_disk_unlocked(it->first);
    drop_unlocked(it->first);
    return true;
}

//
// API helpers
//

// conversations need the durable store: without it create() could only hand out
// ids that nothing can retrieve
static void require_store() {
    if (!server_conversations_store::instance().enabled()) {
        throw server_conversations_error("conversations are disabled (--responses-store-max 0)",
                                         ERROR_TYPE_NOT_SUPPORTED);
    }
}

static void check_persisted(bool ok, const std::string & id) {
    if (!ok) {
        throw server_conversations_error("failed to persist conversation: " + id, ERROR_TYPE_SERVER);
    }
}

static void validate_metadata(const json & metadata) {
    if (metadata.is_null()) {
        return;
    }
    if (!metadata.is_object()) {
        throw std::invalid_argument("'metadata' must be an object");
    }
    if (metadata.size() > 16) {
        throw std::invalid_argument("'metadata' must have at most 16 key-value pairs");
    }
    for (const auto & el : metadata.items()) {
        if (!el.value().is_string()) {
            throw std::invalid_argument("'metadata' values must be strings");
        }
        if (el.key().size() > 64) {
            throw std::invalid_argument("'metadata' keys must be at most 64 characters");
        }
        if (el.value().get<std::string>().size() > 512) {
            throw std::invalid_argument("'metadata' values must be at most 512 characters");
        }
    }
}

static const char * item_id_prefix(const std::string & type) {
    if (type == "message")              return "msg_";
    if (type == "function_call")        return "fc_";
    if (type == "function_call_output") return "fco_";
    if (type == "reasoning")            return "rs_";
    return "item_";
}

static bool include_has(const json & include, const char * name) {
    if (!include.is_array()) {
        return false;
    }
    for (const auto & v : include) {
        if (v.is_string() && v.get<std::string>() == name) {
            return true;
        }
    }
    return false;
}

static json item_list_to_json(const std::vector<json> & data, bool has_more) {
    json out = json{
        {"object",   "list"},
        {"data",     data},
        {"has_more", has_more},
        {"first_id", data.empty() ? json(nullptr) : data.front().value("id", json(nullptr))},
        {"last_id",  data.empty() ? json(nullptr) : data.back().value("id", json(nullptr))},
    };
    return out;
}

json server_conversation_to_json(const server_conversation_entry & e) {
    return json{
        {"id",         e.id},
        {"created_at", e.created_at},
        {"metadata",   e.metadata},
        {"object",     "conversation"},
    };
}

void server_conversation_item_normalize(json & item) {
    if (!item.is_object()) {
        throw std::invalid_argument("conversation items must be objects");
    }
    std::string type = json_value(item, "type", std::string());
    if (type.empty() && item.contains("role")) {
        type = "message";
    }
    if (type.empty()) {
        throw std::invalid_argument("each conversation item must have a 'type'");
    }
    item["type"] = type;
    if (type == "message") {
        const std::string role = json_value(item, "role", std::string());
        if (role != "user" && role != "assistant" && role != "system" && role != "developer") {
            throw std::invalid_argument("'role' must be one of: user, assistant, system, developer");
        }
        if (!item.contains("content") || item.at("content").is_null()) {
            throw std::invalid_argument("message items must have 'content'");
        }
        if (item.at("content").is_string()) {
            const std::string text = item.at("content").get<std::string>();
            if (role == "assistant") {
                item["content"] = json::array({ json{
                    {"type",        "output_text"},
                    {"text",        text},
                    {"annotations", json::array()},
                    {"logprobs",    json::array()},
                } });
            } else {
                item["content"] = json::array({ json{
                    {"type", "input_text"},
                    {"text", text},
                } });
            }
        } else if (!item.at("content").is_array()) {
            throw std::invalid_argument("message 'content' must be a string or an array");
        }
        if (!item.contains("status")) {
            item["status"] = "completed";
        }
    } else if (type == "function_call") {
        for (const char * key : { "call_id", "name", "arguments" }) {
            if (!item.contains(key) || !item.at(key).is_string()) {
                throw std::invalid_argument(std::string("function_call items must have '") + key + "'");
            }
        }
    } else if (type == "function_call_output") {
        for (const char * key : { "call_id", "output" }) {
            if (!item.contains(key)) {
                throw std::invalid_argument(std::string("function_call_output items must have '") + key + "'");
            }
        }
    }
    if (!item.contains("id") || !item.at("id").is_string() || item.at("id").get<std::string>().empty()) {
        item["id"] = std::string(item_id_prefix(type)) + random_string();
    }
}

json server_conversation_item_filter(const json & item, const json & include) {
    json out = item;
    const std::string type = json_value(out, "type", std::string());
    if (type == "file_search_call") {
        if (!include_has(include, "file_search_call.results")) {
            out.erase("results");
        }
    } else if (type == "web_search_call") {
        if (!include_has(include, "web_search_call.results")) {
            out.erase("results");
        }
        if (!include_has(include, "web_search_call.action.sources") &&
                out.contains("action") && out.at("action").is_object()) {
            out.at("action").erase("sources");
        }
    } else if (type == "computer_call_output") {
        if (!include_has(include, "computer_call_output.output.image_url") &&
                out.contains("output") && out.at("output").is_object()) {
            out.at("output").erase("image_url");
        }
    } else if (type == "code_interpreter_call") {
        if (!include_has(include, "code_interpreter_call.outputs")) {
            out.erase("outputs");
        }
    } else if (type == "reasoning") {
        if (!include_has(include, "reasoning.encrypted_content")) {
            out.erase("encrypted_content");
        }
    }
    if (out.contains("content") && out.at("content").is_array()) {
        const bool want_logprobs = include_has(include, "message.output_text.logprobs");
        const bool want_image_url = include_has(include, "message.input_image.image_url");
        for (auto & part : out.at("content")) {
            if (!part.is_object()) {
                continue;
            }
            const std::string part_type = json_value(part, "type", std::string());
            if (part_type == "output_text" && !want_logprobs) {
                part.erase("logprobs");
            }
            if (part_type == "input_image" && !want_image_url) {
                part.erase("image_url");
            }
        }
    }
    return out;
}

json server_conversations_include_from_param(const std::string & raw) {
    json out = json::array();
    size_t pos = 0;
    while (pos <= raw.size() && !raw.empty()) {
        const size_t comma = raw.find(',', pos);
        const size_t end = comma == std::string::npos ? raw.size() : comma;
        size_t b = pos;
        size_t e = end;
        while (b < e && (raw[b] == ' ' || raw[b] == '\t')) ++b;
        while (e > b && (raw[e - 1] == ' ' || raw[e - 1] == '\t')) --e;
        if (e > b) {
            out.push_back(raw.substr(b, e - b));
        }
        if (comma == std::string::npos) {
            break;
        }
        pos = comma + 1;
    }
    return out;
}

static void item_array_validate(const json & items) {
    if (!items.is_array()) {
        throw std::invalid_argument("'items' must be an array");
    }
    if (items.size() > 20) {
        throw std::invalid_argument("'items' can include at most 20 items per request");
    }
}

json server_conversations_create(const json & body) {
    require_store();
    if (!body.is_object()) {
        throw std::invalid_argument("request body must be a JSON object");
    }
    server_conversation_entry e;
    e.id = "conv_" + random_string();
    e.created_at = now_unix();
    if (body.contains("metadata")) {
        validate_metadata(body.at("metadata"));
        e.metadata = body.at("metadata");
    }
    if (body.contains("items") && !body.at("items").is_null()) {
        const json & items = body.at("items");
        item_array_validate(items);
        for (auto item : items) {
            server_conversation_item_normalize(item);
            e.items.push_back(std::move(item));
        }
    }
    check_persisted(server_conversations_store::instance().put(e), e.id);
    return server_conversation_to_json(e);
}

json server_conversations_get(const std::string & id) {
    auto entry = server_conversations_store::instance().get(id);
    if (!entry.has_value()) {
        throw server_conversations_error("conversation not found: " + id, true);
    }
    return server_conversation_to_json(*entry);
}

json server_conversations_update(const std::string & id, const json & body) {
    require_store();
    if (!body.is_object() || !body.contains("metadata")) {
        throw std::invalid_argument("'metadata' is required");
    }
    validate_metadata(body.at("metadata"));
    const json metadata = body.at("metadata");
    json out;
    const bool found = server_conversations_store::instance().update(id,
        [&](server_conversation_entry & e) {
            e.metadata = metadata;
            out = server_conversation_to_json(e);
        });
    if (!found) {
        throw server_conversations_error("conversation not found: " + id, true);
    }
    return out;
}

json server_conversations_delete(const std::string & id) {
    require_store();
    if (!server_conversations_store::instance().erase(id)) {
        throw server_conversations_error("conversation not found: " + id, true);
    }
    return json{
        {"id",      id},
        {"object",  "conversation.deleted"},
        {"deleted", true},
    };
}

json server_conversations_add_items(const std::string & id, const json & body) {
    require_store();
    if (!body.is_object() || !body.contains("items") || body.at("items").is_null()) {
        throw std::invalid_argument("'items' array is required");
    }
    const json & items = body.at("items");
    item_array_validate(items);
    std::vector<json> added;
    for (auto item : items) {
        server_conversation_item_normalize(item);
        added.push_back(std::move(item));
    }
    const json added_list = item_list_to_json(added, false);
    const bool found = server_conversations_store::instance().update(id,
        [&](server_conversation_entry & e) {
            for (auto & item : added) {
                e.items.push_back(std::move(item));
            }
        });
    if (!found) {
        throw server_conversations_error("conversation not found: " + id, true);
    }
    return added_list;
}

json server_conversations_list_items(const std::string & id, const std::string & after,
                                     const std::string & order, int64_t limit, const json & include) {
    auto entry = server_conversations_store::instance().get(id);
    if (!entry.has_value()) {
        throw server_conversations_error("conversation not found: " + id, true);
    }
    if (order != "asc" && order != "desc") {
        throw std::invalid_argument("'order' must be 'asc' or 'desc'");
    }
    if (limit <= 0 || limit > 100) {
        throw std::invalid_argument("'limit' must be between 1 and 100");
    }
    const json & items = entry->items;
    const int64_t n = (int64_t) items.size();
    int64_t idx_after = -1;
    if (!after.empty()) {
        for (int64_t i = 0; i < n; ++i) {
            if (json_value(items.at(i), "id", std::string()) == after) {
                idx_after = i;
                break;
            }
        }
        if (idx_after < 0) {
            throw std::invalid_argument("'after' item not found in conversation: " + after);
        }
    }
    std::vector<json> data;
    bool has_more = false;
    if (order == "desc") {
        int64_t i = after.empty() ? n - 1 : idx_after - 1;
        for (; i >= 0 && (int64_t) data.size() < limit; --i) {
            data.push_back(server_conversation_item_filter(items.at(i), include));
        }
        has_more = i >= 0;
    } else {
        int64_t i = after.empty() ? 0 : idx_after + 1;
        for (; i < n && (int64_t) data.size() < limit; ++i) {
            data.push_back(server_conversation_item_filter(items.at(i), include));
        }
        has_more = i < n;
    }
    return item_list_to_json(data, has_more);
}

json server_conversations_get_item(const std::string & id, const std::string & item_id, const json & include) {
    auto entry = server_conversations_store::instance().get(id);
    if (!entry.has_value()) {
        throw server_conversations_error("conversation not found: " + id, true);
    }
    for (const auto & item : entry->items) {
        if (json_value(item, "id", std::string()) == item_id) {
            return server_conversation_item_filter(item, include);
        }
    }
    throw server_conversations_error("item not found in conversation: " + item_id, true);
}

json server_conversations_delete_item(const std::string & id, const std::string & item_id) {
    require_store();
    json out;
    const bool found = server_conversations_store::instance().update(id,
        [&](server_conversation_entry & e) {
            bool removed = false;
            json kept = json::array();
            for (const auto & item : e.items) {
                if (!removed && json_value(item, "id", std::string()) == item_id) {
                    removed = true;
                    continue;
                }
                kept.push_back(item);
            }
            if (!removed) {
                throw server_conversations_error("item not found in conversation: " + item_id, true);
            }
            e.items = std::move(kept);
            out = server_conversation_to_json(e);
        });
    if (!found) {
        throw server_conversations_error("conversation not found: " + id, true);
    }
    return out;
}

bool server_conversations_append_turn(const std::string & id, const json & input_items, const json & output_items) {
    bool found = false;
    try {
        found = server_conversations_store::instance().update(id,
            [&](server_conversation_entry & e) {
                auto push_item = [&](const json & item) {
                    if (!item.is_object()) {
                        return;
                    }
                    try {
                        json copy = item;
                        server_conversation_item_normalize(copy);
                        e.items.push_back(std::move(copy));
                    } catch (const std::exception & e) {
                        LOG_WRN("%s: dropped conversation item: %s\n", __func__, e.what());
                    }
                };
                if (input_items.is_array()) {
                    for (const auto & item : input_items) {
                        push_item(item);
                    }
                }
                if (output_items.is_array()) {
                    for (const auto & item : output_items) {
                        push_item(item);
                    }
                }
            });
    } catch (...) {
        // a failed write is already reported by the store; nothing to add here
        return true;
    }
    return found;
}
