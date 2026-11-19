#include "server-conversations.h"
#include "server-openai-store.h"

#include <algorithm>
#include <filesystem>

json server_conversation_normalize_item(const json & item) {
    if (!item.is_object()) {
        throw std::invalid_argument("conversation items must be objects");
    }
    json out = item;
    if (!out.contains("id")) {
        out["id"] = "item_" + random_string();
    }
    std::string type = out.value("type", std::string());
    if (type.empty()) {
        if (!out.contains("role")) {
            throw std::invalid_argument("conversation item requires a type or a role");
        }
        type = "message";
        out["type"] = type;
    }
    if (type == "message") {
        std::string role = out.value("role", std::string("user"));
        out["role"] = role;
        if (!out.contains("status")) {
            out["status"] = "completed";
        }
        if (!out.contains("content")) {
            out["content"] = json::array();
        }
        json & content = out["content"];
        if (content.is_string()) {
            const std::string part_type = role == "assistant" ? "output_text" : "input_text";
            content = json::array({ json {
                {"type", part_type},
                {"text", content.get<std::string>()},
            } });
        } else if (content.is_object()) {
            content = json::array({ content });
        } else if (!content.is_array()) {
            throw std::invalid_argument("conversation message content must be a string or an array");
        }
    }
    return out;
}

json server_conversation::to_json() const {
    return json {
        {"id",         id},
        {"object",     "conversation"},
        {"created_at", created_at},
        {"metadata",   metadata},
        {"items",      items},
    };
}

bool server_conversation::from_json(const json & data, server_conversation & out) {
    if (!data.is_object()) {
        return false;
    }
    out.id         = data.value("id", std::string());
    out.created_at = data.value("created_at", (int64_t) 0);
    out.metadata   = data.value("metadata", json(nullptr));
    out.items      = data.value("items", json::array());
    return !out.id.empty();
}

server_conversations_store::server_conversations_store(
        const std::string & base_path, size_t max_items_, int64_t ttl_seconds_) :
    path(base_path),
    max_items(max_items_),
    ttl_seconds(ttl_seconds_) {
    if (!path.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(std::filesystem::path(path) / "conversations", ec);
    }
}

std::string server_conversations_store::file_path(const std::string & id) const {
    return server_openai_store::item_path(path, "conversations", id);
}

bool server_conversations_store::load_one_locked(const std::string & id, server_conversation & out) {
    if (path.empty()) {
        return false;
    }
    json data;
    if (!server_openai_store::read_json(file_path(id), data)) {
        return false;
    }
    server_conversation conversation;
    if (!server_conversation::from_json(data, conversation)) {
        return false;
    }
    if (ttl_seconds > 0 && server_openai_store::now() - conversation.created_at > ttl_seconds) {
        server_openai_store::remove_file(file_path(id));
        return false;
    }
    items[id] = conversation;
    out = std::move(conversation);
    return true;
}

void server_conversations_store::save_locked(const server_conversation & conversation) {
    if (!path.empty()) {
        server_openai_store::write_json(file_path(conversation.id), conversation.to_json());
    }
}

void server_conversations_store::prune_locked() {
    const int64_t now = server_openai_store::now();
    if (ttl_seconds > 0) {
        for (auto it = items.begin(); it != items.end();) {
            if (now - it->second.created_at > ttl_seconds) {
                server_openai_store::remove_file(file_path(it->first));
                it = items.erase(it);
            } else {
                ++it;
            }
        }
    }
    while (max_items > 0 && items.size() > max_items) {
        auto oldest = items.begin();
        for (auto it = items.begin(); it != items.end(); ++it) {
            if (it->second.created_at < oldest->second.created_at) {
                oldest = it;
            }
        }
        server_openai_store::remove_file(file_path(oldest->first));
        items.erase(oldest);
    }
}

server_conversation server_conversations_store::create(const json & metadata, const json & input_items) {
    server_conversation conversation;
    conversation.id         = "conv_" + random_string();
    conversation.created_at = server_openai_store::now();
    conversation.metadata   = metadata;
    conversation.items      = json::array();
    if (!input_items.is_null()) {
        if (!input_items.is_array()) {
            throw std::invalid_argument("'items' must be an array");
        }
        for (const auto & item : input_items) {
            conversation.items.push_back(server_conversation_normalize_item(item));
        }
    }
    {
        std::lock_guard<std::mutex> lock(mutex);
        items[conversation.id] = conversation;
        prune_locked();
        save_locked(conversation);
    }
    return conversation;
}

bool server_conversations_store::get(const std::string & id, server_conversation & out) {
    std::lock_guard<std::mutex> lock(mutex);
    const int64_t now = server_openai_store::now();
    const auto it = items.find(id);
    if (it != items.end()) {
        if (ttl_seconds > 0 && now - it->second.created_at > ttl_seconds) {
            server_openai_store::remove_file(file_path(id));
            items.erase(it);
        } else {
            out = it->second;
            return true;
        }
    }
    return load_one_locked(id, out);
}

bool server_conversations_store::update_metadata(const std::string & id, const json & metadata) {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = items.find(id);
    if (it == items.end()) {
        server_conversation conversation;
        if (!load_one_locked(id, conversation)) {
            return false;
        }
        it = items.find(id);
    }
    it->second.metadata = metadata;
    save_locked(it->second);
    return true;
}

bool server_conversations_store::erase(const std::string & id) {
    std::lock_guard<std::mutex> lock(mutex);
    const bool erased = items.erase(id) > 0;
    if (!path.empty()) {
        server_openai_store::remove_file(file_path(id));
    }
    return erased;
}

bool server_conversations_store::add_items(const std::string & id, const json & input_items, json * added) {
    if (!input_items.is_array()) {
        throw std::invalid_argument("'items' must be an array");
    }
    std::lock_guard<std::mutex> lock(mutex);
    auto it = items.find(id);
    if (it == items.end()) {
        server_conversation conversation;
        if (!load_one_locked(id, conversation)) {
            return false;
        }
        it = items.find(id);
    }
    json added_items = json::array();
    for (const auto & item : input_items) {
        json normalized = server_conversation_normalize_item(item);
        added_items.push_back(normalized);
        it->second.items.push_back(std::move(normalized));
    }
    save_locked(it->second);
    if (added != nullptr) {
        *added = std::move(added_items);
    }
    return true;
}

bool server_conversations_store::get_item(const std::string & id, const std::string & item_id, json & out) {
    server_conversation conversation;
    if (!get(id, conversation)) {
        return false;
    }
    for (const auto & item : conversation.items) {
        if (item.value("id", std::string()) == item_id) {
            out = item;
            return true;
        }
    }
    return false;
}

bool server_conversations_store::delete_item(const std::string & id, const std::string & item_id, server_conversation & out) {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = items.find(id);
    if (it == items.end()) {
        server_conversation conversation;
        if (!load_one_locked(id, conversation)) {
            return false;
        }
        it = items.find(id);
    }
    json remaining = json::array();
    bool removed = false;
    for (const auto & item : it->second.items) {
        if (item.value("id", std::string()) == item_id) {
            removed = true;
        } else {
            remaining.push_back(item);
        }
    }
    if (!removed) {
        return false;
    }
    it->second.items = std::move(remaining);
    save_locked(it->second);
    out = it->second;
    return true;
}

bool server_conversations_store::list_items(
        const std::string & id,
        const std::string & after,
        int limit,
        const std::string & order,
        json & data,
        bool & has_more,
        json & first_id,
        json & last_id) {
    server_conversation conversation;
    if (!get(id, conversation)) {
        return false;
    }
    std::vector<json> items_list;
    for (const auto & item : conversation.items) {
        items_list.push_back(item);
    }
    if (order == "asc") {
        // keep insertion order
    } else if (order == "desc") {
        std::reverse(items_list.begin(), items_list.end());
    } else {
        throw std::invalid_argument("'order' must be asc or desc");
    }

    size_t start = 0;
    if (!after.empty()) {
        for (size_t i = 0; i < items_list.size(); i++) {
            if (items_list[i].value("id", std::string()) == after) {
                start = i + 1;
                break;
            }
        }
    }
    limit = std::max(1, std::min(limit, 100));
    data     = json::array();
    first_id = nullptr;
    last_id  = nullptr;
    for (size_t i = start; i < items_list.size() && (int) data.size() < limit; i++) {
        if (first_id.is_null()) {
            first_id = items_list[i].value("id", json(nullptr));
        }
        last_id = items_list[i].value("id", json(nullptr));
        data.push_back(items_list[i]);
    }
    has_more = start + data.size() < items_list.size();
    return true;
}
