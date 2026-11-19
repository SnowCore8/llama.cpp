#include "server-chat-completions.h"

#include "common.h"

#include <algorithm>
#include <ctime>
#include <stdexcept>

namespace {

std::string chat_tool_name(const json & tool) {
    if (!tool.is_object()) {
        return "";
    }
    if (tool.contains("function") && tool.at("function").is_object()) {
        return tool.at("function").value("name", std::string());
    }
    return tool.value("name", std::string());
}

int64_t response_completion_tokens(const json & response) {
    if (response.contains("usage") && response.at("usage").is_object()) {
        return response.at("usage").value("completion_tokens", (int64_t) 0);
    }
    return 0;
}

int64_t response_prompt_tokens(const json & response) {
    if (response.contains("usage") && response.at("usage").is_object()) {
        return response.at("usage").value("prompt_tokens", (int64_t) 0);
    }
    return 0;
}

bool metadata_matches(const json & candidate, const json & filter) {
    if (filter.is_null()) {
        return true;
    }
    if (!filter.is_object() || !candidate.is_object()) {
        return false;
    }
    for (const auto & [key, value] : filter.items()) {
        bool found = false;
        for (const auto & [candidate_key, candidate_value] : candidate.items()) {
            if (candidate_key == key) {
                found = candidate_value == value;
                break;
            }
        }
        if (!found) {
            return false;
        }
    }
    return true;
}

} // namespace

bool server_chat_completions_store::get(const std::string & id, server_chat_completions_stored & out) const {
    std::lock_guard<std::mutex> lock(mutex);
    const auto it = items.find(id);
    if (it == items.end()) {
        return false;
    }
    out = it->second;
    return true;
}

void server_chat_completions_store::put(const std::string & id, server_chat_completions_stored item) {
    std::lock_guard<std::mutex> lock(mutex);
    if (items.size() >= max_items && items.find(id) == items.end()) {
        items.erase(items.begin());
    }
    items[id] = std::move(item);
}

bool server_chat_completions_store::erase(const std::string & id) {
    std::lock_guard<std::mutex> lock(mutex);
    return items.erase(id) > 0;
}

bool server_chat_completions_store::update_metadata(const std::string & id, const json & metadata) {
    std::lock_guard<std::mutex> lock(mutex);
    const auto it = items.find(id);
    if (it == items.end()) {
        return false;
    }
    it->second.metadata = metadata;
    if (it->second.response.is_object()) {
        it->second.response["metadata"] = metadata;
    }
    return true;
}

std::vector<std::pair<std::string, server_chat_completions_stored>> server_chat_completions_store::list() const {
    std::lock_guard<std::mutex> lock(mutex);
    std::vector<std::pair<std::string, server_chat_completions_stored>> result;
    result.reserve(items.size());
    for (const auto & [id, stored] : items) {
        result.emplace_back(id, stored);
    }
    return result;
}

server_chat_completions_request server_chat_completions_parse_request(const json & body) {
    if (!body.is_object()) {
        throw std::invalid_argument("request body must be a JSON object");
    }
    if (!body.contains("messages") || !body.at("messages").is_array()) {
        throw std::invalid_argument("'messages' must be an array");
    }

    server_chat_completions_request req;
    req.model              = body.value("model", std::string());
    req.store              = body.value("store", false);
    req.n                  = body.value("n", 1);
    req.messages           = body.at("messages");
    req.metadata           = body.value("metadata", json(nullptr));
    req.user               = body.value("user", std::string());
    req.safety_identifier  = body.value("safety_identifier", std::string());
    req.service_tier       = body.value("service_tier", std::string());

    if (req.n < 1 || req.n > 128) {
        throw std::invalid_argument("'n' must be between 1 and 128");
    }
    server_validate_metadata(req.metadata);
    return req;
}

void server_chat_completions_normalize_request(json & body) {
    if (!body.contains("tool_choice") || body.at("tool_choice").is_null()) {
        return;
    }
    const json & choice = body.at("tool_choice");
    if (choice.is_string()) {
        return;
    }
    if (!choice.is_object()) {
        throw std::invalid_argument("'tool_choice' must be a string or an object");
    }

    const std::string type = choice.value("type", std::string());
    if (type == "function") {
        std::string name = choice.value("name", std::string());
        if (name.empty() && choice.contains("function") && choice.at("function").is_object()) {
            name = choice.at("function").value("name", std::string());
        }
        if (name.empty()) {
            throw std::invalid_argument("'tool_choice' function requires a 'name'");
        }
        json filtered = json::array();
        if (body.contains("tools") && body.at("tools").is_array()) {
            for (const auto & tool : body.at("tools")) {
                if (chat_tool_name(tool) == name) {
                    filtered.push_back(tool);
                }
            }
        }
        if (filtered.empty()) {
            throw std::invalid_argument("tool_choice refers to an unknown tool: " + name);
        }
        body["tools"]       = filtered;
        body["tool_choice"] = "required";
        return;
    }
    if (type == "allowed_tools") {
        const json allowed = choice.value("allowed_tools", json::object());
        std::vector<std::string> names;
        if (allowed.contains("tools") && allowed.at("tools").is_array()) {
            for (const auto & tool : allowed.at("tools")) {
                names.push_back(chat_tool_name(tool));
            }
        }
        json filtered = json::array();
        if (body.contains("tools") && body.at("tools").is_array()) {
            for (const auto & tool : body.at("tools")) {
                const std::string name = chat_tool_name(tool);
                if (std::find(names.begin(), names.end(), name) != names.end()) {
                    filtered.push_back(tool);
                }
            }
        }
        if (filtered.empty()) {
            throw std::invalid_argument("'tool_choice' allowed_tools selects no available tool");
        }
        const std::string mode = allowed.value("mode", std::string("auto"));
        body["tools"]       = filtered;
        body["tool_choice"] = mode == "required" ? "required" : "auto";
        return;
    }
    throw std::invalid_argument("unsupported 'tool_choice' type: " + type);
}

server_chat_completions_context::server_chat_completions_context(
        server_chat_completions_store & store,
        server_chat_completions_request req,
        const llama_vocab * vocab) :
    store(store),
    req(std::move(req)),
    vocab(vocab) {
}

void server_chat_completions_context::decorate(json & response) const {
    int64_t n_reasoning = 0;
    if (response.contains("choices") && response.at("choices").is_array()) {
        for (auto & choice : response.at("choices")) {
            if (!choice.is_object() || !choice.contains("message") || !choice.at("message").is_object()) {
                continue;
            }
            json & message = choice.at("message");
            if (!message.contains("refusal")) {
                message["refusal"] = nullptr;
            }
            if (vocab != nullptr && message.contains("reasoning_content") && message.at("reasoning_content").is_string()) {
                n_reasoning += (int64_t) common_tokenize(vocab, message.at("reasoning_content").get<std::string>(), false, false).size();
            }
        }
    }

    if (response.contains("usage") && response.at("usage").is_object()) {
        json & usage = response.at("usage");
        const int64_t completion_tokens = usage.value("completion_tokens", (int64_t) 0);
        usage["completion_tokens_details"] = json {
            {"reasoning_tokens",           n_reasoning},
            {"text_tokens",                std::max<int64_t>(0, completion_tokens - n_reasoning)},
            {"accepted_prediction_tokens", 0},
            {"rejected_prediction_tokens", 0},
            {"audio_tokens",               0},
        };
        if (!usage.contains("prompt_tokens_details") || !usage.at("prompt_tokens_details").is_object()) {
            usage["prompt_tokens_details"] = json::object();
        }
        json & prompt_details = usage.at("prompt_tokens_details");
        if (!prompt_details.contains("cached_tokens")) {
            prompt_details["cached_tokens"] = 0;
        }
        if (!prompt_details.contains("cache_write_tokens")) {
            prompt_details["cache_write_tokens"] = 0;
        }
        prompt_details["audio_tokens"] = 0;
    }

    if (!req.metadata.is_null()) {
        response["metadata"] = req.metadata;
    }
    if (!req.service_tier.empty()) {
        response["service_tier"] = req.service_tier;
    }
}

json server_chat_completions_context::make_messages(const json & response) const {
    json messages = json::array();
    if (!response.contains("choices") || !response.at("choices").is_array()) {
        return messages;
    }
    for (const auto & choice : response.at("choices")) {
        if (!choice.is_object() || !choice.contains("message") || !choice.at("message").is_object()) {
            continue;
        }
        json message = choice.at("message");
        message["id"] = "msg_" + random_string();
        messages.push_back(std::move(message));
    }
    return messages;
}

void server_chat_completions_context::store_response(const json & response) {
    if (!req.store || !response.is_object()) {
        return;
    }
    const std::string id = response.value("id", std::string());
    if (id.empty()) {
        return;
    }
    server_chat_completions_stored entry;
    entry.response         = response;
    entry.request_messages = req.messages;
    entry.messages         = make_messages(response);
    entry.metadata         = req.metadata;
    entry.user             = req.user;
    entry.safety_identifier = req.safety_identifier;
    entry.created_at       = (int64_t) std::time(0);
    store.put(id, std::move(entry));
}

void server_chat_completions_context::finish(json & response) {
    decorate(response);
    store_response(response);
}

void server_chat_completions_context::finish_stream(json response) {
    const std::string id = response.value("id", std::string());
    if (id.empty()) {
        return;
    }

    pending_stream & p = pending[id];
    if (p.response.is_null()) {
        p.response = response;
        p.response["choices"] = json::array();
        p.prompt_tokens = response_prompt_tokens(response);
    }

    if (response.contains("choices") && response.at("choices").is_array()) {
        for (const auto & choice : response.at("choices")) {
            if (!choice.is_object()) {
                continue;
            }
            const int index = choice.value("index", 0);
            if (p.choices.find(index) == p.choices.end()) {
                p.completion_tokens += response_completion_tokens(response);
            }
            p.choices[index] = choice;
        }
    }

    if ((int) p.choices.size() < req.n) {
        return;
    }

    json choices = json::array();
    for (int i = 0; i < req.n; i++) {
        const auto it = p.choices.find(i);
        if (it != p.choices.end()) {
            choices.push_back(it->second);
        }
    }
    p.response["choices"] = std::move(choices);
    if (!p.response.contains("usage") || !p.response.at("usage").is_object()) {
        p.response["usage"] = json::object();
    }
    p.response["usage"]["completion_tokens"] = p.completion_tokens;
    p.response["usage"]["total_tokens"]      = p.prompt_tokens + p.completion_tokens;

    decorate(p.response);
    store_response(p.response);
    pending.erase(id);
}
