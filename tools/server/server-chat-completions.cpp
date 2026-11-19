#include "server-chat-completions.h"
#include "server-openai-store.h"

#include "common.h"

#include <filesystem>

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


void validate_number(const json & body, const char * key, double min_value, double max_value) {
    if (!body.contains(key) || body.at(key).is_null()) {
        return;
    }
    if (!body.at(key).is_number()) {
        throw std::invalid_argument(std::string("'") + key + "' must be a number");
    }
    const double value = body.at(key).get<double>();
    if (value < min_value || value > max_value) {
        throw std::invalid_argument(std::string("'") + key + "' is out of range");
    }
}

void validate_positive_int(const json & body, const char * key) {
    if (!body.contains(key) || body.at(key).is_null()) {
        return;
    }
    if (!body.at(key).is_number_integer() || body.at(key).get<int64_t>() < 1) {
        throw std::invalid_argument(std::string("'") + key + "' must be a positive integer");
    }
}

void validate_enum(const json & body, const char * key, const std::vector<std::string> & allowed) {
    if (!body.contains(key) || body.at(key).is_null()) {
        return;
    }
    if (!body.at(key).is_string()) {
        throw std::invalid_argument(std::string("'") + key + "' must be a string");
    }
    const std::string value = body.at(key).get<std::string>();
    if (std::find(allowed.begin(), allowed.end(), value) == allowed.end()) {
        throw std::invalid_argument(std::string("unsupported '") + key + "' value: " + value);
    }
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

json server_chat_completions_stored::to_json() const {
    return json {
        {"response",          response},
        {"request_messages",  request_messages},
        {"messages",          messages},
        {"metadata",          metadata},
        {"user",              user},
        {"safety_identifier", safety_identifier},
        {"created_at",        created_at},
    };
}

bool server_chat_completions_stored::from_json(const json & data, server_chat_completions_stored & out) {
    if (!data.is_object()) {
        return false;
    }
    out.response          = data.value("response", json(nullptr));
    out.request_messages  = data.value("request_messages", json::array());
    out.messages          = data.value("messages", json::array());
    out.metadata          = data.value("metadata", json(nullptr));
    out.user              = data.value("user", std::string());
    out.safety_identifier = data.value("safety_identifier", std::string());
    out.created_at        = data.value("created_at", (int64_t) 0);
    return !out.response.is_null();
}

server_chat_completions_store::server_chat_completions_store(
        const std::string & base_path, size_t max_items_, int64_t ttl_seconds_) :
    path(base_path),
    max_items(max_items_),
    ttl_seconds(ttl_seconds_) {
    if (!path.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(std::filesystem::path(path) / "chat_completions", ec);
    }
}

std::string server_chat_completions_store::file_path(const std::string & id) const {
    return server_openai_store::item_path(path, "chat_completions", id);
}

bool server_chat_completions_store::load_one_locked(const std::string & id, server_chat_completions_stored & out) {
    if (path.empty()) {
        return false;
    }
    json data;
    if (!server_openai_store::read_json(file_path(id), data)) {
        return false;
    }
    server_chat_completions_stored stored;
    if (!server_chat_completions_stored::from_json(data, stored)) {
        return false;
    }
    if (ttl_seconds > 0 && server_openai_store::now() - stored.created_at > ttl_seconds) {
        server_openai_store::remove_file(file_path(id));
        return false;
    }
    items[id] = stored;
    out = std::move(stored);
    return true;
}

void server_chat_completions_store::prune_locked() {
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

bool server_chat_completions_store::get(const std::string & id, server_chat_completions_stored & out) {
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

void server_chat_completions_store::put(const std::string & id, server_chat_completions_stored item) {
    std::lock_guard<std::mutex> lock(mutex);
    prune_locked();
    items[id] = std::move(item);
    prune_locked();
    const auto it = items.find(id);
    if (it != items.end() && !path.empty()) {
        server_openai_store::write_json(file_path(id), it->second.to_json());
    }
}

bool server_chat_completions_store::erase(const std::string & id) {
    std::lock_guard<std::mutex> lock(mutex);
    const bool erased = items.erase(id) > 0;
    if (!path.empty()) {
        server_openai_store::remove_file(file_path(id));
    }
    return erased;
}

bool server_chat_completions_store::update_metadata(const std::string & id, const json & metadata) {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = items.find(id);
    if (it == items.end()) {
        server_chat_completions_stored stored;
        if (!load_one_locked(id, stored)) {
            return false;
        }
        it = items.find(id);
    }
    it->second.metadata = metadata;
    if (it->second.response.is_object()) {
        it->second.response["metadata"] = metadata;
    }
    if (!path.empty()) {
        server_openai_store::write_json(file_path(id), it->second.to_json());
    }
    return true;
}

std::vector<std::pair<std::string, server_chat_completions_stored>> server_chat_completions_store::list() {
    std::lock_guard<std::mutex> lock(mutex);
    if (!path.empty()) {
        const std::string dir = (std::filesystem::path(path) / "chat_completions").string();
        for (const auto & file : server_openai_store::list_files(dir)) {
            const std::string id = std::filesystem::path(file).stem().string();
            if (items.find(id) != items.end()) {
                continue;
            }
            server_chat_completions_stored stored;
            json data;
            if (server_openai_store::read_json(file, data) && server_chat_completions_stored::from_json(data, stored)) {
                if (ttl_seconds == 0 || server_openai_store::now() - stored.created_at <= ttl_seconds) {
                    items[id] = std::move(stored);
                } else {
                    server_openai_store::remove_file(file);
                }
            }
        }
        prune_locked();
    }
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
    req.model              = json_value(body, "model", std::string());
    req.store              = body.value("store", false);
    req.parallel_tool_calls = body.value("parallel_tool_calls", true);
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

    if (body.contains("store") && !body.at("store").is_null() && !body.at("store").is_boolean()) {
        throw std::invalid_argument("'store' must be a boolean");
    }
    validate_number(body, "temperature", 0.0, 2.0);
    validate_number(body, "top_p", 0.0, 1.0);
    validate_number(body, "frequency_penalty", -2.0, 2.0);
    validate_number(body, "presence_penalty", -2.0, 2.0);
    validate_positive_int(body, "max_completion_tokens");
    validate_positive_int(body, "max_tokens");
    validate_enum(body, "service_tier", {"auto", "default", "flex", "scale", "priority", "fast"});
    validate_enum(body, "reasoning_effort", {"none", "minimal", "low", "medium", "high", "max"});
    validate_enum(body, "verbosity", {"low", "medium", "high"});

    if (body.contains("stop") && !body.at("stop").is_null()) {
        const json & stop = body.at("stop");
        if (stop.is_array()) {
            if (stop.size() > 4) {
                throw std::invalid_argument("'stop' supports at most 4 strings");
            }
            for (const auto & value : stop) {
                if (!value.is_string()) {
                    throw std::invalid_argument("'stop' must contain strings");
                }
            }
        } else if (!stop.is_string()) {
            throw std::invalid_argument("'stop' must be a string or an array of strings");
        }
    }

    if (body.contains("top_logprobs") && !body.at("top_logprobs").is_null()) {
        if (!body.at("top_logprobs").is_number_integer()) {
            throw std::invalid_argument("'top_logprobs' must be an integer");
        }
        const int64_t value = body.at("top_logprobs").get<int64_t>();
        if (value < 0 || value > 20) {
            throw std::invalid_argument("'top_logprobs' must be between 0 and 20");
        }
        if (!body.value("logprobs", false)) {
            throw std::invalid_argument("'top_logprobs' requires 'logprobs' to be true");
        }
    }

    if (body.contains("modalities") && !body.at("modalities").is_null()) {
        const json & modalities = body.at("modalities");
        if (!modalities.is_array()) {
            throw std::invalid_argument("'modalities' must be an array");
        }
        for (const auto & value : modalities) {
            if (!value.is_string() || value.get<std::string>() != "text") {
                throw std::invalid_argument("only the 'text' output modality is supported");
            }
        }
    }
    if (body.contains("audio") && !body.at("audio").is_null()) {
        throw std::invalid_argument("audio output is not supported");
    }
    if (body.contains("web_search_options") && !body.at("web_search_options").is_null()) {
        throw std::invalid_argument("'web_search_options' is not supported");
    }
    if (body.contains("prediction") && !body.at("prediction").is_null()) {
        throw std::invalid_argument("'prediction' is not supported");
    }
    if (body.contains("tools") && body.at("tools").is_array()) {
        for (const auto & tool : body.at("tools")) {
            if (!tool.is_object()) {
                throw std::invalid_argument("'tools' entries must be objects");
            }
            const std::string type = tool.value("type", std::string("function"));
            if (type != "function") {
                throw std::invalid_argument("unsupported tool type: " + type);
            }
        }
    }
    return req;
}

void server_chat_completions_normalize_request(json & body) {
    // legacy functions -> tools
    if (body.contains("functions") && !body.at("functions").is_null()) {
        if (body.contains("tools") && !body.at("tools").is_null()) {
            throw std::invalid_argument("'functions' and 'tools' cannot be used together");
        }
        const json & functions = body.at("functions");
        if (!functions.is_array()) {
            throw std::invalid_argument("'functions' must be an array");
        }
        json tools = json::array();
        for (const auto & function : functions) {
            if (!function.is_object() || !function.contains("name")) {
                throw std::invalid_argument("each legacy function requires a 'name'");
            }
            tools.push_back(json {
                {"type", "function"},
                {"function", function},
            });
        }
        body["tools"] = tools;
        body.erase("functions");
    }

    // legacy function_call -> tool_choice
    if (body.contains("function_call") && !body.at("function_call").is_null()) {
        const json function_call = body.at("function_call");
        if (function_call.is_string()) {
            body["tool_choice"] = function_call;
        } else if (function_call.is_object() && function_call.contains("name")) {
            body["tool_choice"] = json {
                {"type", "function"},
                {"function", json { {"name", function_call.at("name")} }},
            };
        } else {
            throw std::invalid_argument("unsupported 'function_call'");
        }
        body.erase("function_call");
    }

    // legacy message shape
    if (body.contains("messages") && body.at("messages").is_array()) {
        json & messages = body["messages"];
        for (auto & message : messages) {
            if (!message.is_object()) {
                continue;
            }
            if (message.value("role", std::string()) == "function") {
                message["role"] = "tool";
            }
            if (message.contains("function_call") && !message.contains("tool_calls")) {
                const json & function_call = message.at("function_call");
                if (!function_call.is_object() || !function_call.contains("name")) {
                    throw std::invalid_argument("legacy assistant function_call requires a 'name'");
                }
                const json arguments_json = function_call.value("arguments", json(""));
                const std::string arguments = arguments_json.is_string() ? arguments_json.get<std::string>() : arguments_json.dump();
                message["tool_calls"] = json::array({ json {
                    {"id", "call_" + random_string()},
                    {"type", "function"},
                    {"function", json {
                        {"name", function_call.at("name")},
                        {"arguments", arguments},
                    }},
                } });
                message.erase("function_call");
            }
        }
    }

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
        server_chat_completions_request req) :
    store(store),
    req(std::move(req)) {
}

void server_chat_completions_context::decorate(json & response) const {
    // sum the exact count of each choice; the temporary field is removed here
    int64_t n_reasoning   = 0;
    bool    has_reasoning = false;
    if (response.contains("choices") && response.at("choices").is_array()) {
        has_reasoning = true;
        for (auto & choice : response.at("choices")) {
            if (!choice.is_object() || !choice.contains("message") || !choice.at("message").is_object()) {
                has_reasoning = false;
                continue;
            }
            json & message = choice.at("message");
            if (!message.contains("refusal")) {
                message["refusal"] = nullptr;
            }
            if (!req.parallel_tool_calls && message.contains("tool_calls") &&
                    message.at("tool_calls").is_array() && message.at("tool_calls").size() > 1) {
                json kept = json::array();
                kept.push_back(message.at("tool_calls")[0]);
                message["tool_calls"] = std::move(kept);
            }
            if (choice.contains("n_reasoning_tokens") && choice.at("n_reasoning_tokens").is_number_integer()) {
                n_reasoning += choice.at("n_reasoning_tokens").get<int64_t>();
            } else {
                has_reasoning = false;
            }
            choice.erase("n_reasoning_tokens");
        }
    }

    if (response.contains("usage") && response.at("usage").is_object()) {
        json & usage = response.at("usage");
        const int64_t completion_tokens = usage.value("completion_tokens", (int64_t) 0);
        if (has_reasoning) {
            usage["completion_tokens_details"] = json {
                {"reasoning_tokens",           n_reasoning},
                {"text_tokens",                std::max<int64_t>(0, completion_tokens - n_reasoning)},
                {"accepted_prediction_tokens", 0},
                {"rejected_prediction_tokens", 0},
                {"audio_tokens",               0},
            };
        } else if (!usage.contains("completion_tokens_details") || !usage.at("completion_tokens_details").is_object()) {
            usage["completion_tokens_details"] = json {
                {"reasoning_tokens",           0},
                {"text_tokens",                completion_tokens},
                {"accepted_prediction_tokens", 0},
                {"rejected_prediction_tokens", 0},
                {"audio_tokens",               0},
            };
        }
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


json server_chat_completions_context::filter_chunks(json chunks) const {
    auto decorate_chunk = [this](json & chunk) {
        if (chunk.is_object() && !req.service_tier.empty()) {
            chunk["service_tier"] = req.service_tier;
        }
    };
    if (req.parallel_tool_calls) {
        if (chunks.is_array()) {
            for (auto & chunk : chunks) {
                decorate_chunk(chunk);
            }
        } else {
            decorate_chunk(chunks);
        }
        return chunks;
    }
    auto filter_chunk = [this, &decorate_chunk](json & chunk) {
        decorate_chunk(chunk);
        if (!chunk.is_object() || !chunk.contains("choices") || !chunk.at("choices").is_array()) {
            return;
        }
        for (auto & choice : chunk["choices"]) {
            if (!choice.is_object() || !choice.contains("delta") || !choice.at("delta").is_object()) {
                continue;
            }
            json & delta = choice["delta"];
            if (!delta.contains("tool_calls") || !delta.at("tool_calls").is_array()) {
                continue;
            }
            json kept = json::array();
            for (const auto & tool_call : delta.at("tool_calls")) {
                if (tool_call.value("index", 0) == 0) {
                    kept.push_back(tool_call);
                }
            }
            if (kept.empty()) {
                delta.erase("tool_calls");
            } else {
                delta["tool_calls"] = std::move(kept);
            }
        }
    };

    if (chunks.is_array()) {
        for (auto & chunk : chunks) {
            filter_chunk(chunk);
        }
    } else {
        filter_chunk(chunks);
    }
    return chunks;
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
