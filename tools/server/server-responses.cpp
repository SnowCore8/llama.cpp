#include "server-responses.h"
#include "server-conversations.h"
#include "server-openai-store.h"

#include "common.h"
#include "base64.hpp"

#include <filesystem>

#include <algorithm>
#include <chrono>
#include <ctime>
#include <stdexcept>

namespace {

json response_logprob_json(const completion_token_output & p) {
    json top = json::array();
    for (const auto & t : p.probs) {
        top.push_back(json {
            {"token",   t.txt},
            {"logprob", completion_token_output::logarithm(t.prob)},
        });
    }
    std::string text = p.text_to_send;
    text.resize(validate_utf8(text));
    return json {
        {"token",              text},
        {"bytes",              p.str_to_bytes(p.text_to_send)},
        {"logprob",            completion_token_output::logarithm(p.prob)},
        {"top_logprobs",       top},
    };
}


json normalize_input(const json & input) {
    json items = json::array();
    if (input.is_string()) {
        items.push_back(json {
            {"type", "message"},
            {"role", "user"},
            {"content", json::array({ json {
                {"type", "input_text"},
                {"text", input.get<std::string>()},
            }})},
        });
        return items;
    }
    if (input.is_object()) {
        items.push_back(input);
        return items;
    }
    if (!input.is_array()) {
        throw std::invalid_argument("'input' must be a string or an array");
    }
    for (const auto & item : input) {
        if (!item.is_object()) {
            throw std::invalid_argument("input items must be objects");
        }
        items.push_back(item);
    }
    return items;
}

} // namespace

//
// server_responses_event_log
//

void server_responses_event_log::append(int64_t seq, std::string event) {
    if (event.empty()) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (done) {
            return;
        }
        n_bytes += event.size();
        events.emplace_back(seq, std::move(event));
        // keep the newest events, the oldest ones are dropped when the cap is reached
        while (n_bytes > max_bytes && events.size() > 1) {
            n_bytes -= events.front().second.size();
            events.pop_front();
        }
        first_seq = events.front().first;
    }
    cv.notify_all();
}

void server_responses_event_log::close() {
    {
        std::lock_guard<std::mutex> lock(mutex);
        done = true;
    }
    cv.notify_all();
}

bool server_responses_event_log::is_done() const {
    std::lock_guard<std::mutex> lock(mutex);
    return done;
}

bool server_responses_event_log::validate(int64_t next_seq) const {
    std::lock_guard<std::mutex> lock(mutex);
    return next_seq >= first_seq;
}

bool server_responses_event_log::read(int64_t & next_seq, std::string & out, const std::function<bool()> & should_stop) {
    std::unique_lock<std::mutex> lock(mutex);
    while (true) {
        if (should_stop && should_stop()) {
            return false;
        }
        if (next_seq < first_seq) {
            return false; // the reader fell behind the retained window
        }
        if (!events.empty() && next_seq >= events.front().first) {
            const size_t index = (size_t) (next_seq - events.front().first);
            if (index < events.size()) {
                for (size_t i = index; i < events.size(); i++) {
                    out += events[i].second;
                }
                next_seq = events.back().first + 1;
                return true;
            }
        }
        if (done) {
            return false;
        }
        // wake up regularly to notice a dropped connection
        cv.wait_for(lock, std::chrono::milliseconds(200));
    }
}

json server_responses_stored::to_json() const {
    return json {
        {"response",    response},
        {"input_items", input_items},
        {"created_at",  created_at},
    };
}

bool server_responses_stored::from_json(const json & data, server_responses_stored & out) {
    if (!data.is_object()) {
        return false;
    }
    out.response    = data.value("response", json(nullptr));
    out.input_items = data.value("input_items", json::array());
    out.created_at  = data.value("created_at", (int64_t) 0);
    return !out.response.is_null();
}

server_responses_store::server_responses_store(const std::string & base_path, size_t max_items_, int64_t ttl_seconds_) :
    path(base_path),
    max_items(max_items_),
    ttl_seconds(ttl_seconds_) {
    if (!path.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(std::filesystem::path(path) / "responses", ec);
    }
}

std::string server_responses_store::file_path(const std::string & id) const {
    return server_openai_store::item_path(path, "responses", id);
}

void server_responses_store::prune_locked() {
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

bool server_responses_store::get(const std::string & id, server_responses_stored & out) {
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

    if (!path.empty()) {
        json data;
        if (server_openai_store::read_json(file_path(id), data)) {
            server_responses_stored stored;
            if (server_responses_stored::from_json(data, stored)) {
                if (ttl_seconds > 0 && now - stored.created_at > ttl_seconds) {
                    server_openai_store::remove_file(file_path(id));
                } else {
                    items[id] = stored;
                    out = std::move(stored);
                    return true;
                }
            }
        }
    }
    return false;
}

void server_responses_store::put(const std::string & id, server_responses_stored item) {
    std::lock_guard<std::mutex> lock(mutex);
    prune_locked();
    items[id] = std::move(item);
    prune_locked();
    const auto it = items.find(id);
    if (it != items.end() && !path.empty()) {
        server_openai_store::write_json(file_path(id), it->second.to_json());
    }
}

bool server_responses_store::erase(const std::string & id) {
    std::lock_guard<std::mutex> lock(mutex);
    const bool erased = items.erase(id) > 0;
    if (!path.empty()) {
        server_openai_store::remove_file(file_path(id));
    }
    return erased;
}

server_responses_request server_responses_parse_request(const json & body) {
    if (!body.is_object()) {
        throw std::invalid_argument("request body must be a JSON object");
    }
    if (!body.contains("input")) {
        throw std::invalid_argument("'input' is required");
    }

    server_responses_request req;
    req.model                     = body.value("model", std::string());
    req.stream                    = body.value("stream", false);
    req.store                     = body.value("store", true);
    req.timings_per_token         = body.value("timings_per_token", false);
    req.return_progress           = body.value("return_progress", false);
    req.id_slot                   = body.value("id_slot", -1);
    req.background                = body.value("background", false);
    req.parallel_tool_calls       = body.value("parallel_tool_calls", true);
    req.previous_response_id      = body.value("previous_response_id", std::string());
    req.truncation                = body.value("truncation", std::string("disabled"));
    req.user                      = body.value("user", std::string());
    req.safety_identifier         = body.value("safety_identifier", std::string());
    req.service_tier              = body.value("service_tier", std::string());
    req.instructions              = body.value("instructions", json(nullptr));
    req.metadata                  = body.value("metadata", json(nullptr));
    req.reasoning                 = body.value("reasoning", json(nullptr));
    req.text                      = body.value("text", json(nullptr));
    req.include                   = body.value("include", json(nullptr));
    req.max_output_tokens         = body.value("max_output_tokens", -1);
    req.temperature               = body.value("temperature", -1.0);
    req.top_p                     = body.value("top_p", -1.0);

    if (body.contains("conversation") && !body.at("conversation").is_null()) {
        const json & conversation = body.at("conversation");
        if (conversation.is_string()) {
            req.conversation_id = conversation.get<std::string>();
        } else if (conversation.is_object()) {
            req.conversation_id = conversation.value("id", std::string());
        } else {
            throw std::invalid_argument("'conversation' must be a string or an object");
        }
        if (req.conversation_id.empty()) {
            throw std::invalid_argument("'conversation' is missing an id");
        }
    }
    if (!req.previous_response_id.empty() && !req.conversation_id.empty()) {
        throw std::invalid_argument("'conversation' and 'previous_response_id' cannot be used together");
    }
    if (!req.service_tier.empty()) {
        const std::vector<std::string> tiers = { "auto", "default", "flex", "scale", "priority", "fast" };
        if (std::find(tiers.begin(), tiers.end(), req.service_tier) == tiers.end()) {
            throw std::invalid_argument("unsupported 'service_tier' value: " + req.service_tier);
        }
    }
    if (req.reasoning.is_object()) {
        const std::vector<std::string> summaries = { "auto", "concise", "detailed" };
        for (const char * key : { "summary", "generate_summary" }) {
            const std::string value = req.reasoning.value(key, std::string());
            if (!value.empty() && std::find(summaries.begin(), summaries.end(), value) == summaries.end()) {
                throw std::invalid_argument(std::string("unsupported 'reasoning.") + key + "' value: " + value);
            }
        }
    }
    if (body.contains("prompt") && !body.at("prompt").is_null()) {
        throw std::invalid_argument("'prompt' templates are not supported");
    }
    if (req.truncation != "disabled" && req.truncation != "auto") {
        throw std::invalid_argument("'truncation' must be 'auto' or 'disabled'");
    }
    if (req.temperature >= 0.0 && req.temperature > 2.0) {
        throw std::invalid_argument("'temperature' must be between 0 and 2");
    }
    if (req.top_p >= 0.0 && req.top_p > 1.0) {
        throw std::invalid_argument("'top_p' must be between 0 and 1");
    }
    if (req.max_output_tokens == 0) {
        throw std::invalid_argument("'max_output_tokens' must be greater than 0");
    }
    if (body.contains("top_logprobs") && !body.at("top_logprobs").is_null()) {
        if (!body.at("top_logprobs").is_number_integer()) {
            throw std::invalid_argument("'top_logprobs' must be an integer");
        }
        req.top_logprobs = (int32_t) body.at("top_logprobs").get<int64_t>();
        if (req.top_logprobs < 0 || req.top_logprobs > 20) {
            throw std::invalid_argument("'top_logprobs' must be between 0 and 20");
        }
    }
    if (!req.include.is_null()) {
        if (!req.include.is_array()) {
            throw std::invalid_argument("'include' must be an array");
        }
        for (const auto & value : req.include) {
            if (!value.is_string()) {
                throw std::invalid_argument("'include' values must be strings");
            }
            const std::string include_value = value.get<std::string>();
            if (include_value == "message.output_text.logprobs") {
                req.include_logprobs = true;
            } else if (include_value == "reasoning.encrypted_content") {
                throw std::invalid_argument("'include' does not support encrypted reasoning content");
            } else {
                throw std::invalid_argument("unsupported 'include' value: " + include_value);
            }
        }
    }
    server_validate_metadata(req.metadata);

    if (body.contains("tools")) {
        const json & tools = body.at("tools");
        if (!tools.is_array()) {
            throw std::invalid_argument("'tools' must be an array");
        }
        for (const auto & tool : tools) {
            if (!tool.is_object()) {
                throw std::invalid_argument("tool entries must be objects");
            }
            const std::string type = tool.value("type", std::string());
            if (type != "function") {
                throw std::invalid_argument("unsupported tool type: " + type);
            }
            common_chat_tool parsed;
            parsed.name = tool.value("name", std::string());
            if (parsed.name.empty()) {
                throw std::invalid_argument("function tools require a 'name'");
            }
            parsed.description = tool.value("description", std::string());
            const json parameters = tool.value("parameters", json(nullptr));
            parsed.parameters = parameters.is_object() ? parameters.dump() : std::string("{}");
            parsed.defer_loading = tool.value("defer_loading", false);
            req.tools_parsed.push_back(std::move(parsed));
        }
        req.tools = tools;
    }

    req.tool_choice = body.value("tool_choice", json("auto"));
    if (req.tool_choice.is_string()) {
        const std::string choice = req.tool_choice.get<std::string>();
        if (choice == "auto") {
            req.tool_choice_parsed = COMMON_CHAT_TOOL_CHOICE_AUTO;
        } else if (choice == "none") {
            req.tool_choice_parsed = COMMON_CHAT_TOOL_CHOICE_NONE;
        } else if (choice == "required") {
            req.tool_choice_parsed = COMMON_CHAT_TOOL_CHOICE_REQUIRED;
        } else {
            throw std::invalid_argument("'tool_choice' must be auto, none or required");
        }
    } else if (req.tool_choice.is_object()) {
        const std::string type = req.tool_choice.value("type", std::string());
        if (type != "function") {
            throw std::invalid_argument("unsupported 'tool_choice' type: " + type);
        }
        req.tool_choice_name = req.tool_choice.value("name", std::string());
        if (req.tool_choice_name.empty()) {
            throw std::invalid_argument("'tool_choice' function requires a 'name'");
        }
        req.tool_choice_parsed = COMMON_CHAT_TOOL_CHOICE_REQUIRED;
    } else {
        throw std::invalid_argument("'tool_choice' must be a string or an object");
    }

    if ((req.tool_choice_parsed == COMMON_CHAT_TOOL_CHOICE_REQUIRED || !req.tool_choice_name.empty()) && req.tools_parsed.empty()) {
        throw std::invalid_argument("'tool_choice' requires at least one tool");
    }
    if (!req.tool_choice_name.empty()) {
        const auto it = std::find_if(req.tools_parsed.begin(), req.tools_parsed.end(), [&](const common_chat_tool & tool) {
            return tool.name == req.tool_choice_name;
        });
        if (it == req.tools_parsed.end()) {
            throw std::invalid_argument("tool_choice refers to an unknown tool: " + req.tool_choice_name);
        }
    }

    req.input       = body.at("input");
    req.input_items = normalize_input(req.input);
    for (auto & item : req.input_items) {
        if (item.is_object() && !item.contains("id")) {
            item["id"] = "item_" + random_string();
        }
    }

    return req;
}

void server_responses_resolve_input(server_responses_request & req, server_responses_store & store, server_conversations_store * conversations) {
    json items = json::array();
    if (!req.conversation_id.empty()) {
        server_conversation conversation;
        if (conversations == nullptr || !conversations->get(req.conversation_id, conversation)) {
            throw std::invalid_argument("conversation not found: " + req.conversation_id);
        }
        for (const auto & item : conversation.items) {
            items.push_back(item);
        }
    }
    if (!req.previous_response_id.empty()) {
        server_responses_stored previous;
        if (!store.get(req.previous_response_id, previous)) {
            throw std::invalid_argument("previous response not found: " + req.previous_response_id);
        }
        for (const auto & item : previous.input_items) {
            items.push_back(item);
        }
        if (previous.response.is_object()) {
            for (const auto & item : previous.response.value("output", json::array())) {
                items.push_back(item);
            }
        }
    }
    for (const auto & item : req.input_items) {
        items.push_back(item);
    }
    req.effective_input = std::move(items);
}

json server_responses_build_params(
        const llama_vocab * vocab,
        const server_chat_params & opt,
        const server_responses_request & req,
        common_chat_session & session,
        std::vector<raw_buffer> & files) {
    common_chat_templates_inputs inputs;
    inputs.use_jinja             = opt.use_jinja;
    inputs.reasoning_format      = opt.reasoning_format;
    inputs.force_pure_content    = opt.force_pure_content;
    inputs.parallel_tool_calls   = req.parallel_tool_calls;
    inputs.chat_template_kwargs  = opt.chat_template_kwargs;
    inputs.enable_thinking       = opt.enable_thinking;

    auto convert_item = [](const json & item) -> json {
        if (!item.is_object()) {
            throw std::invalid_argument("input items must be objects");
        }
        const std::string type = item.value("type", std::string());
        if (type.empty() || type == "message") {
            std::string role = item.value("role", std::string("user"));
            if (role == "developer") {
                role = "system";
            }
            const json content_in = item.value("content", json(""));
            json content;
            if (content_in.is_string()) {
                content = content_in;
            } else if (content_in.is_array()) {
                json parts = json::array();
                for (const auto & part : content_in) {
                    if (!part.is_object()) {
                        throw std::invalid_argument("content parts must be objects");
                    }
                    const std::string part_type = part.value("type", std::string());
                    if (part_type == "input_text" || part_type == "output_text" || part_type == "text") {
                        parts.push_back(json {
                            {"type", "text"},
                            {"text", part.value("text", std::string())},
                        });
                    } else if (part_type == "input_image") {
                        const std::string url = part.value("image_url", part.value("url", std::string()));
                        if (url.empty()) {
                            throw std::invalid_argument("input_image requires 'image_url'");
                        }
                        parts.push_back(json {
                            {"type", "image_url"},
                            {"image_url", json { {"url", url} }},
                        });
                    } else if (part_type == "refusal") {
                        parts.push_back(json {
                            {"type", "text"},
                            {"text", part.value("refusal", std::string())},
                        });
                    } else if (part_type == "input_file") {
                        const std::string file_data = part.value("file_data", std::string());
                        if (file_data.empty()) {
                            throw std::invalid_argument("input_file requires 'file_data'; file_id and file_url are not supported");
                        }
                        std::string decoded;
                        try {
                            if (file_data.rfind("data:", 0) == 0) {
                                const size_t comma = file_data.find(',');
                                if (comma == std::string::npos) {
                                    throw std::invalid_argument("invalid input_file data URI");
                                }
                                const std::string header = file_data.substr(0, comma);
                                const std::string payload = file_data.substr(comma + 1);
                                decoded = header.find("base64") != std::string::npos ? base64::decode(payload) : payload;
                            } else {
                                decoded = base64::decode(file_data);
                            }
                        } catch (const std::exception &) {
                            throw std::invalid_argument("input_file data is not valid base64");
                        }
                        bool is_text = true;
                        for (const unsigned char c : decoded) {
                            if (c == 0 || (c < 9) || (c > 13 && c < 32)) {
                                is_text = false;
                                break;
                            }
                        }
                        if (!is_text) {
                            throw std::invalid_argument("only text input_file content is supported");
                        }
                        parts.push_back(json {
                            {"type", "text"},
                            {"text", decoded},
                        });
                    } else {
                        throw std::invalid_argument("unsupported content part type: " + part_type);
                    }
                }
                content = parts;
            } else {
                throw std::invalid_argument("message content must be a string or an array");
            }
            return json {
                {"role", role},
                {"content", content},
            };
        }
        if (type == "function_call") {
            const json arguments_in = item.value("arguments", json(""));
            const std::string arguments = arguments_in.is_string() ? arguments_in.get<std::string>() : arguments_in.dump();
            return json {
                {"role", "assistant"},
                {"content", nullptr},
                {"tool_calls", json::array({ json {
                    {"id", item.value("call_id", item.value("id", std::string()))},
                    {"type", "function"},
                    {"function", json {
                        {"name", item.value("name", std::string())},
                        {"arguments", arguments},
                    }},
                } })},
            };
        }
        if (type == "function_call_output") {
            const json output = item.value("output", json(""));
            return json {
                {"role", "tool"},
                {"tool_call_id", item.value("call_id", std::string())},
                {"content", output.is_string() ? output.get<std::string>() : output.dump()},
            };
        }
        if (type == "reasoning") {
            return nullptr;
        }
        throw std::invalid_argument("unsupported input item type: " + type);
    };

    json chat_messages = json::array();
    auto add_chat_message = [&](json message) {
        if (message.contains("content") && message.at("content").is_array()) {
            oaicompat_content_load_media(message["content"], opt, files);
        }
        chat_messages.push_back(std::move(message));
    };

    if (!req.instructions.is_null()) {
        if (req.instructions.is_string()) {
            add_chat_message(json {
                {"role", "system"},
                {"content", req.instructions.get<std::string>()},
            });
        } else if (req.instructions.is_array()) {
            for (const auto & item : req.instructions) {
                json message = convert_item(item);
                if (!message.is_null()) {
                    add_chat_message(std::move(message));
                }
            }
        } else {
            throw std::invalid_argument("'instructions' must be a string or an array");
        }
    }

    for (const auto & item : req.effective_input) {
        json message = convert_item(item);
        if (!message.is_null()) {
            add_chat_message(std::move(message));
        }
    }

    std::string verbosity;
    if (req.text.is_object()) {
        verbosity = req.text.value("verbosity", std::string());
    }
    if (!verbosity.empty()) {
        std::string instruction = "Answer with moderate detail.";
        if (verbosity == "low") {
            instruction = "Answer concisely.";
        } else if (verbosity == "high") {
            instruction = "Answer in detail.";
        } else if (verbosity != "medium") {
            throw std::invalid_argument("unsupported 'text.verbosity' value: " + verbosity);
        }
        json messages_with_hint = json::array();
        messages_with_hint.push_back(json {
            {"role", "system"},
            {"content", instruction},
        });
        for (const auto & message : chat_messages) {
            messages_with_hint.push_back(message);
        }
        chat_messages = std::move(messages_with_hint);
    }
    inputs.messages = common_chat_msgs_parse_oaicompat(chat_messages);
    inputs.tools = req.tools_parsed;
    if (!req.tool_choice_name.empty()) {
        const auto it = std::find_if(inputs.tools.begin(), inputs.tools.end(), [&](const common_chat_tool & tool) {
            return tool.name == req.tool_choice_name;
        });
        if (it == inputs.tools.end()) {
            throw std::invalid_argument("tool_choice refers to an unknown tool: " + req.tool_choice_name);
        }
        inputs.tools = { *it };
    }
    inputs.tool_choice = req.tool_choice_parsed;

    if (!inputs.tools.empty() && !opt.use_jinja) {
        throw std::invalid_argument("tools require --jinja");
    }

    if (req.text.is_object() && req.text.contains("format") && !req.text.at("format").is_null()) {
        const json & format = req.text.at("format");
        if (!format.is_object()) {
            throw std::invalid_argument("'text.format' must be an object");
        }
        const std::string type = format.value("type", std::string("text"));
        if (type == "json_schema") {
            if (!format.contains("schema")) {
                throw std::invalid_argument("'text.format.schema' is required");
            }
            inputs.json_schema = format.at("schema").dump();
        } else if (type == "json_object") {
            inputs.json_schema = R"({"type":"object"})";
        } else if (type != "text") {
            throw std::invalid_argument("unsupported 'text.format.type': " + type);
        }
    }

    if (req.reasoning.is_object()) {
        const std::string effort = req.reasoning.value("effort", std::string());
        if (effort == "none") {
            inputs.enable_thinking = false;
        } else if (!effort.empty()) {
            inputs.chat_template_kwargs["reasoning_effort"] = json(effort).dump();
        }
    }

    common_chat_session_params session_params;
    session = common_chat_session(opt.tmpls.get(), vocab, inputs, session_params);

    json data;
    data["prompt"]      = session.prompt().str();
    data["stream"]      = req.stream;
    data["temperature"] = req.temperature >= 0.0 ? req.temperature : 1.0;
    data["top_p"]       = req.top_p >= 0.0 ? req.top_p : 1.0;
    if (req.max_output_tokens > 0) {
        data["n_predict"] = req.max_output_tokens;
    }
    data["timings_per_token"] = req.timings_per_token;
    data["return_progress"]   = req.return_progress;
    if (req.include_logprobs) {
        data["n_probs"] = req.top_logprobs > 0 ? req.top_logprobs : 20;
    }
    if (req.id_slot >= 0) {
        data["id_slot"] = req.id_slot;
    }

    if (!session.thinking_end_tags().empty() && opt.reasoning_budget >= 0) {
        data["reasoning_budget_tokens"]    = opt.reasoning_budget;
        data["reasoning_budget_start_tag"] = session.thinking_start_tag();
        data["reasoning_budget_end_tags"]  = session.thinking_end_tags();
        data["reasoning_budget_message"]   = opt.reasoning_budget_message;
    }

    return data;
}

server_responses_context::server_responses_context(
        server_responses_store & store,
        server_responses_request req,
        std::string model_name,
        const llama_vocab * vocab,
        server_conversations_store * conversations) :
    store(store),
    req(std::move(req)),
    model_name(std::move(model_name)),
    vocab(vocab),
    conversations(conversations) {
    id         = "resp_" + random_string();
    created_at = (int64_t) std::time(0);
    if (this->req.reasoning.is_object()) {
        const std::string summary = this->req.reasoning.value("summary", std::string());
        const std::string generated = this->req.reasoning.value("generate_summary", std::string());
        reasoning_summary = !summary.empty() || !generated.empty();
    }
    // a background response is generated as a stream, its events can be replayed to new
    // connections. background + stream is served synchronously and needs no log
    if (this->req.background && !this->req.stream) {
        event_log = std::make_shared<server_responses_event_log>();
    }
}

json server_responses_context::push(json & events, const std::string & type, json data) const {
    data["type"]            = type;
    data["sequence_number"] = seq++;
    json event = {
        {"event", type},
        {"data", std::move(data)},
    };
    events.push_back(std::move(event));
    return events.back();
}

void server_responses_context::publish(const json & events) const {
    if (!event_log) {
        return;
    }
    auto publish_one = [this](const json & event) {
        event_log->append(event.at("data").value("sequence_number", (int64_t) 0), format_oai_resp_sse(event));
    };
    if (events.is_array()) {
        for (const auto & event : events) {
            publish_one(event);
        }
    } else {
        publish_one(events);
    }
}

json server_responses_context::make_response(
        const std::string & status,
        const json & output,
        const json & usage,
        const json & error,
        const json & incomplete_details) const {
    const std::string response_model = req.model.empty() ? model_name : req.model;
    json response = {
        {"id",                 id},
        {"object",             "response"},
        {"created_at",         (double) created_at},
        {"status",             status},
        {"background",         req.background},
        {"error",              error},
        {"incomplete_details", incomplete_details},
        {"instructions",       req.instructions},
        {"max_output_tokens",  req.max_output_tokens > 0 ? json(req.max_output_tokens) : json(nullptr)},
        {"model",              response_model},
        {"output",             output},
        {"parallel_tool_calls", req.parallel_tool_calls},
        {"previous_response_id", req.previous_response_id.empty() ? json(nullptr) : json(req.previous_response_id)},
        {"conversation",       req.conversation_id.empty() ? json(nullptr) : json { {"id", req.conversation_id} }},
        {"reasoning",          req.reasoning},
        {"store",              req.store},
        {"temperature",        req.temperature >= 0.0 ? json(req.temperature) : json(1.0)},
        {"text",               req.text},
        {"tool_choice",        req.tool_choice},
        {"tools",              req.tools},
        {"top_p",              req.top_p >= 0.0 ? json(req.top_p) : json(1.0)},
        {"truncation",         req.truncation},
        {"usage",              usage},
        {"user",               req.user.empty() ? json(nullptr) : json(req.user)},
        {"metadata",           req.metadata},
        {"safety_identifier",  req.safety_identifier.empty() ? json(nullptr) : json(req.safety_identifier)},
        {"service_tier",       req.service_tier.empty() ? json(nullptr) : json(req.service_tier)},
    };
    if (status == "completed") {
        response["completed_at"] = (double) std::time(0);
    }
    return response;
}

json server_responses_context::usage_from(const server_task_result_cmpl_final & result, const std::string & reasoning_text) const {
    int64_t n_reasoning = 0;
    if (vocab != nullptr && !reasoning_text.empty()) {
        n_reasoning = (int64_t) common_tokenize(vocab, reasoning_text, false, false).size();
    }
    return json {
        {"input_tokens", result.n_prompt_tokens},
        {"input_tokens_details", json {
            {"cached_tokens",     result.n_prompt_tokens_cache},
            {"cache_write_tokens", 0},
        }},
        {"output_tokens", result.n_decoded},
        {"output_tokens_details", json {
            {"reasoning_tokens", n_reasoning},
        }},
        {"total_tokens", result.n_prompt_tokens + result.n_decoded},
    };
}

json server_responses_context::message_item(const std::string & item_id, const std::string & text, const std::string & status) const {
    return json {
        {"id",     item_id},
        {"type",   "message"},
        {"role",   "assistant"},
        {"status", status},
        {"content", json::array({ json {
            {"type",        "output_text"},
            {"text",        text},
            {"annotations", json::array()},
            {"logprobs",    message_logprobs},
        }})},
    };
}

json server_responses_context::reasoning_item(const std::string & item_id, const std::string & text, const std::string & status) const {
    json content = json::array();
    json summary = json::array();
    if (!text.empty()) {
        if (reasoning_summary) {
            summary.push_back(json {
                {"type", "summary_text"},
                {"text", text},
            });
        } else {
            content.push_back(json {
                {"type", "reasoning_text"},
                {"text", text},
            });
        }
    }
    return json {
        {"id",                item_id},
        {"type",              "reasoning"},
        {"summary",           summary},
        {"content",           content},
        {"encrypted_content", nullptr},
        {"status",            status},
    };
}

json server_responses_context::function_call_item(
        const std::string & item_id,
        const std::string & call_id,
        const std::string & name,
        const std::string & arguments,
        const std::string & status) const {
    return json {
        {"id",        item_id},
        {"type",      "function_call"},
        {"status",    status},
        {"arguments", arguments},
        {"call_id",   call_id},
        {"name",      name},
    };
}

json server_responses_context::output_from_msg(const common_chat_msg & msg) const {
    json output = json::array();
    if (!msg.reasoning_content.empty()) {
        output.push_back(reasoning_item("rs_" + random_string(), msg.reasoning_content, "completed"));
    }
    if (!msg.content.empty()) {
        output.push_back(message_item("msg_" + random_string(), msg.content, "completed"));
    }
    for (const auto & tool_call : msg.tool_calls) {
        const std::string call_id = tool_call.id.empty() ? "call_" + random_string() : tool_call.id;
        output.push_back(function_call_item("fc_" + random_string(), call_id, tool_call.name, tool_call.arguments, "completed"));
        if (!req.parallel_tool_calls) {
            break;
        }
    }
    return output;
}

json server_responses_context::output_from_stream(size_t * n_reasoning) const {
    std::vector<std::pair<size_t, json>> items;
    if (reasoning.added) {
        items.emplace_back(reasoning.output_index, reasoning_item(reasoning.id, reasoning.text, "completed"));
    }
    if (message.added) {
        items.emplace_back(message.output_index, message_item(message.id, message.text, "completed"));
    }
    for (const auto & tool : tools) {
        if (tool.added) {
            items.emplace_back(tool.output_index, function_call_item(tool.item_id, tool.call_id, tool.name, tool.arguments, "completed"));
        }
    }
    std::sort(items.begin(), items.end(), [](const auto & a, const auto & b) { return a.first < b.first; });

    json output = json::array();
    for (auto & item : items) {
        output.push_back(std::move(item.second));
    }
    if (n_reasoning != nullptr) {
        *n_reasoning = reasoning.text.size();
    }
    return output;
}

void server_responses_context::ensure_started(json & events) {
    if (started) {
        return;
    }
    started = true;
    const json in_progress = make_response("in_progress", json::array(), nullptr, nullptr, nullptr);
    push(events, "response.created",   json {{"response", in_progress}});
    push(events, "response.in_progress", json {{"response", in_progress}});
}

void server_responses_context::add_reasoning_delta(json & events, const std::string & delta) {
    if (!reasoning.added) {
        reasoning.id           = "rs_" + random_string();
        reasoning.output_index = next_output_index++;
        reasoning.added        = true;
        push(events, "response.output_item.added", json {
            {"output_index", reasoning.output_index},
            {"item", json {
                {"id",      reasoning.id},
                {"type",    "reasoning"},
                {"summary", json::array()},
                {"content", json::array()},
                {"encrypted_content", nullptr},
                {"status",  "in_progress"},
            }},
        });
    }
    if (!reasoning.part_added) {
        reasoning.part_added = true;
        if (reasoning_summary) {
            push(events, "response.reasoning_summary_part.added", json {
                {"item_id",       reasoning.id},
                {"output_index",  reasoning.output_index},
                {"summary_index", 0},
                {"part", json {
                    {"type", "summary_text"},
                    {"text", ""},
                }},
            });
        } else {
            push(events, "response.content_part.added", json {
                {"item_id",       reasoning.id},
                {"output_index",  reasoning.output_index},
                {"content_index", 0},
                {"part", json {
                    {"type", "reasoning_text"},
                    {"text", ""},
                }},
            });
        }
    }
    reasoning.text += delta;
    if (reasoning_summary) {
        push(events, "response.reasoning_summary_text.delta", json {
            {"item_id",       reasoning.id},
            {"output_index",  reasoning.output_index},
            {"summary_index", 0},
            {"delta",         delta},
        });
    } else {
        push(events, "response.reasoning_text.delta", json {
            {"item_id",       reasoning.id},
            {"output_index",  reasoning.output_index},
            {"content_index", 0},
            {"delta",         delta},
        });
    }
}

void server_responses_context::add_text_delta(json & events, const std::string & delta) {
    if (!message.added) {
        message.id           = "msg_" + random_string();
        message.output_index = next_output_index++;
        message.added        = true;
        push(events, "response.output_item.added", json {
            {"output_index", message.output_index},
            {"item", json {
                {"id",      message.id},
                {"type",    "message"},
                {"role",    "assistant"},
                {"status",  "in_progress"},
                {"content", json::array()},
            }},
        });
    }
    if (!message.part_added) {
        message.part_added = true;
        push(events, "response.content_part.added", json {
            {"item_id",       message.id},
            {"output_index",  message.output_index},
            {"content_index", 0},
            {"part", json {
                {"type",        "output_text"},
                {"text",        ""},
                {"annotations", json::array()},
                {"logprobs",    json::array()},
            }},
        });
    }
    message.text += delta;
    json event_data = json {
        {"item_id",       message.id},
        {"output_index",  message.output_index},
        {"content_index", 0},
        {"delta",         delta},
    };
    if (req.include_logprobs) {
        event_data["logprobs"] = json::array();
    }
    push(events, "response.output_text.delta", std::move(event_data));
}

void server_responses_context::add_tool_delta(json & events, size_t index, const common_chat_tool_call & delta) {
    if (!req.parallel_tool_calls && index > 0) {
        return;
    }
    if (tools.size() <= index) {
        tools.resize(index + 1);
    }
    tool_item & tool = tools[index];

    if (!tool.added) {
        tool.arguments += delta.arguments;
        if (delta.name.empty()) {
            return;
        }
        tool.item_id      = "fc_" + random_string();
        tool.call_id      = delta.id.empty() ? "call_" + random_string() : delta.id;
        tool.name         = delta.name;
        tool.output_index = next_output_index++;
        tool.added        = true;
        push(events, "response.output_item.added", json {
            {"output_index", tool.output_index},
            {"item", function_call_item(tool.item_id, tool.call_id, tool.name, "", "in_progress")},
        });
        if (!tool.arguments.empty()) {
            push(events, "response.function_call_arguments.delta", json {
                {"item_id",      tool.item_id},
                {"output_index", tool.output_index},
                {"delta",        tool.arguments},
            });
        }
        return;
    }

    if (!delta.id.empty() && tool.call_id.empty()) {
        tool.call_id = delta.id;
    }
    if (!delta.arguments.empty()) {
        tool.arguments += delta.arguments;
        push(events, "response.function_call_arguments.delta", json {
            {"item_id",      tool.item_id},
            {"output_index", tool.output_index},
            {"delta",        delta.arguments},
        });
    }
}

void server_responses_context::apply_diffs(json & events, const std::vector<common_chat_msg_diff> & diffs) {
    for (const auto & diff : diffs) {
        if (!diff.reasoning_content_delta.empty()) {
            add_reasoning_delta(events, diff.reasoning_content_delta);
        }
        if (!diff.content_delta.empty()) {
            add_text_delta(events, diff.content_delta);
        }
        if (diff.tool_call_index != std::string::npos) {
            add_tool_delta(events, diff.tool_call_index, diff.tool_call_delta);
        }
    }
}

void server_responses_context::finalize_items(json & events) {
    if (reasoning.added && !reasoning.done) {
        reasoning.done = true;
        if (reasoning_summary) {
            push(events, "response.reasoning_summary_text.done", json {
                {"item_id",       reasoning.id},
                {"output_index",  reasoning.output_index},
                {"summary_index", 0},
                {"text",          reasoning.text},
            });
            push(events, "response.reasoning_summary_part.done", json {
                {"item_id",       reasoning.id},
                {"output_index",  reasoning.output_index},
                {"summary_index", 0},
                {"part", json {
                    {"type", "summary_text"},
                    {"text", reasoning.text},
                }},
            });
        } else {
            push(events, "response.reasoning_text.done", json {
                {"item_id",       reasoning.id},
                {"output_index",  reasoning.output_index},
                {"content_index", 0},
                {"text",          reasoning.text},
            });
            push(events, "response.content_part.done", json {
                {"item_id",       reasoning.id},
                {"output_index",  reasoning.output_index},
                {"content_index", 0},
                {"part", json {
                    {"type", "reasoning_text"},
                    {"text", reasoning.text},
                }},
            });
        }
        push(events, "response.output_item.done", json {
            {"output_index", reasoning.output_index},
            {"item", reasoning_item(reasoning.id, reasoning.text, "completed")},
        });
    }
    if (message.added && !message.done) {
        message.done = true;
        json done_data = json {
            {"item_id",       message.id},
            {"output_index",  message.output_index},
            {"content_index", 0},
            {"text",          message.text},
        };
        if (req.include_logprobs) {
            done_data["logprobs"] = message_logprobs;
        }
        push(events, "response.output_text.done", std::move(done_data));
        push(events, "response.content_part.done", json {
            {"item_id",       message.id},
            {"output_index",  message.output_index},
            {"content_index", 0},
            {"part", json {
                {"type",        "output_text"},
                {"text",        message.text},
                {"annotations", json::array()},
                {"logprobs",    req.include_logprobs ? message_logprobs : json::array()},
            }},
        });
        push(events, "response.output_item.done", json {
            {"output_index", message.output_index},
            {"item", message_item(message.id, message.text, "completed")},
        });
    }
    for (auto & tool : tools) {
        if (!tool.added || tool.done) {
            continue;
        }
        tool.done = true;
        push(events, "response.function_call_arguments.done", json {
            {"item_id",      tool.item_id},
            {"output_index", tool.output_index},
            {"arguments",    tool.arguments},
        });
        push(events, "response.output_item.done", json {
            {"output_index", tool.output_index},
            {"item", function_call_item(tool.item_id, tool.call_id, tool.name, tool.arguments, "completed")},
        });
    }
}

json server_responses_context::consume(const server_task_result_cmpl_partial & result) {
    json events = json::array();
    ensure_started(events);

    const size_t n_events = events.size();
    apply_diffs(events, result.oaicompat_msg_diffs);
    if (req.include_logprobs && !result.prob_output.probs.empty()) {
        json entry = response_logprob_json(result.prob_output);
        text_logprobs.push_back(entry);
        for (size_t i = events.size(); i > 0; i--) {
            json & event = events[i - 1];
            if (event.value("event", std::string()) == "response.output_text.delta") {
                event.at("data")["logprobs"] = json::array({ entry });
                break;
            }
        }
    }
    if (events.size() == n_events && result.is_progress) {
        push(events, "response.in_progress", json {
            {"response", make_response("in_progress", json::array(), nullptr, nullptr, nullptr)},
        });
    }

    if (!events.empty()) {
        json & data = events.back().at("data");
        if (result.stats.is_set()) {
            data["timings"] = result.stats.to_json();
        }
        if (result.is_progress) {
            data["prompt_progress"] = result.progress.to_json();
        }
    }
    publish(events);
    return events;
}

json server_responses_context::finish(const server_task_result_cmpl_final & result) {
    json events = json::array();
    ensure_started(events);
    apply_diffs(events, result.oaicompat_msg_diffs);

    const common_chat_msg & msg = result.oaicompat_msg;
    if (!reasoning.added && !msg.reasoning_content.empty()) {
        add_reasoning_delta(events, msg.reasoning_content);
    }
    if (!message.added && !msg.content.empty()) {
        add_text_delta(events, msg.content);
    }
    for (size_t i = 0; i < msg.tool_calls.size(); i++) {
        if (!req.parallel_tool_calls && i > 0) {
            break;
        }
        if (i < tools.size() && tools[i].added) {
            continue;
        }
        add_tool_delta(events, i, msg.tool_calls[i]);
    }

    if (req.include_logprobs) {
        json list = json::array();
        if (!result.probs_output.empty()) {
            for (const auto & p : result.probs_output) {
                list.push_back(response_logprob_json(p));
            }
        } else {
            for (const auto & entry : text_logprobs) {
                list.push_back(entry);
            }
        }
        message_logprobs = std::move(list);
    }

    finalize_items(events);

    const bool incomplete = result.truncated || result.stop == STOP_TYPE_LIMIT;
    const json usage      = usage_from(result, reasoning.text);
    const json details    = incomplete ? json {{"reason", "max_output_tokens"}} : json(nullptr);
    const std::string status = incomplete ? "incomplete" : "completed";
    const json response   = make_response(status, output_from_stream(), usage, nullptr, details);

    if (incomplete) {
        push(events, "response.incomplete", json {{"response", response}});
    } else {
        push(events, "response.completed", json {{"response", response}});
    }
    if (result.stats.is_set()) {
        events.back().at("data")["timings"] = result.stats.to_json();
    }
    publish(events);
    close_events();
    store_response(response);
    return events;
}

json server_responses_context::final_response(const server_task_result_cmpl_final & result) {
    common_chat_msg msg = result.oaicompat_msg;
    if (msg.empty()) {
        msg.role    = "assistant";
        msg.content = result.content.text;
    }

    if (req.include_logprobs) {
        json list = json::array();
        for (const auto & p : result.probs_output) {
            list.push_back(response_logprob_json(p));
        }
        message_logprobs = std::move(list);
    }

    const bool incomplete = result.truncated || result.stop == STOP_TYPE_LIMIT;
    const json usage      = usage_from(result, msg.reasoning_content);
    const json details    = incomplete ? json {{"reason", "max_output_tokens"}} : json(nullptr);
    const std::string status = incomplete ? "incomplete" : "completed";
    const json response   = make_response(status, output_from_msg(msg), usage, nullptr, details);
    store_response(response);
    return response;
}

json server_responses_context::failed_event(const std::string & message, const std::string & code) {
    json error = json {
        {"code",    code.empty() ? std::string("server_error") : code},
        {"message", message},
    };
    json response = make_response("failed", json::array(), nullptr, error, nullptr);
    if (req.store && !stored.exchange(true)) {
        server_responses_stored entry;
        entry.response    = response;
        entry.input_items = req.effective_input;
        entry.created_at  = created_at;
        store.put(id, std::move(entry));
    }
    json event = json {
        {"event", "response.failed"},
        {"data", json {
            {"type",            "response.failed"},
            {"sequence_number", seq++},
            {"response",        response},
        }},
    };
    publish(event);
    close_events();
    return event;
}

void server_responses_context::store_response(const json & response) {
    if (cancelled.load()) {
        return;
    }
    if (conversations != nullptr && !req.conversation_id.empty() && !conversation_appended) {
        conversation_appended = true;
        json to_add = json::array();
        for (const auto & item : req.input_items) {
            to_add.push_back(item);
        }
        if (response.is_object() && response.contains("output")) {
            for (const auto & item : response.at("output")) {
                to_add.push_back(item);
            }
        }
        conversations->add_items(req.conversation_id, to_add);
    }

    if (!req.store) {
        return;
    }
    if (stored.exchange(true)) {
        return;
    }
    server_responses_stored entry;
    entry.response    = response;
    entry.input_items = req.effective_input;
    entry.created_at  = created_at;
    store.put(id, std::move(entry));
}

json server_responses_context::begin_background() {
    req.store = true;
    json response = make_response("in_progress", json::array(), nullptr, nullptr, nullptr);
    server_responses_stored entry;
    entry.response    = response;
    entry.input_items = req.effective_input;
    entry.created_at  = created_at;
    store.put(id, std::move(entry));
    if (event_log) {
        // publish the opening events right away, a subscriber can attach before the first token
        json events = json::array();
        ensure_started(events);
        publish(events);
    }
    return response;
}

json server_responses_context::cancel() {
    cancelled.store(true);
    if (stored.exchange(true)) {
        return nullptr;
    }
    json response = make_response("cancelled", json::array(), nullptr, nullptr, nullptr);
    {
        std::lock_guard<std::mutex> lock(cancel_mutex);
        cancelled_response = response;
    }
    server_responses_stored entry;
    entry.response    = response;
    entry.input_items = req.effective_input;
    entry.created_at  = created_at;
    store.put(id, std::move(entry));
    return response;
}

void server_responses_context::fail(const std::string & message) {
    if (stored.exchange(true)) {
        return;
    }
    json error = json {
        {"code",    "server_error"},
        {"message", message},
    };
    json response = make_response("failed", json::array(), nullptr, error, nullptr);
    server_responses_stored entry;
    entry.response    = response;
    entry.input_items = req.effective_input;
    entry.created_at  = created_at;
    store.put(id, std::move(entry));
    if (event_log) {
        json events = json::array();
        ensure_started(events);
        push(events, "response.failed", json {{"response", response}});
        publish(events);
        close_events();
    }
}

void server_responses_context::finish_cancelled() {
    if (!event_log || event_log->is_done()) {
        return;
    }
    json response;
    {
        std::lock_guard<std::mutex> lock(cancel_mutex);
        response = cancelled_response;
    }
    if (response.is_null()) {
        response = make_response("cancelled", json::array(), nullptr, nullptr, nullptr);
    }
    json events = json::array();
    ensure_started(events);
    // the stored replay maps a cancelled response to the completed event, keep the live stream equal
    push(events, "response.completed", json {{"response", response}});
    publish(events);
    close_events();
}
