#include "server-responses.h"

#include "common.h"

#include <algorithm>
#include <ctime>
#include <stdexcept>

namespace {

std::string content_to_text(const json & content) {
    if (content.is_null()) {
        return "";
    }
    if (content.is_string()) {
        return content.get<std::string>();
    }
    if (!content.is_array()) {
        throw std::invalid_argument("'content' must be a string or an array");
    }
    std::string text;
    for (const auto & part : content) {
        if (!part.is_object()) {
            throw std::invalid_argument("content parts must be objects");
        }
        const std::string type = part.value("type", std::string());
        if (type == "input_text" || type == "output_text" || type == "text") {
            text += part.value("text", std::string());
        } else if (type == "refusal") {
            text += part.value("refusal", std::string());
        } else if (type == "input_image" || type == "input_file") {
            throw std::invalid_argument("multimodal input is not supported");
        } else {
            throw std::invalid_argument("unsupported content part type: " + type);
        }
    }
    return text;
}

std::string tool_output_to_text(const json & output) {
    if (output.is_null()) {
        return "";
    }
    if (output.is_string()) {
        return output.get<std::string>();
    }
    if (output.is_array()) {
        return content_to_text(output);
    }
    return output.dump();
}

common_chat_msg parse_input_item(const json & item) {
    common_chat_msg msg;

    if (item.is_string()) {
        msg.role = "user";
        msg.content = item.get<std::string>();
        return msg;
    }
    if (!item.is_object()) {
        throw std::invalid_argument("input items must be objects");
    }

    const std::string type = item.value("type", std::string());
    if (type.empty() || type == "message") {
        std::string role = item.value("role", std::string("user"));
        if (role == "developer") {
            role = "system";
        }
        if (role != "user" && role != "system" && role != "assistant") {
            throw std::invalid_argument("unsupported message role: " + role);
        }
        msg.role = role;
        msg.content = content_to_text(item.value("content", json("")));
        return msg;
    }
    if (type == "function_call") {
        common_chat_tool_call tool_call;
        tool_call.id = item.value("call_id", item.value("id", std::string()));
        tool_call.name = item.value("name", std::string());
        if (tool_call.name.empty()) {
            throw std::invalid_argument("function_call item is missing 'name'");
        }
        const json arguments = item.value("arguments", json(""));
        tool_call.arguments = arguments.is_string() ? arguments.get<std::string>() : arguments.dump();
        msg.role = "assistant";
        msg.tool_calls.push_back(std::move(tool_call));
        return msg;
    }
    if (type == "function_call_output") {
        msg.role = "tool";
        msg.tool_call_id = item.value("call_id", std::string());
        msg.content = tool_output_to_text(item.value("output", json("")));
        return msg;
    }
    if (type == "reasoning") {
        // reasoning items are model output, they are not replayed into the prompt
        return msg;
    }
    if (type == "item_reference") {
        throw std::invalid_argument("item_reference is not supported");
    }
    throw std::invalid_argument("unsupported input item type: " + type);
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

bool server_responses_store::get(const std::string & id, server_responses_stored & out) const {
    std::lock_guard<std::mutex> lock(mutex);
    const auto it = items.find(id);
    if (it == items.end()) {
        return false;
    }
    out = it->second;
    return true;
}

void server_responses_store::put(const std::string & id, server_responses_stored item) {
    std::lock_guard<std::mutex> lock(mutex);
    if (items.size() >= max_items && items.find(id) == items.end()) {
        items.erase(items.begin());
    }
    items[id] = std::move(item);
}

bool server_responses_store::erase(const std::string & id) {
    std::lock_guard<std::mutex> lock(mutex);
    return items.erase(id) > 0;
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

    if (req.background) {
        throw std::invalid_argument("'background' is not supported");
    }
    const bool has_conversation = body.contains("conversation") && !body.at("conversation").is_null();
    if (!req.previous_response_id.empty() && has_conversation) {
        throw std::invalid_argument("'conversation' and 'previous_response_id' cannot be used together");
    }
    if (has_conversation) {
        throw std::invalid_argument("'conversation' is not supported");
    }
    if (body.contains("prompt") && !body.at("prompt").is_null()) {
        throw std::invalid_argument("'prompt' templates are not supported");
    }
    if (req.truncation != "disabled") {
        throw std::invalid_argument("'truncation' only supports 'disabled'");
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
        throw std::invalid_argument("'top_logprobs' is not supported");
    }
    if (!req.include.is_null()) {
        if (!req.include.is_array()) {
            throw std::invalid_argument("'include' must be an array");
        }
        for (const auto & value : req.include) {
            if (!value.is_string()) {
                throw std::invalid_argument("'include' values must be strings");
            }
            if (value.get<std::string>() == "message.output_text.logprobs") {
                throw std::invalid_argument("'include' does not support logprobs yet");
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

    return req;
}

void server_responses_resolve_input(server_responses_request & req, const server_responses_store & store) {
    json items = json::array();
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
        common_chat_session & session) {
    common_chat_templates_inputs inputs;
    inputs.use_jinja             = opt.use_jinja;
    inputs.reasoning_format      = opt.reasoning_format;
    inputs.force_pure_content    = opt.force_pure_content;
    inputs.parallel_tool_calls   = req.parallel_tool_calls;
    inputs.chat_template_kwargs  = opt.chat_template_kwargs;
    inputs.enable_thinking       = opt.enable_thinking;

    if (!req.instructions.is_null()) {
        if (req.instructions.is_string()) {
            common_chat_msg msg;
            msg.role    = "system";
            msg.content = req.instructions.get<std::string>();
            inputs.messages.push_back(std::move(msg));
        } else if (req.instructions.is_array()) {
            for (const auto & item : req.instructions) {
                common_chat_msg msg = parse_input_item(item);
                if (msg.empty()) {
                    continue;
                }
                if (msg.role != "system" && msg.role != "user" && msg.role != "assistant") {
                    throw std::invalid_argument("instructions support system, developer and user messages only");
                }
                inputs.messages.push_back(std::move(msg));
            }
        } else {
            throw std::invalid_argument("'instructions' must be a string or an array");
        }
    }

    for (const auto & item : req.effective_input) {
        common_chat_msg msg = parse_input_item(item);
        if (msg.empty()) {
            continue;
        }
        inputs.messages.push_back(std::move(msg));
    }

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
        const llama_vocab * vocab) :
    store(store),
    req(std::move(req)),
    model_name(std::move(model_name)),
    vocab(vocab) {
    id         = "resp_" + random_string();
    created_at = (int64_t) std::time(0);
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
            {"logprobs",    json::array()},
        }})},
    };
}

json server_responses_context::reasoning_item(const std::string & item_id, const std::string & text, const std::string & status) const {
    json content = json::array();
    if (!text.empty()) {
        content.push_back(json {
            {"type", "reasoning_text"},
            {"text", text},
        });
    }
    return json {
        {"id",                item_id},
        {"type",              "reasoning"},
        {"summary",           json::array()},
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
    reasoning.text += delta;
    push(events, "response.reasoning_text.delta", json {
        {"item_id",       reasoning.id},
        {"output_index",  reasoning.output_index},
        {"content_index", 0},
        {"delta",         delta},
    });
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
    push(events, "response.output_text.delta", json {
        {"item_id",       message.id},
        {"output_index",  message.output_index},
        {"content_index", 0},
        {"delta",         delta},
    });
}

void server_responses_context::add_tool_delta(json & events, size_t index, const common_chat_tool_call & delta) {
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
        push(events, "response.output_item.done", json {
            {"output_index", reasoning.output_index},
            {"item", reasoning_item(reasoning.id, reasoning.text, "completed")},
        });
    }
    if (message.added && !message.done) {
        message.done = true;
        push(events, "response.output_text.done", json {
            {"item_id",       message.id},
            {"output_index",  message.output_index},
            {"content_index", 0},
            {"text",          message.text},
        });
        push(events, "response.content_part.done", json {
            {"item_id",       message.id},
            {"output_index",  message.output_index},
            {"content_index", 0},
            {"part", json {
                {"type",        "output_text"},
                {"text",        message.text},
                {"annotations", json::array()},
                {"logprobs",    json::array()},
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
        if (i < tools.size() && tools[i].added) {
            continue;
        }
        add_tool_delta(events, i, msg.tool_calls[i]);
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
    store_response(response);
    return events;
}

json server_responses_context::final_response(const server_task_result_cmpl_final & result) {
    common_chat_msg msg = result.oaicompat_msg;
    if (msg.empty()) {
        msg.role    = "assistant";
        msg.content = result.content.text;
    }

    const bool incomplete = result.truncated || result.stop == STOP_TYPE_LIMIT;
    const json usage      = usage_from(result, msg.reasoning_content);
    const json details    = incomplete ? json {{"reason", "max_output_tokens"}} : json(nullptr);
    const std::string status = incomplete ? "incomplete" : "completed";
    const json response   = make_response(status, output_from_msg(msg), usage, nullptr, details);
    store_response(response);
    return response;
}

json server_responses_context::error_event(const std::string & message, const std::string & code) const {
    json data = {
        {"type",            "error"},
        {"message",         message},
        {"code",            code.empty() ? json(nullptr) : json(code)},
        {"param",           nullptr},
        {"sequence_number", seq++},
    };
    return json {
        {"event", "error"},
        {"data", std::move(data)},
    };
}

void server_responses_context::store_response(const json & response) {
    if (!req.store || stored) {
        return;
    }
    stored = true;
    server_responses_stored entry;
    entry.response    = response;
    entry.input_items = req.effective_input;
    entry.created_at  = created_at;
    store.put(id, std::move(entry));
}
