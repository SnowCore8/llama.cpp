#include "server-chat.h"
#include "server-common.h"
#include "server-responses.h"

#include <algorithm>
#include <chrono>
#include <deque>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

json server_chat_convert_responses_to_chatcmpl(const json & response_body) {
    if (!response_body.contains("input")) {
        throw std::invalid_argument("'input' is required");
    }
    // previous_response_id is resolved by server_responses_prepare_request() before conversion.

    const json input_value = response_body.at("input");
    json chatcmpl_body = response_body;
    chatcmpl_body.erase("input");
    chatcmpl_body.erase("previous_response_id");
    std::vector<json> chatcmpl_messages;

    if (response_body.contains("instructions")) {
        if (!response_body.at("instructions").is_string() && !response_body.at("instructions").is_null()) {
            throw std::invalid_argument("'instructions' must be a string");
        }
        chatcmpl_messages.push_back({
            {"role",    "system"},
            {"content", json_value(response_body, "instructions", std::string())},
        });
        chatcmpl_body.erase("instructions");
    }

    if (input_value.is_string()) {
        // #responses_create-input-text_input
        chatcmpl_messages.push_back({
            {"role",    "user"},
            {"content", input_value},
        });
    } else if (input_value.is_array()) {
        // #responses_create-input-input_item_list

        static auto exists_and_is_array = [](const json & j, const char * key) -> bool {
            return j.contains(key) && j.at(key).is_array();
        };
        static auto exists_and_is_string = [](const json & j, const char * key) -> bool {
            return j.contains(key) && j.at(key).is_string();
        };

        std::deque<json> pending;
        std::unordered_set<std::string> seen_compaction;
        size_t compaction_expansions = 0;
        static constexpr size_t k_max_compaction_expansions = 64;
        static constexpr size_t k_max_pending_items = 4096;
        for (const auto & it : input_value) {
            pending.push_back(it);
        }
        while (!pending.empty()) {
            if (pending.size() > k_max_pending_items) {
                throw std::invalid_argument("compaction expand exceeded pending item limit");
            }
            json item = std::move(pending.front());
            pending.pop_front();
            bool merge_prev = !chatcmpl_messages.empty() && chatcmpl_messages.back().value("role", "") == "assistant";

            if (exists_and_is_string(item, "content")) {
                // #responses_create-input-input_item_list-input_message-content-text_input
                // Only "Input message" contains item["content"]::string
                // After converting item["content"]::string to item["content"]::array,
                // we can treat "Input message" as sum of "Item-Input message" and "Item-Output message"
                item["content"] = json::array({
                    json {
                        {"text", item.at("content")},
                        {"type", "input_text"}
                    }
                });
            }

            // Expand local. compaction into real prior items (not a placeholder string).
            if (exists_and_is_string(item, "type") && item.at("type") == "compaction") {
                const std::string enc = json_value(item, "encrypted_content", std::string());
                if (enc.empty() || !seen_compaction.insert(enc).second) {
                    continue;
                }
                if (++compaction_expansions > k_max_compaction_expansions) {
                    throw std::invalid_argument("compaction expand exceeded depth limit");
                }
                json folded;
                if (server_responses_expand_local_blob(enc, folded) &&
                        folded.contains("items") && folded.at("items").is_array()) {
                    const auto & items = folded.at("items");
                    for (auto it = items.begin(); it != items.end(); ++it) {
                        pending.push_front(*it);
                    }
                }
                continue;
            }

            if (exists_and_is_array(item, "content") &&
                exists_and_is_string(item, "role") &&
                (item.at("role") == "user" ||
                    item.at("role") == "system" ||
                    item.at("role") == "developer")
            ) {
                // #responses_create-input-input_item_list-item-input_message
                std::vector<json> chatcmpl_content;

                for (const json & input_item : item.at("content")) {
                    const std::string type = json_value(input_item, "type", std::string());

                    if (type == "input_text") {
                        if (!input_item.contains("text")) {
                            throw std::invalid_argument("'Input text' requires 'text'");
                        }
                        json part = {
                            {"text", input_item.at("text")},
                            {"type", "text"},
                        };
                        // carry the key as-is; shape is checked by oaicompat_chat_params_parse
                        if (input_item.contains("prompt_cache_breakpoint") && !input_item.at("prompt_cache_breakpoint").is_null()) {
                            part["prompt_cache_breakpoint"] = input_item.at("prompt_cache_breakpoint");
                        }
                        chatcmpl_content.push_back(std::move(part));
                    } else if (type == "input_image") {
                        // While `detail` is marked as required,
                        // it has default value("auto") and can be omitted.

                        if (!input_item.contains("image_url")) {
                            throw std::invalid_argument("'image_url' is required");
                        }
                        json part = {
                            {"image_url", json {
                                {"url", input_item.at("image_url")}
                            }},
                            {"type", "image_url"},
                        };
                        // carry the key as-is; shape is checked by oaicompat_chat_params_parse
                        if (input_item.contains("prompt_cache_breakpoint") && !input_item.at("prompt_cache_breakpoint").is_null()) {
                            part["prompt_cache_breakpoint"] = input_item.at("prompt_cache_breakpoint");
                        }
                        chatcmpl_content.push_back(std::move(part));
                    } else {
                        throw std::invalid_argument("'type' must be one of 'input_text' or 'input_image'");
                    }
                }

                if (item.contains("type")) {
                    item.erase("type");
                }
                if (item.contains("status")) {
                    item.erase("status");
                }
                item["content"] = chatcmpl_content;

                chatcmpl_messages.push_back(item);
            } else if (exists_and_is_string(item, "role") &&
                item.at("role") == "assistant" &&
                // EasyInputMessage omits type; Output message uses type=message.
                (!item.contains("type") || item.at("type").is_null() ||
                 (exists_and_is_string(item, "type") && item.at("type") == "message"))
            ) {
                // #responses_create-input-input_item_list-item-output_message
                // Also OpenAI EasyInputMessage with role=assistant (no type).
                auto chatcmpl_content = json::array();

                // Handle both string content and array content
                if (item.contains("content") && item.at("content").is_string()) {
                    // String content - convert to text content part
                    chatcmpl_content.push_back({
                        {"text", item.at("content")},
                        {"type", "text"},
                    });
                } else if (exists_and_is_array(item, "content")) {
                    // Array content - process each item
                    for (const auto & output_text : item.at("content")) {
                        const std::string type = json_value(output_text, "type", std::string());
                        if (type == "output_text" || type == "input_text") {
                            // Accept both output_text and input_text (string content gets converted to input_text)
                            if (!exists_and_is_string(output_text, "text")) {
                                throw std::invalid_argument("'Output text' requires 'text'");
                            }
                            chatcmpl_content.push_back({
                                {"text", output_text.at("text")},
                                {"type", "text"},
                            });
                        } else if (type == "refusal") {
                            if (!exists_and_is_string(output_text, "refusal")) {
                                throw std::invalid_argument("'Refusal' requires 'refusal'");
                            }
                            chatcmpl_content.push_back({
                                {"refusal", output_text.at("refusal")},
                                {"type", "refusal"},
                            });
                        } else if (type.empty() && exists_and_is_string(output_text, "text")) {
                            // Some clients send bare {text: "..."} parts
                            chatcmpl_content.push_back({
                                {"text", output_text.at("text")},
                                {"type", "text"},
                            });
                        } else {
                            throw std::invalid_argument("'type' must be one of 'output_text' or 'refusal'");
                        }
                    }
                }

                if (merge_prev) {
                    auto & prev_msg = chatcmpl_messages.back();
                    if (!exists_and_is_array(prev_msg, "content")) {
                        prev_msg["content"] = json::array();
                    }
                    auto & prev_content = prev_msg["content"];
                    prev_content.insert(chatcmpl_content);
                } else {
                    item.erase("status");
                    item.erase("type");
                    item["content"] = chatcmpl_content;
                    chatcmpl_messages.push_back(item);
                }
            } else if (exists_and_is_string(item, "arguments") &&
                exists_and_is_string(item, "call_id") &&
                exists_and_is_string(item, "name") &&
                exists_and_is_string(item, "type") &&
                item.at("type") == "function_call"
            ) {
                // #responses_create-input-input_item_list-item-function_tool_call
                json tool_call = {
                    {"function", json {
                        {"arguments", item.at("arguments")},
                        {"name",      item.at("name")},
                    }},
                    {"id",   item.at("call_id")},
                    {"type", "function"},
                };

                if (merge_prev) {
                    auto & prev_msg = chatcmpl_messages.back();
                    if (!exists_and_is_array(prev_msg, "tool_calls")) {
                        prev_msg["tool_calls"] = json::array();
                    }
                    prev_msg["tool_calls"].push_back(tool_call);
                } else {
                    chatcmpl_messages.push_back(json {
                        {"role",       "assistant"},
                        {"tool_calls", json::array({tool_call})}
                    });
                }
            } else if (exists_and_is_string(item, "call_id") &&
                exists_and_is_string(item, "input") &&
                exists_and_is_string(item, "name") &&
                exists_and_is_string(item, "type") &&
                item.at("type") == "custom_tool_call"
            ) {
                // #responses_create-input-input_item_list-item-custom_tool_call
                json tool_call = {
                    {"custom", json {
                        {"input", item.at("input")},
                        {"name",  item.at("name")},
                    }},
                    {"id",   item.at("call_id")},
                    {"type", "custom"},
                };

                if (merge_prev) {
                    auto & prev_msg = chatcmpl_messages.back();
                    if (!exists_and_is_array(prev_msg, "tool_calls")) {
                        prev_msg["tool_calls"] = json::array();
                    }
                    prev_msg["tool_calls"].push_back(tool_call);
                } else {
                    chatcmpl_messages.push_back(json {
                        {"role",       "assistant"},
                        {"tool_calls", json::array({tool_call})}
                    });
                }
            } else if (exists_and_is_string(item, "call_id") &&
                (exists_and_is_string(item, "output") || exists_and_is_array(item, "output")) &&
                exists_and_is_string(item, "type") &&
                item.at("type") == "function_call_output"
            ) {
                // #responses_create-input-input_item_list-item-function_tool_call_output
                if (item.at("output").is_string()) {
                    chatcmpl_messages.push_back(json {
                        {"content",      item.at("output")},
                        {"role",         "tool"},
                        {"tool_call_id", item.at("call_id")},
                    });
                } else {
                    json chatcmpl_outputs = item.at("output");
                    for (json & chatcmpl_output : chatcmpl_outputs) {
                        if (!chatcmpl_output.contains("type") || chatcmpl_output.at("type") != "input_text") {
                            throw std::invalid_argument("Output of tool call should be 'Input text'");
                        }
                        chatcmpl_output["type"] = "text";
                    }
                    chatcmpl_messages.push_back(json {
                        {"content",      chatcmpl_outputs},
                        {"role",         "tool"},
                        {"tool_call_id", item.at("call_id")},
                    });
                }
            } else if (exists_and_is_string(item, "call_id") &&
                (exists_and_is_string(item, "output") || exists_and_is_array(item, "output")) &&
                exists_and_is_string(item, "type") &&
                item.at("type") == "custom_tool_call_output"
            ) {
                // #responses_create-input-input_item_list-item-custom_tool_call_output
                if (item.at("output").is_string()) {
                    chatcmpl_messages.push_back(json {
                        {"content",      item.at("output")},
                        {"role",         "tool"},
                        {"tool_call_id", item.at("call_id")},
                    });
                } else {
                    json chatcmpl_outputs = item.at("output");
                    for (json & chatcmpl_output : chatcmpl_outputs) {
                        if (!chatcmpl_output.contains("type") || chatcmpl_output.at("type") != "input_text") {
                            throw std::invalid_argument("Output of tool call should be 'Input text'");
                        }
                        chatcmpl_output["type"] = "text";
                    }
                    chatcmpl_messages.push_back(json {
                        {"content",      chatcmpl_outputs},
                        {"role",         "tool"},
                        {"tool_call_id", item.at("call_id")},
                    });
                }
            } else if (exists_and_is_array(item, "summary") &&
                exists_and_is_string(item, "type") &&
                item.at("type") == "reasoning") {
                // #responses_create-input-input_item_list-item-reasoning

                // Official input reasoning items carry summary (+ optional encrypted_content)
                // and no content; the local conversion prefers content[0].text when present.
                std::string reasoning_text;
                if (exists_and_is_array(item, "content") && !item.at("content").empty()) {
                    if (!exists_and_is_string(item.at("content")[0], "text")) {
                        throw std::invalid_argument("item['content']['text'] is not a string");
                    }
                    reasoning_text = item.at("content")[0].at("text").get<std::string>();
                } else {
                    if (exists_and_is_array(item, "summary")) {
                        for (const auto & part : item.at("summary")) {
                            if (exists_and_is_string(part, "text")) {
                                if (!reasoning_text.empty()) {
                                    reasoning_text += "\n";
                                }
                                reasoning_text += part.at("text").get<std::string>();
                            }
                        }
                    }
                    if (reasoning_text.empty()) {
                        const std::string enc = json_value(item, "encrypted_content", std::string());
                        json folded;
                        if (!enc.empty() && server_responses_expand_local_blob(enc, folded) &&
                                folded.contains("text") && folded.at("text").is_string()) {
                            reasoning_text = folded.at("text").get<std::string>();
                        }
                    }
                }
                if (reasoning_text.empty()) {
                    // nothing to replay (empty summary and no local encrypted content)
                    continue;
                }

                if (merge_prev) {
                    auto & prev_msg = chatcmpl_messages.back();
                    prev_msg["reasoning_content"] = reasoning_text;
                } else {
                    chatcmpl_messages.push_back(json {
                        {"role", "assistant"},
                        {"content", json::array()},
                        {"reasoning_content", reasoning_text},
                    });
                }
            } else {
                throw std::invalid_argument("Cannot determine type of 'item'");
            }
        }
    } else {
        throw std::invalid_argument("'input' must be a string or array of objects");
    }

    // Coalesce leading system/developer messages into a single system message.
    // Codex (and other Responses clients) often send both top-level `instructions`
    // and an input item with role `developer`. After conversion that becomes two
    // system messages; Qwen-family chat templates reject any system message that
    // is not at index 0 ("System message must be at the beginning.").
    {
        auto extract_text = [](const json & msg) -> std::string {
            if (!msg.contains("content")) {
                return {};
            }
            const auto & content = msg.at("content");
            if (content.is_string()) {
                return content.get<std::string>();
            }
            if (!content.is_array()) {
                return {};
            }
            std::string out;
            for (const auto & part : content) {
                if (part.contains("text") && part.at("text").is_string()) {
                    out += part.at("text").get<std::string>();
                }
            }
            return out;
        };

        size_t n_sys = 0;
        while (n_sys < chatcmpl_messages.size()) {
            const std::string role = chatcmpl_messages[n_sys].value("role", "");
            if (role != "system" && role != "developer") {
                break;
            }
            ++n_sys;
        }

        if (n_sys > 1) {
            std::string merged;
            for (size_t i = 0; i < n_sys; ++i) {
                const std::string text = extract_text(chatcmpl_messages[i]);
                if (text.empty()) {
                    continue;
                }
                if (!merged.empty()) {
                    merged += "\n\n";
                }
                merged += text;
            }
            chatcmpl_messages.erase(chatcmpl_messages.begin(), chatcmpl_messages.begin() + n_sys);
            chatcmpl_messages.insert(chatcmpl_messages.begin(), json {
                {"role",    "system"},
                {"content", merged},
            });
        } else if (n_sys == 1 && chatcmpl_messages[0].value("role", "") == "developer") {
            chatcmpl_messages[0]["role"] = "system";
        }
    }

    chatcmpl_body["messages"] = chatcmpl_messages;

    if (response_body.contains("tools")) {
        if (!response_body.at("tools").is_array()) {
            throw std::invalid_argument("'tools' must be an array of objects");
        }
        std::vector<json> chatcmpl_tools;
        for (json resp_tool : response_body.at("tools")) {
            json chatcmpl_tool;

            const std::string type = json_value(resp_tool, "type", std::string());
            if (server_is_local_web_search_tool_type(type)) {
                // Handled by server_openai_apply_web_search_semantics (prepare_request).
                continue;
            }
            if (type == "custom") {
                // #responses_create-tools-tool-custom: flat shape, Chat Completions uses {custom:{...}}
                resp_tool.erase("type");
                chatcmpl_tool["type"] = "custom";
                chatcmpl_tool["custom"] = std::move(resp_tool);
                chatcmpl_tools.push_back(std::move(chatcmpl_tool));
                continue;
            }
            if (type != "function") {
                // Do not silently drop other hosted/cloud tools (file_search, mcp, …).
                throw std::invalid_argument(
                    "hosted Responses tool type '" + type + "' is not supported on this server "
                    "(only type=function, type=custom or local web_search). Cloud tool execution is unavailable locally.");
            }
            resp_tool.erase("type");
            chatcmpl_tool["type"] = "function";

            if (!resp_tool.contains("strict")) {
                resp_tool["strict"] = true;
            }
            chatcmpl_tool["function"] = resp_tool;
            chatcmpl_tools.push_back(chatcmpl_tool);
        }
        chatcmpl_body.erase("tools");
        if (!chatcmpl_tools.empty()) {
            chatcmpl_body["tools"] = chatcmpl_tools;
        }
    }

    if (response_body.contains("max_output_tokens")) {
        chatcmpl_body.erase("max_output_tokens");
        chatcmpl_body["max_tokens"] = response_body["max_output_tokens"];
    }

    if (response_body.contains("reasoning")) {
        // Official reasoning fields are shape-validated + echoed on the Response object.
        // Only effort maps into Chat Completions `reasoning_effort` for local thinking.
        const json & reasoning = response_body.at("reasoning");
        if (reasoning.is_object()) {
            std::string effort;
            if (reasoning.contains("effort") && !reasoning.at("effort").is_null() &&
                    reasoning.at("effort").is_string()) {
                effort = reasoning.at("effort").get<std::string>();
            }
            // mode=pro boosts local thinking depth when effort is unset or below high.
            if (reasoning.contains("mode") && reasoning.at("mode").is_string() &&
                    reasoning.at("mode").get<std::string>() == "pro") {
                static const std::unordered_set<std::string> k_below_high = {
                    "", "none", "minimal", "low", "medium",
                };
                if (k_below_high.count(effort)) {
                    effort = "high";
                }
            }
            if (!effort.empty()) {
                chatcmpl_body["reasoning_effort"] = effort;
            }
        }
        chatcmpl_body.erase("reasoning");
    }

    // Responses text → Chat: verbosity hint + format → response_format/grammar.
    if (response_body.contains("text") && response_body.at("text").is_object()) {
        const json & text = response_body.at("text");
        if (text.contains("verbosity") && text.at("verbosity").is_string()) {
            chatcmpl_body["verbosity"] = text.at("verbosity");
        }
        if (text.contains("format") && text.at("format").is_object()) {
            const json & fmt = text.at("format");
            const std::string ftype = json_value(fmt, "type", std::string());
            if (ftype == "json_object") {
                chatcmpl_body["response_format"] = json{{"type", "json_object"}};
            } else if (ftype == "json_schema") {
                if (!fmt.contains("schema") || !fmt.at("schema").is_object()) {
                    throw std::invalid_argument("'text.format.schema' is required for type=json_schema");
                }
                const std::string name = json_value(fmt, "name", std::string("response"));
                chatcmpl_body["response_format"] = json{
                    {"type", "json_schema"},
                    {"json_schema", {
                        {"name", name},
                        {"schema", fmt.at("schema")},
                    }},
                };
            } else if (!ftype.empty() && ftype != "text") {
                throw std::invalid_argument(
                    "'text.format.type' must be one of: text, json_object, json_schema");
            }
        }
    }

    // Responses API exposes top_logprobs without a separate logprobs boolean.
    // Chat Completions requires logprobs=true when top_logprobs is set.
    if (response_body.contains("top_logprobs") && !chatcmpl_body.contains("logprobs")) {
        chatcmpl_body["logprobs"] = true;
    }
    // include=["message.output_text.logprobs"] must enable sampling probs before strip.
    if (response_body.contains("include") && response_body.at("include").is_array()) {
        for (const auto & inc : response_body.at("include")) {
            if (inc.is_string() &&
                    inc.get<std::string>() == "message.output_text.logprobs") {
                chatcmpl_body["logprobs"] = true;
                if (!chatcmpl_body.contains("top_logprobs")) {
                    // keep the top-1 candidate: n_probs must be > 0 or the sampler records
                    // no probabilities at all, and include alone has no top_logprobs
                    chatcmpl_body["top_logprobs"] = 1;
                }
                break;
            }
        }
    }

    // Strip Responses-only request fields that are not Chat Completions params
    // (they remain on the prepared request for Response enrichment / store).
    static const char * responses_only_keys[] = {
        "store", "background", "conversation", "prompt", "include",
        "prompt_cache_key", "prompt_cache_options", "prompt_cache_retention",
        "safety_identifier", "service_tier", "truncation", "text",
        "context_management", "moderation", "max_tool_calls", "stream_options",
        "metadata", "user",
    };
    for (const char * key : responses_only_keys) {
        chatcmpl_body.erase(key);
    }

    return chatcmpl_body;
}

// custom tool call input: the raw text; when the arguments hold JSON, unwrap a
// JSON string or an object with a string "input" field
std::string server_chat_custom_tool_input(const std::string & arguments) {
    try {
        const json parsed = json::parse(arguments);
        if (parsed.is_string()) {
            return parsed.get<std::string>();
        }
        if (parsed.is_object() && parsed.contains("input") && parsed.at("input").is_string()) {
            return parsed.at("input").get<std::string>();
        }
    } catch (const common_json_error &) {
        // not JSON: the raw text is the input
    }
    return arguments;
}

bool server_chat_is_custom_tool_name(const std::string & name, const std::vector<std::string> & custom_tool_names) {
    return !name.empty() &&
        std::find(custom_tool_names.begin(), custom_tool_names.end(), name) != custom_tool_names.end();
}

std::unordered_set<size_t> server_chat_custom_tool_call_indices(
        const common_chat_msg & msg, const std::vector<std::string> & custom_tool_names) {
    std::unordered_set<size_t> indices;
    for (size_t i = 0; i < msg.tool_calls.size(); ++i) {
        if (server_chat_is_custom_tool_name(msg.tool_calls[i].name, custom_tool_names)) {
            indices.insert(i);
        }
    }
    return indices;
}

// Rewrite the tool_calls of a serialized Chat message into the official custom shape.
// Used by the non-stream Chat serializer; the stream path translates per-delta instead.
void server_chat_apply_custom_tool_calls(json & message_obj, const std::vector<std::string> & custom_tool_names) {
    if (custom_tool_names.empty() || !message_obj.contains("tool_calls") || !message_obj.at("tool_calls").is_array()) {
        return;
    }
    for (json & tool_call : message_obj["tool_calls"]) {
        if (!tool_call.is_object() || !tool_call.contains("function") || !tool_call.at("function").is_object()) {
            continue;
        }
        const json & function = tool_call.at("function");
        const std::string name = json_value(function, "name", std::string());
        if (!server_chat_is_custom_tool_name(name, custom_tool_names)) {
            continue;
        }
        tool_call["type"] = "custom";
        tool_call["custom"] = json {
            {"input", server_chat_custom_tool_input(json_value(function, "arguments", std::string()))},
            {"name",  name},
        };
        tool_call.erase("function");
    }
}

json server_chat_msg_diff_to_json_oaicompat(const common_chat_msg_diff & diff,
        const std::unordered_set<size_t> & custom_tool_call_indices) {
    json delta = json::object();
    if (!diff.reasoning_content_delta.empty()) {
        delta["reasoning_content"] = diff.reasoning_content_delta;
    }
    if (!diff.content_delta.empty()) {
        delta["content"] = diff.content_delta;
    }
    if (diff.tool_call_index != std::string::npos) {
        const bool is_custom = custom_tool_call_indices.count(diff.tool_call_index) > 0;
        json tool_call;
        tool_call["index"] = diff.tool_call_index;
        if (!diff.tool_call_delta.id.empty()) {
            tool_call["id"]   = diff.tool_call_delta.id;
            tool_call["type"] = is_custom ? "custom" : "function";
        }
        bool emit = true;
        if (is_custom) {
            // No official Chat Completions stream shape exists for custom tools;
            // mirror the non-stream shape with the decoded input delta as it arrives.
            json custom = json::object();
            if (!diff.tool_call_delta.name.empty()) {
                custom["name"] = diff.tool_call_delta.name;
            }
            if (!diff.tool_call_delta.arguments.empty()) {
                custom["input"] = diff.tool_call_delta.arguments;
            }
            if (custom.empty()) {
                emit = false; // raw envelope fragments are withheld until decodable
            } else {
                tool_call["custom"] = custom;
            }
        } else if (!diff.tool_call_delta.name.empty() || !diff.tool_call_delta.arguments.empty()) {
            json function = json::object();
            if (!diff.tool_call_delta.name.empty()) {
                function["name"] = diff.tool_call_delta.name;
            }
            if (!diff.tool_call_delta.arguments.empty()) {
                function["arguments"] = diff.tool_call_delta.arguments;
            }
            tool_call["function"] = function;
        }
        if (emit) {
            delta["tool_calls"] = json::array({ tool_call });
        }
    }
    return delta;
}

json convert_transcriptions_to_chatcmpl(
        const json & inp_body,
        const common_chat_templates * tmpls,
        const std::map<std::string, uploaded_file> & in_files,
        std::vector<raw_buffer> & out_files) {
    // TODO @ngxson : this function may need to be improved in the future
    // handle input files
    out_files.clear();
    auto it = in_files.find("file");
    if (it != in_files.end()) {
        out_files.push_back(it->second.data);
    } else {
        throw std::invalid_argument("No input file found for transcription");
    }

    // handle input data
    std::string prompt          = json_value(inp_body, "prompt", std::string());
    std::string language        = json_value(inp_body, "language", std::string());
    std::string response_format = json_value(inp_body, "response_format", std::string("json"));
    if (response_format != "json") {
        throw std::invalid_argument("Only 'json' response_format is supported for transcription");
    }
    const common_chat_prompt_preset preset = common_chat_get_asr_prompt(tmpls);
    if (prompt.empty()) {
        prompt = preset.user;
    }
    if (!language.empty()) {
        prompt += string_format(" (language: %s)", language.c_str());
    }
    prompt += get_media_marker();

    json messages = json::array();
    if (!preset.system.empty()) {
        messages.push_back({{"role", "system"}, {"content", preset.system}});
    }
    messages.push_back({{"role", "user"}, {"content", prompt}});

    json chatcmpl_body = inp_body; // copy all fields
    chatcmpl_body["messages"] = messages;

    // because input from form-data, everything is string, we need to correct the types here
    std::string stream = json_value(inp_body, "stream", std::string("false"));
    chatcmpl_body["stream"] = stream == "true";

    if (inp_body.contains("max_tokens")) {
        std::string inp = inp_body["max_tokens"].get<std::string>();
        chatcmpl_body["max_tokens"] = std::stoul(inp);
    }

    if (inp_body.contains("temperature")) {
        std::string inp = inp_body["temperature"].get<std::string>();
        chatcmpl_body["temperature"] = std::stof(inp);
    }

    return chatcmpl_body;
}
