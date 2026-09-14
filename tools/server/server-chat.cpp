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

// Edits the cch section of an "x-anthropic-billing-header" system prompt.
// Does nothing to any other prompt.
//
// This is a claude message with a "cch=ef01a" attribute that breaks prefix caching.
// The cch stamp is a whitebox end-to-end integrity hint. It's not meaningful as a
// system prompt data, particularly to llama.cpp, but its presence means the prefix
// cache will not get past it: It changes on each request.
//
// Reference: https://github.com/ggml-org/llama.cpp/pull/21793
// Example header:
// ```
// x-anthropic-billing-header: cc_version=2.1.101.e51; cc_entrypoint=cli; cch=a5145;You are Claude Code, Anthropic's official CLI for Claude.
//                                                                            ^^^^^
// ```
static void normalize_anthropic_billing_header(std::string & system_text) {
    if (system_text.rfind("x-anthropic-billing-header:", 0) != 0) {
        return;
    }

    const size_t header_prefix_length = strlen("x-anthropic-billing-header:");
    const size_t cch_length = 5;
    const size_t index_cch = system_text.find("cch=", header_prefix_length);
    if (index_cch == std::string::npos) {
        return;
    }

    const size_t index_replace = index_cch + 4;
    if (index_replace + cch_length < system_text.length() && system_text[index_replace + cch_length] == ';') {
        for (size_t i = 0; i < cch_length; ++i) {
            system_text[index_replace + i] = 'f';
        }
    } else {
        LOG_ERR("anthropic string not as expected: %s", system_text.c_str());
    }
}


json server_chat_convert_anthropic_to_oai(const json & body, bool count_tokens) {
    json oai_body;

    // Anthropic cache_control -> local prompt cache: the marker becomes a local explicit
    // breakpoint (on the annotated part or on the message object), and the TTL feeds the
    // internal cache TTL channel (__prompt_cache_ttl). The local prompt cache holds a single
    // TTL per request, so multiple different TTLs collapse to the last one seen
    // (see OFFICIAL_API_SCOPE.md).
    std::string cache_ttl;
    auto note_cache_control = [&cache_ttl](const json & cache_control) {
        const std::string cc_type = json_value(cache_control, "type", std::string());
        if (cc_type != "ephemeral") {
            throw std::invalid_argument("'cache_control.type' must be 'ephemeral'");
        }
        const std::string ttl = json_value(cache_control, "ttl", std::string());
        if (!ttl.empty()) {
            if (ttl != "5m" && ttl != "1h") {
                throw std::invalid_argument("'cache_control.ttl' must be '5m' or '1h'");
            }
            cache_ttl = ttl;
        }
    };
    auto mark_cache_breakpoint = [&note_cache_control](json & part, const json & block) {
        if (!block.contains("cache_control") || block.at("cache_control").is_null()) {
            return;
        }
        note_cache_control(block.at("cache_control"));
        part["prompt_cache_breakpoint"] = {{"mode", "explicit"}};
    };

    // Convert system prompt
    json oai_messages = json::array();
    auto system_param = json_value(body, "system", json());
    if (!system_param.is_null()) {
        std::string system_content;
        json system_parts = json::array();
        bool system_has_breakpoint = false;

        if (system_param.is_string()) {
            system_content = system_param.get<std::string>();
            normalize_anthropic_billing_header(system_content);
        } else if (system_param.is_array()) {
            for (const auto & block : system_param) {
                if (json_value(block, "type", std::string()) == "text") {
                    auto system_text = json_value(block, "text", std::string());
                    normalize_anthropic_billing_header(system_text);
                    system_content += system_text;

                    json part = {
                        {"type", "text"},
                        {"text", system_text},
                    };
                    mark_cache_breakpoint(part, block);
                    system_has_breakpoint = system_has_breakpoint || part.contains("prompt_cache_breakpoint");
                    system_parts.push_back(part);
                }
            }
        }

        json system_msg = {{"role", "system"}};
        if (system_has_breakpoint) {
            // a breakpoint needs part form; plain strings cannot carry one
            system_msg["content"] = system_parts;
        } else {
            system_msg["content"] = system_content;
        }
        oai_messages.push_back(system_msg);
    }

    // Convert messages
    if (!body.contains("messages")) {
        throw std::runtime_error("'messages' is required");
    }
    const json & messages = body.at("messages");
    if (messages.is_array()) {
        for (const auto & msg : messages) {
            std::string role = json_value(msg, "role", std::string());

            if (!msg.contains("content")) {
                if (role == "assistant") {
                    continue;
                }
                oai_messages.push_back(msg);
                continue;
            }

            const json & content = msg.at("content");

            if (content.is_string()) {
                oai_messages.push_back(msg);
                continue;
            }

            if (!content.is_array()) {
                oai_messages.push_back(msg);
                continue;
            }

            json tool_calls = json::array();
            json converted_content = json::array();
            json tool_results = json::array();
            std::string reasoning_content;
            bool has_tool_calls = false;
            // a tool_use block with cache_control marks this whole message: tool parts have no
            // standalone part form, so the breakpoint rides on the message object
            bool msg_breakpoint = false;

            for (const auto & block : content) {
                std::string type = json_value(block, "type", std::string());

                if (type == "text") {
                    json part = block;
                    mark_cache_breakpoint(part, block);
                    converted_content.push_back(part);
                } else if (type == "thinking") {
                    reasoning_content += json_value(block, "thinking", std::string());
                } else if (type == "image") {
                    json source = json_value(block, "source", json::object());
                    std::string source_type = json_value(source, "type", std::string());

                    if (source_type == "base64") {
                        std::string media_type = json_value(source, "media_type", std::string("image/jpeg"));
                        std::string data = json_value(source, "data", std::string());
                        std::ostringstream ss;
                        ss << "data:" << media_type << ";base64," << data;

                        json part = {
                            {"type", "image_url"},
                            {"image_url", {
                                {"url", ss.str()}
                            }}
                        };
                        mark_cache_breakpoint(part, block);
                        converted_content.push_back(part);
                    } else if (source_type == "url") {
                        std::string url = json_value(source, "url", std::string());
                        json part = {
                            {"type", "image_url"},
                            {"image_url", {
                                {"url", url}
                            }}
                        };
                        mark_cache_breakpoint(part, block);
                        converted_content.push_back(part);
                    }
                } else if (type == "tool_use") {
                    if (block.contains("cache_control") && !block.at("cache_control").is_null()) {
                        note_cache_control(block.at("cache_control"));
                        msg_breakpoint = true;
                    }
                    tool_calls.push_back({
                        {"id", json_value(block, "id", std::string())},
                        {"type", "function"},
                        {"function", {
                            {"name", json_value(block, "name", std::string())},
                            {"arguments", json_value(block, "input", json::object()).dump()}
                        }}
                    });
                    has_tool_calls = true;
                } else if (type == "tool_result") {
                    // cache_control on a tool_result marks the tool message it converts into
                    const bool result_breakpoint =
                        block.contains("cache_control") && !block.at("cache_control").is_null();
                    if (result_breakpoint) {
                        note_cache_control(block.at("cache_control"));
                    }

                    std::string tool_use_id = json_value(block, "tool_use_id", std::string());

                    auto result_content = json_value(block, "content", json());
                    if (result_content.is_string()) {
                        tool_results.push_back({
                            {"role", "tool"},
                            {"tool_call_id", tool_use_id},
                            {"content", result_content.get<std::string>()}
                        });
                    } else if (result_content.is_array()) {
                        // Single-pass: build both text and content_parts, decide format at the end
                        std::string result_text;
                        json content_parts = json::array();
                        bool has_images = false;

                        for (const auto & c : result_content) {
                            std::string c_type = json_value(c, "type", std::string());
                            if (c_type == "text") {
                                std::string text = json_value(c, "text", std::string());
                                result_text += text;
                                content_parts.push_back({
                                    {"type", "text"},
                                    {"text", text}
                                });
                            } else if (c_type == "image") {
                                has_images = true;
                                json source = json_value(c, "source", json::object());
                                std::string source_type = json_value(source, "type", std::string());
                                if (source_type == "base64") {
                                    std::string media_type = json_value(source, "media_type", std::string("image/jpeg"));
                                    std::string data = json_value(source, "data", std::string());
                                    std::string url = "data:" + media_type + ";base64," + data;
                                    content_parts.push_back({
                                        {"type", "image_url"},
                                        {"image_url", {{"url", url}}}
                                    });
                                } else if (source_type == "url") {
                                    content_parts.push_back({
                                        {"type", "image_url"},
                                        {"image_url", {{"url", json_value(source, "url", std::string())}}}
                                    });
                                }
                            }
                        }

                        if (!has_images) {
                            // Text-only: collapse to a plain string for maximum compatibility
                            tool_results.push_back({
                                {"role", "tool"},
                                {"tool_call_id", tool_use_id},
                                {"content", result_text}
                            });
                        } else {
                            // Mixed or image-only: use array content parts (OpenAI multimodal tool format)
                            tool_results.push_back({
                                {"role", "tool"},
                                {"tool_call_id", tool_use_id},
                                {"content", content_parts}
                            });
                        }
                    } else {
                        tool_results.push_back({
                            {"role", "tool"},
                            {"tool_call_id", tool_use_id},
                            {"content", ""}
                        });
                    }

                    if (result_breakpoint) {
                        tool_results.back()["prompt_cache_breakpoint"] = {{"mode", "explicit"}};
                    }
                }
            }

            if (!converted_content.empty() || has_tool_calls || !reasoning_content.empty()) {
                json new_msg = {{"role", role}};
                if (!converted_content.empty()) {
                    new_msg["content"] = converted_content;
                } else if (has_tool_calls || !reasoning_content.empty()) {
                    new_msg["content"] = "";
                }
                if (!tool_calls.empty()) {
                    new_msg["tool_calls"] = tool_calls;
                }
                if (!reasoning_content.empty()) {
                    new_msg["reasoning_content"] = reasoning_content;
                }
                // a tool_use cache_control marks the message, but only when no text/image part
                // already carries the marker: several markers on one message are one breakpoint
                if (msg_breakpoint) {
                    bool part_has_breakpoint = false;
                    for (const auto & part : converted_content) {
                        if (part.contains("prompt_cache_breakpoint")) {
                            part_has_breakpoint = true;
                            break;
                        }
                    }
                    if (!part_has_breakpoint) {
                        new_msg["prompt_cache_breakpoint"] = {{"mode", "explicit"}};
                    }
                }
                oai_messages.push_back(new_msg);
            }

            for (const auto & tool_msg : tool_results) {
                oai_messages.push_back(tool_msg);
            }
        }
    }

    oai_body["messages"] = oai_messages;

    // Convert tools
    if (body.contains("tools")) {
        const json & tools = body.at("tools");
        if (tools.is_array()) {
            json oai_tools = json::array();
            bool tools_breakpoint = false;
            for (const auto & tool : tools) {
                // cache_control on a tool entry anchors right after the tools section
                if (tool.contains("cache_control") && !tool.at("cache_control").is_null()) {
                    note_cache_control(tool.at("cache_control"));
                    tools_breakpoint = true;
                }
                // strict / input_examples are accepted and ignored: the local tool grammar is
                // always schema-driven, so neither changes behavior
                oai_tools.push_back({
                    {"type", "function"},
                    {"function", {
                        {"name", json_value(tool, "name", std::string())},
                        {"description", json_value(tool, "description", std::string())},
                        {"parameters", tool.contains("input_schema") ? tool.at("input_schema") : json::object()}
                    }}
                });
            }
            oai_body["tools"] = oai_tools;
            if (tools_breakpoint && !oai_messages.empty()) {
                // the anchor sits at the start of the first message, i.e. after the tools;
                // server-context resolves the "tools" role to that position
                oai_body.at("messages").at(0)["prompt_cache_breakpoint"] = {{"mode", "explicit"}, {"anchor", "tools"}};
            }
        }
    }

    // Convert tool_choice
    if (body.contains("tool_choice")) {
        const json & tc = body.at("tool_choice");
        if (tc.is_object()) {
            std::string type = json_value(tc, "type", std::string());
            if (type == "auto") {
                oai_body["tool_choice"] = "auto";
            } else if (type == "any") {
                oai_body["tool_choice"] = "required";
            } else if (type == "tool") {
                // a named tool_choice forces exactly that tool
                oai_body["tool_choice"] = {
                    {"type", "function"},
                    {"function", {
                        {"name", json_value(tc, "name", std::string())}
                    }}
                };
            } else if (type == "none") {
                oai_body["tool_choice"] = "none";
            }
            if (json_value(tc, "disable_parallel_tool_use", false)) {
                oai_body["parallel_tool_calls"] = false;
            }
        }
    }

    // Convert stop_sequences to stop
    if (body.contains("stop_sequences")) {
        oai_body["stop"] = body.at("stop_sequences");
    }

    // max_tokens is required by the official Messages API (0 pre-warms the cache without
    // generating); /v1/messages/count_tokens has no such field at all
    if (!count_tokens && (!body.contains("max_tokens") || body.at("max_tokens").is_null())) {
        throw std::invalid_argument("'max_tokens' is required");
    }
    if (body.contains("max_tokens") && !body.at("max_tokens").is_null()) {
        if (!body.at("max_tokens").is_number_integer() || body.at("max_tokens").get<int>() < 0) {
            throw std::invalid_argument("'max_tokens' must be a non-negative integer");
        }
        oai_body["max_tokens"] = body.at("max_tokens");
    }

    // Pass through common params
    for (const auto & key : {"temperature", "top_p", "top_k", "stream", "chat_template_kwargs"}) {
        if (body.contains(key)) {
            oai_body[key] = body.at(key);
        }
    }

    // service_tier / container / inference_geo carry no local semantics, but the official
    // shapes are validated so invalid values fail early instead of being silently dropped.
    // The values are not forwarded: the local server has neither capacity tiers nor
    // containers, and the response always reports the standard tier (see server-task.cpp).
    if (body.contains("service_tier") && !body.at("service_tier").is_null()) {
        const std::string tier = json_value(body, "service_tier", std::string());
        if (tier != "auto" && tier != "standard_only") {
            throw std::invalid_argument("'service_tier' must be 'auto' or 'standard_only'");
        }
    }
    if (body.contains("container") && !body.at("container").is_null()) {
        if (!body.at("container").is_string() && !body.at("container").is_object()) {
            throw std::invalid_argument("'container' must be a string or an object");
        }
    }
    if (body.contains("inference_geo") && !body.at("inference_geo").is_null()) {
        const std::string geo = json_value(body, "inference_geo", std::string());
        if (geo != "global" && geo != "us") {
            throw std::invalid_argument("'inference_geo' must be 'global' or 'us'");
        }
    }

    // Handle Anthropic-specific output_config param
    // effort -> local reasoning_effort (the official enum is a subset of the local one)
    // format -> local response_format (json_schema)
    if (body.contains("output_config") && body.at("output_config").is_object()) {
        const json & output_config = body.at("output_config");
        if (output_config.contains("effort") && !output_config.at("effort").is_null()) {
            const std::string effort = json_value(output_config, "effort", std::string());
            if (effort != "low" && effort != "medium" && effort != "high" &&
                    effort != "xhigh" && effort != "max") {
                throw std::invalid_argument(
                    "'output_config.effort' must be one of: low, medium, high, xhigh, max");
            }
            oai_body["reasoning_effort"] = effort;
        }
        auto output_format = json_value(output_config, "format", json());
        if (output_format.is_object()) {
            const std::string format_type = json_value(output_format, "type", std::string());
            if (format_type == "json_schema") {
                oai_body["response_format"] = {
                    {"type", "json_schema"},
                    {"json_schema", {
                        {"schema", json_value(output_format, "schema", json::object())}
                    }}
                };
            }
        }
    }

    // Handle Anthropic-specific thinking param
    if (body.contains("thinking")) {
        const json thinking = json_value(body, "thinking", json::object());
        const std::string thinking_type = json_value(thinking, "type", std::string());
        if (thinking_type == "enabled") {
            const bool has_budget = thinking.contains("budget_tokens") && !thinking.at("budget_tokens").is_null();
            const int budget_tokens = json_value(thinking, "budget_tokens", 10000);
            if (has_budget) {
                // official bound: at least 1024. An omitted budget keeps the permissive local
                // default (the local reasoning budget is capped later).
                const bool has_max_tokens = oai_body.contains("max_tokens");
                const int  max_tokens     = json_value(oai_body, "max_tokens", 0);
                if (budget_tokens < 1024) {
                    throw std::invalid_argument("'thinking.budget_tokens' must be at least 1024");
                }
                // count_tokens carries no max_tokens, so the upper bound is unverifiable there
                if (has_max_tokens && budget_tokens >= max_tokens) {
                    throw std::invalid_argument("'thinking.budget_tokens' must be less than 'max_tokens'");
                }
            }
            oai_body["thinking_budget_tokens"] = budget_tokens;
        } else if (thinking_type == "disabled") {
            // "none" is the local way to disable thinking. It wins over output_config.effort:
            // the client asked for no thinking at all, so no effort hint is forwarded.
            oai_body["reasoning_effort"] = "none";
        } else if (thinking_type == "adaptive") {
            // adaptive lets the model size its own thinking: no token budget to set, so
            // effort / the local default decides (see output_config.effort above)
        }
        // display=omitted hides the thinking text in the response but keeps the block,
        // its signature and the signature_delta; the local serializers read this flag
        if (thinking.contains("display") && !thinking.at("display").is_null()) {
            const std::string display = json_value(thinking, "display", std::string());
            if (display != "summarized" && display != "omitted") {
                throw std::invalid_argument("'thinking.display' must be 'summarized' or 'omitted'");
            }
            oai_body["anthropic_thinking_display"] = display;
        }
    }

    // Top-level cache_control: the official "automatic caching" breakpoint sits on the last
    // cacheable block and moves forward as the conversation grows. Locally a breakpoint
    // anchors at a message boundary, so mark the last message part that can carry one.
    if (body.contains("cache_control") && !body.at("cache_control").is_null()) {
        note_cache_control(body.at("cache_control"));
        json & conv_messages = oai_body.at("messages");
        for (size_t i = conv_messages.size(); i-- > 0;) {
            json & msg = conv_messages.at(i);
            if (json_value(msg, "role", std::string()) == "tool" || !msg.contains("content")) {
                continue;
            }
            json & content = msg.at("content");
            if (content.is_string()) {
                const std::string text = content.get<std::string>();
                if (!text.empty()) {
                    content = json::array({{
                        {"type", "text"},
                        {"text", text},
                        {"prompt_cache_breakpoint", {{"mode", "explicit"}}},
                    }});
                    break;
                }
            }
            if (content.is_array() && !content.empty()) {
                json & part = content.back();
                const std::string part_type = json_value(part, "type", std::string());
                if (part_type == "text" || part_type == "image_url") {
                    part["prompt_cache_breakpoint"] = {{"mode", "explicit"}};
                    break;
                }
            }
        }
    }

    // cache_control.ttl -> internal cache TTL channel: the official prompt_cache_options.ttl
    // only accepts 30m, so the Anthropic 5m/1h values must bypass that whitelist
    if (!cache_ttl.empty()) {
        oai_body["__prompt_cache_ttl"] = cache_ttl;
    }

    // Handle Anthropic-specific metadata param
    if (body.contains("metadata")) {
        json metadata = json_value(body, "metadata", json::object());
        std::string user_id = json_value(metadata, "user_id", std::string());
        if (!user_id.empty()) {
            oai_body["__metadata_user_id"] = user_id;
        }
    }

    return oai_body;
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
