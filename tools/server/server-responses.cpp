#include "server-responses.h"
#include "server-openai-persist.h"
#include "server-web-search.h"
#include "server-conversations.h"

#include "server-common.h"
#include "log.h"

#include <algorithm>
#include <chrono>
#include <ctime>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <vector>

std::string server_responses_new_id() {
    return "resp_" + random_string();
}

namespace {
std::mutex & responses_seq_mutex() {
    static std::mutex mu;
    return mu;
}
std::unordered_map<std::string, int32_t> & responses_seq_map() {
    static std::unordered_map<std::string, int32_t> m;
    return m;
}

// Local opaque compaction token: "local." + base64(json). Not OpenAI cloud crypto.
std::string local_b64_encode(const std::string & in) {
    static const char * T = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve(((in.size() + 2) / 3) * 4);
    size_t i = 0;
    while (i + 2 < in.size()) {
        const uint32_t n = ((uint32_t) (unsigned char) in[i] << 16) |
                           ((uint32_t) (unsigned char) in[i + 1] << 8) |
                           ((uint32_t) (unsigned char) in[i + 2]);
        out.push_back(T[(n >> 18) & 63]);
        out.push_back(T[(n >> 12) & 63]);
        out.push_back(T[(n >> 6) & 63]);
        out.push_back(T[n & 63]);
        i += 3;
    }
    if (i + 1 == in.size()) {
        const uint32_t n = ((uint32_t) (unsigned char) in[i] << 16);
        out.push_back(T[(n >> 18) & 63]);
        out.push_back(T[(n >> 12) & 63]);
        out.push_back('=');
        out.push_back('=');
    } else if (i + 2 == in.size()) {
        const uint32_t n = ((uint32_t) (unsigned char) in[i] << 16) |
                           ((uint32_t) (unsigned char) in[i + 1] << 8);
        out.push_back(T[(n >> 18) & 63]);
        out.push_back(T[(n >> 12) & 63]);
        out.push_back(T[(n >> 6) & 63]);
        out.push_back('=');
    }
    return out;
}

bool local_b64_decode(const std::string & in, std::string & out) {
    auto val = [](char c) -> int {
        if (c >= 'A' && c <= 'Z') return c - 'A';
        if (c >= 'a' && c <= 'z') return c - 'a' + 26;
        if (c >= '0' && c <= '9') return c - '0' + 52;
        if (c == '+') return 62;
        if (c == '/') return 63;
        return -1;
    };
    if (in.size() % 4 != 0) {
        return false;
    }
    out.clear();
    out.reserve(in.size() / 4 * 3);
    for (size_t i = 0; i < in.size(); i += 4) {
        const int a = val(in[i]);
        const int b = val(in[i + 1]);
        const int c = in[i + 2] == '=' ? 0 : val(in[i + 2]);
        const int d = in[i + 3] == '=' ? 0 : val(in[i + 3]);
        if (a < 0 || b < 0 || (in[i + 2] != '=' && c < 0) || (in[i + 3] != '=' && d < 0)) {
            return false;
        }
        const uint32_t n = ((uint32_t) a << 18) | ((uint32_t) b << 12) | ((uint32_t) c << 6) | (uint32_t) d;
        out.push_back((char) ((n >> 16) & 0xff));
        if (in[i + 2] != '=') {
            out.push_back((char) ((n >> 8) & 0xff));
        }
        if (in[i + 3] != '=') {
            out.push_back((char) (n & 0xff));
        }
    }
    return true;
}

std::string make_local_compaction_token(const json & folded) {
    return "local." + local_b64_encode(folded.dump());
}

bool expand_local_compaction_token(const std::string & enc, json & folded_out) {
    static const std::string prefix = "local.";
    if (enc.size() <= prefix.size() || enc.compare(0, prefix.size(), prefix) != 0) {
        return false;
    }
    std::string raw;
    if (!local_b64_decode(enc.substr(prefix.size()), raw)) {
        return false;
    }
    try {
        folded_out = json::parse(raw);
        return folded_out.is_object();
    } catch (...) {
        return false;
    }
}
} // namespace

int32_t server_responses_next_seq(const std::string & resp_id) {
    std::lock_guard<std::mutex> lock(responses_seq_mutex());
    return responses_seq_map()[resp_id]++;
}

void server_responses_reset_seq(const std::string & resp_id) {
    std::lock_guard<std::mutex> lock(responses_seq_mutex());
    responses_seq_map().erase(resp_id);
}

static json server_responses_ensure_input_item_ids(json items) {
    if (!items.is_array()) {
        return items;
    }
    for (auto & item : items) {
        if (!item.is_object()) {
            continue;
        }
        if (!item.contains("id") || !item.at("id").is_string() ||
                item.at("id").get<std::string>().empty()) {
            const std::string typ = json_value(item, "type", std::string("message"));
            if (typ == "function_call" || typ == "function_call_output") {
                item["id"] = "fc_" + random_string();
            } else {
                item["id"] = "msg_" + random_string();
            }
        }
        if (!item.contains("type")) {
            item["type"] = "message";
        }
        if (item.contains("phase") && !item.at("phase").is_null()) {
            if (!item.at("phase").is_string()) {
                throw std::invalid_argument("'phase' must be 'commentary' or 'final_answer'");
            }
            const std::string phase = item.at("phase").get<std::string>();
            if (phase != "commentary" && phase != "final_answer") {
                throw std::invalid_argument("'phase' must be 'commentary' or 'final_answer'");
            }
        }
    }
    return items;
}

json server_responses_normalize_input(const json & input) {
    if (input.is_string()) {
        return server_responses_ensure_input_item_ids(json::array({
            json {
                {"role",    "user"},
                {"content", input.get<std::string>()},
                {"type",    "message"},
            },
        }));
    }
    if (input.is_array()) {
        return server_responses_ensure_input_item_ids(input);
    }
    throw std::invalid_argument("'input' must be a string or array of objects");
}

json server_responses_output_item_to_input(const json & output_item) {
    const std::string type = json_value(output_item, "type", std::string());

    if (type == "message") {
        json item = output_item;
        // EasyInputMessage / Input message style for the next turn
        if (!item.contains("type")) {
            item["type"] = "message";
        }
        return item;
    }

    if (type == "function_call") {
        // Pass through as function_call input item
        return output_item;
    }

    if (type == "reasoning") {
        // Keep reasoning in history when present (some clients echo it)
        return output_item;
    }

    if (type == "compaction") {
        // Opaque local/cloud compaction blob — expanded in prepare_request when local.*
        return output_item;
    }

    // Unknown output types: skip by returning null; caller filters
    return nullptr;
}

json server_responses_prepare_request(json body) {
    return server_responses_prepare_request(std::move(body), nullptr, 0);
}

json server_responses_prepare_request(
        json body,
        const llama_vocab * vocab,
        int32_t n_ctx_slot) {
    // OpenAI Responses create requires `model`.
    // Optional `prompt.id` expands local templates under --openai-files-path/prompts/.
    if (!body.contains("model") || !body.at("model").is_string() ||
            body.at("model").get<std::string>().empty()) {
        throw std::invalid_argument("'model' is required");
    }

    // Local prompt templates under --openai-files-path/prompts/{id}.json
    // Expands instructions/input from template + {{variables}} before history merge.
    auto substitute_vars = [](std::string text, const json & vars) {
        if (!vars.is_object()) {
            return text;
        }
        for (const auto & el : vars.items()) {
            if (!el.value().is_string()) {
                continue;
            }
            const std::string needle = "{{" + el.key() + "}}";
            const std::string repl = el.value().get<std::string>();
            for (;;) {
                const auto pos = text.find(needle);
                if (pos == std::string::npos) {
                    break;
                }
                text.replace(pos, needle.size(), repl);
            }
        }
        return text;
    };
    if (body.contains("prompt") && body.at("prompt").is_object()) {
        const json & prompt = body.at("prompt");
        const std::string pid = json_value(prompt, "id", std::string());
        if (pid.empty()) {
            throw std::invalid_argument("'prompt.id' is required and must be a non-empty string");
        }
        json tmpl;
        const std::string & root = openai_persist::root();
        if (root.empty()) {
            throw std::invalid_argument(
                "prompt templates require --openai-files-path (prompts/<id>.json)");
        }
        const std::string path =
            root + "/prompts/" + openai_persist::safe_id(pid) + ".json";
        std::ifstream in(path);
        if (!in) {
            throw std::invalid_argument("No such prompt template: " + pid);
        }
        try {
            std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            tmpl = json::parse(content);
        } catch (...) {
            throw std::invalid_argument("Invalid prompt template JSON: " + pid);
        }
        if (!tmpl.is_object() || tmpl.empty()) {
            throw std::invalid_argument("Prompt template is empty: " + pid);
        }
        const json vars = prompt.contains("variables") && prompt.at("variables").is_object()
                              ? prompt.at("variables")
                              : json::object();
        if (tmpl.contains("instructions") && tmpl.at("instructions").is_string()) {
            const std::string instr =
                substitute_vars(tmpl.at("instructions").get<std::string>(), vars);
            if (!body.contains("instructions") || body.at("instructions").is_null() ||
                    (body.at("instructions").is_string() &&
                     body.at("instructions").get<std::string>().empty())) {
                body["instructions"] = instr;
            }
        }
        if (tmpl.contains("input") && tmpl.at("input").is_string()) {
            const std::string inp =
                substitute_vars(tmpl.at("input").get<std::string>(), vars);
            if (!body.contains("input") || body.at("input").is_null()) {
                body["input"] = inp;
            }
        }
    }

    json new_input = json::array();
    bool expanded = false;
    const bool drop_hist_reasoning = server_responses_drop_history_reasoning(body);
    auto push_hist_item = [&](json item) {
        if (drop_hist_reasoning && item.is_object() &&
                json_value(item, "type", std::string()) == "reasoning") {
            return;
        }
        new_input.push_back(std::move(item));
    };

    const std::string prev_id = json_value(body, "previous_response_id", std::string());

    // Official conversation support: items from the conversation are prepended to the input
    // for this request; the finished turn is appended back after the response completes.
    std::string conv_id;
    if (body.contains("conversation") && !body.at("conversation").is_null()) {
        const json & conv = body.at("conversation");
        if (conv.is_string()) {
            conv_id = conv.get<std::string>();
        } else if (conv.is_object() && conv.contains("id") && conv.at("id").is_string()) {
            conv_id = conv.at("id").get<std::string>();
        } else {
            throw std::invalid_argument("'conversation' must be a string or an object with an 'id'");
        }
        if (conv_id.empty()) {
            throw std::invalid_argument("'conversation' id must not be empty");
        }
        if (!prev_id.empty()) {
            throw std::invalid_argument(
                "'conversation' cannot be used in conjunction with 'previous_response_id'");
        }
        auto conv_entry = server_conversations_store::instance().get(conv_id);
        if (!conv_entry.has_value()) {
            throw std::invalid_argument("conversation not found: " + conv_id);
        }
        for (const auto & item : conv_entry->items) {
            json as_input = server_responses_output_item_to_input(item);
            if (as_input.is_null() && json_value(item, "type", std::string()) == "function_call_output") {
                as_input = item;
            }
            if (!as_input.is_null()) {
                push_hist_item(std::move(as_input));
            }
        }
        expanded = true;
        // Official Response objects echo the conversation as an object with an id.
        // The request's own items join the conversation below, once they are normalized.
        body["conversation"] = json { {"id", conv_id} };
    }

    if (!prev_id.empty()) {
        auto prev = server_responses_store::instance().get(prev_id);
        if (!prev.has_value()) {
            throw std::invalid_argument(
                "previous_response_id not found or expired: " + prev_id);
        }
        // a store=false response is retained for GET/cancel only, never as context
        if (!json_value(prev->response, "store", true)) {
            throw std::invalid_argument(
                "previous_response_id not found or expired: " + prev_id);
        }
        // Compact responses already fold history into output (users + local. blob).
        // Expanding their stored input again would duplicate turns.
        const bool prev_is_compaction =
            prev->response.is_object() &&
            json_value(prev->response, "object", std::string()) == "response.compaction";
        if (!prev_is_compaction) {
            json prev_input = server_responses_normalize_input(prev->input);
            for (const auto & item : prev_input) {
                push_hist_item(item);
            }
        }
        if (prev->output.is_array()) {
            for (const auto & out_item : prev->output) {
                if (!out_item.is_object()) {
                    continue;
                }
                if (json_value(out_item, "type", std::string()) == "compaction") {
                    const std::string enc = json_value(out_item, "encrypted_content", std::string());
                    json folded;
                    if (expand_local_compaction_token(enc, folded) &&
                            folded.contains("items") && folded.at("items").is_array()) {
                        for (const auto & item : folded.at("items")) {
                            push_hist_item(item);
                        }
                    }
                    continue;
                }
                json as_input = server_responses_output_item_to_input(out_item);
                if (!as_input.is_null()) {
                    push_hist_item(std::move(as_input));
                }
            }
        }
        expanded = true;
        // the Response object echoes previous_response_id; keep the request value for enrich
        body["__oai_prev_response_id"] = prev_id;
        body.erase("previous_response_id");
    }

    // input is optional in the official API; normalize once so the response echo, the
    // conversation items and the prompt all carry the same item ids
    const json curr_input = (body.contains("input") && !body.at("input").is_null())
                                ? server_responses_normalize_input(body.at("input"))
                                : json::array();
    body["input"] = curr_input;
    if (!conv_id.empty()) {
        // only the request's own items join the conversation; the prepended history is already there
        body["__oai_conv_input"] = curr_input;
    }
    for (const auto & item : curr_input) {
        new_input.push_back(item);
    }

    if (expanded) {
        body["input"] = std::move(new_input);
        LOG_DBG("Expanded previous_response context -> %zu input items\n",
                body.at("input").size());
    }

    // context_management: auto-apply local compaction when history is long enough.
    bool want_context_compact = false;
    if (body.contains("context_management") && body.at("context_management").is_array()) {
        for (const auto & cm : body.at("context_management")) {
            if (cm.is_object() && json_value(cm, "type", std::string()) == "compaction") {
                want_context_compact = true;
                break;
            }
        }
    }
    if (want_context_compact) {
        json input = server_responses_normalize_input(body.at("input"));
        size_t foldable = 0;
        for (const auto & item : input) {
            if (!item.is_object()) {
                continue;
            }
            if (json_value(item, "role", std::string()) != "user") {
                foldable++;
            }
        }
        if (foldable >= 2 || input.size() >= 4) {
            body["input"] = server_responses_fold_input_compaction(input);
            LOG_DBG("context_management compaction folded input -> %zu items\n",
                    body.at("input").size());
        }
    }

    // Cloud-shaped fields: validate wire shape/enums (invalid → 400), then apply local deepen.
    server_openai_validate_cloud_shaped_fields(body, /*allow_prompt=*/true);
    server_openai_apply_prompt_cache_semantics(body);
    // Local web_search deepen (strip hosted web_search tools, inject results, emit later).
    server_openai_apply_web_search_semantics(body);
    // Official Responses `reasoning` object (effort/context/summary/generate_summary/mode).
    server_openai_validate_reasoning_object(body);

    // Local truncation policy (OpenAI-shaped deepen):
    // real tokenizer counts vs slot context budget (n_ctx_slot - max_output_tokens).
    {
        std::string trunc = "disabled";
        if (body.contains("truncation") && !body.at("truncation").is_null()) {
            if (!body.at("truncation").is_string()) {
                throw std::invalid_argument("'truncation' must be 'auto' or 'disabled'");
            }
            trunc = body.at("truncation").get<std::string>();
        }
        if (trunc != "auto" && trunc != "disabled") {
            throw std::invalid_argument("'truncation' must be 'auto' or 'disabled'");
        }
        if (vocab != nullptr && n_ctx_slot > 0) {
            auto item_text = [](const json & item) -> std::string {
                std::string text;
                auto take = [&](const json & j) {
                    if (j.is_string()) {
                        text += j.get<std::string>();
                    } else if (j.is_array()) {
                        for (const auto & p : j) {
                            if (p.is_object() && p.contains("text") && p.at("text").is_string()) {
                                text += p.at("text").get<std::string>();
                            } else if (p.is_string()) {
                                text += p.get<std::string>();
                            }
                        }
                    }
                };
                if (item.is_object()) {
                    if (item.contains("content")) {
                        take(item.at("content"));
                    }
                    if (item.contains("text") && item.at("text").is_string()) {
                        text += item.at("text").get<std::string>();
                    }
                }
                return text;
            };
            auto estimate_item_tokens = [&](const json & item) -> size_t {
                const std::string text = item_text(item);
                if (text.empty()) {
                    return 1;
                }
                const auto toks = common_tokenize(vocab, text, false, true);
                return std::max<size_t>(1, toks.size());
            };
            int32_t max_out = 0;
            if (body.contains("max_output_tokens") && body.at("max_output_tokens").is_number_integer()) {
                max_out = std::max(0, body.at("max_output_tokens").get<int>());
            }
            size_t budget = (size_t) std::max(1, n_ctx_slot - max_out);
            // Reserve room for instructions + chat-template / role wrappers.
            // Raw content token counts under-estimate the final prompt (often by hundreds).
            size_t overhead = (size_t) std::min(512, std::max(128, n_ctx_slot / 8));
            if (body.contains("instructions") && body.at("instructions").is_string()) {
                const auto itoks = common_tokenize(
                    vocab, body.at("instructions").get<std::string>(), false, true);
                overhead += itoks.size();
            }
            if (overhead + 1 < budget) {
                budget -= overhead;
            } else {
                budget = 1;
            }
            json input = server_responses_normalize_input(body.at("input"));
            size_t total = 0;
            for (const auto & item : input) {
                total += estimate_item_tokens(item);
            }
            if (total > budget) {
                if (trunc == "disabled") {
                    throw std::invalid_argument(
                        "input exceeds context token budget (" +
                        std::to_string(budget) +
                        " tokens); set truncation=auto to drop oldest items");
                }
                json kept = json::array();
                size_t kept_tok = 0;
                for (size_t i = input.size(); i-- > 0; ) {
                    const size_t t = estimate_item_tokens(input.at(i));
                    if (t > budget) {
                        // Skip items that alone cannot fit; continue dropping older ones.
                        continue;
                    }
                    if (kept_tok + t > budget) {
                        break;
                    }
                    kept.push_back(input.at(i));
                    kept_tok += t;
                }
                if (kept.empty()) {
                    throw std::invalid_argument(
                        "input exceeds context token budget (" +
                        std::to_string(budget) + " tokens)");
                }
                // common_json::iterator is forward-only, so reverse manually
                json reversed = json::array();
                for (size_t ri = kept.size(); ri-- > 0;) {
                    reversed.push_back(kept.at(ri));
                }
                kept = std::move(reversed);
                body["input"] = std::move(kept);
                LOG_DBG("truncation=auto tokens %zu -> %zu budget=%zu items=%zu\n",
                        total, kept_tok, budget, body.at("input").size());
            }
        }
    }

    return body;
}

// A finished turn joins the conversation named on the response object (official:
// input items and output items are added after the response completes). store=false
// only controls previous_response_id retention, so this runs before those early-outs.
static void server_responses_conversation_attach(const json & response_obj, const json & conversation_input) {
    if (!response_obj.is_object() || !response_obj.contains("conversation")) {
        return;
    }
    const json & conv = response_obj.at("conversation");
    std::string conv_id;
    if (conv.is_string()) {
        conv_id = conv.get<std::string>();
    } else if (conv.is_object()) {
        conv_id = json_value(conv, "id", std::string());
    }
    if (conv_id.empty()) {
        return;
    }
    const std::string status = json_value(response_obj, "status", std::string());
    if (status != "completed" && status != "incomplete") {
        return; // only a finished turn joins the conversation
    }
    const json & output_items = response_obj.contains("output") ? response_obj.at("output") : json::array();
    if (!server_conversations_append_turn(conv_id, conversation_input, output_items)) {
        LOG_WRN("%s: conversation not found: %s\n", __func__, conv_id.c_str());
    }
}

void server_responses_remember(
    const json & response_obj,
    const json & prepared_request_input,
    const json & instructions,
    const json & conversation_input,
    const std::string & request_model) {
    server_responses_conversation_attach(response_obj, conversation_input);

    if (server_responses_store::instance().max_entries() <= 0) {
        return;
    }
    if (!response_obj.contains("id") || !response_obj.at("id").is_string()) {
        return;
    }
    // OpenAI: store=false keeps the response out of previous_response_id chains. Background
    // responses are still retained: GET and cancel must work while they run.
    const bool should_store  = json_value(response_obj, "store", true);
    const bool is_background = json_value(response_obj, "background", false);
    if (!should_store && !is_background) {
        return;
    }

    // Background cancel wins over a late completed/failed write.
    {
        const std::string nid = response_obj.at("id").get<std::string>();
        const std::string nst = json_value(response_obj, "status", std::string());
        auto cur = server_responses_store::instance().get(nid);
        if (cur.has_value()) {
            const std::string st = json_value(cur->response, "status", std::string());
            if (st == "cancelled" && nst != "cancelled") {
                return;
            }
        }
    }

    server_responses_store_entry entry;
    entry.id            = response_obj.at("id").get<std::string>();
    entry.created_at    = json_value(response_obj, "created_at", (int64_t) 0);
    entry.model         = json_value(response_obj, "model", std::string());
    entry.request_model = request_model;
    entry.instructions  = instructions.is_null() ? json(nullptr) : instructions;
    entry.input         = prepared_request_input.is_null()
                              ? json::array()
                              : server_responses_normalize_input(prepared_request_input);
    entry.output        = response_obj.contains("output") ? response_obj.at("output") : json::array();
    entry.usage         = response_obj.contains("usage") ? response_obj.at("usage") : json::object();
    entry.response      = response_obj;

    server_responses_store::instance().put(std::move(entry));
}

static void server_responses_echo_prompt_cache_options(json & response_obj, const json & request_body) {
    if (!request_body.contains("prompt_cache_options") ||
            !request_body.at("prompt_cache_options").is_object()) {
        return;
    }
    if (!response_obj.contains("prompt_cache_options") ||
            !response_obj.at("prompt_cache_options").is_object()) {
        response_obj["prompt_cache_options"] = request_body.at("prompt_cache_options");
    }
    // The SDK requires mode+ttl on the echoed object; fill local defaults, user values win.
    json & opts = response_obj.at("prompt_cache_options");
    if (!opts.contains("mode") || opts.at("mode").is_null()) {
        opts["mode"] = "implicit";
    }
    if (!opts.contains("ttl") || opts.at("ttl").is_null()) {
        opts["ttl"] = "30m";
    }
}

// Request tools plus the web_search tools stripped before templating; the Response
// echo carries this shape, so the comparison must see both sides alike.
static json server_responses_tools_for_compare(const json & request_body) {
    json tools = json::array();
    if (request_body.contains("tools") && request_body.at("tools").is_array()) {
        tools = request_body.at("tools");
    }
    if (request_body.contains("__oai_web_search_echo_tools") &&
            request_body.at("__oai_web_search_echo_tools").is_array()) {
        for (const auto & t : request_body.at("__oai_web_search_echo_tools")) {
            tools.push_back(t);
        }
    }
    return tools;
}

static void server_responses_inject_prompt_cache_diagnostics(json & response_obj, const json & request_body) {
    if (!request_body.contains("prompt_cache_options") ||
            !request_body.at("prompt_cache_options").is_object()) {
        return;
    }
    const json & opts = request_body.at("prompt_cache_options");
    if (!opts.contains("comparison_response_id") || !opts.at("comparison_response_id").is_string()) {
        return;
    }
    const std::string comparison_id = opts.at("comparison_response_id").get<std::string>();
    if (comparison_id.empty()) {
        return;
    }
    if (!response_obj.contains("usage") || !response_obj.at("usage").is_object()) {
        return;
    }
    const json & usage = response_obj.at("usage");
    if (!usage.contains("input_tokens") || !usage.at("input_tokens").is_number()) {
        return;
    }
    const int64_t cur_input = usage.at("input_tokens").get<int64_t>();
    if (cur_input <= 0) {
        return;
    }
    int64_t cur_cached = 0;
    if (usage.contains("input_tokens_details") && usage.at("input_tokens_details").is_object()) {
        const json & itd = usage.at("input_tokens_details");
        if (itd.contains("cached_tokens") && itd.at("cached_tokens").is_number()) {
            cur_cached = itd.at("cached_tokens").get<int64_t>();
        }
    }

    const auto entry = server_responses_store::instance().get(comparison_id);
    int64_t base_input = 0;
    bool have_base = false;
    if (entry.has_value() && entry->response.contains("usage") && entry->response.at("usage").is_object()) {
        const json & base_usage = entry->response.at("usage");
        if (base_usage.contains("input_tokens") && base_usage.at("input_tokens").is_number()) {
            base_input = base_usage.at("input_tokens").get<int64_t>();
            have_base = base_input > 0;
        }
    }
    if (!have_base) {
        response_obj["prompt_cache_diagnostics"] = json {
            {"type", "comparison_response_not_found"},
        };
        return;
    }

    // Expected = the prefix the two requests can share at most; a repeated prompt
    // locally reuses all but its last 4 tokens, so allow that slack. Short prompts
    // have no tail to re-evaluate: require the full shared prefix.
    const int64_t expected = std::min(cur_input, base_input);
    if (cur_cached >= expected - (expected > 4 ? 4 : 0)) {
        response_obj["prompt_cache_diagnostics"] = json {
            {"type", "cache_hit"},
        };
        return;
    }

    // Miss: attribute it by diffing the baseline response echo against this request,
    // first match wins; the fallback reports input_changed.
    const json & base = entry->response;
    auto member_or_null = [](const json & obj, const char * key) -> json {
        if (obj.is_object() && obj.contains(key) && !obj.at(key).is_null()) {
            return obj.at(key);
        }
        return nullptr;
    };
    auto nested_or_null = [](const json & obj, const char * outer, const char * inner) -> json {
        if (!obj.is_object() || !obj.contains(outer) || !obj.at(outer).is_object()) {
            return nullptr;
        }
        const json & o = obj.at(outer);
        return o.contains(inner) && !o.at(inner).is_null() ? o.at(inner) : json(nullptr);
    };

    std::string reason = "input_changed";
    // The response echo carries the loaded model name, so the model check compares the
    // request against the baseline request-side model stored on the entry.
    if (!entry->request_model.empty() &&
            member_or_null(request_body, "model") != json(entry->request_model)) {
        reason = "model_changed";
    } else if (member_or_null(request_body, "prompt_cache_key") != member_or_null(base, "prompt_cache_key")) {
        reason = "prompt_cache_key_changed";
    } else {
        auto tier_norm = [](const json & v) -> json {
            if (v.is_null()) {
                return "auto"; // official default when the request omits service_tier
            }
            if (v.is_string() && v.get<std::string>() == "fast") {
                return "priority"; // the response echo stores the normalized tier
            }
            return v;
        };
        if (tier_norm(member_or_null(request_body, "service_tier")) != tier_norm(member_or_null(base, "service_tier"))) {
            reason = "service_tier_changed";
        } else if (server_responses_tools_for_compare(request_body) != member_or_null(base, "tools")) {
            reason = "tools_changed";
        } else if (nested_or_null(request_body, "text", "format") != nested_or_null(base, "text", "format")) {
            reason = "text_format_changed";
        } else if (nested_or_null(request_body, "reasoning", "effort") != nested_or_null(base, "reasoning", "effort")) {
            reason = "reasoning_effort_changed";
        } else if (nested_or_null(request_body, "text", "verbosity") != nested_or_null(base, "text", "verbosity")) {
            reason = "verbosity_changed";
        }
    }
    response_obj["prompt_cache_diagnostics"] = json {
        {"type", "cache_miss"},
        {"reason", reason},
        {"cache_missed_tokens", expected - cur_cached},
        {"comparison_reusable_tokens", base_input},
    };
}

json server_responses_enrich_response(json response_obj, const json & request_body) {
    if (!response_obj.contains("object")) {
        response_obj["object"] = "response";
    }
    if (!response_obj.contains("status")) {
        response_obj["status"] = "completed";
    }

    // Echo request fields when present (OpenAI includes many on the Response object)
    static const char * passthrough_keys[] = {
        "temperature", "top_logprobs", "top_p", "truncation", "metadata", "store",
        "service_tier", "user", "max_output_tokens", "tools", "tool_choice",
        "parallel_tool_calls", "text", "reasoning", "instructions",
        "background", "include", "max_tool_calls", "prompt",
        "prompt_cache_key", "prompt_cache_retention", "prompt_cache_options",
        "safety_identifier", "stream_options", "context_management", "conversation",
    };
    for (const char * key : passthrough_keys) {
        if (request_body.contains(key) && !response_obj.contains(key)) {
            response_obj[key] = request_body.at(key);
        }
    }
    // previous_response_id is stripped from the prepared request; enrich sees the kept value
    if (!response_obj.contains("previous_response_id") &&
            request_body.contains("__oai_prev_response_id")) {
        response_obj["previous_response_id"] = request_body.at("__oai_prev_response_id");
    }
    // Fast mode: official responses show service_tier=priority for request fast or priority.
    if (response_obj.contains("service_tier") && response_obj.at("service_tier").is_string() &&
            response_obj.at("service_tier").get<std::string>() == "fast") {
        response_obj["service_tier"] = "priority";
    }

    // OpenAI SDK treats these as required on every Response object.
    if (!response_obj.contains("tools") || response_obj.at("tools").is_null()) {
        response_obj["tools"] = json::array();
    }
    // Restore web_search tools stripped during local deepen so Response echoes request shape.
    if (request_body.contains("__oai_web_search_echo_tools") &&
            request_body.at("__oai_web_search_echo_tools").is_array()) {
        for (const auto & t : request_body.at("__oai_web_search_echo_tools")) {
            response_obj["tools"].push_back(t);
        }
    }
    if (!response_obj.contains("tool_choice") || response_obj.at("tool_choice").is_null()) {
        response_obj["tool_choice"] = "auto";
    }
    if (!response_obj.contains("parallel_tool_calls") || response_obj.at("parallel_tool_calls").is_null()) {
        response_obj["parallel_tool_calls"] = true;
    }
    // Default store=true (OpenAI API default) so clients can rely on the field.
    if (!response_obj.contains("store") || response_obj.at("store").is_null()) {
        response_obj["store"] = true;
    }

    // Convenience field used by OpenAI SDKs
    if (!response_obj.contains("output_text") && response_obj.contains("output") &&
        response_obj.at("output").is_array()) {
        std::string text;
        for (const auto & item : response_obj.at("output")) {
            if (json_value(item, "type", std::string()) != "message") {
                continue;
            }
            if (!item.contains("content") || !item.at("content").is_array()) {
                continue;
            }
            for (const auto & part : item.at("content")) {
                if (json_value(part, "type", std::string()) == "output_text" &&
                    part.contains("text") && part.at("text").is_string()) {
                    text += part.at("text").get<std::string>();
                }
            }
        }
        response_obj["output_text"] = text;
    }

    // Usage details expected by some clients / OpenAI SDK
    if (response_obj.contains("usage") && response_obj.at("usage").is_object()) {
        auto & usage = response_obj.at("usage");
        if (!usage.contains("output_tokens_details") || !usage.at("output_tokens_details").is_object()) {
            usage["output_tokens_details"] = json {
                {"reasoning_tokens", 0},
            };
        } else if (!usage.at("output_tokens_details").contains("reasoning_tokens")) {
            usage["output_tokens_details"]["reasoning_tokens"] = 0;
        }
        if (!usage.contains("input_tokens_details") || !usage.at("input_tokens_details").is_object()) {
            usage["input_tokens_details"] = json {
                {"cached_tokens", 0},
                {"cache_write_tokens", 0},
            };
        } else {
            auto & itd = usage.at("input_tokens_details");
            if (!itd.contains("cached_tokens")) {
                itd["cached_tokens"] = 0;
            }
            if (!itd.contains("cache_write_tokens")) {
                itd["cache_write_tokens"] = 0;
            }
        }
    }

    // Echoed prompt_cache_options complete with local defaults; a comparison id adds diagnostics.
    server_responses_echo_prompt_cache_options(response_obj, request_body);
    server_responses_inject_prompt_cache_diagnostics(response_obj, request_body);

    if (!response_obj.contains("error")) {
        response_obj["error"] = nullptr;
    }
    if (!response_obj.contains("incomplete_details")) {
        response_obj["incomplete_details"] = nullptr;
    }

    // Local web_search deepen: prepend hosted-shaped web_search_call / open_page items.
    if (json_value(request_body, "__oai_web_search", false) &&
            response_obj.contains("output") && response_obj.at("output").is_array()) {
        json prefix = server_web_search_responses_output_items(request_body);
        if (!prefix.empty()) {
            json new_output = json::array();
            for (auto & item : prefix) {
                new_output.push_back(std::move(item));
            }
            for (auto & item : response_obj.at("output")) {
                new_output.push_back(std::move(item));
            }
            response_obj["output"] = std::move(new_output);
        }
        server_web_search_annotate_responses_output(response_obj, request_body);
    }

    return response_obj;
}

json server_responses_build_error_failed_sse_events(
    const std::string & resp_id,
    const std::string & model,
    const json & request_body,
    const std::string & message,
    const std::string & code) {
    const int64_t t = (int64_t) std::time(nullptr);

    json failed = {
        {"id",         resp_id},
        {"object",     "response"},
        {"created_at", t},
        {"completed_at", t},
        {"model",      model},
        {"status",     "failed"},
        {"output",     json::array()},
        {"error",      json {
            {"code",    code},
            {"message", message},
        }},
        {"incomplete_details", nullptr},
    };
    failed = server_responses_enrich_response(std::move(failed), request_body);
    failed["status"] = "failed";
    failed["error"] = json {
        {"code",    code},
        {"message", message},
    };

    auto push = [&](const std::string & event_name, json data) {
        data["type"] = event_name;
        data["sequence_number"] = server_responses_next_seq(resp_id);
        return json {
            {"event", event_name},
            {"data",  std::move(data)},
        };
    };

    json events = json::array();
    events.push_back(push("error", json {
        {"code",    code},
        {"message", message},
        {"param",   nullptr},
    }));
    events.push_back(push("response.failed", json { {"response", failed} }));

    server_responses_remember(
        failed,
        request_body.contains("input") ? request_body.at("input") : json::array(),
        request_body.contains("instructions") ? request_body.at("instructions") : json(nullptr));
    server_responses_reset_seq(resp_id);
    return events;
}

static int64_t server_responses_now_unix() {
    return (int64_t) std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

static json server_responses_usage_zero() {
    return json {
        {"input_tokens",  0},
        {"output_tokens", 0},
        {"total_tokens",  0},
        {"input_tokens_details", json {
            {"cached_tokens", 0},
        }},
        {"output_tokens_details", json {
            {"reasoning_tokens", 0},
        }},
    };
}

static json server_responses_input_item_for_list(json item) {
    if (!item.is_object()) {
        return item;
    }
    if (!item.contains("id") || !item.at("id").is_string()) {
        item["id"] = "msg_" + random_string();
    }
    if (!item.contains("type")) {
        item["type"] = "message";
    }
    if (item.contains("content") && item.at("content").is_string()) {
        item["content"] = json::array({
            json {
                {"type", "input_text"},
                {"text", item.at("content").get<std::string>()},
            },
        });
    }
    if (!item.contains("status")) {
        item["status"] = "completed";
    }
    return item;
}

json server_responses_cancel(const std::string & response_id) {
    auto entry = server_responses_store::instance().get(response_id);
    if (!entry.has_value()) {
        throw std::invalid_argument("response not found or expired: " + response_id);
    }

    json body = entry->response.is_object() && !entry->response.empty()
                    ? entry->response
                    : json {
                          {"id",         entry->id},
                          {"object",     "response"},
                          {"created_at", entry->created_at},
                          {"model",      entry->model},
                          {"output",     entry->output},
                          {"usage",      entry->usage},
                      };
    const std::string status = json_value(body, "status", std::string("completed"));
    // OpenAI: only in-progress/queued (background) responses can be cancelled.
    if (status != "in_progress" && status != "queued") {
        throw std::invalid_argument(
            "Only background responses that are in_progress or queued can be cancelled (status=" + status + ")");
    }
    body["status"] = "cancelled";
    body = server_responses_enrich_response(std::move(body), json::object());

    entry->response = body;
    entry->output   = body.contains("output") ? body.at("output") : entry->output;
    server_responses_store::instance().put(std::move(*entry));
    return body;
}

json server_responses_list_input_items(const std::string & response_id,
                                       const std::string & after,
                                       const std::string & order,
                                       int64_t limit,
                                       const json & include) {
    auto entry = server_responses_store::instance().get(response_id);
    if (!entry.has_value()) {
        throw std::invalid_argument("response not found or expired: " + response_id);
    }
    if (order != "asc" && order != "desc") {
        throw std::invalid_argument("'order' must be 'asc' or 'desc'");
    }
    if (limit <= 0 || limit > 100) {
        throw std::invalid_argument("'limit' must be between 1 and 100");
    }

    const json input = server_responses_normalize_input(entry->input);
    const int64_t n = (int64_t) input.size();
    int64_t idx_after = -1;
    if (!after.empty()) {
        for (int64_t i = 0; i < n; ++i) {
            if (input.at(i).is_object() && json_value(input.at(i), "id", std::string()) == after) {
                idx_after = i;
                break;
            }
        }
        if (idx_after < 0) {
            throw std::invalid_argument("'after' item not found for response: " + after);
        }
    }

    // stored order is oldest first; the official list defaults to newest first (desc)
    std::vector<json> data;
    bool has_more = false;
    if (order == "desc") {
        int64_t i = after.empty() ? n - 1 : idx_after - 1;
        for (; i >= 0 && (int64_t) data.size() < limit; --i) {
            data.push_back(server_responses_input_item_for_list(input.at(i)));
        }
        has_more = i >= 0;
    } else {
        int64_t i = after.empty() ? 0 : idx_after + 1;
        for (; i < n && (int64_t) data.size() < limit; ++i) {
            data.push_back(server_responses_input_item_for_list(input.at(i)));
        }
        has_more = i < n;
    }

    // include gates optional fields (logprobs, encrypted_content, search results)
    json filtered = json::array();
    for (auto & item : data) {
        filtered.push_back(item.is_object() ? server_conversation_item_filter(item, include) : item);
    }

    std::string first_id;
    std::string last_id;
    if (!filtered.empty()) {
        first_id = json_value(filtered.front(), "id", std::string());
        last_id  = json_value(filtered.back(), "id", std::string());
    }

    return json {
        {"object",    "list"},
        {"data",      std::move(filtered)},
        {"first_id",  first_id.empty() ? nullptr : json(first_id)},
        {"last_id",   last_id.empty() ? nullptr : json(last_id)},
        {"has_more",  has_more},
    };
}

json server_responses_apply_output_include(json response_obj, const json & include) {
    if (!response_obj.is_object() || !response_obj.contains("output") ||
            !response_obj.at("output").is_array()) {
        return response_obj;
    }
    json filtered = json::array();
    for (const auto & item : response_obj.at("output")) {
        // official retrieve gates optional item fields behind include
        filtered.push_back(item.is_object() ? server_conversation_item_filter(item, include) : item);
    }
    response_obj["output"] = std::move(filtered);
    return response_obj;
}

json server_responses_compact(json body, const llama_vocab * vocab, int32_t n_ctx_slot) {
    body = server_responses_prepare_request(std::move(body), vocab, n_ctx_slot);

    json input = server_responses_normalize_input(body.at("input"));
    json folded_input = server_responses_fold_input_compaction(input);
    json output = json::array();
    for (const auto & item : folded_input) {
        if (!item.is_object()) {
            continue;
        }
        if (json_value(item, "role", std::string()) == "user") {
            json msg = server_responses_input_item_for_list(item);
            msg["status"] = "completed";
            output.push_back(std::move(msg));
        } else if (json_value(item, "type", std::string()) == "compaction") {
            output.push_back(item);
        }
    }

    json usage = server_responses_usage_zero();
    usage["input_tokens"]  = (int) input.size();
    usage["output_tokens"] = 1;
    usage["total_tokens"]  = (int) input.size() + 1;

    const bool should_store = json_value(body, "store", true);
    json result = {
        {"id",         "resp_" + random_string()},
        {"created_at", server_responses_now_unix()},
        {"object",     "response.compaction"},
        {"status",     "completed"},
        {"model",      json_value(body, "model", std::string())},
        {"store",      should_store},
        {"output",     std::move(output)},
        {"usage",      std::move(usage)},
    };
    // Persist so previous_response_id can expand local. compaction tokens.
    // Store empty input: history lives entirely in output (users + local. blob).
    server_responses_remember(
        result,
        json::array(),
        body.contains("instructions") ? body.at("instructions") : json(nullptr),
        json(nullptr),
        json_value(body, "model", std::string()));
    return result;
}

json server_responses_fold_input_compaction(const json & input) {
    json normalized = server_responses_normalize_input(input);
    json out = json::array();
    json folded_items = json::array();
    for (const auto & item : normalized) {
        if (!item.is_object()) {
            continue;
        }
        const std::string role = json_value(item, "role", std::string());
        const std::string type = json_value(item, "type", std::string());
        if (role == "user") {
            out.push_back(item);
            continue;
        }
        if (type == "compaction") {
            const std::string enc = json_value(item, "encrypted_content", std::string());
            json nested;
            if (expand_local_compaction_token(enc, nested) &&
                    nested.contains("items") && nested.at("items").is_array()) {
                for (const auto & nested_item : nested.at("items")) {
                    folded_items.push_back(nested_item);
                }
            }
            continue;
        }
        folded_items.push_back(item);
    }
    if (folded_items.empty() && out.size() == normalized.size()) {
        return normalized;
    }
    out.push_back(json{
        {"id",                "ftcmp_" + random_string()},
        {"type",              "compaction"},
        {"encrypted_content", make_local_compaction_token(json{{"v", 1}, {"items", folded_items}})},
    });
    return out;
}

int server_responses_max_tool_calls(const json & request_body) {
    if (!request_body.contains("max_tool_calls") || request_body.at("max_tool_calls").is_null()) {
        return -1;
    }
    if (!request_body.at("max_tool_calls").is_number_integer()) {
        throw std::invalid_argument("'max_tool_calls' must be an integer");
    }
    return request_body.at("max_tool_calls").get<int>();
}

int server_responses_effective_tool_call_cap(const json & request_body) {
    int cap = server_responses_max_tool_calls(request_body);
    bool parallel = true;
    if (request_body.contains("parallel_tool_calls") && request_body.at("parallel_tool_calls").is_boolean()) {
        parallel = request_body.at("parallel_tool_calls").get<bool>();
    }
    if (!parallel) {
        if (cap < 0) {
            return 1;
        }
        return std::min(cap, 1);
    }
    return cap;
}

bool server_responses_include_contains(const json & request_body, const std::string & item) {
    if (!request_body.contains("include") || !request_body.at("include").is_array()) {
        return false;
    }
    for (const auto & inc : request_body.at("include")) {
        if (inc.is_string() && inc.get<std::string>() == item) {
            return true;
        }
    }
    return false;
}

bool server_responses_wants_output_logprobs(const json & request_body) {
    if (server_responses_include_contains(request_body, "message.output_text.logprobs")) {
        return true;
    }
    return request_body.contains("top_logprobs") && !request_body.at("top_logprobs").is_null() &&
           json_value(request_body, "top_logprobs", 0) > 0;
}

std::string server_responses_encode_local_blob(const json & obj) {
    return make_local_compaction_token(obj);
}

bool server_responses_expand_local_blob(const std::string & enc, json & out) {
    return expand_local_compaction_token(enc, out);
}

bool server_responses_drop_history_reasoning(const json & request_body) {
    if (!request_body.contains("reasoning") || !request_body.at("reasoning").is_object()) {
        return false;
    }
    // Official: current_turn keeps only the latest turn's reasoning for the model.
    return json_value(request_body.at("reasoning"), "context", std::string()) == "current_turn";
}

std::string server_responses_reasoning_summary_text(const std::string & reasoning_text, const json & request_body) {
    if (reasoning_text.empty() ||
            !request_body.contains("reasoning") || !request_body.at("reasoning").is_object()) {
        return std::string();
    }
    const json & reasoning = request_body.at("reasoning");
    std::string level;
    if (reasoning.contains("summary") && reasoning.at("summary").is_string()) {
        level = reasoning.at("summary").get<std::string>();
    } else if (reasoning.contains("generate_summary") &&
               reasoning.at("generate_summary").is_string()) {
        level = reasoning.at("generate_summary").get<std::string>();
    } else {
        return std::string();
    }
    std::string text = reasoning_text;
    if (level == "concise") {
        // Local concise: first sentence / hard cap — observably shorter than detailed.
        const size_t cut = std::min<size_t>(text.size(), 120);
        size_t end = cut;
        const auto dot = text.find_first_of(".!?\n", 0);
        if (dot != std::string::npos && dot + 1 < cut) {
            end = dot + 1;
        }
        if (end < text.size()) {
            text = text.substr(0, end);
            if (!text.empty() && text.back() != '.' && text.back() != '!' && text.back() != '?') {
                text += "...";
            }
        }
    } else if (level == "auto" && text.size() > 800) {
        text = text.substr(0, 800) + "...";
    }
    // detailed (and short auto/concise) keep fuller reasoning text.
    return text;
}

json server_responses_reasoning_summary(const std::string & reasoning_text, const json & request_body) {
    json summary = json::array();
    std::string text = server_responses_reasoning_summary_text(reasoning_text, request_body);
    if (!text.empty()) {
        summary.push_back(json {
            {"type", "summary_text"},
            {"text", std::move(text)},
        });
    }
    return summary;
}

bool server_responses_want_stream_obfuscation(const json & request_body) {
    // OpenAI includes the obfuscation field by default
    if (!request_body.contains("stream_options") || !request_body.at("stream_options").is_object()) {
        return true;
    }
    return json_value(request_body.at("stream_options"), "include_obfuscation", true);
}

void server_responses_maybe_obfuscate_event(json & event_data, const json & request_body) {
    if (!server_responses_want_stream_obfuscation(request_body)) {
        return;
    }
    if (!event_data.is_object()) {
        return;
    }
    event_data["obfuscation"] = random_string();
}

// One replayed SSE event block: drop the obfuscation field from its data object.
static std::string server_responses_strip_event_obfuscation(const std::string & ev) {
    if (ev.find("\"obfuscation\"") == std::string::npos) {
        return ev;
    }
    const size_t data_pos = ev.find("data: ");
    if (data_pos == std::string::npos) {
        return ev;
    }
    const size_t line_end = ev.find('\n', data_pos);
    if (line_end == std::string::npos) {
        return ev;
    }
    try {
        json obj = json::parse(ev.substr(data_pos + 6, line_end - data_pos - 6));
        if (!obj.is_object() || !obj.contains("obfuscation")) {
            return ev;
        }
        obj.erase("obfuscation");
        return ev.substr(0, data_pos) + "data: " + obj.dump_safe() + ev.substr(line_end);
    } catch (const std::exception &) {
        return ev;
    }
}

std::function<bool(std::string &)> server_responses_strip_obfuscation_from_stream(
        std::function<bool(std::string &)> next) {
    auto pending = std::make_shared<std::string>();
    return [next = std::move(next), pending](std::string & out) -> bool {
        for (;;) {
            std::string chunk;
            const bool has_next = next(chunk);
            pending->append(chunk);
            size_t pos;
            while ((pos = pending->find("\n\n")) != std::string::npos) {
                out += server_responses_strip_event_obfuscation(pending->substr(0, pos + 2));
                pending->erase(0, pos + 2);
            }
            if (!has_next) {
                out += *pending; // a trailing partial block, if any
                pending->clear();
                return false;
            }
            if (!out.empty()) {
                return true;
            }
        }
    };
}
