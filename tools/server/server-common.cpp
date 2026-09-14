#include "common.h"
#include "download.h"
#include "log.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "chat.h"
#include "base64.hpp"

#include "server-common.h"
#include "server-web-search.h"
#include "server-openai-persist.h"

#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <limits>
#include <cstring>
#include <type_traits>
#include <exception>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <chrono>
#include <thread>

#ifdef _WIN32
// windows.h defines min and max as macros, which breaks std::min and std::max
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#include <io.h>
#else
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <unistd.h>
#endif

json format_error_response(const std::string & message, const enum error_type type, const std::string & param, const std::string & code) {
    std::string type_str;
    switch (type) {
        case ERROR_TYPE_INVALID_REQUEST:
            type_str = "invalid_request_error";
            break;
        case ERROR_TYPE_AUTHENTICATION:
            // official 401 body type is invalid_request_error (probed)
            type_str = "invalid_request_error";
            break;
        case ERROR_TYPE_NOT_FOUND:
            type_str = "not_found_error";
            break;
        case ERROR_TYPE_SERVER:
            type_str = "server_error";
            break;
        case ERROR_TYPE_PERMISSION:
            type_str = "permission_error";
            break;
        case ERROR_TYPE_NOT_SUPPORTED:
            type_str = "not_supported_error";
            break;
        case ERROR_TYPE_UNAVAILABLE:
            // official 503 body type is service_unavailable_error (api reference)
            type_str = "service_unavailable_error";
            break;
        case ERROR_TYPE_EXCEED_CONTEXT_SIZE:
            type_str = "exceed_context_size_error";
            break;
    }
    // empty param/code become null, as the official error shape wants
    return json {
        {"code",    code.empty()  ? json(nullptr) : json(code)},
        {"message", message},
        {"param",   param.empty() ? json(nullptr) : json(param)},
        {"type",    type_str},
    };
}

// HTTP status code for an error body, from its "type"; keep in sync with format_error_response
int error_status_from_body(const json & error_data, int fallback) {
    static const std::unordered_map<std::string, int> status_by_type = {
        {"invalid_request_error",     400},
        {"authentication_error",      401},
        {"not_found_error",           404},
        {"server_error",              500},
        {"permission_error",          403},
        {"not_supported_error",       501},
        {"service_unavailable_error", 503},
        {"exceed_context_size_error", 400},
    };
    const auto it = status_by_type.find(json_value(error_data, "type", std::string()));
    return it == status_by_type.end() ? fallback : it->second;
}

json format_oai_model_not_found(const std::string & model_name) {
    return json {{"error", format_error_response(
        string_format("The model `%s` does not exist or you do not have access to it.", model_name.c_str()),
        ERROR_TYPE_INVALID_REQUEST, "", "model_not_found")}};
}

bool server_openai_is_reasoning_effort(const std::string & effort) {
    static const std::unordered_set<std::string> k_efforts = {
        "none", "minimal", "low", "medium", "high", "xhigh", "max",
    };
    return k_efforts.find(effort) != k_efforts.end();
}

void server_openai_validate_reasoning_effort_field(const json & value, const char * field_name) {
    if (value.is_null()) {
        return;
    }
    if (!value.is_string()) {
        throw std::invalid_argument(std::string("'") + field_name + "' must be a string");
    }
    const std::string effort = value.get<std::string>();
    if (!server_openai_is_reasoning_effort(effort)) {
        throw std::invalid_argument(
            std::string("'") + field_name +
            "' must be one of: none, minimal, low, medium, high, xhigh, max");
    }
}

void server_openai_validate_reasoning_object(const json & body) {
    if (!body.contains("reasoning") || body.at("reasoning").is_null()) {
        return;
    }
    if (!body.at("reasoning").is_object()) {
        throw std::invalid_argument("'reasoning' must be an object");
    }
    const json & reasoning = body.at("reasoning");

    if (reasoning.contains("effort")) {
        server_openai_validate_reasoning_effort_field(reasoning.at("effort"), "reasoning.effort");
    }

    if (reasoning.contains("context") && !reasoning.at("context").is_null()) {
        if (!reasoning.at("context").is_string()) {
            throw std::invalid_argument(
                "'reasoning.context' must be 'auto', 'current_turn', or 'all_turns'");
        }
        const std::string ctx = reasoning.at("context").get<std::string>();
        if (ctx != "auto" && ctx != "current_turn" && ctx != "all_turns") {
            throw std::invalid_argument(
                "'reasoning.context' must be 'auto', 'current_turn', or 'all_turns'");
        }
    }

    auto validate_summary_like = [&](const char * field_name) {
        if (!reasoning.contains(field_name) || reasoning.at(field_name).is_null()) {
            return;
        }
        if (!reasoning.at(field_name).is_string()) {
            throw std::invalid_argument(
                std::string("'reasoning.") + field_name +
                "' must be 'auto', 'concise', or 'detailed'");
        }
        const std::string v = reasoning.at(field_name).get<std::string>();
        if (v != "auto" && v != "concise" && v != "detailed") {
            throw std::invalid_argument(
                std::string("'reasoning.") + field_name +
                "' must be 'auto', 'concise', or 'detailed'");
        }
    };
    validate_summary_like("summary");
    validate_summary_like("generate_summary");

    // OpenAI "mode" string; documented values include standard|pro. pro boosts thinking.
    if (reasoning.contains("mode") && !reasoning.at("mode").is_null()) {
        if (!reasoning.at("mode").is_string()) {
            throw std::invalid_argument("'reasoning.mode' must be a string");
        }
    }
}

// Validate Responses `text` object including optional verbosity.
static void server_openai_validate_responses_text_object(const json & body) {
    if (!body.contains("text") || body.at("text").is_null()) {
        return;
    }
    if (!body.at("text").is_object()) {
        throw std::invalid_argument("'text' must be an object");
    }
    const json & text = body.at("text");
    if (text.contains("verbosity") && !text.at("verbosity").is_null()) {
        if (!text.at("verbosity").is_string()) {
            throw std::invalid_argument("'text.verbosity' must be 'low', 'medium', or 'high'");
        }
        const std::string v = text.at("verbosity").get<std::string>();
        if (v != "low" && v != "medium" && v != "high") {
            throw std::invalid_argument("'text.verbosity' must be 'low', 'medium', or 'high'");
        }
    }
    if (text.contains("format") && !text.at("format").is_null()) {
        if (!text.at("format").is_object()) {
            throw std::invalid_argument("'text.format' must be an object");
        }
        const json & fmt = text.at("format");
        if (!fmt.contains("type") || !fmt.at("type").is_string()) {
            throw std::invalid_argument("'text.format.type' is required");
        }
        const std::string ftype = fmt.at("type").get<std::string>();
        if (ftype != "text" && ftype != "json_object" && ftype != "json_schema") {
            throw std::invalid_argument(
                "'text.format.type' must be one of: text, json_object, json_schema");
        }
        if (ftype == "json_schema") {
            if (!fmt.contains("schema") || !fmt.at("schema").is_object()) {
                throw std::invalid_argument("'text.format.schema' is required for type=json_schema");
            }
        }
    }
}

// Validate the shared Chat/Responses `service_tier` enum. Official Responses compact
// documents a narrower enum than the create/chat paths.
static void server_openai_validate_service_tier(const json & body, bool compact = false) {
    if (!body.contains("service_tier") || body.at("service_tier").is_null()) {
        return;
    }
    const char * msg = compact
        ? "'service_tier' must be one of: auto, default, flex, fast, priority"
        : "'service_tier' must be one of: auto, default, flex, scale, priority, fast, ultrafast";
    if (!body.at("service_tier").is_string()) {
        throw std::invalid_argument(msg);
    }
    const std::string tier = body.at("service_tier").get<std::string>();
    if (tier == "auto" || tier == "default" || tier == "flex" || tier == "priority" || tier == "fast") {
        return;
    }
    if (!compact && (tier == "scale" || tier == "ultrafast")) {
        return;
    }
    throw std::invalid_argument(msg);
}

// official compact enum: no scale/ultrafast
void server_openai_validate_compact_service_tier(const json & body) {
    server_openai_validate_service_tier(body, /*compact=*/true);
}

// Validate the official Metadata object: <=16 pairs, keys <=64 chars, string values <=512 chars.
void server_openai_validate_metadata(const json & metadata) {
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

void server_openai_validate_cloud_shaped_fields(const json & body, bool allow_prompt) {
    if (allow_prompt && body.contains("prompt") && !body.at("prompt").is_null()) {
        if (!body.at("prompt").is_object()) {
            throw std::invalid_argument("'prompt' must be an object with string 'id'");
        }
        const json & prompt = body.at("prompt");
        if (!prompt.contains("id") || !prompt.at("id").is_string() ||
                prompt.at("id").get<std::string>().empty()) {
            throw std::invalid_argument("'prompt.id' is required and must be a non-empty string");
        }
        if (prompt.contains("variables") && !prompt.at("variables").is_null() &&
                !prompt.at("variables").is_object()) {
            throw std::invalid_argument("'prompt.variables' must be an object");
        }
        if (prompt.contains("version") && !prompt.at("version").is_null() &&
                !prompt.at("version").is_string()) {
            throw std::invalid_argument("'prompt.version' must be a string");
        }
    }
    if (body.contains("prompt_cache_key") && !body.at("prompt_cache_key").is_null()) {
        if (!body.at("prompt_cache_key").is_string()) {
            throw std::invalid_argument("'prompt_cache_key' must be a string");
        }
    }
    if (body.contains("prompt_cache_retention") && !body.at("prompt_cache_retention").is_null()) {
        if (!body.at("prompt_cache_retention").is_string()) {
            throw std::invalid_argument(
                "'prompt_cache_retention' must be 'in_memory' or '24h'");
        }
        const std::string ret = body.at("prompt_cache_retention").get<std::string>();
        if (ret != "in_memory" && ret != "24h") {
            throw std::invalid_argument(
                "'prompt_cache_retention' must be 'in_memory' or '24h'");
        }
    }
    if (body.contains("prompt_cache_options") && !body.at("prompt_cache_options").is_null()) {
        if (!body.at("prompt_cache_options").is_object()) {
            throw std::invalid_argument("'prompt_cache_options' must be an object");
        }
        const json & opts = body.at("prompt_cache_options");
        if (opts.contains("mode") && !opts.at("mode").is_null()) {
            if (!opts.at("mode").is_string()) {
                throw std::invalid_argument(
                    "'prompt_cache_options.mode' must be 'implicit' or 'explicit'");
            }
            const std::string mode = opts.at("mode").get<std::string>();
            if (mode != "implicit" && mode != "explicit") {
                throw std::invalid_argument(
                    "'prompt_cache_options.mode' must be 'implicit' or 'explicit'");
            }
        }
        if (opts.contains("ttl") && !opts.at("ttl").is_null()) {
            if (!opts.at("ttl").is_string()) {
                throw std::invalid_argument("'prompt_cache_options.ttl' must be a string");
            }
            // Official: 30m is currently the only supported value.
            const std::string ttl = opts.at("ttl").get<std::string>();
            if (ttl != "30m") {
                throw std::invalid_argument("'prompt_cache_options.ttl' must be '30m'");
            }
        }
        // comparison_response_id is a Responses diagnostics hint; an unknown id is not an
        // error (the response reports comparison_response_not_found), so only check the shape.
        if (opts.contains("comparison_response_id") && !opts.at("comparison_response_id").is_null()) {
            if (!opts.at("comparison_response_id").is_string()) {
                throw std::invalid_argument(
                    "'prompt_cache_options.comparison_response_id' must be a string");
            }
        }
        // explicit mode requires a prompt_cache_key (local affinity key).
        if (opts.contains("mode") && opts.at("mode").is_string() &&
                opts.at("mode").get<std::string>() == "explicit") {
            if (!body.contains("prompt_cache_key") || body.at("prompt_cache_key").is_null() ||
                    !body.at("prompt_cache_key").is_string() ||
                    body.at("prompt_cache_key").get<std::string>().empty()) {
                throw std::invalid_argument(
                    "'prompt_cache_key' is required when prompt_cache_options.mode is 'explicit'");
            }
        }
    }
    if (allow_prompt) {
        // Responses-only fields: the chatcmpl conversion strips these before the
        // chat path could validate them, so check the official shape here.
        if (body.contains("metadata") && !body.at("metadata").is_null()) {
            server_openai_validate_metadata(body.at("metadata"));
        }
        if (body.contains("safety_identifier") && !body.at("safety_identifier").is_null()) {
            if (!body.at("safety_identifier").is_string()) {
                throw std::invalid_argument("'safety_identifier' must be a string");
            }
            if (body.at("safety_identifier").get<std::string>().size() > 64) {
                throw std::invalid_argument("'safety_identifier' must be at most 64 characters");
            }
        }
        server_openai_validate_service_tier(body);
        if (body.contains("include") && !body.at("include").is_null()) {
            if (!body.at("include").is_array()) {
                throw std::invalid_argument("'include' must be an array");
            }
            for (const auto & item : body.at("include")) {
                if (!item.is_string()) {
                    throw std::invalid_argument("'include' entries must be strings");
                }
                const std::string inc = item.get<std::string>();
                if (inc != "web_search_call.action.sources" &&
                        inc != "code_interpreter_call.outputs" &&
                        inc != "computer_call_output.output.image_url" &&
                        inc != "file_search_call.results" &&
                        inc != "message.input_image.image_url" &&
                        inc != "message.output_text.logprobs" &&
                        inc != "reasoning.encrypted_content" &&
                        inc != "web_search_call.results") {
                    throw std::invalid_argument("Unknown 'include' value: " + inc);
                }
            }
        }
        server_openai_validate_responses_text_object(body);
    }
}

std::string server_openai_short_hash(const std::string & s) {
    // FNV-1a 64-bit → 16 hex chars (stable local cache key, not crypto).
    uint64_t h = 14695981039346656037ull;
    for (unsigned char c : s) {
        h ^= (uint64_t) c;
        h *= 1099511628211ull;
    }
    char buf[17];
    std::snprintf(buf, sizeof(buf), "%016llx", (unsigned long long) h);
    return std::string(buf);
}

bool server_oai_prompt_cache_keys_compatible(const std::string & a, const std::string & b) {
    // Exact match only: a slot warmed for one prefix must not be reused for another.
    return a == b;
}

std::string server_prompt_cache_anchor_key(const server_tokens & tokens) {
    // enough tokens to tell conversations apart, short enough to survive prompt edits
    constexpr size_t n_anchor_max = 64;

    const size_t n_anchor = std::min(n_anchor_max, tokens.size());

    std::string material;
    material.reserve(n_anchor * 8);
    for (size_t i = 0; i < n_anchor; ++i) {
        material += std::to_string(tokens[i]);
        material += ',';
    }

    return "anch-" + server_openai_short_hash(material);
}

void server_openai_apply_prompt_cache_semantics(json & body) {
    const bool has_key = body.contains("prompt_cache_key") && body.at("prompt_cache_key").is_string() &&
                         !body.at("prompt_cache_key").get<std::string>().empty();
    const bool has_ret = body.contains("prompt_cache_retention") && !body.at("prompt_cache_retention").is_null();
    const bool has_opts = body.contains("prompt_cache_options") && body.at("prompt_cache_options").is_object();
    // Internal TTL channel: the Anthropic layer writes cache_control.ttl here, so 5m/1h stay
    // usable without widening the official prompt_cache_options.ttl enum.
    const bool has_local_ttl = body.contains("__prompt_cache_ttl") && body.at("__prompt_cache_ttl").is_string();
    if (!has_key && !has_ret && !has_opts && !has_local_ttl) {
        return;
    }

    body["cache_prompt"] = true;

    std::string key = has_key ? body.at("prompt_cache_key").get<std::string>() : std::string();
    if (key.empty()) {
        // No client key: mark the request so the server derives a local anchor from the
        // prompt tokens. Do NOT hash the request body here - it changes on every turn of a
        // growing conversation, and a rotating key would invalidate the KV state (and the
        // disk registry entry) right after it was restored.
        body["__oai_prompt_cache_implicit"] = true;
    }

    int32_t ttl = 0; // 0 = process lifetime (in_memory)
    if (has_ret && body.at("prompt_cache_retention").is_string()) {
        const std::string ret = body.at("prompt_cache_retention").get<std::string>();
        if (ret == "24h") {
            ttl = 24 * 3600;
        }
    }
    if (has_opts && body.at("prompt_cache_options").contains("ttl") &&
            body.at("prompt_cache_options").at("ttl").is_string()) {
        // Options TTL is more specific when present; official supports only 30m.
        const std::string opt_ttl = body.at("prompt_cache_options").at("ttl").get<std::string>();
        if (opt_ttl != "30m") {
            throw std::invalid_argument("'prompt_cache_options.ttl' must be '30m'");
        }
        ttl = 30 * 60;
    }
    if (has_local_ttl) {
        // Internal channel (Anthropic cache_control.ttl). It feeds the same local cache TTL
        // as prompt_cache_options.ttl, but keeps the local 5m/1h values off the wire.
        const std::string local_ttl = body.at("__prompt_cache_ttl").get<std::string>();
        if (local_ttl == "5m") {
            ttl = 5 * 60;
        } else if (local_ttl == "30m") {
            ttl = 30 * 60;
        } else if (local_ttl == "1h") {
            ttl = 60 * 60;
        } else {
            throw std::invalid_argument("'__prompt_cache_ttl' must be one of: 5m, 30m, 1h");
        }
    }

    if (!key.empty()) {
        // Check disk TTL *before* touch — otherwise every request refreshes expires_at
        // and server_prompt_cache_key_alive() can never observe expiry mid-process.
        const bool alive_before = server_prompt_cache_key_alive(key);
        if (!alive_before) {
            body["__oai_prompt_cache_expired"] = true;
        }
        body["__oai_prompt_cache_key"] = key;
        body["__oai_prompt_cache_key_explicit"] = true;
        server_prompt_cache_key_touch(key, ttl);
    }
    body["__oai_prompt_cache_ttl"] = ttl;
}

void server_prompt_cache_key_touch(const std::string & key, int32_t ttl_seconds) {
    if (key.empty()) {
        return;
    }
    std::string root = openai_persist::root();
    if (root.empty()) {
        return;
    }
    while (!root.empty() && (root.back() == '/' || root.back() == '\\')) {
        root.pop_back();
    }
    const std::string dir = root + "/prompt_cache_keys";
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    // Filename-safe (no '.' → cannot form ".." segments). Reuse persist helper.
    std::string safe = openai_persist::safe_id(key);
    if (safe.size() > 120) {
        safe = safe.substr(0, 100) + "_" + server_openai_short_hash(key);
    }
    const int64_t now = (int64_t) std::time(nullptr);
    json rec = {
        {"key", key},
        {"updated_at", now},
        {"expires_at", ttl_seconds > 0 ? (now + ttl_seconds) : 0},
        {"ttl_seconds", ttl_seconds},
    };
    std::ofstream out(dir + "/" + safe + ".json");
    if (out) {
        out << rec.dump();
    }
}

static std::string server_prompt_cache_key_filename(const std::string & key) {
    std::string safe = openai_persist::safe_id(key);
    if (safe.size() > 120) {
        safe = safe.substr(0, 100) + "_" + server_openai_short_hash(key);
    }
    return safe;
}

bool server_prompt_cache_key_alive(const std::string & key) {
    if (key.empty()) {
        return true;
    }
    std::string root = openai_persist::root();
    if (root.empty()) {
        return true; // memory-only: treat as alive for process lifetime
    }
    while (!root.empty() && (root.back() == '/' || root.back() == '\\')) {
        root.pop_back();
    }
    const std::string dir = root + "/prompt_cache_keys";
    const std::string safe = server_prompt_cache_key_filename(key);
    std::ifstream in(dir + "/" + safe + ".json");
    if (!in) {
        // Older underscore-subst filenames (pre hex encoding).
        const std::string legacy = openai_persist::safe_id_legacy(key);
        if (legacy != safe) {
            std::string leg = legacy;
            if (leg.size() > 120) {
                leg = leg.substr(0, 100) + "_" + server_openai_short_hash(key);
            }
            in.open(dir + "/" + leg + ".json");
        }
    }
    if (!in) {
        // No durable record yet — allow (first request will touch).
        return true;
    }
    try {
        json rec;
        std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        rec = json::parse(content);
        const int64_t exp = json_value(rec, "expires_at", (int64_t) 0);
        if (exp <= 0) {
            return true;
        }
        return exp > (int64_t) std::time(nullptr);
    } catch (...) {
        return true;
    }
}

bool server_is_local_web_search_tool_type(const std::string & type) {
    return server_web_search_is_tool_type(type);
}

void server_openai_apply_web_search_semantics(json & body) {
    server_web_search_apply(body);
}

double server_oaicompat_probs_score(const std::vector<float> & token_probs) {
    if (token_probs.empty()) {
        return -1e300;
    }
    double s = 0.0;
    for (float p : token_probs) {
        s += std::log(std::max((double) p, 1e-12));
    }
    return s;
}

json server_openai_completions_apply_suffix(
        const llama_vocab * vocab,
        json body,
        int n_batch,
        int n_predict,
        int n_ctx,
        bool spm_infill) {
    if (!body.contains("suffix") || body.at("suffix").is_null() || !body.at("suffix").is_string()) {
        return body;
    }
    const std::string suffix = body.at("suffix").get<std::string>();
    if (suffix.empty()) {
        return body;
    }
    if (!body.contains("prompt")) {
        throw std::invalid_argument("'prompt' is required when 'suffix' is set");
    }

    const bool has_fim =
        vocab != nullptr &&
        llama_vocab_fim_pre(vocab) != LLAMA_TOKEN_NULL &&
        llama_vocab_fim_suf(vocab) != LLAMA_TOKEN_NULL &&
        llama_vocab_fim_mid(vocab) != LLAMA_TOKEN_NULL;

    json prompt_j = body.at("prompt");
    std::string prefix;
    if (prompt_j.is_string()) {
        prefix = prompt_j.get<std::string>();
    } else if (prompt_j.is_array() && !prompt_j.empty() && prompt_j.at(0).is_string()) {
        // Multi-prompt: FIM only the first for local Completions parity.
        prefix = prompt_j.at(0).get<std::string>();
    } else {
        throw std::invalid_argument("'suffix' requires string 'prompt' (or array of strings)");
    }

    if (has_fim) {
        const llama_tokens tokens_prompt; // empty mid prompt
        body["prompt"] = format_prompt_infill(
            vocab, prefix, suffix, json::array(), n_batch, n_predict, n_ctx, spm_infill, tokens_prompt);
    } else {
        // Soft FIM for models without dedicated FIM tokens.
        body["prompt"] =
            std::string("Fill in the missing middle text.\nPREFIX:\n") + prefix +
            "\nSUFFIX:\n" + suffix + "\nMIDDLE:\n";
    }
    // Keep suffix on body for echo/debug; generation uses rewritten prompt.
    return body;
}

void server_openai_validate_completions_create(const json & body) {
    if (!body.contains("prompt")) {
        throw std::invalid_argument("'prompt' is required");
    }
    if (body.contains("echo") && !body.at("echo").is_null()) {
        if (!body.at("echo").is_boolean()) {
            throw std::invalid_argument("'echo' must be a boolean");
        }
        // echo=true is supported: choice text includes the prompt prefix.
    }
    if (body.contains("suffix") && !body.at("suffix").is_null()) {
        if (!body.at("suffix").is_string()) {
            throw std::invalid_argument("'suffix' must be a string");
        }
        // nonempty suffix → FIM (applied in completions route before generation).
    }
    int n_val = 1;
    if (body.contains("n") && !body.at("n").is_null()) {
        if (!body.at("n").is_number_integer()) {
            throw std::invalid_argument("'n' must be an integer");
        }
        n_val = body.at("n").get<int>();
        if (n_val < 1) {
            throw std::invalid_argument("'n' must be >= 1");
        }
    }
    if (body.contains("best_of") && !body.at("best_of").is_null()) {
        if (!body.at("best_of").is_number_integer()) {
            throw std::invalid_argument("'best_of' must be an integer");
        }
        const int best_of = body.at("best_of").get<int>();
        if (best_of < 1) {
            throw std::invalid_argument("'best_of' must be >= 1");
        }
        // Official rule: best_of >= n. Local ranks candidates by token logprob sum.
        if (best_of < n_val) {
            throw std::invalid_argument("'best_of' must be greater than or equal to 'n'");
        }
        // OpenAI: best_of is not compatible with stream (ranking needs all candidates).
        const bool stream = body.contains("stream") && body.at("stream").is_boolean() &&
                            body.at("stream").get<bool>();
        if (stream && best_of > n_val) {
            throw std::invalid_argument("'best_of' > 'n' is not supported with stream=true");
        }
    }
    if (body.contains("user") && !body.at("user").is_null() && !body.at("user").is_string()) {
        throw std::invalid_argument("'user' must be a string");
    }
    if (body.contains("stream_options") && !body.at("stream_options").is_null()) {
        if (!body.at("stream_options").is_object()) {
            throw std::invalid_argument("'stream_options' must be an object");
        }
    }
    auto check_penalty = [](const json & body, const char * key) {
        if (!body.contains(key) || body.at(key).is_null()) {
            return;
        }
        if (!body.at(key).is_number()) {
            throw std::invalid_argument(std::string("'") + key + "' must be a number");
        }
        const double v = body.at(key).get<double>();
        if (v < -2.0 || v > 2.0) {
            throw std::invalid_argument(std::string("'") + key + "' must be in [-2, 2]");
        }
    };
    check_penalty(body, "frequency_penalty");
    check_penalty(body, "presence_penalty");
    if (body.contains("logit_bias") && !body.at("logit_bias").is_null() &&
            !body.at("logit_bias").is_object() && !body.at("logit_bias").is_array()) {
        throw std::invalid_argument("'logit_bias' must be an object or array");
    }
    if (body.contains("seed") && !body.at("seed").is_null() && !body.at("seed").is_number_integer()) {
        throw std::invalid_argument("'seed' must be an integer");
    }
    // Official Completions: logprobs is a non-negative integer (null/omitted = disabled);
    // values above 5 are clamped to 5 by the official API instead of being rejected.
    if (body.contains("logprobs") && !body.at("logprobs").is_null()) {
        if (!body.at("logprobs").is_number_integer() || body.at("logprobs").get<int>() < 0) {
            throw std::invalid_argument("'logprobs' must be a non-negative integer");
        }
    }
    // Official Completions: stop is a string or an array of up to 4 strings.
    if (body.contains("stop") && !body.at("stop").is_null()) {
        if (!body.at("stop").is_string() && !body.at("stop").is_array()) {
            throw std::invalid_argument("'stop' must be a string or an array of strings");
        }
        if (body.at("stop").is_array()) {
            if (body.at("stop").size() > 4) {
                throw std::invalid_argument("'stop' must contain at most 4 sequences");
            }
            for (const auto & item : body.at("stop")) {
                if (!item.is_string()) {
                    throw std::invalid_argument("'stop' must be a string or an array of strings");
                }
            }
        }
    }
}

void server_openai_validate_chat_create_fields(const json & body) {
    if (body.contains("verbosity") && !body.at("verbosity").is_null()) {
        if (!body.at("verbosity").is_string()) {
            throw std::invalid_argument("'verbosity' must be 'low', 'medium', or 'high'");
        }
        const std::string v = body.at("verbosity").get<std::string>();
        if (v != "low" && v != "medium" && v != "high") {
            throw std::invalid_argument("'verbosity' must be 'low', 'medium', or 'high'");
        }
    }
    bool want_audio_out = false;
    if (body.contains("modalities") && !body.at("modalities").is_null()) {
        if (!body.at("modalities").is_array()) {
            throw std::invalid_argument("'modalities' must be an array of 'text' and/or 'audio'");
        }
        for (const auto & m : body.at("modalities")) {
            if (!m.is_string()) {
                throw std::invalid_argument("'modalities' entries must be strings");
            }
            const std::string ms = m.get<std::string>();
            if (ms != "text" && ms != "audio") {
                throw std::invalid_argument("'modalities' entries must be 'text' or 'audio'");
            }
            if (ms == "audio") {
                want_audio_out = true;
            }
        }
    }
    if (body.contains("audio") && !body.at("audio").is_null()) {
        want_audio_out = true;
        if (!body.at("audio").is_object()) {
            throw std::invalid_argument("'audio' must be an object");
        }
    }
    if (want_audio_out) {
        throw std::invalid_argument(
            "audio output (modalities/audio) is not supported on this server");
    }
    if (body.contains("prediction") && !body.at("prediction").is_null()) {
        if (!body.at("prediction").is_object()) {
            throw std::invalid_argument("'prediction' must be an object");
        }
        const json & pred = body.at("prediction");
        if (!pred.contains("type") || !pred.at("type").is_string() ||
                pred.at("type").get<std::string>() != "content") {
            throw std::invalid_argument("'prediction.type' must be 'content'");
        }
        if (!pred.contains("content") ||
                !(pred.at("content").is_string() || pred.at("content").is_array())) {
            throw std::invalid_argument("'prediction.content' must be a string or array");
        }
    }
    auto check_penalty = [](const json & body, const char * key) {
        if (!body.contains(key) || body.at(key).is_null()) {
            return;
        }
        if (!body.at(key).is_number()) {
            throw std::invalid_argument(std::string("'") + key + "' must be a number");
        }
        const double v = body.at(key).get<double>();
        if (v < -2.0 || v > 2.0) {
            throw std::invalid_argument(std::string("'") + key + "' must be in [-2, 2]");
        }
    };
    check_penalty(body, "frequency_penalty");
    check_penalty(body, "presence_penalty");
    if (body.contains("n") && !body.at("n").is_null()) {
        if (!body.at("n").is_number_integer()) {
            throw std::invalid_argument("'n' must be an integer");
        }
        if (body.at("n").get<int>() < 1) {
            throw std::invalid_argument("'n' must be >= 1");
        }
    }
    if (body.contains("metadata") && !body.at("metadata").is_null()) {
        server_openai_validate_metadata(body.at("metadata"));
    }
    if (body.contains("user") && !body.at("user").is_null() && !body.at("user").is_string()) {
        throw std::invalid_argument("'user' must be a string");
    }
    if (body.contains("safety_identifier") && !body.at("safety_identifier").is_null() &&
            !body.at("safety_identifier").is_string()) {
        throw std::invalid_argument("'safety_identifier' must be a string");
    }
    server_openai_validate_service_tier(body);
    if (body.contains("logit_bias") && !body.at("logit_bias").is_null() &&
            !body.at("logit_bias").is_object() && !body.at("logit_bias").is_array()) {
        throw std::invalid_argument("'logit_bias' must be an object or array");
    }
    if (body.contains("stream_options") && !body.at("stream_options").is_null()) {
        if (!body.at("stream_options").is_object()) {
            throw std::invalid_argument("'stream_options' must be an object");
        }
    }
    if (body.contains("reasoning_effort")) {
        server_openai_validate_reasoning_effort_field(body.at("reasoning_effort"), "reasoning_effort");
    }
    // Official Chat Completions: stop is a string or an array of up to 4 strings.
    if (body.contains("stop") && !body.at("stop").is_null()) {
        if (!body.at("stop").is_string() && !body.at("stop").is_array()) {
            throw std::invalid_argument("'stop' must be a string or an array of strings");
        }
        if (body.at("stop").is_array()) {
            if (body.at("stop").size() > 4) {
                throw std::invalid_argument("'stop' must contain at most 4 sequences");
            }
            for (const auto & item : body.at("stop")) {
                if (!item.is_string()) {
                    throw std::invalid_argument("'stop' must be a string or an array of strings");
                }
            }
        }
    }
}

//
// server_slot_stats
//

json server_slot_stats::to_json() const {
    json base = {
        {"cache_n",                n_prompt_cached},

        {"prompt_n",               n_prompt_processed},
        {"prompt_ms",              t_prompt_ms()},
        {"prompt_per_token_ms",    t_prompt_per_token_ms()},
        {"prompt_per_second",      n_prompt_tps()},

        {"predicted_n",            n_gen},
        {"predicted_ms",           t_gen_ms()},
        {"predicted_per_token_ms", t_gen_per_token_ms()},
        {"predicted_per_second",   n_gen_tps()},
    };

    if (n_draft_tokens > 0) {
        base["draft_n"]          = n_draft_tokens;
        base["draft_n_accepted"] = n_draft_accepted;
    }

    return base;
}

//
// random string / id
//

std::string random_string() {
    static const std::string str("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz");

    std::random_device rd;
    std::mt19937 generator(rd());

    std::string result(32, ' ');

    for (int i = 0; i < 32; ++i) {
        result[i] = str[generator() % str.size()];
    }

    return result;
}

std::string gen_chatcmplid() {
    return "chatcmpl-" + random_string();
}

std::string gen_tool_call_id() {
    return random_string();
}

const char * get_media_marker() {
    static const std::string marker = []() {
        // allow user to pin a reproducible marker via env var
        const char * env = getenv("LLAMA_MEDIA_MARKER");
        if (env && env[0] != '\0') {
            return std::string(env);
        }
        return std::string("<__media_") + random_string() + "__>";
    }();
    return marker.c_str();
}

//
// lora utils
//

bool lora_all_alora(const std::vector<common_adapter_lora_info> & loras) {
    bool found_alora = false;
    for (const auto & lora : loras) {
        if (lora.scale != 0) {
            if (llama_adapter_get_alora_n_invocation_tokens(lora.ptr) == 0) {
                return false;
            }
            found_alora = true;
        }
    }
    return found_alora;
}

bool lora_should_clear_cache(
        const std::vector<common_adapter_lora_info> & current,
        const std::vector<common_adapter_lora_info> & next) {

    // This should always be called after determining that the two sets are
    // _not_ equal. This assert is therefore some slightly wasted work and
    // should be safe to remove as long as this method is called correctly.
    GGML_ASSERT(!are_lora_equal(current, next));

    return (
        !(lora_get_enabled_ids(current).empty() || lora_all_alora(current)) ||
        !lora_all_alora(next));
}

std::map<int, float> parse_lora_request(const json & data) {
    std::map<int, float> lora;

    // set value
    for (const auto & entry : data) {
        int id      = json_value(entry, "id", -1);
        float scale = json_value(entry, "scale", 0.0f);
        lora[id] = scale;
    }

    return lora;
}

bool are_lora_equal(
        const std::vector<common_adapter_lora_info> & l1,
        const std::vector<common_adapter_lora_info> & l2) {
    if (l1.size() != l2.size()) {
        return false;
    }
    for (size_t i = 0; i < l1.size(); ++i) {
        // we don't check lora.path to reduce the time complexity
        if (l1[i].scale != l2[i].scale || l1[i].ptr != l2[i].ptr) {
            return false;
        }
    }
    return true;
}

std::vector<size_t> lora_get_enabled_ids(const std::vector<common_adapter_lora_info> & loras) {
    std::vector<size_t> enabled_ids;
    for (size_t i = 0; i < loras.size(); ++i) {
        if (loras[i].scale > 0) {
            enabled_ids.push_back(i);
        }
    }
    return enabled_ids;
}

//
// base64 utils (TODO: use the base64::decode from base64.hpp)
//

static const std::string base64_chars =
             "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
             "abcdefghijklmnopqrstuvwxyz"
             "0123456789+/";

static inline bool is_base64(uint8_t c) {
    return (isalnum(c) || (c == '+') || (c == '/'));
}

static inline raw_buffer base64_decode(const std::string & encoded_string) {
    int i = 0;
    int j = 0;
    int in_ = 0;

    int in_len = encoded_string.size();

    uint8_t char_array_4[4];
    uint8_t char_array_3[3];

    raw_buffer ret;

    while (in_len-- && (encoded_string[in_] != '=') && is_base64(encoded_string[in_])) {
        char_array_4[i++] = encoded_string[in_]; in_++;
        if (i == 4) {
            for (i = 0; i < 4; i++) {
                char_array_4[i] = base64_chars.find(char_array_4[i]);
            }

            char_array_3[0] = ((char_array_4[0]      ) << 2) + ((char_array_4[1] & 0x30) >> 4);
            char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
            char_array_3[2] = ((char_array_4[2] & 0x3) << 6) +   char_array_4[3];

            for (i = 0; (i < 3); i++) {
                ret.push_back(char_array_3[i]);
            }

            i = 0;
        }
    }

    if (i) {
        for (j = i; j < 4; j++) {
            char_array_4[j] = 0;
        }

        for (j = 0; j < 4; j++) {
            char_array_4[j] = base64_chars.find(char_array_4[j]);
        }

        char_array_3[0] = ((char_array_4[0]      ) << 2) + ((char_array_4[1] & 0x30) >> 4);
        char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
        char_array_3[2] = ((char_array_4[2] & 0x3) << 6) +   char_array_4[3];

        for (j = 0; j < i - 1; j++) {
            ret.push_back(char_array_3[j]);
        }
    }

    return ret;
}

//
// server_tokens implementation
//

namespace {

constexpr uint32_t SERVER_TOKENS_STATE_VERSION = 1;

uint32_t server_tokens_state_u32(size_t value) {
    if (value > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("Server tokens state is too large");
    }
    return value;
}

class server_tokens_state_writer {
public:
    template <typename T>
    void write(T value) {
        static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable");
        const auto * ptr = reinterpret_cast<const char *>(&value);
        data.insert(data.end(), ptr, ptr + sizeof(value));
    }

    template <typename T>
    void write(const std::vector<T> & values) {
        static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable");
        write(server_tokens_state_u32(values.size()));
        if (values.empty()) {
            return;
        }
        const auto * ptr = reinterpret_cast<const char *>(values.data());
        data.insert(data.end(), ptr, ptr + values.size() * sizeof(T));
    }

    void write_media_chunk(const mtmd_input_chunk * chunk) {
        size_t chunk_size = 0;
        if (mtmd_input_chunk_save(chunk, nullptr, 0, &chunk_size) != 0 || chunk_size == 0) {
            throw std::runtime_error("Cannot serialize media chunk in server tokens");
        }
        std::vector<char> chunk_data(server_tokens_state_u32(chunk_size));
        if (mtmd_input_chunk_save(chunk, chunk_data.data(), chunk_data.size(), nullptr) != 0) {
            throw std::runtime_error("Cannot serialize media chunk in server tokens");
        }
        write(chunk_data);
    }

    std::vector<char> take() {
        data.resize((data.size() + sizeof(llama_token) - 1) / sizeof(llama_token) * sizeof(llama_token), 0);
        return std::move(data);
    }

private:
    std::vector<char> data;
};

class server_tokens_state_reader {
public:
    server_tokens_state_reader(const char * data, size_t size) : data(data), size(size) {}

    template <typename T>
    T read() {
        static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable");
        if (size - pos < sizeof(T)) {
            throw std::runtime_error("Unexpected end of server tokens state");
        }
        T value;
        std::memcpy(&value, data + pos, sizeof(value));
        pos += sizeof(value);
        return value;
    }

    template <typename T>
    std::vector<T> read_vector() {
        static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable");
        const uint32_t n_values = read<uint32_t>();
        // reject before resizing, so that a small corrupted payload cannot request a huge allocation
        if (n_values > remaining() / sizeof(T)) {
            throw std::runtime_error("Unexpected end of server tokens state");
        }
        std::vector<T> values(n_values);
        if (n_values > 0) {
            std::memcpy(values.data(), data + pos, values.size() * sizeof(T));
            pos += values.size() * sizeof(T);
        }
        return values;
    }

    size_t remaining() const {
        return size - pos;
    }

private:
    const char * data;
    size_t size;
    size_t pos = 0;
};

} // namespace

server_tokens::server_tokens(mtmd::input_chunks & mtmd_chunks, bool has_mtmd) : has_mtmd(has_mtmd) {
    for (size_t i = 0; i < mtmd_chunks.size(); ++i) {
        push_back(mtmd_chunks[i]);
    }
}

server_tokens::server_tokens(const llama_tokens & tokens, bool has_mtmd) : has_mtmd(has_mtmd), tokens(tokens) {
}

llama_pos server_tokens::pos_next(int64_t n_tokens) const {
    if (!has_mtmd) {
        if (n_tokens < 0) {
            return tokens.size();
        }

        return n_tokens;
    }

    if (n_tokens < 0) {
        llama_pos res = tokens.size();

        for (auto it = map_idx_to_media.begin(); it != map_idx_to_media.end(); ++it) {
            const auto & chunk = it->second;
            res += mtmd_input_chunk_get_n_pos(chunk.get()) - mtmd_input_chunk_get_n_tokens(chunk.get());
        }

        return res;
    }

    int64_t idx = 0;
    llama_pos pos = 0;

    GGML_ASSERT(n_tokens <= (int64_t)tokens.size());

    while (idx < n_tokens) {
        const auto media_it = map_idx_to_media.find(idx);
        if (media_it != map_idx_to_media.end()) {
            const auto & chunk = media_it->second;
            const llama_pos n_pos = mtmd_input_chunk_get_n_pos(chunk.get());
            const size_t n_tok = mtmd_input_chunk_get_n_tokens(chunk.get());

            pos += n_pos;
            idx += n_tok;
        } else {
            pos++;
            idx++;
        }
    }

    return pos;
}

size_t server_tokens::size_up_to_pos(llama_pos max_pos) const {
    if (!has_mtmd) {
        return std::min((size_t)max_pos, tokens.size());
    }

    size_t idx = 0;
    llama_pos pos = 0;

    while (idx < tokens.size()) {
        const auto media_it = map_idx_to_media.find(idx);
        if (media_it != map_idx_to_media.end()) {
            const auto & chunk = media_it->second;
            const llama_pos n_pos = mtmd_input_chunk_get_n_pos(chunk.get());
            const size_t n_tok = mtmd_input_chunk_get_n_tokens(chunk.get());

            pos += n_pos;
            idx += n_tok;
        } else {
            pos++;
            idx++;
        }

        if (pos >= max_pos) {
            break;
        }
    }

    return idx;
}

std::string server_tokens::str() const {
    std::ostringstream oss;
    oss << "tokens: ";
    for (size_t idx = 0; idx < tokens.size(); ++idx) {
        llama_token t = tokens[idx];
        oss << "idx:" << idx << " ";
        if (t == LLAMA_TOKEN_NULL) {
            oss << "<embd> ";
        } else {
            oss << t << " ";
        }
    }
    oss << "\n";
    oss << "image idx: ";
    for (const auto & it : map_idx_to_media) {
        oss << it.first << ", ";
    }
    return oss.str();
}

const mtmd::input_chunk_ptr & server_tokens::find_chunk(size_t idx) const {
    auto it = map_idx_to_media.find(idx);
    if (it != map_idx_to_media.end()) {
        return it->second;
    }
    throw std::runtime_error("Chunk not found");
}

std::pair<const mtmd::input_chunk_ptr *, size_t> server_tokens::find_next_media_chunk(size_t idx) const {
    auto it = map_idx_to_media.upper_bound(idx);
    if (it != map_idx_to_media.end()) {
        return { &it->second, it->first };
    }
    return { nullptr, 0 };
}

void server_tokens::push_back(llama_token tok) {
    if (tok == LLAMA_TOKEN_NULL) {
        throw std::runtime_error("Invalid token");
    }
    tokens.emplace_back(tok);
}

void server_tokens::push_back(const mtmd_input_chunk * chunk) {
    auto type = mtmd_input_chunk_get_type(chunk);
    if (type == MTMD_INPUT_CHUNK_TYPE_IMAGE || type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
        GGML_ASSERT(has_mtmd);
        const size_t n_tokens = mtmd_input_chunk_get_n_tokens(chunk);
        size_t start_idx = tokens.size();
        for (size_t i = 0; i < n_tokens; ++i) {
            tokens.emplace_back(LLAMA_TOKEN_NULL);
        }
        mtmd::input_chunk_ptr new_chunk(mtmd_input_chunk_copy(chunk));
        map_idx_to_media[start_idx] = std::move(new_chunk);
    } else if (type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
        size_t n_tokens;
        const auto * text_tokens = mtmd_input_chunk_get_tokens_text(chunk, &n_tokens);
        for (size_t i = 0; i < n_tokens; ++i) {
            push_back(text_tokens[i]);
        }
    } else {
        GGML_ABORT("Invalid chunk type");
    }
}

void server_tokens::push_back_placeholder(const mtmd_input_chunk * chunk) {
    auto type = mtmd_input_chunk_get_type(chunk);
    if (type == MTMD_INPUT_CHUNK_TYPE_IMAGE || type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
        GGML_ASSERT(has_mtmd);
        mtmd::input_chunk_ptr new_chunk(mtmd_input_chunk_get_placeholder(chunk));
        GGML_ASSERT(new_chunk != nullptr && "failed to create placeholder chunk");
        const size_t n_tokens = mtmd_input_chunk_get_n_tokens(chunk);
        size_t start_idx = tokens.size();
        for (size_t i = 0; i < n_tokens; ++i) {
            tokens.emplace_back(LLAMA_TOKEN_NULL);
        }
        map_idx_to_media[start_idx] = std::move(new_chunk);
    } else {
        push_back(chunk);
    }
}

void server_tokens::push_back(server_tokens & tokens) {
    size_t start_idx = size();
    for (size_t i = 0; i < tokens.size(); i++) {
        push_back(tokens[i]);
    }
    if (tokens.has_mtmd) {
        // Assert if we are copying MTMD chunks to a server_tokens that does not have mtmd.
        // We could also just check, but this will prevent silently dropping MTMD data.
        GGML_ASSERT(has_mtmd);
        for (auto it = tokens.map_idx_to_media.begin(); it != tokens.map_idx_to_media.end(); ) {
            auto * chunk = tokens.map_idx_to_media[it->first].get();
            mtmd::input_chunk_ptr new_chunk(mtmd_input_chunk_copy(chunk));
            map_idx_to_media[start_idx + it->first] = std::move(new_chunk);
        }
    }
}

void server_tokens::insert(const llama_tokens & inp_tokens) {
    tokens.insert(tokens.end(), inp_tokens.begin(), inp_tokens.end());
}

const llama_tokens & server_tokens::get_tokens() const {
    GGML_ASSERT(!has_mtmd);
    return tokens;
}

std::vector<char> server_tokens::serialize() const {
    static_assert(sizeof(llama_token) == sizeof(uint32_t), "unexpected llama_token size");

    server_tokens_state_writer writer;
    writer.write((llama_token) LLAMA_TOKEN_NULL);
    writer.write(SERVER_TOKENS_STATE_VERSION);
    writer.write(tokens);

    std::vector<uint32_t> media_keys;
    media_keys.reserve(map_idx_to_media.size());
    for (const auto & item : map_idx_to_media) {
        media_keys.push_back(server_tokens_state_u32(item.first));
    }
    writer.write(media_keys);

    for (const auto & item : map_idx_to_media) {
        writer.write_media_chunk(item.second.get());
    }

    return writer.take();
}

server_tokens server_tokens::deserialize(const llama_tokens & packed, bool has_mtmd) {
    static_assert(sizeof(llama_token) == sizeof(uint32_t), "unexpected llama_token size");

    if (packed.empty() || packed[0] != LLAMA_TOKEN_NULL) {
        // plain token list, as written by older versions
        return server_tokens(packed, has_mtmd);
    }

    server_tokens_state_reader reader(reinterpret_cast<const char *>(packed.data()), packed.size() * sizeof(llama_token));
    reader.read<llama_token>(); // format marker
    if (reader.read<uint32_t>() != SERVER_TOKENS_STATE_VERSION) {
        throw std::runtime_error("Unsupported server tokens state version");
    }

    const llama_tokens tokens = reader.read_vector<llama_token>();

    // the media start indices, followed by the media chunks in the same order
    const std::vector<uint32_t> media_keys = reader.read_vector<uint32_t>();
    if (!media_keys.empty() && !has_mtmd) {
        throw std::runtime_error("Cannot restore media tokens without an mmproj");
    }

    server_tokens result(tokens, has_mtmd);

    for (const uint32_t key : media_keys) {
        const size_t start_idx = key;
        const std::vector<char> chunk_data = reader.read_vector<char>();
        if (chunk_data.empty()) {
            throw std::runtime_error("Cannot load media chunk from server tokens state");
        }

        mtmd::input_chunk_ptr chunk(mtmd_input_chunk_load(chunk_data.data(), chunk_data.size()));
        if (!chunk) {
            throw std::runtime_error("Cannot load media chunk from server tokens state");
        }
        result.map_idx_to_media[start_idx] = std::move(chunk);
    }

    if (reader.remaining() >= sizeof(llama_token)) {
        throw std::runtime_error("Trailing data in server tokens state");
    }

    return result;
}

llama_tokens server_tokens::get_text_tokens() const {
    llama_tokens res;
    res.reserve(tokens.size());
    for (llama_token t : tokens) {
        if (t != LLAMA_TOKEN_NULL) {
            res.push_back(t);
        }
    }
    return res;
}

void server_tokens::set_token(llama_pos pos, llama_token id) {
    GGML_ASSERT(!has_mtmd); // only allow this if mtmd is disabled
    tokens[pos] = id;
}

void server_tokens::keep_first(size_t n) {
    GGML_ASSERT(n <= tokens.size());
    if (has_mtmd) {
        if (n == tokens.size()) {
            return; // nothing to do
        }
        // we throw an error if we try to remove a token in the middle of an image
        // for ex. with input of 5 text tokens and 2 images:
        //    [0] [1] [2] [3] [4] [img0] [img0] [img0] [img1] [img1]
        // n  1   2   3   4   5   6      7      8      9      10
        // allowed to resize      ^                    ^
        // disallowed to resize          ^      ^             ^
        if (n > 0) {
            // make sure we never remove tokens in the middle of an image
            // note that the case where we keep a full image at the end is allowed:
            //   tokens[n - 1] == LLAMA_TOKEN_NULL && tokens[n] != LLAMA_TOKEN_NULL
            if (tokens[n - 1] == LLAMA_TOKEN_NULL && tokens[n] == LLAMA_TOKEN_NULL) {
                find_chunk(n - 1); // will throw an error if the token is not begin-of-chunk
            }
        }
        // remove all image chunks that are not used anymore
        for (auto it = map_idx_to_media.begin(); it != map_idx_to_media.end(); ) {
            size_t idx = it->first;
            if (idx >= n) {
                it = map_idx_to_media.erase(it);
            } else {
                ++it;
            }
        }
    }
    tokens.resize(n);
}

std::string server_tokens::detokenize(const llama_context * ctx, bool special) const {
    llama_tokens text_tokens;
    text_tokens.reserve(tokens.size());
    for (const auto & t : tokens) {
        if (t != LLAMA_TOKEN_NULL) {
            text_tokens.push_back(t);
        }
    }
    return common_detokenize(ctx, text_tokens, special);
}

size_t server_tokens::get_common_prefix(const server_tokens & b) const {
    const size_t max_idx = std::min(tokens.size(), b.tokens.size());

    if (!has_mtmd) {
        for (size_t i = 0; i < max_idx; ++i) {
            if (tokens[i] == b.tokens[i]) {
                continue;
            }

            return i;
        }

        return max_idx;
    }

    for (size_t i = 0; i < max_idx; ++i) {
        const llama_token ai =   tokens[i];
        const llama_token bi = b.tokens[i];

        if (ai == LLAMA_TOKEN_NULL && bi == LLAMA_TOKEN_NULL) {
            const auto & a_chunk =   find_chunk(i);
            const auto & b_chunk = b.find_chunk(i);

            GGML_ASSERT(a_chunk && b_chunk);

            const std::string id_ai = mtmd_input_chunk_get_id(a_chunk.get());
            const std::string id_bi = mtmd_input_chunk_get_id(b_chunk.get());

            const size_t n_tok_a = mtmd_input_chunk_get_n_tokens(a_chunk.get());
            const size_t n_tok_b = mtmd_input_chunk_get_n_tokens(b_chunk.get());

            if (id_ai == id_bi && n_tok_a == n_tok_b) {
                GGML_ASSERT(n_tok_a > 0 && "Invalid media chunk"); // should never happen
                i += n_tok_a - 1; // will be +1 by the for loop
                continue;
            }

            return i;
        }

        if (ai == bi) {
            continue;
        }

        return i;
    }

    return max_idx; // all tokens are equal
}

common_chat_msg_spans server_tokens::find_message_spans(const common_chat_msg_delimiters & delims) const {
    std::map<size_t, size_t> skips;
    for (const auto & it : map_idx_to_media) {
        skips[it.first] = mtmd_input_chunk_get_n_tokens(it.second.get());
    }
    return delims.split(tokens, skips);
}

bool server_tokens::validate(const struct llama_context * ctx) const {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    size_t n_media = 0;

    for (size_t i = 0; i < tokens.size(); ++i) {
        const auto & t = tokens[i];
        if (t == LLAMA_TOKEN_NULL) {
            try {
                const auto & chunk = find_chunk(i);
                if (mtmd_input_chunk_get_type(chunk.get()) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                    return false;
                }
                const size_t n_tokens = mtmd_input_chunk_get_n_tokens(chunk.get());
                const llama_pos n_pos = mtmd_input_chunk_get_n_pos(chunk.get());
                if (n_tokens == 0 || n_pos <= 0 || n_tokens > tokens.size() - i) {
                    return false;
                }
                for (size_t j = i; j < i + n_tokens; ++j) {
                    if (tokens[j] != LLAMA_TOKEN_NULL) {
                        return false;
                    }
                }
                ++n_media;
                i += n_tokens - 1;
            } catch (const std::exception & e) {
                return false;
            }
        } else if (t < 0 || t >= n_vocab) {
            return false;
        }
    }
    return n_media == map_idx_to_media.size();
}

server_tokens server_tokens::clone() const {
    server_tokens res;
    res.has_mtmd = has_mtmd;
    res.tokens   = tokens;
    for (auto it = map_idx_to_media.begin(); it != map_idx_to_media.end(); ++it) {
        size_t idx = it->first;
        const mtmd::input_chunk_ptr & chunk = it->second;
        res.map_idx_to_media[idx] = mtmd::input_chunk_ptr(mtmd_input_chunk_copy(chunk.get()));
    }
    return res;
}

//
// tokenizer and input processing utils
//

bool json_is_array_of_numbers(const json & data) {
    if (data.is_array()) {
        for (const auto & e : data) {
            if (!e.is_number_integer()) {
                return false;
            }
        }
        return true;
    }
    return false;
}

bool json_is_array_of_mixed_numbers_strings(const json & data) {
    bool seen_string = false;
    bool seen_number = false;
    if (data.is_array()) {
        for (const auto & e : data) {
            seen_string |= e.is_string();
            seen_number |= e.is_number_integer();
            if (seen_number && seen_string) {
                return true;
            }
        }
    }
    return false;
}

bool json_is_array_and_contains_numbers(const json & data) {
    if (data.is_array()) {
        for (const auto & e : data) {
            if (e.is_number_integer()) {
                return true;
            }
        }
        return false;
    }
    return false;
}

json json_get_nested_values(const std::vector<std::string> & paths, const json & js) {
    json result = json::object();

    for (const std::string & path : paths) {
        json current = js;
        const auto keys = string_split<std::string>(path, /*separator*/ '/');
        bool valid_path = true;
        for (const std::string & k : keys) {
            if (valid_path && current.is_object() && current.contains(k)) {
                current = current[k];
            } else {
                valid_path = false;
            }
        }
        if (valid_path) {
            result[path] = current;
        }
    }
    return result;
}

llama_tokens tokenize_mixed(const llama_vocab * vocab, const json & json_prompt, bool add_special, bool parse_special) {
    // If `add_bos` is true, we only add BOS, when json_prompt is a string,
    // or the first element of the json_prompt array is a string.
    llama_tokens prompt_tokens;

    if (json_prompt.is_array()) {
        bool first = true;
        for (const auto & p : json_prompt) {
            if (p.is_string()) {
                auto s = p.template get<std::string>();

                llama_tokens p;
                if (first) {
                    p = common_tokenize(vocab, s, add_special, parse_special);
                    first = false;
                } else {
                    p = common_tokenize(vocab, s, false, parse_special);
                }

                prompt_tokens.insert(prompt_tokens.end(), p.begin(), p.end());
            } else {
                if (first) {
                    first = false;
                }

                prompt_tokens.push_back(p.template get<llama_token>());
            }
        }
    } else {
        auto s = json_prompt.template get<std::string>();
        prompt_tokens = common_tokenize(vocab, s, add_special, parse_special);
    }

    return prompt_tokens;
}

size_t validate_utf8(const std::string& text) {
    size_t len = text.size();
    if (len == 0) return 0;

    // Check the last few bytes to see if a multi-byte character is cut off
    for (size_t i = 1; i <= 4 && i <= len; ++i) {
        unsigned char c = text[len - i];
        // Check for start of a multi-byte sequence from the end
        if ((c & 0xE0) == 0xC0) {
            // 2-byte character start: 110xxxxx
            // Needs at least 2 bytes
            if (i < 2) return len - i;
        } else if ((c & 0xF0) == 0xE0) {
            // 3-byte character start: 1110xxxx
            // Needs at least 3 bytes
            if (i < 3) return len - i;
        } else if ((c & 0xF8) == 0xF0) {
            // 4-byte character start: 11110xxx
            // Needs at least 4 bytes
            if (i < 4) return len - i;
        }
    }

    // If no cut-off multi-byte character is found, return full length
    return len;
}

server_tokens process_mtmd_prompt(
        mtmd_context * mctx,
        const std::string & prompt,
        const std::vector<raw_buffer> & files,
        const mtmd_helper_init_opt & init_opt,
        bool is_placeholder) {
    // these will be freed upon going out of scope
    mtmd::bitmaps bitmaps;
    std::vector<mtmd_helper::video_ptr> videos;
    for (auto & file : files) {
        auto out = mtmd_helper_bitmap_init_from_buf(mctx, file.data(), file.size(), is_placeholder, init_opt);
        if (!out.bitmap) {
            throw std::runtime_error("Failed to load image or audio file");
        }
        bitmaps.entries.emplace_back(out.bitmap);
        if (out.video_ctx) {
            videos.emplace_back(out.video_ctx);
        }
    }
    // process prompt
    std::vector<server_tokens> inputs;
    // multimodal
    mtmd_input_text inp_txt = {
        prompt.data(),
        prompt.size(),
        /* add_special */   true,
        /* parse_special */ true,
    };
    mtmd::input_chunks chunks(mtmd_input_chunks_init());
    auto bitmaps_c_ptr = bitmaps.c_ptr();
    int32_t tokenized = mtmd_tokenize(mctx,
                                      chunks.ptr.get(),
                                      &inp_txt,
                                      bitmaps_c_ptr.data(),
                                      bitmaps_c_ptr.size());
    if (tokenized != 0) {
        throw std::runtime_error("Failed to tokenize prompt");
    }
    auto result = server_tokens(chunks, true);
    return result;
}

/**
 * break the input "prompt" object into multiple prompt if needed, then tokenize them
 * use tokenize_input_prompts() if the input could be an array.
 * this supports these cases:
 * - "prompt": "string"
 * - "prompt": [12, 34, 56]
 * - "prompt": [12, 34, "string", 56, 78]
 * - "prompt": { "prompt_string": "string", "multimodal_data": [ "base64" ] }
 */
static server_tokens tokenize_input_subprompt(const llama_vocab * vocab, mtmd_context * mctx, const json & json_prompt, bool add_special, bool parse_special, const mtmd_helper_init_opt & init_opt) {
    constexpr char JSON_STRING_PROMPT_KEY[] = "prompt_string";
    constexpr char JSON_MTMD_DATA_KEY[] = "multimodal_data";
    const bool has_mtmd = mctx != nullptr;
    if (json_prompt.is_string() || json_is_array_of_mixed_numbers_strings(json_prompt)) {
        // string or mixed
        llama_tokens tmp = tokenize_mixed(vocab, json_prompt, add_special, parse_special);
        return server_tokens(tmp, false);
    } else if (json_is_array_of_numbers(json_prompt)) {
        // array of tokens
        llama_tokens tmp = json_prompt.get<llama_tokens>();
        return server_tokens(tmp, false);
    } else if (json_prompt.contains(JSON_STRING_PROMPT_KEY)) {
        // JSON object with prompt key.
        if (json_prompt.contains(JSON_MTMD_DATA_KEY)) {
            if (!has_mtmd)
                throw std::runtime_error("Multimodal data provided, but model does not support multimodal requests.");

            // JSON object with prompt and multimodal key.
            std::vector<raw_buffer> files;
            for (const auto & entry : json_prompt.at(JSON_MTMD_DATA_KEY)) {
                files.push_back(base64_decode(entry));
            }
            return process_mtmd_prompt(mctx, json_prompt.at(JSON_STRING_PROMPT_KEY), files, init_opt);
        } else {
            // Not multimodal, but contains a subobject.
            llama_tokens tmp = tokenize_mixed(vocab, json_prompt.at(JSON_STRING_PROMPT_KEY), add_special, parse_special);
            return server_tokens(tmp, false);
        }
   } else {
       throw std::runtime_error("\"prompt\" elements must be a string, a list of tokens, a JSON object containing a prompt string, or a list of mixed strings & tokens.");
   }
}

std::vector<server_tokens> tokenize_input_prompts(const llama_vocab * vocab, mtmd_context * mctx, const json & json_prompt, bool add_special, bool parse_special, const mtmd_helper_init_opt & init_opt) {
    std::vector<server_tokens> result;
    if (json_prompt.is_array() && !json_is_array_and_contains_numbers(json_prompt)) {
        result.reserve(json_prompt.size());
        for (const auto & p : json_prompt) {
            result.push_back(tokenize_input_subprompt(vocab, mctx, p, add_special, parse_special, init_opt));
        }
    } else {
        result.push_back(tokenize_input_subprompt(vocab, mctx, json_prompt, add_special, parse_special, init_opt));
    }
    if (result.empty()) {
        throw std::runtime_error("\"prompt\" must not be empty");
    }
    return result;
}

//
// OAI utils
//

// used by /completions endpoint
json oaicompat_completion_params_parse(const json & body) {
    json llama_params;

    server_openai_validate_completions_create(body);

    // Handle "stop" field
    if (body.contains("stop") && body.at("stop").is_string()) {
        llama_params["stop"] = json::array({body.at("stop").get<std::string>()});
    } else {
        llama_params["stop"] = json_value(body, "stop", json::array());
    }

    // Copy remaining properties to llama_params
    for (const auto & item : body.items()) {
        // Exception: if "n_predict" is present, we overwrite the value specified earlier by "max_tokens"
        if (!llama_params.contains(item.key()) || item.key() == "n_predict") {
            llama_params[item.key()] = item.value();
        }
    }

    // best_of: generate best_of candidates; return only n choices (ranked in handle_completions_impl).
    {
        int n_val = 1;
        if (body.contains("n") && !body.at("n").is_null() && body.at("n").is_number_integer()) {
            n_val = body.at("n").get<int>();
        }
        int best_of = n_val;
        if (body.contains("best_of") && !body.at("best_of").is_null() &&
                body.at("best_of").is_number_integer()) {
            best_of = body.at("best_of").get<int>();
        }
        if (best_of > n_val) {
            llama_params["n"] = best_of;
            llama_params["__oai_return_n"] = n_val;
        }
    }

    return llama_params;
}

// url can be
// - http(s):// for remote files
// - file:// for local files (only allowed if media_path is set)
// - data: for base64 encoded data with uri scheme (e.g. data:image/png;base64,...)
// - raw base64 encoded data
static void handle_media(
        std::vector<raw_buffer> & out_files,
        const std::string & url,
        const std::string & media_path) {
    if (!media_path.empty()) {
        // should already be enforced by arg.cpp, but checking just in case
        GGML_ASSERT(media_path.back() == DIRECTORY_SEPARATOR);
    }

    if (string_starts_with(url, "http")) {
        // download remote image
        // TODO @ngxson : maybe make these params configurable
        common_remote_params params;
        params.max_size = 1024 * 1024 * 10; // 10MB
        params.timeout  = 10; // seconds
        SRV_INF("downloading image from '%s'\n", url.c_str());
        auto res = common_remote_get_content(url, params);
        if (200 <= res.first && res.first < 300) {
            SRV_INF("downloaded %zu bytes\n", res.second.size());
            raw_buffer data;
            data.insert(data.end(), res.second.begin(), res.second.end());
            out_files.push_back(data);
        } else {
            throw std::runtime_error("Failed to download image");
        }

    } else if (string_starts_with(url, "file://")) {
        if (media_path.empty()) {
            throw std::invalid_argument("file:// URLs are not allowed unless --media-path is specified");
        }
        // load local image file
        std::string file_path = url.substr(7); // remove "file://"
        raw_buffer data;
        if (!fs_validate_filename(file_path, true)) {
            throw std::invalid_argument("file path is not allowed: " + file_path);
        }
        SRV_INF("loading image from local file '%s'\n", (media_path + file_path).c_str());
        std::ifstream file(media_path + file_path, std::ios::binary);
        if (!file) {
            throw std::invalid_argument("file does not exist or cannot be opened: " + file_path);
        }
        data.assign((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        out_files.push_back(data);

    } else if (string_starts_with(url, "data:")) {
        // try to decode base64 image, video, or audio
        std::vector<std::string> parts = string_split<std::string>(url, /*separator*/ ',');
        if (parts.size() != 2) {
            throw std::invalid_argument("Invalid uri-encoded base64 value");
        } else if (!string_starts_with(parts[0], "data:image/")
                && !string_starts_with(parts[0], "data:video/")
                && !string_starts_with(parts[0], "data:audio/")) {
            throw std::invalid_argument("Invalid uri format: " + parts[0]);
        } else if (!string_ends_with(parts[0], "base64")) {
            throw std::invalid_argument("uri must be base64 encoded");
        } else {
            auto base64_data = parts[1];
            auto decoded_data = base64_decode(base64_data);
            out_files.push_back(decoded_data);
        }

    } else {
        // try as raw base64 string
        auto decoded_data = base64_decode(url);
        if (decoded_data.empty()) {
            throw std::runtime_error("Invalid base64 value");
        }
        out_files.push_back(decoded_data);
    }
}

// Nearest official reasoning_effort levels, for templates that accept only a subset.
// Ordered by rank distance, higher level first on ties; "none" is never a candidate
// because it disables thinking.
static std::vector<std::string> server_openai_reasoning_effort_fallbacks(const std::string & effort) {
    static const char * levels[] = { "minimal", "low", "medium", "high", "xhigh", "max" };
    const int n_levels = (int) (sizeof(levels) / sizeof(levels[0]));
    int rank = -1;
    for (int i = 0; i < n_levels; ++i) {
        if (effort == levels[i]) {
            rank = i;
            break;
        }
    }
    std::vector<std::string> fallbacks;
    if (rank < 0) {
        return fallbacks;
    }
    for (int dist = 1; dist < n_levels && (int) fallbacks.size() < 3; ++dist) {
        if (rank + dist < n_levels) {
            fallbacks.push_back(levels[rank + dist]);
        }
        if ((int) fallbacks.size() < 3 && rank - dist >= 0) {
            fallbacks.push_back(levels[rank - dist]);
        }
    }
    return fallbacks;
}

// Name of a tool entry, accepting Chat (function/custom nested) and flat Responses shapes.
static std::string oai_tool_entry_name(const json & tool) {
    if (!tool.is_object()) {
        return {};
    }
    if (tool.contains("function") && tool.at("function").is_object()) {
        return json_value(tool.at("function"), "name", std::string());
    }
    if (tool.contains("custom") && tool.at("custom").is_object()) {
        return json_value(tool.at("custom"), "name", std::string());
    }
    return json_value(tool, "name", std::string());
}

// used by /chat/completions endpoint
json oaicompat_chat_params_parse(
    json & body, /* openai api json semantics */
    const server_chat_params & opt,
    std::vector<raw_buffer> & out_files,
    bool openai_defaults)
{
    json llama_params;

    // Shared OpenAI-shaped field validation (prompt_cache_*).
    server_openai_validate_cloud_shaped_fields(body, /*allow_prompt=*/false);
    server_openai_validate_chat_create_fields(body);
    server_openai_apply_prompt_cache_semantics(body);

    // Local deepen: Chat Completions web_search_options → search + inject system context.
    server_openai_apply_web_search_semantics(body);

    // Legacy Chat Completions functions / function_call → tools / tool_choice (1:1 behavior).
    if (body.contains("functions") && !body.at("functions").is_null()) {
        if (!body.at("functions").is_array()) {
            throw std::invalid_argument("'functions' must be an array");
        }
        if (body.contains("tools") && body.at("tools").is_array() && !body.at("tools").empty()) {
            throw std::invalid_argument("Cannot set both 'functions' and 'tools'");
        }
        json tools_from_fn = json::array();
        for (const auto & fn : body.at("functions")) {
            if (!fn.is_object()) {
                throw std::invalid_argument("'functions' entries must be objects");
            }
            tools_from_fn.push_back(json{
                {"type", "function"},
                {"function", fn},
            });
        }
        body["tools"] = std::move(tools_from_fn);
    }
    if (body.contains("function_call") && !body.at("function_call").is_null()) {
        if (body.contains("tool_choice") && !body.at("tool_choice").is_null()) {
            throw std::invalid_argument("Cannot set both 'function_call' and 'tool_choice'");
        }
        const json & fc = body.at("function_call");
        if (fc.is_string()) {
            const std::string s = fc.get<std::string>();
            if (s == "none" || s == "auto") {
                body["tool_choice"] = s;
            } else {
                throw std::invalid_argument("'function_call' string must be 'none' or 'auto'");
            }
        } else if (fc.is_object()) {
            const std::string name = json_value(fc, "name", std::string());
            if (name.empty()) {
                throw std::invalid_argument("'function_call.name' is required");
            }
            body["tool_choice"] = json{
                {"type", "function"},
                {"function", {{"name", name}}},
            };
        } else {
            throw std::invalid_argument("'function_call' must be a string or object");
        }
    }

    json tools = json_value(body, "tools", json());
    auto has_tools = tools.is_array() && !tools.empty();
    auto stream = json_value(body, "stream", false);
    // OpenAI Chat Completions: tool_choice string or
    // {"type":"function","function":{"name":"..."}}.
    // Responses: {"type":"function","name":"..."}.
    // Anthropic convert: {"type":"function","function":{"name":"..."}} from type=tool.
    // Named force must restrict tools to that name (behavior == method), not only "required".
    std::string tool_choice = "auto";
    std::string forced_tool_name;
    if (body.contains("tool_choice") && !body.at("tool_choice").is_null()) {
        const json & tc = body.at("tool_choice");
        if (tc.is_string()) {
            tool_choice = tc.get<std::string>();
        } else if (tc.is_object()) {
            const std::string tc_type = json_value(tc, "type", std::string("auto"));
            if (tc_type == "none" || tc_type == "auto" || tc_type == "required") {
                tool_choice = tc_type;
            } else if (tc_type == "allowed_tools") {
                // Restrict callable tools to the listed subset (official ToolChoiceAllowed).
                // Official shape nests under 'allowed_tools'; the flat local shape stays supported.
                const json & allowed = tc.contains("allowed_tools") && tc.at("allowed_tools").is_object()
                    ? tc.at("allowed_tools") : tc;
                const std::string mode = json_value(allowed, "mode", std::string("auto"));
                if (mode != "auto" && mode != "required") {
                    throw std::invalid_argument("'tool_choice.mode' must be 'auto' or 'required'");
                }
                if (!allowed.contains("tools") || !allowed.at("tools").is_array()) {
                    throw std::invalid_argument("'tool_choice.tools' must be an array");
                }
                std::unordered_set<std::string> allowed_names;
                for (const auto & entry : allowed.at("tools")) {
                    if (!entry.is_object()) {
                        throw std::invalid_argument("'tool_choice.tools' entries must be objects");
                    }
                    const std::string entry_type = json_value(entry, "type", std::string());
                    if (entry_type == "function" || entry_type == "custom") {
                        const std::string name = oai_tool_entry_name(entry);
                        if (!name.empty()) {
                            allowed_names.insert(name);
                        }
                    }
                }
                bool filtered_out = false;
                if (has_tools) {
                    json filtered = json::array();
                    for (const auto & tool : tools) {
                        if (!tool.is_object()) {
                            continue;
                        }
                        if (allowed_names.count(oai_tool_entry_name(tool))) {
                            filtered.push_back(tool);
                        }
                    }
                    filtered_out = filtered.empty();
                    tools = std::move(filtered);
                    has_tools = !tools.empty();
                }
                if (mode == "required" && !has_tools) {
                    throw std::invalid_argument(
                        "tool_choice requires at least one allowed tool present in 'tools'");
                }
                if (mode == "auto" && filtered_out) {
                    throw std::invalid_argument(
                        "tool_choice allowed_tools with mode 'auto' matched no tools in 'tools'");
                }
                tool_choice = mode;
            } else if (tc_type == "function" || tc_type == "tool") {
                if (tc.contains("function") && tc.at("function").is_object()) {
                    forced_tool_name = json_value(tc.at("function"), "name", std::string());
                } else {
                    forced_tool_name = json_value(tc, "name", std::string());
                }
                if (forced_tool_name.empty()) {
                    throw std::invalid_argument("tool_choice function name is required");
                }
                tool_choice = "required";
            } else if (tc_type == "custom") {
                // Official Chat/Responses: force one custom tool by name.
                if (tc.contains("custom") && tc.at("custom").is_object()) {
                    forced_tool_name = json_value(tc.at("custom"), "name", std::string());
                } else {
                    forced_tool_name = json_value(tc, "name", std::string());
                }
                if (forced_tool_name.empty()) {
                    throw std::invalid_argument("tool_choice custom name is required");
                }
                tool_choice = "required";
            } else {
                throw std::invalid_argument("Invalid tool_choice.type: " + tc_type);
            }
        } else {
            throw std::invalid_argument("tool_choice must be a string or object");
        }
    }

    // Chat custom tool names: serializers pick the official custom tool_call output shape
    // by name (everything else stays function-shaped).
    std::vector<std::string> oai_custom_tool_names;
    if (has_tools) {
        for (const auto & tool : tools) {
            if (!tool.is_object()) {
                continue;
            }
            const std::string type = json_value(tool, "type", std::string("function"));
            if (type == "function") {
                continue;
            }
            if (type == "custom") {
                const std::string name = oai_tool_entry_name(tool);
                if (name.empty()) {
                    throw std::invalid_argument("'tools' entry of type 'custom' requires a name");
                }
                oai_custom_tool_names.push_back(name);
                continue;
            }
            throw std::invalid_argument(
                "Chat Completions tool type '" + type + "' is not supported on this server "
                "(only type=function and type=custom). Cloud tool execution is unavailable locally.");
        }
    }

    if (!forced_tool_name.empty()) {
        if (!has_tools) {
            throw std::invalid_argument("tool_choice requires tools");
        }
        json filtered = json::array();
        for (const auto & tool : tools) {
            if (!tool.is_object()) {
                continue;
            }
            if (oai_tool_entry_name(tool) == forced_tool_name) {
                filtered.push_back(tool);
            }
        }
        if (filtered.empty()) {
            throw std::invalid_argument("Unknown tool_choice tool: " + forced_tool_name);
        }
        tools = std::move(filtered);
        has_tools = true;
    }

    if (!opt.use_jinja) {
        if (has_tools) {
            throw std::runtime_error("tools param requires --jinja flag");
        }
        if (tool_choice != "auto") {
            throw std::runtime_error("tool_choice param requires --jinja flag");
        }
    }

    // Handle "stop" field
    if (body.contains("stop") && body.at("stop").is_string()) {
        llama_params["stop"] = json::array({body.at("stop").get<std::string>()});
    } else {
        llama_params["stop"] = json_value(body, "stop", json::array());
    }

    auto json_schema = json_value(body, "json_schema", json());
    auto grammar = json_value(body, "grammar", std::string());
    if (!json_schema.is_null() && !grammar.empty()) {
        throw std::runtime_error("Cannot use both json_schema and grammar");
    }

    // Handle "response_format" field
    if (body.contains("response_format")) {
        json response_format      = json_value(body, "response_format", json::object());
        std::string response_type = json_value(response_format, "type", std::string());
        if (response_type == "json_object") {
            if (response_format.contains("schema") || json_schema.empty()) {
                json_schema = json_value(response_format, "schema", json::object());
            }
        } else if (response_type == "json_schema") {
            auto schema_wrapper = json_value(response_format, "json_schema", json::object());
            json_schema = json_value(schema_wrapper, "schema", json::object());
        } else if (!response_type.empty() && response_type != "text") {
            throw std::invalid_argument("response_format type must be one of \"text\" or \"json_object\", but got: " + response_type);
        }
    }

    // an absent or empty schema means any object
    if (json_schema.is_object() && json_schema.empty()) {
        json_schema["type"] = "object";
    }

    // get input files
    if (!body.contains("messages")) {
        throw std::invalid_argument("'messages' is required");
    }
    json & messages = body.at("messages");
    if (!messages.is_array()) {
        throw std::invalid_argument("Expected 'messages' to be an array");
    }

    // prompt_cache_breakpoint: validate on the raw parts (the media rewrite below mutates
    // them). Breakpoints are located later as {role, message ordinal}; 4 per request max.
    std::vector<std::pair<std::string, int32_t>> oai_prompt_cache_breakpoints;
    {
        std::map<std::string, int32_t> role_counts;
        size_t n_breakpoints = 0;
        for (const auto & msg : messages) {
            const std::string role = json_value(msg, "role", std::string());
            const int32_t ordinal = ++role_counts[role];

            if (!msg.contains("content") || !msg.at("content").is_array()) {
                continue;
            }
            bool msg_has_breakpoint = false;
            for (const auto & p : msg.at("content")) {
                if (!p.is_object() || !p.contains("prompt_cache_breakpoint") ||
                        p.at("prompt_cache_breakpoint").is_null()) {
                    continue;
                }
                msg_has_breakpoint = true;
                n_breakpoints++;
                const json & bp = p.at("prompt_cache_breakpoint");
                if (!bp.is_object()) {
                    throw std::invalid_argument("'prompt_cache_breakpoint' must be an object");
                }
                if (!bp.contains("mode") || !bp.at("mode").is_string() ||
                        bp.at("mode").get<std::string>() != "explicit") {
                    throw std::invalid_argument("'prompt_cache_breakpoint.mode' must be 'explicit'");
                }
            }
            // locate a message once, even if several of its parts carry a breakpoint
            if (msg_has_breakpoint) {
                oai_prompt_cache_breakpoints.emplace_back(role, ordinal);
            }
        }
        if (n_breakpoints > 4) {
            throw std::invalid_argument("'prompt_cache_breakpoint' is allowed at most 4 times per request");
        }
    }

    for (auto & msg : messages) {
        std::string role = json_value(msg, "role", std::string());
        if (role == "assistant" && msg.contains("function_call") && !msg.at("function_call").is_null()) {
            // Legacy single function_call is deprecated but accepted; normalize it to tool_calls.
            const json & fc = msg.at("function_call");
            if (!fc.is_object()) {
                throw std::invalid_argument("'function_call' must be an object");
            }
            if (msg.contains("tool_calls") && msg.at("tool_calls").is_array() && !msg.at("tool_calls").empty()) {
                throw std::invalid_argument("Cannot set both 'function_call' and 'tool_calls'");
            }
            const std::string name = json_value(fc, "name", std::string());
            if (name.empty()) {
                throw std::invalid_argument("'function_call.name' is required");
            }
            msg["tool_calls"] = json::array({ json {
                {"type", "function"},
                {"function", json {
                    {"name",      name},
                    {"arguments", json_value(fc, "arguments", std::string())},
                }},
            }});
            msg.erase("function_call");
        }
        if (role != "assistant" && !msg.contains("content")) {
            throw std::invalid_argument("All non-assistant messages must contain 'content'");
        }
        if (role == "assistant") {
            if (!msg.contains("content") && !msg.contains("tool_calls")) {
                throw std::invalid_argument("Assistant message must contain either 'content' or 'tool_calls'!");
            }
            if (!msg.contains("content")) {
                continue; // avoid errors with no content
            }
        }
        json & content = msg.at("content");
        if (content.is_string() || content.is_null()) {
            continue;
        }

        if (!content.is_array()) {
            throw std::invalid_argument("Expected 'content' to be a string or an array");
        }

        for (size_t i = 0; i < content.size(); ) {
            json & p = content[i];
            std::string type = json_value(p, "type", std::string());
            if (type == "image_url") {
                if (!opt.allow_image) {
                    throw std::runtime_error("image input is not supported - hint: if this is unexpected, you may need to provide the mmproj");
                }

                json image_url = json_value(p, "image_url", json::object());
                std::string url = json_value(image_url, "url", std::string());
                handle_media(out_files, url, opt.media_path);

                p["type"] = "media_marker";
                p["text"] = get_media_marker();
                p.erase("image_url");

            } else if (type == "input_audio") {
                // shape check first: id requires a file store this server does not have,
                // reject it as an invalid request regardless of audio support
                json input_audio = json_value(p, "input_audio", json::object());
                if (input_audio.contains("id") && !input_audio.at("id").is_null()) {
                    throw std::invalid_argument("'input_audio.id' is not supported on this server (no file storage); pass 'input_audio.data' instead");
                }
                if (!opt.allow_audio) {
                    throw std::runtime_error("audio input is not supported - hint: if this is unexpected, you may need to provide the mmproj");
                }

                // note: don't need to validate "format", it's redundant
                std::string url  = json_value(input_audio, "data",
                                        json_value(input_audio, "url", std::string()));
                handle_media(out_files, url, opt.media_path);

                p["type"] = "media_marker";
                p["text"] = get_media_marker();
                p.erase("input_audio");

            } else if (type == "input_video") {
                if (!opt.allow_video) {
                    throw std::runtime_error("video input is not supported - hint: if this is unexpected, you may need to provide the mmproj");
                }

                json input_video = json_value(p, "input_video", json::object());
                std::string url  = json_value(input_video, "data",
                                        json_value(input_video, "url", std::string()));
                handle_media(out_files, url, opt.media_path);

                p["type"] = "media_marker";
                p["text"] = get_media_marker();
                p.erase("input_video");

            } else if (type == "file") {
                // Official FileContentPart: file_data is decoded into the prompt as text,
                // file_id would need a file store this server does not have.
                json file = json_value(p, "file", json::object());
                if (file.contains("file_id") && !file.at("file_id").is_null()) {
                    throw std::invalid_argument("'file.file_id' is not supported on this server (no file storage); pass 'file.file_data' instead");
                }
                std::string data = json_value(file, "file_data", std::string());
                if (data.empty()) {
                    throw std::invalid_argument("'file' content part requires 'file.file_data'");
                }
                // accept both a data URI and plain base64
                const size_t comma = data.find(',');
                if (data.rfind("data:", 0) == 0 && comma != std::string::npos) {
                    data = data.substr(comma + 1);
                }
                const raw_buffer decoded = base64_decode(data);
                if (decoded.empty()) {
                    throw std::invalid_argument("'file.file_data' is not valid base64");
                }
                p = json {
                    {"type", "text"},
                    {"text", std::string(decoded.begin(), decoded.end())},
                };

            } else if (type == "refusal") {
                // Official assistant refusal part; replayed with text semantics.
                if (!p.contains("refusal") || !p.at("refusal").is_string()) {
                    throw std::invalid_argument("'refusal' content part requires a string 'refusal'");
                }
                const std::string refusal = p.at("refusal").get<std::string>();
                p = json {
                    {"type", "text"},
                    {"text", refusal},
                };

            } else if (type == "moderation") {
                // Moderation parts are not produced locally: accept and drop on replay.
                content.erase(i);
                continue;

            } else if (type != "text") {
                throw std::invalid_argument("unsupported content[].type");
            }
            ++i;
        }
    }

    // --reasoning-preserve: when the jinja template lacks supports_preserve_reasoning,
    // fold prior assistant reasoning_content into content as <think>…</think> so history
    // thinking remains in the prompt (local complete semantics for Qwen/etc.).
    {
        auto it_pr = opt.chat_template_kwargs.find("preserve_reasoning");
        bool preserve_on = false;
        if (it_pr != opt.chat_template_kwargs.end()) {
            try {
                json v = json::parse(it_pr->second);
                preserve_on = v.is_boolean() && v.get<bool>();
            } catch (...) {
                preserve_on = (it_pr->second == "true");
            }
        }
        if (preserve_on) {
            auto tmpl_caps = common_chat_templates_get_caps(opt.tmpls.get());
            const bool tmpl_ok = tmpl_caps.count("supports_preserve_reasoning") &&
                                 tmpl_caps.at("supports_preserve_reasoning");
            if (!tmpl_ok) {
                for (auto & msg : messages) {
                    if (!msg.is_object()) {
                        continue;
                    }
                    if (json_value(msg, "role", std::string()) != "assistant") {
                        continue;
                    }
                    if (!msg.contains("reasoning_content") || !msg.at("reasoning_content").is_string()) {
                        continue;
                    }
                    const std::string reasoning = msg.at("reasoning_content").get<std::string>();
                    if (reasoning.empty()) {
                        continue;
                    }
                    if (!msg.contains("content") || msg.at("content").is_null()) {
                        msg["content"] = "<think>\n" + reasoning + "\n</think>\n";
                        continue;
                    }
                    if (!msg.at("content").is_string()) {
                        continue; // multipart content: leave to template
                    }
                    std::string content = msg.at("content").get<std::string>();
                    if (content.find("<think>") != std::string::npos ||
                        content.find("</think>") != std::string::npos) {
                        continue;
                    }
                    msg["content"] = "<think>\n" + reasoning + "\n</think>\n\n" + content;
                }
            }
        }
    }

    auto caps = common_chat_templates_get_caps(opt.tmpls.get());

    common_chat_templates_inputs inputs;
    inputs.messages               = common_chat_msgs_parse_oaicompat(messages);
    inputs.tools                  = common_chat_tools_parse_oaicompat(tools);
    inputs.tool_choice            = common_chat_tool_choice_parse_oaicompat(tool_choice);
    inputs.json_schema            = json_schema.is_null() ? "" : json_schema.dump();
    inputs.grammar                = grammar;
    inputs.use_jinja              = opt.use_jinja;
    inputs.parallel_tool_calls    = json_value(body, "parallel_tool_calls", caps["supports_parallel_tool_calls"]);
    inputs.add_generation_prompt  = json_value(body, "add_generation_prompt", true);
    inputs.continue_final_message = body.contains("continue_final_message") ?
        common_chat_continuation_parse(body.at("continue_final_message")) :
        COMMON_CHAT_CONTINUATION_NONE;
    if (inputs.continue_final_message == COMMON_CHAT_CONTINUATION_NONE && opt.prefill_assistant
        && !inputs.messages.empty() && inputs.messages.back().role == "assistant") {
        if (inputs.messages.size() >= 2 && inputs.messages[inputs.messages.size() - 2].role == "assistant") {
            throw std::invalid_argument("Cannot have 2 or more assistant messages at the end of the list.");
        }
        inputs.continue_final_message = COMMON_CHAT_CONTINUATION_AUTO;
        inputs.add_generation_prompt  = false;
    }
    if (inputs.continue_final_message != COMMON_CHAT_CONTINUATION_NONE && inputs.add_generation_prompt) {
        throw std::invalid_argument("Cannot set both add_generation_prompt and continue_final_message to true.");
    }
    if (inputs.continue_final_message != COMMON_CHAT_CONTINUATION_NONE
        && !inputs.messages.empty()
        && inputs.messages.back().role == "assistant"
        && !inputs.messages.back().tool_calls.empty()) {
        throw std::invalid_argument("Cannot continue an assistant message that contains tool calls.");
    }
    inputs.reasoning_format = opt.reasoning_format;
    if (body.contains("reasoning_format")) {
        inputs.reasoning_format = common_reasoning_format_from_name(body.at("reasoning_format").get<std::string>());
    }
    inputs.enable_thinking = opt.enable_thinking;
    if (!inputs.tools.empty() && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE) {
        if (body.contains("grammar")) {
            throw std::invalid_argument("Cannot use custom grammar constraints with tools.");
        }
        llama_params["parse_tool_calls"] = true;
    }

    // merge the template args provided from command line with the args provided in the user request
    auto chat_template_kwargs_object = json_value(body, "chat_template_kwargs", json::object());
    inputs.chat_template_kwargs = opt.chat_template_kwargs;
    for (const auto & item : chat_template_kwargs_object.items()) {
        inputs.chat_template_kwargs[item.key()] = item.value().dump();
    }

    // parse the "enable_thinking" kwarg to override the default value
    auto enable_thinking_kwarg = json_value(inputs.chat_template_kwargs, "enable_thinking", std::string(""));
    if (enable_thinking_kwarg == "true") {
        inputs.enable_thinking = true;
    } else if (enable_thinking_kwarg == "false") {
        inputs.enable_thinking = false;
    } else if (!enable_thinking_kwarg.empty() && enable_thinking_kwarg[0] == '"') {
        throw std::invalid_argument("invalid type for \"enable_thinking\" (expected boolean, got string)");
    }

    // OpenAI "reasoning_effort": none disables thinking; other official levels enable
    // thinking and map to a local thinking_budget_tokens ladder when unset.
    // Enum already validated in server_openai_validate_chat_create_fields /
    // server_openai_validate_reasoning_object (Responses → chatcmpl conversion).
    // Omitted effort follows the official default (medium) on OpenAI endpoints unless the
    // value was already picked via chat_template_kwargs (client) or --reasoning-effort (server).
    const bool has_body_effort = body.contains("reasoning_effort") && !body.at("reasoning_effort").is_null();
    const bool use_default_effort = openai_defaults && !has_body_effort && inputs.enable_thinking &&
        inputs.chat_template_kwargs.find("reasoning_effort") == inputs.chat_template_kwargs.end();
    std::string oai_injected_effort;
    if (has_body_effort || use_default_effort) {
        std::string reasoning_effort = "medium";
        if (has_body_effort) {
            server_openai_validate_reasoning_effort_field(body.at("reasoning_effort"), "reasoning_effort");
            reasoning_effort = body.at("reasoning_effort").get<std::string>();
        }
        if (reasoning_effort == "none") {
            inputs.enable_thinking = false;
            inputs.chat_template_kwargs["enable_thinking"] = "false";
        } else {
            inputs.enable_thinking = true;
            inputs.chat_template_kwargs["enable_thinking"] = "true";
            // Same encoding as request chat_template_kwargs merge (.dump() of JSON string).
            inputs.chat_template_kwargs["reasoning_effort"] = json(reasoning_effort).dump();
            if (openai_defaults) {
                // fallback retry applies to OpenAI endpoints only
                oai_injected_effort = reasoning_effort;
            }
            // Local deepen: effort → budget when client did not set an explicit budget field.
            if (!body.contains("thinking_budget_tokens") && !body.contains("reasoning_budget_tokens")) {
                int budget = 1024;
                if (reasoning_effort == "minimal") {
                    budget = 64;
                } else if (reasoning_effort == "low") {
                    budget = 256;
                } else if (reasoning_effort == "medium") {
                    budget = 1024;
                } else if (reasoning_effort == "high") {
                    budget = 4096;
                } else if (reasoning_effort == "xhigh" || reasoning_effort == "max") {
                    budget = 8192;
                }
                body["thinking_budget_tokens"] = budget;
            }
        }
    }

    // verbosity: inject a concise/detailed system hint (local observable behavior).
    // An omitted or null verbosity follows the official default (medium) on OpenAI endpoints.
    std::string v = openai_defaults ? "medium" : std::string();
    if (body.contains("verbosity") && !body.at("verbosity").is_null()) {
        // non-string shapes are rejected by server_openai_validate_chat_create_fields
        v = body.at("verbosity").is_string() ? body.at("verbosity").get<std::string>() : std::string();
    }
    std::string hint;
    if (v == "low") {
        hint = "Respond very concisely. Prefer short answers with minimal prose.";
    } else if (v == "medium") {
        hint = "Respond with a balanced amount of detail. Prefer clear, moderately sized answers.";
    } else if (v == "high") {
        hint = "Respond thoroughly and in detail. Prefer expansive explanations.";
    }
    if (!hint.empty()) {
        // System messages must stay first: many templates reject a system message that is
        // not leading. Extend the leading system/developer message instead of adding one.
        if (!inputs.messages.empty() &&
                (inputs.messages[0].role == "system" || inputs.messages[0].role == "developer")) {
            if (!inputs.messages[0].content_parts.empty()) {
                inputs.messages[0].content_parts.push_back({ "text", hint });
            } else if (inputs.messages[0].content.empty()) {
                inputs.messages[0].content = hint;
            } else {
                inputs.messages[0].content += "\n\n" + hint;
            }
        } else {
            common_chat_msg sys;
            sys.role = "system";
            sys.content = hint;
            inputs.messages.insert(inputs.messages.begin(), std::move(sys));
        }
    }

    // prediction: prefill assistant with predicted content (local Predicted Outputs stand-in).
    if (body.contains("prediction") && body.at("prediction").is_object()) {
        const json & pred = body.at("prediction");
        std::string pred_text;
        if (pred.contains("content") && pred.at("content").is_string()) {
            pred_text = pred.at("content").get<std::string>();
        } else if (pred.contains("content") && pred.at("content").is_array()) {
            for (const auto & part : pred.at("content")) {
                if (part.is_string()) {
                    pred_text += part.get<std::string>();
                } else if (part.is_object() && part.contains("text") && part.at("text").is_string()) {
                    pred_text += part.at("text").get<std::string>();
                }
            }
        }
        if (!pred_text.empty()) {
            common_chat_msg asst;
            asst.role = "assistant";
            asst.content = pred_text;
            inputs.messages.push_back(std::move(asst));
            inputs.continue_final_message = COMMON_CHAT_CONTINUATION_AUTO;
            inputs.add_generation_prompt = false;
            // Include prefilled prediction in streamed/final content (OpenAI-shaped continuity).
            inputs.chat_template_kwargs["__prediction_prefill"] = "true";
            llama_params["continue_final_message"] = true;
            // chat_parser echo so clients see the predicted prefix when it matches.
            llama_params["echo"] = true;
        }
    }

    inputs.force_pure_content = opt.force_pure_content;

    // Apply chat template to the list of messages. Some templates accept only a subset of
    // the official reasoning_effort levels; when the server injected the value, retry with
    // the nearest accepted level instead of failing the request. The original exception is
    // rethrown when no fallback applies.
    common_chat_params chat_params;
    try {
        chat_params = common_chat_templates_apply(opt.tmpls.get(), inputs);
    } catch (...) {
        const std::exception_ptr original = std::current_exception();
        bool applied = false;
        if (!oai_injected_effort.empty()) {
            const std::vector<std::string> fallbacks = server_openai_reasoning_effort_fallbacks(oai_injected_effort);
            for (const auto & fallback : fallbacks) {
                inputs.chat_template_kwargs["reasoning_effort"] = json(fallback).dump();
                try {
                    chat_params = common_chat_templates_apply(opt.tmpls.get(), inputs);
                    SRV_INF("reasoning_effort '%s' rejected by chat template, using nearest level '%s'\n",
                            oai_injected_effort.c_str(), fallback.c_str());
                    applied = true;
                    break;
                } catch (...) {
                    // try the next candidate
                }
            }
        }
        if (!applied) {
            std::rethrow_exception(original);
        }
    }

    llama_params["chat_format"] = static_cast<int>(chat_params.format);
    llama_params["prompt"]      = chat_params.prompt;
    if (!chat_params.grammar.empty()) {
        llama_params["grammar"]      = chat_params.grammar;
        llama_params["grammar_type"] = std::string("tool_calls");
    }
    llama_params["grammar_lazy"] = chat_params.grammar_lazy;
    auto grammar_triggers        = json::array();
    for (const auto & trigger : chat_params.grammar_triggers) {
        server_grammar_trigger ct(trigger);
        grammar_triggers.push_back(ct.to_json());
    }
    llama_params["grammar_triggers"]  = grammar_triggers;
    llama_params["preserved_tokens"]  = chat_params.preserved_tokens;
    llama_params["generation_prompt"] = chat_params.generation_prompt;
    for (const auto & stop : chat_params.additional_stops) {
        llama_params["stop"].push_back(stop);
    }
    if (!chat_params.parser.empty()) {
        llama_params["chat_parser"] = chat_params.parser;
    }

    llama_params["message_delimiters"] = chat_params.message_delimiters.to_json();

    // explicit prompt cache breakpoints: message anchors resolved to token positions by
    // server-context (per-request checkpoint positions). Empty = no extra checkpoints.
    if (!oai_prompt_cache_breakpoints.empty()) {
        json bps = json::array();
        for (const auto & bp : oai_prompt_cache_breakpoints) {
            bps.push_back({ {"role", bp.first}, {"ordinal", bp.second} });
        }
        llama_params["__oai_prompt_cache_breakpoints"] = std::move(bps);
    }

    // custom tool names are resolved by name when serializing tool calls (custom shape)
    if (!oai_custom_tool_names.empty()) {
        llama_params["__oai_custom_tool_names"] = oai_custom_tool_names;
    }

    // Reasoning budget: pass parameters through to sampling layer
    {
        const bool client_set_reasoning_budget =
            (body.contains("reasoning_budget_tokens") && !body.at("reasoning_budget_tokens").is_null()) ||
            (body.contains("thinking_budget_tokens") && !body.at("thinking_budget_tokens").is_null());
        int reasoning_budget = json_value(body, "reasoning_budget_tokens",
                               json_value(body, "thinking_budget_tokens", -1));
        if (reasoning_budget == -1) {
            reasoning_budget = opt.reasoning_budget;
        }

        // Local deepen only when the client did not set a budget field: unlimited
        // reasoning often fills a finite max_tokens with only thinking. Never rewrite
        // an explicit thinking_budget_tokens / reasoning_budget_tokens (1:1 with field).
        if (inputs.enable_thinking && !client_set_reasoning_budget) {
            int max_tok = -1;
            if (body.contains("n_predict") && body.at("n_predict").is_number_integer()) {
                max_tok = body.at("n_predict").get<int>();
            } else if (body.contains("max_tokens") && body.at("max_tokens").is_number_integer()) {
                max_tok = body.at("max_tokens").get<int>();
            } else if (body.contains("max_completion_tokens") && body.at("max_completion_tokens").is_number_integer()) {
                max_tok = body.at("max_completion_tokens").get<int>();
            }
            if (max_tok < 0) {
                if (reasoning_budget < 0) {
                    reasoning_budget = 8192;
                }
            } else if (max_tok > 0) {
                const int mt = max_tok;
                int reserve = std::max(1, mt / 4);
                if (mt >= 512) {
                    reserve = std::max(256, mt / 4);
                }
                reserve = std::min(reserve, mt - 1);
                const int cap = std::max(0, mt - reserve);
                if (reasoning_budget < 0 || reasoning_budget > cap) {
                    reasoning_budget = cap;
                }
            }
        }

        if (!chat_params.thinking_end_tags.empty()) {
            llama_params["reasoning_budget_tokens"] = reasoning_budget;
            llama_params["reasoning_budget_start_tag"] = chat_params.thinking_start_tag;
            llama_params["reasoning_budget_end_tags"] = chat_params.thinking_end_tags;
            llama_params["reasoning_budget_message"] = json_value(body, "reasoning_budget_message", opt.reasoning_budget_message);
            llama_params["reasoning_control"] = json_value(body, "reasoning_control", false);
        }
    }

    // Handle "logprobs" field (Chat Completions shape uses content[]).
    if (json_value(body, "logprobs", false)) {
        if (has_tools && stream) {
            throw std::invalid_argument("logprobs is not supported with tools + stream");
        }
        int top_lp = 20;
        if (body.contains("top_logprobs") && !body.at("top_logprobs").is_null()) {
            if (!body.at("top_logprobs").is_number_integer()) {
                throw std::invalid_argument("'top_logprobs' must be an integer between 0 and 20");
            }
            top_lp = body.at("top_logprobs").get<int>();
            if (top_lp < 0 || top_lp > 20) {
                throw std::invalid_argument("'top_logprobs' must be an integer between 0 and 20");
            }
        }
        llama_params["n_probs"] = top_lp;
    } else if (body.contains("top_logprobs") && !body.at("top_logprobs").is_null()) {
        throw std::invalid_argument("top_logprobs requires logprobs to be set to true");
    }

    // Copy remaining properties to llama_params
    // This allows user to use llama.cpp-specific params like "mirostat", ... via OAI endpoint.
    // See "launch_slot_with_task()" for a complete list of params supported by llama.cpp
    for (const auto & item : body.items()) {
        // Exception: if "n_predict" is present, we overwrite the value specified earlier by "max_tokens"
        if (!llama_params.contains(item.key()) || item.key() == "n_predict") {
            llama_params[item.key()] = item.value();
        }
    }

    return llama_params;
}

json format_embeddings_response_oaicompat(
        const json & request,
        const std::string & model_name,
        const json & embeddings,
        bool use_base64) {
    json data = json::array();
    int32_t n_tokens = 0;
    int i = 0;
    for (const auto & elem : embeddings) {
        json embedding_obj;

        if (use_base64) {
            const auto& vec = json_value(elem, "embedding", json::array()).get<std::vector<float>>();
            const char* data_ptr = reinterpret_cast<const char*>(vec.data());
            size_t data_size = vec.size() * sizeof(float);
            embedding_obj = {
                {"embedding", base64::encode(data_ptr, data_size)},
                {"index", i++},
                {"object", "embedding"}
            };
        } else {
            embedding_obj = {
                {"embedding", json_value(elem, "embedding", json::array())},
                {"index", i++},
                {"object", "embedding"}
            };
        }
        data.push_back(embedding_obj);

        n_tokens += json_value(elem, "tokens_evaluated", 0);
    }

    json res = json {
        {"model", json_value(request, "model", model_name)},
        {"object", "list"},
        {"usage", json {
            {"prompt_tokens", n_tokens},
            {"total_tokens", n_tokens}
        }},
        {"data", data}
    };

    return res;
}

json format_response_rerank(
        const json & request,
        const std::string & model_name,
        const json & ranks,
        bool is_tei_format,
        std::vector<std::string> & texts,
        int top_n) {
    int32_t n_tokens = 0;
    bool return_text = is_tei_format && json_value(request, "return_text", false);
    std::vector<json> elements; // Temporary vector to hold unsorted elements
    std::string score_label = is_tei_format ? "score" : "relevance_score";
    for (const auto & rank : ranks) {
        int index = json_value(rank, "index", 0);
        json elem = json{
            {"index", index},
            {score_label, json_value(rank, "score", 0.0)},
        };
        n_tokens += json_value(rank, "tokens_evaluated", 0);
        if (return_text) {
            elem["text"] = std::move(texts[index]);
        }
        elements.push_back(elem);
    }

    std::sort(elements.begin(), elements.end(), [score_label](const json& a, const json& b) {
        return json_value(a, score_label, 0.0) > json_value(b, score_label, 0.0);
    });

    elements.resize(std::min(top_n, (int)elements.size()));
    json results = elements;

    if (is_tei_format) return results;

    json res = json{
        {"model", json_value(request, "model", model_name)},
        {"object", "list"},
        {"usage", json{
            {"prompt_tokens", n_tokens},
            {"total_tokens", n_tokens}
        }},
        {"results", results}
    };

    return res;
}


//
// other utils
//

std::vector<llama_token_data> get_token_probabilities(llama_context * ctx, int idx, size_t n_top) {
    std::vector<llama_token_data> cur;

    const auto * logits = llama_get_logits_ith(ctx, idx);
    const llama_token * sampled_ids = llama_get_sampled_candidates_ith(ctx, idx);

    const int n_logits = llama_get_sampled_logits_count_ith(ctx, idx);

    cur.resize(n_logits);
    if (sampled_ids) {
        for (int i = 0; i < n_logits; i++) {
            cur[i] = llama_token_data{sampled_ids[i], logits[i], 0.0f};
        }
    } else {
        for (llama_token token_id = 0; token_id < n_logits; token_id++) {
            cur[token_id] = llama_token_data{token_id, logits[token_id], 0.0f};
        }
    }

    // sort tokens by logits (partial: only the leading `n_top` need ordering)
    if (n_top > cur.size()) {
        n_top = cur.size();
    }
    if (n_top > 0) {
        std::partial_sort(cur.begin(), cur.begin() + n_top, cur.end(),
            [](const llama_token_data & a, const llama_token_data & b) {
                return a.logit > b.logit;
            });
    }

    // apply softmax
    float max_l = -std::numeric_limits<float>::infinity();
    if (n_top > 0) {
        max_l = cur[0].logit; // partial_sort guarantees the absolute maximum is at index 0
    } else {
        for (const auto & t : cur) {
            max_l = std::max(max_l, t.logit);
        }
    }
    float cum_sum = 0.0f;
    for (auto & t : cur) {
        float p = expf(t.logit - max_l);
        t.p = p;
        cum_sum += p;
    }
    for (auto & t : cur) {
        t.p /= cum_sum;
    }

    return cur;
}

std::string safe_json_to_str(const json & data) {
    return data.dump_safe();
}

// TODO: reuse llama_detokenize
template <class Iter>
static std::string tokens_to_str(const llama_vocab * ctx, Iter begin, Iter end) {
    std::string ret;
    for (; begin != end; ++begin) {
        ret += common_token_to_piece(ctx, *begin);
    }

    return ret;
}

std::string tokens_to_str(llama_context * ctx, const llama_tokens & tokens) {
    auto model = llama_get_model(ctx);
    return tokens_to_str(llama_model_get_vocab(model), tokens.begin(), tokens.end());
}

std::string tokens_to_str(const llama_vocab * vocab, const llama_tokens & tokens) {
    return tokens_to_str(vocab, tokens.begin(), tokens.end());
}

// format incomplete utf-8 multibyte character for output
std::string tokens_to_output_formatted_string(const llama_context * ctx, const llama_token token) {
    std::string out = token == LLAMA_TOKEN_NULL ? "" : common_token_to_piece(ctx, token);

    // if the size is 1 and first bit is 1, meaning it's a partial character
    //   (size > 1 meaning it's already a known token)
    if (out.size() == 1 && (out[0] & 0x80) == 0x80) {
        std::stringstream ss;
        ss << std::hex << (out[0] & 0xff);
        std::string res(ss.str());
        out = "byte: \\x" + res;
    }

    return out;
}

// format server-sent event (SSE), return the formatted string to send
// note: if data is a json array, it will be sent as multiple events, one per item
std::string format_oai_sse(const json & data) {
    std::ostringstream ss;
    auto send_single = [&ss](const json & data) {
        ss << "data: " <<
            safe_json_to_str(data) <<
            "\n\n"; // required by RFC 8895 - A message is terminated by a blank line (two line terminators in a row).
    };

    if (data.is_array()) {
        for (const auto & item : data) {
            send_single(item);
        }
    } else {
        send_single(data);
    }

    return ss.str();
}

std::string format_oai_resp_sse(const json & data) {
    std::ostringstream ss;
    auto send_single = [&ss](const json & event_obj) {
        ss << "event: " << event_obj.at("event").get<std::string>() << "\n";
        ss << "data: " << safe_json_to_str(event_obj.at("data")) << "\n\n";
    };

    if (data.is_array()) {
        for (const auto & item : data) {
            send_single(item);
        }
    } else {
        send_single(data);
    }

    return ss.str();
}

std::string format_anthropic_sse(const json & data) {
    std::ostringstream ss;

    auto send_event = [&ss](const json & event_obj) {
        if (event_obj.contains("event") && event_obj.contains("data")) {
            ss << "event: " << event_obj.at("event").get<std::string>() << "\n";
            ss << "data: " << safe_json_to_str(event_obj.at("data")) << "\n\n";
        } else {
            ss << "data: " << safe_json_to_str(event_obj) << "\n\n";
        }
    };

    if (data.is_array()) {
        for (const auto & event : data) {
            send_event(event);
        }
    } else {
        send_event(data);
    }

    return ss.str();
}

bool is_valid_utf8(const std::string & str) {
    const unsigned char* bytes = reinterpret_cast<const unsigned char*>(str.data());
    const unsigned char* end = bytes + str.length();

    while (bytes < end) {
        if (*bytes <= 0x7F) {
            // 1-byte sequence (0xxxxxxx)
            bytes++;
        } else if ((*bytes & 0xE0) == 0xC0) {
            // 2-byte sequence (110xxxxx 10xxxxxx)
            if (end - bytes < 2 || (bytes[1] & 0xC0) != 0x80)
                return false;
            bytes += 2;
        } else if ((*bytes & 0xF0) == 0xE0) {
            // 3-byte sequence (1110xxxx 10xxxxxx 10xxxxxx)
            if (end - bytes < 3 || (bytes[1] & 0xC0) != 0x80 || (bytes[2] & 0xC0) != 0x80)
                return false;
            bytes += 3;
        } else if ((*bytes & 0xF8) == 0xF0) {
            // 4-byte sequence (11110xxx 10xxxxxx 10xxxxxx 10xxxxxx)
            if (end - bytes < 4 || (bytes[1] & 0xC0) != 0x80 ||
                (bytes[2] & 0xC0) != 0x80 || (bytes[3] & 0xC0) != 0x80)
                return false;
            bytes += 4;
        } else {
            // Invalid UTF-8 lead byte
            return false;
        }
    }

    return true;
}

llama_tokens format_prompt_infill(
        const llama_vocab * vocab,
        const json & input_prefix,
        const json & input_suffix,
        const json & input_extra,
        const int n_batch,
        const int n_predict,
        const int n_ctx,
        const bool spm_infill,
        const llama_tokens & tokens_prompt
    ) {
    // TODO: optimize this block by reducing memory allocations and movement

    // use FIM repo-level pattern:
    // ref: https://arxiv.org/pdf/2409.12186
    //
    // [FIM_REP]myproject
    // [FIM_SEP]filename0
    // extra chunk 0
    // [FIM_SEP]filename1
    // extra chunk 1
    // ...
    // [FIM_SEP]filename
    // [FIM_PRE]prefix[FIM_SUF]suffix[FIM_MID]prompt
    //
    llama_tokens extra_tokens;
    extra_tokens.reserve(n_ctx);

    auto tokens_prefix = tokenize_mixed(vocab, input_prefix, false, false);
    auto tokens_suffix = tokenize_mixed(vocab, input_suffix, false, false);

    if (llama_vocab_fim_rep(vocab) != LLAMA_TOKEN_NULL) {
        // TODO: make project name an input
        static const auto k_fim_repo = common_tokenize(vocab, "myproject\n", false, false);

        extra_tokens.push_back(llama_vocab_fim_rep(vocab));
        extra_tokens.insert(extra_tokens.end(), k_fim_repo.begin(), k_fim_repo.end());
    }
    for (const auto & chunk : input_extra) {
        // { "text": string, "filename": string }
        const std::string text     = json_value(chunk, "text",     std::string());
        const std::string filename = json_value(chunk, "filename", std::string("tmp"));

        if (llama_vocab_fim_sep(vocab) != LLAMA_TOKEN_NULL) {
            const auto k_fim_file = common_tokenize(vocab, filename + "\n", false, false);

            extra_tokens.insert(extra_tokens.end(), llama_vocab_fim_sep(vocab));
            extra_tokens.insert(extra_tokens.end(), k_fim_file.begin(), k_fim_file.end());
        } else {
            // chunk separator in binary form to avoid confusing the AI
            static const char k_chunk_prefix_str[] = {0x0a, 0x0a, 0x2d, 0x2d, 0x2d, 0x20, 0x73, 0x6e, 0x69, 0x70, 0x70, 0x65, 0x74, 0x20, 0x2d, 0x2d, 0x2d, 0x0a, 0x0a, 0x00};
            static const auto k_chunk_prefix_tokens = common_tokenize(vocab, k_chunk_prefix_str, false, false);

            extra_tokens.insert(extra_tokens.end(), k_chunk_prefix_tokens.begin(), k_chunk_prefix_tokens.end());
        }

        const auto chunk_tokens = common_tokenize(vocab, text, false, false);
        extra_tokens.insert(extra_tokens.end(), chunk_tokens.begin(), chunk_tokens.end());
    }

    if (llama_vocab_fim_sep(vocab) != LLAMA_TOKEN_NULL) {
        // TODO: current filename
        static const auto k_fim_file = common_tokenize(vocab, "filename\n", false, false);

        extra_tokens.insert(extra_tokens.end(), llama_vocab_fim_sep(vocab));
        extra_tokens.insert(extra_tokens.end(), k_fim_file.begin(), k_fim_file.end());
    }

    // for now pick FIM context to fit in a batch (ratio prefix:suffix = 3:1, TODO: configurable?)
    const int n_prefix_take = std::min<int>(tokens_prefix.size(),                3*(n_batch/4));
    const int n_suffix_take = std::min<int>(tokens_suffix.size(), std::max<int>(0, (n_batch/4) - (2 + tokens_prompt.size())));

    SRV_DBG("n_prefix_take = %d, n_suffix_take = %d, total = %d\n", n_prefix_take, n_suffix_take, (n_prefix_take + n_suffix_take));

    // fill the rest of the context with extra chunks
    const int n_extra_take = std::min<int>(std::max<int>(0, n_ctx - (n_batch) - 2*n_predict), extra_tokens.size());

    tokens_prefix.erase(tokens_prefix.begin(), tokens_prefix.begin() + tokens_prefix.size() - n_prefix_take);
    tokens_suffix.resize(n_suffix_take);

    tokens_prefix.insert(tokens_prefix.begin(), llama_vocab_fim_pre(vocab));
    tokens_prefix.insert(tokens_prefix.end(),   tokens_prompt.begin(), tokens_prompt.end());
    tokens_suffix.insert(tokens_suffix.begin(), llama_vocab_fim_suf(vocab));

    auto embd_inp = spm_infill ? tokens_suffix : tokens_prefix;
    auto embd_end = spm_infill ? tokens_prefix : tokens_suffix;

    if (llama_vocab_get_add_bos(vocab)) {
        embd_inp.insert(embd_inp.begin(), llama_vocab_bos(vocab));
    }

    SRV_DBG("extra: n_ctx = %d, n_extra_take = %d, n_extra = %d\n", n_ctx, n_extra_take, (int) extra_tokens.size());

    // put the extra context before the FIM prefix
    embd_inp.insert(embd_inp.begin(), extra_tokens.end() - n_extra_take, extra_tokens.end());

    embd_inp.insert(embd_inp.end(), embd_end.begin(), embd_end.end());
    embd_inp.push_back(llama_vocab_fim_mid(vocab));

    return embd_inp;
}

server_tokens format_prompt_rerank(
        const struct llama_model * model,
        const struct llama_vocab * vocab,
        mtmd_context * mctx,
        const std::string & query,
        const std::string & doc,
        const mtmd_helper_init_opt & init_opt) {
    server_tokens result = {};

    const char * rerank_prompt = llama_model_chat_template(model, "rerank");

    if (rerank_prompt != nullptr) {
        std::string prompt = rerank_prompt;
        string_replace_all(prompt, "{query}"   , query);
        string_replace_all(prompt, "{document}", doc  );
        server_tokens tokens = tokenize_input_subprompt(vocab, mctx, prompt, false, true, init_opt);
        result.push_back(tokens);
    } else {
        // Get EOS token - use SEP token as fallback if EOS is not available
        server_tokens query_tokens = tokenize_input_subprompt(vocab, mctx, query, false, false, init_opt);
        server_tokens doc_tokens   = tokenize_input_subprompt(vocab, mctx, doc,   false, false, init_opt);
        llama_token eos_token = llama_vocab_eos(vocab);
        if (eos_token == LLAMA_TOKEN_NULL) {
            eos_token = llama_vocab_sep(vocab);
        }

        if (llama_vocab_get_add_bos(vocab)) {
            result.push_back(llama_vocab_bos(vocab));
        }
        result.push_back(query_tokens);
        if (llama_vocab_get_add_eos(vocab)) {
            result.push_back(eos_token);
        }
        if (llama_vocab_get_add_sep(vocab)) {
            result.push_back(llama_vocab_sep(vocab));
        }
        result.push_back(doc_tokens);
        if (llama_vocab_get_add_eos(vocab)) {
            result.push_back(eos_token);
        }
    }

    return result;
}

//
// server_subproc
//

bool server_subproc::has_output() {
    if (out_handle >= 0) {
        return true;
    }
    FILE * f = sproc.stdout_file(); // combined stdout/stderr
    if (!f) {
        return false;
    }
#ifdef _WIN32
    HANDLE h = (HANDLE) _get_osfhandle(_fileno(f));
    if (h != INVALID_HANDLE_VALUE) {
        out_handle = (intptr_t) h;
    }
#else
    int fd = fileno(f);
    if (fd >= 0) {
        fcntl(fd, F_SETFL, fcntl(fd, F_GETFL, 0) | O_NONBLOCK);
        out_handle = fd;
    }
#endif
    return out_handle >= 0;
}

int server_subproc::read_output(char * buf, size_t len) {
    if (!has_output()) {
        return -1;
    }
#ifdef _WIN32
    HANDLE h     = (HANDLE) out_handle;
    DWORD  avail = 0;
    if (!PeekNamedPipe(h, NULL, 0, NULL, &avail, NULL)) {
        return -1; // pipe broken, child gone
    }
    if (avail == 0) {
        return 0;
    }
    DWORD to_read = avail < (DWORD) len ? avail : (DWORD) len;
    DWORD got     = 0;
    if (!ReadFile(h, buf, to_read, &got, NULL) || got == 0) {
        return -1;
    }
    return (int) got;
#else
    while (true) {
        ssize_t r = read((int) out_handle, buf, len);
        if (r > 0) {
            return (int) r;
        }
        if (r == 0) {
            return -1; // EOF
        }
        if (errno == EINTR) {
            continue;
        }
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            return 0;
        }
        return -1;
    }
#endif
}

server_subproc::waiter::waiter() {
#ifndef _WIN32
    int fds[2];
    GGML_ASSERT(pipe(fds) == 0);
    for (int fd : fds) {
        fcntl(fd, F_SETFL, fcntl(fd, F_GETFL, 0) | O_NONBLOCK);
    }
    wake_fd[0] = fds[0];
    wake_fd[1] = fds[1];
#endif
}

server_subproc::waiter::~waiter() {
#ifndef _WIN32
    close((int) wake_fd[0]);
    close((int) wake_fd[1]);
#endif
}

void server_subproc::waiter::wake() {
#ifndef _WIN32
    char c = 1;
    (void) !write((int) wake_fd[1], &c, 1);
#endif
}

void server_subproc::waiter::wait(const std::vector<server_subproc *> & procs, std::vector<bool> & ready, int64_t timeout_ms) {
    ready.assign(procs.size(), false);
#ifdef _WIN32
    // no waitable wait exists for anonymous pipes, so poll them in 50 ms steps
    bool any = false;
    for (size_t i = 0; i < procs.size(); i++) {
        DWORD avail = 0;
        if (!procs[i]->has_output() || !PeekNamedPipe((HANDLE) procs[i]->out_handle, NULL, 0, NULL, &avail, NULL) || avail > 0) {
            ready[i] = true; // data or broken pipe, read_output() tells which
            any = true;
        }
    }
    if (!any) {
        int64_t step = timeout_ms < 0 ? 50 : std::min<int64_t>(timeout_ms, 50);
        std::this_thread::sleep_for(std::chrono::milliseconds(step));
    }
#else
    std::vector<pollfd> pfds;
    pfds.reserve(procs.size() + 1);
    pfds.push_back({ (int) wake_fd[0], POLLIN, 0 });
    for (auto * p : procs) {
        pfds.push_back({ p->has_output() ? (int) p->out_handle : -1, POLLIN, 0 }); // poll() skips negative fds
    }
    int timeout = timeout_ms < 0 ? -1 : (int) std::min<int64_t>(timeout_ms, std::numeric_limits<int>::max());
    int r = poll(pfds.data(), pfds.size(), timeout);
    if (r < 0 && errno != EINTR) {
        LOG_ERR("%s: poll() failed: %s\n", __func__, strerror(errno));
    }
    if (pfds[0].revents) {
        char buf[64];
        while (read((int) wake_fd[0], buf, sizeof(buf)) > 0) {}
    }
    for (size_t i = 0; i < procs.size(); i++) {
        ready[i] = pfds[i + 1].fd < 0 || pfds[i + 1].revents != 0;
    }
#endif
}
