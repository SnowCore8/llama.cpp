// Chat conversion functions for server (Responses API, Anthropic API, OAI streaming diffs)

#pragma once

#include "chat.h"
#include "server-common.h"
#include "server-http.h"
#include "server-task.h"

#include "json.h"

#include <unordered_set>

// Parse Anthropic request to server_surface_request
// Generates complete server_surface_request including prompt, files, params, and surface payload
// Surface-private data goes into result.surface
server_surface_request parse_anthropic_to_surface_request(
    json & body,
    const server_chat_params & opt,
    const llama_vocab * vocab,
    const common_params & params_base,
    const std::vector<llama_logit_bias> & logit_bias_eog,
    std::vector<raw_buffer> & out_files);

// Parse Responses request to server_surface_request
// Generates complete server_surface_request including prompt, files, params, and surface payload
// prev_response_id, conv_input, web search data, and ws_token are passed directly (no transport keys)
server_surface_request parse_responses_to_surface_request(
    json & body,
    const server_chat_params & opt,
    const llama_vocab * vocab,
    const common_params & params_base,
    const std::vector<llama_logit_bias> & logit_bias_eog,
    std::vector<raw_buffer> & out_files,
    const std::string & prev_response_id = "",
    const json & conv_input = nullptr,
    bool web_search_enabled = false,
    const std::string & web_search_query = "",
    const json & web_search_results = nullptr,
    const json & web_search_actions = nullptr,
    int web_search_n_requests = 0,
    const std::string & ws_token = "");

// convert OpenAI transcriptions API format to OpenAI Chat Completions API format
json convert_transcriptions_to_chatcmpl(
    const json & body,
    const common_chat_templates * tmpls,
    const std::map<std::string, uploaded_file> & in_files,
    std::vector<raw_buffer> & out_files);

json server_chat_msg_diff_to_json_oaicompat(const common_chat_msg_diff & diff,
        const std::unordered_set<size_t> & custom_tool_call_indices = {});

// true when `name` was declared as a 'custom' tool in the request
bool server_chat_is_custom_tool_name(const std::string & name, const std::vector<std::string> & custom_tool_names);

// indices of tool calls that must serialize in the official custom shape
std::unordered_set<size_t> server_chat_custom_tool_call_indices(
        const common_chat_msg & msg, const std::vector<std::string> & custom_tool_names);

// input string of a custom tool call: raw text, JSON string or object {"input": ...}
std::string server_chat_custom_tool_input(const std::string & arguments);

// rewrite tool_calls of a serialized Chat message into the official custom shape
void server_chat_apply_custom_tool_calls(json & message_obj, const std::vector<std::string> & custom_tool_names);
