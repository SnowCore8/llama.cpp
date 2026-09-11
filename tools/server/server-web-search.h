// Local deepen for OpenAI-hosted web_search (not a cloud backend).
#pragma once

#include "server-common.h"

#include <string>
#include <vector>


struct server_web_search_options {
    std::string query;
    std::string context_size = "medium"; // low | medium | high
    std::vector<std::string> allowed_domains;
    std::vector<std::string> blocked_domains;
    int  max_results        = 5;
    bool fetch_pages        = false;
    int  fetch_pages_n      = 0;
    bool external_web_access = true;
    std::string user_location; // free-form / country hint
    std::string explicit_query; // overrides extracted user text when set
};

bool server_web_search_is_tool_type(const std::string & type);

// Parse filters / search_context_size / user_location from Responses tool objects
// and Chat web_search_options.
server_web_search_options server_web_search_options_from_body(const json & body);

// Run local search pipeline. Returns:
// {
//   query, queries:[], results:[{title,url,snippet,source,page_text?}],
//   provider, actions:[{type,query?,url?,pattern?,sources?}],
//   n_requests
// }
json server_web_search_run(const server_web_search_options & opt);

// Apply deepen onto request body (strip web_search tools, inject context, set __oai_web_search_*).
void server_web_search_apply(json & body);

// Build Responses output items (web_search_call[+open_page...]) honoring include[].
json server_web_search_responses_output_items(const json & request_body);

// Attach url_citation annotations (+ optional Sources footer) on message output_text parts.
void server_web_search_annotate_responses_output(json & response_obj, const json & request_body);

// Chat Completions: add message.annotations url_citation list when search ran.
void server_web_search_annotate_chat_message(json & message_obj, const json & web_results);
