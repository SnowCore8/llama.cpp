// OpenAI Responses API helpers: prepare requests, enrich/store responses

#pragma once

#include "common.h"
#include "server-responses-store.h"

#include "server-common.h"

#include <string>


// Expand previous_response_id into a full input array. Throws std::invalid_argument on errors.
// Also strips previous_response_id from the returned body.
// When vocab/n_ctx_slot are set, truncation uses real tokenizer counts vs the slot context budget.
json server_responses_prepare_request(json body);
json server_responses_prepare_request(
    json body,
    const llama_vocab * vocab,
    int32_t n_ctx_slot);

// POST /v1/responses/compact — CompactedResponse (response.compaction).
json server_responses_compact(json body, const llama_vocab * vocab, int32_t n_ctx_slot);

// Convert a Responses output item into an input item for multi-turn continuation.
json server_responses_output_item_to_input(const json & output_item);

// Normalize Responses "input" (string or array) to an array of input items.
json server_responses_normalize_input(const json & input);

// Build store entry from a completed Responses object + request metadata, then put into store.
void server_responses_remember(
    const json & response_obj,
    const json & prepared_request_input,
    const json & instructions);

// Generate a new response id (resp_...).
std::string server_responses_new_id();

// Next sequence number for a response id (process-wide streaming counter).
int32_t server_responses_next_seq(const std::string & resp_id);

// Reset sequence counter for a response id (call when response completes).
void server_responses_reset_seq(const std::string & resp_id);

// Enrich a completed/non-stream response object with commonly expected OpenAI fields.
json server_responses_enrich_response(json response_obj, const json & request_body);

// Mid-stream tail only: error + response.failed (created/in_progress already sent).
json server_responses_build_error_failed_sse_events(
    const std::string & resp_id,
    const std::string & model,
    const json & request_body,
    const std::string & message,
    const std::string & code = "server_error");

// POST /v1/responses/{id}/cancel — mark stored response cancelled and return it.
json server_responses_cancel(const std::string & response_id);

// GET /v1/responses/{id}/input_items — OpenAI list object of input items.
json server_responses_list_input_items(const std::string & response_id);

// Fold non-user input items into a local. compaction item (used by compact + context_management).
json server_responses_fold_input_compaction(const json & input);

// max_tool_calls from request; returns <0 when unset / unlimited.
int server_responses_max_tool_calls(const json & request_body);

// Effective function_call emit cap: honors max_tool_calls and parallel_tool_calls=false.
// Returns <0 when unlimited.
int server_responses_effective_tool_call_cap(const json & request_body);

// True when request.include contains the given string (e.g. reasoning.encrypted_content).
bool server_responses_include_contains(const json & request_body, const std::string & item);

// Local opaque blob (local. + base64 JSON), same codec as compact encrypted_content.
std::string server_responses_encode_local_blob(const json & obj);
// Inverse of encode; false if not a local. blob or payload is corrupt.
bool server_responses_expand_local_blob(const std::string & enc, json & out);

// Text of the Responses reasoning `summary[]` entry for the accumulated reasoning, per
// request reasoning.summary / generate_summary. Empty when unset.
std::string server_responses_reasoning_summary_text(const std::string & reasoning_text, const json & request_body);

// Build Responses reasoning item `summary[]` from request reasoning.summary /
// generate_summary (auto|concise|detailed). Empty when unset.
json server_responses_reasoning_summary(const std::string & reasoning_text, const json & request_body);

// reasoning.context=current_turn → drop historical reasoning items when expanding.
bool server_responses_drop_history_reasoning(const json & request_body);

// Whether Responses SSE should include local obfuscation payloads.
bool server_responses_want_stream_obfuscation(const json & request_body);

// Add obfuscation field when enabled (mutates event data object).
void server_responses_maybe_obfuscate_event(json & event_data, const json & request_body);
