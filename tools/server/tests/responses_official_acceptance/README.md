# Official OpenAI Stable API Acceptance Tool (Responses)

## Purpose

Accept or reject a server against the **public OpenAI stable API**:
Responses (+ helpers), Conversations, Completions, Models - using docs +
`openai` Python SDK.

**Out of scope:** Codex wire shapes; `client.beta.*` platform extras;
hosted cloud tool *execution*.


Normative sources (priority order):

1. [OpenAI Responses API reference](https://platform.openai.com/docs/api-reference/responses)
2. Conversations, Completions + Models references
3. Types & resources in the `openai` Python SDK

## Pass rule

Exit `0` only with no `FAIL` / `PARTIAL` / `NOT_IMPLEMENTED`. `SKIP` is allowed.

## Surface

| Resource | Paths |
|----------|-------|
| Responses | `POST/GET/DELETE /v1/responses…`, cancel, compact, input_items, input_tokens |
| Conversations | `POST/GET/DELETE /v1/conversations…`, update, items sub-resource (add/list/get/delete) |
| Completions | `POST /v1/completions` (+ stream) |
| Models | `GET /v1/models` (OpenAI `Model` shape) |

Also: create-param catalog, forced SSE, live SDK (`responses.create/stream/parse`,
`models.*`, `completions.create`, `conversations.*`), image `input_image`, `tool_choice`
`{auto,required,function,none}`, `reasoning.effort=none`.

## Design principles

| Principle | Meaning |
|-----------|---------|
| Catalog-first | Every official HTTP resource and every `ResponseCreateParams` field appears in the report |
| Schema-first | Successful JSON bodies validated with SDK Pydantic models |
| Honest outcomes | Missing routes are `NOT_IMPLEMENTED`, not silently skipped |
| No client favoritism | Codex wire shapes are **not** acceptance criteria |
| Deterministic CLI | One command, JSON + text report, CI-friendly exit code |

## Outcome vocabulary

| Status | Meaning |
|--------|---------|
| `PASS` | Behavior matches the public API for an implemented surface |
| `FAIL` | Surface exercised but violates the public API or SDK schema |
| `NOT_IMPLEMENTED` | Official route/feature probed; server returns 404/501 or equivalent absence |
| `PARTIAL` | Request accepted but semantics incomplete vs OpenAI |
| `SKIP` | Probe not applicable (rare; helper SDK paths) |

Conditional SSE events are reported when the suite can produce them;
`error` / `response.failed` are not forced, so they end as `SKIP`.

## Architecture

```text
runner.py / __main__.py
  ├─ catalog.py               # Responses + Completions + Models from SDK
  ├─ http_client.py
  ├─ checks_endpoints.py
  ├─ checks_models.py
  ├─ checks_completions.py
  ├─ checks_create_params.py
  ├─ checks_sdk.py            # live openai SDK
  ├─ checks_streaming.py      # SSE events, background resume, logprobs, web_search stream
  ├─ checks_conversations.py  # Conversations + items, response<->conversation membership
  ├─ checks_scenarios.py
  ├─ checks_semantics.py
  ├─ checks_prompt_cache.py
  ├─ checks_ws.py             # WebSocket transport: lanes, stream_id echo, error shapes
  ├─ validators.py
  └─ report.py
```

## Non-goals

- Codex CLI / debug proxy scenarios
- OpenAI cloud object APIs (Files / Batches / Uploads)
- Hosted tool *execution* on OpenAI cloud
- Byte-identical parity with OpenAI cloud
- Starting `llama-server` (assumes a running base URL)

## Usage

```bash
# from tools/server/tests (venv with requirements.txt / openai)
python -m responses_official_acceptance \
  --base-url http://127.0.0.1:8080 \
  --api-key sk-1234567890 \
  --model Qwen3.5-9B-Q4_K_M \
  --extra-json '{"chat_template_kwargs":{"enable_thinking":false}}' \
  --report-json /tmp/responses-official-report.json

# run a subset by suite group (default: all)
python -m responses_official_acceptance --only streaming,conversation
```

`--only` accepts a comma-separated list of groups:
`endpoint`, `models`, `completions`, `create_param`, `sdk`, `streaming`,
`scenario`, `semantic`, `prompt_cache`, `conversation`, `ws`.
`streaming` covers the SSE event checks, `background_stream`, `output_logprobs`
and `web_search_stream`; `conversation` covers the Conversations API and the
response<->conversation membership checks; `ws` covers the Responses WebSocket
transport (default/named lanes, `stream_id` echo + validation, error envelopes,
lane FIFO and the 32-named-stream limit; the 16 in-flight and 60-minute
connection caps are `SKIP`). A crash in one suite is recorded as a
`FAIL` row and later suites still run.

Environment overrides: `RESPONSES_BASE_URL`, `RESPONSES_API_KEY`,
`RESPONSES_MODEL`, `RESPONSES_EXTRA_JSON`, `RESPONSES_ONLY`.

Notable check families:

- `web_search_stream.events` / `web_search_stream.annotation`: streaming
  `response.web_search_call.*` lifecycle order and `url_citation` annotations
  (both `SKIP` when the search provider returns no sources). Point the server at
  `tools/server/tests/fixtures/web_search_fixture.json` via
  `LLAMA_WEB_SEARCH_FIXTURE` for deterministic sources.
- `stream_delta_logprobs.local` and friends: local logprob semantics (`include`
  or `top_logprobs>0` both emit logprobs).
- `items.list.default_limit`: official default page size is 20.

Prompt-template checks (`prompt.id`) need `pmpt_local_*.json` under the server's
`--openai-files-path/prompts/`. The runner installs them from `../fixtures/prompts/`
into `$LLAMA_OPENAI_FILES_PATH/prompts/` (or `/tmp/llama-openai-files/prompts/`
when that directory already exists); set `LLAMA_OPENAI_FILES_PATH` to the server's
`--openai-files-path` so the fixtures land where the server reads them.

Requires `openai>=2.53` (and `websockets`) from `tools/server/tests/requirements.txt`.
