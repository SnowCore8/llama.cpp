# Official OpenAI Stable API Acceptance Tool (Responses)

## Purpose

Accept or reject a server against the **public OpenAI stable API**:
Responses (+ helpers), Completions, Models — using docs + `openai` Python SDK.

**Out of scope:** Codex wire shapes; `client.beta.*` platform extras;
hosted cloud tool *execution*.


Normative sources (priority order):

1. [OpenAI Responses API reference](https://platform.openai.com/docs/api-reference/responses)
2. Completions + Models references
3. Types & resources in the `openai` Python SDK

## Pass rule

Exit `0` only with no `FAIL` / `PARTIAL` / `NOT_IMPLEMENTED`. `SKIP` is allowed.

## Surface

| Resource | Paths |
|----------|-------|
| Responses | `POST/GET/DELETE /v1/responses…`, cancel, compact, input_items, input_tokens |
| Completions | `POST /v1/completions` (+ stream) |
| Models | `GET /v1/models` (OpenAI `Model` shape) |

Also: create-param catalog, forced SSE, live SDK (`responses.create/stream/parse`,
`models.*`, `completions.create`), image `input_image`, `tool_choice`
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
  ├─ checks_streaming.py
  ├─ checks_scenarios.py
  ├─ checks_semantics.py
  ├─ checks_prompt_cache.py
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
```

Environment overrides: `RESPONSES_BASE_URL`, `RESPONSES_API_KEY`,
`RESPONSES_MODEL`, `RESPONSES_EXTRA_JSON`.

Prompt-template checks (`prompt.id`) need `pmpt_local_*.json` under the server's
`--openai-files-path/prompts/`. The runner installs them from `../fixtures/prompts/`
into `$LLAMA_OPENAI_FILES_PATH/prompts/` (or `/tmp/llama-openai-files/prompts/`
when that directory already exists); set `LLAMA_OPENAI_FILES_PATH` to the server's
`--openai-files-path` so the fixtures land where the server reads them.

Requires `openai>=2.53` (and `websockets`) from `tools/server/tests/requirements.txt`.
