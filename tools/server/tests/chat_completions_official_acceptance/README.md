# Official OpenAI Chat Completions Stable API Acceptance Tool

## Purpose

Accept or reject a server against the **public OpenAI Chat Completions API**:
create/stream, store CRUD, Models — docs + `openai` Python SDK.

**Out of scope:** hosted cloud tool *execution*.


## Pass rule

Exit `0` only with no `FAIL` / `PARTIAL` / `NOT_IMPLEMENTED`. `SKIP` is allowed.

## Surface

| Resource | Paths |
|----------|-------|
| Chat Completions | `POST /v1/chat/completions`, `…/input_tokens` |
| Store CRUD | list/retrieve/update/delete under `/v1/chat/completions` |
| Models | `GET /v1/models` |

## Server prerequisites

- `--alias` must equal the `--model` value passed to the suite: `GET /v1/models`
  ids and the `model` field in responses echo it.
- `--mmproj` is required by the `image_url_content` check on VLM setups.
- `LLAMA_WEB_SEARCH_FIXTURE=tools/server/tests/fixtures/web_search_fixture.json`
  gives the `web_search_options` checks a deterministic provider. Without it
  they hit the live DuckDuckGo backend, which rate-limits repeated probes and
  drops the citations (`provider=none`).
- `--openai-files-path` backs the store CRUD checks; start from a clean
  directory. Set `LLAMA_OPENAI_FILES_PATH` to the same path so the suite reads
  the server state (prompt cache checks).

Example:

```bash
LLAMA_WEB_SEARCH_FIXTURE=tools/server/tests/fixtures/web_search_fixture.json \
build/bin/llama-server \
  -m Qwen3.5-9B-Q4_K_M.gguf --mmproj mmproj-F16.gguf \
  --alias Qwen3.5-9B-Q4_K_M \
  --api-key sk-1234567890 \
  --openai-files-path /tmp/oai-chat-run
```

## Usage

```bash
python -m chat_completions_official_acceptance \
  --base-url http://127.0.0.1:8080 \
  --api-key sk-1234567890 \
  --model Qwen3.5-9B-Q4_K_M \
  --extra-json '{"chat_template_kwargs":{"enable_thinking":false}}' \
  --report-json /tmp/chat-official-report.json
```

See `../OFFICIAL_API_SCOPE.md`.
