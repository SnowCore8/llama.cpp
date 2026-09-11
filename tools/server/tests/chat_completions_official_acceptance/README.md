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
