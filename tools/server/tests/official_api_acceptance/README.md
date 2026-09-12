# Official API acceptance meta-runner

Sequential strict acceptance for two public API surfaces against a llama.cpp server (or compatible proxy):

| Suite | Package | Normative API |
|-------|---------|---------------|
| OpenAI Responses | `responses_official_acceptance` | `/v1/responses` + Completions + Models + openai Python SDK |
| OpenAI Chat Completions | `chat_completions_official_acceptance` | `/v1/chat/completions` + openai Python SDK |

## Run all suites

From `tools/server/tests/` (with venv activated):

```bash
python -m official_api_acceptance \
  --base-url http://127.0.0.1:8080 \
  --api-key sk-1234567890 \
  --model Qwen3.5-9B-Q4_K_M \
  --report-dir /tmp/official_api_reports
```

Exit code is `0` only when **every present suite** exits `0`. A missing `chat_completions_official_acceptance` package counts as FAIL.

Reports written to `--report-dir`:

- `responses.json`
- `chat_completions.json`
- `summary.json`

Individual suites can still be run directly, e.g. `python -m responses_official_acceptance`.
