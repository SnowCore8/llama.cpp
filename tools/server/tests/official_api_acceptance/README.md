# Official API acceptance meta-runner

Sequential strict acceptance for the official API acceptance suites against a
llama.cpp server (or compatible proxy):

| Suite | Package | Normative surface |
|-------|---------|-------------------|
| OpenAI Responses | `responses_official_acceptance` | `/v1/responses` + Completions + Models + openai Python SDK |
| OpenAI Chat Completions | `chat_completions_official_acceptance` | `/v1/chat/completions` + openai Python SDK |
| Local durability | `local_durability_acceptance` | Store restart resume/reload, `GET /v1/tools`, `/props.slot_save_path`, compact expand |

The local durability suite runs last: it may kill/restart `llama-server` to
verify durable stores (restart checks need `LLAMA_SERVER_RESTART_CMD`; without
it they are `SKIP`).

## Run all suites

From `tools/server/tests/` (with venv activated):

```bash
python -m official_api_acceptance \
  --base-url http://127.0.0.1:8080 \
  --api-key sk-1234567890 \
  --model Qwen3.5-9B-Q4_K_M \
  --report-dir /tmp/official_api_reports
```

Exit code is `0` only when **every suite** exits `0` (a suite exits `0` only
with no `FAIL` / `PARTIAL` / `NOT_IMPLEMENTED`; `SKIP` is allowed). A suite
package that fails to import is recorded as a `FAIL` row with exit code `1`.

Reports written to `--report-dir`:

- `responses.json`
- `chat_completions.json`
- `local_durability.json`
- `summary.json`

Environment overrides: `OFFICIAL_API_BASE_URL`, `OFFICIAL_API_KEY` (falls back
to `OPENAI_API_KEY`), `OFFICIAL_API_MODEL`, `OFFICIAL_API_EXTRA_JSON`,
`OFFICIAL_API_REPORT_DIR`.

Individual suites can still be run directly, e.g. `python -m responses_official_acceptance`.
