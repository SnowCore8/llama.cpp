# Local Durability & Discovery Acceptance Tool

## Purpose

Accept or reject a server against the **fork-local durable surfaces**: store
restart finalize/reload, the `GET /v1/tools` discovery endpoint,
`/props.slot_save_path`, Slot KV save/restore/erase, and compact expand. This
suite is **not** part of the official OpenAI stable surface - it scores
llama-server local extensions.

## Pass rule

Exit `0` only with no `FAIL` / `PARTIAL` / `NOT_IMPLEMENTED`. `SKIP` is allowed.

`GET /health` is a hard gate: anything other than HTTP 200 is recorded as a
`FAIL` and the run stops before the other areas.

## What it covers

| Area | Checks |
|------|--------|
| discovery | `GET /v1/tools` (tool list), `GET /v1/tools?format=openai`, `GET /props slot_save_path`, `slots_save_restore_erase` (Slot KV `save` / `restore` / `erase` via `/slots/{id}`) |
| semantic | `compact_expand_via_previous_response_id` (compact -> stored result -> expand via `previous_response_id`), `chat_store_persists` (chat `store=true` -> retrieve) |
| restart | `server_restart`, `fixtures_planted`, `responses_in_progress_to_failed`, `responses_completed_reload_prev_id`, `chat_completion_store_reload`, `prompt_cache_dropped_after_restart` |

The restart matrix plants an interrupted `in_progress` response in the store
root and seeds a completed Response + a stored Chat completion through the API;
after the kill/restart it asserts the interrupted entry became `failed` with
`error.code=server_restart`, the completed entries reload (`previous_response_id`
continuation still resolves), and prompt-cache KV warmed before the restart is
not reused after it (`SKIP` when no pre-restart reuse was observed).

## Requirements

- A running `llama-server` with a Responses-capable model (default
  `Qwen3.5-9B-Q4_K_M`), started with:
  - `--openai-files-path <dir>` for the durable stores; set
    `LLAMA_OPENAI_FILES_PATH` to the same directory for the suite process
    (default `/tmp/llama-openai-files`). Memory-only servers fail the restart
    checks.
  - `--slot-save-path <dir>`; the `GET /props slot_save_path` discovery row
    fails without it and `slots_save_restore_erase` is `SKIP`.
  - Built-in tools (`--tools`) or MCP servers (`--mcp-servers-config` /
    `--mcp-servers-json`) so `/v1/tools` is registered; without either the
    route answers 403 and the discovery rows fail.
- For the restart checks, `LLAMA_SERVER_RESTART_CMD` must hold a bash command
  that brings `llama-server` back up with the same port and store flags;
  without it the `server_restart` row is `SKIP`.
- Python venv with `tools/server/tests/requirements.txt`, run from
  `tools/server/tests/` (the suite imports the sibling
  `responses_official_acceptance` package).

Example server start:

```bash
# durable stores, slot persistence and tools for the local acceptance run
build/bin/llama-server \
  -m Qwen3.5-9B-Q4_K_M.gguf \
  --alias Qwen3.5-9B-Q4_K_M \
  --api-key sk-1234567890 \
  --openai-files-path /tmp/oai-local \
  --slot-save-path /tmp/oai-slots \
  --tools all
```

## Environment

| Variable | Default | Meaning |
|----------|---------|---------|
| `LOCAL_DURABILITY_BASE_URL` | `http://127.0.0.1:8080` | `--base-url`; falls back to `OFFICIAL_API_BASE_URL` |
| `LOCAL_DURABILITY_API_KEY` | `sk-1234567890` | `--api-key`; falls back to `OFFICIAL_API_KEY`, then `OPENAI_API_KEY` |
| `LOCAL_DURABILITY_MODEL` | `Qwen3.5-9B-Q4_K_M` | `--model`; falls back to `OFFICIAL_API_MODEL` |
| `LOCAL_DURABILITY_EXTRA_JSON` | - | `--extra-json`; falls back to `OFFICIAL_API_EXTRA_JSON` |
| `LOCAL_DURABILITY_SKIP_RESTART` | - | `1` / `true` / `yes` skips the kill/restart checks |
| `LLAMA_SERVER_RESTART_CMD` | - | bash command that relaunches llama-server (restart checks) |
| `LLAMA_SERVER_RESTART_LOG` | `/tmp/llama-server-restart.log` | log the relaunched server appends to |
| `LLAMA_RESTART_HEALTH_TIMEOUT` | `600` | seconds to wait for `/health` after a restart |
| `LLAMA_OPENAI_FILES_PATH` | `/tmp/llama-openai-files` | store root the restart checks read |

## Usage

```bash
# from tools/server/tests (venv with requirements.txt)
python -m local_durability_acceptance \
  --base-url http://127.0.0.1:8080 \
  --api-key sk-1234567890 \
  --model Qwen3.5-9B-Q4_K_M \
  --extra-json '{"chat_template_kwargs":{"enable_thinking":false}}' \
  --report-json /tmp/local-durability-report.json
```

`--skip-restart` (equivalent to `LOCAL_DURABILITY_SKIP_RESTART=1`) skips the
kill/restart checks, and `--report-json <path>` writes the row table as JSON.
The suite does not kill the server unless `LLAMA_SERVER_RESTART_CMD` is set -
without it, the `server_restart` row is `SKIP` and the post-restart assertions
do not run.

## See also

- `../official_api_acceptance/README.md` - meta-runner; local durability runs
  **last** there and may restart the server.
- `../OFFICIAL_API_SCOPE.md` - scope, evidence and change risk.
