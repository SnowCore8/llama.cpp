# Server tests

Python based server tests scenario using [pytest](https://docs.pytest.org/en/stable/).

Tests target GitHub workflows job runners with 4 vCPU.

Note: If the host architecture inference speed is faster than GitHub runners one, parallel scenario may randomly fail.
To mitigate it, you can increase values in `n_predict`, `kv_size`.

### Install dependencies

`pip install -r requirements.txt`

### Run tests

1. Build the server

```shell
cd ../../..
cmake -B build
cmake --build build --target llama-server
```

2. Start the test: `./tests.sh`

It's possible to override some scenario steps values with environment variables:

| variable                 | description                                                                                    |
|--------------------------|------------------------------------------------------------------------------------------------|
| `PORT`                   | `context.server_port` to set the listening port of the server during scenario, default: `8080` |
| `LLAMA_SERVER_BIN_PATH`  | to change the server binary path, default: `../../../build/bin/llama-server`                         |
| `DEBUG`                  | to enable steps and server verbose mode `--verbose`                                       |
| `N_GPU_LAYERS`           | number of model layers to offload to VRAM `-ngl --n-gpu-layers`                                |
| `LLAMA_CACHE`            | by default server tests re-download models to the `tmp` subfolder. Set this to your cache (e.g. `$HOME/Library/Caches/llama.cpp` on Mac or `$HOME/.cache/llama.cpp` on Unix) to avoid this |

To run slow tests (will download many models, make sure to set `LLAMA_CACHE` if needed):

```shell
SLOW_TESTS=1 ./tests.sh
```

To run with stdout/stderr display in real time (verbose output, but useful for debugging):

```shell
DEBUG=1 ./tests.sh -s -v -x
```

To run all the tests in a file:

```shell
./tests.sh unit/test_chat_completion.py -v -x
```

To run a single test:

```shell
./tests.sh unit/test_chat_completion.py::test_invalid_chat_completion_req
```

Hint: You can compile and run test in single command, useful for local development:

```shell
cmake --build build -j --target llama-server && ./tools/server/tests/tests.sh
```

To see all available arguments, please refer to [pytest documentation](https://docs.pytest.org/en/stable/how-to/usage.html)

### Debugging external llama-server
It can sometimes be useful to run the server in a debugger when invesigating test
failures. To do this, the environment variable `DEBUG_EXTERNAL=1` can be set
which will cause the test to skip starting a llama-server itself. Instead, the
server can be started in a debugger.

Example using `gdb`:
```console
$ gdb --args ../../../build/bin/llama-server \
    --host 127.0.0.1 --port 8080 \
    --temp 0.8 --seed 42 \
    --hf-repo ggml-org/models --hf-file tinyllamas/stories260K.gguf \
    --batch-size 32 --no-slots --alias tinyllama-2 --ctx-size 512 \
    --parallel 2 --n-predict 64
```
And a break point can be set in before running:
```console
(gdb) br server.cpp:4604
(gdb) r
main: server is listening on http://127.0.0.1:8080 - starting the main loop
srv  update_slots: all slots are idle
```

And then the test in question can be run in another terminal:
```console
(venv) $ env DEBUG_EXTERNAL=1 ./tests.sh unit/test_chat_completion.py -v -x
```
And this should trigger the breakpoint and allow inspection of the server state
in the debugger terminal.

### Official OpenAI API acceptance

Standalone harness (not pytest) that accepts a **running** server against the
official OpenAI API surfaces (docs + `openai` Python SDK) and runs the local
durability/discovery checks. Codex wire shapes are intentionally out of scope.
Three suites run through the meta-runner:

- `responses_official_acceptance` - Responses + Conversations + Completions +
  Models + durable Responses store + Responses WebSocket + live `openai` SDK
- `chat_completions_official_acceptance` - Chat Completions + Models + durable
  Chat store + live `openai` SDK
- `local_durability_acceptance` - store restart resume/reload,
  `GET /v1/tools`, `/props.slot_save_path`, compact expand

```shell
# venv with tools/server/tests/requirements.txt (includes openai)
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

.venv/bin/python -m official_api_acceptance \
  --base-url http://127.0.0.1:8080 \
  --api-key sk-1234567890 \
  --model Qwen3.5-9B-Q4_K_M \
  --extra-json '{"chat_template_kwargs":{"enable_thinking":false}}' \
  --report-dir /tmp/official_api_reports
```

Exit `0` only when every suite reports no `FAIL` / `PARTIAL` / `NOT_IMPLEMENTED`
(`SKIP` is allowed). The local durability suite runs last and may kill/restart
`llama-server` to verify durable stores; restart checks need
`LLAMA_SERVER_RESTART_CMD` (a launch script that brings the server back up;
without it these checks are `SKIP`) and can be skipped with
`LOCAL_DURABILITY_SKIP_RESTART=1` (or `--skip-restart` when running that suite
directly).

See [`official_api_acceptance/README.md`](official_api_acceptance/README.md)
for the meta-runner,
[`responses_official_acceptance/README.md`](responses_official_acceptance/README.md),
[`chat_completions_official_acceptance/README.md`](chat_completions_official_acceptance/README.md)
and
[`local_durability_acceptance/README.md`](local_durability_acceptance/README.md)
for the three suites. The scope contract and its differences from the official
OpenAI docs are documented in [`OFFICIAL_API_SCOPE.md`](OFFICIAL_API_SCOPE.md)
and [`OFFICIAL_API_DIFF.md`](OFFICIAL_API_DIFF.md).
