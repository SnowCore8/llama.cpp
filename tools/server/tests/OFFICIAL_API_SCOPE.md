# Official API Acceptance — Scope & Change Risk

SDK pin (tests venv): `openai==2.54.0` (latest at run time).

**对齐目标：** Responses、OpenAI Completions（`/v1/completions`）均按**官方公开 API 形状**对齐（标准 3：可本地观测的行为加深；云端专用能力合法值接受或形状校验，非法 → 400）。

**硬约束：行为与方法/字段一一对应。** 标为「正向行为」的官方方法或 create 字段，必须产生与该名语义一致的可观测效果（例：`tool_choice` 指定工具名 → 仅允许该工具；`max_output_tokens` → 生成上限）。禁止用别的字段静默改写官方字段语义。仅回显/形状校验项不得冒充正向行为；本地 deepen 须单独标明，不得冒充云端等价。

## Suites

| Suite | Package | Normative surface |
|-------|---------|-------------------|
| Responses | `responses_official_acceptance` | OpenAI Responses + Completions + Models + durable Responses store + live `openai` SDK |
| Chat Completions | `chat_completions_official_acceptance` | Chat Completions (+ input_tokens) + Models + durable Chat store + live `openai` SDK |
| Local durability | `local_durability_acceptance` | Store restart resume/reload, `GET /v1/tools`, `/props.slot_save_path`, compact expand |

Meta: `python -m official_api_acceptance …` (local durability runs **last** and may restart the server).

**Suite expansion policy (this fork):** official acceptance suites are the primary
gate for API alignment. Prefer adding catalog/endpoint/SDK probes over leaving
surfaces as undocumented gaps. Hosted cloud-only backends stay explicit 400;
everything durable/local that clients call through the public SDK should be scored.

## Unable / intentionally non-goal (not blocking)

| Surface | Why |
|---------|-----|
| Files / Uploads / Batches / Moderations / Conversations / cloud QoS (`service_tier`, `inference_geo`, `container`, `user_profile_id`) | Cloud-only surfaces: local simulacra were removed, no local implementation here |
| Other hosted tools (`file_search`, remote `mcp`, `code_interpreter`, `code_execution_*`, …) | No cloud backends — **explicit HTTP 400** (not silent skip). **Exception:** local `web_search` deepen (below). |
| `client.beta.*` product APIs | Platform Beta, not stable local HTTP body contract |
| Byte-identical cloud responses | Local models + templates differ |

**Implemented (local):**
- Chat Completions `store=true` CRUD — durable under `--openai-files-path/chat_completions/` (memory if path unset).
- Hosted cloud tools **explicit reject** — Responses non-`function` tools (except `web_search`/`web_search_preview`), Chat `audio` modalities → 400.
- **Local web_search deepen（完整本地托管形态）** — 多后端：`LLAMA_WEB_SEARCH_FIXTURE` / `LLAMA_WEB_SEARCH_LOCAL_DIR` / `LLAMA_WEB_SEARCH_URL` / DuckDuckGo IA+HTML SERP；无命中时 `provider=none`（不伪造空 stub 结果）；fixture/local 命中后跳过远程 SERP；`search_context_size`→结果数+`open_page`/`find_in_page`；`filters.allowed/blocked_domains`（含 Chat `web_search_options.filters`）；Responses：`web_search_call`（`queries`；`include` 门控 `action.sources` / `results`）+ `url_citation` + 流式 `web_search_call.completed`；Chat：`message.annotations`（含流式终块）。
- MCP / local tools visibility — `GET /v1/tools` (+ `?format=openai`); collision `warnings`.
- **Responses store** — durable under `--openai-files-path/responses/` for `previous_response_id` / retrieve/delete; interrupted `in_progress`/`queued` entries finalized as `failed` on restart.
- **Responses `prompt.id`** — file templates under `--openai-files-path/prompts/<id>.json` (`instructions`/`input` + `{{variables}}`); unknown id → 400 (no built-in stub fallback).
- **Responses compact** — local opaque `encrypted_content` (`local.` + base64 JSON of folded non-user items); compact result is **stored** and expandable on subsequent `previous_response_id` (not cloud crypto).
- **Restart durability (local)** — under `--openai-files-path`: interrupted Responses (`in_progress`/`queued`/`cancelling`) → `failed` + `error.code=server_restart`; completed Responses / Chat Completions entries reload; prompt-cache key TTLs are read back from `prompt_cache_keys/`.
- **Acceptance coverage** — `local_durability_acceptance` asserts `/v1/tools`, `/props.slot_save_path`, Slot KV save/restore/erase, compact expand, and the restart matrix above.
- **Responses create deepen (local behavior)** — `max_tool_calls` truncates tool outputs (`incomplete`/`max_tool_calls`); `context_management` compaction auto-folds long history **and expands `local.` back into real messages for the model** (not a placeholder); `stream_options.include_obfuscation` toggles SSE `obfuscation`; `background` async + retrieve to terminal.
- Slot KV persistence — `--slot-save-path`; startup script enables by default; acceptance exercises `/slots/{id}?action=save|restore|erase`.

### Responses create fields: real behavior vs echo-only

| Kind | Fields |
|------|--------|
| **正向行为** | `model`, `input`（`input_text`/`input_image`）, `instructions`, `previous_response_id`, `store`, `stream`, `temperature`/`top_p`/`max_output_tokens`, `tools`/`tool_choice`/`parallel_tool_calls`（`false` 时 emit 最多 1 个 function_call；`type=web_search`→本地搜索 + `web_search_call`）, `text`（`verbosity` hint；`format`→`response_format`/grammar：`json_object`/`json_schema`）, `reasoning`（`effort`→thinking + **budget 阶梯** minimal…max；`context=current_turn`；`summary`/`generate_summary`；`mode=pro`）, `background`, `max_tool_calls`, `context_management`, `stream_options.include_obfuscation`, `include`（`message.output_text.logprobs`；`reasoning.encrypted_content`→`local.` blob）, `truncation`, `metadata`/`user`/`safety_identifier`, `prompt_cache_key`（本地 KV 前缀缓存 + slot 亲和）, `prompt_cache_retention`/`prompt_cache_options`（`cache_prompt`；`implicit` 自动 key；`ttl=5m`/`30m`/`1h`/`24h`；`explicit` 须 key）, `prompt.id`（`--openai-files-path/prompts/<id>.json` 模板展开，未知 id → 400） |

### Chat Completions create fields (shared validators)

| Kind | Fields |
|------|--------|
| **正向行为（Chat）** | `store`+metadata retrieve；`n` 多 choice；`stream_options.include_usage`；`seed` 确定性；`frequency_penalty`/`presence_penalty` 范围 [-2,2]；`modalities=["text"]`；`audio`/含 `audio` 的 modalities → 400；`web_search_options`→本地搜索注入 system + `message.annotations`(`url_citation`)；`user`/`safety_identifier` 在 `store=true` 时持久化 retrieve；legacy `functions`+`function_call`（`none`/`auto`/`{name}`→`tools`/`tool_choice`，命名强制出该 tool）；`reasoning_effort`（同 Responses budget 阶梯）；`verbosity` low/medium/high system hint；`prediction.type=content` 预填 assistant；`prompt_cache_key`/`prompt_cache_retention`/`prompt_cache_options`（同 Responses）；`logprobs`+`top_logprobs`（0..20，超出 → 400；`content[]` 形状） |
| **形状校验** | 非法 `prompt_cache_*` → 400 |
| **正向补充** | `logit_bias` 负偏置抑制目标 token（`/tokenize` 取 id） |

### OpenAI Completions (`/v1/completions`)

| Kind | Fields |
|------|--------|
| **正向行为** | `model`/`prompt` required；`stream`；采样参数经 schema；`n` → 多 completion；`user` 回显；`echo=true` 时 choice.text 含 prompt 前缀；`best_of`≥`n`（生成 `best_of` 候选，**按 token logprob 求和排名**后返回前 `n`）；非空 `suffix` → **FIM**（有 FIM vocab 走 `format_prompt_infill`，否则 soft FIM prompt）；`seed` 同值确定性；`logit_bias` 负偏置抑制目标 token；`logprobs`→官方 Completions 形状（`tokens`/`token_logprobs`/`top_logprobs`/`text_offset`，非 Chat `content[]`；取值 0..5，超出 → 400） |
| **形状校验** | `user`/`stream_options` 类型校验 |

Inference create/stream/parse, create-param catalogs, tools/`tool_choice`, reasoning/thinking off **and** non-`none` `reasoning_effort`/`reasoning.effort` → local thinking, images (where mmproj), Responses store/cancel/compact, Chat Completions store CRUD remain **in scope**.

**Local complete (not cloud):** `--reasoning-preserve` on templates without `supports_preserve_reasoning` folds prior `reasoning_content` into message `content` as `<think>…</think>` before templating (Qwen etc.).

## Server changes for this acceptance pass (risk)

| Change | Risk | Side effects | Rationale |
|--------|------|--------------|-----------|
| Require `model` on `/v1/chat/completions` and `/v1/completions` when missing → 400 | **Medium** | Clients that omit `model` and rely on single-model default break | Matches OpenAI public API; Responses already require `model` |
| Completions `best_of>n` + `stream=true` → 400 | **Low** | Streaming clients that previously got unranked first-candidate SSE now get invalid_request | Matches OpenAI: ranking needs all candidates |
| Background Responses cancel vs fail race | **Low** | Exception path no longer overwrites `cancelled` | Cancel remains sticky once set |
| `prompt_cache_key` slot isolation | **Medium** | Different affinity keys skip LCP reuse and clear the slot KV (the outgoing state is parked in the level-2 cache under its own key first) | Prevents cross-session `cache_read` leakage |
| Key switch parks then restores KV (the drop moved before the level-2 load) | **Low** | A changed explicit key saves the outgoing state, clears the slot and only then loads the state of the incoming key; the drop used to run after the load, so every switch re-processed the whole prompt while the counters still reported a hit | Rotation across more keys than slots restores the parked prefix (`prompt_cache_key_rotation_restore`) |
| Compaction expand + empty web_search | **Medium** | Model sees folded history; no fake `about:local-web-search` hits | Removes placeholder/minimal deepen paths |
| Responses `truncation=auto` tokenizer vs n_ctx + prompt_cache disk TTL clear KV | **Medium** | Oversized input drops oldest by real `common_tokenize` vs slot `n_ctx`; expired `prompt_cache_keys/*.json` clears slot KV before touch refresh | Replaces chars/4 soft budget |
| Completions `best_of` rank order + Responses `text.verbosity` length | **Low–Medium** | `best_of>n` returns choices sorted by Σ logprob; verbosity low≪high | Removes echo-only / weakly-asserted deepen |
| Chat `prompt_cache` warm, penalties, soft FIM expand, Responses `json_schema` | **Low–Medium** | Chat cache hits; banana-count penalty; suffix raises `prompt_tokens`; schema-constrained JSON | Closes remaining local deepen gaps labeled 正向行为 |
| Retention TTL + implicit cache warm (mirrored on Responses) | **Low–Medium** | Chat/Responses registry `expires_at`; cold→warm `cached_tokens` without explicit key; stop text=`1..6` | Closes remaining cache/stop echo-only deepen gaps |
| Completions temp0/stop/penalties | **Low–Medium** | Completions temp0 identical; stop `fr=stop` vs ctrl has 7/10; banana/animal penalties | Closes Completions sampling minimal paths |
| Chat stop/top_p + Completions max_tokens/echo catalog deepen | **Low** | Chat stop=["7"] swallows marker vs ctrl; top_p type 400; Completions `max_tokens=2`→`length` + `echo=true` prefix | Closes remaining Chat/Completions catalog weak rows |
| Hosted cloud tools → HTTP 400 (was silent skip); **web_search local deepen** | **Medium** | `web_search` / `web_search_options` no longer 400; outbound Instant Answer when network allows | Observable local hosted-tool shape; other hosted types stay 400 |
| `/v1/tools` + `?format=openai` (+ MCP collision warnings) | **Low** | Discovery/docs only for default; no change to chat inference unless clients opt in | Agent clients can wire local MCP without silent loss |
| `GET /props` → `slot_save_path` | **Low** | Additive field for discovery | Clarifies whether slot KV persistence is enabled |

## Deepen campaign status (local hosted APIs)

**Done (observable, acceptance-backed):** Responses truncation/TTL/compaction/web_search/verbosity/json_schema/`prompt_cache_*`；Completions echo/`best_of`/FIM/logprobs/temp0/stop/penalties/`max_tokens` cap；Chat store/cache/stop/top_p/retention/implicit/ttl/verbosity/penalties/prediction。

**Intentionally not deepen (still shape/echo or non-goal):** hosted `file_search`/`mcp`/`code_interpreter`（显式 400）；Files/Uploads/Batches/Conversations/Moderations 与云端 QoS 字段（`service_tier`/`inference_geo`/`container`/`user_profile_id`）无本地实现（忽略或拒绝）；byte-identical cloud payloads；scenario 字符串匹配 `PARTIAL`（模型表述噪声，非 API 最小实现）；云端 ramp-rate 计费降级（本地无 TPM 计量，不模拟）；soft FIM（无 FIM vocab 时的 prompt 加深，非真 infill token）；`local.` compaction blob（非云端加密）。

**Stub inventory dissolved (this fork):** `prompt.id` file templates only；web_search empty → `provider=none`。

When the rows above stay green, the “消解最小化实现” campaign for in-scope 正向行为 is **complete**.

## Suite-only fixes (no server risk)

- OpenAI SDK 2.54 `responses.delete` returns `None` — assert delete + retrieve 404, not `deleted=true` object
- Chat `max_tokens` is optional on OpenAI — do not require 400 when omitted
- Completions stream markers avoid model “safety” refusals (`/no_think` + short sentinel)
- Document `betas` as ignored stable-surface header (PASS)
- Chat stream `delta.reasoning_content` forced via `reasoning_effort=low`
- Responses/Chat `verbosity` / `prediction` / Completions `echo`/`best_of`(logprob 排名) / `suffix` FIM 已正向行为
- Responses `prompt.id` 本地模板展开；`prompt_cache_*` 本地调度/亲和语义
