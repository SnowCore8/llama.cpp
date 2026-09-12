# Official API Acceptance — Scope & Change Risk

SDK pin (tests venv): `openai==3.11.0` (latest at run time).

**对齐目标：** Responses、OpenAI Completions（`/v1/completions`）均按**官方公开 API 形状**对齐（标准 3：可本地观测的行为加深；云端专用能力合法值接受或形状校验，非法 → 400）。

**硬约束：行为与方法/字段一一对应。** 标为「正向行为」的官方方法或 create 字段，必须产生与该名语义一致的可观测效果（例：`tool_choice` 指定工具名 → 仅允许该工具；`max_output_tokens` → 生成上限）。禁止用别的字段静默改写官方字段语义。仅回显/形状校验项不得冒充正向行为；本地 deepen 须单独标明，不得冒充云端等价。

## Suites

| Suite | Package | Normative surface |
|-------|---------|-------------------|
| Responses | `responses_official_acceptance` | OpenAI Responses + Conversations + Completions + Models + durable Responses store + live `openai` SDK |
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
| Files / Uploads / Batches / Moderations / cloud QoS (`service_tier`, `inference_geo`, `container`, `user_profile_id`) | Cloud-only surfaces: no local implementation here (`service_tier` accepts the official enum and is echoed, `fast`→`priority` like the official response, but changes no scheduling; the rest are ignored or rejected) |
| `DELETE /v1/models/{model}` | Cloud fine-tune-only surface: local servers have no fine-tuned models to delete (retrieve/list are implemented) |
| Other hosted tools (`file_search`, remote `mcp`, `code_interpreter`, `code_execution_*`, …) | No cloud backends — **explicit HTTP 400** (not silent skip). **Exception:** local `web_search` deepen (below). |
| `client.beta.*` product APIs | Platform Beta, not stable local HTTP body contract |
| Byte-identical cloud responses | Local models + templates differ |

**Implemented (local):**
- Models — `GET /v1/models/{model}` retrieve returns the loaded model object by id (`model_name`; aliases are not accepted), 404 otherwise; every model object carries `shutdown_date: null` (local models never shut down). Router mode serves the same route from the model store (`hidden` cache entries → 404).
- Chat Completions `store=true` CRUD — durable under `--openai-files-path/chat_completions/` (memory if path unset); stored messages at `GET /v1/chat/completions/{id}/messages` (`after`/`limit`/`order`, default `asc`; one row per choice with a stable `msg_` id).
- Hosted cloud tools **explicit reject** — Responses non-`function` tools (except `web_search`/`web_search_preview`), Chat `audio` modalities → 400.
- **Local web_search deepen（完整本地托管形态）** — 多后端：`LLAMA_WEB_SEARCH_FIXTURE` / `LLAMA_WEB_SEARCH_LOCAL_DIR` / `LLAMA_WEB_SEARCH_URL` / DuckDuckGo IA+HTML SERP；无命中时 `provider=none`（不伪造空 stub 结果）；fixture/local 命中后跳过远程 SERP；`search_context_size`→结果数+`open_page`/`find_in_page`；`filters.allowed/blocked_domains`（含 Chat `web_search_options.filters`）；Responses：`web_search_call`（`queries`；`include` 门控 `action.sources` / `results`）+ `url_citation` + 流式 `output_item.added` → `web_search_call.in_progress` → `.searching` → `.completed` → `output_item.done`，`output_text.done` 之前每个 citation 一条 `output_text.annotation.added`，`content_part`/`output_item` done 事件携带最终 annotations；Chat：`message.annotations`（含流式终块）。
- MCP / local tools visibility — `GET /v1/tools` (+ `?format=openai`); collision `warnings`.
- **Responses store** — durable under `--openai-files-path/responses/` for `previous_response_id` / retrieve/delete; interrupted `in_progress`/`queued` entries finalized as `failed` on restart.
- **Conversations API** — conversation objects + ordered items, durable under `--openai-files-path/conversations/` (memory if path unset, shares `--responses-store-max`/`--responses-store-ttl`); eight endpoints (create/retrieve/update/delete, items add/list/get/delete, list defaults `limit=20`+`order=desc`, 20-item add cap); `include` gating uses the official 8-value enum (any other value → 400; array encodings `include[]=a&include[]=b`, repeated `include=a&include=b`, or one comma-separated value all accepted), item ids fall back to the `item_` prefix for types without a dedicated one (`msg_`/`fc_`/`fco_`/`rs_`), and with the store disabled (`--responses-store-max 0`) write endpoints → 501 while reads simply miss (404); Responses create `conversation` prepends the stored items and appends the finished turn back (append survives `store=false`; cancelled/failed turns are not appended).
- **Responses `prompt.id`** — file templates under `--openai-files-path/prompts/<id>.json` (`instructions`/`input` + `{{variables}}`); unknown id → 400 (no built-in stub fallback).
- **Responses compact** — local opaque `encrypted_content` (`local.` + base64 JSON of folded non-user items); compact result is **stored** and expandable on subsequent `previous_response_id` (not cloud crypto).
- **Restart durability (local)** — under `--openai-files-path`: interrupted Responses (`in_progress`/`queued`/`cancelling`) → `failed` + `error.code=server_restart`; completed Responses / Chat Completions entries reload; prompt-cache key TTLs are read back from `prompt_cache_keys/`.
- **Acceptance coverage** — `local_durability_acceptance` asserts `/v1/tools`, `/props.slot_save_path`, Slot KV save/restore/erase, compact expand, and the restart matrix above.
- **Responses create deepen (local behavior)** - `max_tool_calls` drops calls beyond the cap (excess attempts ignored, no `incomplete` marker); `context_management` compaction auto-folds long history **and expands `local.` back into real messages for the model** (not a placeholder); `stream_options.include_obfuscation` toggles SSE `obfuscation`（默认开启、对齐官方默认；`false` 关闭）; `background` async + retrieve/cancel to terminal + resumable stream (`GET /v1/responses/{id}?stream=true` with optional `starting_after` cursor; a resume without cursor follows from the oldest whole event still retained; 404 without a session, 400 when the replay prefix was dropped, and a following client whose replay window is evicted gets a terminal SSE `error` event instead of a silent end); `conversation` membership (prepend stored items, append the finished turn); assistant message `phase` (`commentary`/`final_answer`, invalid → 400).
- **Streaming phases (local)** - no queue stage: the official `response.queued` event is never emitted, `response.created`/`response.in_progress` are followed directly by generation events.
- Slot KV persistence — `--slot-save-path`; startup script enables by default; acceptance exercises `/slots/{id}?action=save|restore|erase`.

### Responses create fields: real behavior vs echo-only

| Kind | Fields |
|------|--------|
| **正向行为** | `model`, `input`（仅 `input_text`/`input_image`，其余 union 成员（`input_file`/`input_audio`/`item_reference` 等）→ 400；assistant 消息可选 `phase`=`commentary`/`final_answer`，非法值 → 400）, `instructions`, `previous_response_id`, `conversation`（会话 items 前置 + 本轮回合追加；与 `previous_response_id` 互斥）, `store`, `stream`, `temperature`/`top_p`/`max_output_tokens`, `tools`/`tool_choice`/`parallel_tool_calls`（`false` 时 emit 最多 1 个 function_call；`type=web_search`→本地搜索 + `web_search_call`；`type=allowed_tools`→子集过滤，`mode=required` 无匹配 → 400）, `text`（`verbosity` hint；`format`→`response_format`/grammar：`json_object`/`json_schema`）, `reasoning`（`effort`→thinking + **budget 阶梯** minimal…max；`context=current_turn`；`summary`/`generate_summary`；`mode=pro`）, `background`, `max_tool_calls`, `context_management`, `stream_options.include_obfuscation`, `include`（`message.output_text.logprobs`→非流式 `output_text` parts + 流式 delta/done 条目；`reasoning.encrypted_content`→`local.` blob）, `top_logprobs`（单独设置即启用 logprobs 记录并控制 top-N，官方该字段只定义数量上限——见「本地契约」）, `truncation`, `metadata`/`user`/`safety_identifier`, `prompt_cache_key`（本地 KV 前缀缓存 + slot 亲和）, `prompt_cache_retention`/`prompt_cache_options`（`cache_prompt`；`implicit` 自动 key；`prompt_cache_options.ttl`=`5m`/`30m`/`1h`（超集，官方仅 `30m`——见「记录（未修）」）；`comparison_response_id`→`prompt_cache_diagnostics`（`cache_hit`/`cache_miss`+`reason`/`comparison_response_not_found`；未知 id 不返回 HTTP 错误，本地口径见「记录（未修）」）；响应回声补默认 `mode=implicit`/`ttl=30m`（显式值优先）；`prompt_cache_retention`=`in_memory`/`24h`；`explicit` 须 key）, `prompt.id`（`--openai-files-path/prompts/<id>.json` 模板展开，未知 id → 400） |
| **Accept-ignore** | `moderation`（官方 create 字段：本地接受但不运行审核，任何形状都不报错、不生效） |

### Chat Completions create fields (shared validators)

| Kind | Fields |
|------|--------|
| **正向行为（Chat）** | `store`+metadata retrieve；`n` 多 choice；`stream_options.include_usage`（常规 chunk `usage: null` + 终块 usage 统计）；`stream_options.include_obfuscation`（流式 chunk 默认含 `obfuscation`、`false` 关闭）；非流式 assistant message `refusal`（无拒答时 `null`）；`seed` 确定性；`frequency_penalty`/`presence_penalty` 范围 [-2,2]；`modalities=["text"]`；`audio`/含 `audio` 的 modalities → 400；`web_search_options`→本地搜索注入 system + `message.annotations`(`url_citation`)；`user`/`safety_identifier` 在 `store=true` 时持久化 retrieve；legacy `functions`+`function_call`（`none`/`auto`/`{name}`→`tools`/`tool_choice`，命名强制出该 tool）；`reasoning_effort`（同 Responses budget 阶梯）；`verbosity` low/medium/high system hint；`prediction.type=content` 预填 assistant；`prompt_cache_key`/`prompt_cache_retention`/`prompt_cache_options`（同 Responses）；`logprobs`+`top_logprobs`（0..20，超出 → 400；`content[]` 形状） |
| **形状校验** | 非法 `prompt_cache_*` → 400 |
| **正向补充** | `logit_bias` 负偏置抑制目标 token（`/tokenize` 取 id） |

### OpenAI Completions (`/v1/completions`)

| Kind | Fields |
|------|--------|
| **正向行为** | `model`/`prompt` required；`stream`；`stream_options.include_usage`（常规 chunk `usage: null` + 终块 usage 统计）；`stream_options.include_obfuscation`（流式 chunk 默认含 `obfuscation`、`false` 关闭）；采样参数经 schema；`n` → 多 completion；`user` 回显；`echo=true` 时 choice.text 含 prompt 前缀；`best_of`≥`n`（生成 `best_of` 候选，**按 token logprob 求和排名**后返回前 `n`）；非空 `suffix` → **FIM**（有 FIM vocab 走 `format_prompt_infill`，否则 soft FIM prompt）；`seed` 同值确定性；`logit_bias` 负偏置抑制目标 token；`logprobs`→官方 Completions 形状（`tokens`/`token_logprobs`/`top_logprobs`/`text_offset`，非 Chat `content[]`；取值 0..5，超出 → 400） |
| **形状校验** | `user`/`stream_options` 类型校验；`include_obfuscation` 非布尔 → 400 |

Inference create/stream/parse, create-param catalogs, tools/`tool_choice`, reasoning/thinking off **and** non-`none` `reasoning_effort`/`reasoning.effort` → local thinking, images (where mmproj), Responses store/cancel/compact, Chat Completions store CRUD remain **in scope**.

**Local complete (not cloud):** `--reasoning-preserve` on templates without `supports_preserve_reasoning` folds prior `reasoning_content` into message `content` as `<think>…</think>` before templating (Qwen etc.).

**记录（未修）:** `reasoning_effort` / `reasoning.effort` 原值透传进 `chat_template_kwargs`; 模板自带词表时会拒绝: Qwen3.8-27B 模板只接受 `xhigh`/`medium`/`low`，`minimal`/`high`/`max` 触发模板异常 -> HTTP 500（Chat 与 Responses 均已复现）。未做 OpenAI 到模板词表的映射; 客户端可改用 `chat_template_kwargs.reasoning_effort` 传模板原生值。审计接受为本地契约/声明修正：Responses SSE `obfuscation` 为本地偏离（官方稳定版仅 `response.shell_call_command.delta` 定义该字段；本地对所有 SSE 事件 data 对象注入，默认开、`stream_options.include_obfuscation=false` 关闭）；`response.reasoning_text.delta`/`.done` 与 `response.refusal.delta`/`.done` 均不发射（reasoning 只暴露 summary）；`prompt_cache_options.ttl` 为本地超集（官方仅 `30m`，本地另接受 `5m`/`1h`；`24h` 仅属于 `prompt_cache_retention`）；`response.created` 事件不带 `usage` 键（官方示例含 `usage: null`，schema 标 optional）；compact 的 `service_tier` 复用 create 校验（7 值：`auto`/`default`/`flex`/`scale`/`priority`/`fast`/`ultrafast`；官方 compact 文档仅 5 值）；Chat `n` 超过 `n_parallel` → 400（本地容量上限）；Chat `logprobs`+`tools`+`stream` 组合 → 400（官方文档未禁止）；Completions 接受 `best_of == n`（官方文档措辞为 “must be greater than `n`”）。`prompt_cache_diagnostics` 为本地简化口径（不做云端 token 计量）：expected = min(本次/基线 `usage.input_tokens`)，本次复用 = 本次 `input_tokens_details.cached_tokens`；`cached >= expected - 4`（本地重复 prompt 尾部 4 个 token 重评的容差；`expected <= 4` 时容差为 0）-> `cache_hit`，否则 `cache_miss`（`cache_missed_tokens = expected - cached`、`comparison_reusable_tokens` = 基线 `input_tokens`）；`reason` 由基线 vs 本次请求体比较、首中即止（`model` 与存储的基线请求侧 model 比较（响应回声为本地加载名），`service_tier` 缺省与显式 `auto` 归一为同值，其后按 response 回声：model_changed -> prompt_cache_key_changed -> service_tier_changed -> tools_changed -> text_format_changed -> reasoning_effort_changed -> verbosity_changed -> 兜底 input_changed），不产出官方枚举中的 `context_compacted`/`unavailable`；基线无条目或无可用 `usage.input_tokens`（`in_progress` stub / compact 结果）-> `comparison_response_not_found`（请求保持 200，不加载历史、不改变缓存行为）。显式 `prompt_cache_options.ttl`=`5m`/`1h` 按用户值回显，而 openai SDK 3.11.0 该字段为 `Literal["30m"]`，严格校验的 SDK 客户端解析此类响应会失败（本地 ttl 超集的已知代价）。

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
| Responses Conversations API + `conversation` membership | **Low** | Local conversation objects (no cloud counterpart); `conversation` rejects `previous_response_id` and unknown ids with 400; `store=false` still appends the turn | Official memory layer: prepend stored items, append the finished turn |
| Background stream resumable session + cancel | **Low** | Background streams survive client disconnects (`starting_after` replay); cancel also cancels the stream; plain streams get 404 on reattach | Matches official reattach semantics |
| Streaming logprobs when `include` has no `top_logprobs` | **Low** | `include=["message.output_text.logprobs"]` now records the top-1 candidate (`n_probs` stayed 0, so the arrays were always empty) | Bug fix: requested logprobs must carry real values |
| Responses cloud field validation (`metadata`, `safety_identifier`, `service_tier`, `include`, `phase`, `allowed_tools` mode) | **Low** | Out-of-range or unknown values that used to pass silently now return 400 | Enumerable official value spaces; shape errors are client bugs |
| Streaming chunks default to `obfuscation` and `usage: null`; Responses SSE `obfuscation` default on | **Low** | Responses obfuscation, previously off by default, now appears in every SSE event's data object | Aligns with the official default behavior |
| Responses `prompt_cache_options.comparison_response_id` diagnostics + echoed-options defaults | **Low** | Unknown ids stay HTTP 200 and report `comparison_response_not_found`; an echoed `prompt_cache_options` gains `mode`/`ttl` defaults (the SDK requires both) | Local deepen of the official diagnostics field |

## Deepen campaign status (local hosted APIs)

**Done (observable, acceptance-backed):** Responses truncation/TTL/compaction/web_search/verbosity/json_schema/`prompt_cache_*`（含 `comparison_response_id` 诊断）/conversations/background stream resume/streaming logprobs/`phase`/`allowed_tools`；Completions echo/`best_of`/FIM/logprobs/temp0/stop/penalties/`max_tokens` cap/stream `obfuscation`+`usage: null`；Chat store/cache/stop/top_p/retention/implicit/ttl/verbosity/penalties/prediction/stream `obfuscation`+`usage: null`/`refusal`；Responses SSE `obfuscation` 默认开。

**Intentionally not deepen (still shape/echo or non-goal):** hosted `file_search`/`mcp`/`code_interpreter`（显式 400）；Files/Uploads/Batches/Moderations 与云端 QoS 字段无本地实现（`service_tier` 接受官方枚举并回显、`fast`→`priority`，不改执行；`inference_geo`/`container`/`user_profile_id` 忽略或拒绝）；byte-identical cloud payloads；scenario 字符串匹配 `PARTIAL`（模型表述噪声，非 API 最小实现）；云端 ramp-rate 计费降级（本地无 TPM 计量，不模拟）；soft FIM（无 FIM vocab 时的 prompt 加深，非真 infill token）；`local.` compaction blob（非云端加密）。

**Stub inventory dissolved (this fork):** `prompt.id` file templates only；web_search empty → `provider=none`。

When the rows above stay green, the “消解最小化实现” campaign for in-scope 正向行为 is **complete**.

## Suite-only fixes (no server risk)

- OpenAI SDK `responses.delete` returns `None` — assert delete + retrieve 404, not `deleted=true` object
- Chat `max_tokens` is optional on OpenAI — do not require 400 when omitted
- Completions stream markers avoid model “safety” refusals (`/no_think` + short sentinel)
- Document `betas` as ignored stable-surface header (PASS)
- Chat stream `delta.reasoning_content` forced via `reasoning_effort=low`
- Responses/Chat `verbosity` / `prediction` / Completions `echo`/`best_of`(logprob 排名) / `suffix` FIM 已正向行为
- Responses `prompt.id` 本地模板展开；`prompt_cache_*` 本地调度/亲和语义

## Local contracts (not cloud)

- **Conversations error codes** — 400 invalid input (including `include` values outside the official enum), 404 missing conversation/item, 500 write failure, 501 store disabled (`--responses-store-max 0`); the usual `{code, message, type}` error envelope.
- **Responses `top_logprobs` without `include`** — local extension: setting `top_logprobs` alone enables logprob recording; officially the arrays are enabled through `include: ["message.output_text.logprobs"]` and `top_logprobs` only caps the count.
- **Responses output logprobs drop zero-length tokens** — the sampled EOG token carries no text and is not listed in the `output_text` / `output_text.done` logprob entries, so the entries line up with the output text (Chat Completions keeps the upstream behavior).
