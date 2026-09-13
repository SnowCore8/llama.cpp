# Official API Acceptance — Scope & Change Risk

SDK pin (tests venv): `openai==3.11.0` (latest at run time).

差异分类汇总（本 fork vs OpenAI 公开文档）见 [OFFICIAL_API_DIFF.md](OFFICIAL_API_DIFF.md)；本文档为契约、依据与验收证据的权威源，两文件同步维护。

**对齐目标：** Responses、OpenAI Completions（`/v1/completions`）均按**官方公开 API 形状**对齐（标准 3：可本地观测的行为加深；云端专用能力合法值接受或形状校验，非法 → 400）。

**硬约束：行为与方法/字段一一对应。** 标为「正向行为」的官方方法或 create 字段，必须产生与该名语义一致的可观测效果（例：`tool_choice` 指定工具名 → 仅允许该工具；`max_output_tokens` → 生成上限）。禁止用别的字段静默改写官方字段语义。仅回显/形状校验项不得冒充正向行为；本地 deepen 须单独标明，不得冒充云端等价。

## Suites

| Suite | Package | Normative surface |
|-------|---------|-------------------|
| Responses | `responses_official_acceptance` | OpenAI Responses + Conversations + Completions + Models + durable Responses store + Responses WebSocket (lanes/steer/inject/warmup) + live `openai` SDK |
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
| `client.beta.*` product APIs | Platform Beta, not stable local HTTP body contract（例外：beta Responses WebSocket 的 `response.inject` 事件已实现，见「Local contracts」） |
| Byte-identical cloud responses | Local models + templates differ |

**Implemented (local):**
- Models — `GET /v1/models/{model}` retrieve returns the loaded model object by id (`model_name`; aliases are not accepted), 404 otherwise; every model object carries `shutdown_date: null` (local models never shut down). Router mode serves the same route from the model store (`hidden` cache entries → 404).
- Chat Completions `store=true` CRUD — durable under `--openai-files-path/chat_completions/` (memory if path unset); stored messages at `GET /v1/chat/completions/{id}/messages` (`after`/`limit`/`order`, default `asc`; one row per choice with a stable `msg_` id).
- Hosted cloud tools **explicit reject** — Responses non-`function` tools (except `web_search`/`web_search_preview`), Chat `audio` modalities → 400.
- **Local web_search deepen（完整本地托管形态）** — 多后端：`LLAMA_WEB_SEARCH_FIXTURE` / `LLAMA_WEB_SEARCH_LOCAL_DIR` / `LLAMA_WEB_SEARCH_URL` / DuckDuckGo IA+HTML SERP；无命中时 `provider=none`（不伪造空 stub 结果）；fixture/local 命中后跳过远程 SERP；`search_context_size`→结果数+`open_page`/`find_in_page`；`filters.allowed/blocked_domains`（含 Chat `web_search_options.filters`）；Responses：`web_search_call`（`queries`；`include` 门控 `action.sources` / `results`）+ `url_citation` + 流式 `output_item.added` → `web_search_call.in_progress` → `.searching` → `.completed` → `output_item.done`，`output_text.done` 之前每个 citation 一条 `output_text.annotation.added`，`content_part`/`output_item` done 事件携带最终 annotations；Chat：`message.annotations`（含流式终块）。
- MCP / local tools visibility — `GET /v1/tools` (+ `?format=openai`); collision `warnings`.
- **Responses store** — durable under `--openai-files-path/responses/` for `previous_response_id` / retrieve/delete; interrupted `in_progress`/`queued` entries finalized as `failed` on restart.
- **Conversations API** — conversation objects + ordered items, durable under `--openai-files-path/conversations/` (memory if path unset, shares `--responses-store-max`/`--responses-store-ttl`); eight endpoints (create/retrieve/update/delete, items add/list/get/delete, list defaults `limit=20`+`order=desc`, 20-item add cap); `include` gating uses the official 8-value enum (any other value → 400; array encodings `include[]=a&include[]=b`, repeated `include=a&include=b`, or one comma-separated value all accepted), item ids fall back to the `item_` prefix for types without a dedicated one (`msg_`/`fc_`/`fco_`/`rs_`), and with the store disabled (`--responses-store-max 0`) write endpoints → 501 while reads simply miss (404); Responses create `conversation` prepends the stored items and appends the finished turn back (append survives `store=false`; cancelled/failed turns are not appended).
- **Responses `prompt.id`** — file templates under `--openai-files-path/prompts/<id>.json` (`instructions`/`input` + `{{variables}}`); unknown id → 400 (no built-in stub fallback). A versioned id resolves to `prompts/<id>@<version>.json` (missing version file → 400); object variable values expand per part type (`input_text` splices into the template, `input_image` keeps its part shape, `input_file` → 400); a template placeholder with no matching variable → 400.
- **Responses compact** — local opaque `encrypted_content` (`local.` + base64 JSON of folded non-user items); compact result is **stored** and expandable on subsequent `previous_response_id` (not cloud crypto).
- **Restart durability (local)** — under `--openai-files-path`: interrupted Responses (`in_progress`/`queued`/`cancelling`) → `failed` + `error.code=server_restart`; completed Responses / Chat Completions entries reload; prompt-cache key TTLs are read back from `prompt_cache_keys/`.
- **Acceptance coverage** — `local_durability_acceptance` asserts `/v1/tools`, `/props.slot_save_path`, Slot KV save/restore/erase, compact expand, and the restart matrix above.
- **Responses create deepen (local behavior)** - `max_tool_calls` drops calls beyond the cap (excess attempts ignored, no `incomplete` marker); `context_management` compaction auto-folds long history **and expands `local.` back into real messages for the model** (not a placeholder); `stream_options.include_obfuscation` toggles SSE `obfuscation`（默认开启、对齐官方默认；`false` 关闭）; `background` async + retrieve/cancel to terminal + resumable stream (`GET /v1/responses/{id}?stream=true` with optional `starting_after` cursor; a resume without cursor follows from the oldest whole event still retained; 404 without a session, 400 when the replay prefix was dropped, and a following client whose replay window is evicted gets a terminal SSE `error` event instead of a silent end); `conversation` membership (prepend stored items, append the finished turn); assistant message `phase` (`commentary`/`final_answer`, invalid → 400).
- **Custom tools (Chat + Responses)** — 官方形状全链路：Chat 定义 `{type:"custom",custom:{name,description?,format?}}`、`tool_choice={type:"custom",custom:{name}}` 强制、输出 `{id,type:"custom",custom:{name,input}}`、回放 `tool_calls[].type=="custom"`；Responses 扁平定义 `{type:"custom",name,description?,format?}`、输出 item `{id:"ctc_…",type:"custom_tool_call",status,call_id,input,name}`、流式 `response.custom_tool_call_input.delta`/`.done`、回放 `custom_tool_call`/`custom_tool_call_output`（本地执行口径见「Local contracts」）。
- **Chat `allowed_tools` 嵌套形状** — `{type:"allowed_tools",allowed_tools:{mode:"auto"|"required",tools:[…]}}`：按子集过滤可用工具；`required` 无匹配 → 400；`auto` 过滤后空集 → 400。
- **Chat legacy 与内容类型补全** — `role:"function"` 消息（官方 deprecated 角色）渲染前映射为 `role:"tool"`（同 `developer`→`system` 模式）；assistant `function_call`（legacy）规范化为 `tool_calls` 回放；assistant `content:[{type:"refusal",…}]` 回放；`file` part：`file_data` data URI 接受、`file_id` → 400（本地无 Files API）；`input_audio.id` → 400。
- **Usage reasoning 计数（Chat + Responses）** — Chat `usage.completion_tokens_details` 恒 5 字段（`reasoning_tokens` 真计数、`text_tokens = completion_tokens - reasoning_tokens`、`accepted_prediction_tokens`/`audio_tokens`/`rejected_prediction_tokens` = 0）；Responses `usage.output_tokens_details.reasoning_tokens` 非流式与流式均为真值（计数口径见「Local contracts」）。
- **`stop` 门控（Chat + Completions）** — 官方形状 string 或 ≤4 条 string 数组；数组超 4 条、元素非 string、或类型不在 string|array|null → 400。
- **Embeddings** — 输入校验：`input:null`/`input:[]`/非 string `encoding_format` → 400 `invalid_request_error`；响应项仅官方三键 `embedding`/`index`/`object`（base64 同形）；未开 `--embeddings` → 501，pooling 非 OAI 兼容（如 `none`）→ 400。
- **Streaming phases (local)** - no queue stage: the official `response.queued` event is never emitted, `response.created`/`response.in_progress` are followed directly by generation events.
- Slot KV persistence — `--slot-save-path`; startup script enables by default; acceptance exercises `/slots/{id}?action=save|restore|erase`.

### Responses create fields: real behavior vs echo-only

| Kind | Fields |
|------|--------|
| **正向行为** | `model`, `input`（仅 `input_text`/`input_image`，其余 union 成员（`input_file`/`input_audio`/`item_reference` 等）→ 400；assistant 消息可选 `phase`=`commentary`/`final_answer`，非法值 → 400）, `instructions`, `previous_response_id`, `conversation`（会话 items 前置 + 本轮回合追加；与 `previous_response_id` 互斥）, `store`, `stream`, `temperature`/`top_p`/`max_output_tokens`, `tools`/`tool_choice`/`parallel_tool_calls`（`false` 时 emit 最多 1 个 function_call；`type=web_search`→本地搜索 + `web_search_call`；`type=allowed_tools`→子集过滤，`mode=required` 无匹配 → 400）, `text`（`verbosity` hint；`format`→`response_format`/grammar：`json_object`/`json_schema`）, `reasoning`（`effort`→thinking + **budget 阶梯** minimal…max；`context=current_turn`；`summary`/`generate_summary`；`mode=pro`）, `background`, `max_tool_calls`, `context_management`, `stream_options.include_obfuscation`, `include`（`message.output_text.logprobs`→非流式 `output_text` parts + 流式 delta/done 条目；`reasoning.encrypted_content`→`local.` blob）, `top_logprobs`（单独设置即启用 logprobs 记录并控制 top-N，官方该字段只定义数量上限——见「本地契约」）, `truncation`, `metadata`/`user`/`safety_identifier`, `prompt_cache_key`（本地 KV 前缀缓存 + slot 亲和）, `prompt_cache_retention`/`prompt_cache_options`（`cache_prompt`；`implicit` 自动 key；`prompt_cache_options.ttl`=`5m`/`30m`/`1h`（超集，官方仅 `30m`——见「记录（未修）」）；`comparison_response_id`→`prompt_cache_diagnostics`（`cache_hit`/`cache_miss`+`reason`/`comparison_response_not_found`；未知 id 不返回 HTTP 错误，本地口径见「记录（未修）」）；响应回声补默认 `mode=implicit`/`ttl=30m`（显式值优先）；`prompt_cache_retention`=`in_memory`/`24h`；`explicit` 须 key）, `prompt.id`（`--openai-files-path/prompts/<id>.json` 模板展开，未知 id → 400；`prompt.version` → `prompts/<id>@<version>.json`，缺文件 → 400；对象变量按 part 类型展开：`input_text` 拼接进模板、`input_image` 保留 part 形状、`input_file` → 400；模板占位符无匹配变量 → 400），`prompt_cache_breakpoint`（text/image part 上的字段，形状校验 + 真实断点，见「本地契约」） |
| **Accept-ignore** | `moderation`（官方 create 字段：本地接受但不运行审核，任何形状都不报错、不生效） |

### Chat Completions create fields (shared validators)

| Kind | Fields |
|------|--------|
| **正向行为（Chat）** | `store`+metadata retrieve；`n` 多 choice；`stream_options.include_usage`（常规 chunk `usage: null` + 终块 usage 统计）；`stream_options.include_obfuscation`（流式 chunk 默认含 `obfuscation`、`false` 关闭）；非流式 assistant message `refusal`（无拒答时 `null`）；`seed` 确定性；`frequency_penalty`/`presence_penalty` 范围 [-2,2]；`modalities=["text"]`；`audio`/含 `audio` 的 modalities → 400；`web_search_options`→本地搜索注入 system + `message.annotations`(`url_citation`)；`user`/`safety_identifier` 在 `store=true` 时持久化 retrieve；legacy `functions`+`function_call`（`none`/`auto`/`{name}`→`tools`/`tool_choice`，命名强制出该 tool）；`reasoning_effort`（同 Responses budget 阶梯）；`verbosity` low/medium/high system hint；`prediction.type=content` 预填 assistant；`prompt_cache_key`/`prompt_cache_retention`/`prompt_cache_options`（同 Responses）；`logprobs`+`top_logprobs`（0..20，超出 → 400；`content[]` 形状）；`prompt_cache_breakpoint`（text/image part 上的字段，形状校验 + 真实断点，见「本地契约」） |
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

另两条记录（未修）：SSE `error` / `response.failed` 在本地不可达——错误发生在流开始前时按非流式 HTTP 错误体返回，SSE `error`+`response.failed` 只在流已开始后 reader 出错时产生，默认配置无可控注入点（验收对应行保持 SKIP；`pre_created_error_is_plain_json` 断言非流式错误体形状）；Embeddings `encoding_format` 非 string 的 400 message 透传 nlohmann 异常原文（`[json.exception.type_error.302] ...`），未做官方措辞级映射（官方 message 为自由文本）。

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
| Responses WebSocket transport (lanes, `stream_id` echo, nested error envelope) | **Low** | New `server-responses-ws` module; same create path as HTTP, WS-only error nesting + lane echo; poll-based read with a 60-minute session clock | Official WS mode shape |
| WS steering interrupt (`STOP_TYPE_STEERED`) + WS-only steer events | **Low–Medium** | A steered response stops at the next decode step and ends `incomplete`/`steered`, then an automatic successor runs the queued steer on the same lane | Mid-turn steering |
| WS `response.inject` + connection-local `store=false` continuation cache | **Low** | Bounded in-process LRU for `store=false` responses, readable only with the issuing connection's token; invalid inject schema closes the connection | Official WS continuation + multi-agent inject |
| WS `generate:false` warmup (also HTTP) | **Low** | A completed output-less response for state warmup; no model run, no conversation append; non-boolean `generate` → 400 | Official warm-up field (guide-only) |
| Custom tools end-to-end (Chat + Responses) | **Low–Medium** | New tool type accepted on both APIs; serialized as the official custom shape; templates still see the function form | Official custom tool surface |
| Legacy `function` role mapped to `tool` before templating | **Low** | Deprecated function-result messages render as `tool` (no more 500 on templates that reject unknown roles) | Official deprecated role alias |
| Reasoning token counts in usage details | **Low** | Chat `completion_tokens_details` and Responses `output_tokens_details.reasoning_tokens` carry real counts (`0` when unavailable) | Official usage detail fields |
| `stop` gating (Chat + Completions) | **Low** | Arrays longer than 4 and non-string types now 400 at validation | Official limit |
| Embeddings input validation → 400 (was 500); per-item key set fixed | **Low** | `input: null`/`[]` and non-string `encoding_format` return `invalid_request_error`; the non-official per-item `encoding_format` key is gone | Official error shape + item schema |

## Deepen campaign status (local hosted APIs)

**Done (observable, acceptance-backed):** Responses truncation/TTL/compaction/web_search/verbosity/json_schema/`prompt_cache_*`（含 `comparison_response_id` 诊断）/conversations/background stream resume/streaming logprobs/`phase`/`allowed_tools`；Completions echo/`best_of`/FIM/logprobs/temp0/stop/penalties/`max_tokens` cap/stream `obfuscation`+`usage: null`；Chat store/cache/stop/top_p/retention/implicit/ttl/verbosity/penalties/prediction/stream `obfuscation`+`usage: null`/`refusal`；Responses SSE `obfuscation` 默认开。

**Gap-fill wave folded in:** WS/HTTP `generate:false` 预热、custom 工具全链路（Chat + Responses）、Chat `allowed_tools` 嵌套形状与 legacy 兼容（`function` role、`function_call`、`refusal`、`file`、`input_audio`）、usage reasoning 真计数、`stop` 门控、embeddings 输入校验与项键修正——均已并入「Server changes」表与 Implemented (local) 清单。

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

- **Responses WebSocket** — 错误一律用官方嵌套信封 `{type:"error", status, error:{type, code, message, param}, stream_id?, sequence_number?}`：SSE 扁平 error 帧转发时转换，`sequence_number` 只在错误来自响应流时保留（官方语义）；连接级错误不带。`error.type` 取官方错误词表（`invalid_request_error`/`authentication_error`/`permission_error`/`not_found_error`/`server_error`）：HTTP 转发错误若响应体已是官方值则原样保留，否则按 HTTP 状态映射（400/401/403/404 → 对应类型，其余 → `server_error`）；SSE error 帧转发按 code 映射（`invalid_prompt` → `invalid_request_error`，其余 → `server_error`；`invalid_json`/`unsupported_event_type` 等连接级自产码均为 `invalid_request_error` 类）。WS create 的 `background` 被忽略且不回显（官方 "background is not supported over WebSocket"；本地不报错、照常生成）。本地补充码：`invalid_json`（帧不是合法 JSON 对象）、`unsupported_event_type`（`type` 不是 `response.create`/`response.steer`/`response.inject`；param=`type`）。lane：`stream_id` 1-256 字符 `[A-Za-z0-9_.-]`，命名 lane 的全部事件（含终止与请求级错误）回显 `stream_id`，默认 lane 不带；限额 = 16 个在飞响应（超出排队）/ 32 个命名 lane（第 33 个报 `websocket_stream_limit_reached` 并回显被拒名字）/ 60 分钟连接寿命（`websocket_connection_limit_reached`，按 1s tick 检查）。WS 自产事件（`response.steer.*`、`response.inject.*`）的 `sequence_number` 用连接级计数器（官方 schema 只规定字段存在，未定义编号基准；响应流事件保留 per-response 计数）。
- **Responses WS `store=false` 继续（官方 connection-local cache 的本地实现）** — 进程内 LRU（容量 256，记 `{连接令牌, prepared input, output}`；`server_responses_remember` 对 `store=false` 非后台响应写入）。每条 WS 连接在建立时生成随机连接令牌（`__oai_ws_local`，只存在于内部请求，不出现在任何响应里），WS 层在签发 create 与自动 successor 时覆盖写入该键；缓存条目按写入者令牌归档，读取要求令牌非空且匹配 → 响应只能在**签发它的同一条连接**内作 `previous_response_id` 继续。失败驱逐（官方指南）：同 lane 续写以请求级 4xx/5xx 失败时驱逐被引用的父；跨 lane fork 失败保留父。保留策略为连接级 LRU-256（官方口径为每 lane 保留最近缓存、源 lane 前进或失败时可淘汰父；本地不主动淘汰，为更宽松超集）。未命中语义：WS 请求（同连接或其他连接）→ 嵌套 `error` status 400 code=`previous_response_not_found`（message 用官方措辞）；HTTP 请求（含带伪造 `__oai_ws_local` 的请求）→ 扁平错误信封以同一 code/param/message 返回（type `invalid_request_error`；官方把该码列在 WebSocket mode errors，HTTP 面未单独定义，本地选跨传输一致）。官方依据：WebSocket mode 指南（connection-local cache、uncached → `previous_response_not_found`、同 lane 失败驱逐）。
- **Responses WS steering（本地执行语义）** — 本地无模型内 steer 注入：接受后目标在下一次 decode 步停表（`STOP_TYPE_STEERED`）→ `response.incomplete` + `incomplete_details.reason="steered"` → 自动 successor（继承原请求设置、同 lane、输入=排队 steer 依序）；目标先正常完成则保留 `completed` 后接 successor；`too_many_pending_steers` 本地阈值 32 条/目标；目标以 failed/cancelled 终止时对未提交 steer 回 `successor_creation_failed`。pending 的 `required_input` 逐 function_call 生成 `{type:"function_call_output", call_id, name}`（本地输出项可达仅 function_call/function_call_output）。
- **Responses WS `response.inject`（beta）** — 本地无 multi-agent 等待态：校验提交后回 `response.inject.created`，注入项由目标响应终止时的 successor 承接；失败码仅官方两值（`response_already_completed`/`response_not_found`）；schema 不合规 → 通用 `error`(400) 并关闭连接（官方行为）；不强制 `OpenAI-Beta: responses_multi_agent=v1`（接受任意请求头）。
- **Responses `generate:false` 预热（本地契约）** — HTTP 与 WS 均接受；`generate` 存在时必须为布尔（否则 400 `'generate' must be a boolean`），缺省 true。流式仅发 `response.created` + `response.completed`（无 `in_progress`；created 即携带 completed 状态对象），非流式直接返回该对象；只做 prepare_request 级校验（不跑模板级字段校验），不实现真实加速（不跑推理）。响应被 remember（store 或连接令牌缓存）可作 `previous_response_id` 链式；不追加 conversation 回合（返回与落库对象均不含 `conversation` 键）。WS 层：warmup create 不执行 steer 前置（不吞排队 steer）。
- **Conversations error codes** — 400 invalid input (including `include` values outside the official enum), 404 missing conversation/item, 500 write failure, 501 store disabled (`--responses-store-max 0`); the usual `{code, message, type}` error envelope.
- **`prompt_cache_breakpoint`（本地执行语义）** — 形状在原始 message parts 上校验（媒体重写前）：必须是 object 且 `mode="explicit"`，每请求最多 4 个 part 携带（按 part 计数、按 message 定位）。锚点 `{role, message ordinal}` 经 `llama_params` 传到 server-context，解析为该消息 span 结束后的 token 位置（官方未定义文本内偏移粒度，本地按消息边界锚定），fill loop 在该位置强制 batch break 并豁免 mid-prompt skip 与 min-step 两个 checkpoint 闸门，使 checkpoint 落在标注边界；prompt 末尾的断点不建 checkpoint（近末尾 fallback 覆盖该位置）。Responses 输入转换在 text 与 image part 上转发该字段；形状只校验一次（Chat 与 Responses 共享解析器）。
- **Responses `top_logprobs` without `include`** — local extension: setting `top_logprobs` alone enables logprob recording; officially the arrays are enabled through `include: ["message.output_text.logprobs"]` and `top_logprobs` only caps the count.
- **Responses output logprobs drop empty-text tokens and candidates** — the sampled EOG token carries no text and is not listed in the `output_text` / `output_text.done` logprob entries, so the entries line up with the output text; within each listed entry the inner `top_logprobs` / `top_probs` candidates whose token text is empty (special/control tokens) are dropped, and an entry left with no candidates is dropped as well (Chat Completions keeps the upstream behavior).
- **Custom tools（本地执行语义，Chat + Responses）** — 模板/grammar 面按合成 function schema 走（`{type:"object",properties:{input:{type:"string"}},required:["input"]}`；custom 条目带该 schema 才进 grammar 生成器），序列化时按工具名表翻译回官方 custom 形状（其余工具保持 function 形状）。`custom.format` 接受但不参与生成（`text` 与 `grammar` 形状均被忽略——本地无 custom 输入的语法约束）。回放：assistant `tool_calls[].type=="custom"` 的 free-form `input` 重新包成 `{"input": …}` 再进模板（与模型生成形态一致）。Chat 流式（官方 Chat streaming 未定义 custom 增量形状，本地镜像非流式）：`delta.tool_calls[]` 带 `type:"custom"` 与 `custom.name`/`custom.input` 增量，`input` 为解码后文本，原始 envelope 片段在可解码前扣留。Responses：非流式 item id `ctc_`+随机、输出项 `ctco_`+随机，流式 item id `ctc_`+index（镜像 `fc_` 命名）；流式在解析完成前按 `response.custom_tool_call_input.delta` 扣留、item 完成时 `.done`。
- **Reasoning token 计数（本地口径）** — 由 reasoning-budget 采样器统计：COUNTING/WAITING_UTF8 状态每 accept 一个 token 记 1；自然命中 end 序列时扣除该序列 token 数（下限 0，空块报 0）；FORCING（预算耗尽后的强制 end）不计数；DONE 后 re-arm 到新 start 标签时累计继续（多 think 块累加）；采样器创建时 accept 的 prefill（模板预置 think 标签及其后续 prompt token）在计数与预算上双清（reset_count），prefill 后 `n_counted` 从 0 开始。创建条件：模板给出 start/end 标签 且（grammar_lazy / 预算 ≥0 / reasoning_control 任一，或未启用 backend sampling）；后端采样且无上述需求时不做统计，`n_reasoning_tokens=-1` 按 0 上报（无 think 标签时 0 为真值，backend sampling 时 0 为 best-effort）。

## Official parameter defaults（机器可读 spec）

四个 create 端点的请求侧默认值，取自官方 OpenAPI spec（`openai/openai-openapi`，分支 `main`，`openapi.yaml`，`info.version` 2.3.0；拉取于 2026-09-13，文件 sha1 `39619f0a6b32c8f9cc606ab83becfbd5abb76d26`）。**默认值全部藏在 `anyOf` 分支内**：只解顶层或只解 `allOf` 会全部漏掉，网页文档与 SDK docstring 也不渲染该位置（表面像"官方没写默认"）。刷新解析必须递归 `anyOf`/`allOf`；复拉：`https://raw.githubusercontent.com/openai/openai-openapi/main/openapi.yaml`。

| Param | `/v1/chat/completions` | `/v1/responses` | `/v1/completions` | `/v1/embeddings` |
|-------|------------------------|-----------------|-------------------|------------------|
| `temperature` | `1` | `1` | `1` | - |
| `top_p` | `1` | `1` | `1` | - |
| `presence_penalty` | `0` | - | `0` | - |
| `frequency_penalty` | `0` | - | `0` | - |
| `n` | `1` | - | `1` | - |
| `stream` | `false` | `false` | `false` | - |
| `logprobs` | `false` | - | `null` | - |
| `top_logprobs` | no default | no default | - | - |
| `stop` | `null` | - | `null` | - |
| `logit_bias` | `null` | - | `null` | - |
| `max_tokens` | no default | - | `16` | - |
| `max_completion_tokens` | no default | - | - | - |
| `max_output_tokens` | - | no default | - | - |
| `seed` | no default | - | no default | - |
| `store` | `false` | `true` | - | - |
| `parallel_tool_calls` | `true` | `true` | - | - |
| `service_tier` | `auto` | `auto` | - | - |
| `reasoning_effort` | `medium` | - | - | - |
| `verbosity` | `medium` | - | - | - |
| `truncation` | - | `disabled` | - | - |
| `background` | - | `false` | - | - |
| `best_of` | - | - | `1` | - |
| `echo` | - | - | `false` | - |
| `suffix` | - | - | `null` | - |
| `encoding_format` | - | - | - | `float` |

- 标记：`-` = 该端点没有该参数；`no default` = 参数存在但 spec 未给默认值（不传参时的服务端行为在 spec 层面未承诺）。
- Responses 没有顶层 `reasoning_effort` / `verbosity`，对应配置在 `reasoning` / `text.verbosity` 子对象内（递归 `anyOf` 后两处默认同为 `medium`）；spec 将 Responses `truncation` 标为 deprecated（默认值仍为 `disabled`，保留供参考）；`stop` 顶层默认 `null`，字符串变体分支另有遗留 `default`（`<|endoftext|>`），以顶层为准。
- 官方响应示例回显实际生效的采样值（`temperature: 1.0`、`top_p: 1.0`、`presence_penalty: 0.0`、`frequency_penalty: 0.0`），与本表一致。
- 本地对照：llama.cpp 服务端采样默认（`temp 0.80`、`top_p 0.95`、`min_p 0.05`、`top_k 40`，`common/common.h`）与官方不同，采样相关的对齐与验收必须显式传参，不依赖"不传参"的跨端等价。

### Local omitted-parameter behavior (probed 2026-09-13)

把上表逐项实测"本地不传参"时的实际行为（探针 `/tmp/defaults_probe.py`、`/tmp/defaults_probe_d3c.py`；证据 `/tmp/defaults-probe-1.json`、`/tmp/defaults-probe-d3c2.json`；服务器 = 验收配置 8080）。证据口径：`GET /props` 的 `default_generation_settings.params` 是服务端缺省；`GET /slots` 的 per-slot `params` 是**上一次请求实际生效的参数**（已用显式 `temperature=1`/`top_p=1` 对照验证会原样回显），故"不传参"时的采样生效值可直接读出。判词：`ok` = 与官方默认等价；`differs` = 不等价（登记）；`spec 未承诺` = spec 未给默认值，仅记录实测。

| Param | `/v1/chat/completions` | `/v1/responses` | `/v1/completions` | `/v1/embeddings` |
|-------|------------------------|-----------------|-------------------|------------------|
| `temperature` | `0.80` differs | `0.80` differs | `0.80` differs | - |
| `top_p` | `0.95` differs | `0.95` differs | `0.95` differs | - |
| `presence_penalty` | `0` ok | - | `0` ok | - |
| `frequency_penalty` | `0` ok | - | `0` ok | - |
| `n` | `1` ok | - | `1` ok | - |
| `stream` | `false`（JSON）ok | `false`（JSON）ok | `false`（JSON）ok | - |
| `logprobs` | `false`（无 logprobs）ok | - | `null` ok | - |
| `top_logprobs` | spec 未承诺 | spec 未承诺 | - | - |
| `stop` / `logit_bias` | 不施加 ok | - | 不施加 ok | - |
| `max_tokens` | spec 未承诺（`n_predict=-1`） | - | **`16` 未实现**：不限，至 EOS/槽位上限（实测 32728 completion tokens，`finish_reason=length`）differs | - |
| `max_completion_tokens` | spec 未承诺（`n_predict=-1`） | - | - | - |
| `max_output_tokens` | - | spec 未承诺（`n_predict=-1`） | - | - |
| `seed` | spec 未承诺（随机，`LLAMA_DEFAULT_SEED`） | - | spec 未承诺（随机） | - |
| `store` | `false`：不落盘（`GET .../messages` -> 404）；显式 `true` 可回取 ok | `true`：落盘（回显 `true`、`GET` 200、store 文件）ok | - | - |
| `parallel_tool_calls` | 缺省取模板 caps（本例 `supports_parallel_tool_calls=true`）ok | 缺省 `true`（回显 + 实际不限并行）ok | - | - |
| `service_tier` | 不传不回显；显式回显（`fast`->`priority`）；无调度（§2） | 同 chat | - | - |
| `reasoning_effort` | 不注入 differs（映射缺口见 §4 #1） | 嵌套 `reasoning.effort`：不注入 differs | - | - |
| `verbosity` | 不注入 hint differs（显式 `low`/`medium`/`high` 才注入） | 嵌套 `text.verbosity`：不注入 differs | - | - |
| `truncation` | - | `disabled`：超预算 -> 400（消息含 `set truncation=auto`）ok | - | - |
| `background` | - | `false`：同步返回 ok | - | - |
| `best_of` | - | - | 缺省 = `n`（`1`）ok | - |
| `echo` | - | - | `false`（无 prompt 前缀）ok | - |
| `suffix` | - | - | 不施加 ok | - |
| `encoding_format` | - | - | - | 代码缺省 `float`（`server-context.cpp:6570`；本服务器未启 `--embeddings`，501 与套件同 SKIP）ok |

- 采样侧其余本地缺省（官方无对应字段）：`min_p=0.05`、`top_k=40`、`repeat_penalty=1.0`、repeat window 64、`dynatemp=0`、`mirostat=0`、`dry=0`、`xtc=0`（`/props.default_generation_settings`）。
- 省略任一 `max_*` 且 thinking 打开、未设预算时，本地自动给 8192 reasoning 预算（`server-common.cpp`），thinking 输出上限实测 8192 tokens（chat 总 8201 / responses 总 8198 completion tokens）。
- `truncation=auto` 按**整项**丢弃最旧 item（多 item 实测保留最新 2 项，`input_tokens=24015`）；单个 item 自身超预算 -> 400（消息不带 `set truncation=auto` 提示）。
- Completions `prompt` 省略 -> 400 `'prompt' is required`（官方遗留字符串分支的 `'<|endoftext|>'` 默认未实现，本地更严）。
- 新增登记偏差已同步 [OFFICIAL_API_DIFF.md](OFFICIAL_API_DIFF.md) §4 #13/#14。
