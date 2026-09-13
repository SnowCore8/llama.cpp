# Official API Diff — 本 fork 与 OpenAI 官方公开文档的差异

差异视角摘要：把本 fork 两个官方对齐套件（`responses_official_acceptance` / `chat_completions_official_acceptance`）已验收的行为，与 OpenAI 公开 API 文档逐面对照。

**权威来源是 [OFFICIAL_API_SCOPE.md](OFFICIAL_API_SCOPE.md)**：全部契约细节、依据与验收证据以该文档为准；本文件只做分类汇总，scope 更新时同步本表。

验收基线：2026-09-13，v100 分支本地提交（未 push）：全量验收起于 tip `551ee1755`，缺省对齐改动后经新构建（8091）全量复验；同日 hint 漂移处置波（gating 提交 `730ae34be` + 套件断言修复）后再次重基线，新结果与差异明细见 §6。

## 1. 未实现 / 显式拒绝（Unable / intentionally non-goal）

| Surface | 本地行为 | 原因 |
|---|---|---|
| Files / Uploads / Batches / Moderations | 无本地端点 | 云端专用面（例外：Responses create 字段 `moderation` 接受但不生效，见 §2） |
| `DELETE /v1/models/{model}` | 未实现（retrieve / list 已实现） | 云端仅微调模型可删除，本地无微调模型 |
| 其他 hosted tools：`file_search` / 远程 `mcp` / `code_interpreter` / `code_execution_*` | **显式 HTTP 400**（不静默跳过） | 无云端后端；`web_search` 例外：本地实现，见 §5 |
| 云端 QoS：`service_tier` / `inference_geo` / `container` / `user_profile_id` | `service_tier` 接受官方枚举并回显（`fast`→`priority`）但不改调度；其余忽略或拒绝（明细见 §2） | 本地无云端调度能力 |
| `client.beta.*` 产品 API | 不实现 | Platform Beta 非稳定本地 HTTP 面（例外：beta Responses WebSocket `response.inject` 已实现，见 §3） |
| 逐字节等同云端响应 | 不可能 | 本地模型与模板不同 |

## 2. 接受但忽略（Accept-ignore）

| Field / Header | 官方面含义 | 本地行为 |
|---|---|---|
| Responses create `moderation` | 请求侧审核 | 接受任意形状，不报错、不运行审核 |
| `betas` 请求头 | 选择 beta 面 | 稳定面忽略（套件记录为 PASS） |
| `service_tier`（执行侧） | 影响调度/优先级 | 接受+回显官方枚举，`fast`→`priority`，不改变执行 |
| `inference_geo` / `container` / `user_profile_id` | 云端 QoS/容器 | 忽略或显式拒绝（不生效） |

## 3. 本地契约（官方未定义或有意偏离）

| 面 | 本地契约 |
|---|---|
| Responses WS 错误信封 | 一律官方嵌套信封 `{type:"error", status, error:{type, code, message, param}, stream_id?, sequence_number?}`；SSE 扁平 error 帧转发时转换；`error.type` 按官方词表映射（HTTP 状态或 SSE code 推导）；本地补充连接级码 `invalid_json` / `unsupported_event_type` |
| WS lane 与限额 | `stream_id` 1-256 字符 `[A-Za-z0-9_.-]`，命名 lane 的全部事件（含终止与请求级错误）回显 `stream_id`，默认 lane 不带；限额 = 16 个在飞响应（超出排队）/ 32 个命名 lane（第 33 个报 `websocket_stream_limit_reached`）/ 60 分钟连接寿命 |
| WS 自产事件序号 | `response.steer.*` / `response.inject.*` 的 `sequence_number` 用连接级计数器（官方只规定字段存在，未定义编号基准）；响应流事件保留 per-response 计数 |
| WS create 的 `background` | 忽略且不回显（官方 "background is not supported over WebSocket"）；不报错、照常生成 |
| WS `store=false` 续写 | 连接本地 LRU-256（写入者令牌归档）；只能由**签发它的同一条连接**作 `previous_response_id` 继续；同 lane 续写失败驱逐父、跨 lane fork 失败保留父；未命中统一 `previous_response_not_found`（HTTP 面同码，见「Local contracts」） |
| WS steering | 本地无模型内 steer 注入：目标在下一次 decode 步停表（`STOP_TYPE_STEERED`）→ `incomplete`/`steered` → 自动 successor 承接排队 steer（同 lane）；`too_many_pending_steers` 阈值 32 条/目标；`required_input` 逐 function_call 生成 output stub |
| WS `response.inject`（beta） | 校验后回 `response.inject.created`，注入项由目标终止时的 successor 承接；失败码仅官方两值；schema 不合规 → 通用 `error`(400) 并关闭连接；不强制 `OpenAI-Beta` 请求头 |
| `generate:false` 预热 | HTTP 与 WS 均接受；布尔校验（否则 400）；流式仅 `response.created`+`response.completed`（无 `in_progress`）；只做 prepare 级校验、不跑推理；被 remember 可作 `previous_response_id`，不追加 conversation 回合 |
| Conversations 错误码 | 400 invalid input（含 `include` 超出官方 8 值枚举）、404 missing、500 write failure、501 store disabled（`--responses-store-max 0`） |
| `prompt_cache_breakpoint` | 形状在原始 message parts 上校验（object 且 `mode="explicit"`、≤4 part/请求）；锚点按**消息边界**（官方未定义文本内偏移粒度）；断点强制 batch break 并豁免 mid-prompt skip 与 min-step 闸门；prompt 末尾断点不建 checkpoint |
| `top_logprobs` 无 `include` | 单独设置即启用 logprobs 记录并控制 top-N（官方该字段只定义数量上限，数组由 `include` 启用） |
| Responses output logprobs 空文本过滤 | EOG token 无文本、不列入条目；条目内空文本候选（special/control token）剔除，条目被剔空则丢弃（Chat 保留上游行为） |
| custom 工具执行 | 模板/grammar 面按合成 function schema 走（`{"input":string}` required）；序列化按工具名表翻译回官方 custom 形状；`custom.format` 接受但不参与生成；回放 `input` 重包 `{"input":…}`；Chat 流式（官方未定义）镜像非流式、原始 envelope 片段可解码前扣留；Responses 非流式 id `ctc_`/`ctco_`+随机、流式 `ctc_`+index |
| Reasoning token 计数 | reasoning-budget 采样器口径：COUNTING/WAITING_UTF8 每 token 记 1，自然 end 序列扣减（下限 0）；FORCING 不计数；多 think 块 re-arm 累加；prefill（模板预置标签）计数与预算双清；不可得（-1）按 0 上报 |
| `reasoning_effort` / `verbosity` 缺省与模板词表回退（2026-09-13） | 省略（或 `null`）等价官方缺省 `medium`：`reasoning_effort` 省略、未显式给值（`chat_template_kwargs.reasoning_effort` / `--reasoning-effort`）且 thinking 未关闭时注入 `medium`（thinking 开；budget 阶梯同显式值、请求无显式 budget 时缺省 1024），`verbosity` 省略时注入 medium hint；hint 并入首位 system/developer 消息（追加在内容尾部；无则前置新 system，不新增第二条 system 消息）。服务端注入的 effort 先按原值入模板；模板拒绝时按 rank 阶梯（`minimal`<`low`<`medium`<`high`<`xhigh`<`max`，同距取更高者、最多 3 个候选、排除 `none`）改用最近被接受值并记 INFO 日志；无候选可用时抛原异常（HTTP 500）。`none` 语义不变（关闭 thinking，不注入）。客户端 `chat_template_kwargs.reasoning_effort` 与服务端 `--reasoning-effort` 原生通道原值直传、无回退。缺省注入（及随之的回退）仅作用于 OpenAI 端点（Chat / Responses）；`/v1/messages`、`count_tokens`、`/apply-template`、`transcriptions` 维持既有行为 |
| 流式阶段 | 无 queue 阶段：官方 `response.queued` 事件从不发射，`response.created`/`response.in_progress` 后直接进入生成事件 |
| `max_tool_calls` 截断 | 超上限的调用被丢弃（多余尝试忽略，不打 `incomplete` 标记） |

## 4. 记录未修偏差（Recorded deviations，审计接受）

| # | 项 | 官方 | 本地 | 说明 |
|---|---|---|---|---|
| 1 | Responses SSE `obfuscation` | 稳定版仅 `response.shell_call_command.delta` 定义该字段 | 对所有 SSE 事件 data 对象注入，默认开（`stream_options.include_obfuscation=false` 关闭） | 本地偏离，声明接受 |
| 2 | `response.reasoning_text.delta`/`.done`、`response.refusal.delta`/`.done` | 官方定义 | 不发射（reasoning 只暴露 summary） | 声明接受 |
| 3 | `prompt_cache_options.ttl` | 仅 `30m` | 超集：另接受 `5m`/`1h`（`24h` 仅属于 `prompt_cache_retention`） | openai SDK 3.11.0 该字段为 `Literal["30m"]`，严格校验的 SDK 解析此类响应会失败 |
| 4 | `response.created` 事件 | 官方示例含 `usage: null`（schema 标 optional） | 不带 `usage` 键 | 兼容风险低 |
| 5 | compact `service_tier` 校验 | 官方 compact 文档 5 值 | 复用 create 校验的 7 值（+`fast`/`ultrafast`） | 宽松超集 |
| 6 | Chat `n` 上限 | 无声明上限 | `n` 超过 `n_parallel` → 400 | 本地容量上限 |
| 7 | Chat `logprobs`+`tools`+`stream` | 官方未禁止 | 组合 → 400 | 本地限制 |
| 8 | Completions `best_of` | 文档措辞 "must be greater than `n`" | 接受 `best_of == n` | 宽松 |
| 9 | `prompt_cache_diagnostics` | 云端 token 计量 | 本地简化口径：expected=min(本次/基线 `input_tokens`)，`cached >= expected-4` → `cache_hit`；不产出 `context_compacted`/`unavailable`；`reason` 首中即止 | 详见 scope「记录（未修）」 |
| 10 | SSE `error` / `response.failed` | 流中错误事件 | 默认配置**不可达**：流开始前的错误按非流式 HTTP 错误体返回；SSE error+failed 只在流已开始后 reader 出错时产生 | 验收对应行保持 SKIP；`pre_created_error_is_plain_json` 断言非流式错误体形状 |
| 11 | Embeddings `encoding_format` 非 string | 400 | 400，但 message 透传 nlohmann 异常原文（`[json.exception.type_error.302] ...`） | 官方 message 为自由文本，未做措辞映射 |

## 5. 本地超集与扩展（Supersets / local-only surfaces）

| 面 | 本地实现 | 说明 |
|---|---|---|
| `web_search`（Responses/Chat） | 多后端本地搜索：`LLAMA_WEB_SEARCH_FIXTURE` / `LOCAL_DIR` / `URL` / DuckDuckGo IA+HTML；无命中 `provider=none`（不伪造 stub） | 官方为云端搜索；本地形状对齐（`web_search_call`、`url_citation`、流式事件序） |
| `GET /v1/tools`（+`?format=openai`） | 本地工具发现面（MCP 碰撞 `warnings`） | 无官方对应端点（agent 客户端便利） |
| Slot KV | `--slot-save-path`；`/slots/{id}?action=save\|restore\|erase`；`/props.slot_save_path` | 本地状态持久化 |
| 本地持久化 store | `--openai-files-path` 下 Responses store（`previous_response_id`/retrieve/delete/重启恢复矩阵）、Chat store（`GET /v1/chat/completions/{id}/messages`）、Conversations store（8 端点） | 官方形状 + 本地存储语义 |
| `prompt.id` 文件模板 | `prompts/<id>.json`（`{{variables}}`） + `prompt.version` → `prompts/<id>@<version>.json`；未知 id/缺版本文件 → 400 | 官方为云端 prompt 对象 |
| compact 结果 | 本地不透明 `local.` blob（base64 JSON），可存储并在 `previous_response_id` 上展开 | 非云端加密 |
| `prompt_cache_*` 机制 | 本地 KV 前缀缓存 + slot 亲和 + 磁盘 key TTL（`prompt_cache_keys/`） | 官方为云端缓存 |
| Reasoning budget 阶梯 | `reasoning.effort` → 本地 thinking budget（minimal…max 阶梯） | 本地执行映射 |

## 6. 验收现状（Acceptance status）

| Suite | 模型 | 结果 | 证据 |
|---|---|---|---|
| Responses 全量 | Qwen3.5-0.8B-F16（V100，夹具服务器） | **373 = 343 PASS / 5 FAIL / 1 PARTIAL / 24 SKIP** | `/tmp/drift/resp-full-final.json`（hint 漂移处置后重基线） |
| Chat 全量 | Qwen3.5-0.8B-F16 | **143 = 137 PASS / 2 FAIL / 1 PARTIAL / 3 SKIP** | `/tmp/drift/chat-full-final.json`（同上） |
| Responses 子集（7 组，工具调用行复核） | Qwen3.5-9B-Q4_K_M | **263 = 254 PASS / 0 FAIL / 9 SKIP** | `/tmp/drift/resp-9b.json` |
| Chat 全量（工具调用行复核） | Qwen3.5-9B-Q4_K_M | **142 = 141 PASS / 0 FAIL / 1 SKIP** | `/tmp/drift/chat-9b.json` |

- 0.8B Responses 的 5 FAIL 中 3 为 hint 确定性登记（`max_tool_calls_with_forced_tool` / `parallel_tool_calls_false_single` / `max_tool_calls_cap_completes`）、2 为 flaky（`previous_response_id.reasoning_replay` / `max_tool_calls`）；1 PARTIAL = `forced_function_tool_call`（hint 确定性）；`function_call_output_continue` 为级联 SKIP。9B 复核全部 PASS → 模型能力类，登记见 SCOPE「记录（未修）」。
- 0.8B Chat 的 2 个 FAIL：`delta.tool_calls.forced`（hint 确定性登记；fallback 行，9B 主探针成功故不生成）、`tool_choice_required_emits_tool_call`（基线既有 FAIL；9B 复核 PASS）。
- SKIP 均为环境/机型条件：hybrid 模型无 blob checkpoint（`prompt_cache_single_text_append`，判据相对化后按设计 SKIP）、单模型服务器（`prompt_cache_cross_model_isolation`）、不可强注入（`response.failed`/`error` 类）、no probe value、no seed id、9B sdk `embeddings.create`（未启 embeddings）等。
- 9B 复核服务器须带 `LLAMA_WEB_SEARCH_FIXTURE`（与 0.8B 验收配置一致），否则 `web_search` 类行走真实外网、结果不可复现。
- 2026-09-13 缺省对齐后复验（新构建 8091）：Responses 373 / Chat 142 与当时基线逐行 status 一致、0 新增 FAIL（历史记录）；基线证据保留在 `/tmp/full-verify-resp4.json`、`/tmp/chat-full2.json`。
- 2026-09-13 hint 漂移处置后重基线（gating 提交 `730ae34be` + 套件断言修复）：4 处 cache 断言前提修复与 `input_tokens_matches_create_usage` 对齐（SCOPE「Suite-only fixes」）、行为漂移登记（SCOPE「记录（未修）」）；逐行差异（旧基线 / A 环境 / after 三路对拍）与断言改动 diff 见 `/tmp/drift-report.md`；旧全量证据 `/tmp/oai-defs-resp.json`、`/tmp/oai-defs-chat.json`，A 环境证据 `/tmp/a-effort/report-resp.json`、`/tmp/a-effort/report-chat-env.json`。

## 维护约定

- 本文件随 [OFFICIAL_API_SCOPE.md](OFFICIAL_API_SCOPE.md) 同步：新增/消化任一偏差时，同时更新对应分类表与「验收现状」。
- 分类归属：未实现 → §1；接受不生效 → §2；官方未定义/有意偏离的语义 → §3；已记录接受的偏差 → §4；本地新增面 → §5。
- 引用数字以套件 report JSON 与 scope 文档为准，勿凭记忆改写。
