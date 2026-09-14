# Official API Diff — 本 fork 与 OpenAI 官方公开文档的差异

差异视角摘要：把本 fork 两个官方对齐套件（`responses_official_acceptance` / `chat_completions_official_acceptance`）已验收的行为，与 OpenAI 公开 API 文档逐面对照。

**权威来源是 [OFFICIAL_API_SCOPE.md](OFFICIAL_API_SCOPE.md)**：全部契约细节、依据与验收证据以该文档为准；本文件只做分类汇总，scope 更新时同步本表。

验收基线：2026-09-13，v100 分支本地提交（未 push）：全量验收起于 tip `b74f3a2bf`，缺省对齐改动后经新构建（8091）全量复验；同日 hint 漂移处置波（gating 提交 `2dcc946ac` + 套件断言修复）后再次重基线，新结果与差异明细见 §6。

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
| Responses WS 错误信封 | 一律官方嵌套信封 `{type:"error", status, error:{type, code, message, param}, stream_id?, sequence_number?}`；SSE 扁平 error 帧转发时转换；`error.type` 按词表映射（HTTP 状态或 SSE code 推导；`invalid_request_error` 官方实证，其余为本地词表，明细见 SCOPE「Local contracts」）；本地补充连接级码 `invalid_json` / `unsupported_event_type` |
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
| Anthropic `cache_control` 映射（2026-09-14） | → 本地 `prompt_cache_breakpoint{mode:"explicit"}`（system 块与 message 的 text/image 块）；顶层 `cache_control` = 最后一条非 `tool` 消息的可缓存 part（对应官方 "automatic caching"）；`ttl` → `prompt_cache_options.ttl`；**本地每请求单一 TTL**，多个不同 ttl 折叠为最后一个；`tools[]`/`tool_use`/`tool_result` 上的 `cache_control` 不支持（登记） |
| Anthropic `thinking.disabled` 与 `output_config.effort` 并存（2026-09-14） | disabled 胜（转发 `reasoning_effort="none"`，effort 提示丢弃）；`usage.cache_creation_input_tokens` 恒 `0`（本地 `input_tokens` 已扣除缓存命中，报"创建"会破坏官方求和不变式） |
| Anthropic SSE `error` 帧（2026-09-14） | 已按官方包裹 `{"type":"error","error":{…}}`；默认配置**运行期不可达**（超上下文在流开始前即普通 400 JSON 返回），记为代码层对齐、运行期未验证 |

## 4. 记录未修偏差（Recorded deviations，审计接受）

注：编号稳定优先（跨文件引用依赖，如 PRD §3.2 的「DIFF §4 #12/#13」）；已消化项 #4/#5 的编号保留、不复用，见 §4.1。

| # | 项 | 官方 | 本地 | 说明 |
|---|---|---|---|---|
| 1 | Responses SSE `obfuscation` | 稳定版仅 `response.shell_call_command.delta` 定义该字段 | 对所有 SSE 事件 data 对象注入，默认开（`stream_options.include_obfuscation=false` 关闭） | 本地偏离，声明接受 |
| 2 | `response.reasoning_text.delta`/`.done`、`response.refusal.delta`/`.done` | 官方定义 | 不发射（reasoning 只暴露 summary） | 声明接受 |
| 3 | `prompt_cache_options.ttl` | 仅 `30m` | 超集：另接受 `5m`/`1h`（`24h` 仅属于 `prompt_cache_retention`） | openai SDK 3.11.0 该字段为 `Literal["30m"]`，严格校验的 SDK 解析此类响应会失败 |
| 6 | Chat `n` 上限 | OpenAPI 声明 `n <= 128`（无槽位/并发语义） | `n` 超过 `min(n_parallel, 128)` → 400 | 本地容量上限 + 官方上限取小；常见槽数（4）下槽数先绑定 |
| 7 | Chat `logprobs`+`tools`+`stream` | 官方未禁止 | 组合 → 400 | 本地限制 |
| 8 | Completions `best_of` | 文档措辞 "must be greater than `n`" | 接受 `best_of == n` | 宽松 |
| 9 | `prompt_cache_diagnostics` | 云端 token 计量 | 本地简化口径：expected=min(本次/基线 `input_tokens`)，`cached >= expected-4` → `cache_hit`；不产出 `context_compacted`/`unavailable`；`reason` 首中即止 | 详见 scope「Recorded deviations」 |
| 10 | SSE `error` / `response.failed` | 流中错误事件 | 默认配置**不可达**：流开始前的错误按非流式 HTTP 错误体返回；SSE error+failed 只在流已开始后 reader 出错时产生 | 验收对应行保持 SKIP；`pre_created_error_is_plain_json` 断言非流式错误体形状 |
| 11 | Embeddings `encoding_format` 非 string | 400 | 400，但 message 透传 nlohmann 异常原文（`[json.exception.type_error.302] ...`） | 官方 message 为自由文本，未做措辞映射 |
| 12 | legacy `/v1/completions` 未知 model | 两类实拍均非“从未存在 id” | 不校验（echo served name） | 从未存在 id 0 实拍，不发明形状 |
| 13 | `GET /v1/models/{未知}` | 无错误形状定义（OpenAPI 只声明 200） | 404 + 四字段错误体（`code:null`）+ `The model 'x' does not exist` | 无实拍可依，保留现状 |
| 14 | router 模式未知 model（本地扩展） | 无 router 概念 | proxy 层统一 404 + A 族（chat/responses/embeddings） | 与单模型 responses 的 400 + B 族分层不同 |
| 15 | WS `/v1/responses` 未知 model | WS error 事件（本地信封） | 400 + `model_not_found`（B 族 message） | 套件未覆盖，未实测断言 |
| 16 | legacy `/v1/completions` `echo=true` + `logprobs>0`（非流式；2026-09-14 已提交 `63fd0490c`；探针已验证、三套件已跑） | 官方文档从未规定语义（负结果）；实拍/社区证据见 `/tmp/oai-probe/web-b2/report.md`（官方文档/实拍，非本地验证） | 分片前置 prompt 行、第 0 行 `token_logprobs`/`top_logprobs` 双 `null`、`text_offset` 原点 = 完整 text 开头；代价：该类请求禁用 prompt 前缀复用与后端采样；`n_outputs_max` 抬高（上界 `n_batch*n_vocab*4` 约 2.0 GB、常驻增量 +1812 MiB 实测，见 SCOPE「Recorded deviations」） | 本地限制；`n_outputs_max` 为已接受代价（accepted cost，已随 `63fd0490c` 入库；用户裁定无条件抬高保持）；探针 36/36 PASS（V100+9B，`/tmp/oai-probe/b2a-verify/`），三套件 504 PASS / 24 SKIP / 0 FAIL（`/tmp/acc-baseline-9b/`） |
| 18 | MTP 下生成行 `top_logprobs` 退化（预存，非本波引入；未修） | 生成行 `top_logprobs` 与 prompt 行同形（全词表 top-N） | `--spec-type draft-mtp` 时被接受的草稿 token 所在生成行只有 1 个键；prompt 行不受影响（始终 5..6 键） | 独立实例（无 `--spec-type`）实测每行恰 5 键；MTP 日志 `draft acceptance = 1.00000 (5 accepted / 5 generated)` 对应 1 键生成行（`/tmp/oai-probe/b2a-verify/server3.log`、`server-nospec.log`）；`echo=false` 亦复现，本波未触碰生成行路径 |
| 20 | Anthropic `/v1/messages` 未接的官方字段（2026-09-14） | 官方定义 | `thinking{type:"adaptive"}` 与 `display`、`tools[]`/`tool_use`/`tool_result` 上的 `cache_control`、`service_tier`/`container`/`inference_geo`、`tools[].strict`/`input_examples`、响应 `usage.service_tier`/`stop_details`/`container`/`inference_geo`、`message_delta.usage` 的 input/cache_* 均不生效；非流式错误体仍为跨端点统一 `{"error":{…}}` | 本地无对应机制；非流式错误体属 `ex_wrapper` 的跨端点设计（路由无关），未按 Anthropic 官方 `{"type":"error","error":{…}}` 改；逐项见 SCOPE「Anthropic Messages」 |
| 21 | Anthropic SSE `error` 帧（2026-09-14） | 流中错误事件 data = `{"type":"error","error":{type,message}}` | 已按官方包裹（`server-context.cpp` 的 Anthropic SSE 错误格式化分支），但默认配置不可达 -> **运行期未验证** | 可达点两处（流中出现 error 任务 / 流循环抛异常）；探针实测超上下文（流开始前即普通 400 JSON）、长生成、6 并发流式均无流中 error 帧（SCOPE「Anthropic Messages」） |

### 4.1 已对齐（原登记偏差已消化 + 本轮合规修正，2026-09-14）

本轮已提交（`63fd0490c` + `a9b2ab765` + `2e248d5a9`；本文档随本次提交）；acceptance 三套件已跑全绿（504 PASS / 24 SKIP / 0 FAIL，证据 `/tmp/acc-baseline-9b/`），非流式 `/v1/completions` echo+logprobs 与 `logprobs` 夹取已由探针验证（36/36）。**B2b 流式对偶（原 #17）已实现并验证、随本轮提交**（代码切片 `f27a83df2`）：三套件 505 PASS / 24 SKIP / 0 FAIL（证据 `/tmp/acc-b2b-262144/`）。**Completions `best_of` 容量/官方上限（原 #19 + 新）已实现并验证、随本次提交**（代码切片 `c52034650`）：三套件 530 = 506 PASS / 24 SKIP / 0 FAIL（与 B2b 基线 520 个共有键逐行 0 状态变化、仅 +1 新行；证据 `/tmp/best-of-slots/reports-ttlfix/`）。该切片首轮现场为 530 = 504 PASS / 24 SKIP / 2 FAIL（2 行预存 cache 偶发、非本切片引入），已随同切片套件修复消化（见 §6 尾条）。

| 原 # | 项 | 官方依据 | 本地现状 |
|---|---|---|---|
| 4 | `response.created` / `response.in_progress` 骨架 `usage` 键 | 官方 Response 对象字节始终带 `usage` 键（完成前为 null） | 缺省补 `"usage": null`（`server_responses_enrich_response`）；登记项移出 §4 |
| 5 | compact `service_tier` 枚举 | 官方 compact 文档 5 值（`auto`/`default`/`flex`/`fast`/`priority`） | compact 仅接受 5 值，`scale`/`ultrafast` -> 400；create/chat 仍 7 值（`server_openai_validate_compact_service_tier`）；登记项移出 §4 |
| 新 | Completions `logprobs` 上限 | 文档 "The maximum value for `logprobs` is 5."（`/tmp/oai-probe/web-b2/report.md` S1/S3）；实拍 `logprobs=20` 每行恰 5 键（同报告 S8） | 由「`>5` -> 400」改为「`>5` 夹到 5，HTTP 200」（合规修正）；负数/非整数仍 400（`'logprobs' must be a non-negative integer`）；探针实测 `logprobs=10` -> 200 且每行 <=6 键（`/tmp/oai-probe/b2a-verify/c3-clamp.json`） |
| 17 | legacy Completions `echo=true` + `logprobs>0` 流式（SSE，B2b，2026-09-14；已提交 `f27a83df2`） | 官方文档「streamed 与 non-streamed response object 形状相同」+「data-only SSE as they become available」（`openai-docs/api/reference/resources/completions/methods/create.md`）；实拍分块 `text_offset` 递增（`web-b2` S8） | 已对齐：首块 `text` = prompt + 首个生成 token、首块并进 prompt 行（第 0 行 `token_logprobs`/`top_logprobs` 双 `null`）、`text_offset` 跨块从完整 text 开头累计（echo=true 首块从 0、echo=false 从 prompt 的 UTF-8 字节数起）、末事件不再重复 prompt。实现方对照（vLLM `completion/serving.py` / SGLang `serving_completions.py`）：两家均首块并进 + offset 跨块累加 + prompt 行只发一次，与本实现同向；唯一差异是两家 offset 单位为字符、本地为 UTF-8 字节（B2a 已记录）。契约与运行期证据见 SCOPE「OpenAI Completions」 |
| 19 | `/v1/completions` `best_of` 超槽数（原为请求挂起，2026-09-14） | 每个候选在排序结束前独占一个槽位；官方与实现方都把该类界限放在请求校验层（vLLM `protocol.py` / SGLang `CompletionRequest` 只声明 `best_of`、不实现排序） | 请求校验阶段（路由 `post_completions_oai`）按 `best_of > min(n_parallel, 20)` -> 400 `Field 'best_of': Value must be between 1 <= value <= <界>, but got <N>`（与 `n` 的 schema 硬限同形）；界值本身（`best_of == 界`）仍正常服务。原状：`--parallel 1` 上 `best_of=2` 无限挂起（>300 s 无响应）；现状 0.7 ms 返回 400。探针 4 槽（界 4）、21 槽（界 20）、129 槽（界 20）三实例全 PASS（`/tmp/best-of-slots/`） |
| 新 | 官方声明上限 `best_of <= 20`、`n <= 128` | OpenAPI `CreateCompletionRequest`（`best_of` `maximum: 20`、`n` `maximum: 128`）与 `CreateChatCompletionRequest`（`n` `maximum: 128`）；越界属请求校验错误 | 已落地并与槽数取小：`n_cmpl` schema 硬限 `min(n_parallel, 128)`（`server-schema.cpp`）、`best_of` 路由界 `min(n_parallel, 20)`。实测：21 槽实例 `best_of=21` -> 400（界 20）、`n=22` -> 400（界 21）；129 槽实例 `n=129` -> 400（界 128）、`n=128` -> 200（128 choices）；4 槽实例行为不变（界 4/4）。新增验收行 `completions_best_of_gt_slots_rejected` 取 `min(total_slots, 20)` 作界，故换环境不误判 |

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
| Responses 全量 | Qwen3.5-0.8B-F16（V100，夹具服务器） | **375 = 344 PASS / 5 FAIL / 1 PARTIAL / 25 SKIP** | `/tmp/b1-responses.json`（未知 model 形状落地后重基线） |
| Chat 全量 | Qwen3.5-0.8B-F16 | **144 = 138 PASS / 2 FAIL / 1 PARTIAL / 3 SKIP** | `/tmp/b1-chat.json`（同上） |
| Responses embeddings 组（`--embeddings --pooling mean` 实例） | Qwen3.5-0.8B-F16 | **13 = 12 PASS / 1 SKIP** | `/tmp/b1-responses-emb.json`（同上） |
| Responses embeddings+sdk 组（`--embeddings --pooling mean` 实例，8094） | Qwen3.5-9B-Q4_K_M | **30 = 29 PASS / 0 FAIL / 1 SKIP**（12 个 embeddings 行 11 PASS + sdk 17 行全 PASS；唯一 SKIP = `embeddings.dimensions`） | `/tmp/env-emb-8094/resp-emb-sdk.json` |
| Responses 子集（7 组，工具调用行复核） | Qwen3.5-9B-Q4_K_M | **263 = 254 PASS / 0 FAIL / 9 SKIP** | `/tmp/drift/resp-9b.json` |
| Chat 全量（工具调用行复核） | Qwen3.5-9B-Q4_K_M | **142 = 141 PASS / 0 FAIL / 1 SKIP** | `/tmp/drift/chat-9b.json` |

- **验证口径（2026-09-14 起）**：统一 V100 + Qwen3.5-9B-Q4_K_M（探针与 acceptance 全量）；上表历史 0.8B 数字保留为历史记录，不改写。错误词表修正（401 `invalid_request_error`、503 `service_unavailable_error`）后的 9B 全量新基线**已回填**：acceptance 三套件 530 = 506 PASS / 24 SKIP / 0 FAIL（`/tmp/best-of-slots/reports-ttlfix/`，该构建含词表修正 `997e76504`），401/503/exceed 运行期探针 17 PASS / 0 FAIL（`/tmp/env-503-aux/`，见 SCOPE「错误 type 词表对齐」条）；同批单元测试在网络命名空间隔离（`unshare -n`，零 HF 下载）下全绿（`unit/test_security.py::test_incorrect_api_key` 2 passed、`unit/test_router.py::test_router_api_key_required` 1 passed，证据 `/tmp/env-503-aux/unit-security.log`、`unit-router.log`）。
- 0.8B Responses 的 5 FAIL 中 3 为 hint 确定性登记（`max_tool_calls_with_forced_tool` / `parallel_tool_calls_false_single` / `max_tool_calls_cap_completes`）、2 为 flaky（`previous_response_id.reasoning_replay` / `max_tool_calls`）；1 PARTIAL = `forced_function_tool_call`（hint 确定性）；`function_call_output_continue` 为级联 SKIP。9B 复核全部 PASS → 模型能力类，登记见 SCOPE「Recorded deviations」。
- 0.8B Chat 的 2 个 FAIL：`delta.tool_calls.forced`（hint 确定性登记；fallback 行，9B 主探针成功故不生成）、`tool_choice_required_emits_tool_call`（基线既有 FAIL；9B 复核 PASS）。
- SKIP 均为环境/机型条件：hybrid 模型无 blob checkpoint（`prompt_cache_single_text_append`，判据相对化后按设计 SKIP）、单模型服务器（`prompt_cache_cross_model_isolation`）、不可强注入（`response.failed`/`error` 类）、no probe value、no seed id 等（原「9B sdk `embeddings.create` 未启 embeddings 而 SKIP」已随下条关闭）。
- 9B 复核服务器须带 `LLAMA_WEB_SEARCH_FIXTURE`（与 0.8B 验收配置一致），否则 `web_search` 类行走真实外网、结果不可复现。
- 2026-09-13 缺省对齐后复验（新构建 8091）：Responses 373 / Chat 142 与当时基线逐行 status 一致、0 新增 FAIL（历史记录）；基线证据保留在 `/tmp/full-verify-resp4.json`、`/tmp/chat-full2.json`。
- 2026-09-13 hint 漂移处置后重基线（gating 提交 `2dcc946ac` + 套件断言修复）：4 处 cache 断言前提修复与 `input_tokens_matches_create_usage` 对齐（SCOPE「Suite-only fixes」）、行为漂移登记（SCOPE「Recorded deviations」）；逐行差异（旧基线 / A 环境 / after 三路对拍）与断言改动 diff 见 `/tmp/drift-report.md`；旧全量证据 `/tmp/oai-defs-resp.json`、`/tmp/oai-defs-chat.json`，A 环境证据 `/tmp/a-effort/report-resp.json`、`/tmp/a-effort/report-chat-env.json`。
- 2026-09-13 未知 model 形状对齐后重基线（提交 `8cb9a5257`）：三端点新增行各 PASS（chat 404 A 族 / responses 400 B 族 / embeddings 404 A 族）；B 族 wire 形状按原始响应 bytes 核定为 **error 包裹 4 字段**（SDK 异常输出的“扁平”形是其拆包后的 `.body` 视角）；两套件相对上一基线仅新增行差异、0 意外状态变化；证据 `/tmp/b1-{chat,responses,responses-emb}.json`，报告 `/tmp/b1-fix-report.md`。
- 2026-09-14 本轮（**已提交**：`63fd0490c` + `a9b2ab765` + `2e248d5a9`；本文档随本次提交）：legacy Completions `echo=true` + `logprobs>0` 非流式 prompt 位置行、Completions `logprobs` 夹取 5、Responses 骨架 `usage: null`、compact `service_tier` 5 值；套件行改动（`checks_completions.py` echo/clamp/scenario、`checks_sdk.py` custom_tool `temperature=0`）。**探针已跑（36/36 PASS）**：`/tmp/oai-probe/verify_b2a_echo_logprobs.py` 对非流式 `/v1/completions` echo+logprobs（V100 + Qwen3.5-9B-Q4_K_M，证据 `/tmp/oai-probe/b2a-verify/`）；**acceptance 三套件已跑全绿**（504 PASS / 24 SKIP / 0 FAIL，HEAD `2e248d5a9`，证据 `/tmp/acc-baseline-9b/`），本轮改动已并入基线并跑通；历史 0.8B 基线数字仍不改写。
- 2026-09-14 B2b 流式对偶（`echo=true` + `logprobs>0` 的 SSE；已提交 `f27a83df2`，本文档随本次提交）：新增套件行 `completions_echo_logprobs_stream_chunks`。**ctx=262144 全量三套件 529 = 505 PASS / 24 SKIP / 0 FAIL**（responses 377 = 355/22、chat 143 = 142/1、local_durability 9 = 8/1），与同口径 B2b 前快照（`/tmp/acc-ctx262144-verify/`）逐行 **0 状态变化**、仅 +1 新行（suite+name 去重键 519 -> 520）；新行 3 连跑稳定（`same_text=True`，V100 + Qwen3.5-9B-Q4_K_M）。证据 `/tmp/acc-b2b-262144/`。
- 2026-09-14 Completions `best_of` 容量与官方声明上限（原 #19 + §4.1「新」；代码切片 `c52034650`）：新增套件行 `completions_best_of_gt_slots_rejected`（读 `/props.total_slots`，界取 `min(total_slots, 20)`）。**ctx=262144 全量三套件 530 = 504 PASS / 24 SKIP / 2 FAIL**（responses 378 = 354/22/2、chat 143 = 142/1、local_durability 9 = 8/1）；与 B2b 基线 520 个共有 suite+name 键相比，**仅 2 行 cache 偶发 FAIL 状态不同**（其余 518 键逐行 0 状态变化）、另 +1 新行（去重键 520 -> 521）；该 2 行为预存断言前提问题、非本切片引入：`prompt_cache_ttl_expiry_clears_kv`（`expired_cached=20`，期望 0）及其连带 `prompt_cache_hit_monitor`，残差 20 = 注入 hint 常量前缀（槽位 L1 残留竞态）；**同二进制单跑 `--only prompt_cache` 全绿**（25 = 23 PASS / 2 SKIP / 0 FAIL，同行为 `expired_cached=0`，证据 `/tmp/best-of-slots/ttl-rerun1.json`），已裁定另开切片修该行 -> **已修**（见下条，修复后该 2 行转 PASS、全量 506/24/0）。本切片新增行各轮全 PASS；`best_of`/`n` 探针在 4 槽（界 4/4）、21 槽（`best_of` 界 20、`n` 界 21）、129 槽（`best_of` 界 20、`n` 界 128，`n=128` -> 200 且 128 choices）三实例全 PASS（V100 + Qwen3.5-9B-Q4_K_M 与 CPU + Qwen3.5-0.8B）；修复前同实例 `best_of=5`（槽数 4）无响应 30 s 超时，`--parallel 1` 上 `best_of=2` 由无限挂起改为 0.7 ms 返回 400。证据 `/tmp/best-of-slots/`。
- 2026-09-14 `prompt_cache_ttl_expiry_clears_kv` 去偶发（纯套件修复，无服务端改动；套件切片 `8a6fb0bb4`，本文档随本次提交）：该探针改为**唯一首条 system 消息**（prompt 首 token 唯一），消除「磁盘 TTL 过期只清本 key 槽位、别的槽位仍按键无关的纯 token LCP 命中注入 hint 头」导致的 `expired_cached=20` 偶发；断言 `expired_cached == 0` 强度不变（真实回归时第 3 次仍命中该 key 自己的槽位/L2 -> `cached ≈ warm` -> FAIL）。证据：整条 responses lane 修复后复跑 378 = 356 PASS / 22 SKIP / 0 FAIL（`/tmp/best-of-slots/ttl-fix-run1.json`），**全量三套件 530 = 506 PASS / 24 SKIP / 0 FAIL**（`/tmp/best-of-slots/reports-ttlfix/`；与首轮现场 `reports-final/` 的 504/24/2 对拍仅这 2 行 FAIL -> PASS，其余 519 个共有键 0 变化）；失败现场与 `server.log` 两路对照见 SCOPE「Suite-only fixes」。
- 2026-09-14 9B embeddings live SDK 回归（纯验证波，无提交；关闭 §8.2 #3）：`--embeddings --pooling mean` 实例（9B、8094）上 `--only embeddings,sdk` = **30 = 29 PASS / 0 FAIL / 1 SKIP**（exit=0）。12 个 embeddings 行 11 PASS（`dim=4096`；唯一 SKIP = `embeddings.dimensions`，官方仅 text-embedding-3+ 支持、本地接受但忽略，按设计记录不断言）；sdk 17 行全 PASS，其中 `embeddings.create` 由「未启 embeddings 而 SKIP」转为 **PASS**（`n=1 dim=4096 model='Qwen3.5-9B-Q4_K_M' usage=prompt_tokens=5/total_tokens=5`）。与 B1 波 0.8B embeddings 组（13 = 12 PASS / 1 SKIP）同向、无新增 FAIL。证据 `/tmp/env-emb-8094/resp-emb-sdk.json`。
- 2026-09-14 Anthropic `/v1/messages` 合理加深（**已提交** `ad9e816dd`；本文档随本次提交）：把官方字段接到本地既有机制 —— 请求侧命名 `tool_choice`（不再退化为 `required`）、`tool_choice{none}`、`disable_parallel_tool_use` -> `parallel_tool_calls=false`、`output_config.effort` -> `reasoning_effort`（官方 5 值子集；越界 400）、`output_config.format` -> `response_format`、`thinking{disabled}` -> `reasoning_effort="none"`（覆盖 effort）、`cache_control` -> 本地显式 `prompt_cache_breakpoint`（含顶层自动缓存落点与 `ttl` -> `prompt_cache_options.ttl`）；响应侧 `usage.cache_creation_input_tokens`（恒 0）、`usage.output_tokens_details.thinking_tokens`、thinking `content_block_start.signature`、SSE `error` 帧官方包裹。**验收**：`unit/test_compat_anthropic.py -m "not slow" -k "not vision"` = **34 passed / 4 deselected**（30.54 s，`/tmp/anthropic-deepen/unit-anthropic-full.log`）+ live 探针（V100 + Qwen3.5-9B-Q4_K_M、8095）**15 PASS / 2 FAIL**（2 FAIL 均为 S4 运行期不可触发，`/tmp/anthropic-deepen/probe.log`、`probe-results.json`）。**同批暴露 42 个预存单测失败**（`unit/test_chat_completion.py` + `unit/test_tool_call.py` = 42 failed / 16 passed / 196 deselected，42.04 s，`unshare -n`；根因 = model 校验 `8cb9a5257` 引入的 `'model' is required` 400 与 `model_not_found` 404，本波 `git diff --stat` 未触碰校验路径；运行期复验见 SCOPE「Suite-only fixes」），属独立波次、仅登记。契约见 SCOPE「Anthropic Messages (`/v1/messages`) deepen」，差异登记见 §3 与 §4 #20/#21，行为变化面见 PRD §9.1。

## 维护约定

- 本文件随 [OFFICIAL_API_SCOPE.md](OFFICIAL_API_SCOPE.md) 同步：新增/消化任一偏差时，同时更新对应分类表与「验收现状」。
- 分类归属：未实现 -> §1；接受不生效 -> §2；官方未定义/有意偏离的语义 -> §3；已记录接受的偏差 -> §4；已消化/合规修正 -> §4.1（编号保留、不复用）；本地新增面 -> §5。
- 引用数字以套件 report JSON 与 scope 文档为准，勿凭记忆改写。
