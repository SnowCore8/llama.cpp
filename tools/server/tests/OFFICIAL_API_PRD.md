# Official API 对齐 PRD（llama.cpp fork / v100 分支）

- 文档日期：2026-09-14。定位：campaign「OpenAI 官方 API 全量对齐」的整合视图（背景 / 范围 / 证据 / 验收 / 历史 / 判定 / 基线 / 未决 / 风险）。
- 权威源：契约细节与验收证据以 [OFFICIAL_API_SCOPE.md](OFFICIAL_API_SCOPE.md) 为准；差异分类汇总以 [OFFICIAL_API_DIFF.md](OFFICIAL_API_DIFF.md) 为准。本文档与两者同步维护；若源文件间措辞冲突，以 SCOPE 为准（本文相应标注存疑处）。
- 验证口径（2026-09-14 起）：统一 V100（`CUDA_VISIBLE_DEVICES=0`）+ Qwen3.5-9B-Q4_K_M（探针与 acceptance 全量）；历史 0.8B 数字保留为历史基线（见 §7）。
- 分支状态：v100 分支本地工作（未 push）；envelope 波与词表修正波已本地提交 - 代码+单测 `6bf39f10a`、SCOPE/DIFF 登记 `68776830a`、README 示例 `137493cd0`、本文档 `91a53751d`。
- 本轮（B2a 非流式 Completions `echo`+`logprobs` + 三处小对齐）**已提交**（`47bf58b08` + `84f929fce` + `7137fab81`；本文档随本次提交；HEAD = `7137fab81`），且 acceptance 三套件**已跑全绿**（504 PASS / 24 SKIP / 0 FAIL，证据 `/tmp/acc-baseline-9b/`）；运行期探针 36/36 PASS，见 §5 #9 / §8.1。
- 本轮（B2b Completions `echo`+`logprobs` 流式对偶）**已实现并验证**（B2b 切片，已提交 `65b93d9ad`）：acceptance 三套件 **505 PASS / 24 SKIP / 0 FAIL**（ctx=262144，证据 `/tmp/acc-b2b-262144/`），新行 `completions_echo_logprobs_stream_chunks` 3 连跑稳定，与同口径 B2b 前快照逐行 0 状态变化；实现方 vLLM/SGLang 同向，见 §5 #10 / §7.1。

## 目录

1. [背景与目标](#1-背景与目标)
2. [范围](#2-范围)
3. [规范与证据等级](#3-规范与证据等级)
4. [验收体系](#4-验收体系)
5. [执行历史（波次表）](#5-执行历史波次表)
6. [规范判定表（2026-09-14）](#6-规范判定表2026-09-14)
7. [验收基线（最新）](#7-验收基线最新)
8. [未决与后续](#8-未决与后续)
9. [风险与回退](#9-风险与回退)
10. [维护约定](#10-维护约定)

## 1. 背景与目标

- 本 fork 是 llama.cpp 的本地推理服务器：单模型模式 + router 多模型模式（`--models-dir`）。
- 目标：把 OpenAI 公开 API 面做到**形状与可观测语义的全量对齐**。四条主端点：Chat Completions、Responses（HTTP + WebSocket）、Embeddings、Models；另含 Conversations、OpenAI Completions（legacy `/v1/completions`）等面。
- campaign 执行口径（用户裁定）：**官方标准 = 红线、完全让路、不为兼容改标准**。
- 对齐策略（SCOPE）：本地可观测的行为按官方方法/字段语义加深（deepen，验收支撑）；云端专用能力的合法值接受或做形状校验，非法值 -> 400；不做逐字节云响应模拟（本地模型与模板不同，不可能等同）。
- 硬约束（SCOPE）：**行为与方法/字段一一对应**。标为「正向行为」的官方方法或 create 字段必须产生与该名语义一致的可观测效果；禁止用别的字段静默改写官方字段语义；仅回显/形状校验项不得冒充正向行为，本地 deepen 单独标明、不得冒充云端等价。

## 2. 范围

### 2.1 对齐范围（in scope）

| 面 | 内容 | 验收载体 |
|----|------|----------|
| Chat Completions | create/stream、store CRUD（`GET /v1/chat/completions/{id}/messages`）、`input_tokens`、tools/`tool_choice`/`allowed_tools`、legacy 兼容与内容类型补全、logprobs、`prompt_cache_*` | `chat_completions_official_acceptance` + live SDK |
| Responses | create/stream、store/`previous_response_id`、Conversations 成员、compact、`prompt.id`、`input_items`/`input_tokens`、background 流恢复、WS（lanes/steer/inject/warmup）、custom tools | `responses_official_acceptance` + live SDK |
| Embeddings | 输入校验（400）、响应项三键（`embedding`/`index`/`object`）、base64、pooling 兼容 | responses 套件 embeddings 组（`--embeddings` 实例） |
| Models | list + retrieve（含 `shutdown_date: null`；router 从 model store 同路由） | 两主套件 |
| Conversations | 8 端点（create/retrieve/update/delete + items add/list/get/delete） | `responses_official_acceptance` |
| OpenAI Completions | `/v1/completions`（echo/`best_of`/FIM/logprobs/seed/stop 等） | `responses_official_acceptance` |
| 本地扩展面（DIFF §5） | web_search（Responses/Chat）、`GET /v1/tools`、Slot KV、本地持久化 store、`prompt.id` 文件模板、compact `local.` blob、`prompt_cache_*` 调度/亲和/TTL、reasoning budget 阶梯 | `local_durability_acceptance` + 两主套件 |

### 2.2 非目标（out of scope，DIFF §1）

| Surface | 原因 |
|---------|------|
| Files / Uploads / Batches / Moderations | 云端专用面，无本地端点（例外：Responses create 字段 `moderation` 接受但不生效） |
| `DELETE /v1/models/{model}` | 云端微调模型面；本地无微调模型（list/retrieve 已实现） |
| 其他 hosted tools（`file_search` / remote `mcp` / `code_interpreter` / `code_execution_*` 等） | 无云端后端，**显式 HTTP 400**（不静默跳过）；例外：本地 `web_search` deepen |
| 云端 QoS 字段（`service_tier` / `inference_geo` / `container` / `user_profile_id`） | `service_tier` 接受官方枚举并回显（`fast` -> `priority`）但不改调度；其余忽略或拒绝 |
| `client.beta.*` 产品 API | Platform Beta 非稳定本地 HTTP 面（例外：WS `response.inject` 已实现） |
| 逐字节等同云端响应 | 本地模型与模板不同 |

## 3. 规范与证据等级

### 3.1 规范来源

| 来源 | 位置 | 用途 |
|------|------|------|
| 官方文档语料 | `/root/openai-docs`（`api/llms-full.txt`，122,619 行；`api/reference/` 各面参考页） | 字段/错误码/默认值的文档依据 |
| 官方实拍件 | `/tmp/oai-probe/`（如 `first-attempt-401-models.json` 401 原始响应；`web-b1/` B1 取证；`web-b2/` B2 社区原文） | 原始响应 bytes 级依据 |
| 机器可读 spec | `openai/openai-openapi` `openapi.yaml`（`info.version` 2.3.0，2026-09-13 拉取，sha1 `39619f0a...`） | 四个 create 端点请求侧默认值（SCOPE「Official parameter defaults」） |
| 官方 SDK 行为 | tests venv pin `openai==3.11.0`；SDK 拆包行为（openai-python `_client.py`：`data = body.get("error", body)`） | 行为佐证（如 wire 形状 vs SDK 异常 `.body` 视角） |

### 3.2 证据等级（campaign 约定）

- 等级：**实拍（原始响应 bytes） > 官方文档语料 > 推断（必须标注）**。
- 先例：
  - B1 波在任务书措辞与原始件冲突时「以原始件为准」（B 族 wire 形状裁定为 error 包裹 4 字段，b1-fix-report §1.1）；
  - B2 波把社区观察明确标记「推断/未验证」（notes-b2-shape.md）；
  - **「0 实拍不发明形状」**：legacy Completions 未知 model、`GET /v1/models/{未知}` 在无实拍/无规范时维持现状（DIFF §4 #12/#13）。

## 4. 验收体系

### 4.1 套件与判定规则

| 套件（package） | 覆盖（Normative surface） | 说明 |
|-----------------|---------------------------|------|
| `responses_official_acceptance` | Responses + Conversations + Completions + Models + durable Responses store + WS（lanes/steer/inject/warmup）+ live SDK | 主套件（2026-09-14 口径 377 行；历史 0.8B 口径 375 行） |
| `chat_completions_official_acceptance` | Chat Completions（+ `input_tokens`）+ Models + durable Chat store + live SDK | 主套件（2026-09-14 口径 143 行；历史 0.8B 口径 144 行） |
| `local_durability_acceptance` | 重启恢复矩阵、`GET /v1/tools`、`/props.slot_save_path`、compact expand | 最后运行（可能重启服务器） |
| `official_api_acceptance` | meta-runner：顺序跑三套件；`--report-dir` 输出 `responses.json` / `chat_completions.json` / `local_durability.json` / `summary.json` | `python -m official_api_acceptance ...` |

- 判定规则：套件 exit 0 仅当**无 FAIL / PARTIAL / NOT_IMPLEMENTED**（**SKIP 允许**）；套件包 import 失败记为 FAIL 行（exit 1）。
- 设计原则（suite README）：catalog-first（每个官方资源/字段出现在报告中）、schema-first（成功体用 SDK Pydantic 校验）、honest outcomes（缺路由 = `NOT_IMPLEMENTED`，不静默跳过）、no client favoritism（Codex 形状非验收标准）。
- Suite expansion policy（SCOPE）：两套件是 API 对齐主门禁；可经公共 SDK 调用的 durable/local 面都应计分，云端专用面保持显式 400。

### 4.2 基线机制与运行前提

- 基线机制：每次运行导出 report JSON（行级 status）；基线对照以文件为准（如 `/tmp/b1-responses.json`、`/tmp/drift/resp-9b.json`），禁止凭记忆改写数字。
- 运行前提：web_search 行需 `LLAMA_WEB_SEARCH_FIXTURE`（否则走真实外网、结果不可复现）；durability 重启检查需 `LLAMA_SERVER_RESTART_CMD`（否则 SKIP）；embeddings 组需 `--embeddings --pooling mean` 实例；store 类行需 `LLAMA_OPENAI_FILES_PATH`。
- 9B 复核机制（历史双轨）：0.8B-F16 为主基线、Qwen3.5-9B-Q4_K_M 交叉复核；工具调用类 FAIL 在 9B 全 PASS -> 判定「模型能力类」（登记不修实现、不弱化断言）。

### 4.3 验证口径（2026-09-14 起）

**统一 V100（`CUDA_VISIBLE_DEVICES=0`）+ Qwen3.5-9B-Q4_K_M**：所有验证（探针 / 套件 / 回归）在该口径下执行，取代此前「0.8B 主基线 + 9B 复核」双轨；历史 0.8B 数字保留为历史基线（见 §7），不改写。来源：SCOPE 头部口径行 + DIFF §6 注（2026-09-14）；重基线执行计划见 §8.1。

## 5. 执行历史（波次表）

| # | 波次 | 时间 | 范围 | 关键提交 | 结果 / 证据 |
|---|------|------|------|----------|-------------|
| 1 | 初始 W1-W5 全量补齐 | 09-12 晚 - 09-13 | 正向行为全量补齐：WS transport/steering/inject、custom tools、`generate:false`、`prompt.version`、`prompt_cache_breakpoint`、stop 门控、embeddings 校验（500 -> 400）、reasoning 计数、Conversations/Chat store/Completions/Models 补面 + 套件扩展 | 多批，示例 `36d477622`、`2808e6d69`、`f62b7f7d5`、`d686fe77f`、`b186ac938`、`47f6f7d5f`、`a88ff9201`、`332ae3eb8`；契约记录 `551ee1755`；差异文档 `0ce0f252a` | 各面落地并获验收支撑（SCOPE「Deepen campaign status」）；全量验收起于 tip `551ee1755`（DIFF §6） |
| 2 | 9B 复核 | 09-13（14:40 起；漂移波内再复核） | 用 Qwen3.5-9B-Q4_K_M 交叉复核 0.8B 高风险行为行（工具调用类） | 无提交（纯证据波） | 首轮：responses 子集 244 = 236 PASS / 0 FAIL / 8 SKIP（`/tmp/w9b-review.json`）、chat 142 = 141 PASS / 0 FAIL / 1 SKIP（`/tmp/w9b-chat.json`）；漂移波扩组：263 = 254 PASS / 0 FAIL / 9 SKIP（`/tmp/drift/resp-9b.json`）、142 = 141 PASS / 0 FAIL / 1 SKIP（`/tmp/drift/chat-9b.json`）-> 判定模型能力类（drift-report §4） |
| 3 | 二轮缺省对齐 | 09-13 晚 | `reasoning_effort`/`reasoning.effort` 与 `verbosity`/`text.verbosity` 省略/null = 官方缺省 `medium`（服务端注入；模板词表拒绝时按 rank 阶梯就近回退并记 INFO）；gating：缺省注入（及回退）仅作用于 OpenAI 端点（Chat/Responses） | `730ae34be`（+ `c10fb3ef7` 采样缺省、`2874b2903` Completions `max_tokens`=16、`f9bc994df` docs） | 实测对齐（`/tmp/defs-align-probe.json`、`/tmp/a-effort/final-report.md`、`/tmp/a-effort/gating-report.md`）；`/v1/messages`、`count_tokens`、`/apply-template`、`transcriptions` 恢复 pre-A 行为 |
| 4 | hint 漂移处置波 | 09-13 22:48 - 23:10 | hint 常量前缀破坏的 4 处 cache 断言前提修复 + `input_tokens_matches_create_usage` 两条路径对齐 + 行为漂移登记（保持断言强度、禁止恒真） | `9f8337a42`、`60f4f9bc0` | 重基线：responses 373 = 343 PASS / 5 FAIL / 1 PARTIAL / 24 SKIP、chat 143 = 137 PASS / 2 FAIL / 1 PARTIAL / 3 SKIP（drift-report §2.4）；9B 全 PASS -> 模型能力类 |
| 5 | 文档体系重排 | 09-13 23:26 | SCOPE 主体重排（新增 Recorded deviations 节、Server changes 改名、删元叙述）；DIFF 3 处引用名最小修正；相对链接全检 | `68bd07398` | 0 断链、原始条目 -> 新位置全量对照（doc-reorg-report）；语义/数字/证据路径逐字未动 |
| 6 | B1 未知 model 形状 | 09-13 23:10 - 23:36 | chat/embeddings 未知 model -> 404 + A 族（反引号 message、`param:null`、`code:"model_not_found"`）；responses -> 400 + B 族（`The requested model 'X' does not exist.`、`param:"model"`）；router 统一 404 + A 族 | `07ede89c8`、`82dcce5ee`、`801c7d7d6` | 3 个新验收行 PASS、0 意外状态变化（b1-fix-report §4）；B 族 wire 形状按 raw bytes 裁定（§1.1） |
| 7 | envelope 错误体泛化 | 09-13 23:43 - 09-14 00:12 | 全端点统一四字段 envelope（`code`/`param` 空 -> null，插入序 code,message,param,type）+ HTTP 状态与 body 解耦（`error_status_from_body`）；99 处调用点；401 补 `code:"invalid_api_key"` | `6bf39f10a`（10 文件 +54/-59，含单测与 401 `code`） | 探针 12 PASS；B1 A/B 族与 router 回归 byte-equal；三套件 0 行状态差异（envelope-report §4/§5）；9B responses 子集待回填（§5.5） |
| 8 | 词表对齐修正（401/503） | 2026-09-14 | 401 body `type` -> `invalid_request_error`（官方实拍）；503 body `type` -> `service_unavailable_error`（官方文档）；`error_status_from_body` 与 WS 转发映射同步 | `6bf39f10a`（代码）/ `68776830a`（SCOPE/DIFF 登记）/ `137493cd0`（README 示例） | 实现与文档登记已提交；探针/套件验证推迟，9B 全量新基线待重跑回填（DIFF §6 注；`/tmp/envelope-report.md` §5/§8） |
| 9 | B2a：非流式 Completions `echo`+`logprobs` + 三处小对齐 | 2026-09-14（深夜；用户要求不起模型） | legacy `/v1/completions` `echo=true` + `logprobs>0` 非流式前置 prompt 位置行（第 0 行 `token_logprobs`/`top_logprobs` 双 `null`、`text_offset` 原点 = 完整 text 开头、单位沿用 UTF-8 字节）；Completions `logprobs>5` 夹到 5（HTTP 200，原为 400）；Responses 骨架补 `"usage": null`；compact `service_tier` 收窄 5 值；套件行更新（`checks_completions.py` echo/clamp/scenario、`checks_sdk.py` custom_tool `temperature=0`） | **已提交**（`47bf58b08` + `84f929fce` + `7137fab81`；文档随本次提交） | **探针 36/36 + acceptance 三套件全绿**：`/tmp/oai-probe/verify_b2a_echo_logprobs.py` 对非流式 `/v1/completions` echo+logprobs 36/36 PASS（V100 + Qwen3.5-9B-Q4_K_M，证据 `/tmp/oai-probe/b2a-verify/`）；acceptance 三套件 504 PASS / 24 SKIP / 0 FAIL（responses 354/22、chat 142/1、local_durability 8/1；HEAD `7137fab81`，证据 `/tmp/acc-baseline-9b/`；与历史 9B 证据共同行 0 状态变化）。`n_outputs_max` 抬高代价改正为上界 `n_batch x n_vocab x 4` 约 2.0 GB（`n_vocab = 248320` / `n_batch = 2048`）、实测常驻增量 +1812 MiB；契约登记见 SCOPE「OpenAI Completions」+「Recorded deviations」，差异登记见 DIFF §4 #16/#18/#19 + §4.1；本轮改动已并入基线并跑通 |
| 10 | B2b：Completions `echo`+`logprobs` 流式对偶 | 2026-09-14 | legacy `/v1/completions` `echo=true` + `logprobs>0` 的 SSE：首块并进 prompt 文本与 prompt 行（第 0 行双 `null`）、`text_offset` 跨块从完整 text 开头累计（echo=false 从 prompt 的 UTF-8 字节数起）、末事件不重复 prompt；新增套件行 `completions_echo_logprobs_stream_chunks`（echo/logprobs/temperature=0） | **已实现并验证**（B2b 切片，已提交 `65b93d9ad`；本文档随本次提交） | **acceptance 三套件全绿**：ctx=262144 全量 529 = 505 PASS / 24 SKIP / 0 FAIL（responses 377 = 355/22、chat 143 = 142/1、local_durability 9 = 8/1；证据 `/tmp/acc-b2b-262144/`）；与同口径 B2b 前快照（`/tmp/acc-ctx262144-verify/`）逐行 0 状态变化、仅 +1 新行（去重键 519 -> 520）；新行 3 连跑稳定（`same_text=True`）；实现方 vLLM/SGLang 同向、仅 offset 单位差异（字符 vs 本地 UTF-8 字节）；契约登记见 SCOPE「OpenAI Completions」，差异登记见 DIFF §4.1（原 #17） |
| 11 | Completions `best_of` 超槽数修复 + 官方声明上限 20/128 | 2026-09-14 | legacy `/v1/completions` 路由层（`post_completions_oai`）拒绝 `best_of > min(n_parallel, 20)` -> 400 `Field 'best_of': Value must be between 1 <= value <= <界>, but got <N>`（与 `n` 的 schema 硬限同形；原状：超槽数会无限挂起）；同时落地官方 OpenAPI 声明上限 `n_cmpl` schema 硬限 `min(n_parallel, 128)`（覆盖 Chat/Completions/Responses/native）与 `best_of <= 20`；新增套件行 `completions_best_of_gt_slots_rejected`（读 `/props.total_slots`，界取 `min(total_slots, 20)`，环境无关） | **已实现并验证**（代码切片 `6cc674611`；本文档随本次提交） | **acceptance 三套件**：ctx=262144 全量 530 = 504 PASS / 24 SKIP / 2 FAIL（responses 378 = 354/22/2、chat 143 = 142/1、local_durability 9 = 8/1；证据 `/tmp/best-of-slots/`）；2 行 FAIL 为预存 cache 偶发（同二进制单跑 `--only prompt_cache` 全绿，`expired_cached=0`，证据 `/tmp/best-of-slots/ttl-rerun1.json`），非本切片引入，已裁定另开切片修；除这 2 行外与 B2b 基线 520 个共有 suite+name 键逐行 0 状态变化、仅 +1 新行（去重键 520 -> 521）；探针 4 槽（界 4/4）、21 槽（`best_of` 界 20、`n` 界 21）、129 槽（`best_of` 界 20、`n` 界 128，`n=128` -> 200 且 128 choices）全 PASS；修复前同实例 `best_of=5`（槽数 4）无响应 30 s、`--parallel 1` 上 `best_of=2` 无限挂起 -> 现 0.7 ms 400；契约登记见 SCOPE「Recorded deviations」，差异登记见 DIFF §4.1（原 #19 + 「新」） |

## 6. 规范判定表（2026-09-14）

三类分栏：①合规修正（按官方值对齐）；②已处置妥协（401/503 词表）；③无官方实证的本地词表登记（保留沿用）。
依据标注：实拍 = 原始响应件；文档 = `/root/openai-docs` 语料；无证据 = 语料 0 命中。

### 6.1 ① 合规修正（已对齐）

| 项 | 官方依据 | 本地现状（2026-09-14） |
|----|----------|------------------------|
| 四字段 envelope（`code` / `message` / `param` / `type`；`code`/`param` 可为 null） | 官方 spec `Error` required = `[type, message, param, code]`（b-audit/report.md 记录）；官方 401 实拍 4 字段（`/tmp/oai-probe/first-attempt-401-models.json`） | 全部 OpenAI 兼容端点错误体、非 OAI 端点（`/tools`）与 SSE/WS 错误载荷统一四字段（envelope-report §2/§3.1） |
| `code` 语义：string 或 null（空 -> null） | 实拍：`code:"invalid_api_key"`、`code:"model_not_found"`；400 场景 `code:null`（env-probe） | 由 int（原复刻 HTTP 状态）改为 string/null；HTTP 状态经 `error_status_from_body` 从 `type` 推导（8 项映射与旧 int code 逐一等价） |
| `invalid_api_key`（401 `code`） | 实拍：`/tmp/oai-probe/first-attempt-401-models.json` | 401 body `code` 对齐；`type` 处置见 §6.2 |
| `previous_response_not_found` | 文档：`llms-full.txt:32248`；WS mode errors 段 | responses / WS / HTTP 统一该 `code` + `param:"previous_response_id"`（envelope-report §3.2） |
| `model_not_found` | 语料 0 命中；原始件 `/tmp/oai-probe/web-b1`（raw bytes，如 #1278979） | chat/embeddings 404 + A 族；responses 400 + B 族（error 包裹 4 字段）；router 404 + A 族（B1 波；SCOPE） |
| Completions `logprobs` 上限 | 文档 "The maximum value for `logprobs` is 5."（`/tmp/oai-probe/web-b2/report.md` S1/S3）；实拍 `logprobs=20` 每行恰 5 键（同报告 S8） | 由「`>5` -> 400」改为「`>5` 夹到 5（HTTP 200）」；负数/非整数仍 400 `'logprobs' must be a non-negative integer`（**已提交 `47bf58b08`；探针 `logprobs=10` -> 200 且每行 <=6 键，验收行 `clamp=200` PASS**） |
| Completions `best_of` / `n` 上限 | OpenAPI `CreateCompletionRequest`（`best_of` `maximum: 20`、`n` `maximum: 128`，含 "best_of must be greater than n"）与 `CreateChatCompletionRequest`（`n` `maximum: 128`）；越界属请求校验错误 | 与官方值对齐并按本地容量取小：`n_cmpl` schema 硬限 `min(n_parallel, 128)`（覆盖 Chat/Completions/Responses/native）、`best_of` 路由界 `min(n_parallel, 20)`；越界 -> 400（**代码切片 `6cc674611`；探针 4/21/129 槽三实例验证官方界生效，`n=128` -> 200 且 128 choices**） |
| Responses 骨架 `usage` 键 | 官方 Response 对象字节始终带 `usage` 键（完成前为 null） | `response.created`/`response.in_progress` 缺省补 `"usage": null`（**已提交 `84f929fce`；acceptance 三套件全绿**） |
| Responses compact `service_tier` | 官方 compact 文档 5 值（`auto`/`default`/`flex`/`fast`/`priority`） | compact 仅接受 5 值，`scale`/`ultrafast` -> 400；create/chat 仍 7 值枚举（**已提交 `84f929fce`；acceptance 三套件全绿**） |

### 6.2 ② 已处置妥协（词表对齐修正波）

| 项 | 官方值 / 依据 | 本地处置（2026-09-14） | 妥协余项 |
|----|---------------|------------------------|----------|
| 401 body `type` | `invalid_request_error`（实拍 `/tmp/oai-probe/first-attempt-401-models.json`） | 由 `authentication_error` 对齐为 `invalid_request_error`（旧值退场；SCOPE「错误 type 词表对齐」条） | message 保持本地 `"Invalid API Key"`（官方为长文案） |
| 503 body `type` | `service_unavailable_error`（文档 `llms-full.txt:32238`、`llms-full.txt:76944`；官方 `code`=`server_is_overloaded`） | 由 `unavailable_error` 对齐为 `service_unavailable_error`（旧值退场） | `code` 不设（null；`server_is_overloaded` 属云端过载语义） |

- 同步面：`format_error_response` 映射（`ERROR_TYPE_AUTHENTICATION` -> `invalid_request_error`、`ERROR_TYPE_UNAVAILABLE` -> `service_unavailable_error`）、`error_status_from_body`（`service_unavailable_error` -> 503；`authentication_error` 保留为可接受转发值 -> 401）、WS 转发映射（server-responses-ws.cpp）。HTTP 状态码与四字段形状不变（SCOPE 同条）。
- 状态注：实现与文档登记已落地并提交（`6bf39f10a`/`68776830a`/`137493cd0`）；复验（探针/套件）推迟，最终数字以 `/tmp/envelope-report.md` §5/§8 为准。

### 6.3 ③ 无官方实证的本地词表登记（保留沿用）

| `type` | 本地用途 | 官方证据 |
|--------|----------|----------|
| `not_found_error` | 404（默认路由等） | 语料 0 命中 |
| `permission_error` | 403 | 语料 0 命中 |
| `not_supported_error` | 501 | 语料 0 命中 |
| `exceed_context_size_error` | 400 超上下文（本地语义；体含 `n_prompt_tokens`/`n_ctx` 额外字段） | 语料 0 命中 |
| `server_error`（作 `type` 时） | 500 兜底 | 语料仅见 Responses `error.code: "server_error"`（`api/reference/resources/responses.md`）；作顶层 `type` 0 命中 |
| `feature_disabled` | 403 UI 特性开关 | 语料 0 命中 |

- 原则：无实拍不发明官方对应（SCOPE「本地错误 type 词表（官方无实拍值，2026-09-14 核对，保留沿用）」条）；后续取得实拍可再对齐。

## 7. 验收基线（最新）

### 7.1 基线表（B1 波后数字）

| 套件 / 组 | 口径 | 结果 | 证据 |
|-----------|------|------|------|
| Responses 全量 | 历史基线（0.8B 口径） | 375 = 344 PASS / 5 FAIL / 1 PARTIAL / 25 SKIP | `/tmp/b1-responses.json` |
| Chat 全量 | 历史基线（0.8B 口径） | 144 = 138 PASS / 2 FAIL / 1 PARTIAL / 3 SKIP | `/tmp/b1-chat.json` |
| Responses embeddings 组（`--embeddings --pooling mean` 实例） | 历史基线（0.8B 口径） | 13 = 12 PASS / 1 SKIP | `/tmp/b1-responses-emb.json` |
| Responses 子集（7 组，工具调用行复核） | 9B 复核 | 263 = 254 PASS / 0 FAIL / 9 SKIP | `/tmp/drift/resp-9b.json` |
| Chat 全量（工具调用行复核） | 9B 复核 | 142 = 141 PASS / 0 FAIL / 1 SKIP | `/tmp/drift/chat-9b.json` |
| 统一口径全量重基线（V100 + Qwen3.5-9B-Q4_K_M） | 2026-09-14 口径切换 | **已完成**：504 PASS / 24 SKIP / 0 FAIL（responses 354/22、chat 142/1、local_durability 8/1），HEAD `7137fab81` | `/tmp/acc-baseline-9b/` |
| 同口径全量复核，仅 ctx 改 262144（生产口径；探针修复 `dbb7ac3e6` 后） | 2026-09-14 修复后复核 | **已完成**：504 PASS / 24 SKIP / 0 FAIL，与上行逐行状态 **0 变化**（两侧 suite+name 去重键各 519，键集相同） | `/tmp/acc-ctx262144-verify/` |
| 同口径全量复核，B2b 流式对偶（ctx=262144） | 2026-09-14 B2b 切片（代码 `65b93d9ad`） | **已完成**：529 = 505 PASS / 24 SKIP / 0 FAIL（responses 377 = 355/22、chat 143 = 142/1、local_durability 9 = 8/1）；与上行逐行 **0 状态变化**、仅 +1 新行（`completions_echo_logprobs_stream_chunks`，suite+name 去重键 519 -> 520） | `/tmp/acc-b2b-262144/` |
| 同口径全量复核，Completions `best_of` 上限（ctx=262144） | 2026-09-14 本切片（代码 `6cc674611`） | **已完成**：530 = 504 PASS / 24 SKIP / 2 FAIL（responses 378 = 354/22/2、chat 143 = 142/1、local_durability 9 = 8/1）；2 行 FAIL 为预存 cache 偶发（见下「FAIL 归因」），除这 2 行外与上行逐行 **0 状态变化**、仅 +1 新行（`completions_best_of_gt_slots_rejected`，suite+name 去重键 520 -> 521） | `/tmp/best-of-slots/` |

**套件口径**（2026-09-14 全量重基线的运行配置）：单模型 llama-server（**不是 router**）+ `--ctx-size 32768` + `--openai-files-path` 与套件侧 `LLAMA_OPENAI_FILES_PATH` 指向同一空目录 + `LLAMA_WEB_SEARCH_FIXTURE=tools/server/tests/fixtures/web_search_fixture.json` + `--parallel -1`（多槽）。router 口径多 2 行 FAIL 属于**非缺陷**：`POST /v1/responses (unknown model)`（router 分层 404 A 族 vs 套件期望的单模型 400 B 族）；`GET /v1/responses/{id}?stream=true`（router 子进程 store 取不回）。ctx 口径曾经多 1 行 FAIL、**已修**（`dbb7ac3e6`）：`pre_created_error_is_plain_json` 原来硬编码 `huge = "test " * 60000`（约 6 万 token），只假定 ctx 远小于约 6 万 token；ctx 大于该值时载荷不超预算，请求被正常受理（HTTP 200 + SSE），该行以**配置原因**FAIL，不是产品行为差异。现在载荷按 `/props.n_ctx` 取 `max(60000, int(n_ctx * 1.5) + 512)`，与其余 ctx 相关行（`n_ctx//5` 自适应）口径一致。定位依据（同一代码同一请求、只差 ctx）：旧行为 ctx=262144 -> `FAIL HTTP 200 ct='text/event-stream'`、ctx=32768 -> `PASS HTTP 400 ... msg='input exceeds context token budget (32248 tokens); set truncation=auto to drop oldest item'`（32248 = 32768 - 512 - 8，与历史 0.8B 报告 `/tmp/b1-responses.json`、`/tmp/full-verify-resp.json` 的 detail 逐字节相同）；修后 ctx=262144 -> `PASS HTTP 400 ... msg='input exceeds context token budget (261624 tokens)'`（261624 = 262144 - 512 - 8）。

FAIL / PARTIAL / SKIP 归因（DIFF §6）：

- 0.8B responses 5 FAIL：3 hint 确定性登记（`max_tool_calls_with_forced_tool` / `parallel_tool_calls_false_single` / `max_tool_calls_cap_completes`）+ 2 flaky（`previous_response_id.reasoning_replay` / `max_tool_calls`）；1 PARTIAL = `forced_function_tool_call`（hint 确定性）。
- 0.8B chat 2 FAIL：`delta.tool_calls.forced`（hint 确定性登记；fallback 行，9B 主探针成功故不生成）、`tool_choice_required_emits_tool_call`（基线既有 FAIL；9B 复核 PASS）。
- SKIP 均为环境/机型条件：hybrid 无 blob checkpoint、单模型服务器（`prompt_cache_cross_model_isolation`）、不可强注入（`response.failed`/`error` 类）、no probe value、no seed id、9B sdk `embeddings.create`（未启 embeddings）等。
- 9B 复核目标行全部 PASS -> 漂移判定「模型能力类」（登记见 SCOPE「Recorded deviations」）。
- 本轮 2 行 cache 偶发 FAIL（`prompt_cache_ttl_expiry_clears_kv` + 连带 `prompt_cache_hit_monitor`）：预存断言前提问题（残差 20 = 注入 hint 常量前缀，槽位 L1 残留竞态），非本切片引入；同二进制单跑 `--only prompt_cache` 全绿（25 = 23 PASS / 2 SKIP / 0 FAIL，`expired_cached=0`，`/tmp/best-of-slots/ttl-rerun1.json`）；已裁定另开切片修该行（保持断言强度）。

注：**envelope 波 + 词表修正波确认后的最终数字见 `/tmp/envelope-report.md`（§5/§8），以届时核验为准。** 当前该报告 §5.5 标「延迟至下轮（用户要求夜间不起模型；本轮无任何套件运行）」、§8 已更新（词表对齐修正）；9B responses 子集与词表修正复验待下轮执行（见 §8.1）。

注：本轮 B2a 已提交（`47bf58b08` + `84f929fce` + `7137fab81`）且 acceptance 三套件**已跑全绿**（504 PASS / 24 SKIP / 0 FAIL，证据 `/tmp/acc-baseline-9b/`，见 §5 #9 / §8.1）：§7.1 表中「统一口径全量重基线」已并入基线并跑通；本轮新增/更新的套件行已纳入该次运行；历史 0.8B 基线数字不改写。

注：本轮 B2b 流式对偶**已实现并验证**（B2b 切片，已提交 `65b93d9ad`）且 acceptance 三套件**已跑全绿**（ctx=262144 全量 505 PASS / 24 SKIP / 0 FAIL，证据 `/tmp/acc-b2b-262144/`，见 §5 #10 / §7.1 / §8.1）；与同口径 B2b 前快照逐行 **0 状态变化**、仅 +1 新行，新行 3 连跑稳定。

注：本轮 Completions `best_of` 上限（含官方声明上限 20/128）**已实现并验证**（代码切片 `6cc674611`）：ctx=262144 全量 530 = 504 PASS / 24 SKIP / 2 FAIL（2 行 FAIL 为预存 cache 偶发，同二进制单跑 `--only prompt_cache` 全绿，见上「FAIL 归因」；证据 `/tmp/best-of-slots/`，见 §5 #11 / §7.1 / §8.1）；除这 2 行外与 B2b 基线 520 个共有键逐行 **0 状态变化**、仅 +1 新行；`best_of`/`n` 探针在 4/21/129 槽三实例验证官方界生效。

## 8. 未决与后续

### 8.1 下一轮验证计划（统一口径 V100 + Qwen3.5-9B-Q4_K_M）

**9B 全量重基线**（承接词表修正波与统一口径切换）：

1. responses 全量；
2. chat 全量；
3. embeddings 组（`--embeddings --pooling mean` 实例）；
4. 401 / 503 / exceed 探针（错误体四字段 + 新词表 + 状态码）；
5. B1 四路径回归（chat 404 A / responses 400 B / embeddings 404 A / router 404 A）；
6. 本轮 B2a 新增/更新的套件行（`checks_completions.py` echo/clamp/scenario、`checks_sdk.py` custom_tool `temperature=0`）；本轮已提交（`47bf58b08` + `84f929fce` + `7137fab81`）并已并入基线并跑通。
7. B2b 流式对偶新增行 `completions_echo_logprobs_stream_chunks` 已随本轮实现并跑通（代码 `65b93d9ad`；ctx=262144 全量 505 PASS / 24 SKIP / 0 FAIL，证据 `/tmp/acc-b2b-262144/`；新行 3 连跑稳定）。
8. Completions `best_of` 上限新增行 `completions_best_of_gt_slots_rejected` 已随本轮实现并跑通（代码 `6cc674611`；ctx=262144 全量 530 = 504 PASS / 24 SKIP / 2 FAIL，其中 2 行为预存 cache 偶发、非本切片引入，证据 `/tmp/best-of-slots/`；`best_of`/`n` 探针 4/21/129 槽三实例全 PASS）。
9. 另开切片修 `prompt_cache_ttl_expiry_clears_kv` 偶发（保持断言强度：区分共享 hint 前缀残留与整段前缀命中，或强制驱逐槽位 KV）；该行及其连带 `prompt_cache_hit_monitor` 同二进制单跑已全绿（`/tmp/best-of-slots/ttl-rerun1.json`）。

来源：DIFF §6 注（「9B 全量新基线待重跑回填」）+ 2026-09-14 用户口径指令 + 本轮 §5 #9 / §5 #10 / §5 #11。

**本轮已跑运行期探针**（`/tmp/oai-probe/verify_b2a_echo_logprobs.py`，非流式 `/v1/completions` echo+logprobs 36/36 PASS；V100 + Qwen3.5-9B-Q4_K_M，证据 `/tmp/oai-probe/b2a-verify/`）；**acceptance 三套件已跑全绿**（504 PASS / 24 SKIP / 0 FAIL，HEAD `7137fab81`，证据 `/tmp/acc-baseline-9b/`），本轮已提交并并入基线。**B2b 流式切片**同样全绿：ctx=262144 全量 529 = 505 PASS / 24 SKIP / 0 FAIL（证据 `/tmp/acc-b2b-262144/`）。

### 8.2 功能未决项

| # | 项 | 现状 | 依据 |
|---|----|------|------|
| 1 | B2：legacy Completions `echo=true` + `logprobs` 组合完整语义 | **非流式（B2a）已实现并提交**（`47bf58b08`；探针 36/36 + 三套件全绿）：分片前置 prompt 行、第 0 行 `token_logprobs`/`top_logprobs` 双 `null`、行 p+1 取自位置 p 的 logits、`text_offset` 原点 = 完整 text 开头（单位沿用 UTF-8 字节）、`logprobs>5` 夹到 5；**流式（B2b）已实现并验证**（B2b 切片，已提交 `65b93d9ad`；首块并进 prompt 文本与 prompt 行、`text_offset` 跨块累计、末事件不重复 prompt；三套件全绿 + 新行 3 连跑稳定） | `/tmp/oai-probe/web-b2/report.md`（官方文档/实拍）、实现方 vLLM/SGLang 源码对照、b-audit/report.md B2；本轮改动见 §5 #9 / §5 #10 |
| 2 | SDK `responses.create.custom_tool` 行 temperature 硬化 | **已处理**（已提交 `7137fab81`；套件已验证）：该调用补 `temperature=0`（按先例 `58d28395b`，forced-tool 检查 pin temperature 0）；该行曾 flaky（PASS/FAIL 双峰），本次运行时复验 PASS | drift-report §3；checks_sdk.py |
| 3 | embeddings live 实测 | sdk `embeddings.create` 在未启 `--embeddings` 时 SKIP（9B 全量即如此）；需在 embeddings 实例上补 live SDK 回归 | DIFF §6 SKIP 说明 |
| 4 | `/v1/messages` 显式 verbosity/reasoning_effort | anthropic 转换器不透传顶层字段（pre-A 既有）；缺省注入已按 gating 限定 OpenAI 端点；若要支持显式值需扩转换器 | gating-report §6.2 |

### 8.3 SCOPE「Recorded deviations」登记项（2026-09-14 时点；编号稳定，原 #5/#6/#21/#25 已对齐）

标题清单（细节与处置以 SCOPE 为准）：

1. `reasoning_effort` / `verbosity` 原值通道与缺省注入范围（二轮，2026-09-13）；
2. Responses SSE `obfuscation` 本地偏离；
3. `response.reasoning_text.*` / `response.refusal.*` 不发射；
4. `prompt_cache_options.ttl` 超集（openai SDK 3.11.0 `Literal["30m"]` 严格校验代价）；
5. `response.created` 不带 `usage` 键 - **已对齐**（骨架缺省补 `usage: null`；移出 SCOPE「Recorded deviations」，见 DIFF §4.1）；
6. compact `service_tier` 复用 create 校验（7 值 vs 官方 compact 文档 5 值）- **已对齐**（compact 收窄 5 值，`scale`/`ultrafast` -> 400；见 DIFF §4.1）；
7. Chat `n` 超过 `n_parallel` -> 400；
8. Chat `logprobs` + `tools` + `stream` 组合 -> 400；
9. Completions 接受 `best_of == n`；
10. `prompt_cache_diagnostics` 本地简化口径；
11. SSE `error` / `response.failed` 本地不可达；
12. Embeddings `encoding_format` 非 string 的 message 透传；
13. 未知 model 两个证据缺口（legacy completions 不校验；`GET /v1/models/{未知}` 形状保留）；
14. router 模式未知 model 形状（本地扩展）；
15. WS `/v1/responses` 未知 model（未实测断言）；
16. hint 缺省引入的行为漂移（2026-09-13 验收登记，未修）；
17. 错误 `type` 词表对齐（2026-09-14；见本文 §6.2）；
18. 本地错误 `type` 词表登记（2026-09-14；见本文 §6.3）；
19. legacy Completions `echo=true` + `logprobs>0`（非流式）禁用 prompt 前缀复用（2026-09-14；已提交 `47bf58b08`）；
20. legacy Completions `echo=true` + `logprobs>0`（非流式）禁用后端采样（同上）；
21. legacy Completions `echo=true` + `logprobs>0` 的 SSE/流式 - **已对齐**（B2b，已提交 `65b93d9ad`；首块并进 prompt 文本与 prompt 行、`text_offset` 跨块累计、末事件不重复 prompt；移出 SCOPE「Recorded deviations」，见 DIFF §4.1 原 #17）；与 context shift / checkpoint / KV 组合仍**未验证**（unverified）；
22. legacy Completions `echo=true` + `logprobs>0`（非流式）的未解码行返回 `null`（同上）；
23. `params_base.n_outputs_max` 抬高：已接受的已知代价（accepted cost），所有请求常驻增量实测 +1812 MiB（上界 `n_batch x n_vocab x 4` 约 2.0 GB，`n_vocab`=248320 / `n_batch`=2048）；
24. MTP 下生成行 `top_logprobs` 退化（`--spec-type draft-mtp`；预存、非本波引入、未修）；
25. `/v1/completions` `best_of > n_parallel` 无校验且请求挂起 - **已修**（本切片，代码 `6cc674611`；路由层按 `best_of > min(n_parallel, 20)` -> 400，并落地官方声明上限 `n <= 128` / `best_of <= 20`；见 DIFF §4.1 原 #19 + 「新」）；

### 8.4 其它已登记（波报告未决点）

- 本轮 2 行 cache 偶发 FAIL（`prompt_cache_ttl_expiry_clears_kv` + 连带 `prompt_cache_hit_monitor`）：残差 20 = 注入 hint 常量前缀（槽位 L1 残留竞态），同二进制单跑 `--only prompt_cache` 全绿；已裁定另开切片修（见 §8.1 #9）。
- WS 未知 model 单独探针未单列（envelope-report §6.5）；
- router 与单模型的分层差异（B1 既有，保持；envelope-report §6.4）；
- `prompt_cache_single_text_append` 新判据未在 dense 模型复验（drift-report §6.1）；
- Anthropic SSE 透传错误帧 `code` int -> null（顺带效果，登记）（envelope-report §6.2）；
- background response error 体仅 `{message}`（协议对象，明确范围外）（envelope-report §6.3）。

## 9. 风险与回退

### 9.1 行为变化面（第三方客户端影响）

| 变更 | 波次 | 影响 | 缓解 |
|------|------|------|------|
| 错误体 `code` 由数字（int）变 string/null | envelope 波 | 按数字读取 `code` 的第三方客户端 | 与官方形状一致；验收套件对 `code:null` 天然兼容（envelope-report §5.6） |
| 401 `type` 由 `authentication_error` 变 `invalid_request_error`；503 由 `unavailable_error` 变 `service_unavailable_error` | 词表修正波 | 按键值字符串匹配旧词表的客户端 | 对齐官方实拍/文档；单点修改可独立回退 |
| 未知 model 校验引入（chat/embeddings/responses/router） | B1 波 | 回显任意 model 名的客户端现在收到 404/400 | 与官方一致；A/B 族 body 逐字节固定 |
| 缺省 hint 注入（常量 20-token system 前缀） | 二轮缺省对齐 | prompt 前缀变化；0.8B 工具调用类验收行漂移（已登记）；`/v1/messages` 等不受影响（gating） | 漂移判定模型能力类；套件保持真实状态、未弱化断言 |
| Completions `echo=true`+`logprobs>0` 新语义 + `logprobs>5` 夹取 + Responses 骨架 `usage: null` + compact `service_tier` 收窄 | 本轮（B2a；已提交 `47bf58b08` + `84f929fce` + `7137fab81`） | 依赖 `logprobs>5 -> 400` 的客户端现在收 200（夹到 5）；该类请求失去 prompt 前缀复用/后端采样（延迟上升）；未完成 Responses 对象新增 `usage: null` 键；`n_outputs_max` 抬高带来常驻内存（实测 +1812 MiB，上界 `n_batch x n_vocab x 4` 约 2.0 GB；已随 `47bf58b08` 入库，用户裁定无条件抬高保持） | 与官方一致；本轮为独立提交切片，已并入基线并跑通，可整体回退 |
| MTP 下生成行 `top_logprobs` 退化为 1 键 | 预存（非本轮引入） | `--spec-type draft-mtp` 时被接受草稿 token 的生成行只带自身分值；prompt 行不受影响 | 用户裁定先不修；默认关闭 MTP（`/root/llama_gguf/models.ini` 两段 `spec-type = draft-mtp` 已注释）；与本轮切片无关 |
| Completions `best_of` 超槽数由挂起改为 400；`best_of`/`n` 上限收窄到 `min(n_parallel, 20)` / `min(n_parallel, 128)` | 本轮（本切片；代码 `6cc674611`） | 原来 `best_of > n_parallel` 时请求无限挂起（无错误、无响应；`--parallel 1` 上 `best_of=2`），现返回 400 `Field 'best_of': ...`；`n` 上限由原先的仅槽数界改为槽数与官方界 128 取小（21/129 槽实例上界值收敛）；4 槽等常见实例行为不变 | 与官方 OpenAPI 声明上限一致（`best_of <= 20`、`n <= 128`）且在请求校验层提前拒绝（官方/vLLM/SGLang 同层）；独立切片，可单独回退 |

### 9.2 回退方式

- 各波为独立切片：按提交 revert（hash 见 §5 波次表）；envelope 波与词表修正波为 `6bf39f10a`（代码）/ `68776830a`（登记）/ `137493cd0`（README）/ `91a53751d`（本文档），可独立回退。
- 回退判定：以 report JSON 行级 status 对照基线（§7），任何回退后重跑套件核对 0 意外变化。

## 10. 维护约定

- 本文档与 [OFFICIAL_API_SCOPE.md](OFFICIAL_API_SCOPE.md) / [OFFICIAL_API_DIFF.md](OFFICIAL_API_DIFF.md) 同步：契约/证据变更 -> 先改 SCOPE，再同步 DIFF 分类与本文整合视图；冲突时以 SCOPE 为准。
- 数字引用一律取自套件 report JSON / 各波报告（`/tmp/b1-*.json`、`/tmp/drift/*`、`/tmp/envelope-report.md`、`/tmp/a-effort/*` 等），禁止凭记忆改写。
- 验证口径变更（如 2026-09-14 统一 9B）需在 SCOPE 头部、DIFF §6 注与本文 §4.3/§7 三处同步。
