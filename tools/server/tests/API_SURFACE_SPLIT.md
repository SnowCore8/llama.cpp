# API Surface Split - 目标形状与切片

**状态**：设计稿（2026-09-14 生效裁定：**只留真正需要共享的部分，不做妥协合并兼容**）。
**前身**：本文档由"三面（OpenAI Chat / OpenAI Responses / Anthropic Messages）能否完全分开实现"的评估收束而来，结论是**不复制引擎**、只拆面层，并去掉现有的一切"兼容合并"。
**关联**：[OFFICIAL_API_SCOPE.md](OFFICIAL_API_SCOPE.md)（契约口径见其「Anthropic Messages deep」段与「Recorded deviations」）、[OFFICIAL_API_DIFF.md](OFFICIAL_API_DIFF.md)、[OFFICIAL_API_PRD.md](OFFICIAL_API_PRD.md)、[README-dev.md](../README-dev.md)（请求管线与例程走读）。

---

## 0. 裁定与目标

**裁定**：三个 API 面各自走完整的面层逻辑，只有**引擎**共享；不允许为了复用而把面语义折进共享结构（现状的 chatcmpl 中间层、`__oai_*` 借道键、共享函数内的跨面形状兼容分支，全部属于要消除的"妥协合并"）。

**同时明确不做**：不复制引擎。模板渲染、工具文法、prompt 组装、task 生命周期、slot/KV/checkpoint、采样、prompt cache 是"真正需要共享"的部分，按面复制会破坏"同一模型同一语义"，且显存与缓存语义无法自洽。

**判据**：一个共享点"是否必须拆"，以两条硬证据为准 ——
1. 共享代码里**已经存在的 `res_type` 面条件分支**；
2. 面语义**借道共享 JSON 的私有键**。

---

## 1. 现状边界（实测，基线 `f7e015193`）

### 1.1 唯一引擎入口

`server-context.cpp:4591` 的 `handle_completions_impl` 是唯一引擎入口，8 个调用点收敛于此：

| 调用点 | 面 | `res_type` |
|---|---|---|
| 5530 | native infill | `NONE` |
| 5542 | native completion | `NONE` |
| 5595 | legacy `/v1/completions` | `OAI_CMPL` |
| **5640** | **chat completions** | `OAI_CHAT` |
| **5996 / 6073** | **responses**（含 warmup / background 内层） | `OAI_RESP` |
| 6335 | audio transcriptions | `OAI_ASR` |
| **6358** | **anthropic messages** | `ANTHROPIC` |

`server-context.cpp` 的分工已经是两段：`server_context` / `server_context_impl`（引擎，1-4590）与 `server_routes`（HTTP 处理器，4591-6920）。三个面的路由入口在 `5603` / `5860` / `6343`（`count_tokens` 在 `6366`）。

### 1.2 必须拆：共享代码里的面条件分支

**(A) 面 -> 引擎的参数映射块**（`server-context.cpp:4739-4830`，位于 `handle_completions_impl` 内）。四个面各自往 `task.params` 灌私有字段：

| 面 | 行 | 灌入 |
|---|---|---|
| `OAI_RESP` | 4739-4761 | `__oai_resp_id/input/instructions/request` + steer hold（含 tools 检测） |
| `ANTHROPIC` | 4762-4766 | `anthropic_thinking_display` -> `anthropic_thinking_display_omitted` |
| `OAI_CHAT` | 4767-4781 | `__oai_chat_store/metadata/user/safety_identifier/service_tier` |
| `OAI_CMPL` | 4779-4782、4811-4826 | `user`；以及 `echo`/`best_of`/`n`/rank-by-logprob 整套 legacy 语义 |

**(B) 共享参数结构体是三面并集包**。`server-task.h` 的 `task_params` 同时含 `oaicompat_resp_*`、`anthropic_thinking_display_omitted`、`oaicompat_chat_*`、`oaicompat_cmpl_*`。最硬症状：`need_prompt_logprobs()`（`server-task.h:282`）在**引擎**参数上写 `params.res_type == TASK_RESPONSE_TYPE_OAI_CMPL && ...` —— 面的需求（legacy echo+logprobs 要 prompt 逐位 logits）渗进了引擎。

**(C) 错误体 / 状态码映射**：`res->error(json, bool anthropic)`（`server-context.cpp:4844 / 4856 / 4944`，把面判定降级为 bool 传入共享 res 层）；`is_anthropic_api_path` 三处内联（`server-http.cpp:251` 401、`:270-275` 未就绪 503、`server.cpp:83` `ex_wrapper`）；流式错误帧三分派（`server-context.cpp:4985-5012`）。

**(D) 流式 / 事件模型分派**：`server-context.cpp:4958-4960`（首帧）、`5068-5078`（后续帧）的 `format_anthropic_sse` / `format_oai_resp_sse` / `format_oai_sse` 三分支；`server-task.cpp:1507/1510/1522` 的 `OAI_RESP` 流式状态机（`oai_resp_created` / `oai_web_search_streamed` / `reasoning_summary_frozen`）。

**(E) 收敛层的跨面形状兼容**：`oaicompat_chat_params_parse`（`server-common.cpp:2024`，377 行）单函数内兼容四种形状，函数内注释自证：`Responses:`、`Anthropic convert:`、`Legacy Chat Completions functions`、`Chat custom tool names`。

### 1.3 必须拆：借道私有键（约 28 个）

`__oai_resp_*`(5)、`__oai_chat_*`(5)、`__oai_web_search_*`(8)、`__oai_prompt_cache_*`、`__prompt_cache_ttl`、`__oai_custom_tool_names`、`__oai_return_n`、`anthropic_thinking_display` 等，分布在 `server-context.cpp` / `server-chat.cpp`。

### 1.4 事实上的内核接口（现状）

`handle_completions_impl` 在 `4591-5110` 内实际读取的 `data` 键（实测枚举）分三类：

- **引擎必需（保留进 IR）**：`prompt`、`stream`、`n`、`id_slot`、`message_delimiters`、`__oai_prompt_cache_key`(+`_explicit`/`_implicit`/`_expired`)、`__oai_prompt_cache_breakpoints`、`__oai_prompt_cache_ttl`。
- **面间共享的工具/搜索（保留进 IR）**：`__oai_custom_tool_names`、`__oai_web_search_*`。
- **面私有（必须搬走）**：`__oai_resp_request/id/input/instructions/background`、`__oai_chat_metadata/user/store/service_tier/safety_identifier`、`anthropic_thinking_display`、`echo`、`best_of`、`logprobs`、`user`、`__oai_return_n`。

### 1.5 已分（不再改）

- 响应序列化已按 `res_type` 分派：`server_task_result_*::to_json()`（`server-task.cpp:428`）；Anthropic 序列化在 `server-task.cpp:1197 / 1283 / 1941`。
- Responses 面自成一文件：`server-responses.cpp`（1847 行，含 SSE 事件模型 / store / compaction / obfuscation / WS）。

### 1.6 不可拆（真正需要共享）

- 模板与文法核：`oaicompat_chat_params_parse` 的模板/文法部分（`server-common.cpp:2024`）、`common/chat.cpp`（1577 行）、`common/jinja`（4336 行）、`common/json-schema-to-grammar*`。
- 引擎：`process_single_task`（`server-context.cpp:2639`）、slot/KV/checkpoint（`server-context.cpp:3763-3943`）、`common/sampling.cpp`（931 行）、prompt cache、批处理、`--parallel`。

---

## 2. 目标形状（内核 IR）

面 -> 内核只经一个显式结构：

```cpp
// 面 -> 内核的唯一入口；内核只认这个
struct server_surface_request {
    json                    prompt;   // 模板核产出（含已注入的 tool 文法）
    std::vector<raw_buffer> files;    // 多模态附件
    server_task_params      params;   // 引擎字段：采样 / 缓存键 / stream / id_slot / n_cmpl / 需 prompt logits
    json                    surface;  // 面私有载荷（序列化与流式用），内核不读
};
```

约束：

1. `res_type` 只用于**选面的发送器**（响应侧已分派），不再参与参数构造。
2. 面私有载荷走 `surface`（或由面的流式闭包捕获），不再用 `__oai_*` 穿过共享 JSON。
3. `server_task_params` 只保留引擎字段；面专属字段（`oaicompat_resp_*` / `oaicompat_chat_*` / `oaicompat_cmpl_*` / `anthropic_*`）移出。
4. "面需要 prompt 逐位 logits"这类需求以**显式布尔**表达（取代 `server-task.h:282` 的面分支）。

---

## 3. 切片

每片一个提交，各片独立可回退；顺序保证中间状态不破。

| 片 | 内容 | 触及 |
|---|---|---|
| S1 | 把 `handle_completions_impl` 的参数构造抽成 `server_task_params` 生产者（**等价搬迁**，仍从 `data` 读键），要求同口径 acceptance **逐行 0 状态变化** | `server-context.cpp:4739-4830` |
| S2 | 拆 `oaicompat_chat_params_parse`：共享核（messages -> prompt / 模板 / 文法 / 采样映射）+ 三份 `parse_*_request`（各自字段校验、缺省注入、错误文本）；删掉函数内跨面形状兼容分支与注释 | `server-common.cpp:2024` |
| S3 | Anthropic / Responses 直接产出 `server_surface_request`；删 `server_chat_convert_anthropic_to_oai` 与 `server_chat_convert_responses_to_chatcmpl` 的 chatcmpl 中间产物路径（转换核保留为"面形状 -> IR"） | `server-chat.cpp:12 / 621`、`server-context.cpp:5860 / 6343 / 6366` |
| S4 | 清掉约 28 个借道私有键，面改自持载荷 | 全 server |
| S5 | 错误格式化器按面注册；删 `is_anthropic_api_path` 三处内联 | `server.cpp:83`、`server-http.cpp:251 / 270` |
| S6 | 三个 `format_*_sse` 分派与 `OAI_RESP` 流式状态机移进各面 | `server-context.cpp:4958-5012 / 5068-5078`、`server-task.cpp:1507 / 1510 / 1522` |
| S7 | `need_prompt_logprobs` 显式化；`task_params` 只留引擎字段，删 `server-task.h:282` 面分支 | `server-task.h:87 / 282` |
| S8 | 契约更新：SCOPE 的 Anthropic 段口径由"全部为接线，不引入新子系统"改为"内核单份 + 面层独立"；DIFF / PRD 登记本波 | 三份文档 |

**范围估计**：S1-S7 合计改动量在千行量级（推断，非实测），引擎（§1.6）一行不复制。

---

## 4. 每片验收

统一口径（与三份契约文档一致）：

| 项 | 命令/方式 | 通过条件 |
|---|---|---|
| acceptance 三套件 | `python -m official_api_acceptance --base-url ... --report-dir ...`（V100 + Qwen3.5-9B-Q4_K_M，单模型，非 router） | 530 = 506 PASS / 24 SKIP / 0 FAIL；**S1 / S3 额外要求与上基线逐行 0 状态变化**（suites+name 去重键比对） |
| WS 面 | `--only ws` | 41 = 39 PASS / 2 SKIP / 0 FAIL |
| 单测（离线，`unshare -n`） | `unit/test_compat_anthropic.py` + `unit/test_security.py` | 74 passed / 4 deselected |
| fork CI | push `v100` 后 `CI (fork v100)` | 0 failed（最近基线 400 passed / 3 skipped） |
| 运行期探针 | Anthropic 401/400/200/529 信封、WS 未知 model、echo+logprobs x checkpoint/KV | 全 PASS |

---

## 5. 风险与不变量

**风险（要提前认领）**：面层拆分后，"改一处三面同准"变成"三面各自准"。跨面同一契约必须在三处都成立并各自登记，否则必然漂移。当前已识别的跨面同一契约：

- 错误 `type` 词表（401 `invalid_request_error`、503 `service_unavailable_error`）与 Anthropic 的 503 -> 529 映射；
- legacy Completions `logprobs > 5` 夹取；
- `cache_control.ttl`（Anthropic，5m/1h）与 `prompt_cache_options.ttl`（OpenAI 形状，仅 30m）合并到同一套 prompt cache 语义；
- Responses SSE `obfuscation` 收窄到 `*.delta`（Chat / Completions 仍在整块上）。

**不变量（拆分不得破坏）**：

1. 同一模型在三面上得到相同的 token 化、模板渲染、工具文法与采样结果。
2. prompt cache / checkpoint 只在引擎里存在一份；三面共享同一套断点与键语义。
3. 槽位池、`--parallel`、`--cache-ram`、`--ctx-checkpoints` 的语义与现状一致。
4. 三面仍共用同一进程、同一端口、同一 `--api-key` 名单（面级鉴权与租户配额不在本设计范围）。

---

## 6. 不做项（明确排除）

- **不复制引擎**（§1.6 全部保持单份）。
- **不做面级启停开关**：端点仍全部注册（`server.cpp:342 / 357`）；面级限流/租户配额是另一条工作线，不在本设计内。
- **不动 router 模式**：router 下三面均被代理（`server.cpp:247-250`），本设计不改其行为。
- **不动非聊天面**：`server-context.cpp:4634`（mtmd）、`6735 / 6747 / 6841`（embd）与 ASR 的同型分支不在本次范围。

---

## 7. 变更记录

- 2026-09-14：首版。依据"只留真正需要共享的部分，不做妥协合并兼容"的裁定；边界与行号基于 `v100` tip `f7e015193` 实测。
