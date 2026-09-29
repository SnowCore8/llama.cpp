# 共有接口定义

跨 API 面的共享契约：错误信封、鉴权、model 校验、参数默认值、非目标面、验收套件、共享本地契约。

## 错误信封

全部端点统一四字段：

```json
{"error": {"message": "...", "type": "...", "param": null, "code": null}}
```

- 插入序：`code`, `message`, `param`, `type`
- `code` / `param` 空时为 `null`（不是省略）
- HTTP 状态与 body `type` 解耦（`error_status_from_body` 映射）

### `type` 词表

| HTTP 状态 | `type` | 依据 |
|-----------|--------|------|
| 400 | `invalid_request_error` | 官方实拍 |
| 401 | `invalid_request_error` | 官方实拍（`code: "invalid_api_key"`） |
| 403 | `permission_error` | 本地词表 |
| 404 | `not_found_error` | 本地词表 |
| 500 | `server_error` | 本地词表 |
| 501 | `not_supported_error` | 本地词表 |
| 503 | `service_unavailable_error` | 官方文档 |
| 400 超上下文 | `exceed_context_size_error` | 本地词表（附加 `n_prompt_tokens` / `n_ctx` 与四字段同层） |

原则：无实拍不发明官方对应。

## 鉴权

`Authorization: Bearer <api-key>` 或 `x-api-key: <api-key>`；`--api-key` 设定。未配 key 时不校验。401 体 `type: "invalid_request_error"`, `code: "invalid_api_key"`。

## Model 校验

| 面 | 未知 model 行为 |
|----|----------------|
| Chat / Embeddings | 404 + A 族（反引号 message, `param: null`, `code: "model_not_found"`） |
| Responses | 400 + B 族（`"The requested model 'X' does not exist."`, `param: "model"`, `code: "model_not_found"`） |
| Completions (legacy) | 不校验（echo served name；0 实拍，维持现状） |
| Router 模式 | proxy 层统一 404 + A 族（chat/responses/embeddings） |

## 参数默认值

取自官方 OpenAPI spec（`openai/openai-openapi` v2.3.0，`anyOf` 分支内）。

| Param | Chat | Responses | Completions | Embeddings |
|-------|------|-----------|-------------|------------|
| `temperature` | `1` | `1` | `1` | - |
| `top_p` | `1` | `1` | `1` | - |
| `presence_penalty` | `0` | - | `0` | - |
| `frequency_penalty` | `0` | - | `0` | - |
| `n` | `1` | - | `1` | - |
| `stream` | `false` | `false` | `false` | - |
| `logprobs` | `false` | - | `null` | - |
| `stop` | `null` | - | `null` | - |
| `logit_bias` | `null` | - | `null` | - |
| `max_tokens` | 无默认 | - | `16` | - |
| `max_output_tokens` | - | 无默认 | - | - |
| `seed` | 无默认 | - | 无默认 | - |
| `store` | `false` | `true` | - | - |
| `parallel_tool_calls` | `true` | `true` | - | - |
| `service_tier` | `auto` | `auto` | - | - |
| `reasoning_effort` | `medium` | `medium`（嵌套 `reasoning.effort`） | - | - |
| `verbosity` | `medium` | `medium`（嵌套 `text.verbosity`） | - | - |
| `truncation` | - | `disabled` | - | - |
| `background` | - | `false` | - | - |
| `best_of` | - | - | `1` | - |
| `echo` | - | - | `false` | - |
| `suffix` | - | - | `null` | - |
| `encoding_format` | - | - | - | `float` |

本地采样其余缺省：`min_p=0.05`, `top_k=40`, `repeat_penalty=1.0`, window 64, `dynatemp=0`, `mirostat=0`。

### 缺省注入行为

- `reasoning_effort` / `verbosity` 省略或 null → 注入 `medium`（服务端注入；模板词表拒绝时按 rank 阶梯就近回退并记 INFO）
- 注入仅作用于 OpenAI 端点（Chat/Responses）；`/apply-template`、`transcriptions`、Anthropic 维持既有行为
- 省略 `max_*` 且 thinking 打开、未设预算时，自动给 8192 reasoning 预算
- `truncation=auto` 按整项丢弃最旧 item；单项超预算 → 400

## 非目标面（显式拒绝或忽略）

| 面 | 本地行为 | 原因 |
|----|----------|------|
| Files / Uploads / Batches / Moderations | 无端点（例外：Responses `moderation` 接受不生效） | 云端专用 |
| `DELETE /v1/models/{model}` | 未实现 | 云端微调面 |
| `file_search` / 远程 `mcp` / `code_interpreter` | 显式 400 | 无云端后端 |
| `service_tier` 执行侧 | 接受+回显（`fast`→`priority`），不改调度 | 无云端调度 |
| `client.beta.*` | 不实现（例外：WS `response.inject` 已实现） | Beta 非稳定面 |

## 验收套件

| 套件 | 覆盖 |
|------|------|
| `official_api_acceptance.responses` | Responses + Conversations + Completions + Models + store + WS + SDK |
| `official_api_acceptance.chat_completions` | Chat Completions + input_tokens + Models + store + SDK |
| `official_api_acceptance.local_durability` | 重启恢复, `GET /v1/tools`, `/props.slot_save_path`, compact expand |

Meta-runner: `python -m official_api_acceptance ...`（durability 最后跑）。判定：exit 0 仅当无 FAIL/PARTIAL/NOT_IMPLEMENTED（SKIP 允许）。

SDK pin: `openai==3.11.0`。验证口径: V100 + Qwen3.5-9B-Q4_K_M。

## 共享本地契约

### `reasoning_effort` / `verbosity` 原值通道

- `chat_template_kwargs.reasoning_effort` 与 `--reasoning-effort` 原值直传、无回退
- 缺省注入（及回退）仅作用于 OpenAI 端点

### SSE `obfuscation`

- Responses: 只在 `*.delta` 事件注入，默认开，`stream_options.include_obfuscation=false` 关闭
- Chat / Completions: 每个 chunk 都带，末块 `""`（官方 chunk schema 定义在整块上）

### Streaming phases

无 queue 阶段：`response.queued` 事件不发射，`created`/`in_progress` 后直接进入生成事件。

### `stop` 门控（Chat + Completions）

string 或 ≤4 条 string 数组；超 4 条、非 string、类型不在 string|array|null → 400。

### Usage reasoning 计数

- Chat: `completion_tokens_details` 恒 5 字段（`reasoning_tokens` 真计数，`text_tokens = completion - reasoning`，其余 = 0）
- Responses: `output_tokens_details.reasoning_tokens` 非流式与流式均为真值
- 计数口径：reasoning-budget 采样器 COUNTING/WAITING_UTF8 每 token 记 1，自然 end 序列扣减（下限 0）；FORCING 不计数；多 think 块 re-arm 累加；prefill 计数与预算双清；不可得（-1）按 0 上报

### `prompt_cache_options.ttl`

只收 `30m`（官方「only supported value」），其它值 → 400。`24h` 只属于 `prompt_cache_retention`。

### `prompt_cache_diagnostics`

本地简化：expected = min(本次/基线 `input_tokens`)，`cached >= expected - 4` → `cache_hit`；`reason` 依次回退首中即止；不产出 `unavailable`。

### `top_logprobs` 无 `include`

单独设置即启用 logprobs 记录并控制 top-N（本地扩展；官方该字段只定义数量上限）。

### Responses output logprobs 空文本过滤

EOG token 无文本不列入条目；条目内空文本候选剔除，条目被剔空则丢弃（Chat 保留上游行为）。

### Custom tools（Chat + Responses）

模板/grammar 面按合成 function schema（`{"input": string}` required）走；序列化按工具名表翻译回官方 custom 形状；`custom.format` 接受但不参与生成。

### `prompt_cache_breakpoint`

形状在原始 message parts 上校验（object 且 `mode="explicit"`, ≤4 part/请求）；锚点按消息边界；强制 batch break 并豁免 mid-prompt skip 与 min-step 闸门。

### Restart durability

`--openai-files-path` 下：interrupted Responses（`in_progress`/`queued`/`cancelling`）→ `failed` + `error.code=server_restart`；completed 条目重载；prompt-cache key TTLs 从 `prompt_cache_keys/` 读回。

## Models 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/v1/models` | 列出（OpenAI `Model` 形状） |
| GET | `/v1/models/{model}` | 检索；按 `model_name` 匹配，404 otherwise；`shutdown_date: null` |

Router 模式从 model store 同路由服务（`hidden` cache → 404）。

## Embeddings

`POST /v1/embeddings` — 输入校验：`input:null`/`input:[]`/非 string `encoding_format` → 400 `invalid_request_error`；响应项仅 `embedding`/`index`/`object` 三键；未开 `--embeddings` → 501；pooling 非 OAI 兼容 → 400。

## Slot KV

`--slot-save-path`；`/slots/{id}?action=save|restore|erase`。
