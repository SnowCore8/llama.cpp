# Responses API (`/v1/responses`)

## 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/v1/responses` | 创建（含 stream/background） |
| GET | `/v1/responses/{id}` | 检索（`?stream=true` 恢复流） |
| DELETE | `/v1/responses/{id}` | 删除 |
| POST | `/v1/responses/{id}/cancel` | 取消 |
| POST | `/v1/responses/compact` | 压缩（`response.compaction`） |
| GET | `/v1/responses/{id}/input_items` | 列出输入项 |
| POST | `/v1/responses/input_tokens` | Token 计数 |
| WS | `/v1/responses` | WebSocket 传输 |

## Create 字段

| Kind | 字段 |
|------|------|
| **正向行为** | `model`, `input`（仅 `input_text`/`input_image`，其余 → 400；assistant 可选 `phase`=`commentary`/`final_answer`，非法 → 400）, `instructions`, `previous_response_id`, `conversation`（items 前置 + 本轮追加；与 `previous_response_id` 互斥）, `store`, `stream`, `temperature`/`top_p`/`max_output_tokens`, `tools`/`tool_choice`/`parallel_tool_calls`（`false` → emit 最多 1 个 function_call；`web_search` → 本地搜索；`allowed_tools` → 子集过滤，`required` 无匹配 → 400）, `text`（`verbosity` hint；`format` → grammar：`json_object`/`json_schema`）, `reasoning`（`effort` → thinking + budget 阶梯；`context=current_turn`；`summary`/`generate_summary`；`mode=pro`）, `background`, `max_tool_calls`, `context_management`, `stream_options.include_obfuscation`, `include`（`message.output_text.logprobs` / `reasoning.encrypted_content`）, `top_logprobs`, `truncation`, `metadata`/`user`/`safety_identifier`, `prompt_cache_key`/`prompt_cache_retention`/`prompt_cache_options`, `prompt.id`/`prompt.version`, `prompt_cache_breakpoint` |
| **Accept-ignore** | `moderation`（接受不生效） |
| **Model 校验** | 未知 → 400 + B 族 |

## 本地契约

### Store

- Durable under `--openai-files-path/responses/`；`store=true` 默认
- `previous_response_id` 链式续写
- Interrupted `in_progress`/`queued` → restart 后 `failed` + `error.code=server_restart`

### Compact

- `encrypted_content` = 本地 opaque blob（`local.` + base64 JSON of folded non-user items）
- Compact 结果被存储，可在后续 `previous_response_id` 展开
- `service_tier` 仅接受 5 值（`auto`/`default`/`flex`/`fast`/`priority`），`scale`/`ultrafast` → 400

### Conversations API

- Durable under `--openai-files-path/conversations/`
- 8 端点：create/retrieve/update/delete + items add/list/get/delete
- List defaults: `limit=20`, `order=desc`；add cap 20 items
- `include` gating: 官方 8 值枚举，其它 → 400
- Item ids fallback: `item_` prefix（无专用前缀时）
- Store disabled（`--responses-store-max 0`）: write → 501, read → 404
- `conversation` create: 前置 stored items + 追加完成回合（`store=false` 仍追加；cancelled/failed 不追加）
- 错误码: 400 invalid input, 404 missing, 500 write failure, 501 store disabled

### Background stream

- `background: true` → 立即返回 `in_progress`；可 retrieve/cancel
- Resumable: `GET /v1/responses/{id}?stream=true` + `starting_after` cursor
- 无 cursor → 从最老完整事件起；404 无 session；400 replay prefix 已丢
- Replay window 被逐 → 终端 SSE `error` 事件（不静默结束）

### `max_tool_calls`

超限调用丢弃（不打 `incomplete` 标记）。

### `context_management`

Compaction 自动折叠历史 + 展开 `local.` back into real messages for model。

### `generate:false` warmup

- HTTP + WS 均接受；`generate` 必须布尔（否则 400）
- 流式仅 `response.created` + `response.completed`（无 `in_progress`）
- 只做 prepare 级校验，不跑推理
- 被 remember 可作 `previous_response_id`；不追加 conversation 回合

### `prompt.id` 模板

- `--openai-files-path/prompts/<id>.json`（`instructions`/`input` + `{{variables}}`）
- 未知 id → 400；versioned: `prompts/<id>@<version>.json`
- 变量展开: `input_text` 拼接、`input_image` 保留 shape、`input_file` → 400
- 占位符无匹配变量 → 400

### Web search（本地完整托管形态）

- 多后端: fixture / local dir / URL / DuckDuckGo SERP
- 无命中 → `provider=none`
- `web_search_call` 生命周期: `output_item.added` → `in_progress` → `searching` → `completed` → `output_item.done`
- `url_citation` annotations: `output_text.done` 前每条 citation 一条 `output_text.annotation.added`
- `include` 门控 `action.sources` / `results`
- `search_context_size` → 结果数 + `open_page`/`find_in_page`
- `filters.allowed/blocked_domains`

## WebSocket

- 错误信封: 官方嵌套 `{type:"error", status, error:{type, code, message, param}, stream_id?, sequence_number?}`
- `stream_id`: 1-256 字符 `[A-Za-z0-9_.-]`；命名 lane 全部事件回显 `stream_id`；默认 lane 不带
- 限额: 16 在飞响应（超出排队）/ 32 命名 lane（第 33 报 `websocket_stream_limit_reached`）/ 60 分钟连接寿命
- 连接级错误不带 `stream_id`；自产事件（`steer.*`/`inject.*`）序号用连接级计数器

### Steering

- 目标在下次 decode 步停表（`STOP_TYPE_STEERED`）→ `incomplete`/`steered` → 自动 successor
- `too_many_pending_steers` 阈值 32/目标
- WS-only events: `response.steer.accepted`/`.pending`/`.failed`

### `response.inject`（beta）

- 校验后回 `response.inject.created`；注入项由目标 successor 承接
- 失败码: `response_already_completed` / `response_not_found`
- Schema 不合规 → `error`(400) + 关闭连接

### `store=false` 连接本地缓存

- 进程内 LRU-256（`{连接令牌, prepared input, output}`）
- 只能由签发它的同一条连接作 `previous_response_id` 继续
- 同 lane 续写失败驱逐父；跨 lane fork 失败保留父
- 未命中: WS → 嵌套 `error` 400 `previous_response_not_found`；HTTP → 扁平信封同 code

## Recorded deviations

- `response.reasoning_text.delta`/`.done` 与 `response.refusal.delta`/`.done` 不发射（reasoning 只暴露 summary）
- SSE `error` / `response.failed` 默认配置不可达（流开始前错误按 HTTP 错误体返回）
- `prompt_cache_diagnostics` 本地简化（见 [common.md](common.md)）
- WS 未知 model: 400 + `model_not_found`（B 族 message）
