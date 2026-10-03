# Anthropic Messages API (`/v1/messages`)

口径：**合理加深**——把官方公开字段接到本地既有机制上，不宣称完整兼容。

官方基准：`/root/anthropic-docs`（`llms-full.txt` 全量正文 628 页）。

实现落点：`parse_anthropic_to_surface_request`（`server-chat.cpp`，转换逻辑内联）、`to_json_anthropic` / `to_json_anthropic_stream`（`server-task.cpp`）、SSE 错误帧（`server-context.cpp`）。

架构（v100 API Surface Split）：面层独立转换，产出 `server_surface_request`；内核单份共享，不再依赖 `__oai_*` 私有键穿层。新增 `parse_anthropic_to_surface_request`（`server-chat.cpp`）作为面→内核入口。

## 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/v1/messages` | 创建 |
| POST | `/v1/messages/count_tokens` | Token 计数 |

## 请求侧

### Official field constraints

SDK 基准: `anthropic-sdk-python` main 分支 `MessageCreateParams` / `Message`。

| 字段 | 官方约束 |
|------|----------|
| `model` | 必填，模型标识符 |
| `max_tokens` | 必填，≥0（`0` = 仅预热缓存）；各模型有各自上限 |
| `messages` | 必填，≤100,000 条；必须 `user`/`assistant` 交替 |
| `system` | `string` 或 `TextBlockParam[]` |
| `temperature` | 已弃用（Opus 4.6 后不支持，仅接受 `1.0`，其余 → 400） |
| `top_p` | 0..1 |
| `top_k` | ≥0 |
| `stop_sequences` | string 数组，不能为单 string |
| `service_tier` | `auto`/`standard_only` |
| `thinking.type` | `enabled`/`adaptive`/`disabled` |
| `thinking.budget_tokens` | ≥1024 且 < `max_tokens` |
| `output_config.effort` | `low`/`medium`/`high`/`xhigh`/`max` |
| `tool_choice.type` | `auto`/`any`/`tool`/`none` |
| `cache_control.type` | 必须 `ephemeral`；`ttl`: `5m`/`1h` |
| `metadata.user_id` | ≤512 字符，禁止 PII |
| `diagnostics.previous_message_id` | 启用 `cache_miss_reason` 诊断 |
| `container` | 跨请求容器标识或参数对象 |

### 本地行为映射

| 字段 | 本地行为 |
|------|----------|
| `tool_choice{type:"tool",name}` | 转发为命名强制 function tool；名字不在 `tools` 中 → 400 |
| `tool_choice{type:"none"}` | 转发 `"none"` |
| `tool_choice.disable_parallel_tool_use` | → `parallel_tool_calls=false` |
| `output_config.effort` | → 本地 `reasoning_effort`（5 值 low/medium/high/xhigh/max）；越界 → 400 |
| `output_config.format`（`json_schema`） | → 本地 `response_format` |
| `thinking{type:"disabled"}` | → `reasoning_effort="none"`；与 `output_config.effort` 并存时 disabled 胜 |
| `thinking{type:"adaptive"}` | 不设预算，交由 effort/本地缺省决定 |
| `thinking.display` | `summarized`（缺省）/`omitted`；越界 → 400 |
| `cache_control`（system/message part） | → `prompt_cache_breakpoint{mode:"explicit"}`；`type` 必须 `ephemeral`、`ttl` 必须 `5m`/`1h` |
| 顶层 `cache_control` | 倒序找最后一条非 `tool` 消息的可缓存 part 置断点 |
| `cache_control.ttl` | `5m`/`1h` → 内部通道 `__prompt_cache_ttl`；其它 → 400。本地每请求单一 TTL |
| `cache_control` 落在 `tools[]` | 消息级断点，`anchor:"tools"` |
| `cache_control` 落在 `tool_use`/`tool_result` | 提升到消息级断点（近似，粒度更粗） |
| `max_tokens`（create 必填） | 缺/null → 400；非整数/负 → 400；`0` 按官方语义接受（只预热缓存） |
| `count_tokens` body | 无法定 `max_tokens`（豁免） |
| `service_tier`/`container`/`inference_geo` | 类型/枚举校验后丢弃 |

## 响应侧

| 字段 | 本地行为 |
|------|----------|
| `usage.cache_creation_input_tokens` | 恒 `0`（保住求和不变式） |
| `usage.output_tokens_details.thinking_tokens` | 非流式真值 `max(0, n_reasoning_tokens)` |
| thinking `content_block_start` | 补 `{"signature": ""}` |
| `container`/`stop_details` | 恒 `null` |
| `usage.service_tier` | 恒 `"standard"` |
| `usage.inference_geo` | 恒 `null` |
| `usage.cache_creation` | 恒 `{ephemeral_1h: 0, ephemeral_5m: 0}` |
| `usage.server_tool_use` | 恒 `{web_fetch_requests: 0, web_search_requests: 0}` |
| `message_delta.usage` | 累积 `input_tokens`/`output_tokens`/`cache_creation`(0)/`cache_read` |
| SSE `error` 帧 | 官方包裹 `{"type":"error","error":{...}}` |
| 非流式错误体 | 官方包裹 `{"type":"error","error":{type,message},"request_id":null}`；503 → 529 `overloaded_error` |

## Message 响应对象

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 唯一标识 |
| `type` | `"message"` | 固定 |
| `role` | `"assistant"` | 固定 |
| `model` | string | |
| `content` | array | `ContentBlock[]`（`text`/`tool_use`/`thinking`/`redacted_thinking`/`server_tool_use`） |
| `stop_reason` | string? | 见下表 |
| `stop_sequence` | string? | 触发停止的自定义序列 |
| `usage` | object | 见下 |
| `container` | object? | 本地恒 `null` |
| `stop_details` | object? | 本地恒 `null` |
| `diagnostics` | object? | `cache_miss_reason` 等 |

**stop_reason 枚举**: `end_turn`/`max_tokens`/`stop_sequence`/`tool_use`/`pause_turn`/`refusal`/`model_context_window_exceeded`。

**Usage**: `input_tokens`/`output_tokens`/`cache_creation_input_tokens`/`cache_read_input_tokens`/`output_tokens_details.thinking_tokens`/`service_tier`/`inference_geo`/`cache_creation`/`server_tool_use`。

## SSE 事件类型

| 事件 | 数据 |
|------|------|
| `message_start` | `{type:"message_start", message:{...content:[]}}` |
| `content_block_start` | `{type, index, content_block:{type,...}}` |
| `content_block_delta` | `{type, index, delta:{type:"text_delta"/"input_json_delta"/"thinking_delta"/"signature_delta",...}}` |
| `content_block_stop` | `{type, index}` |
| `message_delta` | `{type, delta:{stop_reason,stop_sequence}, usage:{output_tokens,...}}` — usage 为累积值 |
| `message_stop` | `{type:"message_stop"}` |
| `ping` | `{type:"ping"}` |
| `error` | `{type:"error", error:{type,message}}` |

## 错误词表

| HTTP 状态 | `error.type` |
|-----------|-------------|
| 400 | `invalid_request_error` |
| 401 | `authentication_error` |
| 402 | `billing_error` |
| 403 | `permission_error` |
| 404 | `not_found_error` |
| 409 | `conflict_error` |
| 413 | `request_too_large` |
| 429 | `rate_limit_error` |
| 500 | `api_error` |
| 504 | `timeout_error` |
| 529 | `overloaded_error` |

错误体格式: `{"type":"error","error":{"type":"...","message":"..."},"request_id":"..."}`。

## 仍不生效 / 近似

- **不生效**: `tools[].strict` / `input_examples`（接受即忽略）
- **近似**: `service_tier`/`container`/`inference_geo` 请求侧只校验不参与调度；响应侧恒 0/null；`cache_control` 在 `tool_use`/`tool_result` 为消息级断点（非 part 级）；`request_id` 恒 null 且不发响应头
- **stop_reason 本地实际**: 仅产出 `end_turn`/`max_tokens`/`tool_use`（不产出 `pause_turn`/`refusal`/`model_context_window_exceeded`）
- **错误类型映射**: 本地 `exceed_context_size_error`/`not_supported_error` → `invalid_request_error`(400)；`service_unavailable_error` → `overloaded_error`(529)；未识别 → `api_error`(500)

## 运行期验证

已验证路径（单模型 9B）：
- 错 `x-api-key` / 无鉴权 → 401 `{"type":"error","error":{"type":"authentication_error",...}}`
- 缺 `max_tokens` → 400 `{"type":"error","error":{"type":"invalid_request_error","message":"'max_tokens' is required"}}`
- `count_tokens` 不带 `max_tokens` → 200
- 启动窗口 → 529 `overloaded_error`

未触发（登记）：SSE error 帧在默认配置下不可达（超上下文在流开始前以 400 JSON 返回）。

## 验收

单测 `unit/test_compat_anthropic.py`（44 条）+ `unit/test_security.py`（30 条）：合并 74 passed / 4 deselected。
