# Anthropic Messages API (`/v1/messages`)

口径：**合理加深**——把官方公开字段接到本地既有机制上，不宣称完整兼容。

官方基准：`/root/anthropic-docs`（`llms-full.txt` 全量正文 628 页）。

实现落点：`server_chat_convert_anthropic_to_oai`（`server-chat.cpp`）、`to_json_anthropic` / `to_json_anthropic_stream`（`server-task.cpp`）、SSE 错误帧（`server-context.cpp`）。

## 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/v1/messages` | 创建 |
| POST | `/v1/messages/count_tokens` | Token 计数 |

## 请求侧

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

## 仍不生效 / 近似

- **不生效**: `tools[].strict` / `input_examples`（接受即忽略）
- **近似**: `service_tier`/`container`/`inference_geo` 请求侧只校验不参与调度；响应侧恒 0/null；`cache_control` 在 `tool_use`/`tool_result` 为消息级断点（非 part 级）；`request_id` 恒 null 且不发响应头

## 运行期验证

已验证路径（单模型 9B）：
- 错 `x-api-key` / 无鉴权 → 401 `{"type":"error","error":{"type":"authentication_error",...}}`
- 缺 `max_tokens` → 400 `{"type":"error","error":{"type":"invalid_request_error","message":"'max_tokens' is required"}}`
- `count_tokens` 不带 `max_tokens` → 200
- 启动窗口 → 529 `overloaded_error`

未触发（登记）：SSE error 帧在默认配置下不可达（超上下文在流开始前以 400 JSON 返回）。

## 验收

单测 `unit/test_compat_anthropic.py`（44 条）+ `unit/test_security.py`（30 条）：合并 74 passed / 4 deselected。
