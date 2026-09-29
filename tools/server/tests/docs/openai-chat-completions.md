# Chat Completions API (`/v1/chat/completions`)

## 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/v1/chat/completions` | 创建 |
| GET | `/v1/chat/completions` | list（store CRUD） |
| GET | `/v1/chat/completions/{id}` | retrieve |
| POST | `/v1/chat/completions/{id}` | update |
| DELETE | `/v1/chat/completions/{id}` | delete |
| GET | `/v1/chat/completions/{id}/messages` | 列出消息 |
| POST | `/v1/chat/completions/input_tokens` | Token 计数 |
| POST | `/v1/chat/completions/control` | 实时控制 in-flight 完成 |

## Official field constraints

SDK 基准: `openai-python` main 分支 `ChatCompletionCreateParams` / `ChatCompletion`。

| 字段 | 官方约束 |
|------|----------|
| `temperature` | 0..2 |
| `top_p` | 0..1 |
| `top_logprobs` | 0..20（需 `logprobs=true`） |
| `frequency_penalty`/`presence_penalty` | -2..2 |
| `stop` | string 或 ≤4 条 string 数组 |
| `metadata` | ≤16 键值对，键≤64 字符，值≤512 字符 |
| `safety_identifier` | ≤64 字符 |
| `service_tier` | `auto`/`default`/`flex`/`scale`/`priority`/`fast`（默认 `auto`） |
| `reasoning_effort` | `none`/`low`/`medium`/`high`/`max` |
| `verbosity` | `low`/`medium`/`high`（默认 `medium`） |
| `max_completion_tokens` | 替代已弃用的 `max_tokens`；与 o 系列不兼容的 `max_tokens` 已弃用 |
| `prompt_cache_retention` | 已弃用，改用 `prompt_cache_options.ttl`；值 `in_memory`/`24h` |
| `logit_bias` | 键为 token ID，值 -100..100 |
| `modalities` | `["text"]`/`["audio"]`/`["text","audio"]` |
| `store` | bool |
| `moderation` | 输入输出审核（o 系列支持） |
| `response_format` | `text`/`json_object`/`json_schema` |

## ChatCompletion 响应对象

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 唯一标识 |
| `object` | `"chat.completion"` | 固定 |
| `created` | int | Unix 秒 |
| `model` | string | |
| `choices` | array | `Choice[]`（`n` 决定长度） |
| `usage` | object? | `CompletionUsage` |
| `system_fingerprint` | string? | 后端配置指纹 |
| `service_tier` | string? | |
| `metadata` | object? | ≤16 KV |
| `moderation` | object? | |

**Choice**: `index`/`message`/`finish_reason`/`logprobs`。`finish_reason`: `stop`/`length`/`tool_calls`/`content_filter`/`function_call`。

**CompletionUsage**: `prompt_tokens`/`completion_tokens`/`total_tokens`/`completion_tokens_details`/`prompt_tokens_details`。

## Create 字段

| Kind | 字段 |
|------|------|
| **正向行为** | `model`（必填）, `messages`（必填）, `store`+metadata retrieve, `n` 多 choice（上限 `min(n_parallel, 128)`）, `stream`, `stream_options.include_usage`（常规 chunk `usage: null` + 终块统计）, `stream_options.include_obfuscation`, 非流式 `refusal`（无拒答时 `null`）, `seed` 确定性, `frequency_penalty`/`presence_penalty` [-2,2], `modalities=["text"]`, `web_search_options` → 本地搜索 + `message.annotations`(`url_citation`), `user`/`safety_identifier`（`store=true` 时持久化）, legacy `functions`+`function_call`, `reasoning_effort`（budget 阶梯）, `verbosity` system hint, `prediction.type=content` 预填 assistant, `prompt_cache_key`/`prompt_cache_retention`/`prompt_cache_options`, `logprobs`+`top_logprobs`（0..20，超出 → 400）, `prompt_cache_breakpoint`, `response_format`（`text`/`json_object`/`json_schema`）, `max_completion_tokens`（替代 `max_tokens`）, `moderation`（接受不生效） |
| **形状校验** | `audio`/含 `audio` 的 modalities → 400；非法 `prompt_cache_*` → 400 |
| **正向补充** | `logit_bias` 负偏置抑制目标 token |
| **Model 校验** | 未知 → 404 + A 族 |

## Store CRUD

- Durable under `--openai-files-path/chat_completions/`（memory if path unset）
- `GET .../messages`: `after`/`limit`/`order`（默认 `asc`）；one row per choice with `msg_` id
- `user`/`safety_identifier` 在 `store=true` 时持久化 retrieve

## 工具

### Custom tools

- 定义: `{type:"custom", custom:{name, description?, format?}}`
- `tool_choice`: `{type:"custom", custom:{name}}` 强制
- 输出: `{id, type:"custom", custom:{name, input}}`
- 回放: `tool_calls[].type=="custom"`
- 流式: `delta.tool_calls[]` 带 `type:"custom"` + `custom.name`/`custom.input` 增量

### `allowed_tools` 嵌套

`{type:"allowed_tools", allowed_tools:{mode:"auto"|"required", tools:[...]}}`：按子集过滤可用工具；`required` 无匹配 → 400；`auto` 过滤后空集 → 400。

### Legacy 兼容

- `role:"function"` → 渲染前映射 `role:"tool"`
- assistant `function_call` → 规范化为 `tool_calls` 回放
- assistant `content:[{type:"refusal",...}]` 回放
- `file` part: `file_data` data URI 接受, `file_id` → 400
- `input_audio.id` → 400

## Web search（Chat 形态）

- `web_search_options` → 本地搜索注入 system + `message.annotations`(`url_citation`)
- `filters.allowed/blocked_domains`（同 Responses）
- 流式终块携带 `message.annotations`

## Usage

- `completion_tokens_details` 恒 5 字段: `reasoning_tokens`（真计数）, `text_tokens`, `accepted_prediction_tokens`/`audio_tokens`/`rejected_prediction_tokens`（= 0）

## Recorded deviations

- `n` 超过 `min(n_parallel, 128)` → 400（本地容量上限）
- `logprobs`+`tools`+`stream` 组合 → 400（官方未禁止，本地限制）
