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

## Create 字段

| Kind | 字段 |
|------|------|
| **正向行为** | `model`（必填）, `messages`（必填）, `store`+metadata retrieve, `n` 多 choice（上限 `min(n_parallel, 128)`）, `stream`, `stream_options.include_usage`（常规 chunk `usage: null` + 终块统计）, `stream_options.include_obfuscation`, 非流式 `refusal`（无拒答时 `null`）, `seed` 确定性, `frequency_penalty`/`presence_penalty` [-2,2], `modalities=["text"]`, `web_search_options` → 本地搜索 + `message.annotations`(`url_citation`), `user`/`safety_identifier`（`store=true` 时持久化）, legacy `functions`+`function_call`, `reasoning_effort`（budget 阶梯）, `verbosity` system hint, `prediction.type=content` 预填 assistant, `prompt_cache_key`/`prompt_cache_retention`/`prompt_cache_options`, `logprobs`+`top_logprobs`（0..20，超出 → 400）, `prompt_cache_breakpoint` |
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
