# Completions API (`/v1/completions`) — Legacy

## 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/v1/completions` | 创建 |

## Create 字段

| Kind | 字段 |
|------|------|
| **正向行为** | `model`/`prompt` required（`prompt` 省略 → 400）, `stream`, `stream_options.include_usage`（常规 chunk `usage: null` + 终块统计）, `stream_options.include_obfuscation`, 采样参数经 schema, `n` 多 completion（上限 `min(n_parallel, 128)`，越界 → 400）, `user` 回显, `echo=true` → choice.text 含 prompt 前缀, `best_of` ≥ `n`（按 token logprob 求和排名后返回前 `n`；上限 `min(n_parallel, 20)`）, 非空 `suffix` → FIM（有 FIM vocab 走 `format_prompt_infill`，否则 soft FIM prompt）, `seed` 确定性, `logit_bias` 负偏置, `logprobs` 官方形状（见下） |
| **形状校验** | `user`/`stream_options` 类型校验；`include_obfuscation` 非布尔 → 400 |
| **Model 校验** | 不校验（echo served name；0 实拍，维持现状） |

## `logprobs`

官方 Completions 形状：`tokens` / `token_logprobs` / `top_logprobs` / `text_offset`（非 Chat `content[]`）。

- 有效范围 0..5：`>5` 夹到 5（HTTP 200）、负数/非整数 → 400
- `top_logprobs` 保留"选中 token 不在 top-k 时补入"行为（官方允许 `logprobs+1` 键）

## `echo=true` + `logprobs>0`（非流式 + 流式同形）

`choices[].logprobs` 四数组前置 prompt 分片行，其后为生成 token 行：
- 第 0 行（首个 prompt token）的 `token_logprobs[0]` 与 `top_logprobs[0]` 为 `null`
- 其余 prompt 行真实值：行 p+1 取自位置 p 的 logits
- 行数跟随分词后的 prompt（非 raw 字符串）
- `text_offset` 原点 = 完整 text 开头（UTF-8 字节）
- 流式：首块并进 prompt 文本与 prompt 行；`text_offset` 跨块连续累计；末事件不重复 prompt

### 代价

- 该类请求禁用 prompt 前缀复用与后端采样
- `n_outputs_max` 抬高（上界 `n_batch * n_vocab * 4` 约 2.0 GB；实测常驻增量 +1812 MiB）

## `best_of`

- `best_of < n` → 400 `'best_of' must be greater than or equal to 'n'`
- 生成 `best_of` 候选，按 token logprob 求和排名后返回前 `n`
- 上限 `min(n_parallel, 20)`，越界 → 400
- `best_of>n` + `stream=true` → 400（排名需全部候选）

## FIM

非空 `suffix` → FIM：
- 有 FIM vocab → `format_prompt_infill`
- 无 FIM vocab → soft FIM prompt（非真 infill token）

## Recorded deviations

- Legacy `/v1/completions` 未知 model: 不校验（0 实拍，维持现状）
- `best_of`/`n` 上限: 官方 OpenAPI 声明（`best_of` ≤ 20, `n` ≤ 128）与本地槽数取小
