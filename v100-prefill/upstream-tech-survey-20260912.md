# V100 / llama.cpp 可拿来开发的技术调研（2026-09-12）

背景：等内核构建期间，盘点「大模型最新技术里哪些可以拿过来进行开发」。
本机约束：V100（sm_70；无 cp.async / ldmatrix / fp8）、llama.cpp fork（Responses 分支）、本地模型 Qwen3.5-9B / Qwen3.8-27B（qwen35 hybrid）。

## 0. 上游同步现状（已核实，git fetch + ancestry 检查）

- fork 已合并到上游 **2026-09-11 15:50（8172e6577）**；本次 `git fetch origin master` 后，上游只多 6 个提交（hexagon 多卡、webgpu Dawn、server subproc、MSVC cmake、CI、tests），**没有 CUDA/FA 相关新改动可拿**。
- 近 3 周上游的关键 CUDA/FA/模型提交**均已在树内**：XOR swizzle 内核 (e4b9af007)、f16 FA divergent barrier 修复 (b74f590ea)、sparse-FA（DSV4/GLM/qwen4exp）(8e93a9773)、GGML_FA_QUANTS (5a4d0feca)、GDN max→rsqrt 修复 (5fdfa6282)、Qwen3.5 recurrent_layers (9a7570587)、kv-cells 解码优化 (b356fa262)、Q4_K/Q5_K branchless MMVQ (73ab7599b)、Qwen3.8-Flash-Next (qwen4exp, 6c84c7d5d) 等。
- 结论：**不存在"上游有新内核可以直接白嫖"的空间**。可拿来的分三类：① 本机可用但没开的功能（配置级，零代码）；② V100 缺口（要开发才能用）；③ 我们自己的内核 R&D 清单（当前线继续）。

## 1. 配置级：不写代码就能用

| 项 | 说明 | 证据 | 状态 |
|---|---|---|---|
| **MTP 推测解码** | `llama-server --spec-type draft-mtp`（可配 `--spec-draft-n-max`）。本地 9B/27B GGUF **自带 nextn 头**（`nextn_predict_layers` 键 + `nextn.*` 张量齐全），qwen35 运行时支持加载 MTP 层，且传 `draft-mtp` 时自动 `load_mtp=true`，**无需额外 draft 模型** | docs/speculative.md:369；common/common.cpp:1714；src/models/qwen35.cpp:38；两个 GGUF 头部扫描 | ⚠ 需实测 tg 收益 |
| n-gram 系推测 | `--spec-type ngram-mod / ngram-map-k4v / ngram-simple ...`（`--spec-default` 直接开 ngram-mod）；零模型成本、ngram-mod 哈希池跨 slot 共享；适合代码/重复文本场景 | docs/speculative.md | ⚠ 需实测 |
| backend sampling | `--backend-sampling`：受支持的采样器搬到 GPU 上跑（可选 `--spec-draft-backend-sampling`） | common/arg.cpp:2303 | ⚠ 需实测 |
| EAGLE-3 / DFlash / DSpark drafts | 上游已实现三种 draft 类型 + HF 自动下载；公开 draft 以 Qwen3 系为主（如 `z-lab/Qwen3-4B-DFlash`、`deepseek-ai/dspark_qwen3_4b_block7`、`AngelSlim/Qwen3-*-eagle3`） | docs/speculative.md | 待查是否有 Qwen3.5/3.8 对应 draft |
| kv-cells（解码） | 上游实测 Qwen3.8-Flash-Next @71k：tg 69.3 → 72.7（+4.9%），pp 不变；已在树，服务器重建即得 | b356fa262 提交说明 | ✅ 已含 |
| GDN 归一化修复 | qwen35/qwen3next 的 GDN 层 max→rsqrt（正确性修复） | 5fdfa6282 | ✅ 已含 |

注：上表 1-4 项都只是 llama-server 参数，落点在 `models.ini` 对应模型配置；服务器按用户指示暂不启动，等恢复时再验证。

## 2. V100 缺口（要"开发"才能用）

| 项 | 现状 | 开发点 |
|---|---|---|
| **sparse-FA**（DSV4 / GLM / Qwen3.8-Flash-Next 用） | `ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse` 要求 `turing_mma_available(cc)` ⇒ **V100 上被关闭，退化为稠密注意力**（带 indexer 的模型在 V100 拿不到稀疏收益） | 给 Volta 打开 sparse gather：Volta 无 cp.async / 无多级加载（相关 static_assert 不冲突），但需移植 gather 逻辑并过正确性门禁；仅当要在 V100 跑这类模型才有意义 |
| FA2 / FA3 / FA4 | 官方不支持 sm_70（已评估：移植不值得，见 fa2-volta-port-assessment） | 不再投入 |
| 新 arch 支持（qwen4exp 等） | 已在树 | 无需动作 |

## 3. 当前线的内核 R&D 清单（V100 预填）

| 编号 | 想法 | 依据 / 前提 | 备注 |
|---|---|---|---|
| V-B | `(256,256,64, 256,1, 64,128,128,64,1,false)`：nbatch_fa 32→64 ⇒ 迭代与同步再减半；smem ≈72,192（1 CTA 内）；KQ_C +8 regs | V-A 结果出来后决定是否试 | **⏸ 降级（2026-09-12 晚更新）**：V-A 实测失败（墙钟 −5~−9%、内核 +52.4%），归因=单 CTA/SM 失去延迟掩盖 + MIO 突发 + 指令膨胀（详见台账 §10）；V-B 只会把同一方向推得更远（smem 更大、仍 1 CTA/SM），不推荐继续 |
| Volta XOR swizzle | 现有 swizzle 被 `TURING_MMA_AVAILABLE` 门禁（fattn-swizzle.cuh:16）；给 LDS 仿真路径设计 XOR 映射可降 bank 冲突 | 先测 e3d/V-A 的 bank conflict 基数（指标已加入 A/B 脚本）再动手 | **❌ 降级（2026-09-12 晚更新）**：基数已测——e3d ld 冲突占 ld wavefronts 仅 **3.2%**（8.93M/280.9M），swizzle 收益上限只有几个百分点，不再优先 |
| E2 复活 | 寄存器预取下一 K tile（唯一无 cp.async 的软流水方案） | e3d 254/255 无空间；V-A 若释放 ~12 regs 即可试 | 候选（V-A 实测 253 regs，仅多释放 2 个且已回退 ⇒ 仍不满足前提） |
| warp specialization | Volta 用命名屏障做 producer/consumer（FA3 思路的无 TMA/cp.async 适配） | 大改（数百行） | 🔬 推断 |
| L2 / CTA 栅格化 | e3d 的 dram read 已降到 1.58 GB；是否还有 L2 复用空间需先测 CTA 排布 | 低优先 | 🔬 推断 |

## 4. 已否 / 不适用（避免重复投入）

- FA2 移植（结论：不值得，V100 上性能上限与现内核同级）
- FA3/FA4 特性（TMA / warpgroup / fp8 均为 Hopper+ 硬件）
- 用 GTX 1650 Ti 分担计算（带宽/显存/PCIe 互联不划算）
- ncols>64 新模板实例（项目规范：新增实例需 maintainer 同意，不擅自加）
- GGML_FA_QUANTS 收窄编译（**澄清**：它只作用于 vec 内核实例；mma-f16 内核与 K/V 类型无关、恒编译，收窄它对我们的重建时长无收益）

## 5. 工程事实更新（供引用）

- 内核重建实测：改 `fattn-mma-f16.cuh` 后重编「21 个 fattn 实例 TU + fattn.cu」= **4 分钟**（2026-09-12 12:10:42 → 12:14:42，`-j7` + ccache，空闲机器；证据 `/root/llama.cpp/v100-prefill/fa-logs/build-va-27b.log`）。历史"25-35 分钟"估计作废（当时的并发负载/参数不同）。
- ccache 已启用（构建日志：`ccache found, compilation results will be cached`）。
- ncu 可用的 bank conflict 指标：`l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` / `_op_st.sum`（已加入 `27b-fa-ab-va.sh`，用于给 Volta swizzle 决策提供基数）。
