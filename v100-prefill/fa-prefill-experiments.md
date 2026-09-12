# Qwen3.5-9B / Qwen3.8-27B / V100 预填充（FlashAttention 内核）实验记录

用途：记录「把 V100 预填充速度拉上去」这条线上的所有实测数据。
规则：
1. **每条数据都带来源文件 + 会话/日期**；跨会话绝对耗时不可直接比（本机时钟漂移实测 2.4-6%，今天上午的漂移 <1%）；只有同会话同二进制可直比。
2. **每个 ncu 报告文件名带配置前缀**，且引用前用文件内特征值（smem / warps_active / duration）反查身份，禁止凭记忆归类。
3. 二进制身份：`build/bin/llama-bench` 只是 17KB 启动器，**真正的内核在 `build/bin/libggml-cuda.so.0.23.0`**；每次测完记录该 .so 的行为特征（smem 大小最能区分配置）。

## 0. 环境与协议

| 项 | 值 |
|---|---|
| GPU | Tesla PG503-216（V100 32GB，**72 SM**，boost 上限 1530 MHz / ncu 采样期间实测均值 ~1.31 GHz（`sm__cycles_elapsed÷duration`），smem 96 KB/SM，L2 6 MB，HBM2 **1133 GB/s**（mem clk 1107 MHz × 4096-bit）；以上为 2026-09-12 本机 `cudaDeviceProp` 实测，证据 `/tmp/mma-peak.log`） |
| 第二块卡 | GTX 1650 Ti（cc 7.5）——**必须** `CUDA_DEVICE_ORDER=PCI_BUS_ID`，否则 `-dev CUDA1` 会落到它上面 |
| 模型 | `/root/llama_gguf/Qwen3.5-9B-Q4_K_M/Qwen3.5-9B-Q4_K_M.gguf`（qwen35，9.20B，32 层 = 8 全注意力 + 24 SSM，n_head 16 / n_head_kv 4，head_dim 256） |
| 源码/构建 | 实验基线 HEAD = 8def8f970（当时干净树）；本实验改动（`fattn-mma-f16.cuh` 的 Volta 配置行，见 §8）已提交为 **19a667d32**（本地 commit，未推送）；`CMAKE_CUDA_ARCHITECTURES=70-real`，`GGML_CUDA_FA_ALL_QUANTS=ON` |
| 编译 | `cmake --build /root/llama.cpp/build --target llama-bench -j6`（改 `fattn-mma-f16.cuh` 会重建 21 个 fattn 实例 TU，实测 25-35 分钟） |
| 被 profile 的内核 | `flash_attn_ext_f16<256, 256, 16, 4, false, false, false>`，block (32,4,1)=128 线程，grid (1024,1,1) |
| 墙钟命令 | `CUDA_DEVICE_ORDER=PCI_BUS_ID ./build/bin/llama-bench -m <模型> -dev CUDA1 -fa 1 -ctk q8_0 -ctv q8_0 -b 4096 -ub 4096 -p 4096 -d 0,16384,32768 -n 0 -r 2` |
| ncu 命令 | `CUDA_DEVICE_ORDER=PCI_BUS_ID ncu -k "regex:flash_attn_ext_f16" --launch-skip 39 -c 1 --metrics <list> --csv ./build/bin/llama-bench -m <模型> -dev CUDA1 -fa 1 -ctk q8_0 -ctv q8_0 -b 4096 -ub 4096 -p 4096 -d 16384 -n 0 -r 1` |
| 纪律 | 同一时刻只跑一个 llama-bench（显存不够）；**跑测前先清场：结束旧的/在跑的干扰进程（含模型服务），排除干扰源**（2026-09-12 用户指示，取代早前"不动正在跑的进程"）；改配置实验一次只动一个变量 |
| **A/B 协议（2026-09-12 建立）** | 内核库按配置备份到 `/tmp/so-<标签>.so`，再放进 `/tmp/falib-<标签>/`（内含 `libggml-cuda.so` / `.so.0` / `.so.0.23.0`）；`llama-bench` 用 **RUNPATH** 而非 RPATH，故 `LD_LIBRARY_PATH=/tmp/falib-<标签>` 可优先加载指定内核库，**完全不动 `build/bin`**（避免实验期间的 .so 影响用户重启服务器）；对比时按 A/B/A/B 轮换跑，消除时钟漂移。（库快照位置后由 `/tmp` 迁到 **`/root/llama.cpp/v100-prefill/fa-libs/<标签>/`**，用法不变） |
| 已知备份内核库 | `stock`（8def8f970 干净树，sha256 `754fbf6f5aa7…`，167,425,512 B）；`E1`（sha256 `d3ff2615084c…`）；**`E3-(d)`（sha256 `e26604048a3a…`，= 当时的 `build/bin`；后续快照与当前基线见 §16.5）**。启动器 `build/bin/llama-bench` sha256 `fd90ef249ae5…`（全程未变）。**注：2026-09-12 按用户要求清理 `/tmp`**；A/B 内核库快照现统一存放于 `/root/llama.cpp/v100-prefill/fa-libs/<标签>/`（现有 `stock`、`e3d`、`deltanet-pipe`、`mmq-exp`、`cublas-off`、`mmq-gate`，用法同上一行），全部可由 git 重建（stock = 8def8f970 树；E1 = 其上加 `Q_in_reg=false`） |
| 证据归档 | 2026-09-12 起全部日志/证据由 `/tmp` 迁至 **`/root/llama.cpp/v100-prefill/fa-logs/`**（同名文件；**本文各处引用的 `/tmp/<name>` 均按此映射**；清单见该目录 `.archived-list.txt`）；未引用的 nsys 原始 profile（`pp8192-prof.*`）已直接删除 |

**2026-09-12 晚目录/分支调整（用户决定）**：整套资料由 `/root/llama_gguf/` 迁入本仓库 `v100` 分支的 `v100-prefill/`（本文件所在处），文内路径已同步改写。三个 V100 提交已重放到 `v100` 分支：`19a667d32` → `b899cd647`（FA 调优）、`3148fd037` → `f1803060d`（delta-net 流水线）、`eecfa404f` → `6f81ea836`（MMQ 门槛），`git range-diff` 校验逐补丁一致；Responses 已复位到 `8def8f970`。本文出现的以上旧哈希均为当时的 Responses 版本。`fa-libs/` 六份库快照本地保留、不入 git（`.git/info/exclude` 排除；单文件 167 MB，超 GitHub 100 MiB 上限）。

## 1. 墙钟吞吐（llama-bench，t/s；ub=4096、KV=q8_0、fa=1）

| 日期/会话 | 配置 | pp4096 (d0) | d16384 | d32768 | d65536 | d262144 | 来源文件 |
|---|---|---|---|---|---|---|---|
| 2026-09-12 02:42 | stock | 3328.26 | 2397.91 | - | 1296.33 | 456.95 | `/tmp/lb-stock-q8-depth.log`（历史参考） |
| 2026-09-12 08:13 | stock | 3327.81 ± 27.35 | 2401.99 ± 9.86 | - | 1300.17 ± 3.35 | 被叫停 | `/tmp/lb-ab-stock.log`（256k 轮次按用户要求取消） |
| 2026-09-12 08:18 | **stock（A 侧基线）** | 3294.69 ± 23.93 | 2382.60 ± 8.46 | 1854.81 ± 2.72 | - | - | `/tmp/lb-ab-stock-32k.log` |
| 2026-09-12 08:31 | **E1**：256/256/ncols64 用 `Q_in_reg=false` | **3385.22 ± 34.22** | **2667.72 ± 11.10** | **2195.80 ± 2.33** | - | - | `/tmp/lb-e1.log` |
| 2026-09-12 08:47 | E1 vs stock **同会话 A/B/A/B 轮换**：stock | 3329.20 / 3252.86 | 2396.07 / 2377.94 | 1869.52 / 1852.76 | - | - | `/tmp/fa-ab-swap.log` |
| 2026-09-12 08:47 | 同轮换：**E1** | 3351.71 / 3324.46 | 2648.49 / 2619.89 | 2179.48 / 2166.92 | - | - | `/tmp/fa-ab-swap.log` |
| 2026-09-12 10:27 | **E3-(d)**：瓦片减半 `fa/K2/V2/combine = 32/64/64/64`，smem 43,776 ⇒ 2 CTA/SM | 3407.36 ± 23.03 | 2830.40 ± 8.74 | 2402.73 ± 10.13 | - | - | `/tmp/lb-e3d.log` |
| 2026-09-12 10:30 | E3-(d) vs **E1 同会话 A/B/A/B 轮换**：E1 | 3368.17 / 3326.40 | 2648.45 / 2625.15 | 2180.27 / 2168.91 | - | - | `/tmp/e3d-measure.log` |
| 2026-09-12 10:30 | 同轮换：**E3-(d)** | 3368.40 / 3342.01 | 2786.50 / 2758.56 | 2380.00 / 2363.89 | - | - | `/tmp/e3d-measure.log` |

**E1 vs stock（同会话 A/B 均值）**：d0 3291.03 → 3338.09（**+1.4%**）、d16384 2387.01 → 2634.19（**+10.4%**）、d32768 1861.14 → 2173.20（**+16.8%**）。
两侧各自两轮的离散度 ≤2.3%（d0）与 ≤1.1%（有深度点），提升远超噪声；与跨会话版本（+2.7% / +12.0% / +18.4%）一致到 1.5 个百分点内。

**E3-(d) vs E1（同会话 A/B/A/B 均值）**：d0 3347.29 → 3355.21（**+0.2%**，对照组：d0 不含 FA）、d16384 2636.80 → **2772.53（+5.15%）**、d32768 2174.59 → **2371.95（+9.08%）**。
两侧各自两轮离散度 ≤1.3%（d0）与 ≤1.0%（有深度点）；d0 基本重合说明两侧二进制除内核外完全等价、本会话噪声极小。

**累计 vs stock（同会话链式：stock→E1 @08:47，E1→E3-(d) @10:30）**：d0 1.014×1.002 = **+1.6%**、d16384 1.104×1.0515 = **+16.1%**、d32768 1.168×1.0908 = **+27.4%**。
与跨会话直接对比（3355.21/3291.03、2772.53/2387.01、2371.95/1861.14 = +2.0% / +16.2% / +27.5%）**吻合到 0.5 个百分点内**——两条独立路径互证。

### 1.1 清场复核（2026-09-12 11:05-11:14，服务全停、无干扰；`/tmp/cln-verify.log`）

服务整组停止（V100 显存 18,599 → 10 MiB）后，`stock / e1 / e3d` **三路 ×2 轮**同会话轮换（沿用 §0 的 A/B 协议与解析方式）：

| 侧（2 轮均值） | d0 | d16384 | d32768 |
|---|---|---|---|
| stock | 3306.22 | 2397.64 | 1874.08 |
| e1 | 3332.51 | 2643.61 | 2186.14 |
| **e3d** | **3340.08** | **2782.96** | **2367.54** |

| 比值 | 清场复核 | 带干扰时代 | 偏差 |
|---|---|---|---|
| e1 vs stock | +0.80% / +10.26% / **+16.65%** | +1.4% / +10.4% / +16.8% | ≤0.6pp |
| e3d vs e1 | +0.23% / +5.27% / **+8.30%** | +0.24% / +5.15% / +9.08% | ≤0.8pp |
| e3d vs stock（累计） | +1.02% / +16.06% / **+26.33%** | +1.6% / +16.1% / +27.4% | ≤1.1pp |

六个绝对数字的跨会话位移 **≤0.7%**（区间 −0.19% ~ +0.70%）⇒ **空闲共驻服务对本实验的结论与绝对数字影响 ≤0.7%，逻辑未被干扰**
（空闲服务不占 SM/带宽；ncu duration 本就是 base-clock 锁定、不受共驻影响，故内核级数字无需复测）。

**256k 实测点（e3d、清场、`/tmp/cln-256k-e3d.log`）**：`pp4096 @ d262144` = **769.87 t/s**（5.320 s/批）。
stock 侧 256k 长测按用户指示取消（**追踪问题点只用 32k**）；stock 的 256k 引用历史实测 456.95 t/s（清场位移 ≤0.7% 已量化）。

## 2. ncu 报告汇总（同一启动点：--launch-skip 39、-c 1、grid (1024,1,1)、block 128）

| 来源文件（时间） | 配置（判定依据） | smem/block | 占用 | duration | dram read | tensor% | issue% | l1tex% | 备注 |
|---|---|---|---|---|---|---|---|---|---|
| `fa-ncu.csv` (02:13) | 旧构建 0f4562718 + 干净表（smem 67,584） | 67,584 | 1 block（waves 14.22） | — | — | — | — | — | 未合并前的构建 |
| `fa-ncu3.csv` (02:21) | 8def8f970 + **nbatch_combine=64**（smem 34,816 ⇒ 2 blocks） | 34,816 | 2 blocks（7.92 warps/SM） | — | — | — | 15.04* | — | *One or More Eligible；No Eligible 84.96% |
| `fa-ncu5.csv` (07:57) | 同上（2 blocks，registers 255） | — | 12.37%（7.91 warps） | 53.61 ms | 19.43 GB | 19.72（active 口径） | 15.05 | 35.38 | long_sb 8.89 / wait 1.45 |
| `fa-ncu-occ2.csv` (07:59) | 同上（2 blocks，LaunchStats 全） | 34,816 | 2 blocks（7.91 warps） | 53.50 ms | 19.41 GB | 19.70 | 15.01 | 35.34 | Block Limit SM 32 / Reg 2 / Smem 2 / Warps 16 |
| `fa-ncu-ablate.csv` (08:05) | stock + KV 负载全钉 key0 | — | 6.25%（4 warps） | 43.37 ms | 0.565 GB | 24.25 | 17.95 | 39.54 | long_sb 2.21 |
| `fa-ncu4.csv` (02:22) | 8def8f970 干净表（duration 与 occ2/ablate 都不同） | — | — | 53.28 ms | — | 18.79（elapsed 口径） | — | — | LDS 波前 1.0499e9；bank conflict 81.77e6 |
| `ncu-stock-pipes.csv` (08:20) | **干净 stock（smem 67,584 ⇒ 1 block）** | 67,584 | 1 block（waves 14.22，4 warps） | **52.47 ms** | 2.38 GB | 49.2% 指令占比 | 15.20 | — | registers 255（顶格） |
| `ncu-stock-stalls.csv` (08:20) | 同上 | — | 6.25% | — | — | — | 15.20 | — | long_sb 3.21 / wait 1.42 / short_sb 0.36 / barrier 0.21 |
| `ncu-e1-pipes.csv` (08:34) | **E1（Q_in_reg=false；smem 67,584 = combine 项主导，未变）** | 67,584 | 1 block（waves 14.22，4 warps） | **34.53 ms** | 3.48 GB | 54.0% 指令占比 | 21.14 | — | registers **251**、LDS 波前 1.20e9 |
| `ncu-e1-stalls.csv` (08:34) | 同上 | — | 6.25% | — | — | — | 21.14 | — | long_sb **1.21** / wait 1.28 / short_sb 0.47 / barrier 0.18 |
| `ncu-e3d-pipes.csv` (10:27) | **E3-(d)（小瓦片组合 ⇒ smem 43,776、2 CTA/SM）** | **43,776** | 2 CTA（waves **7.11**，8 warps） | **23.97 ms** | 1.58 GB | 51.4% 指令占比 | **33.62** | — | registers **254**、LDS 波前 1.223e9 |
| `ncu-e3d-stalls.csv` (10:28) | 同上 | — | 12.33% | — | — | — | 33.62 | — | long_sb **0.90** / wait 1.23 / mio_throttle 0.64 / math_pipe 0.49 / barrier 0.47 |

**重要修正**：07:57-07:59 那次所谓「occ2 vs stock」的对比，两侧其实都是 2 blocks/SM（34,816 smem）的构建——当时的"stock 参考"二进制里已经带着 combine=64 的改动，所以那次 ≈2% 的差异是噪声，**不能**据此说"加 warp 无用"。真正的 1 block（67,584）vs 2 CTA/SM 干净对比后来由 **E3-(d) 补上**（§3.2，同会话同协议）：1 CTA 的 E1 34.53 ms vs 2 CTA 的 E3-(d) 23.97 ms（**−30.6%**）⇒ 2 CTA/SM 是主因；但注意 E3-(d) 同时把 K2/V2/combine 从 128 减到 64（多变量），不能把全部收益归于占用率，`warps_active` ×2 且 `long_sb` 1.21→0.90 支持"占用率为主因"。

## 3. 指令流分解（干净 stock，2026-09-12 08:20，`/tmp/ncu-stock-pipes.csv`）

总指令 `smsp__inst_executed.sum` = **2,887,841,792**（1.387M warp-迭代 × 2082 条；其中 1025 条/迭代是 HMMA）

| pipe | 指令数 | 占比 | 说明 |
|---|---|---|---|
| tensor (HMMA) | 1,421,869,056 | 49.2% | 与 `sm__inst_executed_pipe_tensor.sum` 一致 |
| LSU | 699,033,600 | 24.2% | 见下方 SASS 细分 |
| fp16 (HMUL2/HFMA2) | 267,911,168 | 9.3% | softmax 缩放/转换 |
| fma (FFMA) | 246,034,432 | 8.5% | rowsum/exp 归约 |
| alu | 177,164,288 | 6.1% | 整数与逻辑 |
| xu (MUFU 等) | 65,581,056 | 2.3% | exp2 等 |
| adu | 8,192 | 0.0% | |

SASS 访存细分：

| 指令 | 数量 | 占比 | 说明 |
|---|---|---|---|
| shared_ld (LDS) | 371,068,928 | 12.8% | K/V/Q 的 ldmatrix 仿真（Volta 每条 = 2×LDS.64） |
| shared_st (STS) | 33,726,464 | 1.2% | K/V tile 与 KQ 结果 |
| global_ld (LDG) | 33,591,296 | 1.2% | KV tile |
| local_ld (LDL) | **142,987,264** | 5.0% | **寄存器溢出读** |
| local_st (STL) | **85,366,784** | 3.0% | **寄存器溢出写** |

其它：LDS 波前 1,023,412,499（load）+ 126,713,863（store）；`launch__registers_per_thread` = **255（顶格）**；dram read 2.38 GB / 52.47 ms ≈ **45 GB/s**（峰值 ~900 GB/s ⇒ 5%，带宽远没吃满）。

### 3.1 stock vs E1 逐项对比（同启动点、同协议，08:20 vs 08:34）

| 指标 | stock | E1 | 变化 |
|---|---|---|---|
| 内核 duration | 52.47 ms | **34.53 ms** | **−34.2%** |
| 总指令 `smsp__inst_executed.sum` | 2,887,841,792 | 2,633,771,008 | −8.8% |
| tensor（HMMA） | 1,421,869,056 | 1,421,869,056 | 0%（算法没变） |
| LSU | 699,033,600 | 563,019,776 | −19.5% |
| fp16 | 267,911,168 | 267,911,168 | 0% |
| fma | 246,034,432 | 205,342,720 | −16.5% |
| alu | 177,164,288 | 99,823,616 | −43.7% |
| xu | 65,581,056 | 65,519,616 | ≈0 |
| **local_ld（溢出读）** | 142,987,264 | **20,373,504** | **−85.7%** |
| **local_st（溢出写）** | 85,366,784 | **25,927,680** | **−69.6%** |
| shared_ld（LDS） | 371,068,928 | 430,182,400 | **+15.9%**（Q 改从 smem 读） |
| global_ld | 33,591,296 | 33,591,296 | 0% |
| registers/thread | 255（顶格） | **251** | −4（但不再溢出） |
| smem/block | 67,584 | 67,584 | 0（combine 项主导，Q 搬走不影响总量） |
| occupancy | 1 block / 4 warps | 1 block / 4 warps | 0 |
| issue_active | 15.20% | **21.14%** | +39%（相对） |
| long_scoreboard | 3.21 | **1.21** | **−62%** |

结论：E1 的收益**不是**来自"少发指令"（只 −8.8%），而是来自 **溢出流量消失后 long_scoreboard 从 3.21 掉到 1.21**——溢出到 local 的访存本来会叠加在 KV 全局加载的同一延迟链上。

### 3.2 E1 vs E3-(d) 逐项对比（同启动点、同协议，08:34 vs 10:27；`/tmp/ncu-e1-pipes.csv` / `/tmp/ncu-e3d-pipes.csv`）

配置差异（都在 Volta 的 256/256/ncols64 行）：E1 = `nbatch_fa 32, K2 128, V2 128, combine 128, Q_in_reg false`；
E3-(d) = `nbatch_fa 32, K2 64, V2 64, combine 64, Q_in_reg false`（把 smem 从 67,584 压到 43,776 ⇒ 2 CTA/SM）。

| 指标 | E1 | **E3-(d)** | 变化 |
|---|---|---|---|
| **内核 duration** | 34.53 ms | **23.97 ms** | **−30.6%** |
| smem/block（动态） | 67,584 | **43,776** | −35.2%（= Q 33,792 + KV 8,704 + mask 1,280） |
| waves per SM | 14.22 | **7.11** | **1 → 2 CTA/SM 确认** |
| warps_active | 6.25% | **12.33%** | ×2 |
| registers/thread | 251 | 254 | +3（**2×128×254 = 65,024 / 65,536，只剩 512 个寄存器余量，再 +2 regs 就会掉回 1 CTA/SM**） |
| 总指令 `smsp__inst_executed.sum` | 2,633,771,008 | 2,765,373,440 | +5.0% |
| tensor（HMMA） | 1,421,869,056 | 1,421,869,056 | 0%（算法未变，同 §3.1 口径） |
| fp16 | 267,911,168 | 267,911,168 | 0% |
| fma | 205,342,720 | 231,798,784 | +12.9% |
| **alu** | 99,823,616 | **193,157,120** | **+93.5%**（瓦片减半 ⇒ 外层循环/索引/同步翻倍） |
| xu | 65,519,616 | 65,585,152 | ≈0 |
| LSU | 563,019,776 | 566,943,744 | +0.7% |
| local_ld / local_st | 20,373,504 / 25,927,680 | 20,373,504 / 22,241,280 | 0% / −14%（无新溢出） |
| shared_ld（LDS） | 430,182,400 | 430,313,472 | ≈0 |
| global_ld | 33,591,296 | 33,591,296 | 0% |
| **dram read** | 3.48 GB | **1.58 GB** | **−55%**（推断：2 CTA/SM 下同 SM 上两块扫 KV 的时序重叠，L2 命中变好） |
| LDS 波前（load） | 1.20e9 | 1.223e9 | +2%（bank 冲突比例 2.79→2.84，微增） |
| **issue_active** | 21.14% | **33.62%** | **+59%** |
| long_scoreboard | 1.21 | **0.90** | −26%（KV 全局加载延迟被另一 CTA 掩盖） |
| wait | 1.28 | 1.23 | ≈0 |
| **mio_throttle** | 0.05 | **0.64** | **×12.8**（MIO/共享内存队列饱和：小瓦片 2 倍 LDS/STS） |
| **math_pipe_throttle** | 0.11 | **0.49** | ×4.5（tensor 管线开始排队 = 更接近算力） |
| barrier | 0.18 | 0.47 | ×2.6（小瓦片 ⇒ 同步次数翻倍） |
| short_scoreboard | 0.47 | 0.62 | +32% |
| not_selected | 0.00 | 0.26 | 新增（8 warps 争发射槽） |
| no_instruction | 0.21 | 0.12 | −43% |
| lg_throttle | 0.18 | 0.05 | −72% |

结论：占用率翻倍把 issue 拉高 59%、把 KV 延迟停顿压到 0.90，换来 **内核 −30.6%**；代价是 2 倍的小瓦片带来的
`mio_throttle`（共享内存队列）与 `barrier` 停顿——即"用 MIO 压力换延迟隐藏"，同时 `math_pipe_throttle` 上升到 0.49
说明 tensor 管开始排队、离算力上限更近了。**寄存器只剩 512 的余量（254/255）是这个配置的硬约束**。

## 4. 停顿分解（同一启动点，`/tmp/ncu-stock-stalls.csv`）

`sm__warps_active` = 6.25%（= 4 warps/SM，1 block），`smsp__issue_active` = 15.20%（内核空转 85%），
每条已发指令背后的停顿（per issue active）：

| 停顿源 | 值 | 占比 | 含义 |
|---|---|---|---|
| long_scoreboard | 3.21 | ~55% | 等 L1TEX 全局数据（KV 的 LDG） |
| wait | 1.42 | ~24% | 等固定延迟依赖（HMMA/FFMA 链） |
| short_scoreboard | 0.36 | 6% | 等 smem |
| no_instruction | 0.28 | 5% | 取指/分支 |
| barrier | 0.21 | 4% | `__syncthreads` |
| 其余（dispatch/mio/lg/math_pipe/not_selected） | <0.1 各 | ~6% | |

## 5. 负载成本模型（把墙钟拆成「权重项 + FA 项」）

写法：每批 4096 token 的总时间 `T(kv) = W + c × kv`，`W` 与深度无关（权重/SSM/MLP 计算 + KV 写入），`c` 是 FA 的边际系数。
用 d16384 / d32768 两点解出（d0 点用于自检）：

| 配置 | 解出的 `c`（每 1k KV） | 解出的 `W` | d0 自检（实测 vs 反推） |
|---|---|---|---|
| stock（08:47 A/B 均值） | 30.3 µs / 1k KV（= 0.4850 s @16k） | ≈ 1.231 s | 1.2446 s vs 1.2310 s（+1.1%） |
| **E1（08:47 A/B 均值）** | **20.6 µs / 1k KV**（= 0.3298 s @16k） | ≈ 1.225 s | 1.2270 s vs 1.2251 s（+0.2%） |
| **E3-(d)（10:30 A/B 均值）** | **15.6 µs / 1k KV**（= 0.2495 s @16k） | ≈ 1.228 s | 1.2206 s vs 1.2278 s（−0.6%） |

（本表的 `c` 由此前的 A/B 均值行直接换算，修正了早先两处 µs/1k 换算笔误：29.6→30.3、20.1→20.6；"@16k 的秒数"与旧值一致。）

⇒ **FA 边际项**：30.3 → 20.6 → **15.6 µs/1k KV**（E1 −32.0%，E3-(d) 再 −24.4%，**累计 −48.6%**），权重项三家一致（1.231 / 1.225 / 1.228 s，噪声内）。与 ncu 内核 duration 52.47 → 34.53 → 23.97 ms 逐级互证。

外推自检（用 stock 的独立历史点，未参与拟合）：

| 深度 | 实测 t/s（stock） | 模型预测 | 偏差 |
|---|---|---|---|
| d65536 | 1296.33 → 3.160 s | 1.231 + 4×0.4850 = 3.171 s | +0.3% |
| d262144 | 456.95 → 8.964 s | 1.231 + 16×0.4850 = 8.991 s | +0.3% |

**对用户目标（256k 上下文填充）的外推**（模型只在 0-32k 上拟合；上表用 stock 的历史 64k/256k 实测验证过线性到 256k 仍成立）：

| 配置 | 预期 256k 每批时间 | 预期 t/s | 相对 stock |
|---|---|---|---|
| stock | 1.231 + 256×30.3 µs = 8.99 s | ~455（历史实测 456.95） | 1.00× |
| E1（已实现，A/B 复核） | 1.225 + 256×20.6 µs = 6.50 s | ~630 | 1.38× |
| **E3-(d)（本次交付配置）** | 1.228 + 256×15.6 µs = **5.22 s** | **~785** | **1.72×** |
| **E3-(d) 清场实测校验** | **5.320 s** | **769.87** | **1.685×** |
| FA 再快 1 倍（E3-(d) 基础上） | 3.22 s | ~1270 | 2.79× |
| FA 归零（理论上限） | 1.23 s | ~3340 | 7.3× |

**实测 vs 外推**：E3-(d) 的 256k 已于 2026-09-12 11:16 在清场环境实测（`/tmp/cln-256k-e3d.log`，`-r 1`）= **769.87 t/s**，对比模型预测 785 t/s 偏差 **−1.9%**（模型只在 0-32k 拟合、8× 外推）；对 stock 历史 256k（456.95 t/s）= **1.685×**。
FA 占比自检：256k 时 FA 时间 (5.320−1.228)/5.320 = **77%**，与"长上下文 FA 占 80%+"一致。

**算力口径（2026-09-12 本机微基准实测；同晚复核修正，勘误与证据见 §17.5——原"93-102 TFLOP/s 峰值"与"2048 FLOP/条"不再引用）**：
- 本卡 fp16 / fp32 累加**张量实战峰值**：cublas 8192³ fp16/fp32acc GEMM 实测 **85.5 TFLOP/s**（复跑确认；原测 85.4）。证据 `fa-logs/cublas-peak.log`、复跑 `fa-logs/mma-peak-recheck-20260912.log`，源码归档 `/root/llama.cpp/v100-prefill/{mma-peak.cu,cublas-peak.cu}`。
- 同形 HMMA 微基准（**非峰值、仅作同形相对锚点**）：纯 `mma.sync.m8n8k4` 指令流 **45.05 Gmma/s**（= 0.409 mma/SM/cycle；复跑复现原值 45.38 的 −0.7%；12 配置扫描区间 45.0–46.7）。按几何量 512 FLOP/条换算仅 ≈23 TFLOP/s，低于 cuBLAS ⇒ 不能当本卡张量上限（§17.5）。
- 单条 `mma.sync.aligned.m8n8k4`（与仓库 `mma.cuh` Volta 分支同形）编译为 **4 条 SASS `HMMA.884.F32.F32.STEP0..3`**：ncu 交叉实测 73,728,000 ÷ 18,432,000 = 4.0、147,456,000 ÷ 36,864,000 = 4.0；静态 SASS 768×4 条。一条 `mma.sync` 的几何量 = 8×8×4 = 256 MAC = 512 FLOP。
- 被 profile 的那次 FA 启动（stock/E1/E3-(d) 三侧同点，指令数同为 1.4219e9 条 HMMA）：stock 52.47 ms = 27.1 G/s、E1 34.53 ms = **41.2 G/s**、E3-(d) 23.97 ms = **59.3 G/s** ⇒ 同形微基准忙度 **15.0% / 22.9% / 32.9%**（锚值 180.2 G/s = 45.05×4）。绝对 FLOP 口径撤回——原"Q=4096、KV≈20480"的几何与动态指令数互斥（差 6.8×），无法复核，见 §17.5。
⇒ 结论不变：**内核受延迟而非算力限制**，喂饱 tensor 管理论上仍有 3.0–4.4× 空间（同形微基准锚；E1 22.9% / E3-(d) 32.9% 忙）；长上下文预填充里 FA 占 80%+ 时间，杠杆很大。

## 6. 结论（2026-09-12 更新）

1. **不是 DRAM 带宽受限**：当前启动点只用到 ~45 GB/s（5%）；把全部 KV 读钉到同一行（−97% 流量）也只快 17.6%（53.3 → 43.4 ms）。内存≈18%。
2. **是延迟/依赖受限**：issue 15.2%（每 SM 每 cycle 仅 0.6 条），85% 的周期无指令可发；主因是 KV 全局加载（long_sb ~55%）与 HMMA/FFMA 依赖链（wait ~24%）。
3. **寄存器顶格 + 溢出**（**E1 已解决**）：stock 为 255 regs/thread（上限）+ 228M 条 LDL/STL（占全部指令 7.9%）；E1 用 `Q_in_reg=false` 后降到 251 regs、溢出减 80%，**内核 duration −34%**。
4. **tensor 占 49.2% 指令，但 tensor 忙度仅 15%（stock）→ 22.9%（E1）**（E1 后 54% 指令占比；同形微基准口径，见 §5 末）：把 tensor 管喂饱的理论空间 ≈ 4.4×（E1）/ 3.0×（E3-(d)）。
5. **Volta 先天缺 cp.async 与 ldmatrix**：`nstages` 被强制 0（无软流水）、`swizzle` 关闭（tile stride = nbatch+4）、ldmatrix 用 2×LDS.64 仿真。
6. **ncols 上限 64**（现有 20 个模板实例化；新增实例要加 .cu 文件，按项目规范不做）⇒ 无法靠更大 ncols 削减 KV 重读。
7. **TILE（FP32 SIMT）路径已排除**：`fattn-tile.cuh` 纯 FFMA，理论上限 ~14 TFLOP/s（72 SM × 64 FP32 × 2 × 1.53 GHz），且完全不使用 tensor 管——tensor 路径的本卡实战峰值已达 cuBLAS 85.5 TFLOP/s（§5 末；原"MMA 路径 ~40 TFLOP/s"的比较属已勘误口径，见 §17.5）。
8. **收益随深度放大**：FA 边际系数 30.3 → 20.6（E1）→ **15.6 µs/1k KV**（E3-(d)，**累计 −48.6%**）；同会话链式对比 d0 只 +1.6%、d16384 +16.1%、d32768 **+27.4%**（清场复核 +16.1% / +26.3%，一致）；**256k 清场实测 5.320 s / 769.87 t/s**（对 stock 历史 456.95 ⇒ **1.69×**，模型预测偏差 −1.9%）（§5）。
9. **溢出（spill）是"隐性延迟源"**：溢出本身只占 7.9% 指令，但它把 local 访存塞进 KV 加载的同一延迟链，long_sb 3.21 → 1.21 的下降才是 E1 那 34% 加速的主因。
10. **2 CTA/SM 是第二阶段的最大杠杆（E3-(d)）**：瓦片减半使 smem 67,584 → 43,776，同 SM 共存 2 个 CTA；warps_active ×2（6.25→12.33%）、issue 21.14→33.62%、内核 34.53 → 23.97 ms（−30.6%）；对干净 stock 累计 **−54.3%**（52.47 → 23.97 ms）。
11. **寄存器只剩 512 个余量（254/255）是 E3-(d) 的硬边界**：2×128×254 = 65,024 / 65,536；再涨 2 个寄存器/线程就会跌回 1 CTA/SM。`__launch_bounds__(nthreads, 2)` 早已声明占用率 2，编译器一直在压寄存器——任何可能增加寄存器压力的改动（如 E2 预取）必须先验证这一条。

## 7. 实验队列

| 编号 | 内容 | 依据 | 状态 |
|---|---|---|---|
| E1 | Volta 的 (256,256,64) 行改 `Q_in_reg=false`（Q 从寄存器搬到 smem） | 255 regs 顶格 + 228M 条溢出指令 | **✅ 已完成**：内核 −34.2%、同会话 A/B d32768 +16.8%（d0 对照 +1.4%）、256k 外推 8.99 → 6.50 s（+38%） |
| E2 | 手工双缓冲 KV 预取（无 cp.async，把下一 tile 先取进寄存器） | long_sb 仍占首次采样的大头；E1 后 regs 251 仍紧 | 候选（需先评估寄存器余量） |
| E3 | 干净对比 1 CTA vs 2 CTA/SM（小瓦片压 smem） | §2 的重要修正 | **✅ 已完成（= (d) 变体胜出）**：`GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 64, 128, 2, 32, 64, 64, 64, 2, false)` ⇒ smem 43,776、2 CTA/SM。内核 −30.6% vs E1（对 stock 累计 52.47 → 23.97 ms = **−54.3%**）、同会话 A/B d16384 +5.15% / d32768 +9.08%、256k 外推 ~785 t/s。试错：(a)(b) `nbatch_fa=16` 违反宏断言 `nbatch_fa%32==0`（编译期失败，设计阶段弃）；(c) `Q_in_reg=true + combine=64` 被寄存器压力/溢出否掉；(d) 胜出。E2 的寄存器余量问题在 (d) 下更紧（254/255，只剩 512）。 |
| E4 | `models.ini` 9B `ubatch-size=4096` | 减少 ubatch 切分次数 | 已改，待用户重启 router |
| E5 | 继续压非 tensor 指令（E3-(d) 后 alu 翻倍、mio_throttle 0.64、LDS 占比升） | §3.1 / §3.2 | E3-(d) 已交付（issue 21→34%）；下一步依赖 E2 或新想法 |
| E6 | 正确性门禁：`test-backend-ops test -b CUDA1 -o FLASH_ATTN_EXT` | 改的是 Volta 配置行 | **✅ 通过（E1、E3-(d) 各跑一次）**：`-p "hsk=256"` 141 用例 `OK=141 / FAIL=0`（汇总行 `141/141 tests passed`；此前误记为 143 = `grep -c OK` 行数，含判决行，见 §17.4），用例描述序列与 stock **逐例完全一致**（diff 为空）；日志 `/tmp/tbo-e1-256.log`、`/tmp/tbo-e3d-256.log`、`/tmp/tbo-stock-256.log`。另：不带过滤的全量跑会在 (320,256) 用例上 abort，已用 stock 库对照确认是**既有问题**（与本改动无关） |
| E7 | 清场复核（停服务、排除干扰源） | 用户指示"跑实验直接结束旧的把干扰源排除掉" | **✅ 已完成（11:05-11:20）**：三路 A/B/C ×2 轮 ⇒ 比值与结论未变（偏差 ≤1.1pp）、六个绝对数字位移 ≤0.7%；e3d 的 256k 实测 **769.87 t/s**（模型预测偏差 −1.9%）；stock 的 256k 长测按指示取消（**问题点追踪只用 32k**）；日志 `/tmp/cln-verify.log`、`/tmp/cln-256k-e3d.log` |
| E8 | V-B 变体：`nbatch_fa=64`（K2=V2=128），在 V-A 瓦片几何上再减半迭代数 | V-A 结果 | **⏸ 降级（不推荐继续）**：V-A 已证大瓦片 + 单 CTA 方向负面（mio_throttle ×3.5、barrier/long_sb 无掩盖、指令 +15.6%）；V-B 的 smem 只会更大（≈72 KB），仍是 1 CTA/SM，须先解释上述恶化来源再谈 |
| E9 | Volta 版 XOR swizzle（现 swizzle 被 `TURING_MMA_AVAILABLE` 门禁，给 LDS 仿真路径设计等效映射以降 bank conflicts） | `fattn-swizzle.cuh` 门禁 + bank conflict 基数 | **❌ 降级（基数太小）**：e3d 实测 ld 冲突 8.93M / ld wavefronts 280.9M = **3.2%**（st 11.1% 但总量小）⇒ 收益上限仅几个百分点，不再优先（见 §10） |
| E10 | E2 复活：寄存器预取下一 K tile | V-A 若释放寄存器（e3d 254/255 无空间） | 候选（V-A 实测 253 regs，仅多释放 2 个且已回退；仍受寄存器余量限制） |

（2026-09-12 另有一份非内核方向的调研：`/root/llama.cpp/v100-prefill/upstream-tech-survey-20260912.md` —— 上游同步现状、MTP/DFlash/DSpark/EAGLE3 推测解码、sparse-FA 的 V100 缺口等。）

## 8. 最终改动内容（E1 + E3-(d) 合并为一行，位于 `ggml_cuda_fattn_mma_get_config_volta`）

```cpp
// 256/256: keep Q in SMEM, Q_in_reg makes the kernel spill at 255 registers per thread.
// Small K/V and combine tiles so that the whole tile fits twice in 96 KB of SMEM.
GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 64, 128, 2,  32,  64,  64,  64, 2, false);
```

字段依次为：`DKQ=256, DV=256, ncols=64, nthreads=128, occupancy=2, nbatch_fa=32, nbatch_K2=64, nbatch_V2=64, nbatch_combine=64, nstages_target=2, Q_in_reg=false`。

- 相对原先回退到的 Ampere 表共两处差异：`Q_in_reg` 由 true 改 false（E1，消寄存器溢出）；`nbatch_K2/V2/combine` 由 128 减到 64（E3-(d)，动态 smem 67,584 → 43,776 ⇒ 2 CTA/SM）。
- 影响面：所有 `(DKQ=256, DV=256, ncols1×ncols2=64)` 组合在 Volta（cc 7.0）上的 FA 前向。用户服务器 9B 路径（`--flash-attn auto --cache-type-k/v q8_0`、ub=2048）经 `fattn.cu` 调度逐行核对命中此行。
- 实测（清场复核 A/B，停服务排除干扰源后三路 ×2 轮）：相对 E1，d16384 +5.27%、d32768 +8.30%；相对 stock，d16384 +16.06%、d32768 +26.33%；**256k 实测 5.320 s / 769.87 t/s（对 stock 历史 456.95 ⇒ 1.685×）**，与模型外推 ~785 t/s 偏差 −1.9%。
- 正确性：见 §7 E6（141/141，与 stock 逐例一致）。

## 9. 27B 模型扩展验证（Qwen3.8-27B-Q4_K_M，2026-09-12，8k/16k 协议）

背景：27B（qwen35，65 block / 17 个全注意力层，embedding 5120，n_head 24 / n_head_kv 4 = GQA 6，head_dim 256）的 FA 实例化是 `<256,256,32,2>`，与 9B 的 `<256,256,16,4>` **共享 §8 那同一行配置**（同 DKQ/DV=256、ncols=64）；需要确认该行在 27B 上同样生效。
协议（用户指示：问题点追踪统一 8k/16k）：`-dev CUDA1 -fa 1 -ctk q8_0 -ctv q8_0 -b 8192 -ub 4096 -p 8192 -d 8192,16384 -n 0 -r 2`；清场后 A/B 轮换（stock ↔ e3d 各 2 轮）。日志：`/root/llama.cpp/v100-prefill/fa-logs/27b-fa-ab-deep-20260912.log`（lib sha：stock `754fbf6f…` / e3d `e2660404…`，与 §0/§8 一致）。

### 9.1 墙钟 A/B（t/s）

| 深度 | stock r1 / r2 | stock 均值 | e3d r1 / r2 | e3d 均值 | 提升 |
|---|---|---|---|---|---|
| d8192 | 874.36 ± 0.21 / 872.56 ± 1.61 | 873.46 | 988.86 ± 0.30 / 991.67 ± 0.55 | 990.27 | **+13.4%** |
| d16384 | 759.51 ± 2.64 / 766.58 ± 1.90 | 763.05 | 919.60 ± 0.80 / 920.15 ± 1.23 | 919.88 | **+20.6%** |

离散度 ≤1.0%；收益同样随深度放大（同 9B 的方向）。

### 9.2 ub 深点（e3d 侧，对照 9B 的 ub 结论）

| ub | d8192 | d16384 |
|---|---|---|
| 2048 | 940.92 ± 1.18 | 872.09 ± 2.49 |
| **4096** | **990.27（A/B 均值）** | **919.88（A/B 均值）** |
| 8192 | 959.82 ± 1.39 | 900.19 ± 3.09 |

⇒ 27B 上 ub4096 同样是峰值（d8192 比 ub2048 高 5.2%、比 ub8192 高 3.2%），这支撑 `models.ini` 里 27B `ubatch-size=4096` 的取值。

### 9.3 内核级 ncu（27B、同一启动点协议：`--launch-skip 39 -c 1`、d8192）

| 侧 | smem/block | waves | registers | duration |
|---|---|---|---|---|
| stock | 67,584 | 21.33 | 255 | **12.198 ms** |
| **e3d** | 45,056 | 10.67 | 255 | **5.658 ms** |

⇒ 内核 **−53.6%**（对 9B 的 −54.3%，两侧一致互证）；45,056 = Q 33,792 + KV 8,704 + mask 2,560（27B 的 mask 大一倍：ncols1=32 vs 9B 的 16）；waves 减半 = 2 CTA/SM 在 27B 上同样生效。

### 9.4 strace 复核（e3d、d8192、`/root/llama.cpp/v100-prefill/fa-logs/27b-strace-summary-20260912.txt`）

整进程 846,611 次 syscall、总 syscall 时间 8.28 s：poll 5.24 s（63%）+ sched_yield 839,751 次 1.62 s（19.6%）为主，mmap 0.78 s / ioctl 0.35 s；strace 侧墙钟 993.51 t/s 与无 strace 的 e3d r1（988.86）持平。
⇒ 内核边界（驱动/分配/同步）没有隐藏开销，瓶颈确定在 GPU 计算（内核），后续优化继续只着眼内核内部。

## 10. V-A 变体实验（8 warp / 1 CTA，2026-09-12 —— **失败，已回退**）

动机：E3-(d) 已把「2 CTA/SM」杠杆用满，剩余瓶颈是每迭代固定开销 + 延迟。V-A 反向设计：**把 8 个 warp 放进 1 个 CTA**（nthreads 128→256、occupancy 2→1），并把 `nbatch_K2/V2` 64→128——K 沿 DKQ/2=128 只需 1 个 chunk（e3d 需要 2 个），K 侧同步次数减半；代价是动态 smem 45,056 → 69,632（>96 KB/2 ⇒ 回到 1 CTA/SM）。

改动（单行，diff 2+/2-）：`GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 64, 256, 1, 32, 128, 128, 64, 1, false);`；构建 4 分钟；lib sha `8a710a69…`（快照 `fa-libs/va`）。A/B 脚本与日志：`/root/llama.cpp/v100-prefill/fa-logs/27b-fa-ab-va-20260912.log`；两侧正确性门禁通过（141 OK / 0 FAIL、用例集合排序后 diff=0）。

### 10.1 墙钟 A/B（t/s，两轮，e3d ↔ va 轮换）

| 模型 | 深度 | e3d（r1 / r2 → 均值） | va（r1 / r2 → 均值） | Δ |
|---|---|---|---|---|
| 27B | d8192 | 1012.16 / 984.50 → 998.33 | 944.93 / 928.47 → 936.70 | **−6.2%** |
| 27B | d16384 | 931.97 / 911.06 → 921.52 | 846.29 / 836.69 → 841.49 | **−8.7%** |
| 9B | d8192 | 2988.80 / 2982.01 → 2985.41 | 2839.79 / 2829.24 → 2834.52 | **−5.1%** |
| 9B | d16384 | 2722.25 / 2713.05 → 2717.65 | 2533.27 / 2534.20 → 2533.74 | **−6.8%** |

四个组合、每一轮全部为负（9B 侧两轮离散 ≤0.4%）⇒ 结论稳健。

### 10.2 ncu 归因（27B、d8192、同一启动点）

| 指标 | e3d | va | 解读 |
|---|---|---|---|
| duration | 5.589 ms | **8.517 ms（+52.4%）** | 内核直接慢一半 |
| smem/block | 45,056 | 69,632 | 构造预期命中；但装不下 2 个 CTA |
| waves | 10.67 | 21.33 | 掉回 1 CTA/SM |
| regs / warps_active | 255 / 12.25% | 253 / 12.50% | 每 SM 仍 8 warp（2 CTA×4 vs 1 CTA×8） |
| issue_active | 33.2% | 24.5% | 可发射窗口更少 |
| barrier | 0.49 | 0.69 | **假设证实**：单 CTA 无共驻 warp 组掩盖同步 |
| long_scoreboard | 0.97 | 1.60 | 访存延迟同样无处隐藏 |
| mio_throttle | 0.50 | **1.75（×3.5）** | 8 warp 单 CTA 突发压 LSU/MIO 队列 |
| math_pipe_throttle | 0.47 | 0.83 | tensor 突发也更强 |
| short_scoreboard | 0.73 | 0.20 | 反而下降（LDS 等待少） |
| inst 总量 | 649.6M | 750.8M（+15.6%） | fp16 管 +88.6%、fma +28.5%、alu +21.9%；tensor 320.9M 不变 |
| dram 读 | 391.9 MB | 435.9 MB（+11.2%） | 非瓶颈（≈51 GB/s） |
| bank conflicts ld/st | 8.93M / 3.69M | 3.43M / 1.72M | **反而更低** |
| shared wavefronts ld | 280.9M | 330.0M（+17.5%） | LDS 流量上升（shared_ld 指令 +14.9%） |

失败机制：**单 CTA/SM 失去延迟掩盖**（barrier + long_scoreboard 同涨）+ **大瓦片导致 MIO 突发与指令膨胀**（mio_throttle ×3.5、总指令 +15.6%）；K 同步次数减半的收益远不能补偿。⇒ 已按预案 `git restore` 回退该单行改动，并重编译使 build/bin 与源码一致（sha 校验见 §10.3）。

### 10.3 回退与校验

- 回退：`git restore --source=HEAD --worktree -- ggml/src/ggml-cuda/fattn-mma-f16.cuh`，工作区仅此一处、回退后 `git status --short` 为空。
- 重编译：`bash /root/llama.sh make`（`-j8` + ccache），日志 `fa-logs/build-revert-e3d-20260912.log`；**已校验（2026-09-12）**：`build/bin/libggml-cuda.so.0.23.0` sha256 = `e2660404…`，与 `fa-libs/e3d` 快照**逐字节一致**；冒烟复测 9B d8192 = 3051.52 t/s（≈/≥ e3d 基线 2988.80），无异常。

**衍生裁决（E9）**：bank conflicts 基数很小——ld 冲突仅占 ld wavefronts 的 **3.2%**（8.93M/280.9M）、st 侧 11.1% 但总量小；即使 Volta swizzle 全部消掉，收益上限也只有几个百分点 ⇒ **E9（Volta XOR swizzle）降级，不再优先**。

## 11. 微优化实验：Volta rescale 跳过（2026-09-12 —— **负收益，已回退**）

动机：SASS 分解显示主循环每 warp 每迭代 1,497 条指令中有 **128 条**是 `VKQ_C` 的重缩放 `HMUL2`（8.5%），而 `KQ_max_scale[col] = expf(KQ_max[col]-KQ_max_new[col])` 在 running max 不变时恒为 **1.0f**（`SOFTMAX_FTZ_THRESHOLD=-20.0f` ⇒ diff=0 不会被清零，见 §3），256 次迭代里最大值真正上升只有少数几次 ⇒ 绝大多数迭代是恒等乘法。改动即：加 warp 一致守卫（`__any_sync`）跳过。

改动（未提交，`fattn-mma-f16.cuh` 单文件 13+/5-，`#else // Volta` 分支内）：把原「无条件 128×HMUL2 重缩放循环」包进 `bool needs_rescale` + `__any_sync(0xFFFFFFFF, needs_rescale)` 守卫，`needs_rescale = OR_col(KQ_max_scale[col] != 1.0f)`。

构建日志 `fa-logs/build-rescale-20260912.log`，lib sha `8e573ba4…`（快照 `fa-libs/rescale`，实验后已随回退删除）。正确性门禁 141 OK / 0 FAIL（`fa-logs/tbo-rescale-hsk256-9b.log`）。

### 11.1 墙钟 A/B（9B 单模型、单变量；`fa-logs/9b-fa-ab-rescale-20260912.log`）

按用户指示：**9B 与 27B 同架构 ⇒ 只测 9B**；协议沿用 8k/16k 深度、两轮 e3d↔rescale 轮换。

| 深度 | e3d（r1 / r2 → 均值） | rescale（r1 / r2 → 均值） | Δ |
|---|---|---|---|
| d8192 | 3061.62 / 3038.06 → 3049.84 | 3045.47 / 3018.71 → 3032.09 | **−0.58%** |
| d16384 | 2795.97 / 2772.57 → 2784.27 | 2781.04 / 2764.46 → 2772.75 | **−0.41%** |

4 组配对（每轮内比较）全部偏向 e3d（+0.53% / +0.64% / +0.53% / +0.29%）。

**方法学警告（本轮新发现）**：跑序是 ABAB（e3d,r | e3d,r），而机时存在**单调下漂 ~0.8%/轮**（3061.62 → 3045.47 → 3038.06 → 3018.71）。线性漂移下 ABAB 的一阶项不抵消，会**系统性高估 A（e3d）**约半个轮步（≈0.4%）；按线性漂移修正后净效应约 **−0.2%（噪声边缘）**。以后 ≤1% 量级的效应必须用 **ABBA** 跑序，或只按轮内配对统计并在报告里明示漂移。

### 11.2 内核级 ncu（9B d8192、同一启动点 `--launch-skip 39 -c 1`；`fa-logs/9b-ncu-rescale-20260912.log`）

| 指标 | e3d | rescale | Δ | 解读 |
|---|---|---|---|---|
| duration | 17.099 ms | 17.388 ms | **+1.69%** | 内核反而更慢 |
| 总指令 | 1,986.8M | 1,897.6M | −4.49% | 只降 4.5%（守卫抵掉一半） |
| fp16 管（rescale） | 192.4M | **64.4M** | **−66.5%** | **守卫确实生效**（≈128M 条 HMUL2 消失） |
| fma 管 | 167.3M | 192.5M | **+15.1%** | 守卫开销（比较/OR/寄存器重排） |
| alu 管 | 140.2M | 154.5M | **+10.2%** | 同上（`__any_sync`=VOTE + 分支） |
| tensor / lsu / xu | 1019.2M / 407.0M / 47.2M | 1019.2M / 406.0M / 46.2M | 0 / −0.2% / −2.1% | 张量工作量完全一致 |
| issue_active | 33.65% | 31.58% | **−2.07pp** | 每 SM 可发射窗口更少 |
| wait / long_sb / short_sb | 1.23 / 0.88 / 0.61 | 1.35 / 1.11 / 0.80 | **+10% / +26% / +31%** | 停顿全面上升 |
| math_pipe / mio_throttle | 0.49 / 0.65 | 0.35 / 0.58 | −29% / −11% | 仅队列类停顿下降 |
| regs / smem | 254 / 43,776 | 254 / 43,776 | = | 占用率未变（仍 2 CTA/SM） |

**机制（结论性）**：这 128 条 HMUL2 与 tensor/LDS 等待**重叠**，是**廉价的"填缝"工作**——内核受**延迟**（依赖链 + 访存）而非指令数限制；删掉填缝后没有新工作可填，加上守卫引入的 warp 汇合点，停顿反而上升（long_sb +26%、short_sb +31%、wait +10%，issue −2.1pp）。内核 +1.69% × FA 占 9B d8192 墙钟（§5 模型口径 ≈7%，按 FA 占比反推需 ≈29% 才能解释实测墙钟）——两个口径不一致本身说明：该量级下**排序伪影与噪声占主导**；但两条独立证据（内核停顿结构 + 4 组配对墙钟）方向一致 ⇒ **判负**。

### 11.3 回退与校验

- 回退：`git restore --source=HEAD --worktree -- ggml/src/ggml-cuda/fattn-mma-f16.cuh`；`git status --short` 为空。
- 重编译：`fa-logs/build-revert-rescale-20260912.log`（`BUILD_EXIT=0`）；**已校验**：`build/bin/libggml-cuda.so.0.23.0` sha256 = `e2660404…`，与 `fa-libs/e3d` **逐字节一致**（构建可复现）。
- 快照 `fa-libs/rescale` 按"失败变体不留存"原则删除（sha `8e573ba4…` + 上方 diff 足以复现该实验）。

**派生结论（写给后续会话）**：
1. **微优化方向封闭**：在 2 CTA/SM 的延迟受限形态下，"减少指令"不产生收益（本次 −4.5% 指令 ⇒ 内核 +1.7%）。任何后续优化必须先证明它**缩短关键路径**或**增加每 warp 独立工作量（ILP）**，否则不做。
2. **VKQ_C=128 寄存器是硬墙**（§6.11）：`T_C_VKQ VKQ_C[DV/(2*T_C_VKQ::J)]` 与 `np`/`ncols` **无关** ⇒ Volta 上任何配置都不可能跌到 ≤170 regs；3 CTA/SM 同样不可能（Q 瓦片 33,792 B > 32,768 B 的 3 CTA 预算）。
3. **A/B 方法学**：≤1% 量级的效应必须 ABBA 跑序，或只按轮内配对统计并明示漂移。

---

## 12. KV 拆分实验：强制 stream-K（2026-09-12 —— **负收益，已回退**）

**动因**：用户批准尝试"KV-split 并行化"（把一个 Q 瓦片的 KV 扫描拆给多个 CTA 再 reduce）——这是最后一个未测的结构级杠杆。

**关键前置发现（省掉 250-400 行新代码）**：该机制在本仓库**已经存在**，即 upstream 的 **stream-K**：
- 内核（`fattn-mma-f16.cuh`）按 `kbc` 连续工作空间把整个（sequence, z_KV, zt_gqa, jt, kb）任务平铺给 `gridDim.x` 个块；跨块落在同一输出瓦片的接缝用 `dstk_fixup`（max/rowsum 元数据 + 部分 O）记录，再由 `flash_attn_stream_k_fixup_uniform`（块数 = 瓦片数整数倍）或 `flash_attn_stream_k_fixup_general`（其他）归约（`fattn-common.cuh`）。
- 开关在 `launch_fattn` 内的 `should_use_stream_k`：NVIDIA 上仅 Ada+ 无条件启用；**Volta/Turing 走 `tiles_efficiency_percent < 75`**，我们场景（2048 瓦片 vs 144 槽 ≈ 95%）因此**被关掉**。
- 所以"试 KV 拆分"= 强制打开该开关（4 行 host 端改动，无新增内核代码）。

**补丁**（`fattn-common.cuh`，+4 行，单变量）：`if (NVIDIA && cc >= VOLTA && cc < TURING && DKQ == 256) return true;`
构建：`fa-logs/build-streamk-20260912.log`（`BUILD_EXIT=0`，本次改动牵动 fattn-vec/mma/tile 共 121 个实例 TU，编译明显慢于只动 `fattn-mma-f16.cuh` 的轮次）；lib sha `2105ffd0…`，快照 `fa-libs/streamk`。

**生效性验证**（`fa-logs/ncu-streamk-gridcheck-9b.log`，9B、`-p 8192 -d 0`、`--launch-skip 3 -c 1`）：

| 指标 | e3d | streamk |
| --- | --- | --- |
| grid 尺寸 | 1024 | **144**（= 2 CTA/SM × 72 SM） |
| waves/MP | 7.11 | 1.00 |
| 内核时长（同一启动点） | 3.86 ms | 6.50 ms（+68%） |

**正确性门禁**：`tbo-streamk-hsk256-9b.log` → **141 OK / 0 FAIL**（KV 拆分路径数值正确，用例集合与既有侧一致）。

### 12.1 墙钟 A/B（9B 单模型、ABBA 次序；`fa-logs/9b-fa-ab-streamk-20260912.log`）

| 深度 | e3d（r1 / r2） | streamk（r1 / r2） | 均值 Δ | 轮内配对 Δ |
| --- | --- | --- | --- | --- |
| d8192 | 3111.89 / 3060.40 | 3038.33 / 3026.07 | **−1.75%** | −2.36% / −1.12% |
| d16384 | 2839.14 / 2785.51 | 2776.36 / 2756.43 | **−1.63%** | −2.21% / −1.04% |

### 12.2 内核级 ncu（9B、d8192、KV≈16384 启动点、`--launch-skip 39 -c 1`；`fa-logs/9b-ncu-streamk-20260912.log`）

| 指标 | e3d | streamk | Δ | 解读 |
| --- | --- | --- | --- | --- |
| gpu__time_duration | 17.133 ms | 19.507 ms | **+13.8%** | 内核净变慢 |
| **dram__bytes** | 1.205 GB | **8.994 GB** | **×7.5** | DRAM 流量爆炸 |
| **lts__t_sector_hit_rate** | 92.23% | **41.92%** | **−50.3pp** | L2 命中腰斩 |
| long_scoreboard | 0.88 | 1.34 | +52% | 访存等待主导了退化 |
| wait / barrier | 1.23 / 0.47 | 1.23 / 0.48 | = | 同步/屏障无变化 |
| mio_throttle / math_pipe | 0.65 / 0.49 | 0.62 / 0.51 | ≈ | 队列类停顿持平 |
| issue_active | 33.52% | 31.86% | −1.66pp | 发射率略降 |
| 指令总数 / tensor 指令 | 1986.8M / 1019.2M | 1984.6M / 1019.3M | ≈0 | **无重复计算** |
| fixup 归约内核 | — | 49.6 µs / 22.9 MB DRAM | — | **归约本身可忽略** |

### 12.3 机制（结论性）

现状内核的速度**依赖 L2 广播**：同一时刻常驻的 ~144 个块都在**同一个 KV head、同一个 kb 相位**上锁步推进（瓦片按 jt 最快排序，144 个块恰好落在同 head 的相邻 jt 上），于是一个 K/V 字节从 DRAM 取一次即被上百个块经 L2 复用（命中 92%）。
KV 拆分后，144 个块平铺到**整个工作空间**：块分属不同 head、且各自在自己的瓦片序列上重启 kb 循环 ⇒ **相位错开、广播消失**（命中 41.9%、DRAM ×7.5），long_scoreboard +52%，波尾收益被完全吃掉还倒亏。归约内核本身（49.6 µs）不是问题所在。

⇒ **KV 拆分在 V100（6 MB L2）上与现行"锁步广播"形态互斥**。这与 upstream 的启发式一致：stream-K 只在"瓦片数填不满机器"时才启用，且 Ada+ 的 L2 大 1-2 个数量级，能容纳被拆散后的流量。
⇒ 另一形态（uniform，`blocks = ntiles × bpt`）虽保留"同 head、双窗口"的部分局部性，但广播宽度从 144 降到 ~72（DRAM 反推约翻倍 ⇒ 内核 +7~8%），而波尾收益本就不确定（stream-K 的零波尾并未换来净胜）⇒ **不推荐再试**。
⇒ **方法学教训**：`-d 0`（KV=4096）启动点测得 +68%，bench 协议（KV≈16k）启动点测得 +13.8%——不同 KV 下 L2 经济性不同，内核级结论必须取**与 bench 同 KV 量级**的启动点。

### 12.4 回退与校验

- 回退：`git restore --source=HEAD --worktree -- ggml/src/ggml-cuda/fattn-common.cuh`；`git status --short` 为空。
- 重编译：`fa-logs/build-revert-streamk-20260912.log`（`BUILD_EXIT=0`，ccache 命中、主要是链接）；**已校验**：`build/bin/libggml-cuda.so.0.23.0` sha256 = `e2660404…`，与 `fa-libs/e3d` **逐字节一致**。
- 快照 `fa-libs/streamk` 按"失败变体不留存"原则删除（sha `2105ffd0…` + 上述 diff 足以复现）；`fa-libs/` 恢复为 `stock` + `e3d` 两份。

---

## 13. 非 FA（W 项）归因：9B 预填充固定开销拆解（2026-09-12 —— 纯 profiling，无代码改动）

**动因**：用户批准转向"非 FA 部分（W 项）"。FA 侧已封顶（§8–§12），W = 每 4096-token 批的固定开销 ≈ 1.23 s（256k 时占单批 23%），此行之前从未拆过。

**协议**：`-fa 1 -ctk q8_0 -ctv q8_0 -b 8192 -ub 4096 -p 8192 -d 0 -n 0 -r 1`（d0 使 W 主导；nsys 全程内核级汇总，脚本 `9b-nsys-wterm.sh`）；另用 ncu 对三个大头内核定性（`9b-ncu-wterm.sh`，`--cache-control none`，d8192 启动点）。
**先说一条结构性事实**：9B 是 **hybrid SSM 模型**（`config.json`：32 层 = 24 层 `linear_attention`（gated delta net）+ 8 层 `full_attention`，`full_attention_interval=4`）——所以"KV 深度无关"的那部分不只有 dense GEMM，还有整个 delta-net 主干。

### 13.1 每 4096-token 批的 W 构成（nsys 轨迹含 warmup+实测共 4 批；下表已除以 4）

| 内核族 | ms/批 | 占 W | 说明 |
| --- | ---: | ---: | --- |
| **cuBLAS f16 GEMM 合计** | **684.7** | **60.4%** | cutlass `Kernel2` 128x128 = 507.1；`volta_s884gemm_fp16_256x128` = 166.4；`volta_s884gemm_fp16_64x64` = 11.2 |
| **`gated_delta_net_cuda`**（24 层） | **173.8** | **15.3%** | 线性注意力核，单次 7.2–8.2 ms |
| 权重 Q4_K/Q6_K → f16 dequant | 54.1 | 4.8% | 每批 264 次（q4_K 216 + q6_K 32 + q8_0 16；q8_0 仅 0.36 ms/批） |
| 激活 f32 → f16 `convert_unary` | 45.0 | 4.0% | 每批 248 次 |
| `flash_attn_ext_f16`（KV≤8k） | 50.2 | 4.4% | 8 层 × 4 批 = 32 次 |
| `unary_gated_op`(silu) | 25.8 | 2.3% | MLP + delta-net gate |
| `rms_norm_f32` ×3 变体 | 30.3 | 2.6% | 612 次（轨迹合计，= 153 次/批） |
| `concat_non_cont` | 21.1 | 1.8% | 24 层线性注意力，每层 1 次 |
| `ssm_conv_long_token_f32` | 14.5 | 1.3% | 短卷积（kernel=4） |
| `k_bin_bcast`(add) | 14.5 | 1.3% | 残差加 |
| 其余（scale/cpy/wmma/fwht/rope/KV 量化写/mask 等） | ~24 | 2.1% | |
| **合计（GPU 内核时间）** | **1135** | 100% | 同协议墙钟 1207 ms/批（bench 3394.61 t/s）⇒ GPU 占空比 **94%** |

- 深度无关性成立：delta-net 在 KV≈12k 启动点 8.16 ms vs d0 均值 7.24 ms（≈持平）；GEMM/dequant/convert 只与 token 数相关。
- 结论：**W 的 60% 是 GEMM、15% 是 delta-net、8.8% 是"量化权重→f16"的转换税、10% 是内存受限小算符**。

### 13.2 内核级定性（ncu）

| 指标 | cutlass `Kernel2` 128x128 | `volta_s884gemm_fp16_256x128` | `gated_delta_net_cuda` |
| --- | --- | --- | --- |
| gpu__time_duration | 3.412 ms | 5.703 ms | 8.156 ms |
| grid / block | (256,8,1) / 128 | (16,32,3) / 256 | (32,1,32) / (32,4,1) |
| 寄存器 / waves | 236 / 14.22 | 250 / 21.33 | 47 / **1.42** |
| `sm__throughput` | **84.6%** | **75.1%** | **48.2%** |
| tensor 指令占比 | 536.9M / 680.1M = 79% | 805.3M / 939.1M = 86% | — |
| `issue_active` | 54.8% | 45.3% | 56.3% |
| DRAM（占峰值） | 590 MB（15.3%） | 960 MB（14.9%） | 605 MB（**6.5%**） |
| 主导停顿 | — | — | long_sb 4.51 / short_sb 2.92 / wait 2.70（per issue-active） |

**判读**：
1. **GEMM（W 的 60%）已近 cuBLAS 实用上限**：两块最大 GEMM 的 SM 吞吐 75–85%、张量指令占比 79–86%；聚合 82 TFLOPS ≈ V100 名义 125 的 66%（混合了 64x64 小瓦片与逐个矩阵的尾段）。路径选择有两层原因：**profile 当时的 build 显式 `-DGGML_CUDA_FORCE_CUBLAS=ON`**（当时 `/root/llama.sh` 的配置；**已于 2026-09-12 改为 OFF，见 §16.5——本表 ub=4096 ⇒ 每批 ne11≥64 ⇒ 两种旗标下都走 cuBLAS，本节判定与结论不受影响**），`ggml_cuda_should_use_mmq` 在 `#ifdef GGML_CUDA_FORCE_CUBLAS` 下直接 return false；即便不设该宏，代码级门槛也是 `ne11 < MMQ_DP4A_MAX_BATCH_SIZE(=64)` 才用 dp4a MMQ（`mmq.cu:334`），ub=4096 同样落到 cuBLAS。
2. **delta-net（W 的 15%）是延迟受限**：只有 1.42 波、SM 48%、DRAM 6.5%、`long_scoreboard` 主导 ⇒ 每 token 的串行链**无访存预取**，等 load 的时间占一半；605 MB DRAM/次 ≈ 理论最小值的 3 倍（`v_t[col]` 逐 warp 标量散读 + 32 个 z 块重复读 q/k）。这是**唯一有明确改造出口的大项**（软件流水预取 + v 向量化/共享，2–3× 量级）。
3. **转换税 8.8%**：每批 512 次小 kernel（264 权重 dequant + 248 激活 convert）；dequant 有效带宽仅 ~244 GB/s（8.12 GiB f16 写 + 4.16 GiB 读）。显存现实：256k 时 KV≈20 GiB + 权重 5.4 GiB，**常驻 f16 权重缓存（+8.1 GiB）放不下**，只能摊薄（如 ub=8192）或提速内核。
4. **小算符 ~10%**：多次 1–2 ms 级内存受限内核（norm/silu/concat/ssm_conv/add），约 50% 带宽效率。

### 13.3 杠杆清单与预期（诚实上限）

| 方向 | 可动对象 | 预期 W 降幅 | 换算 256k 墙钟 | 换算 8k/16k 协议 |
| --- | --- | ---: | ---: | ---: |
| A. delta-net 流水化（预取 + v 向量化） | `gated_delta_net.cu` | 6–10% | +1.5–2.3% | +6–10% |
| B. 转换税摊薄/提速（ub↑、更快 dequant） | 参数 / `convert.cu` | 2–4% | +0.5–1% | +2–4% |
| C. 小算符合并/提速 | norm/silu/concat 等 | 3–5% | +0.7–1.2% | +3–5% |
| D. GEMM 本体 | cuBLAS 支配 | ≈0（无手段） | — | — |

⇒ **W 项总量级约 10–15%（对 256k 墙钟 ≈ +2.5–3.5%）**；注意用户测试协议（8k/16k）里 W 占 80–90%，此方向在该区间接近 1:1 生效。

### 13.4 产物与工具小记

- `fa-logs/nsys-wterm-9b-d0.nsys-rep` / `-stats.txt` / `-kern.csv`、`fa-logs/9b-nsys-wterm-20260912.log`、脚本 `9b-nsys-wterm.sh`。
- `fa-logs/9b-ncu-wterm-20260912.log`（delta-net + volta_256x128）、`fa-logs/ncu-wterm-kernel2-9b.log`、脚本 `9b-ncu-wterm.sh`。
- 工具小记：ncu `-k` 的 `regex:` 匹配**简名**——`regex:cutlass::Kernel2` 匹配不到（报 "No kernels were profiled" + 内核列表），改用 `regex:Kernel2` 命中；`--cache-control none` 避免 replay 前的缓存刷新失真。
- 本行无代码改动（`git status --short` 为空，HEAD `19a667d32`）。

## 14. delta-net 软件流水（W 项杠杆 A，单一变量）2026-09-12

改动：`ggml/src/ggml-cuda/gated_delta_net.cu` 单文件——手动双缓冲（寄存器 `k_pre/q_pre/g_pre/v_pre/beta_pre`）在 token t 计算前发出 t+1 的载入，用访存延迟覆盖归约链。**数学与索引映射完全不变**（仅调度次序）。这是 W 项实验里第一处代码改动。

### 14.1 内核级 A/B（ncu，d8192、launch-skip 24、cache-control none，与 §13 同一位点）

| 指标 | e3d（基线） | deltanet-pipe | 变化 |
|---|---|---|---|
| `gpu__time_duration.sum` | 8.025 ms | 7.130 ms | **−11.2%** |
| `sm__throughput` | 47.68% | 65.23% | +17.6 pt |
| `sm__warps_active` | 55.56% | 47.80% | −7.8 pt |
| long_scoreboard / issue-active | 4.24 | 1.48 | −65% |
| short_scoreboard | 2.91 | 2.27 | −22% |
| wait | 2.70 | 1.83 | −32% |
| not_selected | 1.47 | 2.42 | +65% |
| mio_throttle | 0.31 | 0.22 | −29% |
| dram__bytes.sum | 470,166,208 B | 313,604,192 B | −33% |
| dram throughput | 5.09% | 3.87% | −24% |

判定：
1. **流水化按设计生效**：long_scoreboard 崩 65%、SM 吞吐 48%→65%、`not_selected` 上升（warp 有活干、调度器挑不过来，健康信号）。DRAM 字节降 1/3 是延迟对齐后 q/k 的 32 份重复读更多落在 L2 的副产物（对带宽不构成杠杆，V100 带宽本就不是瓶颈）。
2. **只快 11.2%，不是预期的 2–3×**：被隐藏掉的是访存延迟；剩下的瓶颈是每 token 两次 `warp_reduce_sum`（5 轮 `__shfl_xor_sync`，sm_70 无 64-bit 合并，共 ~10 条 shfl 的纯延迟链）＋ token 内串行依赖（kv → delta → s 更新 → attn 归约），流水化动不了这一段。
3. 内核 2612 cycles/token（新）vs 2939（旧，1.5 GHz 折算）；归约链只占其中 ~5%，说明即使消掉两个归约，端到端也只再省个位数百分比。

### 14.2 墙钟 ABBA（9B，`-fa 1 -ctk q8_0 -ctv q8_0 -b 8192 -ub 4096 -p 8192 -d 0,8192,16384 -n 0 -r 2`，8 次运行）

| 次序 | 库 | d0 | d8192 | d16384 |
|---|---|---|---|---|
| 1 | e3d | 3415.68 | 3091.16 | 2831.00 |
| 2 | pipe | 3424.27 | 3098.51 | 2831.33 |
| 3 | pipe | 3398.92 | 3069.79 | 2807.97 |
| 4 | e3d | 3347.58 | 3035.30 | 2780.75 |
| 5 | e3d | 3378.54 | 3046.44 | 2799.40 |
| 6 | pipe | 3385.60 | 3062.40 | 2802.13 |
| 7 | pipe | 3359.64 | 3031.31 | 2787.04 |
| 8 | e3d | 3318.47 | 3008.20 | 2757.42 |
| **均值增益** | | **+0.80%** | **+0.66%** | **+0.54%** |
| 逐对增益 | | +0.25/+1.53/+0.21/+1.24 | +0.24/+1.14/+0.52/+0.77 | +0.01/+0.98/+0.10/+1.07 |

- 折算绝对值：每 4096-token 批省 **10.4 / 8.9 / 7.9 ms**（d0 / 8k / 16k）。
- 全序列单库漂移 ≈ −0.6%/次运行（pipe 侧逐轮均值 −0.63%、e3d 侧 −0.95%；首尾 e3d 3415.68→3318.47），**ABBA 是必需品**，单轮比较会给出 +0.2%~+1.5% 的散乱结论。
- 单轮增益的峰谷差（d0 上 +0.21% vs +1.53%）说明效应量与噪声同阶，均值 ±0.3% 是合理不确定性。
- **未解释的差额（留档）**：ncu 内核省 0.895 ms × 24 层 = 21.5 ms/批，墙钟只收到 ~10 ms/批（约一半）。可能原因：ncu replay 的缓存状态比真实运行更利于预取版本，或释放出的 GPU 时间被同批其它算子重叠吸收。按墙钟数字为准。

### 14.3 预期换算（用户协议）

- d0 ≈ W 主导：+0.80%（实测，与 W 的 14.4% 份额 × 内核 −11.2% ≈ +1.6% 的估算相比打了对折，见 14.2 第 4 条）。
- 256k：delta-net 每批工作量不随 KV 变，故绝对节省不变，份额被 FA 稀释 → 预期 +0.2% 量级（估，未实测；256k 位点单次测量成本高）。

### 14.4 正确性门禁

`test-backend-ops test -b CUDA1 -o GATED_DELTA_NET` **36/36 OK**（覆盖 KDA=0/1、K=1..4 快照、head_size 16/32/64/128、n_seqs、permuted=1）；库 sha256 `405cfc0d131efcd03129ed68842099e25a34d4cd01998dac38ff3c873afcec05`（快照 `fa-libs/deltanet-pipe`，基线 `fa-libs/e3d` = `e2660404…`）。

### 14.5 产物

- 源码：`ggml/src/ggml-cuda/gated_delta_net.cu`（本行唯一脏文件）。
- 日志：`fa-logs/9b-deltanet-ncu-ab-20260912.log`、`fa-logs/9b-deltanet-ab-20260912.log`（第一块 ABBA）、`fa-logs/9b-deltanet-ab2-20260912.log`（第二块 ABBA）、`fa-logs/build-wterm-deltanet-20260912.log`。
- 脚本：`9b-deltanet-ncu-ab.sh`、`9b-deltanet-ab.sh`。
- 库快照：`fa-libs/deltanet-pipe/`（保留，因 A/B 结果为正且内核效率严格变好）。

## 15. delta-net 分块（chunked）算法：设计与数值预验证 2026-09-12（用户选定方向 B，尚未写 CUDA）

动机：§14 证明流水化只能榨到 −11%（串行 scan 的 token 内依赖无法用流水化消除）。分块把 O(n) 串行 scan 变成 chunk 内密集矩阵乘 → 并行度与算术强度上升一个量级，是 W 内唯一 10% 量级的项。

### 15.1 关键推导（非 KDA、标量衰减 γ_t=e^{g_t}、Γ_t=∏_{r≤t}γ_r、A[i][j] 即 S）

每个 chunk（C 个 token，进入状态 S_in）内：
- `A[t][s] = β_t·exp(GS_t − GS_s)·(k_s·k_t)`（严格下三角），`T = (I + A)^{-1}`，`GS` = g 的 log 空间前缀和
- `P = T·(β⊙v)`，`W = T·diag(β·Γ)`，`u = P − W·(K·S_in)`
- 输出 `o_t = scale·[ Γ_t·(S_inᵀq_t) + Σ_{s≤t} exp(GS_t−GS_s)(k_s·q_t)·u_s ]`
- 出块状态 `S_out = Γ_{C-1}·S_in + Σ_s exp(GS_{C-1}−GS_s)·k_s u_sᵀ`
- 全部可改写为稠密乘：`Q_eff = diag(Γ)·Q − (tril(QKᵀ)⊙R)·W`，则 `o = scale·(Q_eff·S_in + (tril(QKᵀ)⊙R)·P)`；状态扫描每 chunk 只需 `K·S_in`、`Q_eff·S_in`、`W·(K S_in)`、`Kᵀdiag(w)·(...)` 四个乘。

**硬约束（原型踩出来的）**：不许做 `ū = u/Γ` 归一化——g 落在门禁的 [-20,-1e-4] 区间时，C=64 的块内累计衰减 exp(-640) 已在 f64 边缘、C=128 直接下溢为 0 → 除零 NaN。所有衰减权重必须在 log 空间用 `exp(GS_t − GS_s)`（差 ≤ 0，恒在 (0,1]）算，不含任何除法。

### 15.2 数值预验证（`dnet-chunk-prototype.py`，numpy，f64）

判据同门禁：NMSE = Σ(a−b)²/Σa²，阈值 1e-7；输入分布同门禁（q/k l2 归一、g~U(-20,-1e-4)、β~U(0,1)、v~U(-0.3,5)、state~U(-1,1)）。

| 场景 | NMSE(attn) | NMSE(state) |
|---|---|---|
| d=128, n=256, C=32/64/128 | 1.2e-30 / 7.3e-30 / 9.3e-29 | 1.6e-32 |
| d=128, n=4096, C=64/128 | 1.0e-29 / 3.8e-29 | 1.9e-29 |
| d=128, n=2/33/65/100/127（含尾块） | ~1e-29 | 1e-32 |
| d=64, n=256, C=16/32/64/128 | 6.4e-31 ~ 7.1e-30 | 1.2e-32 |
| 极端：g≡-20 / g 交替 / β≡1 | ≤1.2e-26 | ≤7.5e-27 |

结论：分块与串行**数学等价、误差纯舍入**（f64 下 ~1e-29），f32 实现预计 NMSE ~1e-12，距门禁 1e-7 有 5 个数量级余量；上表含 C=128 与各种尾块，说明块长不敏感。**尚未验证**：KDA（逐行衰减）与 K>1 快照路径的分块形式（当前设计里这两者回退旧内核）。

### 15.3 实现计划（待批准后动手）

- 内核 A（并行 over (seq, v-head, chunk)）：算 GS、KKᵀ、A、T（前代/分块求解）、P、W、`tril(QKᵀ)⊙R`、`Q_eff`、`Kw = Kᵀdiag(w)P`，写 scratch（`ggml_cuda_pool_alloc(ctx.pool())`，ggml-cuda 既有惯例）。
- 内核 B（每 (seq, v-head) 一个 block 串行过 chunk）：`K S_in`、`Q_eff S_in`、`W(K S_in)`、`Kᵀdiag(w)(P − W K S_in)`，输出 o 与最终状态。
- 覆盖面：先只做热路径（非 KDA、K=1、连续 q/k/v、n_tokens ≥ C）；KDA、K>1 快照、permuted、kda 小 head 尺寸**全部回退现有内核**（保证门禁 36 例与整模型行为不变）。
- 门禁：`test-backend-ops -o GATED_DELTA_NET` 36 例（其中 n_seq_tokens=64..256 的用例会走新路径）+ 整模型 logits 对照（`fused_gdn_ch=false` 走纯原语分解路径可作对照）。
- 预期收益（估）：内核 3~4×（174 → 50~60 ms/批）→ W −9~11% → d0 +9~11%、8k +8~10%、16k +7~9%、256k +2.2~2.6%；代价是 d² 级 FLOP 翻倍但算术强度/并行度大幅改善（现内核 FP32 效率仅 11%）。

### 15.4 风险

1. **工作量与维护**：新增 ~600-900 行 CUDA（新内核 + dispatch + scratch），是本分支最大的一处非上游新增代码；仓库规范要求大改先设计再动手（本文即该设计）。
2. **并行度上限**：内核 B 每个 (seq,head) 串行过 chunk，只有 H·n_seqs 个 block（9B 预填充 = 32），低于 72 SM；先按简单版做，实测若成瓶颈再上分段扫描（hierarchical scan）。**这是收益能否兑现的最大不确定性。**
3. **数值**：15.2 已把主要风险消掉，但 f32 下的舍入还要看实现顺序；仍有 5 个数量级余量。
4. **256k 收益有限**：即便内核 4×，W 也只 −10%，256k 因 FA 占 76% 只有 +2.2~2.6%。

### 15.5 产物

- `dnet-chunk-prototype.py`（数值原型，含安全形式的完整实现与门禁同分布的自测）。
- 本 §15 即设计文档；未写任何 CUDA，`git status --short` 为空（HEAD `3148fd037`）。

## 16. cuBLAS 能否被替换：强制 dp4a MMQ 的实证否证（2026-09-12，单一变量 A/B）

### 16.1 结论

- **预填充不可换**：强制 dp4a MMQ（等价 `-DGGML_CUDA_FORCE_CUBLAS=OFF -DGGML_CUDA_FORCE_MMQ=ON`）在 ub=4096 下比 cuBLAS **慢 1.8–2.0×**。
- **小批窗口是 MMQ 的地盘**：pp32（ne11=32）MMQ 比 cuBLAS **快 2.4×** —— 与上游门槛 `ne11 < 64` 完全一致。
- **cuBLAS 效率标尺**（§13.2 + 本机微基准，账本 229–230 行；复核见 §17.5）：本机 cuBLAS 最佳工况 8192³ fp16/fp32acc = 85.5 TFLOPS（复跑确认）；模型真实形状混合聚合 ≈82 TFLOPS ≈ 自身最佳的 ~96% ⇒ **无更换空间**（手写内核的理论天花板也只有 GEMM 段 +10~15%，工期以周计）。
- **顺带的配置勘误**：本 build 的 `FORCE_CUBLAS=ON` 实际影响面 = **ne11 ∈ [9,63]**（把上游默认给 MMQ 的小批窗口强制给 cuBLAS）；decode（ne11≤8）走 MMVQ、预填充（≥64）走 cuBLAS，都不经此判定。生产 `models.ini` 开了 `spec-type=draft-mtp`，其 verify 批次（槽位 × (1+draft)）正可能落在这个窗口。
- 对 256k 目标的意义：GEMM ≈ 13%（684.7 ms/4096-token 批 vs 256k 整批 5.32 s）⇒ 即便 GEMM 完全免费也只 +15%；"换库"路径实证为零，方向关闭。

### 16.2 实测（ABBA 2 轮，9B Q4_K_M，`-fa 1 -ctk/-ctv q8_0 -b 8192 -ub 4096`）

| 协议 | cuBLAS（405cfc0d） | 强制 MMQ（f5dab96c） | 比值 |
| --- | ---: | ---: | ---: |
| pp8192 | 3466.8 / 3357.3 → 均值 **3412** | 1700.6 / 1700.9 → **1701** | MMQ 慢 **2.01×** |
| pp8192 @ d8192 | 3121.5 / 3040.4 → **3081** | 1619.8 / 1619.3 → **1620** | 慢 **1.90×** |
| pp8192 @ d16384 | 2859.4 / 2784.9 → **2822** | 1545.2 / 1542.4 → **1544** | 慢 **1.83×** |
| pp32（ne11=32） | 353.2 / 359.5 → **356** | 857.8 / 858.1 → **858** | MMQ **快 2.41×** |

- 漂移：cuBLAS 侧 r1→r2 掉 2.6–3.2%（热漂移，ABBA 已抵消方向）；MMQ 侧两轮几乎不动（1700.58/1700.91）。2× 量级结论不受影响。
- pp32 单次仅 ~90/37 ms，轮内误差 ±38/±110（±11–13%），但两轮一致（353.2/359.5、857.8/858.1），2.41× 远超噪声。
- 基线绝对值与 §13.1 的 3394.61 t/s 同档（3412/3357，±2% 漂移），基线库与历史一致。

### 16.3 方法与产物

- mmq-exp 库：临时等价补丁（禁用 `FORCE_CUBLAS` 提前返回（mmq.cu:267）+ NVIDIA 分支强制 `return true`（mmq.cu:334）），等价于 `FORCE_CUBLAS=OFF + FORCE_MMQ=ON`；**补丁已 `git checkout` 回退，工作区干净**；基线重建后 sha 与实验前一致（405cfc0d…，构建可复现）。
- 快照：`fa-libs/mmq-exp`（f5dab96c…）；基线 `fa-libs/deltanet-pipe`（405cfc0d…）。
- 脚本 `9b-mmq-ab.sh`（ABBA + pp32 附测 + 每侧打印 `ldd` 解析结果）；日志 `fa-logs/9b-mmq-ab-20260912.log`；构建日志 `build-mmq-exp-20260912.log` / `build-mmq-revert-20260912.log`。

### 16.4 基础设施勘误（影响所有快照换库实验，含历史）

1. **`LD_LIBRARY_PATH` 快照换库要求目录内存在 `libggml-cuda.so.0` 符号链接**：本库 SONAME = `libggml-cuda.so.0`（DT_NEEDED 同名），加载器按精确文件名搜索；快照目录若只有 `libggml-cuda.so.0.23.0`，会**静默回退**到 RUNPATH（`/root/llama.cpp/build/bin`）——A/B 两侧实际加载同一个库。
   - 本次已给 `deltanet-pipe`/`mmq-exp` 补齐 `.so.0`/`.so` 链接，并在 A/B 脚本打印每侧 `ldd` 解析结果作为证据（`resolved_*` 行，本次均指向快照目录）。
   - 历史复核：`e3d`/`stock` 目录当时已有符号链接（换库真实生效）；`deltanet-pipe` 目录当时**没有**链接，实际靠回退到 build/bin（当时 build/bin 恰好就是目标新库，13:40:41 构建）——§14 的 ncu 与墙钟结论仍成立，机制按本条修正为准。
2. **`FORCE_CUBLAS` 影响面修正**：只影响 `ggml_cuda_should_use_mmq` 一个函数；decode（ne11≤8）在 `should_use_mmvq`（`mmvq.cu:318`，V100 走 `ne11 <= MMVQ_MAX_BATCH_SIZE(=8)`）就已分流，不受影响；受影响窗口 = ne11 ∈ [9,63]。
3. flag 作用点全树核实：`GGML_CUDA_FORCE_MMQ` / `GGML_CUDA_FORCE_CUBLAS` 各自只守 `mmq.cu:329` / `mmq.cu:267` 一处行为 + `ggml-cuda.cu:5627` 特性字符串（`grep -rn` 全树唯一）。

### 16.5 已采纳并验证（2026-09-12）

- `llama.sh`：`-DGGML_CUDA_FORCE_CUBLAS=ON` → **OFF**（两个 FORCE_* 都不开 = 上游默认按批分流），中文注释同步改为实测依据（用户确认采纳）。
- 验证（`9b-cublas-off-ab.sh`，ABBA 2 轮；旧 `fa-libs/deltanet-pipe` vs 新 `fa-libs/cublas-off`）：

| 协议 | 旧（ON） | 新（OFF） | Δ |
| --- | ---: | ---: | ---: |
| pp32（ne11=32） | 359.63 / 359.63 → **359.6** | 859.69 / 859.17 → **859.4** | **+2.39×** |
| pp8192 | 3454.7 / 3408.9 → **3431.8** | 3433.3 / 3422.1 → **3427.7** | −0.12%（噪声内） |
| pp8192 @ d8192 | 3115.2 / 3074.5 → **3094.8** | 3102.2 / 3090.1 → **3096.2** | +0.04%（噪声内） |

- flag 生效硬证据：feature 串 `FORCE_CUBLAS` 在旧库出现 1 次、新库 **0** 次（`strings … | grep -c '^FORCE_CUBLAS$'`）。
- 新库独立复现了 §16.2 MMQ 侧结果（pp32 859.4 ≈ 858），交叉确认 mmq-exp 等价补丁的正确性。
- 新快照：`fa-libs/cublas-off`（sha `eb6041f8…`）；旧基线保留 `fa-libs/deltanet-pipe`（`405cfc0d…`）。今后代码 A/B 的基线用 `cublas-off`。

## 17. 当前构建全量复核与账本勘误（2026-09-12 下午）

目的（用户指令）：账本必须完全准确、诚实，且反映**当前代码**的性能；所有头条数字必须机器实测。本节把 §1–§16 的全部头条数字在**当前构建**上重测，并修正上一轮对账发现的无依据推断。

### 17.1 口径与产物

- **当前代码** = HEAD `3148fd037`（在 FA 调优 `19a667d32` 与 merge `8def8f970` 之上叠加 delta-net 流水化），`git status --short` 为空。
- **当前生产库** = `build/bin/libggml-cuda.so.0.23.0`，sha256 `eb6041f8…`（快照 `fa-libs/cublas-off`）；`llama.sh` = `FORCE_CUBLAS=OFF + FORCE_MMQ=OFF`（§16.5 已采纳并验证）。
- 清场条件：`llama-server` 停止、`nvidia-smi --query-compute-apps` 为空；本机两卡（PCI 序：0=GTX 1650 Ti / 1=Tesla PG503-216），所有脚本显式 `CUDA_DEVICE_ORDER=PCI_BUS_ID` 后以 `-dev CUDA1` 指 V100（默认设备序下 CUDA1 是 GTX，见 §17.5 陷阱）。
- 本轮产物（终态路径）：
  - `9b-current-verify.sh` → `fa-logs/9b-current-verify-20260912.log`（S1–S6，`CURVERIFY_DONE 14:59:29`）
  - `9b-mmq-crossover.sh` → `fa-logs/9b-mmq-crossover-20260912.log`（`CROSSOVER_DONE 15:09:27`，§17.7）
  - `9b-attr-ab.sh` → `fa-logs/9b-attr-ab-20260912.log`（`ATTRAB_DONE 15:14:40`，§17.8）
  - `mma-peak2.cu`（微基准扫描版）→ `fa-logs/mma-peak-recheck-20260912.log`（§17.5）
  - 重生成门禁日志 `fa-logs/tbo-current-256.log`、`fa-logs/tbo-stock-256.log`（各 32553 B）
- 换库证据：每个脚本对每侧打印 sha256 与 `ldd` 解析（本轮三侧 `resolved_e3d` / `resolved_deltanet-pipe` / `resolved_cublas-off` 均指向 `fa-libs/` 对应目录）。

### 17.2 头条数字重测（当前构建）

| 协议 | 账本历史值 | 当前构建实测 | Δ |
| --- | ---: | ---: | ---: |
| pp4096 d0（§1.1 e3d 清场基线） | 3340.08 | **3375.36**（三库 ABBA 的 cublas-off 均值，§17.8） | +0.6% |
| pp4096 @d16384（同上） | 2782.96 | **2800.64** | +0.3% |
| pp4096 @d32768（同上） | 2367.54 | **2375.35** | +0.1% |
| pp8192（§16.5 cublas-off AB） | 3427.7 | 3430.08 ± 2.86 | +0.07% |
| pp8192 @d8192（§16.5） | 3096.2 | 3096.86 ± 7.54 | +0.02% |
| pp8192 @d16384（§16.2 cuBLAS 侧） | 2822 | 2832.87 ± 5.76 | +0.4% |
| pp4096 @d262144（§1.1 256k，-r 1） | 769.87 | **774.50 ± 0.70**（-r 2） | +0.6% |
| 小批剖面（新基线） | - | pp8/9/32/63/64/128 = 320.91±47.34 / 335.75±42.96 / 875.09±73.93 / 1120.88±68.87 / 614.27±26.92 / 1083.62±42.03 | 见 §17.7 |

- S1 的单次读数（14:5x，`-b 4096 -ub 4096 -p 4096 -d 0,16384,32768 -r 2`）= 3454.41 ± 24.67 / 2853.04 ± 12.42 / 2418.13 ± 13.40；其相对 §1.1 的 +3.4/+2.5/+2.1% **未能复现**——同会话 ABBA（§17.8）判明为热漂移，净代码增量 = +0.6/+0.3/+0.1%。
- 结论：全部头条数字在当前构建上复现，偏差 ≤0.6% 且方向一致地略快；无失真、无退化。

### 17.3 正确性门禁（8 个 tbo 日志全量核对）

在 `fa-logs/` 下可复现：

```sh
# 每个日志的用例名集合指纹（排序后 md5；应为同一指纹）
for f in tbo-*.log; do printf "%s " "$f"; grep -oE '^  FLASH_ATTN_EXT\([^:]*\):' "$f" | sort | md5sum; done
# 彩色判决行计数（应为 143 = 141 用例 + Backend CUDA1 + 程序收尾）
grep -ac $'\033\[1;32mOK' tbo-*.log
```

- 8 个日志（`tbo-current-256`、`tbo-stock-256`、`tbo-e1-256`、`tbo-e3d-256`、`tbo-e3d-hsk256-20260912`、`tbo-rescale-hsk256-9b`、`tbo-streamk-hsk256-9b`、`tbo-va-hsk256-20260912`）：各 **141** 个用例，集合指纹全部 = `b6bdb91af86e86cd4bdf28ff136a640f`（两两相同）；彩色 `OK` 全部 = **143**；剥色后 `FAIL` 计数 0。
- 判决落点两种：同行 `): OK`（130 例）或独立行 `OK`（11 例，CUDA-graph warmup 消息插在 `): ` 与 `OK` 之间）；两类之和每文件 = 141。

### 17.4 计数勘误（"143" 的最终解释）

- 用例数 = **141**；"143" = 141 用例判决 + `Backend CUDA1: OK` + 程序收尾 `OK`（彩色 `grep -c OK` 把三类都算上；逐文件核实均 143）。
- 行级提取的两个伪结论（均已在 §7 E6 等就地修正）：①"stock 130 例 vs e3d 129 例"——实为一例（`nh=4,nr23=[6,1],kv=4096,nb=64,…,kv_view=1`）的判决行被 warmup 消息拆行，按"行尾 `): OK`"提取漏判；按用例名核对后 8 日志集合全同。②"143 是用例数"——见上。

### 17.5 微基准复核与 §5 勘误

- 复跑 `mma-peak2.cu`（与原始 `mma-peak.cu` 同一内联汇编与形态）：4 chains / 576×128 = **45.05 Gmma/s**（原始 45.38，±0.7% 内复现）；12 配置扫描（4/8/16 chains × 576×128/256）= 30.80–46.69 Gmma/s，最大在 16 chains / 576×256。
- **设备序陷阱**：CUDA 默认 `FASTEST_FIRST` 下 `dev=1` = GTX 1650 Ti（本次首跑 0.21 Gmma/s、SMs 16 即落错设备）；必须 `CUDA_DEVICE_ORDER=PCI_BUS_ID`。原始 `mma-peak.log` 的 "device 1: Tesla PG503-216 | SMs 72" 与 PCI 序一致，历史数据成立。
- 每条 `mma.sync.m8n8k4` = 4 个 SASS STEP：ncu 两次独立测得 73,728,000 ÷ 18,432,000 = **4.0**、147,456,000 ÷ 36,864,000 = **4.0**；静态 SASS 亦为 FA 内核 768×4 = 3072 条 `HMMA.884.F32.F32.STEP0..3`；`mma.cuh` Volta 分支每个逻辑 mma() 展开 2 条 `mma.sync`。
- cuBLAS 标尺复跑：8192³ fp16/fp32acc = 64.3 ms → **85.5 TFLOPS**（原始 85.4）。
- **§5 勘误（已就地修正）**：45.05 Gmma/s 只是"同形态相对锚"（换算 23.1 TFLOPS ≪ cuBLAS 85.5，未打满），不能当卡峰值；"93–102 TFLOPS / 2048 FLOP 每 mma / 200 G/s / 20.6% / 13.6% / 29.6% / 58 TFLOPS 57–63% / 40–47% / 2.4–3.4×"等推断全部撤回。改为可证口径：三配置同形状 STEP 速率 **27.1 / 41.2 / 59.3 G STEP/s**（= 参考锚 45.05×4 = 180.2 G/s 的 **15.0% / 22.9% / 32.9%** 忙度；对应 ncu 内核时长 52.472 / 34.531 / 23.970 ms，STEP 计数同为 1.4219e9）。
- 绝对 FLOP 口径同时撤回：按"Q=4096 / KV≈20480 / nh=16"因果几何推算需 ~2.42e9 次 mma，与实测执行 3.55e8 次相差 **6.8×**，说明几何模型不能与内核计数直接换算。

### 17.6 就地编辑索引与路径约定

- 本轮就地编辑 4 处：§5 算力口径块、§6 第 4 条、§6 第 7 条、§16.1 cuBLAS 标尺；上一轮另有 9 处（含 `143→141` 等计数勘误）。
- 路径约定（§0 已声明）：文中 `/tmp/<name>` 均映射到 `fa-logs/<name>`（迁移清单 `fa-logs/.archived-list.txt`）；库快照统一 `fa-libs/<标签>/`，目录需含 `libggml-cuda.so.0` 符号链接（§16.4）。
- 快照清单（本轮逐一复核 sha256 前缀）：`cublas-off`（eb6041f8，唯一生产基线）、`deltanet-pipe`（405cfc0d）、`e3d`（e2660404）、`mmq-exp`（f5dab96c）、`stock`（754fbf6f）。

### 17.7 小批分流交叉实验：上游门槛 `ne11 < 64` 在 V100 上切错（新发现）

同会话 ABBA（2 轮 × 8 档；当前默认分流库 `cublas-off` vs 强制 MMQ 库 `mmq-exp`；`-b 8192 -ub 4096 -d 0 -r 3`）：

| ne11 | 默认分流（cuBLAS 侧）r1 / r2 → 均值 | 强制 MMQ（mmq-exp）r1 / r2 → 均值 | 优胜 |
| ---: | ---: | ---: | --- |
| 48 | 977.98 / 977.86 → 977.9 | 974.22 / 978.87 → 976.5 | 两侧同走 MMQ（门槛 <64），持平 |
| **64** | 606.53 / 605.70 → **606.1** | 1106.30 / 1114.99 → **1110.6** | **MMQ 快 1.83×** |
| 128 | 1071.25 / 1069.80 → 1070.5 | 1262.40 / 1263.75 → 1263.1 | MMQ 快 1.18× |
| 256 | 1764.05 / 1762.65 → 1763.4 | 1450.63 / 1451.36 → 1451.0 | cuBLAS 快 1.215× |
| 512 | 2584.16 / 2570.63 → 2577.4 | 1589.78 / 1589.01 → 1589.4 | cuBLAS 快 1.62× |
| 1024 | 3093.63 / 3020.19 → 3056.9 | 1674.42 / 1673.71 → 1674.1 | cuBLAS 快 1.83× |
| 2048 | 3405.33 / 3377.42 → 3391.4 | 1698.22 / 1697.81 → 1698.0 | cuBLAS 快 2.00× |
| 4096 | 3461.71 / 3414.83 → 3438.3 | 1704.53 / 1705.51 → 1705.0 | cuBLAS 快 2.02× |

- 读法：默认分流在 `ne11 < 64` 走 MMQ、`>= 64` 走 cuBLAS；**真实交叉点线性内插 ≈ ne11 177**（128 处 MMQ 快 18%，256 处 cuBLAS 快 21.5%）。即 ne11 ∈ [64, ~176] 区间默认分流选了慢 1.2–1.83× 的 cuBLAS。
- 稳定性：MMQ 侧跨轮零漂移（pp4096：1704.53/1705.51）；cuBLAS 侧跨轮漂 1.0–2.4%（热态敏感）。ABBA 位序已平衡（r1：cuBLAS→MMQ，r2 反序）。
- 旁证：§16.5 的 pp32（ne11=32）MMQ +2.39× 同源（<64 窗口本就归 MMQ）；§17.2 的 S3 剖面 pp63 = 1120.88 → pp64 = 614.27 即本表的 64 台阶（−45%）。
- 影响面与后续：`-ub 4096` 主预填充只在**末尾残块**落窗；受影响更大的是小批负载（短 prompt、投机解码 verify 批）。修复 = 改 `ggml-cuda` 的 MMQ 门槛（sm_70 特化，64 → ~192，需留余量），属**代码改动**，本轮未动代码，只记录实测交叉点；其他量化类型 / 更深上下文下的门槛需另行 A/B。

### 17.8 S1 归因：同协议三路 ABBA 与热漂移量化

三库同会话 ABBA（`9b-attr-ab.sh`；协议与 §1.1 完全相同：`-b 4096 -ub 4096 -p 4096 -d 0,16384,32768 -n 0 -r 2`；r1 顺序 e3d→deltanet-pipe→cublas-off，r2 反序，位序平衡）：

| 库（sha） | pp4096 | @d16384 | @d32768 |
| --- | ---: | ---: | ---: |
| e3d（e2660404，§1.1 基线快照） | 3396.04 / 3316.12 → **3356.08** | 2826.23 / 2759.04 → **2792.64** | 2401.57 / 2344.44 → **2373.01** |
| deltanet-pipe（405cfc0d，+delta-net） | 3407.35 / 3356.20 → **3381.78** | 2824.29 / 2776.68 → **2800.49** | 2394.40 / 2370.33 → **2382.37** |
| cublas-off（eb6041f8，= 当前生产库） | 3383.20 / 3367.51 → **3375.36** | 2808.49 / 2792.78 → **2800.64** | 2385.95 / 2364.74 → **2375.35** |

- **归因结论**：当前库相对 §1.1 基线 e3d = **+0.57% / +0.29% / +0.10%**（d0/d16384/d32768）；增量主要来自 delta-net 流水化（e3d→deltanet-pipe = +0.77% / +0.28% / +0.39%，与 §14 的 +0.80% / +0.66% / +0.54% 同量级、方向一致）；配置切换在预填充档无可测差异（deltanet-pipe vs cublas-off = −0.19% / +0.01% / −0.29%，噪声内，与 §16.5 一致）。
- **S1 的 +3.4/+2.5/+2.1% 不可复现**：e3d 快照本轮 = 3356.08（对 §1.1 的 3340.08 为 +0.5%，基线本身可复现）；同一命令测 cublas-off 在今天 14:5x = 3454.41、15:12 = 3375.36（−2.3%）。本轮 ABBA 内部全部读数随时间单调下降 ~2.4%（3396→3316，3.5 分钟）；S1 起测时 GPU 冷态（135 MHz / 33°C，见 `9b-current-verify` 日志首行 nvidia-smi），ABBA 则紧跟上一实验开测（开测前 55°C）。
- **口径修正**：本机 ±2% 级热漂移是常态；跨时段只可比趋势（±3%），亚百分比比较必须用同会话 ABBA。账本头条以 §17.2 的复现值/ABBA 均值为准。

---

## §18 V100 MMQ 分流门槛修复：代码改动 + 正确性门禁 + 三侧 ABBA（2026-09-12）

### 18.1 改动内容

- 位置：`ggml/src/ggml-cuda/mmq.cu` → `ggml_cuda_should_use_mmq()` 的 NVIDIA 分支（原第 334 行）。
- 旧行为（V100 / sm_70，FORCE_CUBLAS=OFF、FORCE_MMQ=OFF）：量化 `MUL_MAT` 仅在 `ne11 < MMQ_DP4A_MAX_BATCH_SIZE`（=64）时走 MMQ，其余走 cuBLAS。§17 实测交点 ≈177 → **[64,177) 整个窗口白白丢掉 MMQ**。
- 新行为：V100 专档，`volta_mma_available(cc)` 时 `ne11 < 176` 走 MMQ；其余 NVIDIA 显卡判定不变。

```diff
     if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
+        if (volta_mma_available(cc)) {
+            // on V100 MMQ is faster than cuBLAS for batch sizes below ~176 (measured crossover)
+            return ne11 < 176;
+        }
         return !fp16_mma_hardware_available(cc) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
     }
```

- 影响面（代码可证）：只有「sm_70 + 量化类型 + ne11∈[64,176)」判定翻转；非量化类型在更早的 `mmq_supported` 处返回 false，非 Volta 显卡不进入该分支，其余判定原样。`/root/llama.sh` 的路由注释同步更新（中文）。
- 复现：`/root/llama.sh make`（日志 `fa-logs/build-mmq-gate-20260912.log`，30 s，ccache 命中）。

### 18.2 构建产物与快照

- 新库：`build/bin/libggml-cuda.so.0.23.0`，sha256 `4efef6a8…`；快照 `fa-libs/mmq-gate/`（含 `.so.0`/`.so` 软链）。
- 基线库：`fa-libs/cublas-off`（sha `eb6041f8…`，= 改动前生产库）；对照库：`fa-libs/mmq-exp`（sha `f5dab96c…`，强制全 MMQ）。
- 验证脚本：`9b-mmq-gate-verify.sh`（门禁）、`9b-mmq-gate-ppl-matrix.sh`（ppl 矩阵）、`9b-mmq-gate-nsys-b176.sh`（内核级归因）、`9b-mmq-gate-ab.sh`（ABBA）。

### 18.3 正确性门禁

**(a) test-backend-ops（vs CPU 参考）**，均在 V100（`-b CUDA1`）上跑：

| 运行 | 库 | 结果 |
| --- | --- | --- |
| MUL_MAT_ID（含窗口内 bs=64/128/129 量化用例） | mmq-gate | 4/4 backends passed，0 FAIL（日志内 606 条 OK；窗口内量化用例 31 条全 OK） |
| MUL_MAT_ID（对照） | cublas-off | 4/4 backends passed，0 FAIL（600 条 OK；窗口内量化用例 29 条全 OK） |
| MUL_MAT 全家族（窗口外回归） | mmq-gate | 4/4 backends passed，0 FAIL（745 条 OK） |

dense `MUL_MAT` 用例的 n 只有 <64 与 ≥512（窗口外，判定不变），窗口内覆盖全部来自 `MUL_MAT_ID` 的 bs=64/128/129（量化 type_a：q4_0/q8_0/q4_k/q6_k/iq2_xs 等）。

**(b) ppl 矩阵（9B Q4_K_M，README 语料 2525 token；同配置双库对比，确定性可复现）**：

| `-b/-ub` | mmq-gate | cublas-off | mmq-exp（强制 MMQ） | 判定路径（新/旧） |
| ---: | ---: | ---: | ---: | --- |
| 48 | 3.7081 | 3.7081 | — | MMQ / MMQ，逐位一致 |
| 64 | 3.7065 | 3.6737 | 3.7065 | MMQ / cuBLAS；gate == exp（逐位一致） |
| 128 | 3.7113 | 3.6740 | 3.7113 | 同上 |
| 175 | 3.7164 | 3.6738 | — | 同上 |
| 176 | 3.6878 | 3.6743 | — | 权重矩阵乘均为 cuBLAS；差 0.012% 来自 LM head（见下） |
| 512 | 3.6737 | 3.6737 | — | cuBLAS / cuBLAS，逐位一致 |

- **窗口内 MMQ 的代价是 +0.9%~+1.2% ppl**（对 cuBLAS）。这是 MMQ 对激活做 q8_1 量化的固有精度差，不是新 bug：`mmq-gate` 与「强制全 MMQ」库在 b=64/128 上**逐位一致**（3.7065 / 3.7113），差异只随「是否走 MMQ」出现，不随库出现。上游对 Turing+ 本来就是全量 MMQ（V100 的 `ne11≤63` 也一直如此），该量级的取舍与上游现状同源。
- **ppl 对路由完全敏感、是可靠的路由探针**：`-b/-ub=48`（新旧都 MMQ）与 `512`（新旧都 cuBLAS）两处双库逐位一致，验证了方法学。

**(c) 内核级归因（nsys，`-c 176 -b 176 -ub 176` 各库一次，`cuda_gpu_kern_sum` 对照）**：

- 唯一差异内核：gate 侧 `mul_mat_q<(ggml_type)14, …>`（14 次 × 6.99 ms）+ 配套 `quantize_mmq_q8_1`（14 次）；base 侧对应 `volta_s884gemm_fp16_256x64_ldg8_tn`（14 次 × 4.23 ms）；`dequantize_block_q6_K` 次数 448 vs 462（base 多 14 次）。
- 结论：c=176 时权重矩阵乘的 ne11=176（两侧都 cuBLAS），换掉的 14 次内核是 **LM head（Q6_K，type 14）在 ne11=175（n_outputs = chunk-1）上的一次/每 chunk** —— 即 18.3(b) 中 0.012% 差异的全部来源。
- 附带发现：**LM head 形状（K=4096, N≈152k）的 MMQ 在 ne11=175 比 cuBLAS 慢 65%**（6.99 vs 4.23 ms，单个 GEMM）；该形状的交点明显低于常规权重矩阵（N=4k~14k）。对 server 预填充无影响（logits 只算最后 1 个 token，ne11=1 → MMVQ）；只影响「整批算 logits」的离线工具。

### 18.4 性能验证（三侧 ABBA）

协议与 §17 交叉点实验一致：9B Q4_K_M `-b 8192 -ub 4096 -p 48,64,128,175,176,192,512,4096 -d 0 -n 0 -r 3`，r1 顺序 base→gate→exp、r2 反序，位序平衡；每侧 sha256 + ldd 校验（`9b-mmq-gate-ab.sh`，日志 `fa-logs/9b-mmq-gate-ab-20260912.log`）。

| pp | cublas-off（旧，r1/r2） | mmq-gate（新，r1/r2） | mmq-exp（强制 MMQ） | gate/base | gate/exp |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 48 | 976.7 / 977.0 | 978.4 / 977.3 | 978.5 / 977.5 | 1.001 | 1.000 |
| 64 | 605.9 / 606.1 | 1114.6 / 1113.5 | 1113.5 / 1113.6 | **1.838** | 1.000 |
| 128 | 1070.2 / 1070.0 | 1264.5 / 1263.9 | 1263.5 / 1263.4 | **1.181** | 1.001 |
| 175 | 1353.7 / 1348.3 | 1318.0 / 1315.8 | 1315.6 / 1316.6 | 0.975 | 1.001 |
| 176 | 1369.0 / 1370.7 | 1367.6 / 1365.9 | 1324.1 / 1323.8 | 0.998 | **1.032** |
| 192 | 1393.0 / 1394.1 | 1396.0 / 1392.9 | 1430.2 / 1429.9 | 1.001 | **0.975** |
| 512 | 2579.2 / 2572.4 | 2578.3 / 2572.1 | 1589.1 / 1589.7 | 1.000 | **1.620** |
| 4096 | 3450.2 / 3415.3 | 3446.3 / 3424.4 | 1705.5 / 1704.3 | 1.001 | **2.015** |

- **断言全部成立**：pp48（噪声地板）三侧同值；pp64/128/175 gate 与强制 MMQ 逐点一致（1.000~1.001），说明窗口内确实全部换到 MMQ；pp176/192/512/4096 gate 与旧库一致（0.998~1.001），而强制 MMQ 在这几档明显更慢（1.032 / 0.975 / 1.620 / 2.015），说明 ≥176 仍走 cuBLAS。**门槛恰好落在 176**。
- **收益**：窗口内预填充 +18%（pp128）~ +84%（pp64）；≥176 与改动前逐点相同（±0.2% 内），生产档（ub=4096）不受影响。

### 18.5 交点复核与门槛取值的诚实标注

- §17 用 pp128/pp256 两点线性内插得交点 ≈177；本轮新增 pp175 实测点后修正为 **≈169**：pp175 上 cuBLAS 比 MMQ 快 2.5%（1351.0 vs 1316.9），pp128 上 MMQ 快 18.1% → 曲线在 128~175 区间不是直线。
- 因此门槛 176 在 **[169,176) 有 ≤2.6% 的小回退**（仅在 ub/ne11 落在 169~175 时出现，该区间极窄）；换来的收益是 [64,169) 的 +18%~+84%。取舍按「最大化窗口收益」取向，并在此如实记录。
- 若要消除该小回退，可把阈值下调到 168；不影响 [64,168) 的收益，仅放弃 169~175 的一段（该段本身≤2.6%）。**待定，见 18.6。**

### 18.6 结论与未决项

- 结论：改动达到预期 —— V100 上 [64,176) 窗口的预填充从 cuBLAS 切到 MMQ，实测 +18%~+84%，≥176 无损；正确性门禁（tbo 双库 0 FAIL + ppl 探针 + nsys 归因）全部通过；窗口内 ppl +0.9%~+1.2%（MMQ 固有）。
- 未决：阈值是否从 176 下调到 168（见 18.5）；LM head 特例（N≈152k 时 MMQ 在 175 慢 65%）是否值得按形状单独处理（当前判定函数拿不到 N，需改签名，暂不做）。
- 代码状态：改动仅 `mmq.cu` 4 行 + `/root/llama.sh` 注释；**已提交 `6f81ea836`**（`v100` 分支，本地未推送，正文含 `Assisted-by: Qwen Code`；原 Responses 版本 `eecfa404f` 已随目录/分支调整重放，映射见 §0 注）。

