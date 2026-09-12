#!/usr/bin/env bash
# W 项内核级定性：①gated_delta_net（W 的 15%，串行 scan）看是否延迟/访存受限；
#                  ②最大 f16 GEMM（cutlass 128x128，W 的 ~50%）看离张量核峰值还差多少。
# 协议沿用 bench 位点：-p 8192 -d 8192 -n 0 -r 1；--cache-control none 保持与真实运行一致的缓存行为。
set -u

cd /root/llama.cpp || exit 1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

M9=/root/llama_gguf/Qwen3.5-9B-Q4_K_M/Qwen3.5-9B-Q4_K_M.gguf
B=./build/bin/llama-bench

COMMON="gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,smsp__issue_active.avg.pct_of_peak_sustained_active,sm__warps_active.avg.pct_of_peak_sustained_active,dram__bytes.sum,launch__grid_size,launch__block_size,launch__registers_per_thread,launch__waves_per_multiprocessor"
STALLS="smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,smsp__average_warps_issue_stalled_wait_per_issue_active.ratio,smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio,smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio,smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio"
TENSOR="smsp__inst_executed.sum,sm__inst_executed_pipe_tensor.sum,smsp__inst_executed_pipe_tensor.sum"

echo "== fa-ncu-wterm 9B (W-term kernel characterization) 2026-09-12"

echo ""
echo "===== NCU gated_delta_net (d8192, launch-skip 24) ====="
ncu -k "regex:gated_delta_net" --launch-skip 24 -c 1 --cache-control none --csv \
    --metrics "$COMMON,$STALLS" \
    $B -m "$M9" -dev CUDA1 -fa 1 -ctk q8_0 -ctv q8_0 -b 8192 -ub 4096 -p 8192 -d 8192 -n 0 -r 1
echo "EXIT_ncu_deltanet=$?"

echo ""
echo "===== NCU cutlass f16 GEMM (d8192, launch-skip 100) ====="
ncu -k "regex:cutlass::Kernel2" --launch-skip 100 -c 1 --cache-control none --csv \
    --metrics "$COMMON,$TENSOR" \
    $B -m "$M9" -dev CUDA1 -fa 1 -ctk q8_0 -ctv q8_0 -b 8192 -ub 4096 -p 8192 -d 8192 -n 0 -r 1
echo "EXIT_ncu_gemm=$?"

echo ""
echo "===== NCU volta_s884gemm_fp16_256x128 (d8192, launch-skip 50) ====="
ncu -k "regex:volta_s884gemm_fp16_256x128" --launch-skip 50 -c 1 --cache-control none --csv \
    --metrics "$COMMON,$TENSOR" \
    $B -m "$M9" -dev CUDA1 -fa 1 -ctk q8_0 -ctv q8_0 -b 8192 -ub 4096 -p 8192 -d 8192 -n 0 -r 1
echo "EXIT_ncu_volta=$?"

echo NCU_WTERM_9B_DONE
