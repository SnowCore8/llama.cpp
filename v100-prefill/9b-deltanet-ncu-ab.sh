#!/usr/bin/env bash
# delta-net 流水化的内核级 A/B：同一 ncu 位点（d8192、launch-skip 24、cache-control none）
# 分别测基线库（fa-libs/e3d）与新库（fa-libs/deltanet-pipe），对比内核时长与 stall 结构。
# 位点与 §13 基线一致（当时 8.156 ms / long_scoreboard 4.51），便于逐项对照。
set -u

cd /root/llama.cpp || exit 1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

M9=/root/llama_gguf/Qwen3.5-9B-Q4_K_M/Qwen3.5-9B-Q4_K_M.gguf
B=./build/bin/llama-bench

COMMON="gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes.sum"
STALLS="smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,smsp__average_warps_issue_stalled_wait_per_issue_active.ratio,smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio,smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio"

for tag in e3d deltanet-pipe; do
    echo ""
    echo "===== NCU gated_delta_net [$tag] (d8192, launch-skip 24) ====="
    # LD_LIBRARY_PATH 覆盖 llama-bench 的 RUNPATH，仅替换 libggml-cuda.so 一个变量
    LD_LIBRARY_PATH=/root/llama.cpp/v100-prefill/fa-libs/$tag \
    ncu -k "regex:gated_delta_net" --launch-skip 24 -c 1 --cache-control none --csv \
        --metrics "$COMMON,$STALLS" \
        $B -m "$M9" -dev CUDA1 -fa 1 -ctk q8_0 -ctv q8_0 -b 8192 -ub 4096 -p 8192 -d 8192 -n 0 -r 1
    echo "EXIT_ncu_$tag=$?"
done

echo NCU_DELTANET_AB_DONE
