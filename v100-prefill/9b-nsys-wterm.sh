#!/usr/bin/env bash
# 非 FA（W 项）归因：用 nsys 采 9B 预填充的内核级时间分布，回答"W 花在哪"。
# 协议与既有 benchmark 完全一致：-fa 1 -ctk q8_0 -ctv q8_0 -b 8192 -ub 4096 -p 8192 -n 0 -r 1
# 取 -d 0 的深（KV 深度 0..8192），此区间 W 占绝对主导，便于把 W 拆到内核族。
# 产出：fa-logs/nsys-wterm-9b-d0.nsys-rep（原始轨迹）+ -stats.txt（可读汇总）+ -kern.csv（供聚合）。
set -u

cd /root/llama.cpp || exit 1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

M9=/root/llama_gguf/Qwen3.5-9B-Q4_K_M/Qwen3.5-9B-Q4_K_M.gguf
B=./build/bin/llama-bench
OUT=/root/llama.cpp/v100-prefill/fa-logs/nsys-wterm-9b-d0

echo "== nsys W-term attribution, 9B, d0, $(date '+%F %T')"

nsys profile --force-overwrite=true --trace=cuda --sample=none --cpuctxsw=none --cuda-graph-trace=node \
    -o "$OUT" \
    $B -m "$M9" -dev CUDA1 -fa 1 -ctk q8_0 -ctv q8_0 -b 8192 -ub 4096 -p 8192 -d 0 -n 0 -r 1
echo "EXIT_nsys_profile=$?"

nsys stats --force-export=true --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum "$OUT.nsys-rep" > "$OUT-stats.txt" 2>&1
echo "EXIT_nsys_stats=$?"
echo "---- stats begin ----"
cat "$OUT-stats.txt"
echo "---- stats end ----"

nsys stats --force-export=true --report cuda_gpu_kern_sum --format csv "$OUT.nsys-rep" > "$OUT-kern.csv" 2>&1
echo "EXIT_nsys_csv=$?  lines=$(wc -l < "$OUT-kern.csv")"

echo NSYS_WTERM_9B_DONE
