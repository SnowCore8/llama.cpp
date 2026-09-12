#!/usr/bin/env bash
# c=176 微差归因（2026-09-12）：nsys 内核追踪，找出门槛改动在 176 批配置下真正换掉的内核。
# 背景：ppl -c176 -b176 双库均确定性，但 gate=4.2403 vs base=4.2408（差 0.012%），
#       说明图内存在某个 ne11 属于 [64,176) 的量化矩阵乘；用内核名/次数差定位。
set -u
cd /root/llama.cpp || exit 1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

FLOGS=/root/llama.cpp/v100-prefill/fa-logs
M=/root/llama_gguf/Qwen3.5-9B-Q4_K_M/Qwen3.5-9B-Q4_K_M.gguf
PPL=./build/bin/llama-perplexity

for side in mmq-gate cublas-off; do
    OUT=$FLOGS/nsys-gate-b176-$side
    echo "== nsys $side $(date '+%T')"
    LD_LIBRARY_PATH=/root/llama.cpp/v100-prefill/fa-libs/$side nsys profile --force-overwrite=true --trace=cuda \
        --sample=none --cpuctxsw=none --cuda-graph-trace=node -o "$OUT" \
        $PPL -m "$M" -dev CUDA1 -fa 1 -ctk q8_0 -ctv q8_0 -f README.md -c 176 -b 176 -ub 176 > /dev/null 2>&1
    echo "EXIT_profile_$side=$?"
    nsys stats --force-export=true --report cuda_gpu_kern_sum --format csv "$OUT.nsys-rep" > "$OUT-kern.csv" 2>&1
    echo "EXIT_stats_$side=$? lines=$(wc -l < "$OUT-kern.csv")"
done
echo "NSYS_GATE_B176_DONE $(date '+%F %T')"
