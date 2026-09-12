#!/bin/bash
# A/B 轮换墙钟对比：A = stock 内核库，B = E1 内核库
# 用 LD_LIBRARY_PATH 切换内核库（llama-bench 是 RUNPATH，LD_LIBRARY_PATH 优先），不动 build/bin
# A/B/A/B 四轮同会话跑，消除时钟漂移；每轮 3 个深度点、r=2
set -u
cd /root/llama.cpp || exit 1

MODEL=/root/llama_gguf/Qwen3.5-9B-Q4_K_M/Qwen3.5-9B-Q4_K_M.gguf
BASE_CMD="./build/bin/llama-bench -m $MODEL -dev CUDA1 -fa 1 -ctk q8_0 -ctv q8_0 -b 4096 -ub 4096 -p 4096 -d 0,16384,32768 -n 0 -r 2"

for round in 1 2; do
    for side in stock e1; do
        echo "=== round $round side $side ==="
        LD_LIBRARY_PATH=/tmp/falib-$side CUDA_DEVICE_ORDER=PCI_BUS_ID $BASE_CMD 2>&1 | grep -E "^\| qwen35|^build:"
    done
done
echo "AB_DONE"
