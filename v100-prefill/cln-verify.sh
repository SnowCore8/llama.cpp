#!/usr/bin/env bash
# 清场后复核（服务已全部停止，无 GPU 干扰）：3 路 A/B/C 轮换 + 256k 实测点
# 用途：验证此前"E1/E3-(d) 结论"是否受共驻服务干扰；同时把 256k 外推换成实测
# 产物：/tmp/cln-verify.log（本文件）、/tmp/cln-256k-{e3d,stock}.log；侧标签 stock|e1|e3d 由 LD_LIBRARY_PATH 切换
set -u
cd /root/llama.cpp
M=/root/llama_gguf/Qwen3.5-9B-Q4_K_M/Qwen3.5-9B-Q4_K_M.gguf
B=./build/bin/llama-bench
COMMON="-m $M -dev CUDA1 -fa 1 -ctk q8_0 -ctv q8_0 -b 4096 -ub 4096"

echo "cln-verify (clean field, no service) 2026-09-12；侧: stock|e1|e3d"
echo "STEP0_START $(date '+%F %T')"

# 阶段 A：3 路 × 2 轮轮换（每轮含 d0/16384/32768）
for round in 1 2; do
    for side in stock e1 e3d; do
        echo "=== round $round side $side ==="
        LD_LIBRARY_PATH=/tmp/falib-$side CUDA_DEVICE_ORDER=PCI_BUS_ID $B $COMMON -p 4096 -d 0,16384,32768 -n 0 -r 2 2>&1 | grep -E "^\| qwen35|^build:"
    done
done
echo "STEP_AB_DONE $(date '+%F %T')"

# 阶段 B：256k 真实单点（e3d 先跑，随后 stock 对照；-r 1 控制时长）
for side in e3d stock; do
    echo "=== 256k side $side ==="
    LD_LIBRARY_PATH=/tmp/falib-$side CUDA_DEVICE_ORDER=PCI_BUS_ID $B $COMMON -p 4096 -d 262144 -n 0 -r 1 > /tmp/cln-256k-$side.log 2>&1
    echo "256K_EXIT_$side=$?"
    grep -E "^\| qwen35" /tmp/cln-256k-$side.log | grep 262144 || true
done
echo "STEP_256K_DONE $(date '+%F %T')"
