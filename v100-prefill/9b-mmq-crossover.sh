#!/usr/bin/env bash
# MMQ ↔ cuBLAS 交叉点测量（sm_70 / V100, 9B Q4_K_M, ub=4096, 单 chunk 形状）：
# 目的：上游门槛 ne11<64 在 V100 上是否最优——找两实现真正的交叉点。
# 侧：cublas-off（当前构建：<64 MMQ、>=64 cuBLAS） vs mmq-exp（全强制 MMQ）
# 对照点：pp48（两侧同为 MMQ，用于量化噪声地板）
# 次序 A,B,B,A（ABBA）；每侧一条命令跑全部尺寸，尺寸=48,64,128,256,512,1024,2048,4096
# 预期（已知点）：63 MMQ~1121 / 64 cuBLAS~614 / 128 cuBLAS~1084 / 4096 cuBLAS~3454 vs MMQ~1700
set -u
cd /root/llama.cpp || exit 1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

LOG=/root/llama.cpp/v100-prefill/fa-logs/9b-mmq-crossover-20260912.log
exec > >(tee "$LOG") 2>&1

M=/root/llama_gguf/Qwen3.5-9B-Q4_K_M/Qwen3.5-9B-Q4_K_M.gguf
B=./build/bin/llama-bench
LIBS=/root/llama.cpp/v100-prefill/fa-libs
ARGS=(-m "$M" -dev CUDA1 -fa 1 -ctk q8_0 -ctv q8_0 -b 8192 -ub 4096 -p 48,64,128,256,512,1024,2048,4096 -d 0 -n 0 -r 3)

echo "== 9b-mmq-crossover 开始 $(date '+%F %T') =="
echo "git_head=$(git rev-parse --short HEAD)  worktree_lines=$(git status --short | wc -l)"
echo "GPU占用（应为空）: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr '\n' ' ')"
for side in cublas-off mmq-exp; do
    echo "lib_${side}_sha256=$(sha256sum "$LIBS/$side/libggml-cuda.so.0.23.0" | awk '{print $1}')"
    echo "resolved_${side}=$(LD_LIBRARY_PATH="$LIBS/$side" ldd $B | awk '/libggml-cuda.so.0/ {print $3}')"
done

for round in 1 2; do
    if [ "$round" = 1 ]; then order="cublas-off mmq-exp"; else order="mmq-exp cublas-off"; fi
    for side in $order; do
        echo ""
        echo "===== r${round} ${side} pp48..4096 ====="
        LD_LIBRARY_PATH="$LIBS/$side" $B "${ARGS[@]}"
        echo "EXIT_r${round}_${side}=$?"
    done
done
echo "CROSSOVER_DONE $(date '+%F %T')"
