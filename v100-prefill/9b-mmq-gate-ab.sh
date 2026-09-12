#!/usr/bin/env bash
# V100 MMQ 门槛改动的性能验证（2026-09-12）：三侧 ABBA + 边界断言。
# 侧：cublas-off（旧门槛 <64 走 MMQ） / mmq-gate（新门槛 <176 走 MMQ） / mmq-exp（全强制 MMQ，锚）
# 断言表：
#   pp48/64/128/175：新 == 强制MMQ（窗口内换到 MMQ），旧侧在 64..175 落后
#   pp176/192/512/4096：新 == 旧（>=176 仍走 cuBLAS，无回归）
# 尺寸刻意含 175/176 边界两侧；r3 + 两轮（正向/反向）抵消漂移。
set -u
cd /root/llama.cpp || exit 1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

LOG=/root/llama.cpp/v100-prefill/fa-logs/9b-mmq-gate-ab-20260912.log
exec > >(tee "$LOG") 2>&1

M=/root/llama_gguf/Qwen3.5-9B-Q4_K_M/Qwen3.5-9B-Q4_K_M.gguf
B=./build/bin/llama-bench
LIBS=/root/llama.cpp/v100-prefill/fa-libs
ARGS=(-m "$M" -dev CUDA1 -fa 1 -ctk q8_0 -ctv q8_0 -b 8192 -ub 4096 -p 48,64,128,175,176,192,512,4096 -d 0 -n 0 -r 3)

echo "== 9b-mmq-gate-ab 开始 $(date '+%F %T') =="
echo "git_head=$(git rev-parse --short HEAD)  worktree_lines=$(git status --short | wc -l)"
echo "GPU占用（应为空）: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr '\n' ' ')"
nvidia-smi --query-gpu=index,name,clocks.sm,temperature.gpu --format=csv,noheader
for side in cublas-off mmq-gate mmq-exp; do
    echo "lib_${side}_sha256=$(sha256sum "$LIBS/$side/libggml-cuda.so.0.23.0" | awk '{print $1}')"
    echo "resolved_${side}=$(LD_LIBRARY_PATH="$LIBS/$side" ldd $B | awk '/libggml-cuda.so.0/ {print $3}')"
done

for round in 1 2; do
    if [ "$round" = 1 ]; then order="cublas-off mmq-gate mmq-exp"; else order="mmq-exp mmq-gate cublas-off"; fi
    for side in $order; do
        echo ""
        echo "===== r${round} ${side} pp48..4096 ====="
        LD_LIBRARY_PATH="$LIBS/$side" $B "${ARGS[@]}"
        echo "EXIT_r${round}_${side}=$?"
    done
done
echo "GPU收尾: $(nvidia-smi --query-gpu=index,name,clocks.sm,temperature.gpu --format=csv,noheader | tr '\n' '|')"
echo "MMQGATE_AB_DONE $(date '+%F %T')"
