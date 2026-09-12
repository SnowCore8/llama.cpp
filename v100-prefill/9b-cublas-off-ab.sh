#!/usr/bin/env bash
# 配置变更验证 A/B：deltanet-pipe（旧配置 FORCE_CUBLAS=ON）vs cublas-off（改为 OFF 后的构建）。
# 预期：pp32 从 ~356 升到 ~858（ne11=32 改走 MMQ）；pp8192 d0/d8192 不变（>=64 仍走 cuBLAS）。
# ABBA 2 轮；每侧打印 ldd 解析结果作换库生效证据。
set -u

cd /root/llama.cpp || exit 1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

M9=/root/llama_gguf/Qwen3.5-9B-Q4_K_M/Qwen3.5-9B-Q4_K_M.gguf
B=./build/bin/llama-bench
LIBS=/root/llama.cpp/v100-prefill/fa-libs
BASE=(-dev CUDA1 -fa 1 -ctk q8_0 -ctv q8_0 -b 8192 -ub 4096 -p 8192 -d 0,8192 -n 0 -r 2)
SMALL=(-dev CUDA1 -fa 1 -ctk q8_0 -ctv q8_0 -b 8192 -ub 4096 -p 32 -d 0 -n 0 -r 5)

echo "== fa-ab-cublas-off (9B only) $(date +%F)"
echo "git_head=$(git rev-parse --short HEAD)"
echo "worktree_diff=$(git diff --stat | tail -1)"
for side in deltanet-pipe cublas-off; do
    echo "lib_${side}_sha256=$(sha256sum "$LIBS/$side/libggml-cuda.so.0.23.0" | awk '{print $1}')"
    echo "resolved_${side}=$(LD_LIBRARY_PATH="$LIBS/$side" ldd $B | awk '/libggml-cuda.so.0 =>/ {print $3}')"
done

# ABBA：r1 = 旧,新；r2 = 新,旧
for round in 1 2; do
    if [ "$round" = 1 ]; then order="deltanet-pipe cublas-off"; else order="cublas-off deltanet-pipe"; fi
    for side in $order; do
        echo ""
        echo "===== SMALL r${round} ${side} pp32 ne11=32 ====="
        LD_LIBRARY_PATH="$LIBS/$side" $B -m "$M9" "${SMALL[@]}"
        echo "EXIT_small_r${round}_${side}=$?"
        echo ""
        echo "===== WALL r${round} ${side} 9B ub4096 d0/8192 ====="
        LD_LIBRARY_PATH="$LIBS/$side" $B -m "$M9" "${BASE[@]}"
        echo "EXIT_r${round}_${side}=$?"
    done
done
echo ABCUBLASOFF_9B_DONE
