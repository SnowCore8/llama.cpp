#!/usr/bin/env bash
# 单一变量 A/B：gated_delta_net 软件流水（预取下一拍 k/q/v/g/beta，数学不变）。
# 单模型 = 9B；对照 = 已提交基线快照 e3d；协议：d0 / 8k / 16k 深度、ABBA 次序抵消漂移。
# d0 是纯 W 位点（KV 分支不参与），W 项改动的信号在这里最干净。
set -u

cd /root/llama.cpp || exit 1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

M9=/root/llama_gguf/Qwen3.5-9B-Q4_K_M/Qwen3.5-9B-Q4_K_M.gguf
B=./build/bin/llama-bench
LIBS=/root/llama.cpp/v100-prefill/fa-libs
BASE=(-dev CUDA1 -fa 1 -ctk q8_0 -ctv q8_0 -b 8192 -ub 4096 -p 8192 -d 0,8192,16384 -n 0 -r 2)

echo "== fa-ab-deltanet-pipe (9B only) 2026-09-12"
echo "git_head=$(git rev-parse --short HEAD)"
echo "worktree_diff=$(git diff --stat | tail -1)"
for side in e3d deltanet-pipe; do
    echo "lib_${side}_sha256=$(sha256sum "$LIBS/$side/libggml-cuda.so.0.23.0" | awk '{print $1}')"
done

# ABBA：r1 = e3d,pipe；r2 = pipe,e3d
for round in 1 2; do
    if [ "$round" = 1 ]; then order="e3d deltanet-pipe"; else order="deltanet-pipe e3d"; fi
    for side in $order; do
        echo ""
        echo "===== WALL r${round} ${side} 9B ub4096 ====="
        LD_LIBRARY_PATH="$LIBS/$side" $B -m "$M9" "${BASE[@]}"
        echo "EXIT_r${round}_${side}=$?"
    done
done
echo ABDELTANET_9B_DONE
