#!/usr/bin/env bash
# 单一变量 A/B：在 Volta(DKQ=256) 上强制启用 stream-K（KV 扫描拆分 + dstk_fixup 跨块 reduce）。
# 单模型 = 9B（与 27B 同架构，按用户指示不重复测）；对照 = 已提交的 e3d 快照。
# 协议：清场、8k/16k 深度、ABBA 次序（e3d -> streamk -> streamk -> e3d，抵消线性漂移）。
set -u

cd /root/llama.cpp || exit 1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

M9=/root/llama_gguf/Qwen3.5-9B-Q4_K_M/Qwen3.5-9B-Q4_K_M.gguf
B=./build/bin/llama-bench
LIBS=/root/llama.cpp/v100-prefill/fa-libs
LOGS=/root/llama.cpp/v100-prefill/fa-logs
BASE=(-dev CUDA1 -fa 1 -ctk q8_0 -ctv q8_0 -b 8192 -ub 4096 -p 8192 -d 8192,16384 -n 0 -r 2)

echo "== fa-ab-streamk (9B only) 2026-09-12"
echo "git_head=$(git rev-parse --short HEAD)"
echo "worktree_diff=$(git diff --stat | tail -1)"
for side in e3d streamk; do
    echo "lib_${side}_sha256=$(sha256sum "$LIBS/$side/libggml-cuda.so.0.23.0" | awk '{print $1}')"
done

echo ""
echo "===== CORRECTNESS streamk hsk=256 ====="
TBO_LOG="$LOGS/tbo-streamk-hsk256-9b.log"
LD_LIBRARY_PATH="$LIBS/streamk" ./build/bin/test-backend-ops test -b CUDA1 -o FLASH_ATTN_EXT -p "hsk=256" > "$TBO_LOG" 2>&1
echo "EXIT_tbo_streamk=$?"
echo "OK=$(grep -c OK "$TBO_LOG") FAIL=$(grep -c FAIL "$TBO_LOG")"

# ABBA：r1 = e3d,streamk；r2 = streamk,e3d
for round in 1 2; do
    if [ "$round" = 1 ]; then order="e3d streamk"; else order="streamk e3d"; fi
    for side in $order; do
        echo ""
        echo "===== WALL r${round} ${side} 9B ub4096 ====="
        LD_LIBRARY_PATH="$LIBS/$side" $B -m "$M9" "${BASE[@]}"
        echo "EXIT_r${round}_${side}=$?"
    done
done
echo ABSTREAMK_9B_DONE
