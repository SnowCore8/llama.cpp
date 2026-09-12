#!/usr/bin/env bash
# S1 同协议归因 ABBA：e3d（§1.1 基线快照）-> deltanet-pipe（delta-net 构建）-> cublas-off（当前生产库）
# 目的：判定 S1 相对 §1.1 的 +2~3% 是代码变更带来的，还是会话漂移；三方同协议同会话直接对比
# 协议与 cln-verify.sh 阶段 A 完全一致：-b 4096 -ub 4096 -p 4096 -d 0,16384,32768 -n 0 -r 2
set -u
cd /root/llama.cpp
M=/root/llama_gguf/Qwen3.5-9B-Q4_K_M/Qwen3.5-9B-Q4_K_M.gguf
B=./build/bin/llama-bench
COMMON="-m $M -dev CUDA1 -fa 1 -ctk q8_0 -ctv q8_0 -b 4096 -ub 4096"
LIBDIR=/root/llama.cpp/v100-prefill/fa-libs

echo "== 9b-attr-ab 开始 $(date '+%F %T') =="
echo "git_head=$(git rev-parse --short HEAD)  worktree_lines=$(git status --short | wc -l)"
echo "GPU占用（应为空）:"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true

# 每侧打印 sha256 与 ldd 实际解析路径，作为换库生效证据
for side in e3d deltanet-pipe cublas-off; do
    d=$LIBDIR/$side
    echo "lib_${side}_sha256=$(sha256sum "$(readlink -f $d/libggml-cuda.so.0)" | cut -d' ' -f1)"
    echo "resolved_${side}=$(LD_LIBRARY_PATH=$d ldd $B | grep libggml-cuda | awk '{print $3}')"
done

for round in 1 2; do
    if [ "$round" -eq 1 ]; then order="e3d deltanet-pipe cublas-off"; else order="cublas-off deltanet-pipe e3d"; fi
    for side in $order; do
        echo "===== r$round $side pp4096 d0/16384/32768 ====="
        LD_LIBRARY_PATH=$LIBDIR/$side CUDA_DEVICE_ORDER=PCI_BUS_ID $B $COMMON -p 4096 -d 0,16384,32768 -n 0 -r 2 2>&1 | grep -E "^\| qwen35|^build:"
        echo "EXIT_r${round}_${side}=${PIPESTATUS[0]}"
    done
done
echo "ATTRAB_DONE $(date '+%F %T')"
