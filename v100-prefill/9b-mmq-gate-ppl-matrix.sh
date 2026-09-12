#!/usr/bin/env bash
# MMQ 门槛改动的数值差异归因（2026-09-12）：逐 batch 尺寸双库 ppl 矩阵。
# 逻辑：若某尺寸两库判定路径相同（都走 MMQ 或都走 cuBLAS），ppl 应完全一致；
#       只有落在 [64,176) 且分属不同实现时才出现差异 -> 差异可完全归因于 MMQ vs cuBLAS。
# 尺寸：48/64（旧新都是 MMQ，对照）/ 128/175（新 MMQ、旧 cuBLAS）/ 176（旧新都是 cuBLAS，对照）
set -u
cd /root/llama.cpp || exit 1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

LOG=/root/llama.cpp/v100-prefill/fa-logs/9b-mmq-gate-ppl-matrix-20260912.log
exec > >(tee "$LOG") 2>&1

M=/root/llama_gguf/Qwen3.5-9B-Q4_K_M/Qwen3.5-9B-Q4_K_M.gguf
PPL=./build/bin/llama-perplexity
LIBS=/root/llama.cpp/v100-prefill/fa-libs
TXT=/root/llama.cpp/README.md

echo "== 9b-mmq-gate-ppl-matrix 开始 $(date '+%F %T') =="
for bs in 48 64 175 176; do
    for side in mmq-gate cublas-off; do
        echo "--- bs=$bs side=$side"
        LD_LIBRARY_PATH="$LIBS/$side" $PPL -m "$M" -dev CUDA1 -fa 1 -ctk q8_0 -ctv q8_0 -f "$TXT" -b $bs -ub $bs 2>&1 | grep -E "Final estimate"
    done
done
echo "PPL_MATRIX_DONE $(date '+%F %T')"
