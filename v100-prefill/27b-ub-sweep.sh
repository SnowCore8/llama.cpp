#!/usr/bin/env bash
# Qwen3.8-27B-Q4_K_M 预填充 ubatch 扫描（V100 / CUDA1，清场无服务）
# 协议对齐 9B 的 lb-ub-sweep（见 fa-logs/lb-ub-sweep.log）：-b 8192 固定、变 -ub、
# pp8192、KV q8_0、fa=1、-r 用默认值（5）。
set -u
cd /root/llama.cpp || exit 1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
M=/root/llama_gguf/Qwen3.8-27B-Q4_K_M/Qwen3.8-27B-Q4_K_M.gguf
echo "27b-ub-sweep 2026-09-12 clean-field (no service)"
echo "build=$(git rev-parse --short HEAD)  so_sha256=$(sha256sum build/bin/libggml-cuda.so.0.23.0 | cut -d' ' -f1)"
echo "model=$M"
echo "proto: -b 8192 -ub <v> -p 8192 -n 0 (r default) -fa 1 -ctk/v q8_0 -dev CUDA1"
for ub in 512 1024 2048 4096 8192; do
    echo ""
    echo "===== ub=$ub ====="
    ./build/bin/llama-bench -m "$M" -dev CUDA1 -fa 1 -ctk q8_0 -ctv q8_0 -b 8192 -ub "$ub" -p 8192 -n 0
    echo "EXIT_ub${ub}=$?"
done
echo SWEEP_DONE
