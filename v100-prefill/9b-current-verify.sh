#!/usr/bin/env bash
# 当前构建复核（2026-09-12）：把账本头条数字重新绑定到"当前代码"。
# 当前代码 = HEAD 3148fd037（FA 调优 19a667d32 + delta-net 流水化 3148fd037）+ FORCE_CUBLAS=OFF
# 当前内核库 = build/bin/libggml-cuda.so.0.23.0（sha eb6041f8…，快照 fa-libs/cublas-off）
# 目的：账本中所有头条数字均为机器实测且反映当前构建；同时补正确性门禁与缺失的 stock 侧证据。
# 预期对照：S1 ↔ fa-logs/cln-verify.log 的 e3d 清场值（3340.08 / 2782.96 / 2367.54）
#           S2 ↔ fa-logs/9b-cublas-off-ab-20260912.log（d0 3433.3/3422.1、d8192 3102.2/3090.1）
#           S3 = 新基线（小批分流剖面：8→MMVQ，9..63→MMQ，>=64→cuBLAS）
#           S4 ↔ 769.87 t/s（5.320 s/批，fa-logs/cln-256k-e3d.log）
set -u
cd /root/llama.cpp || exit 1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

LOG=/root/llama.cpp/v100-prefill/fa-logs/9b-current-verify-20260912.log
exec > >(tee "$LOG") 2>&1

M=/root/llama_gguf/Qwen3.5-9B-Q4_K_M/Qwen3.5-9B-Q4_K_M.gguf
B=./build/bin/llama-bench
TBO=./build/bin/test-backend-ops
COMMON=(-m "$M" -dev CUDA1 -fa 1 -ctk q8_0 -ctv q8_0 -n 0)

echo "== 9b-current-verify 开始 $(date '+%F %T') =="
echo "git_head=$(git rev-parse --short HEAD)  worktree_lines=$(git status --short | wc -l)"
echo "build_lib_sha256=$(sha256sum build/bin/libggml-cuda.so.0.23.0 | awk '{print $1}')"
echo "resolved_lib=$(ldd $B | awk '/libggml-cuda.so.0/ {print $3}')"
echo "GPU占用（应为空）: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr '\n' ' ')"
nvidia-smi --query-gpu=index,name,clocks.sm,temperature.gpu,power.draw --format=csv,noheader

echo; echo "===== S1 协议§1：p4096 d0/16384/32768，r2（对照 cln-verify e3d 清场值）====="
$B "${COMMON[@]}" -b 4096 -ub 4096 -p 4096 -d 0,16384,32768 -r 2
echo "EXIT_S1=$?"

echo; echo "===== S2 协议§14/§16：p8192 d0/8192/16384，r2（对照 cublas-off AB 值）====="
$B "${COMMON[@]}" -b 8192 -ub 4096 -p 8192 -d 0,8192,16384 -r 2
echo "EXIT_S2=$?"

echo; echo "===== S3 小批分流剖面：p8,9,32,63,64,128 d0，r5（新基线）====="
$B "${COMMON[@]}" -b 8192 -ub 4096 -p 8,9,32,63,64,128 -d 0 -r 5
echo "EXIT_S3=$?"

echo; echo "===== S4 256k 实测点：p4096 d262144，r2（对照 769.87 t/s）$(date '+%T') ====="
$B "${COMMON[@]}" -b 4096 -ub 4096 -p 4096 -d 262144 -r 2
echo "EXIT_S4=$?"

echo; echo "===== S5 正确性门禁（当前构建）：FLASH_ATTN_EXT hsk=256 ====="
$TBO test -b CUDA1 -o FLASH_ATTN_EXT -p "hsk=256" > /root/llama.cpp/v100-prefill/fa-logs/tbo-current-256.log 2>&1
echo "EXIT_S5=$?"
grep -E "tests passed|Backend CUDA1" /root/llama.cpp/v100-prefill/fa-logs/tbo-current-256.log

echo; echo "===== S6 补档：stock 侧正确性门禁（E6 缺失证据重建）====="
echo "s6_resolved_lib=$(LD_LIBRARY_PATH=/root/llama.cpp/v100-prefill/fa-libs/stock ldd $TBO | awk '/libggml-cuda.so.0/ {print $3}')"
LD_LIBRARY_PATH=/root/llama.cpp/v100-prefill/fa-libs/stock $TBO test -b CUDA1 -o FLASH_ATTN_EXT -p "hsk=256" > /root/llama.cpp/v100-prefill/fa-logs/tbo-stock-256.log 2>&1
echo "EXIT_S6=$?"
grep -E "tests passed|Backend CUDA1" /root/llama.cpp/v100-prefill/fa-logs/tbo-stock-256.log

echo; echo "GPU收尾: $(nvidia-smi --query-gpu=index,name,clocks.sm,temperature.gpu,power.draw --format=csv,noheader | tr '\n' '|')"
echo "CURVERIFY_DONE $(date '+%F %T')"
