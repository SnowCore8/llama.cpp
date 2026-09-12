#!/usr/bin/env bash
# V100 MMQ 门槛验证（2026-09-12）：正确性门禁 + 端到端一致性 + 换库对照。
# 改动：ggml_cuda_should_use_mmq() 的 NVIDIA 分支新增 V100 专档 volta_mma_available -> ne11 < 176。
# 影响面（代码可证）：仅 sm_70 + 量化类型 + ne11 属于 [64,176) 时分流改变；其余分支判定完全不变。
# tbo 窗口覆盖：MUL_MAT_ID 有 bs=64/128/129（落在窗口内，改动后走 MMQ）；dense MUL_MAT 全部 <64 或 >=512（窗口外）。
# 端到端：llama-perplexity -b/-ub 128（窗口内，双库对比）与 512（窗口外，双库走同一路径，应完全一致）。
set -u
cd /root/llama.cpp || exit 1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

LOG=/root/llama.cpp/v100-prefill/fa-logs/9b-mmq-gate-verify-20260912.log
exec > >(tee "$LOG") 2>&1

M=/root/llama_gguf/Qwen3.5-9B-Q4_K_M/Qwen3.5-9B-Q4_K_M.gguf
TBO=./build/bin/test-backend-ops
PPL=./build/bin/llama-perplexity
GATE=/root/llama.cpp/v100-prefill/fa-libs/mmq-gate
BASE=/root/llama.cpp/v100-prefill/fa-libs/cublas-off
TXT=/root/llama.cpp/README.md
FLOGS=/root/llama.cpp/v100-prefill/fa-logs

echo "== 9b-mmq-gate-verify 开始 $(date '+%F %T') =="
echo "git_head=$(git rev-parse --short HEAD) worktree_lines=$(git status --short | wc -l)"
echo "gate_lib_sha256=$(sha256sum $GATE/libggml-cuda.so.0.23.0 | awk '{print $1}')"
echo "base_lib_sha256=$(sha256sum $BASE/libggml-cuda.so.0.23.0 | awk '{print $1}')"
echo "GPU占用（应为空）: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr '\n' ' ')"
nvidia-smi --query-gpu=index,name,clocks.sm,temperature.gpu --format=csv,noheader

tbo_run () { # $1=标签 $2=库路径 $3=op 过滤 $4=日志名
  echo; echo "===== tbo[$1] -o $3 ====="
  echo "resolved_lib=$(LD_LIBRARY_PATH=$2 ldd $TBO | awk '/libggml-cuda.so.0/ {print $3}')"
  local t0=$SECONDS
  LD_LIBRARY_PATH=$2 $TBO test -b CUDA1 -o "$3" > "$FLOGS/$4" 2>&1
  local rc=$?
  echo "EXIT_rc=$rc 用时=$((SECONDS-t0))s"
  grep -E "backends passed" "$FLOGS/$4" | tail -1
}

ppl_run () { # $1=标签 $2=库路径 $3=batch 大小 $4=日志名
  echo; echo "===== ppl[$1] -b $3 -ub $3 ====="
  echo "resolved_lib=$(LD_LIBRARY_PATH=$2 ldd $PPL | awk '/libggml-cuda.so.0/ {print $3}')"
  local t0=$SECONDS
  LD_LIBRARY_PATH=$2 $PPL -m "$M" -dev CUDA1 -fa 1 -ctk q8_0 -ctv q8_0 -f "$TXT" -b $3 -ub $3 > "$FLOGS/$4" 2>&1
  echo "EXIT_rc=$? 用时=$((SECONDS-t0))s"
  grep -E "Final estimate" "$FLOGS/$4"
}

# S1/S2：窗口内覆盖（MUL_MAT_ID bs=64/128/129，量化类型）+ 换库对照
tbo_run gate $GATE MUL_MAT_ID "tbo-mmqgate-mmid.log"
tbo_run base $BASE MUL_MAT_ID "tbo-base-mmid.log"

# S3：dense MUL_MAT 全家族（窗口外，双库判定一致；任一失败即回归）
tbo_run gate $GATE MUL_MAT "tbo-mmqgate-mm.log"

# S4-S7：端到端 2x2（窗口内 b128 双库对比；窗口外 b512 作方法学零假设对照）
ppl_run gate128 $GATE 128 "ppl-mmqgate-b128.log"
ppl_run base128 $BASE 128 "ppl-base-b128.log"
ppl_run gate512 $GATE 512 "ppl-mmqgate-b512.log"
ppl_run base512 $BASE 512 "ppl-base-b512.log"

echo; echo "GPU收尾: $(nvidia-smi --query-gpu=index,name,clocks.sm,temperature.gpu --format=csv,noheader | tr '\n' '|')"
echo "MMQGATE_VERIFY_DONE $(date '+%F %T')"
