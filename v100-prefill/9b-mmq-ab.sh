#!/usr/bin/env bash
# 单一变量 A/B：cuBLAS 路径（deltanet-pipe = 3148fd037 构建）vs 强制 dp4a MMQ 路径（mmq-exp）。
# mmq-exp 用临时等价补丁构建（禁用 FORCE_CUBLAS 提前返回 + NVIDIA 分支强制 return true，
# 等价于 -DGGML_CUDA_FORCE_CUBLAS=OFF -DGGML_CUDA_FORCE_MMQ=ON）；补丁已回退，工作区干净。
# 9B only；协议：d0/8k/16k（-p 8192 -ub 4096 -r 2）+ 小批 pp32（ne11=32，
# 这是 FORCE_CUBLAS 在生产里唯一影响的窗口：9..63 落在 MMQ/cuBLAS 分界上）。
# ABBA 次序抵消漂移；每侧先打印 ldd 解析结果——验证 LD_LIBRARY_PATH 换库真的生效。
set -u

cd /root/llama.cpp || exit 1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

M9=/root/llama_gguf/Qwen3.5-9B-Q4_K_M/Qwen3.5-9B-Q4_K_M.gguf
B=./build/bin/llama-bench
LIBS=/root/llama.cpp/v100-prefill/fa-libs
BASE=(-dev CUDA1 -fa 1 -ctk q8_0 -ctv q8_0 -b 8192 -ub 4096 -p 8192 -d 0,8192,16384 -n 0 -r 2)
SMALL=(-dev CUDA1 -fa 1 -ctk q8_0 -ctv q8_0 -b 8192 -ub 4096 -p 32 -d 0 -n 0 -r 5)

echo "== fa-ab-mmq-exp (9B only) $(date +%F)"
echo "git_head=$(git rev-parse --short HEAD)"
echo "worktree_diff=$(git diff --stat | tail -1)"
for side in deltanet-pipe mmq-exp; do
    echo "lib_${side}_sha256=$(sha256sum "$LIBS/$side/libggml-cuda.so.0.23.0" | awk '{print $1}')"
    echo "resolved_${side}=$(LD_LIBRARY_PATH="$LIBS/$side" ldd $B | awk '/libggml-cuda.so.0 =>/ {print $3}')"
done

# ABBA：r1 = cuBLAS,MMQ；r2 = MMQ,cuBLAS
for round in 1 2; do
    if [ "$round" = 1 ]; then order="deltanet-pipe mmq-exp"; else order="mmq-exp deltanet-pipe"; fi
    for side in $order; do
        echo ""
        echo "===== WALL r${round} ${side} 9B ub4096 ====="
        LD_LIBRARY_PATH="$LIBS/$side" $B -m "$M9" "${BASE[@]}"
        echo "EXIT_r${round}_${side}=$?"
        echo ""
        echo "===== SMALL r${round} ${side} pp32 ne11=32 ====="
        LD_LIBRARY_PATH="$LIBS/$side" $B -m "$M9" "${SMALL[@]}"
        echo "EXIT_small_r${round}_${side}=$?"
    done
done
echo ABMMQ_9B_DONE
