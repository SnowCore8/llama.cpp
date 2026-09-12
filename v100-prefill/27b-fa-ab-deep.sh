#!/usr/bin/env bash
# 27B FA 内核 A/B（stock vs tuned=e26604）+ ubatch 深度点 + strace（V100 / CUDA1，清场）
# 协议修订（2026-09-12 用户要求缩短）: 统一 8k/16k 上下文，最大深度 16384
# 前置: /root/llama.cpp/v100-prefill/fa-libs/{stock,e3d}/ 各含 libggml-cuda.so.0.23.0 及同名链接
# 墙钟 A/B: 2 轮 x {stock,e3d}, -p 8192 -d 8192,16384 -b 8192 -ub 4096 -n 0 -r 2
# 深度点: e3d, -ub 2048,8192 列表一次跑完（省一次模型加载）
# strace: e3d, -f -c 汇总一次 d8192 运行的系统调用开销
# ncu: pipes 双侧 + stalls 仅 e3d, -k regex:flash_attn_ext_f16 --launch-skip 39 -c 1 -d 8192 -r 1
set -u

cd /root/llama.cpp || exit 1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

M=/root/llama_gguf/Qwen3.8-27B-Q4_K_M/Qwen3.8-27B-Q4_K_M.gguf
B=./build/bin/llama-bench
LIBS=/root/llama.cpp/v100-prefill/fa-libs
LOGS=/root/llama.cpp/v100-prefill/fa-logs
COMMON=(-m "$M" -dev CUDA1 -fa 1 -ctk q8_0 -ctv q8_0)

echo "== 27b-fa-ab-deep v2 2026-09-12 (清场, 8k/16k 协议)"
echo "git_head=$(git rev-parse --short HEAD)"
echo "worktree_fa_diff=$(git diff --stat -- ggml/src/ggml-cuda/fattn-mma-f16.cuh | tail -1)"
for side in stock e3d; do
    if [ ! -e "$LIBS/$side/libggml-cuda.so.0.23.0" ]; then
        echo "FATAL: missing $LIBS/$side/libggml-cuda.so.0.23.0"
        exit 1
    fi
    echo "lib_${side}_sha256=$(sha256sum "$LIBS/$side/libggml-cuda.so.0.23.0" | awk '{print $1}')"
done

for round in 1 2; do
    for side in stock e3d; do
        echo ""
        echo "===== WALL r${round} ${side} ub4096 d8192,16384 ====="
        LD_LIBRARY_PATH="$LIBS/$side" $B "${COMMON[@]}" -b 8192 -ub 4096 -p 8192 -d 8192,16384 -n 0 -r 2
        echo "EXIT_wall_r${round}_${side}=$?"
    done
done

echo ""
echo "===== WALL e3d ub2048,8192 d8192,16384 ====="
LD_LIBRARY_PATH="$LIBS/e3d" $B "${COMMON[@]}" -b 8192 -ub 2048,8192 -p 8192 -d 8192,16384 -n 0 -r 2
echo "EXIT_wall_e3d_ub_list=$?"

STRACE_OUT=$LOGS/27b-strace-summary-20260912.txt
echo ""
echo "===== STRACE e3d ub4096 d8192 ====="
LD_LIBRARY_PATH="$LIBS/e3d" strace -f -c -o "$STRACE_OUT" \
    $B "${COMMON[@]}" -b 8192 -ub 4096 -p 8192 -d 8192 -n 0 -r 1
echo "EXIT_strace_e3d=$?"
echo "----- strace summary ($STRACE_OUT) -----"
cat "$STRACE_OUT"

PIPES="dram__bytes_read.sum,gpu__time_duration.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum,launch__registers_per_thread,launch__shared_mem_per_block_dynamic,launch__waves_per_multiprocessor,sm__inst_executed_pipe_tensor.sum,smsp__inst_executed_pipe_adu.sum,smsp__inst_executed_pipe_alu.sum,smsp__inst_executed_pipe_fma.sum,smsp__inst_executed_pipe_fp16.sum,smsp__inst_executed_pipe_lsu.sum,smsp__inst_executed_pipe_xu.sum,smsp__inst_executed.sum,smsp__sass_inst_executed_op_global_ld.sum,smsp__sass_inst_executed_op_global_st.sum,smsp__sass_inst_executed_op_local_ld.sum,smsp__sass_inst_executed_op_local_st.sum,smsp__sass_inst_executed_op_shared_ld.sum,smsp__sass_inst_executed_op_shared_st.sum"
STALLS="smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio,smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio,smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio,smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio,smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio,smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio,smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio,smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,smsp__average_warps_issue_stalled_wait_per_issue_active.ratio,smsp__issue_active.avg.pct_of_peak_sustained_active,sm__warps_active.avg.pct_of_peak_sustained_active"

for side in stock e3d; do
    echo ""
    echo "===== NCU ${side} pipes ====="
    LD_LIBRARY_PATH="$LIBS/$side" ncu -k "regex:flash_attn_ext_f16" --launch-skip 39 -c 1 --csv --metrics "$PIPES" \
        $B "${COMMON[@]}" -b 8192 -ub 4096 -p 8192 -d 8192 -n 0 -r 1
    echo "EXIT_ncu_${side}_pipes=$?"
done

echo ""
echo "===== NCU e3d stalls ====="
LD_LIBRARY_PATH="$LIBS/e3d" ncu -k "regex:flash_attn_ext_f16" --launch-skip 39 -c 1 --csv --metrics "$STALLS" \
    $B "${COMMON[@]}" -b 8192 -ub 4096 -p 8192 -d 8192 -n 0 -r 1
echo "EXIT_ncu_e3d_stalls=$?"

echo ABDEEP_DONE
