#!/usr/bin/env bash
# V-A 变体（nthreads=256 / 1 CTA / K2=V2=128 / combine=64 / Q_in_reg=false）vs 既有 tuned（e3d）
# 清场、统一 8k/16k 协议（用户指示）。该配置行同时服务 27B（ncols1=32,ncols2=2）与 9B（16,4），两边都测。
# 前置: /root/llama.cpp/v100-prefill/fa-libs/{e3d,va}/libggml-cuda.so.0.23.0
set -u

cd /root/llama.cpp || exit 1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

M27=/root/llama_gguf/Qwen3.8-27B-Q4_K_M/Qwen3.8-27B-Q4_K_M.gguf
M9=/root/llama_gguf/Qwen3.5-9B-Q4_K_M/Qwen3.5-9B-Q4_K_M.gguf
B=./build/bin/llama-bench
LIBS=/root/llama.cpp/v100-prefill/fa-libs
LOGS=/root/llama.cpp/v100-prefill/fa-logs
BASE=(-dev CUDA1 -fa 1 -ctk q8_0 -ctv q8_0 -b 8192 -ub 4096 -p 8192 -d 8192,16384 -n 0 -r 2)

echo "== fa-ab-va 2026-09-12 (清场, 8k/16k)"
echo "git_head=$(git rev-parse --short HEAD)"
echo "worktree_diff=$(git diff --stat | tail -1)"
for side in e3d va; do
    if [ ! -e "$LIBS/$side/libggml-cuda.so.0.23.0" ]; then
        echo "FATAL: missing $LIBS/$side/libggml-cuda.so.0.23.0"
        exit 1
    fi
    echo "lib_${side}_sha256=$(sha256sum "$LIBS/$side/libggml-cuda.so.0.23.0" | awk '{print $1}')"
done

# 正确性门禁（E6 协议）：hsk=256 的 FLASH_ATTN_EXT 全集，期望 OK=143 / FAIL=0
for side in e3d va; do
    echo ""
    echo "===== CORRECTNESS ${side} hsk=256 ====="
    TBO_LOG="$LOGS/tbo-${side}-hsk256-20260912.log"
    LD_LIBRARY_PATH="$LIBS/$side" ./build/bin/test-backend-ops test -b CUDA1 -o FLASH_ATTN_EXT -p "hsk=256" > "$TBO_LOG" 2>&1
    echo "EXIT_tbo_${side}=$?"
    tail -3 "$TBO_LOG"
done

for round in 1 2; do
    for side in e3d va; do
        echo ""
        echo "===== WALL r${round} ${side} 27B ub4096 ====="
        LD_LIBRARY_PATH="$LIBS/$side" $B -m "$M27" "${BASE[@]}"
        echo "EXIT_27b_r${round}_${side}=$?"
    done
done

for round in 1 2; do
    for side in e3d va; do
        echo ""
        echo "===== WALL r${round} ${side} 9B ub4096 ====="
        LD_LIBRARY_PATH="$LIBS/$side" $B -m "$M9" "${BASE[@]}"
        echo "EXIT_9b_r${round}_${side}=$?"
    done
done

PIPES="dram__bytes_read.sum,gpu__time_duration.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,launch__registers_per_thread,launch__shared_mem_per_block_dynamic,launch__waves_per_multiprocessor,sm__inst_executed_pipe_tensor.sum,smsp__inst_executed_pipe_adu.sum,smsp__inst_executed_pipe_alu.sum,smsp__inst_executed_pipe_fma.sum,smsp__inst_executed_pipe_fp16.sum,smsp__inst_executed_pipe_lsu.sum,smsp__inst_executed_pipe_xu.sum,smsp__inst_executed.sum,smsp__sass_inst_executed_op_global_ld.sum,smsp__sass_inst_executed_op_global_st.sum,smsp__sass_inst_executed_op_local_ld.sum,smsp__sass_inst_executed_op_local_st.sum,smsp__sass_inst_executed_op_shared_ld.sum,smsp__sass_inst_executed_op_shared_st.sum"
STALLS="smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio,smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio,smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio,smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio,smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,smsp__average_warps_issue_stalled_wait_per_issue_active.ratio,smsp__issue_active.avg.pct_of_peak_sustained_active,sm__warps_active.avg.pct_of_peak_sustained_active"

for side in e3d va; do
    echo ""
    echo "===== NCU ${side} pipes+stalls 27B d8192 ====="
    LD_LIBRARY_PATH="$LIBS/$side" ncu -k "regex:flash_attn_ext_f16" --launch-skip 39 -c 1 --csv --metrics "$PIPES,$STALLS" \
        $B -m "$M27" -dev CUDA1 -fa 1 -ctk q8_0 -ctv q8_0 -b 8192 -ub 4096 -p 8192 -d 8192 -n 0 -r 1
    echo "EXIT_ncu_${side}=$?"
done

echo ABVA_DONE
