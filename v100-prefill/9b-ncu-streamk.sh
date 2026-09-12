#!/usr/bin/env bash
# 单一变量内核级复核：Volta 强制 stream-K（9B、d8192、沿用既有 ncu 协议 --launch-skip 39 -c 1）。
# 目的：①确认 stream-K 真生效（grid size 2048 -> 144）；②内核时长/停顿/DRAM/L2 命中变化；
#       ③streamk 侧额外观测 fixup 归约内核的开销。
set -u

cd /root/llama.cpp || exit 1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

M9=/root/llama_gguf/Qwen3.5-9B-Q4_K_M/Qwen3.5-9B-Q4_K_M.gguf
B=./build/bin/llama-bench
LIBS=/root/llama.cpp/v100-prefill/fa-libs

PIPES="gpu__time_duration.sum,smsp__inst_executed.sum,sm__inst_executed_pipe_tensor.sum,smsp__inst_executed_pipe_fp16.sum,smsp__inst_executed_pipe_lsu.sum,smsp__inst_executed_pipe_fma.sum,smsp__inst_executed_pipe_alu.sum,smsp__inst_executed_pipe_xu.sum,smsp__issue_active.avg.pct_of_peak_sustained_active,sm__warps_active.avg.pct_of_peak_sustained_active,launch__registers_per_thread,launch__shared_mem_per_block_dynamic"
STALLS="smsp__average_warps_issue_stalled_wait_per_issue_active.ratio,smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio,smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio,smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio,smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio,smsp__sass_inst_executed_op_shared_ld.sum,smsp__sass_inst_executed_op_local_ld.sum,smsp__sass_inst_executed_op_local_st.sum"
EXTRA="launch__grid_size,launch__waves_per_multiprocessor,dram__bytes.sum,lts__t_sector_hit_rate.pct"

echo "== fa-ncu-streamk 9B (single variable) 2026-09-12"
for side in e3d streamk; do
    echo ""
    echo "===== NCU ${side} 9B d8192 (flash_attn_ext_f16) ====="
    LD_LIBRARY_PATH="$LIBS/$side" ncu -k "regex:flash_attn_ext_f16" --launch-skip 39 -c 1 --csv --metrics "$PIPES,$STALLS,$EXTRA" \
        $B -m "$M9" -dev CUDA1 -fa 1 -ctk q8_0 -ctv q8_0 -b 8192 -ub 4096 -p 8192 -d 8192 -n 0 -r 1
    echo "EXIT_ncu_${side}=$?"
done

echo ""
echo "===== NCU streamk 9B d8192 (stream_k_fixup only) ====="
LD_LIBRARY_PATH="$LIBS/streamk" ncu -k "regex:stream_k_fixup" -c 1 --csv --metrics "gpu__time_duration.sum,launch__grid_size,smsp__inst_executed.sum,dram__bytes.sum" \
    $B -m "$M9" -dev CUDA1 -fa 1 -ctk q8_0 -ctv q8_0 -b 8192 -ub 4096 -p 8192 -d 8192 -n 0 -r 1
echo "EXIT_ncu_streamk_fixup=$?"
echo NCU_STREAMK_9B_DONE
