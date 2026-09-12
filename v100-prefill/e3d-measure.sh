#!/bin/bash
# E3-(d) 实验测量脚本：nbatch_fa=32 + nbatch_K2/V2=64 + combine=64 + Q_in_reg=false（目标 2 CTA/SM、8 warps、不溢出）
# 流程：等编译 -> 校验 -> 备份产物 -> 墙钟 -> ncu 管线 -> ncu 停顿 -> 与 E1 的 A/B/A/B 轮换
# 身份校验点：ncu 的 launch__shared_mem_per_block_dynamic 应为 43,776；warps_active 应为 12.5%（2 CTA/SM）
set -u
cd /root/llama.cpp || exit 1

MODEL=/root/llama_gguf/Qwen3.5-9B-Q4_K_M/Qwen3.5-9B-Q4_K_M.gguf
BENCH=./build/bin/llama-bench

MET_PIPES=smsp__inst_executed.sum,sm__inst_executed_pipe_tensor.sum,smsp__inst_executed_pipe_lsu.sum,smsp__inst_executed_pipe_fma.sum,smsp__inst_executed_pipe_fp16.sum,smsp__inst_executed_pipe_alu.sum,smsp__inst_executed_pipe_xu.sum,smsp__inst_executed_pipe_adu.sum,smsp__sass_inst_executed_op_shared_ld.sum,smsp__sass_inst_executed_op_shared_st.sum,smsp__sass_inst_executed_op_global_ld.sum,smsp__sass_inst_executed_op_global_st.sum,smsp__sass_inst_executed_op_local_ld.sum,smsp__sass_inst_executed_op_local_st.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum,dram__bytes_read.sum,gpu__time_duration.sum,launch__registers_per_thread,launch__shared_mem_per_block_dynamic,launch__waves_per_multiprocessor
MET_STALLS=smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,smsp__average_warps_issue_stalled_wait_per_issue_active.ratio,smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio,smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio,smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio,smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio,smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio,smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio,smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio,smsp__issue_active.avg.pct_of_peak_sustained_active,sm__warps_active.avg.pct_of_peak_sustained_active

# 0) 等待编译结束（最多 45 分钟）
for i in $(seq 1 270); do
    grep -q "BUILD_EXIT" /tmp/build-e3d.log 2>/dev/null && break
    sleep 10
done

if ! grep -q "BUILD_EXIT=0" /tmp/build-e3d.log 2>/dev/null; then
    echo "STEP1_BUILD_NOT_OK"
    grep -i "error" /tmp/build-e3d.log | tail -20
    exit 1
fi
echo "STEP1_BUILD_OK"

# 1) 产物指纹 + 放进 A/B 目录
cp -f "$BENCH" /tmp/lb-e3d
cp -f build/bin/libggml-cuda.so.0.23.0 /tmp/so-e3d.so
mkdir -p /tmp/falib-e3d
cp -f /tmp/so-e3d.so /tmp/falib-e3d/libggml-cuda.so.0.23.0
ln -sf libggml-cuda.so.0.23.0 /tmp/falib-e3d/libggml-cuda.so.0
ln -sf libggml-cuda.so.0.23.0 /tmp/falib-e3d/libggml-cuda.so
sha256sum /tmp/lb-e3d /tmp/so-e3d.so | tee /tmp/e3d-fingerprint.txt

# 2) 墙钟曲线
echo "STEP2_WALL_START"
CUDA_DEVICE_ORDER=PCI_BUS_ID $BENCH -m $MODEL -dev CUDA1 -fa 1 -ctk q8_0 -ctv q8_0 \
    -b 4096 -ub 4096 -p 4096 -d 0,16384,32768 -n 0 -r 2 > /tmp/lb-e3d.log 2>&1
echo "WALL_EXIT=$?"

# 3) ncu 指令管线
echo "STEP3_NCU_PIPES_START"
CUDA_DEVICE_ORDER=PCI_BUS_ID ncu -k "regex:flash_attn_ext_f16" --launch-skip 39 -c 1 --csv \
    --metrics $MET_PIPES \
    $BENCH -m $MODEL -dev CUDA1 -fa 1 -ctk q8_0 -ctv q8_0 -b 4096 -ub 4096 -p 4096 -d 16384 -n 0 -r 1 \
    > /tmp/ncu-e3d-pipes.csv 2>&1
echo "NCU_P_EXIT=$?"

# 4) ncu 停顿
echo "STEP4_NCU_STALLS_START"
CUDA_DEVICE_ORDER=PCI_BUS_ID ncu -k "regex:flash_attn_ext_f16" --launch-skip 39 -c 1 --csv \
    --metrics $MET_STALLS \
    $BENCH -m $MODEL -dev CUDA1 -fa 1 -ctk q8_0 -ctv q8_0 -b 4096 -ub 4096 -p 4096 -d 16384 -n 0 -r 1 \
    > /tmp/ncu-e3d-stalls.csv 2>&1
echo "NCU_S_EXIT=$?"

# 5) 与 E1（当前最佳）A/B/A/B 轮换：A=E1 库，B=E3-(d) 库
BASE_CMD="$BENCH -m $MODEL -dev CUDA1 -fa 1 -ctk q8_0 -ctv q8_0 -b 4096 -ub 4096 -p 4096 -d 0,16384,32768 -n 0 -r 2"
for round in 1 2; do
    for side in e1 e3d; do
        echo "=== round $round side $side ==="
        LD_LIBRARY_PATH=/tmp/falib-$side CUDA_DEVICE_ORDER=PCI_BUS_ID $BASE_CMD 2>&1 | grep -E "^\| qwen35|^build:"
    done
done
echo "STEP5_AB_DONE"
