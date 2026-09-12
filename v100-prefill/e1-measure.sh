#!/bin/bash
# E1 实验测量脚本（Q_in_reg=false 配置）
# 流程：校验编译结果 -> 记录产物指纹 -> 墙钟曲线 -> ncu 指令管线 -> ncu 停顿采样
# 所有产物写入 /tmp，脚本本身不修改仓库源码
set -u
cd /root/llama.cpp || exit 1

MODEL=/root/llama_gguf/Qwen3.5-9B-Q4_K_M/Qwen3.5-9B-Q4_K_M.gguf
BENCH=./build/bin/llama-bench
LB=/tmp/lb-e1.log

# 与 stock(A 侧) 完全一致的 ncu 指标集，保证可比
MET_PIPES=smsp__inst_executed.sum,sm__inst_executed_pipe_tensor.sum,smsp__inst_executed_pipe_lsu.sum,smsp__inst_executed_pipe_fma.sum,smsp__inst_executed_pipe_fp16.sum,smsp__inst_executed_pipe_alu.sum,smsp__inst_executed_pipe_xu.sum,smsp__inst_executed_pipe_adu.sum,smsp__sass_inst_executed_op_shared_ld.sum,smsp__sass_inst_executed_op_shared_st.sum,smsp__sass_inst_executed_op_global_ld.sum,smsp__sass_inst_executed_op_global_st.sum,smsp__sass_inst_executed_op_local_ld.sum,smsp__sass_inst_executed_op_local_st.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum,dram__bytes_read.sum,gpu__time_duration.sum,launch__registers_per_thread,launch__shared_mem_per_block_dynamic,launch__waves_per_multiprocessor
MET_STALLS=smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,smsp__average_warps_issue_stalled_wait_per_issue_active.ratio,smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio,smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio,smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio,smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio,smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio,smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio,smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio,smsp__issue_active.avg.pct_of_peak_sustained_active,sm__warps_active.avg.pct_of_peak_sustained_active

# 0) 等待 E1 编译结束（最多 40 分钟，编译在另一个后台任务里进行）
for i in $(seq 1 240); do
    grep -q "BUILD_EXIT" /tmp/build-e1.log 2>/dev/null && break
    sleep 10
done

# 1) 编译结果必须成功，否则直接退出
if ! grep -q "BUILD_EXIT=0" /tmp/build-e1.log 2>/dev/null; then
    echo "STEP1_BUILD_NOT_OK"
    grep "error" /tmp/build-e1.log | tail -20
    exit 1
fi
echo "STEP1_BUILD_OK"

# 2) 产物指纹：带标签复制启动器与内核库，便于事后核对实验身份
cp -f "$BENCH" /tmp/lb-e1-qinreg0
cp -f build/bin/libggml-cuda.so.0.23.0 /tmp/so-e1-qinreg0.so
sha256sum /tmp/lb-e1-qinreg0 /tmp/so-e1-qinreg0.so | tee /tmp/e1-fingerprint.txt
grep -c "flash_attn_ext_f16" /tmp/build-e1.log >/dev/null

# 3) 墙钟曲线：与 A 侧（clean stock）同参数、同深度点
echo "STEP3_WALL_START"
CUDA_DEVICE_ORDER=PCI_BUS_ID $BENCH -m $MODEL -dev CUDA1 -fa 1 -ctk q8_0 -ctv q8_0 \
    -b 4096 -ub 4096 -p 4096 -d 0,16384,32768 -n 0 -r 2 > $LB 2>&1
echo "WALL_EXIT=$?"

# 4) ncu 指令管线采样（深度 16384，跳过前 39 次启动取第 40 次）
echo "STEP4_NCU_PIPES_START"
CUDA_DEVICE_ORDER=PCI_BUS_ID ncu -k "regex:flash_attn_ext_f16" --launch-skip 39 -c 1 --csv \
    --metrics $MET_PIPES \
    $BENCH -m $MODEL -dev CUDA1 -fa 1 -ctk q8_0 -ctv q8_0 -b 4096 -ub 4096 -p 4096 -d 16384 -n 0 -r 1 \
    > /tmp/ncu-e1-pipes.csv 2>&1
echo "NCU_P_EXIT=$?"

# 5) ncu 停顿采样
echo "STEP5_NCU_STALLS_START"
CUDA_DEVICE_ORDER=PCI_BUS_ID ncu -k "regex:flash_attn_ext_f16" --launch-skip 39 -c 1 --csv \
    --metrics $MET_STALLS \
    $BENCH -m $MODEL -dev CUDA1 -fa 1 -ctk q8_0 -ctv q8_0 -b 4096 -ub 4096 -p 4096 -d 16384 -n 0 -r 1 \
    > /tmp/ncu-e1-stalls.csv 2>&1
echo "NCU_S_EXIT=$?"

echo "STEP6_DONE"
