// V100 (sm_70) HMMA m8n8k4 峰值吞吐微基准 + 设备属性输出
// 用途：为 FA 内核的「占理论峰值百分比」提供本机实测口径
// 指令形态与仓库 ggml/src/ggml-cuda/mma.cuh (Volta 分支) 完全一致：D=8xf32, A=2x f16x2, B=2x f16x2
#include <cstdio>
#include <cuda_runtime.h>

#define MMA_D8(d, a0, a1, b0, b1) \
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 " \
        "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9}, {%10, %11}, {%0, %1, %2, %3, %4, %5, %6, %7};" \
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]), "+r"(d[6]), "+r"(d[7]) \
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1))

__global__ void mma_peak_kernel(float * out, long long iters) {
    // 4 条独立累加链，避免测到依赖延迟而非吞吐
    int d0[8] = {0}, d1[8] = {0}, d2[8] = {0}, d3[8] = {0};
    const int a0 = 0x3c003c00, a1 = 0x3c003c00, b0 = 0x3c003c00, b1 = 0x3c003c00;
    for (long long i = 0; i < iters; ++i) {
        MMA_D8(d0, a0, a1, b0, b1);
        MMA_D8(d1, a0, a1, b0, b1);
        MMA_D8(d2, a0, a1, b0, b1);
        MMA_D8(d3, a0, a1, b0, b1);
    }
    const int s = d0[0] + d1[1] + d2[2] + d3[3];
    out[threadIdx.x + blockIdx.x * blockDim.x] = (float) s;
}

int main() {
    const int dev = 1;
    cudaSetDevice(dev);
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, dev);
    printf("device %d: %s\n", dev, p.name);
    printf("  cc %d.%d, SMs %d, clock %.0f MHz, mem %.1f GB\n", p.major, p.minor,
           p.multiProcessorCount, p.clockRate / 1000.0, p.totalGlobalMem / 1073741824.0);
    printf("  smem/SM %zu B, smem/block optin %zu B, L2 %d B, regs/SM %d\n",
           p.sharedMemPerMultiprocessor, p.sharedMemPerBlockOptin, p.l2CacheSize, p.regsPerMultiprocessor);
    printf("  mem clock %.0f MHz, bus %d bit -> BW %.1f GB/s\n",
           p.memoryClockRate / 1000.0, p.memoryBusWidth,
           p.memoryClockRate * 2.0 * (p.memoryBusWidth / 8.0) / 1e6);

    const int nthreads = 128, nblocks = p.multiProcessorCount * 8;
    const long long iters = 100000;
    float * dout = nullptr;
    cudaMalloc(&dout, (size_t) nblocks * nthreads * sizeof(float));

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    mma_peak_kernel<<<nblocks, nthreads>>>(dout, 100);  // 预热
    cudaDeviceSynchronize();
    cudaEventRecord(t0);
    mma_peak_kernel<<<nblocks, nthreads>>>(dout, iters);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, t0, t1);

    // 每线程每迭代 4 条指令；整卡指令数
    const double instr = (double) nblocks * (nthreads / 32.0) * (double) iters * 4.0;
    const double t = ms * 1e-3;
    printf("  launch: %d blocks x %d threads, %.2f ms\n", nblocks, nthreads, ms);
    printf("  instr rate: %.2f Gmma/s = %.3f mma/SM/cycle\n", instr / t / 1e9,
           instr / t / p.multiProcessorCount / (p.clockRate * 1000.0));
    for (int f : {512, 1024, 2048}) {
        printf("  if %d FLOP/mma -> %.1f TFLOP/s\n", f, instr * f / t / 1e12);
    }
    cudaFree(dout);
    return 0;
}
