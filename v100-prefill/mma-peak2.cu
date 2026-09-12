// 复核微基准：mma.sync.m8n8k4 吞吐（4/8/16 链、不同占用）+ ncu 交叉测量模式
// 目的：为账本 §5 提供"微指令/条"与峰值率的可复查证据
// 形态与 /root/llama_gguf/mma-peak.cu 完全一致（同一条 asm、D=8xf32）
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define MMA_D8(d, a0, a1, b0, b1) \
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 " \
        "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9}, {%10, %11}, {%0, %1, %2, %3, %4, %5, %6, %7};" \
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]), "+r"(d[6]), "+r"(d[7]) \
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1))

template<int CHAINS>
__global__ void mma_peak_kernel(float * out, long long iters) {
    int d[CHAINS][8];
    #pragma unroll
    for (int c = 0; c < CHAINS; ++c)
        #pragma unroll
        for (int j = 0; j < 8; ++j) d[c][j] = 0;
    const int a0 = 0x3c003c00, a1 = 0x3c003c00, b0 = 0x3c003c00, b1 = 0x3c003c00;
    for (long long i = 0; i < iters; ++i) {
        #pragma unroll
        for (int c = 0; c < CHAINS; ++c) MMA_D8(d[c], a0, a1, b0, b1);
    }
    int s = 0;
    #pragma unroll
    for (int c = 0; c < CHAINS; ++c) s += d[c][0];
    out[threadIdx.x + blockIdx.x * blockDim.x] = (float) s;
}

static float run(int chains, int nblocks, int nthreads, long long iters) {
    float * dout = nullptr;
    cudaMalloc(&dout, (size_t) nblocks * nthreads * sizeof(float));
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    if (chains == 4) {
        mma_peak_kernel<4><<<nblocks, nthreads>>>(dout, 100);
        cudaDeviceSynchronize();
        cudaEventRecord(t0);
        mma_peak_kernel<4><<<nblocks, nthreads>>>(dout, iters);
    } else if (chains == 8) {
        mma_peak_kernel<8><<<nblocks, nthreads>>>(dout, 100);
        cudaDeviceSynchronize();
        cudaEventRecord(t0);
        mma_peak_kernel<8><<<nblocks, nthreads>>>(dout, iters);
    } else {
        mma_peak_kernel<16><<<nblocks, nthreads>>>(dout, 100);
        cudaDeviceSynchronize();
        cudaEventRecord(t0);
        mma_peak_kernel<16><<<nblocks, nthreads>>>(dout, iters);
    }
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, t0, t1);
    const double instr = (double) nblocks * (nthreads / 32.0) * (double) iters * chains;
    cudaFree(dout);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    return (float) (instr / (ms * 1e-3) / 1e9); // Gmma/s
}

int main(int argc, char ** argv) {
    const int dev = 1;
    cudaSetDevice(dev);
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, dev);
    const double clk = p.clockRate * 1000.0;
    printf("device %d: %s | SMs %d | clock %.0f MHz\n", dev, p.name, p.multiProcessorCount, clk / 1e6);

    if (argc > 1 && argv[1][0] == 'n') { // ncu 交叉测量：单配置 2 次（第 2 次被测）
        const int chains = atoi(argv[1] + 1);
        const long long iters = 2000;
        float * dout = nullptr;
        cudaMalloc(&dout, (size_t) p.multiProcessorCount * 8 * 128 * sizeof(float));
        for (int i = 0; i < 2; ++i) {
            if (chains == 4) mma_peak_kernel<4><<<p.multiProcessorCount * 8, 128>>>(dout, iters);
            else if (chains == 8) mma_peak_kernel<8><<<p.multiProcessorCount * 8, 128>>>(dout, iters);
            else mma_peak_kernel<16><<<p.multiProcessorCount * 8, 128>>>(dout, iters);
            cudaDeviceSynchronize();
        }
        printf("ncu mode done: chains=%d iters=%lld grid=(%d,1,1)x128 (2 launches)\n",
               chains, iters, p.multiProcessorCount * 8);
        cudaFree(dout);
        return 0;
    }

    const long long iters = 20000;
    const int sm = p.multiProcessorCount;
    const int cfgs[][2] = {{sm * 8, 128}, {sm * 16, 128}, {sm * 4, 256}, {sm * 8, 256}};
    for (int chains : {4, 8, 16}) {
        for (auto & c : cfgs) {
            const float g = run(chains, c[0], c[1], iters);
            printf("chains=%2d blocks=%4d threads=%3d | %7.2f Gmma/s = %.3f mma/SM/cycle\n",
                   chains, c[0], c[1], g, g * 1e9 / sm / clk);
        }
    }
    return 0;
}
