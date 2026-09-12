// cublas fp16×fp16+fp32 累加 GEMM 峰值实测：用于独立校准 HMMA 的 FLOP/指令口径
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

int main() {
    const int dev = 1;
    cudaSetDevice(dev);
    const int N = 8192;
    half * A = nullptr, * B = nullptr, * C = nullptr;
    cudaMalloc(&A, (size_t) N * N * sizeof(half));
    cudaMalloc(&B, (size_t) N * N * sizeof(half));
    cudaMalloc(&C, (size_t) N * N * sizeof(half));
    cudaMemset(A, 0x3c, (size_t) N * N * sizeof(half));
    cudaMemset(B, 0x3c, (size_t) N * N * sizeof(half));

    cublasHandle_t h;
    cublasCreate(&h);
    cublasSetMathMode(h, CUBLAS_DEFAULT_MATH);

    const float alpha = 1.0f, beta = 0.0f;
    // 预热
    cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, B, CUDA_R_16F, N, A, CUDA_R_16F, N,
                 &beta, C, CUDA_R_16F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    cudaDeviceSynchronize();

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    const int reps = 5;
    cudaEventRecord(t0);
    for (int i = 0; i < reps; ++i) {
        cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, B, CUDA_R_16F, N, A, CUDA_R_16F, N,
                     &beta, C, CUDA_R_16F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    }
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, t0, t1);
    const double flops = 2.0 * (double) N * N * N * reps;
    printf("cublas fp16/fp32acc GEMM %d^3 x%d: %.1f ms -> %.1f TFLOP/s\n", N, reps, ms, flops / (ms * 1e-3) / 1e12);

    cublasDestroy(h);
    cudaFree(A); cudaFree(B); cudaFree(C);
    return 0;
}
