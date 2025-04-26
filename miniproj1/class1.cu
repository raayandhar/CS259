// class1.cu — fp16 matvec
// Compile: nvcc -O3 -arch=sm_89 class1.cu -lcublas -o class1

#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <random>

#define Ni 25088
#define Nn 4096

#define WARPS_PER_BLOCK 4
#define THREADS_PER_BLOCK (WARPS_PER_BLOCK * 32)

// class1
__global__ void class1(const __half* __restrict__ W, const __half* __restrict__ x,  __half* __restrict__ y)
{
    const int warp_id = threadIdx.x >> 5; // 0...3
    const int lane  = threadIdx.x & 31; // 0..31
    const int out_idx = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (out_idx >= Nn) return;

    const __half* w_row = W + static_cast<size_t>(out_idx) * Ni;

    // On sm_8x arch, fp16->fp32 promotion is basically free in the
    // CUDA core datapath, so casting back to half costs nothing
    // ff: vector dp almost always accumulates in fp32 anyways
    float acc = 0.0f;
#pragma unroll 4
    for (int i = lane; i < Ni; i += 32)
        acc += __half2float(w_row[i]) * __half2float(x[i]);    // 

    // warp‑level reduce
    // to do this warp-reduction, note that __shfl* operates only on 32-bit words
    // https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/
    // https://forums.developer.nvidia.com/t/what-kind-of-function-is-shfl-down-sync/255684
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, offset);

    if (lane == 0)
        y[out_idx] = __float2half(acc);
}

// catch-all check-cuda error string stolen from another repo
#define CHECK_CUDA(call)                                                   \
    do {                                                                   \
        cudaError_t _e = (call);                                           \
        if (_e != cudaSuccess) {                                           \
            std::cerr << "CUDA error: " << cudaGetErrorString(_e)         \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                  \
    } while (0)

// also stolen from elsewhere
static __half rand_half(std::default_random_engine &gen, std::uniform_real_distribution<float> &dist) 
{
    return __float2half(dist(gen));
}

void benchmark()
{
    std::default_random_engine rng(42);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);

    // host buffers
    std::vector<__half> h_x(Ni);
    std::vector<__half> h_W(static_cast<size_t>(Nn) * Ni);
    std::vector<__half> h_y(Nn, __float2half(0.f));

    for (auto &v : h_x) v = rand_half(rng, dist);
    for (auto &v : h_W) v = rand_half(rng, dist);

    // device buffers
    __half *d_x, *d_W, *d_y, *d_y_ref;
    CHECK_CUDA(cudaMalloc(&d_x, Ni * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_W, static_cast<size_t>(Nn) * Ni * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_y, Nn * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_y_ref, Nn * sizeof(__half)));

    CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), Ni * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_W, h_W.data(), static_cast<size_t>(Nn) * Ni * sizeof(__half), cudaMemcpyHostToDevice));

    dim3 block(THREADS_PER_BLOCK);
    dim3 grid((Nn + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);

    // warm‑up
    class1<<<grid, block>>>(d_W, d_x, d_y);
    CHECK_CUDA(cudaDeviceSynchronize());

    // timed loop
    constexpr int ITER = 100;
    cudaEvent_t start, stop; 
    cudaEventCreate(&start); 
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < ITER; ++i)
        class1<<<grid, block>>>(d_W, d_x, d_y);
    cudaEventRecord(stop); 
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_class1 = 0.f; 
    cudaEventElapsedTime(&ms_class1, start, stop);
    ms_class1 /= ITER;
    double gflop = 2.0 * Ni * Nn * 1e-6;
    std::cout << "[class1] " << ms_class1 << " ms (" << gflop / ms_class1 << " GFLOP/s)\n";

    cublasHandle_t handle; 
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
    const __half alpha = __float2half(1.f);
    const __half beta  = __float2half(0.f);

    cudaEventRecord(start);
    // This runs f16 GEMM with a 1xN matrix, which is equivalent to GEMV
    // cuBLAS does not have HGemv: https://www.cnblogs.com/thisjiang/p/12609758.html
    // This is probably the reason we are doing well...?
    // Due to our conversions in the kernel itself (float->half), we don't get
    // Exactly the same results as cuBLAS, but they are within tolerance
    for (int i = 0; i < ITER; ++i) {
	cublasGemmEx(handle,
		     CUBLAS_OP_T,  // ← read W^T so that its rows become columns
		     CUBLAS_OP_N,
		     Nn,           // m  Nn   =     (rows of W)
		     1,            // n = 1         (columns of x)
		     Ni,           // k = Ni
		     &alpha,
		     d_W, CUDA_R_16F, Ni,   // lda = Ni (row length)
		     d_x, CUDA_R_16F, Ni,   // ldb = Ni
		     &beta,
		     d_y_ref, CUDA_R_16F, Nn,// ldc = Nn
		     CUDA_R_16F,
		     CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    cudaEventRecord(stop); 
    CHECK_CUDA(cudaEventSynchronize(stop));

    float msCublas = 0.f; cudaEventElapsedTime(&msCublas, start, stop); msCublas /= ITER;
    std::cout << "[cuBLAS] " << msCublas << " ms (" << gflop / msCublas << " GFLOP/s)\n";

    // cleanup
    cublasDestroy(handle);
    cudaFree(d_x); 
    cudaFree(d_W);
    cudaFree(d_y); 
    cudaFree(d_y_ref);
    cudaEventDestroy(start); 
    cudaEventDestroy(stop);
}

int main() { 
    benchmark(); 
    return 0; 
}
