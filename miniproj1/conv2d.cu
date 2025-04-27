#include <mma.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#define MMA_M 16
#define MMA_N 8
#define MMA_K 16
#define WARP_SIZE 32

#define div_ceil(a, b) (((a) + (b) - 1) / (b))

#define LDMATRIX_X4(R0, R1, R2, R3, addr)                                \
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" \
                 : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                \
                 : "r"(addr));

#define LDMATRIX_X2(R0, R1, addr)                                        \
    asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n" \
                 : "=r"(R0), "=r"(R1)                                    \
                 : "r"(addr));

#define HMMA16816(RC0, RC1, RA0, RA1, RA2, RA3, RB0, RB1, C0, C1)        \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n" \
                 : "=r"(RC0), "=r"(RC1)                                 \
                 : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1), "r"(C0), "r"(C1));

#define CUDA_KERNEL_LOOP_TYPE(i, n, index_type)                         \
  int64_t _i_n_d_e_x = ((int64_t) blockIdx.x) * blockDim.x + threadIdx.x;           \
  for (index_type i=_i_n_d_e_x; _i_n_d_e_x < (n); _i_n_d_e_x+=blockDim.x * gridDim.x, i=_i_n_d_e_x)

#define CUDA_KERNEL_LOOP(i, n) CUDA_KERNEL_LOOP_TYPE(i, n, int)

constexpr int CUDA_NUM_THREADS = 1024;

inline int GET_BLOCKS(const int64_t N, const int64_t max_threads_per_block=CUDA_NUM_THREADS) {
  TORCH_INTERNAL_ASSERT(N > 0, "CUDA kernel launch blocks must be positive, but got N=", N);
  constexpr int64_t max_int = std::numeric_limits<int>::max();

  auto block_num = (N - 1) / max_threads_per_block + 1;
  TORCH_INTERNAL_ASSERT(block_num <= max_int, "Can't schedule too many blocks on CUDA device");

  return static_cast<int>(block_num);
}

template <typename T, typename U>
__host__ __device__ __forceinline__
constexpr auto CEIL_DIV(T a, U b) {
    return (a + b - 1) / b;
}

__device__ __forceinline__ void print_matrix(const char* name, half* matrix, int rows, int cols) {
    if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
        printf("Matrix %s (%d x %d):\n", name, rows, cols);
        for (int i = 0; i < min(rows, 5); i++) {
            printf("Row %d: ", i);
            for (int j = 0; j < min(cols, 5); j++) {
                printf("%.4f ", matrix[i*cols + j]);
            }
            printf("...\n");
        }
        printf("...\n");
    }
    __syncthreads();
}

template <typename dt>
__global__ void im2col_kernel(
    const int64_t n,
    const dt* data_im,
    const int64_t height,
    const int64_t width,
    const int64_t kernel_height,
    const int64_t kernel_width,
    const int64_t pad_height,
    const int64_t pad_width,
    const int64_t stride_height,
    const int64_t stride_width,
    const int64_t dilation_height,
    const int64_t dilation_width,
    const int64_t height_col,
    const int64_t width_col,
    dt* data_col) {
  CUDA_KERNEL_LOOP_TYPE(index, n, int64_t) {
    int64_t w_out = index % width_col;
    int64_t idx = index / width_col;
    int64_t h_out = idx % height_col;
    int64_t channel_in = idx / height_col;
    int64_t channel_out = channel_in * kernel_height * kernel_width;
    int64_t h_in = h_out * stride_height - pad_height;
    int64_t w_in = w_out * stride_width - pad_width;
    dt* col = data_col + (channel_out * height_col + h_out) * width_col + w_out;
    const dt* im = data_im + (channel_in * height + h_in) * width + w_in;
    for (int64_t i = 0; i < kernel_height; ++i) {
      for (int64_t j = 0; j < kernel_width; ++j) {
        int64_t h = h_in + i * dilation_height;
        int64_t w = w_in + j * dilation_width;
        *col = (h >= 0 && w >= 0 && h < height && w < width)
            ? im[i * dilation_height * width + j * dilation_width]
            : static_cast<dt>(0);
        col += height_col * width_col;
      }
    }
  }
}

template <typename dt>
void im2col(
    cudaStream_t stream,
    const dt* data_im,
    const int64_t channels,
    const int64_t height,
    const int64_t width,
    const int64_t height_col,
    const int64_t width_col,
    const int64_t kernel_height,
    const int64_t kernel_width,
    const int64_t pad_height,
    const int64_t pad_width,
    const int64_t stride_height,
    const int64_t stride_width,
    const int64_t dilation_height,
    const int64_t dilation_width,
    dt* data_col) {
  int64_t num_kernels = channels * height_col * width_col;
  im2col_kernel<<<GET_BLOCKS(num_kernels), 1024, 0, stream>>>(
      num_kernels,
      data_im,
      height,
      width,
      kernel_height,
      kernel_width,
      pad_height,
      pad_width,
      stride_height,
      stride_width,
      dilation_height,
      dilation_width,
      height_col,
      width_col,
      data_col);
}

__global__ void matmul(half* A, half* B, half* C, const int m, const int n, const int d){
    // A is (m,d), B is (d,n), C is (m,n)
       
    __shared__ half As[8][64];
    __shared__ half Bs[8][64];


    float C_local[8][8];
    for (int ib = 0; ib < 8; ++ib) {
        for (int jb = 0; jb < 8; ++jb) {
            C_local[ib][jb] = 0;
        }
    }

    for (int k = 0; k < d; k+=8) {
        // each thread will load 8 elements into SRAM
        int local_thread = threadIdx.y * blockDim.x + threadIdx.x; //0-63 threads
        int i = blockIdx.x * 64 + local_thread;
        int j = blockIdx.y * 64 + local_thread;

        for(int f = 0; f < 8; f++){
            int local_k = k + f;
            As[f][local_thread] = A[local_k*m + i];
            Bs[f][local_thread] = B[local_k*n + j];
        }
        __syncthreads();
        
        #pragma unroll
        for(int f = 0; f < 8; f++){
            half x[8];
            half y[8];

            for (int ib = 0; ib < 8; ++ib) {
                x[ib] = As[f][ib*8+threadIdx.x];
                for (int jb = 0; jb < 8; ++jb) {
                    y[jb] = Bs[f][jb*8+threadIdx.y];
                    C_local[ib][jb] += __half2float(x[ib]) * __half2float(y[jb]);
                }
            }
        }
        __syncthreads();
    }

    for (int ib = 0; ib < 8; ++ib) {
        for (int jb = 0; jb < 8; ++jb) {
            int i = blockIdx.x * 64 + ib * 8 + threadIdx.x;
            int j = blockIdx.y * 64 + jb * 8 + threadIdx.y;
            if (i < m && j < n) {
                C[i*n + j] = __float2half(C_local[ib][jb]);
            }
        }
    }
}

__global__ void mma_matmul(const half *__restrict__ A, const half *__restrict__ B, half *__restrict__ C, size_t M,
                               size_t N, size_t K) {
    const size_t K_tiles = div_ceil(K, MMA_K);
    const size_t warp_row = blockIdx.y * MMA_M;
    const size_t warp_col = blockIdx.x * MMA_N;
    if (warp_row >= M || warp_col >= N) {
        return;
    }
    __shared__ half A_smem[MMA_M][MMA_K];
    __shared__ half B_smem[MMA_N][MMA_K];
    __shared__ half C_smem[MMA_M][MMA_N];
    const size_t lane_id = threadIdx.x % WARP_SIZE;
    uint32_t RC[2] = {0, 0};
#pragma unroll
    for (size_t i = 0; i < K_tiles; ++i) {
        *((int4 *)(&A_smem[lane_id / 2][0]) + lane_id % 2) =
            *((int4 *)(&A[(warp_row + lane_id / 2) * K + i * MMA_K]) + lane_id % 2);
        if (lane_id < MMA_N * 2) {
            *((int4 *)(&B_smem[lane_id / 2][0]) + lane_id % 2) =
                *((int4 *)(&B[i * MMA_K + (warp_col + lane_id / 2) * K]) + lane_id % 2);
        }
        __syncthreads();
        uint32_t RA[4];
        uint32_t RB[2];
        uint32_t A_smem_lane_addr = __cvta_generic_to_shared(&A_smem[lane_id % 16][(lane_id / 16) * 8]);
        LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], A_smem_lane_addr);
        uint32_t B_smem_lane_addr = __cvta_generic_to_shared(&B_smem[lane_id % 8][((lane_id / 8) % 2) * 8]);
        LDMATRIX_X2(RB[0], RB[1], B_smem_lane_addr);
        HMMA16816(RC[0], RC[1], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1], RC[0], RC[1]);
        __syncthreads();
    }
    *((uint32_t *)(&C_smem[lane_id / 4][0]) + lane_id % 4) = RC[0];
    *((uint32_t *)(&C_smem[lane_id / 4 + 8][0]) + lane_id % 4) = RC[1];
    __syncthreads();
    if (lane_id < MMA_M) {
        *((int4 *)(&C[(warp_row + lane_id) * N + warp_col])) = *((int4 *)(&C_smem[lane_id][0]));
    }
}

// Conv2d Layer 1 FP 16
torch::Tensor launch_conv2d_v1(torch::Tensor input, torch::Tensor filters, torch::Tensor output) {
    /* Conv2d version 1 parameters */
    constexpr int in_channels = 64; // Ni
    constexpr int in_height = 224; // Ny
    constexpr int in_width = 224; // Nx
    
    constexpr int out_channels = 64; // Nn
    constexpr int kernel_height = 3; // Ky
    constexpr int kernel_width = 3; // Kx
    
    int stride_height = 1;
    int stride_width = 1;
    int pad_height = 0;
    int pad_width = 0;
    int dilation_height = 1;
    int dilation_width = 1;
    
    int out_height = (in_height - kernel_height + 2 * pad_height) / stride_height + 1;
    int out_width = (in_width - kernel_width + 2 * pad_width) / stride_width + 1;
    
    auto col_buffer = torch::empty({in_channels * kernel_height * kernel_width, out_height * out_width},
                                  input.options());
    col_buffer = col_buffer.contiguous();
    
    auto input_no_batch = input.squeeze(0);
    auto filters_reshaped = filters.view({out_channels, in_channels * kernel_height * kernel_width});
    
    cudaStream_t stream = 0; // use main stream

    im2col<at::Half>(
        stream,
        input_no_batch.data_ptr<at::Half>(),
        in_channels,
        in_height,
        in_width,
        out_height,
        out_width,
        kernel_height,
        kernel_width,
        pad_height,
        pad_width,
        stride_height,
        stride_width,
        dilation_height,
        dilation_width,
        col_buffer.data_ptr<at::Half>()
    );

    auto output_reshaped = output.squeeze(0).view({out_channels, out_height * out_width});
    
    const int m = out_channels;
    const int n = out_height * out_width;
    const int k = in_channels * kernel_height * kernel_width;

    const int n_padded = CEIL_DIV(n, 64) * 64;
//    printf("n_padded: %d\n", n_padded);

//    printf("M: %d\n", m);
//    printf("N: %d\n", n);
//    printf("K: %d\n", k);

    const int num_threads = 8;
    dim3 blockDim(num_threads, num_threads);

    const int grid_size_x = (m+63)/64;
    const int grid_size_y = (n+63)/64;
    dim3 gridDim(grid_size_x, grid_size_y);
    
    // Filter_matrix (No x Ni * Kx * Ky), Im2Col_matrix (Ni * Kx * Ky, H_out * W_out)
    // Output: (No, H_out * W_out)
    filters_reshaped = filters_reshaped.t().contiguous();
    
    matmul<<<gridDim, blockDim>>>(
        reinterpret_cast<half*>(filters_reshaped.data_ptr<at::Half>()),
        reinterpret_cast<half*>(col_buffer.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output_reshaped.data_ptr<at::Half>()),
        m, n, k
    );
    
    
    return output;
}


// FP 16 Conv2d Layer 2
torch::Tensor launch_conv2d_v2(torch::Tensor input, torch::Tensor filters, torch::Tensor output) {
    /* Conv2d layer 2 parameters */
    constexpr int in_channels = 512; // Ni
    constexpr int in_height = 14; // Ny
    constexpr int in_width = 14; // Nx
    
    constexpr int out_channels = 512; // Nn
    constexpr int kernel_height = 3; // Ky
    constexpr int kernel_width = 3; // Kx
    
    int stride_height = 1;
    int stride_width = 1;
    int pad_height = 0;
    int pad_width = 0;
    int dilation_height = 1;
    int dilation_width = 1;
    
    int out_height = (in_height - kernel_height + 2 * pad_height) / stride_height + 1;
    int out_width = (in_width - kernel_width + 2 * pad_width) / stride_width + 1;
    
    auto col_buffer = torch::empty({in_channels * kernel_height * kernel_width, out_height * out_width},
                                  input.options());
    col_buffer = col_buffer.contiguous();
    auto input_no_batch = input.squeeze(0);
    auto filters_reshaped = filters.view({out_channels, in_channels * kernel_height * kernel_width});
    
    cudaStream_t stream = 0; // use main stream
    
    im2col<at::Half>(
        stream,
        input_no_batch.data_ptr<at::Half>(),
        in_channels,
        in_height,
        in_width,
        out_height,
        out_width,
        kernel_height,
        kernel_width,
        pad_height,
        pad_width,
        stride_height,
        stride_width,
        dilation_height,
        dilation_width,
        col_buffer.data_ptr<at::Half>()
    );

    auto output_reshaped = output.squeeze(0).view({out_channels, out_height * out_width});
    
    const int m = out_channels;
    const int n = out_height * out_width;
    const int k = in_channels * kernel_height * kernel_width;

    const int n_padded = CEIL_DIV(n, 64) * 64;
 //   printf("n_padded: %d\n", n_padded);

  //  printf("M: %d\n", m);
   // printf("N: %d\n", n);
    //printf("K: %d\n", k);

   
    const int num_threads = 8;
    dim3 blockDim(num_threads, num_threads);

    const int grid_size_x = (m+63)/64;
    const int grid_size_y = (n+63)/64;
    dim3 gridDim(grid_size_x, grid_size_y);
    

    /*
    dim3 blockDim(WARP_SIZE);
    dim3 gridDim(div_ceil(n, MMA_N), div_ceil(m, MMA_M));
    */

    // Filter_matrix (No x Ni * Kx * Ky), Im2Col_matrix (Ni * Kx * Ky, H_out * W_out)
    // Output: (No, H_out * W_out)
    filters_reshaped = filters_reshaped.t().contiguous();
    matmul<<<gridDim, blockDim>>>(
        reinterpret_cast<half*>(filters_reshaped.data_ptr<at::Half>()),
        reinterpret_cast<half*>(col_buffer.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output_reshaped.data_ptr<at::Half>()),
        m, n, k
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv2d_v1", &launch_conv2d_v1, "im2col + gemm");
    m.def("conv2d_v2", &launch_conv2d_v2, "im2col + gemm");
}
