#include <mma.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include "ptx.h"

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

__global__ void matmul(half* A, half* B, float* C, const int m, const int n, const int d){
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

  //          uint32_t Areg[4];
  //          uint32_t Breg[2];
            for (int ib = 0; ib < 8; ++ib) {
                x[ib] = As[f][ib*8+threadIdx.x];
//                load_matrix_x4(Areg, As[f][ib*8+threadIdx.x]);
                for (int jb = 0; jb < 8; ++jb) {
                    y[jb] = Bs[f][jb*8+threadIdx.y];
//                    load_matrix_x2(Breg, Bs[f][ib*8+threadIdx.x]);
                    //reuse x[ib] and y[jb]
   //                 half temp = __hmul(x[ib], y[jb]);
                    C_local[ib][jb] += __half2float(x[ib]) * __half2float(y[jb]);


 //                   float Creg[4] = {0, 0, 0, 0};
//                    mma_m16n8k8(Areg, Breg, Creg, Creg);
                    // not sure if im doing the Creg correctly?
 //                   C_local[ib][jb] += Creg[0] + Creg[1] + Creg[2] + Creg[3];

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
                C[i*n + j] = C_local[ib][jb];
            }
        }
    }
}

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
  //  printf("n_padded: %d\n", n_padded);

  //  printf("M: %d\n", m);
  //  printf("N: %d\n", n);
  //  printf("K: %d\n", k);

    const int num_threads = 8;
    dim3 blockDim(num_threads, num_threads);

    const int grid_size_x = (m+63)/64;
    const int grid_size_y = (n+63)/64;
    dim3 gridDim(grid_size_x, grid_size_y);
    
 //    dim3 blockDim(16,16);
 //   dim3 gridDim((m/64),(n/64));

    // Filter_matrix (No x Ni * Kx * Ky), Im2Col_matrix (Ni * Kx * Ky, H_out * W_out)
    // Output: (No, H_out * W_out)
    filters_reshaped = filters_reshaped.t().contiguous();
    matmul<<<gridDim, blockDim>>>(
        reinterpret_cast<half*>(filters_reshaped.data_ptr<at::Half>()),
        reinterpret_cast<half*>(col_buffer.data_ptr<at::Half>()),
        output_reshaped.data_ptr<float>(),
        m, n, k
    );
    
    return output;
}

torch::Tensor launch_conv2d_v2(torch::Tensor input, torch::Tensor filters, torch::Tensor output) {
    /* Conv2d version 2 parameters */
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

 //   printf("M: %d\n", m);
 //   printf("N: %d\n", n);
 //   printf("K: %d\n", k);

    const int num_threads = 8;
    dim3 blockDim(num_threads, num_threads);

    const int grid_size_x = (m+63)/64;
    const int grid_size_y = (n+63)/64;
    dim3 gridDim(grid_size_x, grid_size_y);
    
 //    dim3 blockDim(16,16);
 //   dim3 gridDim((m/64),(n/64));

    // Filter_matrix (No x Ni * Kx * Ky), Im2Col_matrix (Ni * Kx * Ky, H_out * W_out)
    // Output: (No, H_out * W_out)
    filters_reshaped = filters_reshaped.t().contiguous();
    matmul<<<gridDim, blockDim>>>(
        reinterpret_cast<half*>(filters_reshaped.data_ptr<at::Half>()),
        reinterpret_cast<half*>(col_buffer.data_ptr<at::Half>()),
        output_reshaped.data_ptr<float>(),
        m, n, k
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv2d_v1", &launch_conv2d_v1, "im2col + gemm");
    m.def("conv2d_v2", &launch_conv2d_v2, "im2col + gemm");
}
