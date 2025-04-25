#include <mma.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
// #include <ptx.h>

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
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;

    if(i < m && j < n){
        float inner_prod = 0.0f;
        for(int k = 0; k < d; k++){
            inner_prod += __half2float(A[i*d+k]) * __half2float(B[k*n+j]);
        }
        C[i*n+j] = inner_prod;
    }
}

template <const uint Nx, const uint Ny, 
         const uint Kx, const uint Ky, 
         const uint Ni, const uint Nn, 
         const uint stride>
__global__ void conv2d_v2(const half* input, const half* filter, float* output) {
    printf("hi");

    constexpr uint OUT_H = (Ny - Ky) / stride + 1;
    constexpr uint OUT_W = (Nx - Kx) / stride + 1;

    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int height_col = (OUT_H*OUT_W);
    int width_col = (Kx*Ky);
    int pad = 0;

    half im2col_matrix[OUT_H * OUT_W * (Kx*Ky)];
    half filter_matrix[Kx*Ky * Nn];

    for (int c = 0; c < Ni; c++){
        int w_out = idx % width_col;
        int h_idx = idx / width_col;
        int h_out = h_idx % height_col;

        int h_in = h_out * stride - pad;
        int w_in = w_out * stride - pad;

        for (int i = 0; i < Ky; ++i) {
            for (int j = 0; j < Kx; ++j) {
                int h = h_in + i;
                int w = w_in + j;

                im2col_matrix[(h_out * width_col + w_out) * width_col + (i*Kx + j)] =
                    (h >= 0 && w >= 0 && h < Ny && w < Nx) ?
                    input[c*Nx*Ny + (h*Nx + w)] : __float2half(0.0f);
                
                for (int oc = 0; oc < Ny; oc++){
                    filter_matrix[(i*Kx + j) * Nn + oc] = filter[oc*Ni*Kx*Ky + (i*Kx+j)];
                }
            }
        }

        print_matrix("im2col", im2col_matrix, height_col, width_col);
        print_matrix("filter", filter_matrix, Kx*Ky, Nn);

        int i = blockIdx.x*blockDim.x + threadIdx.x;
        int j = blockIdx.y*blockDim.y + threadIdx.y;

        int M = h_out * w_out;
        int N = Ni;
        int K = Kx*Ky;
        
        if(i < M && j < N){
            float inner_prod = 0.0f;
            for(int k = 0; k < K; k++){
                inner_prod += __half2float(im2col_matrix[i*K+k]) * __half2float(filter_matrix[k*N+j]);
            }
            output[i*N+j] = inner_prod;
        }
    }
}

torch::Tensor launch_conv2d_v1(torch::Tensor input, torch::Tensor filters, torch::Tensor output) {
    auto in_channels = input.size(1);
    auto in_height = input.size(2);
    auto in_width = input.size(3);
    
    auto out_channels = filters.size(0);
    auto kernel_height = filters.size(2);
    auto kernel_width = filters.size(3);
    
    int stride_height = 1;
    int stride_width = 1;
    int pad_height = 0;
    int pad_width = 0;
    int dilation_height = 1;
    int dilation_width = 1;
    
    auto out_height = (in_height - kernel_height + 2 * pad_height) / stride_height + 1;
    auto out_width = (in_width - kernel_width + 2 * pad_width) / stride_width + 1;
    
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
    
    const int num_threads = 16;
    dim3 blockDim(num_threads, num_threads);
    const int grid_size_x = (m + num_threads - 1) / num_threads;
    const int grid_size_y = (n + num_threads - 1) / num_threads;
    dim3 gridDim(grid_size_x, grid_size_y);

    // Filter_matrix (No x Ni * Kx * Ky), Im2Col_matrix (Ni * Kx * Ky, H_out * W_out)
    // Output: (No, H_out * W_out)
    matmul<<<gridDim, blockDim>>>(
        reinterpret_cast<half*>(filters_reshaped.data_ptr<at::Half>()),
        reinterpret_cast<half*>(col_buffer.data_ptr<at::Half>()),
        output_reshaped.data_ptr<float>(),
        m, n, k
    );
    
    return output;
}

torch::Tensor launch_conv2d_v2(torch::Tensor input, torch::Tensor filters, torch::Tensor output){
    const uint Nx = 224;
    const uint Ny = 224;
    const uint Kx = 3;
    const uint Ky = 3;
    const uint Ni = 64;
    const uint Nn = 64;
    const uint stride = 1;
    const uint BLOCK_SIZE = 16;

    constexpr uint OUT_H = (Ny - Ky) / stride + 1;
    constexpr uint OUT_W = (Nx - Kx) / stride + 1;

    dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dim_grid(CEIL_DIV(OUT_H * OUT_W, BLOCK_SIZE),
            CEIL_DIV(Nn, BLOCK_SIZE));

    printf("Before kernel launch\n");
    conv2d_v2<Nx, Ny, Kx, Ky, Ni, Nn, stride>
    <<<dim_grid, dim_block>>>(
        reinterpret_cast<half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<half*>(filters.data_ptr<at::Half>()),
        output.data_ptr<float>()
    );

    cudaDeviceSynchronize();
    printf("After kernel launch\n");
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv2d_v1", &launch_conv2d_v1, "im2col + gemm");
    m.def("conv2d_v2", &launch_conv2d_v2, "fused im2col + gemm");
}
