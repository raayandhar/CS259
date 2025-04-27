#include <cudnn.h>
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <tuple>

#define checkCUDNN(expression)                             \
  {                                                        \
    cudnnStatus_t status = (expression);                   \
    if (status != CUDNN_STATUS_SUCCESS) {                  \
      std::cerr << "Error on line " << __LINE__ << ": "    \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                             \
    }                                                      \
  }

// Conv2d Layer 1 with cuDNN
std::tuple<torch::Tensor, float> launch_conv2d_cudnn_v1(torch::Tensor input, torch::Tensor filters, torch::Tensor output) {
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
    
    cudaSetDevice(0);
    
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);
    
    cudnnTensorDescriptor_t input_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                                         CUDNN_TENSOR_NCHW,
                                         CUDNN_DATA_HALF,
                                         1,
                                         in_channels,
                                         in_height,
                                         in_width));
    
    cudnnFilterDescriptor_t kernel_descriptor;
    checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
    checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor,
                                         CUDNN_DATA_HALF,
                                         CUDNN_TENSOR_NCHW,
                                         out_channels,
                                         in_channels,
                                         kernel_height,
                                         kernel_width));
    
    cudnnConvolutionDescriptor_t convolution_descriptor;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
    checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                              pad_height,
                                              pad_width,
                                              stride_height,
                                              stride_width,
                                              1,
                                              1,
                                              CUDNN_CROSS_CORRELATION,
                                              CUDNN_DATA_FLOAT));
    
 
//    checkCUDNN(cudnnSetConvolutionMathType(convolution_descriptor, CUDNN_TENSOR_OP_MATH));
    
    int batch_size{0}, channels{0}, height{0}, width{0};
    checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convolution_descriptor,
                                                    input_descriptor,
                                                    kernel_descriptor,
                                                    &batch_size,
                                                    &channels,
                                                    &height,
                                                    &width));
    
    cudnnTensorDescriptor_t output_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
                                         CUDNN_TENSOR_NCHW,
                                         CUDNN_DATA_HALF,
                                         batch_size,
                                         channels,
                                         height,
                                         width));
    
    cudnnConvolutionFwdAlgo_t convolution_algorithm = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    
    size_t workspace_bytes{0};
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                                                     input_descriptor,
                                                     kernel_descriptor,
                                                     convolution_descriptor,
                                                     output_descriptor,
                                                     convolution_algorithm,
                                                     &workspace_bytes));
    
    void* workspace{nullptr};
    if (workspace_bytes > 0) {
        cudaMalloc(&workspace, workspace_bytes);
    }
    
    const float alpha = 1.0f, beta = 0.0f;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    
    checkCUDNN(cudnnConvolutionForward(cudnn,
                                     &alpha,
                                     input_descriptor,
                                     input.data_ptr<at::Half>(),
                                     kernel_descriptor,
                                     filters.data_ptr<at::Half>(),
                                     convolution_descriptor,
                                     convolution_algorithm,
                                     workspace,
                                     workspace_bytes,
                                     &beta,
                                     output_descriptor,
                                     output.data_ptr<at::Half>()));
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    if (workspace) {
        cudaFree(workspace);
    }
    
    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyFilterDescriptor(kernel_descriptor);
    cudnnDestroyConvolutionDescriptor(convolution_descriptor);
    
    cudnnDestroy(cudnn);
    
    return std::make_tuple(output, milliseconds);
}

// Conv2d Layer 2 with cuDNN
std::tuple<torch::Tensor, float> launch_conv2d_cudnn_v2(torch::Tensor input, torch::Tensor filters, torch::Tensor output) {
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
    
    cudaSetDevice(0);
    
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);
    
    cudnnTensorDescriptor_t input_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                                         CUDNN_TENSOR_NCHW,
                                         CUDNN_DATA_HALF,
                                         1,
                                         in_channels,
                                         in_height,
                                         in_width));
    
    cudnnFilterDescriptor_t kernel_descriptor;
    checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
    checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor,
                                         CUDNN_DATA_HALF,
                                         CUDNN_TENSOR_NCHW,
                                         out_channels,
                                         in_channels,
                                         kernel_height,
                                         kernel_width));
    
    cudnnConvolutionDescriptor_t convolution_descriptor;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
    checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                              pad_height,
                                              pad_width,
                                              stride_height,
                                              stride_width,
                                              1,
                                              1,
                                              CUDNN_CROSS_CORRELATION,
                                              CUDNN_DATA_FLOAT));
    
 
//    checkCUDNN(cudnnSetConvolutionMathType(convolution_descriptor, CUDNN_TENSOR_OP_MATH));
    
    int batch_size{0}, channels{0}, height{0}, width{0};
    checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convolution_descriptor,
                                                    input_descriptor,
                                                    kernel_descriptor,
                                                    &batch_size,
                                                    &channels,
                                                    &height,
                                                    &width));
    
    cudnnTensorDescriptor_t output_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
                                         CUDNN_TENSOR_NCHW,
                                         CUDNN_DATA_HALF,
                                         batch_size,
                                         channels,
                                         height,
                                         width));
    
    cudnnConvolutionFwdAlgo_t convolution_algorithm = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED;
    
    size_t workspace_bytes{0};
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                                                     input_descriptor,
                                                     kernel_descriptor,
                                                     convolution_descriptor,
                                                     output_descriptor,
                                                     convolution_algorithm,
                                                     &workspace_bytes));
    
    void* workspace{nullptr};
    if (workspace_bytes > 0) {
        cudaMalloc(&workspace, workspace_bytes);
    }
    
    const float alpha = 1.0f, beta = 0.0f;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    
    checkCUDNN(cudnnConvolutionForward(cudnn,
                                     &alpha,
                                     input_descriptor,
                                     input.data_ptr<at::Half>(),
                                     kernel_descriptor,
                                     filters.data_ptr<at::Half>(),
                                     convolution_descriptor,
                                     convolution_algorithm,
                                     workspace,
                                     workspace_bytes,
                                     &beta,
                                     output_descriptor,
                                     output.data_ptr<at::Half>()));
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    if (workspace) {
        cudaFree(workspace);
    }
    
    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyFilterDescriptor(kernel_descriptor);
    cudnnDestroyConvolutionDescriptor(convolution_descriptor);
    
    cudnnDestroy(cudnn);
    
    return std::make_tuple(output, milliseconds);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv2d_cudnn_v1", &launch_conv2d_cudnn_v1, "cuDNN convolution for layer 1");
    m.def("conv2d_cudnn_v2", &launch_conv2d_cudnn_v2, "cuDNN convolution for layer 2");
}
