#include <cudnn.h>
#include <torch/extension.h>
#include <cuda_fp16.h>

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
torch::Tensor launch_conv2d_cudnn_v1(torch::Tensor input, torch::Tensor filters, torch::Tensor output) {
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
    
    int out_height = (in_height - kernel_height + 2 * pad_height) / stride_height + 1;
    int out_width = (in_width - kernel_width + 2 * pad_width) / stride_width + 1;
    
    cudnnHandle_t cudnn;
    checkCUDNN(cudnnCreate(&cudnn));
    
    cudnnTensorDescriptor_t input_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(
        input_descriptor,
        CUDNN_TENSOR_NHWC,
        CUDNN_DATA_HALF,
        1, in_channels, in_height, in_width
    ));
    
    cudnnTensorDescriptor_t output_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(
        output_descriptor,
        CUDNN_TENSOR_NHWC,
        CUDNN_DATA_HALF,
        1, out_channels, out_height, out_width
    ));
    
    cudnnFilterDescriptor_t filter_descriptor;
    checkCUDNN(cudnnCreateFilterDescriptor(&filter_descriptor));
    checkCUDNN(cudnnSetFilter4dDescriptor(
        filter_descriptor,
        CUDNN_DATA_HALF,
        CUDNN_TENSOR_NCHW,
        out_channels, in_channels, kernel_height, kernel_width
    ));
    
    cudnnConvolutionDescriptor_t conv_descriptor;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&conv_descriptor));
    checkCUDNN(cudnnSetConvolution2dDescriptor(
        conv_descriptor,
        pad_height, pad_width,
        stride_height, stride_width,
        1, 1,
        CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_HALF
    ));
    
    cudnnConvolutionFwdAlgo_t algorithm;
    checkCUDNN(cudnnGetConvolutionForwardAlgorithm(
        cudnn,
        input_descriptor,
        filter_descriptor,
        conv_descriptor,
        output_descriptor,
        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
        0,
        &algorithm
    ));
    
    size_t workspace_bytes = 0;
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn,
        input_descriptor,
        filter_descriptor,
        conv_descriptor,
        output_descriptor,
        algorithm,
        &workspace_bytes
    ));
    
    void* workspace = nullptr;
    if (workspace_bytes > 0) {
        cudaMalloc(&workspace, workspace_bytes);
    }
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    auto input_nhwc = input.permute({0, 2, 3, 1}).contiguous();
    
    auto output_nhwc = output.permute({0, 2, 3, 1}).contiguous();
    
    checkCUDNN(cudnnConvolutionForward(
        cudnn,
        &alpha,
        input_descriptor,
        input_nhwc.data_ptr<at::Half>(),
        filter_descriptor,
        filters.data_ptr<at::Half>(),
        conv_descriptor,
        algorithm,
        workspace,
        workspace_bytes,
        &beta,
        output_descriptor,
        output_nhwc.data_ptr<at::Half>()
    ));
    
    if (workspace) {
        cudaFree(workspace);
    }
    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyFilterDescriptor(filter_descriptor);
    cudnnDestroyConvolutionDescriptor(conv_descriptor);
    cudnnDestroy(cudnn);
    
    return output_nhwc.permute({0, 3, 1, 2}).contiguous();
}

// Conv2d Layer 2 with cuDNN
torch::Tensor launch_conv2d_cudnn_v2(torch::Tensor input, torch::Tensor filters, torch::Tensor output) {
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
    
    int out_height = (in_height - kernel_height + 2 * pad_height) / stride_height + 1;
    int out_width = (in_width - kernel_width + 2 * pad_width) / stride_width + 1;
    
    cudnnHandle_t cudnn;
    checkCUDNN(cudnnCreate(&cudnn));
    
    cudnnTensorDescriptor_t input_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(
        input_descriptor,
        CUDNN_TENSOR_NHWC,
        CUDNN_DATA_HALF,
        1, in_channels, in_height, in_width
    ));
    
    cudnnTensorDescriptor_t output_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(
        output_descriptor,
        CUDNN_TENSOR_NHWC,
        CUDNN_DATA_HALF,
        1, out_channels, out_height, out_width
    ));
    
    cudnnFilterDescriptor_t filter_descriptor;
    checkCUDNN(cudnnCreateFilterDescriptor(&filter_descriptor));
    checkCUDNN(cudnnSetFilter4dDescriptor(
        filter_descriptor,
        CUDNN_DATA_HALF,
        CUDNN_TENSOR_NCHW,
        out_channels, in_channels, kernel_height, kernel_width
    ));
    
    cudnnConvolutionDescriptor_t conv_descriptor;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&conv_descriptor));
    checkCUDNN(cudnnSetConvolution2dDescriptor(
        conv_descriptor,
        pad_height, pad_width,
        stride_height, stride_width,
        1, 1,
        CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_HALF
    ));
    
    cudnnConvolutionFwdAlgo_t algorithm;
    checkCUDNN(cudnnGetConvolutionForwardAlgorithm(
        cudnn,
        input_descriptor,
        filter_descriptor,
        conv_descriptor,
        output_descriptor,
        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
        0,
        &algorithm
    ));
    
    size_t workspace_bytes = 0;
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn,
        input_descriptor,
        filter_descriptor,
        conv_descriptor,
        output_descriptor,
        algorithm,
        &workspace_bytes
    ));
    
    void* workspace = nullptr;
    if (workspace_bytes > 0) {
        cudaMalloc(&workspace, workspace_bytes);
    }
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    auto input_nhwc = input.permute({0, 2, 3, 1}).contiguous();
    
    auto output_nhwc = output.permute({0, 2, 3, 1}).contiguous();
    
    checkCUDNN(cudnnConvolutionForward(
        cudnn,
        &alpha,
        input_descriptor,
        input_nhwc.data_ptr<at::Half>(),
        filter_descriptor,
        filters.data_ptr<at::Half>(),
        conv_descriptor,
        algorithm,
        workspace,
        workspace_bytes,
        &beta,
        output_descriptor,
        output_nhwc.data_ptr<at::Half>()
    ));
    
    if (workspace) {
        cudaFree(workspace);
    }
    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyFilterDescriptor(filter_descriptor);
    cudnnDestroyConvolutionDescriptor(conv_descriptor);
    cudnnDestroy(cudnn);
    
    return output_nhwc.permute({0, 3, 1, 2}).contiguous();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv2d_cudnn_v1", &launch_conv2d_cudnn_v1, "cuDNN convolution for layer 1");
    m.def("conv2d_cudnn_v2", &launch_conv2d_cudnn_v2, "cuDNN convolution for layer 2");
}
