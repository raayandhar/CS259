import torch
from torch.utils.cpp_extension import load
import os
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'  # Ampere
print("\nCompiling Conv2D module...\n")
conv = load(name="conv2d", sources=["conv2d.cu"], extra_cuda_cflags=["-arch=sm_86", "-O3", "-g", "--generate-line-info"])

def time_conv(func, *args, num_runs=10):
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(num_runs):
        result = func(*args)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event) / num_runs
    return result, elapsed_time

def conv2d_ref(input, filters):
    return torch.nn.functional.conv2d(input, filters, stride=1, padding=0)

if __name__ == "__main__":
    # First run the example from the image
    #test_example_from_image()
    
    # Then run the performance test
    print("\n=== Performance Test ===\n")
    
    # Define dimensions
    N, C, H, W = 1, 64, 224, 224  # Input: batch_size, channels, height, width
    K, C_out = 3, 64             # Kernel: size, output_channels

#    N, C, H, W = 1, 2, 3, 3  # Input: batch_size, channels, height, width
#    K, C_out = 2, 2         # Kernel: size, output_channels
    
    input = torch.rand(N, C, H, W, device='cuda')
    filters = torch.rand(C_out, C, K, K, device='cuda')
   
    input_half = input.half()
    filters_half = filters.half()
    
    H_out = H - K + 1
    W_out = W - K + 1
    
    output_v1 = torch.zeros(N, C_out, H_out, W_out, device='cuda')
    output_v2 = torch.zeros(N, C_out, H_out, W_out, device='cuda')
    
    output_v1, conv_v1_time = time_conv(conv.conv2d_v1, input_half, filters_half, output_v1)
#    output_v2, conv_v2_time = time_conv(conv.conv2d_v2, input_half, filters_half, output_v2)
    
    output_ref = conv2d_ref(input, filters)
    
    OPS_PER_ELEMENT = K * K * C * 2 
    TOTAL_OPS = N * C_out * H_out * W_out * OPS_PER_ELEMENT
    NUM_GFLOPS = TOTAL_OPS / 1e9
    
    output_ref_reshaped = output_ref.float()
    print('Conv2d_v1 TEST CHECK:', torch.allclose(output_ref_reshaped, output_v1, rtol=1e-02, atol=1e-02))
    print(f"Conv2d_v1 time: {conv_v1_time:.4f} ms")
    print(f"Conv2d_v1 GFLOPS/s: {(NUM_GFLOPS/(conv_v1_time*1e-3)):.4f}\n")

#    print('Conv2d_v2 TEST CHECK:', torch.allclose(output_ref_reshaped, output_v2, rtol=1e-02, atol=1e-02))
#    print(f"Conv2d_v2 time: {conv_v2_time:.4f} ms")
#    print(f"Conv2d_v2 GFLOPS/s: {(NUM_GFLOPS/(conv_v2_time*1e-3)):.4f}\n")
