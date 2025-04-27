import torch
from torch.utils.cpp_extension import load
import os
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'  # Ampere
print("\nCompiling Conv2D and CUDNN module...\n")
conv = load(name="conv2d", sources=["conv2d.cu"], extra_cuda_cflags=["-arch=sm_86", "-O3", "-g", "--generate-line-info"])
cudnn_conv = load(name="cudnn_conv2d", sources=["cudnn_conv2d.cu"], 
                extra_cuda_cflags=["-arch=sm_86", "-O3", "-g", "--generate-line-info"], extra_ldflags=["-lcudnn"])

def time_conv(func, *args, num_runs=5):
    for _ in range(1): # warmup
        result = func(*args)

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

def time_cudnn_conv(func, *args, num_runs=5):
    for _ in range(1): #warmup
        _,_ = func(*args)

    result, internal_time = func(*args)
    cudnn_times = [internal_time]
    
    for _ in range(num_runs-1):
        _, time = func(*args)
        cudnn_times.append(time)
    
    avg_internal_time = sum(cudnn_times) / len(cudnn_times)
    return result, avg_internal_time

def conv2d_ref(input, filters):
    return torch.nn.functional.conv2d(input, filters, stride=1, padding=0)

if __name__ == "__main__":

    N, C, H, W = 1, 64, 224, 224  # Input: batch_size, channels, height, width
    K, C_out = 3, 64             # Kernel: size, output_channels

    N_v2, C_v2, H_v2, W_v2 = 1, 512, 14, 14  # Input: batch_size, channels, height, width
    K_v2, C_out_v2 = 3, 512             # Kernel: size, output_channels
    
    input = torch.rand(N, C, H, W, device='cuda')
    filters = torch.rand(C_out, C, K, K, device='cuda')

    input_v2 = torch.rand(N_v2, C_v2, H_v2, W_v2, device='cuda')
    filters_v2 = torch.rand(C_out_v2, C_v2, K_v2, K_v2, device='cuda')
   
    input_half = input.half()
    filters_half = filters.half()

    input_half_v2 = input_v2.half()
    filters_half_v2 = filters_v2.half()
    
    H_out = H - K + 1
    W_out = W - K + 1

    H_out_v2 = H_v2 - K_v2 + 1
    W_out_v2 = W_v2 - K_v2 + 1
    
    output_v1 = torch.zeros(N, C_out, H_out, W_out, device='cuda', dtype=torch.float16)
    output_v2 = torch.zeros(N_v2, C_out_v2, H_out_v2, W_out_v2, device='cuda', dtype=torch.float16)
    output_cudnn_v1 = torch.zeros(N, C_out, H_out, W_out, device='cuda', dtype=torch.float16)
    output_cudnn_v2 = torch.zeros(N_v2, C_out_v2, H_out_v2, W_out_v2, device='cuda', dtype=torch.float16)
    
    output_v1, conv_v1_time = time_conv(conv.conv2d_v1, input_half, filters_half, output_v1)
    output_v2, conv_v2_time = time_conv(conv.conv2d_v2, input_half_v2, filters_half_v2, output_v2)
    output_cudnn_v1, cudnn_v1_time = time_cudnn_conv(cudnn_conv.conv2d_cudnn_v1, input_half, filters_half, output_cudnn_v1)
    output_cudnn_v2, cudnn_v2_time = time_cudnn_conv(cudnn_conv.conv2d_cudnn_v2, input_half_v2, filters_half_v2, output_cudnn_v2)
    
    output_ref = conv2d_ref(input, filters)
    output_ref_v2 = conv2d_ref(input_v2, filters_v2)
    
    OPS_PER_ELEMENT = K * K * C * 2 
    TOTAL_OPS = N * C_out * H_out * W_out * OPS_PER_ELEMENT
    NUM_GFLOPS = TOTAL_OPS / 1e9
    
    OPS_PER_ELEMENT_V2 = K_v2 * K_v2 * C_v2 * 2
    TOTAL_OPS_V2 = N_v2 * C_out_v2 * H_out_v2 * W_out_v2 * OPS_PER_ELEMENT_V2
    NUM_GFLOPS_V2 = TOTAL_OPS_V2 / 1e9
    
    output_ref = output_ref.half()
    output_ref_v2 = output_ref_v2.half()

    print('---BENCHMARKING CONV2D LAYER1 PARAMETERS---')
    print('Conv2d_v1 TEST CHECK:', torch.allclose(output_ref, output_v1, rtol=1e-03, atol=1e-05))
    print(f"Conv2d_v1 time: {conv_v1_time:.4f} ms")
    print(f"Conv2d_v1 GFLOPS/s: {(NUM_GFLOPS/(conv_v1_time*1e-3)):.4f}\n")
    print()

    print('---BENCHMARKING CONV2D LAYER2 PARAMETERS---')
    print('Conv2d_v2 TEST CHECK:', torch.allclose(output_ref_v2, output_v2, rtol=1e-03, atol=1e-05))
    print(f"Conv2d_v2 time: {conv_v2_time:.4f} ms")
    print(f"Conv2d_v2 GFLOPS/s: {(NUM_GFLOPS_V2/(conv_v2_time*1e-3)):.4f}\n")
    print()

    print('---BENCHMARKING CUDNN CONV2D LAYER1 PARAMETERS---')
    print('CUDNN Conv2d_v1 TEST CHECK:', torch.allclose(output_ref, output_cudnn_v1, rtol=1e-03, atol=1e-05))
    print(f"CUDNN Conv2d_v1 time (kernel only): {cudnn_v1_time:.4f} ms")
    print(f"CUDNN Conv2d_v1 GFLOPS/s: {(NUM_GFLOPS/(cudnn_v1_time*1e-3)):.4f}\n")
    print()

    print('---BENCHMARKING CUDNN CONV2D LAYER2 PARAMETERS---')
    print('CUDNN Conv2d_v2 TEST CHECK:', torch.allclose(output_ref_v2, output_cudnn_v2, rtol=1e-03, atol=1e-05))
    print(f"CUDNN Conv2d_v2 time (kernel only): {cudnn_v2_time:.4f} ms")
    print(f"CUDNN Conv2d_v2 GFLOPS/s: {(NUM_GFLOPS_V2/(cudnn_v2_time*1e-3)):.4f}\n")
    
    # Speedup comparison
    print("\n---SPEEDUP COMPARISON---")
    print(f"Layer 1 - Mine vs CUDNN: {cudnn_v1_time/conv_v1_time:.2f}x speedup")
    print(f"Layer 2 - Mine vs CUDNN: {cudnn_v2_time/conv_v2_time:.2f}x speedup")
