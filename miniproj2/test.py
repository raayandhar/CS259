import torch
import math
import torch.nn.functional as F
from torch.utils.cpp_extension import load

flash_attention = load(name='flash_attention', sources=['attention.cu'], extra_cuda_cflags=['-O3'])

def time_attn(func, *args, num_runs=1):
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

def manual_attn(q, k, v): 
#    att = (q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))))
    att = (q @ k.transpose(-2, -1))
    att = F.softmax(att, dim=-1)
    y = att @ v
    return y

def main():
    [64, 128, 256, 512, 1024, 2048]
    B = 1 
    H = 4 
    N = 64 
    d = 32 

    # 64:  3.0249 ms,  0.6451
    # 128:  11.7762 ms, 2.2508
    # 256:  46.0249 ms, 8.7235
    # 512:  186.5626 ms, 34.5702

    # Br, Bc, N=128
    # 16, 16: 2.2569 ms
    # 32, 32: 11.7762 ms
    # 64, 16: 3.8963
    # 16, 64: 11.5999 ms

    Q = torch.randn((B, H, N, d), device='cuda', dtype=torch.float32)
    K = torch.randn((B, H, N, d), device='cuda', dtype=torch.float32)
    V = torch.randn((B, H, N, d), device='cuda', dtype=torch.float32)
    O = torch.zeros((B, H, N, d), device='cuda', dtype=torch.float32)
    l = torch.zeros((B, H, N), device='cuda', dtype=torch.float32)
    m = torch.full((B, H, N), -float('inf'), device='cuda', dtype=torch.float32)

    print('Starting attention computations...')

    manual_result, manual_time = time_attn(manual_attn, Q, K, V)
    
    flash_result, flash_time = time_attn(flash_attention.forward, Q, K, V, O, l, m, B, H, N, d)

    print("\nFinal output match:", torch.allclose(manual_result, flash_result, rtol=0, atol=1e-02))
    print(f"Manual time: {manual_time:.4f}")
    print(f"Flash time: {flash_time:.4f}")
    print(f"\nSpeedup achieved: {(manual_time/flash_time):.2f}x")

if __name__ == "__main__":
    main()
