import math
import numpy as np
import matplotlib.pyplot as plt

peak_compute = 35.58 * 10**12  
dram_bandwidth = 936.2 * 10**9 
l2_bandwidth = 1500.0 * 10**9 
l1_bandwidth = 3000.0 * 10**9 
l1_size = 128 * 1024
l2_size = 6 * 1024 * 1024 

# some notes:
# simple roofline model performs so badly because it assumes everything is in DRAM but fa1 computes and loads everything in l1
# for l1 cache/shared mem, i was very conservative and assumed no coalescing, alot of bank conflicts
# also for our model, for each memory hiearchy, i assumed low utilization of each memory level

### Functions for our model ###        
def compute_flops(B, H, N, d, Br, Bc):
    """Compute total FLOPS for flash attention"""
    Tr = math.ceil(N / Br)
    Tc = math.ceil(N / Bc)
    flops_qk = 0
    flops_mij = 0
    flops_pij = 0
    flops_lij = 0
    flops_mi_new = 0
    flops_li_new = 0
    flops_pijv = 0
    flops_oi = 0
    
    for i in range(Tr):
        for j in range(Tc):
            flops_qk += 2 * Br * Bc * d 
            flops_mij += Br * Bc 
            flops_pij += 2 * Br * Bc
            flops_lij += Br * Bc
            flops_mi_new += Br 
            flops_li_new += 6 * Br
            flops_pijv += 2 * Br * Bc * d
            flops_oi += 10 * Br * d
    
    total_flops = B * H * (flops_qk + flops_mij + flops_pij + 
                            flops_lij + flops_mi_new + flops_li_new + 
                            flops_pijv + flops_oi) * 3 
    
    return total_flops

def compute_memory_accesses(B, H, N, d, Br, Bc):
    """Compute memory accesses (bytes) at each level of the memory hierarchy"""
    Tr = math.ceil(N / Br)
    Tc = math.ceil(N / Bc)
    bytes_per_element = 4 

    dram_read_qkv = 3 * (B * H * N * d * bytes_per_element)
    dram_read_lm = 2 * (B * H * N * bytes_per_element)
    dram_write_o = B * H * N * d * bytes_per_element
    dram_write_lm = 2 * B * H * N * bytes_per_element
    dram_total = dram_read_qkv + dram_read_lm + dram_write_o + dram_write_lm
    
    l2_total = dram_total
    
    l1_accesses = 0
    num_thread_blocks = B * H
    
    for j in range(Tc):
        # Load Kj, Vj to shared memory
        l1_accesses += 2 * Bc * d * bytes_per_element
        
        for i in range(Tr):
            # Load Qi to shared memory
            l1_accesses += Br * d * bytes_per_element
            
            # Read from shared memory, Qi, Kj for QK^T
            l1_accesses += (Br * d) * 2 * bytes_per_element
            
            # Write Sij to shared memory
            l1_accesses += Br * Bc * bytes_per_element
            
            # Read Sij for mij computation
            l1_accesses += Br * Bc * bytes_per_element
            
            # Read/write Sij for Pij computation
            l1_accesses += Br * Bc * 2 * bytes_per_element
            
            # Read Sij for lij computation
            l1_accesses += Br * Bc * bytes_per_element
            
            # Read Sij and Vj for Oi
            l1_accesses += Br * Bc * bytes_per_element + Bc * d * bytes_per_element
           
    # No coalescing for L1 cache
    uncoalesced_factor = 32 
    l1_total = l1_accesses * num_thread_blocks * uncoalesced_factor
   
    return dram_total, l2_total, l1_total 
    
def compute_operational_intensity(B, H, N, d, Br, Bc):
    """Calculate operational intensity (FLOPS/byte) at each memory level"""
    flops = compute_flops(B, H, N, d, Br, Bc)
    dram_mem, l2_mem, l1_mem = compute_memory_accesses(B, H, N, d, Br, Bc)
    
    dram_intensity = flops / dram_mem
    l2_intensity = flops / l2_mem
    l1_intensity = flops / l1_mem
    
    print(f"dram_mem: {dram_mem:,}")
    print(f"l2_mem: {l2_mem:,}")
    print(f"l1_mem: {l1_mem:,}")
    
    # flops = 27,7772,351,685.08 * 0.0006451 = 17,417.7 flops
    
    
    #l1_mem = 17,417.7 / 4.60 = 3786.45 bytes
    #l2_mem =  17,417.7 / 48.61 = 358 bytes
    #dram_mem =  17,417.7  / 117.49 = 148.24 bytes

    return dram_intensity, l2_intensity, l1_intensity
    
def compute_execution_times(B, H, N, d, Br, Bc):
    """Calculate execution times for each potential bottleneck"""
    flops = compute_flops(B, H, N, d, Br, Bc)
    dram_total, l2_total, l1_total = compute_memory_accesses(B, H, N, d, Br, Bc)

    # Bank conflict factor for L1 cache/shared memory
    bank_conflict_factor = 6.5

    # Our model
    compute_time = (flops / peak_compute) * 1000
    our_dram_time = (dram_total / (dram_bandwidth * 0.01)) * 1000
    l2_time = (l2_total / (l2_bandwidth * 0.01)) * 1000
    l1_time = (l1_total / (l1_bandwidth * 0.1)) * bank_conflict_factor * 1000

    # Simple roofline model (assume peak bandwith)
    compute_time = (flops / peak_compute) * 1000
    roofline_dram_time = ((dram_total * 3) / (dram_bandwidth)) * 1000
    
    times = [compute_time, our_dram_time, l2_time, l1_time]
    bottleneck_time = max(times)
    simple_roofline_time = min(compute_time, roofline_dram_time)
    bottlenecks = ["Compute", "DRAM", "L2 Cache", "L1 Cache"]
    bottleneck = bottlenecks[times.index(max(times))]
    
    return compute_time, our_dram_time, l2_time, l1_time, bottleneck_time, simple_roofline_time, bottleneck
   

### Plotting functions ### 
def graph_tiles_sizes(B, H, N, d):
    tile_configs = [
        (16, 16),
        (32, 32),
        (64, 16),
        (16, 64)
    ]
    
    labels = ["16x16", "32x32", "64x16", "16x64"]
    predicted_times = []
    roofline_times = []  
    
    for Br, Bc in tile_configs:
        _, _, _, _, bottleneck_time, roofline_time, _ = compute_execution_times(B, H, N, d, Br, Bc)
        predicted_times.append(bottleneck_time)
        roofline_times.append(roofline_time)
    
    measured_times = [2.2569, 11.7762, 3.8963, 11.5999]
    
    plt.figure(figsize=(10, 6))
    
    x = np.arange(len(tile_configs))
    width = 0.25 
    
    plt.bar(x - width, predicted_times, width, label='Our Model')
    plt.bar(x, roofline_times, width, label='Simple Roofline Model')
    plt.bar(x + width, measured_times, width, label='Measured Time')
    
    plt.xlabel('Tile Configuration (Br x Bc)')
    plt.ylabel('Execution Time (ms)')
    plt.title('Flash Attention Performance with Different Tile Sizes (N=128)')
    plt.xticks(x, labels)
    plt.grid(True, axis='y', linestyle='--', linewidth=0.5)
    plt.legend()
    
    for i, v in enumerate(predicted_times):
        plt.text(i - width, v + 0.1, f"{v:.2f}", ha='center', fontsize=8)
    
    for i, v in enumerate(roofline_times):
        plt.text(i, v + 0.1, f"{v:.2f}", ha='center', fontsize=8)
    
    for i, v in enumerate(measured_times):
        plt.text(i + width, v + 0.1, f"{v:.2f}", ha='center', fontsize=8)
        
    plt.savefig('graphs/tile_sizes.png', dpi=300, bbox_inches='tight')
    plt.show()
    
def graph_sequence_lengths(B, H, d):
    """Validate model against specific sequence lengths from measurements"""
    N_values = [64, 128, 256, 512]
    
    measured_times = [0.6451, 2.2508, 8.7235, 34.5702]
    
    our_model_times = []
    roofline_times = []
    
    for N in N_values:
        compute_time, dram_time, l2_time, l1_time, bottleneck_time, roofline_time, _ = compute_execution_times(B, H, N, d, 32, 32)
        our_model_times.append(bottleneck_time)
        roofline_times.append(roofline_time)
    
    our_model_errors = [100 * abs(p - m) / m for p, m in zip(our_model_times, measured_times)]
    roofline_errors = [100 * abs(p - m) / m for p, m in zip(roofline_times, measured_times)]
    
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 1, 1)
    plt.loglog(N_values, measured_times, 'ko-', linewidth=2, label='Measured')
    plt.loglog(N_values, our_model_times, 'bs-', linewidth=2, label='Our Model')
    plt.loglog(N_values, roofline_times, 'rd-', linewidth=2, label='Roofline Model')
    
    for i, N in enumerate(N_values):
        plt.annotate(f"N={N}\n{measured_times[i]:.2f}ms", 
                     xy=(N, measured_times[i]), 
                     xytext=(0, 10),
                     textcoords="offset points",
                     ha='center')
    
    plt.xlabel('Sequence Length (N)')
    plt.ylabel('Time (ms)')
    plt.title('Flash Attention Execution Time vs. Sequence Length')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.semilogx(N_values, our_model_errors, 'bs-', linewidth=2, label='Our Model Error')
    plt.semilogx(N_values, roofline_errors, 'rd-', linewidth=2, label='Roofline Model Error')
    plt.axhline(y=10, color='gray', linestyle='--', label='10% Error Threshold')
    
    for i, N in enumerate(N_values):
        plt.annotate(f"{our_model_errors[i]:.1f}%", 
                     xy=(N, our_model_errors[i]), 
                     xytext=(0, 10),
                     textcoords="offset points",
                     ha='center')
        plt.annotate(f"{roofline_errors[i]:.1f}%", 
                     xy=(N, roofline_errors[i]), 
                     xytext=(0, -15),
                     textcoords="offset points",
                     ha='center')
    
    plt.xlabel('Sequence Length (N)')
    plt.ylabel('Error (%)')
    plt.title('Model Error Rates vs. Sequence Length')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('graphs/seq_lengths.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n===== varying sequences lengths error =====")
    print("Average error for our model: {:.2f}%".format(sum(our_model_errors)/len(our_model_errors)))
    print("Average error for roofline model: {:.2f}%".format(sum(roofline_errors)/len(roofline_errors)))

if __name__ == "__main__":
    B = 1 
    H = 4 
    d = 32
    
    actual_time = [0.6451, 2.2508, 8.7235, 34.5702]
    N_values = [64, 128, 256, 512]
    
    print("\ntesting various sequence lengths when (BR, BC) = (32,32)")
    for N in N_values:
        print(f"\nAnalyzing N={N}:")
        flops = compute_flops(B, H, N, d, 16, 16)
        print(f"Total FLOPS: {flops:,}")
        dram_intensity, l2_intensity, l1_intensity = compute_operational_intensity(B, H, N, d, 16, 16)
        print(f"DRAM intensity: {dram_intensity:.2f} FLOPS/byte")
        print(f"L2 intensity: {l2_intensity:.2f} FLOPS/byte")
        print(f"L1 intensity: {l1_intensity:.2f} FLOPS/byte")

        compute_time, dram_time, l2_time, l1_time, bottleneck_time, roofline_time, bottleneck = compute_execution_times(B, H, N, d, 32, 32)
        print(f"(Our model) Predicted execution time: {bottleneck_time:.6f} ms (Bottleneck: {bottleneck})")
        print(f"(Simple roofline model) Predicted execution time: {roofline_time:.6f} ms")
        print(f"(Actual execution time: {actual_time[N_values.index(N)]:.6f} ms)")
    
    N = 64 
    graph_tiles_sizes(B, H, N, d)
    graph_sequence_lengths(B, H, d)