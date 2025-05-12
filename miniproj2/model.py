import math

# Flops
# QK^T = 2 * Br * Bc * d
# mij = Br
# Pij = 2 * Br * Bc
# lij = Br
# mi_new = Br
# li_new = 6*Br
# PijVj = 2*Br*Bc*d
# O_i = 10 * Br * d
# Total FLOPS = 2 * Br * Bc * d + 2 * Br * Bc + 2 * Br + 6*Br + 2*Br*Bc*d + 10 * Br * d

num_sms = 82  
peak_compute = 35.58 * 10**12  

# Memory parameters
dram_bandwidth = 936 * 10**9 
l2_bandwidth = 2.5 * 10**12 
l1_bandwidth = 13.8 * 10**12 

# Memory capacities
l1_size = 128 * 1024 
l2_size = 6 * 1024 * 1024 
        
def compute_flops(B, H, N, d, Br, Bc):
    """Compute total FLOPS for flash attention"""
    Tr = math.ceil(N / Br)
    Tc = math.ceil(N / Bc)
    
    flops_qk = 2 * Br * Bc * d * Tr * Tc
    flops_mij = Br * Tr * Tc
    flops_pij = 2 * Br * Bc * Tr * Tc
    flops_lij = Br * Tr * Tc
    flops_mi_new = Br * Tr * Tc
    flops_li_new = 6 * Br * Tr * Tc
    flops_pijv = 2 * Br * Bc * d * Tr * Tc
    flops_oi = 10 * Br * d * Tr
    
    total_flops = B * H * (flops_qk + flops_mij + flops_pij + 
                            flops_lij + flops_mi_new + flops_li_new + 
                            flops_pijv + flops_oi)
    
    return total_flops

# Don't know if L2 and L1 is correct
def compute_memory_accesses(B, H, N, d, Br, Bc):
    """Compute memory accesses (bytes) at each level of the memory hierarchy"""
    Tr = math.ceil(N / Br)
    Tc = math.ceil(N / Bc)
    bytes_per_element = 4
    
    # DRAM level
    dram_read_qkvo = 4*(B * H * N * d * bytes_per_element)
    dram_read_lm = 2*(B * H * N * bytes_per_element)

    dram_write_o = B * H * N * d * bytes_per_element
    dram_write_lm = 2 * B * H * N * bytes_per_element
    dram_total = dram_read_qkvo + dram_read_lm + dram_write_o + dram_write_lm
    
    # L2 cache
    l2_read_qkv = B * H * Tr * Tc * Br*d + 2*Bc*d * bytes_per_element
    l2_write_o = B * H * N * d * bytes_per_element
    l2_total = l2_read_qkv + l2_write_o
    
    # L1/Shared memory
    
    #float* Qi = sram;
    #float* Kj = &sram[Br*d];
    #float* Vj = &sram[Br*d + Bc*d];
    #float* Sij = &sram[2*Br*d + 2*Bc*d + 2*Br]; 

    l1_qkv_load = B * H * Tr * Tc * Br + 2*Bc * d * bytes_per_element
    l1_intermediate = (B * H * Tr * Tc * (Br*d + Bc*d + Br*Bc)) * bytes_per_element
    l1_output = B * H * Tr * Bc * d * bytes_per_element
    l1_total = l1_qkv_load + l1_intermediate + l1_output
   
    return dram_total, l2_total, l1_total 
    
def compute_operational_intensity(B, H, N, d, Br, Bc):
    """Calculate operational intensity (FLOPS/byte) at each memory level"""
    flops = compute_flops(B, H, N, d, Br, Bc)
    dram_mem, l2_mem, l1_mem = compute_memory_accesses(B, H, N, d, Br, Bc)
    
    dram_intensity = flops / dram_mem
    l2_intensity = flops / l2_mem
    l1_intensity = flops / l1_mem

    return dram_intensity, l2_intensity, l1_intensity
    
def determine_bottleneck(compute_time, dram_time, l2_time, l1_time):
    """Determine the bottleneck in the system"""
    return max(compute_time, dram_time, l2_time, l1_time)

if __name__ == "__main__":
    B = 1 
    H = 4 
    N = 128
    d = 32
    Br = 32
    Bc = 32
   
    # TOps 
    flops = compute_flops(B, H, N, d, Br, Bc)
    print(f"Total FLOPS: {flops}")
   
    # Achieved Operational Intensity 
    dram_intensity, l2_intensity, l1_intensity = compute_operational_intensity(B, H, N, d, Br, Bc)
    print(f"\nOperational Intensity (FLOPS/byte):")
    print(f"  DRAM: {dram_intensity:.2f}")
    print(f"  L2: {l2_intensity:.2f}")
    print(f"  L1/SPAD: {l1_intensity:.2f}")
   
    # Bottleneck Analysis 
    dram_total, l2_total, l1_total = compute_memory_accesses(B, H, N, d, Br, Bc)

    compute_time = flops / peak_compute
    dram_time = dram_total / dram_bandwidth
    l2_time = l2_total / l2_bandwidth
    l1_time = l1_total / l1_bandwidth

    bottleneck = determine_bottleneck(compute_time, dram_time, l2_time, l1_time)
    print(f"Bottleneck time: {bottleneck:.2f} ms")

    
