#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

/*
Forward kernel for multi-headed flash attention.
Q: input tensor of shape (batch_size, num_heads, seq_len, d_query)
K: input tensor of shape (batch_size, num_heads, seq_len, d_query)
V: input tensor of shape (batch_size, num_heads, seq_len, d_value)
O: output tensor of shape (batch_size, num_heads, seq_len, d_value)
l: row-wise exponential sum (used for softmax)
m: row-maximum (used for softmax)
B: batch size
H: number of heads
*/
__global__ void forward_kernel(const float* Q, const float* K, const float* V, float* O, float* l, float* m,
                    const int B, const int H, const int N, const int d,
                    const int Tc, const int Tr, const int Bc, const int Br, const float innerprod_scale) {

    /*
    b: batch index
    h: head index (in multi-head attention)
    threadIdx.x (tx): iterating over rows of a Q-tile
    threadIdx.y (ty): iterating over rows of a K/V-tile
    gridDim.x (bz): batch size
    gridDim.y (nh): total # of heads

    Parallelism scheme:
    Each block handles the attention computation for a specific batch element and a specific attention head.
    The local computation we deal with is Q[batch_idx][head_idx] @ K[batch_idx][head_idx]^T.

    Let's denote this as Q_i @ K_i^T, where Q_i has shape (seq_len, d_query) and K_i has shape (seq_len, d_query). This reduces
    to a matmul followed by a normalization operation (/sqrt(d_query)) and a softmax before a matmul with V_i:

    A_i (seq_len, seq_len) @ V_i (seq_len, d_value) => O_i (seq_len, d_value)
    */
    int b = blockIdx.x;    
    int h = blockIdx.y;    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bz = gridDim.x;    
    int nh = gridDim.y;    

    extern __shared__ float sram[]; // size specified in <<gridDim, blockDim, sharedMemSize>>

    /*
    Shared memory layout:
    [Q_tile (Br, query_dim), K_tile (Bc, query_dim), V_tile (Bc, query_dim), Sij (Br, Bc)]

    We compute a small partial product of the final attention pattern per thread. Br queries * Bc keys of it, in fact.
    We store all of this data contiguously the shared memory.
    */
    float* Qi = sram; // tile of Q matrix
    float* Kj = &sram[Br*d]; // tile of K matrix
    float* Vj = &sram[Br*d + Bc*d]; // tile of V matrix
    float* Sij = &sram[2*Br*d + 2*Bc*d + 2*Br]; // Q_i * K_j^T
    const int bnhNd = b * nh * N * d;
    const int hNd = h * N * d;

    /*
    Legacy
    //float* Oi = &sram[Br*d + 2*Bc*d];
    //float* li = &sram[2*Br*d + 2*Bc*d];
    //float* mi = &sram[2*Br*d + 2*Bc*d + Br];
    //float* Sij = &sram[Br*d + 2*Bc*d];
    */

    for(int j = 0; j < Tc; j++) {
        // Load Kj, Vj into shared memory
        for(int k = 0; k < d; k++){
            Kj[ty*d + k] = K[bnhNd + hNd + j*Bc*d + (ty*d + k)];
            Vj[ty*d + k] = V[bnhNd + hNd + j*Bc*d + (ty*d + k)];
        }
        __syncthreads();
        
        // Load Qi into shared memory
        for(int i = 0; i < Tr; i++) {
            for(int k = 0; k < d; k++){
                Qi[tx*d + k] = Q[bnhNd + hNd + i*Br*d + (tx*d + k)];
            }
            __syncthreads();

            // Load l, m into shared memory
            float li_old = l[b*nh*N + h*N + i*Br + tx];
            float mi_old = m[b*nh*N + h*N + i*Br + tx];

            // Compute QK^T
            float inner_prod = 0.0f;
            for(int k = 0; k < d; k++){
                inner_prod += Qi[tx*d + k] * Kj[ty*d + k];
            }
            // inner_prod *= innerprod_scale;
            Sij[tx*Bc + ty] = inner_prod;
            __syncthreads();

            // Compute mij 
            float mij = -INFINITY;

            
            for(int jj = 0; jj < Bc; jj++) {
                mij = fmaxf(mij, Sij[tx*Bc + jj]);
            }
            __syncthreads();
            

            // Compute Pij (safe softmax)
            Sij[tx*Bc + ty] = __expf(Sij[tx*Bc + ty] - mij);
            __syncthreads();


            float lij = 0.0f;
            for(int jj = 0; jj < Bc; jj++){
                float val = Sij[tx*Bc+jj];
                lij += val;
            }
            
            __syncthreads();
            
            // Compute mi_new, li_new
            float mi_new = max(mi_old, mij);
            float li_new = __expf(mi_old - mi_new) * li_old + __expf(mij - mi_new) * lij;

            // Write to O
            for(int k = 0; k < d; k++){
                float PijVj = 0.0f;
                for(int jj = 0; jj < Bc; jj++){
                    PijVj += Sij[tx*Bc + jj] * Vj[jj*d + k];
                }

                if(i*Br*d + tx*d < N*d){
                    if(ty == 0){ // make sure ty don't override each other
                    O[bnhNd + hNd + i*Br*d + (tx*d + k)] = 
                        (1.0f / li_new) * (li_old * expf(mi_old - mi_new) * O[bnhNd + hNd + i*Br*d + (tx*d + k)] + 
                                        expf(mij - mi_new) * PijVj);
                    }
                    // Write to l, m
                    l[b*nh*N + h*N + i*Br + tx] = li_new;
                    m[b*nh*N + h*N + i*Br + tx] = mi_new;
                }
            }
        }
        __syncthreads();
    }
}

torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O, torch::Tensor l, torch::Tensor m, const int bz, const int nh, const int N, const int d) {
    int Br = 16;
    int Bc = 16;

    const int Tr = (N + Br - 1) / Br;
    const int Tc = (N + Bc - 1) / Bc;
    const float dot_prod_scale = 1.0f/sqrt(d);

    dim3 grid_size(bz, nh);
    dim3 block_size(Br, Bc);
    const int shared_mem_size = (2*Br*d + 2*Bc*d + 2*Br + Br*Bc) * sizeof(float);

    forward_kernel<<<grid_size, block_size, shared_mem_size>>>(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        O.data_ptr<float>(),
        l.data_ptr<float>(),
        m.data_ptr<float>(),
        bz, nh, N, d, Tc, Tr, Bc, Br, dot_prod_scale
    );
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Flash Attention forward");
}


