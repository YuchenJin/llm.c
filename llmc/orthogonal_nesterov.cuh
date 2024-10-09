#ifndef ORTHOGONAL_NESTEROV_CUH
#define ORTHOGONAL_NESTEROV_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "cuda_common.h"
#include "cuda_utils.cuh"

// OrthogonalNesterov optimizer structure
typedef struct {
    float lr;
    float momentum;
    int zeropower_iters;
} OrthogonalNesterov;

// Initialize OrthogonalNesterov optimizer
__host__ void orthogonal_nesterov_init(OrthogonalNesterov* opt, float lr, float momentum, int zeropower_iters) {
    opt->lr = lr;
    opt->momentum = momentum;
    opt->zeropower_iters = zeropower_iters;
}

// Helper function to convert float to floatX
__device__ __forceinline__ floatX to_floatX(float val) {
    return (floatX)val;
}

// CUDA kernel for Newton-Schulz iteration
__global__ void newton_schulz_kernel(floatX* X, const floatX* G, int n, int steps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        floatX x = X[idx];
        for (int i = 0; i < steps; i++) {
            floatX a = to_floatX(3.4445f) * x;
            floatX b = to_floatX(-4.7750f) * (x * x * x);
            floatX c = to_floatX(2.0315f) * (x * x * x * x * x);
            x = a + b + c;
        }
        X[idx] = x;
    }
}

// CUDA kernel for OrthogonalNesterov update
__global__ void orthogonal_nesterov_kernel(floatX* p, floatX* g, floatX* m, 
                                           float lr, float momentum, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        floatX buf = m[idx];
        buf = buf * to_floatX(momentum) + g[idx];
        floatX update = (momentum > 0.0f) ? 
            buf * to_floatX(1.0f + momentum) - m[idx] * to_floatX(momentum) : 
            buf;
        m[idx] = buf;
        p[idx] -= to_floatX(lr) * update;
    }
}

// Host function to perform OrthogonalNesterov step
__host__ void orthogonal_nesterov_step(OrthogonalNesterov* opt, floatX* p, floatX* g, floatX* m, 
                                       size_t n, cudaStream_t stream) {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;

    // Perform Newton-Schulz iteration
    newton_schulz_kernel<<<grid_size, block_size, 0, stream>>>(g, g, n, opt->zeropower_iters);

    // Perform OrthogonalNesterov update
    orthogonal_nesterov_kernel<<<grid_size, block_size, 0, stream>>>(p, g, m, opt->lr, opt->momentum, n);
}

#endif // ORTHOGONAL_NESTEROV_CUH
