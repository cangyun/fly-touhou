// Source-only candidate. No compile/GPU/capture qualification has occurred.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/macros/Macros.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

namespace {
__global__ void sum_binary_outgoing(const float* spikes, const int64_t* offsets,
        const int32_t* targets, const int64_t* fixed, int64_t* summed,
        int64_t n, int64_t size) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size) return;
    const float z = spikes[i];
    CUDA_KERNEL_ASSERT(z == 0.0f || z == 1.0f);
    if (z != 1.0f) return;
    const int64_t lane = i / n, source = i % n;
    for (int64_t edge = offsets[source]; edge < offsets[source + 1]; ++edge) {
        // Unsigned modular atomic represents signed two's-complement sums.
        // CPU proof: EVERY signed partial sum is inside(-2^63,2^63), independent
        // of event order. Negative unsigned encodings are intentional, not an
        // unproven signed-overflow assumption or a floating atomic reduction.
        atomicAdd(reinterpret_cast<unsigned long long*>(summed + lane * n + targets[edge]),
                  static_cast<unsigned long long>(fixed[edge]));
    }
}

__global__ void round_once_grid25(const int64_t* summed, float* current, int64_t size) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size) return;
    // Explicit RN integer conversion. Power-of-two scaling is exact here:
    // any nonzero value is >=2^-25 and all magnitudes <2^38, normal and finite.
    current[i] = ldexpf(__ll2float_rn(static_cast<long long>(summed[i])), -25);
}
}  // namespace

torch::Tensor fixed_event_forward_cuda(torch::Tensor spikes, torch::Tensor offsets,
                                      torch::Tensor targets, torch::Tensor fixed) {
    c10::cuda::CUDAGuard device_guard(spikes.device());
    const int64_t size = spikes.numel(), n = spikes.size(1);
    const int threads = 256;
    const int64_t blocks64 = (size + threads - 1) / threads;
    TORCH_CHECK(blocks64 <= 2147483647LL, "Finite supported event grid required");
    auto summed = torch::zeros(spikes.sizes(), spikes.options().dtype(torch::kInt64));
    auto current = torch::empty_like(spikes);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(spikes.get_device());
    sum_binary_outgoing<<<static_cast<unsigned int>(blocks64), threads, 0, stream>>>(
        spikes.data_ptr<float>(), offsets.data_ptr<int64_t>(), targets.data_ptr<int32_t>(),
        fixed.data_ptr<int64_t>(), summed.data_ptr<int64_t>(), n, size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    round_once_grid25<<<static_cast<unsigned int>(blocks64), threads, 0, stream>>>(
        summed.data_ptr<int64_t>(), current.data_ptr<float>(), size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    // No default-stream switch, cudaMalloc or explicit synchronization. PyTorch
    // allocation/current stream are intended for future CUDA graph capture;
    // actual capture correctness and int64 atomic performance are unmeasured.
    return current;
}
