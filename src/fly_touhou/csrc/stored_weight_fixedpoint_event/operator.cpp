// NEW round-once stored-W protocol. Source only; NOT compiled in preparation.
#include <torch/extension.h>

torch::Tensor fixed_event_forward_cuda(torch::Tensor spikes, torch::Tensor offsets,
                                      torch::Tensor targets, torch::Tensor fixed);

torch::Tensor fixed_event_forward(torch::Tensor spikes, torch::Tensor offsets,
                                 torch::Tensor targets, torch::Tensor fixed) {
    TORCH_CHECK(spikes.is_cuda() && offsets.is_cuda() && targets.is_cuda() && fixed.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(spikes.device() == offsets.device() && spikes.device() == targets.device() && spikes.device() == fixed.device(), "One explicit device required");
    TORCH_CHECK(spikes.scalar_type() == torch::kFloat32 && offsets.scalar_type() == torch::kInt64 &&
                targets.scalar_type() == torch::kInt32 && fixed.scalar_type() == torch::kInt64, "FP32/int64/int32/int64 contract");
    TORCH_CHECK(spikes.dim() == 2 && spikes.size(0) > 0 && offsets.dim() == 1 && targets.dim() == 1 && fixed.dim() == 1, "Original tensor ranks required");
    TORCH_CHECK(offsets.numel() == spikes.size(1) + 1 && targets.numel() == fixed.numel(), "Bound graph dimensions differ");
    TORCH_CHECK(spikes.is_contiguous() && offsets.is_contiguous() && targets.is_contiguous() && fixed.is_contiguous(), "Contiguous device buffers required");
    return fixed_event_forward_cuda(spikes, offsets, targets, fixed);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fixed_event_forward, "Explicit stored-W fixedpoint round-once event current");
}
