// Original-order first-order SDDMM. One warp reduces one existing edge.
// This changes no nodes, connections, signs, neuron dynamics or loss terms.
extern "C" __global__ void fused_edge_gradient(
    const float* __restrict__ activity_transposed,
    const float* __restrict__ gradient_transposed,
    const long long* __restrict__ pre,
    const long long* __restrict__ post,
    float* __restrict__ output,
    const int batch,
    const long long edges) {
    const long long thread = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long edge = thread >> 5;
    if (edge >= edges) return;
    const int lane = threadIdx.x & 31;
    const long long source = pre[edge] * batch;
    const long long target = post[edge] * batch;
    float value = 0.0f;
    for (int b = lane; b < batch; b += 32) {
        value = __fadd_rn(value, __fmul_rn(activity_transposed[source+b], gradient_transposed[target+b]));
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        value = __fadd_rn(value, __shfl_down_sync(0xffffffffu, value, offset));
    if (lane == 0) output[edge] = value;
}
