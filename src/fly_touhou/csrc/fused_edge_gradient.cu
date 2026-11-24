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

__device__ bool nonfinite(float value) {
    return (__float_as_uint(value) & 0x7f800000u) == 0x7f800000u;
}

extern "C" __global__ void edge_tile_flags(
    const float* __restrict__ activity, const float* __restrict__ gradient,
    unsigned* __restrict__ activity_flags, unsigned* __restrict__ gradient_flags,
    const int batch, const int rows) {
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (row >= rows) return;
    const unsigned full = 0xffffffffu;
    const int lane = threadIdx.x & 31;
    const long long offset = (long long)row * batch;
    unsigned source = 0, target = 0;
    for (int base = 0, tile = 0; base < batch; base += 32, ++tile) {
        const int b = base + lane;
        const float a = b < batch ? activity[offset + b] : 0.f;
        const float g = b < batch ? gradient[offset + b] : 0.f;
        source |= (unsigned)__any_sync(full, a != 0.f) << tile;
        source |= (unsigned)__any_sync(full, nonfinite(a)) << (tile + 16);
        target |= (unsigned)__any_sync(full, g != 0.f) << tile;
        target |= (unsigned)__any_sync(full, nonfinite(g)) << (tile + 16);
    }
    if (lane == 0) {
        activity_flags[row] = source;
        gradient_flags[row] = target;
    }
}

// Assign 16 original edges to each warp. Zero-only tiles reduce to +0
// in the reference loop. A nonfinite operand always keeps the tile active.
extern "C" __global__ void masked_edge_gradient(
    const float* __restrict__ activity, const float* __restrict__ gradient,
    const long long* __restrict__ pre, const long long* __restrict__ post,
    const unsigned* __restrict__ activity_flags, const unsigned* __restrict__ gradient_flags,
    float* __restrict__ out, const int batch, const long long edges) {
    const int edges_per_warp = 16;
    const unsigned full = 0xffffffffu;
    const long long first = (((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5) * edges_per_warp;
    if (first >= edges) return;
    const int lane = threadIdx.x & 31;
    int source = 0, target = 0;
    unsigned live_tiles = 0;
    if (lane < edges_per_warp && first + lane < edges) {
        source = (int)pre[first + lane];
        target = (int)post[first + lane];
        const unsigned a = activity_flags[source], g = gradient_flags[target];
        live_tiles = ((a & g) & 0xffffu) | ((a | g) >> 16);
        if (!live_tiles) out[first + lane] = 0.f;
    }
    unsigned live_edges = __ballot_sync(full, live_tiles != 0);
    while (live_edges) {
        const int local = __ffs(live_edges) - 1;
        const long long from = (long long)__shfl_sync(full, source, local) * batch;
        const long long to = (long long)__shfl_sync(full, target, local) * batch;
        unsigned tiles = __shfl_sync(full, live_tiles, local);
        float sum = 0.f;
        while (tiles) {
            const int b = 32 * (__ffs(tiles) - 1) + lane;
            if (b < batch) sum = __fadd_rn(sum, __fmul_rn(activity[from + b], gradient[to + b]));
            tiles &= tiles - 1;
        }
        for (int offset = 16; offset > 0; offset >>= 1)
            sum = __fadd_rn(sum, __shfl_down_sync(full, sum, offset));
        if (lane == 0) out[first + local] = sum;
        live_edges &= live_edges - 1;
    }
}
