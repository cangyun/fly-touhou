// Four fixed edge subsequences and the deterministic CSR reduction order.
// Each y-tile packs up to 32 candidate spike trains.
extern "C" __global__ void pack_candidates(const float* __restrict__ spikes,
        unsigned* __restrict__ bits, int batch, int rows) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const int first = blockIdx.y * 32;
    const int count = min(32, batch - first);
    unsigned mask = 0;
    for (int b = 0; b < count; ++b) {
        mask |= (unsigned)(spikes[(long long)(first + b) * rows + row] != 0.f) << b;
    }
    bits[(long long)blockIdx.y * rows + row] = mask;
}

__device__ bool nonfinite(float value) {
    return (__float_as_uint(value) & 0x7f800000u) == 0x7f800000u;
}

template<bool finite_weights>
__device__ void binary_csr(const long long* __restrict__ ptr,
        const long long* __restrict__ col, const float* __restrict__ weight,
        const unsigned* __restrict__ flags, float* __restrict__ out, int batch, int rows) {
    const int row = blockIdx.x, lane = threadIdx.x & 31, part = threadIdx.x >> 5;
    const int b = blockIdx.y * 32 + lane;
    const unsigned* bits = flags + (long long)blockIdx.y * rows;
    const unsigned full = 0xffffffffu;
    const long long stop = ptr[row + 1];
    float accum = 0.f;
    for (long long base = ptr[row] + part; base < stop; base += 128) {
        const long long e = base + lane * 4;
        const unsigned mask = e < stop ? bits[col[e]] : 0u;
        const float w = (finite_weights ? mask != 0u : e < stop) ? weight[e] : 0.f;
        unsigned remaining = __ballot_sync(full, mask || (!finite_weights && nonfinite(w)));
        while (remaining) {
            const int source = __ffs(remaining) - 1;
            const unsigned m = __shfl_sync(full, mask, source);
            const float v = __shfl_sync(full, w, source);
            if ((m >> lane) & 1u) accum = __fadd_rn(accum, v);
            else if (!finite_weights && nonfinite(v))
                accum = __fadd_rn(accum, __fmul_rn(v, 0.f));
            remaining &= remaining - 1;
        }
    }
    __shared__ float partial[128];
    partial[threadIdx.x] = accum;
    __syncthreads();
    if (part == 0 && b < batch) {
        float total = __fadd_rn(partial[lane], partial[32 + lane]);
        total = __fadd_rn(total, partial[64 + lane]);
        out[(long long)b * rows + row] = __fadd_rn(total, partial[96 + lane]);
    }
}

// Frozen inference uses the finite-weight specialization.
extern "C" __global__ void spike_csr(const long long* ptr, const long long* col,
        const float* weight, const unsigned* bits, float* out, int batch, int rows) {
    binary_csr<true>(ptr, col, weight, bits, out, batch, rows);
}

// Training evaluates 0 * NaN/Inf exactly as the general FP32 CSR path.
extern "C" __global__ void spike_csr_training(const long long* ptr, const long long* col,
        const float* weight, const unsigned* bits, float* out, int batch, int rows) {
    binary_csr<false>(ptr, col, weight, bits, out, batch, rows);
}
