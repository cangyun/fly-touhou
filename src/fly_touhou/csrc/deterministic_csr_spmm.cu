// Each output has four fixed edge subsequences and one fixed reduction.
// No atomic updates. Order does not depend on warp/block scheduling or batch.
extern "C" __global__ void deterministic_csr_spmm(
    const long long* __restrict__ crow,
    const long long* __restrict__ columns,
    const float* __restrict__ weights,
    const float* __restrict__ activity_NB,
    float* __restrict__ output_NB,
    const int batch,
    const int rows) {
    const int row=blockIdx.x;
    const int lane=threadIdx.x & 31;
    const int part=threadIdx.x >> 5;
    const int b=blockIdx.y*32+lane;
    __shared__ float partial[128];
    float accumulator=0.0f;
    if (row<rows && b<batch) {
        const long long end=crow[row+1];
        for (long long e=crow[row]+part;e<end;e+=4) {
            const float x=activity_NB[columns[e]*batch+b];
            accumulator=__fadd_rn(accumulator,__fmul_rn(weights[e],x));
        }
    }
    partial[part*32+lane]=accumulator;
    __syncthreads();
    if (part==0 && row<rows && b<batch) {
        float total=__fadd_rn(partial[lane],partial[32+lane]);
        total=__fadd_rn(total,partial[64+lane]);
        total=__fadd_rn(total,partial[96+lane]);
        output_NB[(long long)row*batch+b]=total;
    }
}

// Four adjacent lanes compute one row. Each lane retains its original
// subsequence, including multiplication by zero for nonfinite weights.
extern "C" __global__ void deterministic_csr_single(
    const long long* __restrict__ crow,
    const long long* __restrict__ columns,
    const float* __restrict__ weights,
    const float* __restrict__ activity,
    float* __restrict__ output,
    const int batch,
    const int rows) {
    const int row = blockIdx.x * 32 + (threadIdx.x >> 2);
    const int part = threadIdx.x & 3;
    float sum = 0.0f;
    if (row < rows) {
        const long long end = crow[row + 1];
        for (long long e = crow[row] + part; e < end; e += 4) {
            sum = __fadd_rn(sum, __fmul_rn(weights[e], activity[columns[e]]));
        }
    }
    const float second = __shfl_sync(0xffffffffu, sum, 1, 4);
    const float third = __shfl_sync(0xffffffffu, sum, 2, 4);
    const float fourth = __shfl_sync(0xffffffffu, sum, 3, 4);
    if (part == 0 && row < rows) {
        output[row] = __fadd_rn(__fadd_rn(__fadd_rn(sum, second), third), fourth);
    }
}

// Reuse each edge's index and weight across four tiles of 32 candidates.
// Each candidate still sums the same four original edge subsequences.
extern "C" __global__ void deterministic_csr_128(
    const long long* __restrict__ crow,
    const long long* __restrict__ columns,
    const float* __restrict__ weights,
    const float* __restrict__ activity,
    float* __restrict__ output,
    const int batch,
    const int rows) {
    const int row = blockIdx.x;
    const int lane = threadIdx.x & 31;
    const int part = threadIdx.x >> 5;
    float sum[4] = {};
    const long long end = crow[row + 1];
    for (long long e = crow[row] + part; e < end; e += 4) {
        const long long first = columns[e] * 128 + lane;
        const float weight = weights[e];
        #pragma unroll
        for (int tile = 0; tile < 4; ++tile) {
            sum[tile] = __fadd_rn(sum[tile], __fmul_rn(weight, activity[first + 32 * tile]));
        }
    }
    __shared__ float partial[4][128];
    #pragma unroll
    for (int tile = 0; tile < 4; ++tile) {
        partial[tile][threadIdx.x] = sum[tile];
    }
    __syncthreads();
    if (part == 0) {
        #pragma unroll
        for (int tile = 0; tile < 4; ++tile) {
            float total = __fadd_rn(partial[tile][lane], partial[tile][32 + lane]);
            total = __fadd_rn(total, partial[tile][64 + lane]);
            output[(long long)row * 128 + 32 * tile + lane] = __fadd_rn(total, partial[tile][96 + lane]);
        }
    }
}

// Pack the neighbour index and the current signed FP32 weight into one
// aligned load for the single-candidate replay. Each lane still visits the
// original edge subsequence; even 0 * Inf is evaluated as in general CSR.
extern "C" __global__ void deterministic_csr_history(
    const long long* __restrict__ crow,
    const int2* __restrict__ edges,
    const float* __restrict__ activity,
    float* __restrict__ output,
    const int rows) {
    const int row = blockIdx.x * 32 + (threadIdx.x >> 2);
    const int part = threadIdx.x & 3;
    float sum = 0.0f;
    if (row < rows) {
        const long long end = crow[row + 1];
        for (long long e = crow[row] + part; e < end; e += 4) {
            const int2 edge = edges[e];
            sum = __fadd_rn(sum, __fmul_rn(__int_as_float(edge.y), activity[edge.x]));
        }
    }
    const float second = __shfl_sync(0xffffffffu, sum, 1, 4);
    const float third = __shfl_sync(0xffffffffu, sum, 2, 4);
    const float fourth = __shfl_sync(0xffffffffu, sum, 3, 4);
    if (part == 0 && row < rows) {
        output[row] = __fadd_rn(__fadd_rn(__fadd_rn(sum, second), third), fourth);
    }
}
