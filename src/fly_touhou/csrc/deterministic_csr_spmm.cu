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
