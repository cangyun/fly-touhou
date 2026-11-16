// Up to 32 candidate spike trains share each edge read. The four
// subsequences and FP32 addition order are the deployed deterministic CSR.
extern "C" __global__ void pack_candidates(const float* __restrict__ spikes,
        unsigned* __restrict__ bits, int batch, int rows) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows) return;
    unsigned mask = 0;
    for (int b=0; b<batch; ++b)
        if (spikes[(long long)b*rows+i] != 0.f) mask |= 1u << b;
    bits[i] = mask;
}
extern "C" __global__ void spike_csr(const long long* __restrict__ ptr,
        const long long* __restrict__ col, const float* __restrict__ weight,
        const unsigned* __restrict__ bits, float* __restrict__ out, int batch, int rows) {
    const int row=blockIdx.x, lane=threadIdx.x & 31, part=threadIdx.x>>5;
    const long long stop=ptr[row+1];
    float accum=0.f;
    for(long long base=ptr[row]+part; base<stop; base+=128) {
        long long e=base+lane*4;
        unsigned mask=e<stop ? bits[col[e]] : 0u;
        float w=mask ? weight[e] : 0.f;
        unsigned remaining=__ballot_sync(0xffffffffu,mask!=0u);
        while(remaining) {
            int j=__ffs(remaining)-1;
            unsigned m=__shfl_sync(0xffffffffu,mask,j);
            float v=__shfl_sync(0xffffffffu,w,j);
            if ((m>>lane)&1u) accum=__fadd_rn(accum,v);
            remaining &= remaining-1;
        }
    }
    __shared__ float partial[128];
    partial[part*32+lane]=accum;
    __syncthreads();
    if(part==0 && lane<batch) {
        float total=__fadd_rn(partial[lane],partial[32+lane]);
        total=__fadd_rn(total,partial[64+lane]);
        out[(long long)lane*rows+row]=__fadd_rn(total,partial[96+lane]);
    }
}
