// One FP32 LIF step without intermediate whole-brain tensors between operations.
extern "C" __global__ void lif_tick_forward(
        const float* membrane, const float* synaptic, const float* previous_spikes,
        const float* rate, const float* drive, const float* message, const float* bias,
        const float* alpha, const float* syn_alpha, const float* threshold, const float* gain,
        float* next_membrane, float* next_synaptic, float* next_spikes, float* next_rate,
        int rows, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int neuron = i % rows;
    const float leak = alpha[neuron], syn_leak = syn_alpha[neuron], limit = threshold[neuron];
    const float syn = (1.f - syn_leak) * synaptic[i] + syn_leak * message[i];
    const float input = drive[i] + bias[neuron] + gain[0] * syn;
    const float voltage = (1.f - leak) * membrane[i] + leak * input;
    const float fired = (voltage > limit && previous_spikes[i] == 0.f) ? 1.f : 0.f;
    next_membrane[i] = voltage - fired * limit;
    next_synaptic[i] = syn;
    next_spikes[i] = fired;
    next_rate[i] = .75f * rate[i] + .25f * fired;
}

extern "C" __global__ void lif_tick_backward(
        const float* membrane, const float* synaptic, const float* previous_spikes,
        const float* drive, const float* message, const float* bias, const float* alpha,
        const float* syn_alpha, const float* threshold, const float* gain,
        const float* next_synaptic, const float* next_spikes,
        const float* grad_membrane, const float* grad_synaptic, const float* grad_spikes,
        const float* grad_rate, float* mem_out, float* syn_out, float* rate_out,
        float* drive_out, float* msg_out, float* alpha_in, float* alpha_state,
        float* syn_alpha_in, float* syn_alpha_state, float* threshold_reset,
        float* threshold_divisor, float* threshold_offset, float* gain_out, int rows, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int neuron = i % rows;
    const float leak = alpha[neuron], syn_leak = syn_alpha[neuron], limit = threshold[neuron];
    const float syn = next_synaptic[i];
    const float input = drive[i] + bias[neuron] + gain[0] * syn;
    const float voltage = (1.f - leak) * membrane[i] + leak * input;
    const float normalized = (voltage - limit) / limit;
    const float span = 1.f + fabsf(normalized);
    // HardSpike's surrogate applies only outside the one-step refractory interval.
    const float spike_adjoint = (grad_spikes[i] + .25f * grad_rate[i])
        * (previous_spikes[i] == 0.f ? 1.f : 0.f);
    const float surrogate = .1f * spike_adjoint / (span * span);
    const float voltage_adjoint = surrogate / limit;
    const float grad_candidate = grad_membrane[i] + voltage_adjoint;
    const float grad_input = grad_candidate * leak;
    const float grad_syn = grad_synaptic[i] + grad_input * gain[0];
    mem_out[i] = grad_candidate * (1.f - leak);
    syn_out[i] = grad_syn * (1.f - syn_leak);
    rate_out[i] = .75f * grad_rate[i];
    drive_out[i] = grad_input;
    msg_out[i] = grad_syn * syn_leak;
    // Keep the broadcast gradients separate until after the reductions.
    alpha_in[i] = grad_candidate * input;
    alpha_state[i] = grad_candidate * membrane[i];
    syn_alpha_in[i] = grad_syn * message[i];
    syn_alpha_state[i] = grad_syn * synaptic[i];
    threshold_reset[i] = -grad_membrane[i] * next_spikes[i];
    threshold_divisor[i] = -surrogate * (normalized / limit);
    threshold_offset[i] = -voltage_adjoint;
    gain_out[i] = grad_input * syn;
}
