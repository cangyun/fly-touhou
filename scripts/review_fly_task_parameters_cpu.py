"""CPU-only parameter/input saturation evidence; no brain forward or updates."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from pathlib import Path
import argparse
import hashlib
import json
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError('Preserve earlier diagnosis reports')
    import psutil
    current = psutil.Process()
    current.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    current.cpu_affinity(current.cpu_affinity()[-2:])
    import numpy as np
    import torch
    torch.set_num_threads(2)
    assert not torch.cuda.is_initialized()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
    from fly_touhou.thrl_durable_records import write_json
    def digest(path):
        with path.open('rb') as stream:
            return hashlib.file_digest(stream, 'sha256').hexdigest()
    parameters = {}
    pins = {}
    for tag in ('initial', 'final'):
        path = a.run / (tag + '-parameters.pt')
        payload = torch.load(path, map_location='cpu', weights_only=True)
        parameters[tag] = {name: value.numpy().copy()
                           for name, value in payload['student_payload']['brain_parameters'].items()}
        pins[tag] = dict(path=str(path), sha256=digest(path))
    def quantiles(x):
        return dict(zip(('min', 'p01', 'p10', 'median', 'p90', 'p99', 'max'),
                        np.quantile(x, (0, .01, .1, .5, .9, .99, 1)).tolist()))
    encoded = []
    traces = []
    for version in range(60, 64):
        path = a.run / 'traces' / f'rollout-{version:03}.npz'
        with np.load(path, allow_pickle=False) as packet:
            encoded.append(packet['encoded_visual'].reshape(-1, 384)[::4])
        traces.append(dict(path=str(path), sha256=digest(path)))
    encoded = np.concatenate(encoded)
    rows = {}
    for tag, values in parameters.items():
        logits = encoded @ values['input_encoder.weight'].T + values['input_encoder.bias']
        sigmoid = 1 / (1 + np.exp(-np.clip(logits, -80, 80)))
        gains = 1 + .1 * np.tanh(values['recurrent_gain_logits'])
        rows[tag] = dict(input_logits=quantiles(logits),
            input_sigmoid=quantiles(sigmoid),
            input_ports_near_0_or_1_fraction=float(((sigmoid < .01) | (sigmoid > .99)).mean()),
            postsynaptic_gain=quantiles(gains),
            gain_at_99pct_constraint_fraction=float((np.abs(gains - 1) > .099).mean()),
            gain_logit_RMS=float(np.sqrt(np.mean(values['recurrent_gain_logits']**2))))
    delta = {}
    for name, final in parameters['final'].items():
        initial = parameters['initial'][name]
        delta[name] = dict(initial_RMS=float(np.sqrt(np.mean(initial**2))),
            final_RMS=float(np.sqrt(np.mean(final**2))),
            delta_RMS=float(np.sqrt(np.mean((final-initial)**2))),
            changed=int((final != initial).sum()), total=int(final.size))
    # An objective switch retained the eight original Adam histories. Quantify
    # their exact residual contribution, rather than assuming that they help or
    # hurt the new task. This does not create or step an optimizer.
    optimizers = {}
    for tag in ('initial', 'final'):
        path = a.run / (tag + '-training.pt')
        payload = torch.load(path, map_location='cpu', weights_only=True)
        optimizers[tag] = payload['recovery']['optimizer_state']
        pins[tag + '_training'] = dict(path=str(path), sha256=digest(path))
    moments = []
    for old_group, new_group in zip(optimizers['initial']['param_groups'],
                                    optimizers['final']['param_groups']):
        if old_group['name'] != new_group['name'] or old_group['params'] != new_group['params']:
            raise ValueError('Original optimizer parameter identity differs')
        for identifier in new_group['params']:
            old = optimizers['initial']['state'].get(identifier)
            final = optimizers['final']['state'][identifier]
            initial_step = int(old['step']) if old else 0
            steps = int(final['step']) - initial_step
            total = float(final['exp_avg_sq'].double().sum())
            inherited = (float(old['exp_avg_sq'].double().sum()) * new_group['betas'][1]**steps
                         if old else 0.)
            moments.append(dict(group=new_group['name'], state_id=identifier, steps=steps,
                inherited_initial_step=initial_step, learning_rate=new_group['lr'],
                exp_avg_sq_RMS_denominator=float(final['exp_avg_sq'].double().mean().sqrt()),
                inherited_exp_avg_sq_fraction=inherited/total if total else None))
    write_json(a.output, dict(schema='fly-task-parameter-input-cpu-review-v1',
        sources=pins, traces=traces, projection_input_rows=len(encoded),
        projection_input='Actual already-standardized frozen visual384; last4 committed rollouts stride4',
        projections=rows, parameter_changes=delta, Adam_history=moments,
        brain_forwards=0, CUDA_initialized=torch.cuda.is_initialized(), native_games=0,
        optimizer_updates=0,
        limitations=['Saturation statistics alone do not prove neural information transmission',
                     'Residual Adam history share alone cannot establish that reset would improve learning',
                     'Both parameter snapshots are projected on identical saved behavior features']))
    print(json.dumps(dict(output=str(a.output), projections=rows, parameter_changes=delta,
                          Adam_history=moments)))


if __name__ == '__main__':
    main()
