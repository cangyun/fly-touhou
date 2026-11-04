"""Finite CUDA readout learning on real frozen-brain activities, no brain replay."""
from pathlib import Path
import argparse
import json
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '2'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a fresh output; preserve earlier candidates.')
    args.output.mkdir(parents=True)
    started = time.monotonic()
    import psutil
    process = psutil.Process()
    process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    process.cpu_affinity(process.cpu_affinity()[:2])
    from fly_touhou.fly_cached_distillation import import_reader_preserving_CVD
    import_reader_preserving_CVD()
    from fly_touhou.fly_dn_readout_probe import load_cache_manifest, metrics, file_sha, ProbeGuard
    from fly_touhou.thrl_durable_records import write_json
    guard = ProbeGuard()
    manifest, data, _ = load_cache_manifest(args.manifest, load_arrays=True, guard=guard)
    import numpy as np
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(.2)
    torch.manual_seed(613711)
    torch.cuda.manual_seed_all(613711)
    fixed_steps, learning_rate = 512, .01
    source = Path(manifest['model']['frozen_source_path'])
    if file_sha(source) != manifest['model']['outer_checkpoint_sha256']:
        raise ValueError('The activity cache and source parameters must match.')
    provenance = dict(schema='fly-DN-motor-fixed-KL-refit-v1', steps=fixed_steps,
        learning_rate=learning_rate, weight_decay=0., optimizer='Adam', batch_rows=2048,
        loss='T1 teacher-probability KL; original inventory-legal19',
        input='frozen following-state DN1299 filtered spike rates',
        scale='train-only column RMS, no centering/bias; folded into raw motor weights',
        graph_input_and_neural_parameters_frozen=True, held_used_for_training_or_stopping=False,
        source_checkpoint_sha256=file_sha(source), cache_manifest_sha256=file_sha(args.manifest),
        native_worlds=0, neural_forwards=0, source_script_sha256=file_sha(__file__))
    write_json(args.output / 'configuration.json', provenance)
    x = torch.as_tensor(data['train']['DN_features'], device='cuda')
    scale = x.square().mean(0).sqrt()
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    xs = x / scale
    p = torch.as_tensor(data['train']['teacher_probabilities'], device='cuda')
    p = p / p.sum(-1, keepdim=True)
    initial = torch.as_tensor(data['train']['online_readout_weight'], device='cuda')
    weight = torch.nn.Parameter(initial * scale)
    optimizer = torch.optim.Adam([weight], lr=learning_rate)
    for step in range(1, fixed_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = F.kl_div(F.log_softmax(F.linear(xs, weight), dim=-1), p, reduction='batchmean')
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite KL; no candidate was saved.')
        loss.backward()
        optimizer.step()
        if step % 64 == 0:
            guard.check('finite_motor_fit')
            write_json(args.output / 'progress.json', dict(status='training_motor_only',
                optimizer_updates=step, loss=float(loss.detach()), elapsed_seconds=time.monotonic()-started))
    folded = (weight.detach() / scale).cpu().numpy().astype(np.float32)
    if not np.isfinite(folded).all() or folded.shape != (19, 1299):
        raise FloatingPointError('Invalid raw motor weights; no candidate was saved.')
    scores = {role: metrics(split['teacher_probabilities'], split['DN_features'] @ folded.T)
              for role, split in data.items()}
    payload = torch.load(source, map_location='cpu', weights_only=True)
    payload['base_parameter_payload']['trainable_parameters']['brain.action_readout.weight'] = torch.from_numpy(folded.copy())
    payload['motor_refit_provenance'] = provenance
    candidate = args.output / 'candidate.pt'
    torch.save(payload, candidate.with_suffix('.pt.tmp'))
    os.replace(candidate.with_suffix('.pt.tmp'), candidate)
    report = dict(provenance, status='completed', optimizer_updates=fixed_steps,
        train_label_presentations=2048*fixed_steps, metrics=scores,
        candidate=str(candidate.resolve()), candidate_sha256=file_sha(candidate),
        elapsed_seconds=time.monotonic()-started,
        GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        native_avoidance_demonstrated=False)
    write_json(args.output / 'result.json', report)
    print(json.dumps(dict(status='completed', metrics=scores, seconds=report['elapsed_seconds']), ensure_ascii=False))


if __name__ == '__main__':
    main()
