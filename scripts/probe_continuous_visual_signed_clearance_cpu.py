"""One fixed visual384 vs existing DN1299 signed-clearance CPU readout fit.

The visual cache is already FixedStandardizedVisual's output. The additional
train-only scaling is identical to the preceding DN diagnostic's convention.
Only a small head is constructed; no perception, core, policy or teacher runs.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_name] = '2'

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEED = 71931
UPDATES = 500
REFERENCE_SHA = '1f4d38775876106380e07070e4a8c60348d1311013b7fc64ae106fc6b9933f80'
SCHEMA = 'continuous-visual384-vs-DN-signed4f-fixed500-CPU-diagnostic-v1'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def run(a):
    import psutil
    process = psutil.Process()
    process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    process.cpu_affinity(process.cpu_affinity()[-2:])
    if a.output.exists():
        raise FileExistsError('One fresh fixed fit; no overwrite/retry/seed selection')
    a.output.mkdir(parents=True)
    started = time.monotonic()
    ref_path = a.reference / 'report.json'
    if sha(ref_path) != REFERENCE_SHA:
        raise ValueError('Exact previously completed DN500 reference required')
    ref = read(ref_path)
    if ref['diagnostic_Adam'] != UPDATES or ref['recipe']['seed'] != SEED:
        raise ValueError('Same fixed500 DN diagnostic required')
    pred_path = a.reference / 'predictions.npz'
    if sha(pred_path) != ref['predictions_sha256']:
        raise ValueError('Previous row/geometry/prediction evidence differs')
    with np.load(pred_path, allow_pickle=False) as z:
        prior = {k: z[k].copy() for k in ('game_id', 'heldout', 'geometry_valid',
            'geometry_hard', 'target_margin4', 'actual_actions', 'chosen_outcomes',
            'signed4f_original_recipe')}
    with np.load(a.cache / 'labels.npz', allow_pickle=False) as z:
        labels = {k: z[k].copy() for k in ('game_id', 'episode_start',
            'source_rollout', 'source_time', 'world')}
    if sha(a.cache / 'report.json') != ref['sources']['replay_report_sha256']:
        raise ValueError('Previously verified complete final-theta replay identity differs')
    if sha(a.cache / 'labels.npz') != ref['sources']['labels_sha256']:
        raise ValueError('Previously verified source row/reset labels differ')
    encoded_parts = []
    source_records = []
    fields = ('encoded_visual', 'features', 'game_ids', 'episode_starts', 'actions',
        'task_margins', 'task_trusted', 'task_unprotected_mask', 'task_outcomes')
    for number, source in enumerate(ref['sources']['traces']):
        path = Path(source['path'])
        # Original complete source payload hashes were checked by DN500. Reuse
        # that evidence, check its immutable committed pointer, then align every
        # game/row/reset/action/geometry label while extracting only visual384.
        if read(path.with_suffix('.json'))['trace_sha256'] != source['sha256']:
            raise ValueError('Committed source SHA pointer differs')
        with np.load(path, allow_pickle=False) as z:
            b = {k: z[k].copy() for k in fields}
        encoded = b['encoded_visual']
        if encoded.shape != (256, 30, 384) or encoded.dtype != np.float32:
            raise ValueError('Actual FP32 encoded_visual[256,30,384] required')
        if not np.isfinite(encoded).all():
            raise ValueError('Nonfinite saved standardized perception')
        sl = slice(number * 7680, (number + 1) * 7680)
        f = b['features'].reshape(-1, 25)
        margin = np.clip(b['task_margins'][:, :, 0, :], -64., 64.).reshape(-1, 18)
        normal = (f[:, 21] > .5) & (f[:, 20] <= 0) & (f[:, 5] <= 0)
        geometry = b['task_trusted'][:, :, 0].reshape(-1) & b['task_unprotected_mask'].reshape(-1) & normal
        checks = (
            (b['game_ids'].reshape(-1), prior['game_id'][sl]),
            (b['game_ids'].reshape(-1), labels['game_id'][sl]),
            (b['actions'].reshape(-1), prior['actual_actions'][sl]),
            (b['task_outcomes'].reshape(-1), prior['chosen_outcomes'][sl]),
            (b['episode_starts'].reshape(-1), labels['episode_start'][sl]),
            (margin, prior['target_margin4'][sl]),
            (geometry, prior['geometry_valid'][sl]),
            (labels['source_rollout'][sl], np.full(7680, number)),
            (labels['source_time'][sl], np.repeat(np.arange(256), 30)),
            (labels['world'][sl], np.tile(np.arange(30), 256)),
        )
        if not all(np.array_equal(x, y) for x, y in checks):
            raise ValueError('Exact source scene/game/time/reset/action/geometry alignment differs')
        encoded_parts.append(encoded.reshape(-1, 384))
        source_records.append(dict(path=str(path), sha256=source['sha256'],
            payload_sha_rechecked_this_run=False, prior_payload_sha_check=REFERENCE_SHA,
            current_committed_sha_pointer_checked=True, every_row_alignment_checked=True,
            encoded_raw_sha256=hashlib.sha256(encoded.tobytes()).hexdigest()))
    if len(encoded_parts) != 8:
        raise ValueError('Exactly eight source traces, no alternate dataset')
    encoded = np.concatenate(encoded_parts)
    del encoded_parts, b, labels
    games = prior['game_id']
    held = games % 5 == 0
    geometry = prior['geometry_valid']
    margin = prior['target_margin4']
    hard = geometry & (margin <= 0).any(-1) & (margin > 1).any(-1)
    if not np.array_equal(held, prior['heldout']) or not np.array_equal(hard, prior['geometry_hard']):
        raise ValueError('Exact same game-heldout and physical4f hard masks required')
    train = geometry & ~held
    held_geometry = geometry & held
    fit = encoded[train].astype(np.float64)
    mean, sd = fit.mean(0), fit.std(0)
    active = sd >= 1e-4
    floor = float(np.median(sd[active])) if active.any() else 1.
    mean = mean.astype(np.float32)
    scale = np.where(active, np.maximum(sd, floor), 1.).astype(np.float32)
    x = np.asarray((encoded - mean) / scale, np.float32)
    del fit, encoded
    if not np.isfinite(x).all():
        raise ValueError('Finite train-only rescaled visual384 required')

    # Reuse the exact previous diagnostic's scene metrics, not a new definition.
    module_spec = importlib.util.spec_from_file_location('previous_signed_metrics',
        ROOT / 'scripts/probe_continuous_DN_signed_clearance_cpu.py')
    metrics = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(metrics)
    import torch
    from torch import nn
    from torch.nn import functional as F
    torch.set_num_threads(2)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(SEED)
        net = nn.Sequential(nn.Linear(384, 64), nn.SiLU(), nn.Linear(64, 18))
    parameter_count = sum(p.numel() for p in net.parameters())
    if parameter_count != 25810:
        raise AssertionError('384->64SiLU->18linear has25,810 parameters')
    xt, target = torch.from_numpy(x), torch.from_numpy(margin)
    pools = [torch.from_numpy(np.flatnonzero((train[:, None] & cond).reshape(-1)))
        for cond in (margin <= 0, np.abs(margin) < 8, np.ones_like(margin, bool))]
    if [len(p) for p in pools] != list(ref['recipe']['query_pools'][k]
            for k in ('danger_y_le0', 'boundary_abs_y_lt8', 'all')):
        raise ValueError('Same fixed flattened query populations required')
    generator = torch.Generator(device='cpu').manual_seed(SEED)
    optimizer = torch.optim.Adam(net.parameters(), lr=.001, weight_decay=0.)
    history = []
    query_hash = hashlib.sha256()
    paid = backwards = 0
    fit_started = time.monotonic()
    for step in range(UPDATES):
        if time.monotonic() - fit_started >= 60.:
            break
        ix = torch.cat([pool[torch.randint(len(pool), (n,), generator=generator)]
            for pool, n in zip(pools, (1024, 2048, 1024))])
        query_hash.update(ix.numpy().tobytes())
        row, direction = ix // 18, ix % 18
        unique, inverse = torch.unique(row, return_inverse=True)
        prediction = (8. * net(xt[unique]))[inverse, direction]
        y = target[row, direction]
        known = (y <= 0) | (y > 1)
        huber = F.smooth_l1_loss(prediction / 8., y / 8., beta=.1)
        binary = F.binary_cross_entropy_with_logits(-(prediction[known] - .5), (y[known] <= 0).float())
        loss = huber + .2 * binary
        optimizer.zero_grad(set_to_none=True)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite small diagnostic head loss')
        loss.backward()
        backwards += 1
        norm = nn.utils.clip_grad_norm_(net.parameters(), 5.)
        if not bool(torch.isfinite(norm)):
            raise FloatingPointError('Nonfinite small diagnostic head gradients')
        optimizer.step()
        paid += 1
        if paid == 1 or paid % 50 == 0:
            history.append(dict(update=paid, loss=float(loss.detach()), Huber=float(huber.detach()),
                BCE=float(binary.detach()), preclip_grad_norm=float(norm), seconds=time.monotonic() - fit_started))
    fit_seconds = time.monotonic() - fit_started
    with torch.no_grad():
        pred = torch.cat([8. * net(xt[s:s + 1024]) for s in range(0, len(x), 1024)]).numpy()
    reports = {}
    absolute = {}
    for name, p in (('visual384_signed500', pred), ('DN1299_signed500', prior['signed4f_original_recipe'])):
        reports[name] = {role: metrics.summary(-p, margin, mask, prior['actual_actions'], prior['chosen_outcomes'])
            for role, mask in (('train_trusted4f', train), ('held_trusted4f', held_geometry),
                ('train_hard', train & hard), ('held_hard', held & hard))}
        absolute[name] = {}
        with torch.no_grad():
            for role, mask in (('train', train), ('held', held_geometry)):
                danger, safe = mask[:, None] & (margin <= 0), mask[:, None] & (margin > 1)
                pp, yy = torch.from_numpy(p[mask]), torch.from_numpy(margin[mask])
                known = (yy <= 0) | (yy > 1)
                absolute[name][role] = dict(known_danger_queries=int(danger.sum()), known_safe_queries=int(safe.sum()),
                    dangerous_recall_pred_lt1=float((p[danger] < 1).mean()),
                    dangerous_recall_pred_le0=float((p[danger] <= 0).mean()),
                    safe_falsepositive_pred_lt1=float((p[safe] < 1).mean()),
                    absolute_clearance_MAE=float(np.abs(p[mask] - margin[mask]).mean()),
                    boundary_MAE=float(np.abs(p - margin)[mask[:, None] & (np.abs(margin) < 8)].mean()),
                    uniform_query_Huber=float(F.smooth_l1_loss(pp / 8., yy / 8., beta=.1)),
                    uniform_known_BCE=float(F.binary_cross_entropy_with_logits(-(pp[known] - .5), (yy[known] <= 0).float())))
    np.savez_compressed(a.output / 'predictions.npz', predicted_margin4=pred,
        game_id=games, heldout=held, geometry_valid=geometry, geometry_hard=hard,
        source_row=np.arange(61440, dtype=np.int32))
    result = dict(schema=SCHEMA, status='completed500_fixed_steps' if paid == UPDATES else 'partial_finite60s_budget',
        sources=dict(previous_DN_report=str(ref_path), previous_DN_report_sha256=REFERENCE_SHA,
            previous_predictions_sha256=ref['predictions_sha256'], traces=source_records,
            same_geometry_game_reset_alignment=True, no_second_full_source_payload_SHA_sweep=True),
        rows=61440, train_geometry_rows=int(train.sum()), held_geometry_rows=int(held_geometry.sum()),
        held_hard_rows=int((held & hard).sum()), train_games=sorted(set(games[train].tolist())),
        held_games=sorted(set(games[held_geometry].tolist())),
        recipe=dict(**{k: v for k, v in ref['recipe'].items() if k not in ('parameters', 'architecture',
            'active_SD_median', 'active_columns')}, parameters=parameter_count,
            architecture='already-fixed-standardized encoded_visual384->train-only robust scale->64SiLU->18linear; margin8*network',
            active_SD_median=floor, active_columns=int(active.sum()),
            same_query_order_private_generator_seed=SEED, query_sequence_sha256=query_hash.hexdigest()),
        encoded_identity=dict(formula='saved(FrozenV5Perception(rawV5)-fixed_train_mean)/fixed_train_std',
            fixed_statistics_sha256='c7041498d7d5d4aaa3b68be36c4b96dbd4adf535a2146a421a3c4876d2f16627',
            fixed_statistics_reapplied=False, source='continuous_connectome_collector.py:195-210; continuous_connectome_model.py:80-82; fly_visual_distillation.py:88-89',
            source_encoding_no_clip=True, additional_scaling='FP64 population SD on same12982 TRAIN trusted geometry rows; SD>=1e-4 active; scale=max(SD,medianActiveSD),inactive1',
            mean_raw_sha256=hashlib.sha256(mean.tobytes()).hexdigest(), scale_raw_sha256=hashlib.sha256(scale.tobytes()).hexdigest()),
        diagnostic_Adam=paid, diagnostic_backwards=backwards, fit_seconds=fit_seconds,
        fit_history=history, reports=reports, absolute_binary_metrics=absolute,
        parameter_comparison=dict(visual384=parameter_count, DN1299=84370,
            DN_over_visual=84370 / parameter_count, exact_equal_parameter_ablation=False,
            same_seed_does_not_make_different_shapes_identical_initial_weights=True),
        limitations=['PPO body/perception has seen these trajectories; game holdout is only the new small readout fit.',
            'Selected trusted4f physical geometry is a proxy, not unexecuted native collision truth; all sources are early own-policy data.',
            'One fixed seed/500steps/64hidden per representation, not information-theoretic availability or a causal bottleneck proof.',
            'A visual readout cannot be deployed as actor; current actor must continue through the complete graph.'],
        source_runtime_modified=False, full_model_constructed=False, perception_forwards=0,
        full_connectome_forwards=0, teacher_queries=0, CUDA_calls=0, native_actions=0,
        source_model_or_production_Adam_updates=0, policy_or_weights_exported=False,
        resources=dict(threads=2, affinity=process.cpu_affinity(), below_normal=True,
            CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'], RSS_bytes=process.memory_info().rss),
        total_CPU_wall_seconds=time.monotonic() - started,
        predictions_sha256=sha(a.output / 'predictions.npz'))
    (a.output / 'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')
    print(json.dumps(dict(output=str(a.output), paid_steps=paid, fit_seconds=fit_seconds,
        held_hard={k: v['held_hard'] for k, v in reports.items()}, absolute_held={k: v['held'] for k, v in absolute.items()})))
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    run(p.parse_args(argv))


if __name__ == '__main__':
    main()
