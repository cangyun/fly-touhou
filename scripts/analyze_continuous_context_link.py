"""Measure same-scene policy/threat association without any model inference.

A permutation expectation preserves the policy's direction frequencies. It
therefore distinguishes a safer direction prior from state-dependent choices.
This is a selected cached-state diagnostic, not a native avoidance estimate.
"""
from __future__ import annotations
import argparse
import hashlib
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fly_touhou.thrl_durable_records import write_json


def metrics(probability, danger, game):
    n = len(game)
    if not n: return dict(rows=0)
    move = probability[:, :18].astype(np.float64)
    move /= move.sum(-1, keepdims=True)
    selected = move.argmax(-1)
    marginal = np.bincount(selected, minlength=18) / n
    expected_probability = float(move.mean(0) @ danger.mean(0))
    expected_greedy = float(marginal @ danger.mean(0))
    within_probability = within_greedy = 0.
    for g in np.unique(game):
        mask = game == g
        count = int(mask.sum())
        within_probability += count / n * float(move[mask].mean(0) @ danger[mask].mean(0))
        within_greedy += count / n * float((np.bincount(selected[mask], minlength=18) / count) @ danger[mask].mean(0))
    actual = float((move * danger).sum(-1).mean())
    greedy = float(danger[np.arange(n), selected].mean())
    return dict(rows=n, games=len(np.unique(game)),
        conditional_movement_danger_probability=actual,
        same_marginal_permutation_expectation=expected_probability,
        within_game_same_marginal_permutation_expectation=within_probability,
        actual_minus_within_game_expectation=actual - within_probability,
        greedy_danger_count=int(danger[np.arange(n), selected].sum()), greedy_danger_fraction=greedy,
        greedy_same_marginal_permutation_expectation=expected_greedy,
        greedy_within_game_permutation_expectation=within_greedy,
        greedy_actual_minus_within_game_expectation=greedy - within_greedy,
        chosen_direction_counts=np.bincount(selected, minlength=18).tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import psutil
    owner = psutil.Process();owner.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    owner.cpu_affinity(owner.cpu_affinity()[-2:])
    rows = []; reference = None
    for arm in ('trainable', 'fixed'):
        for phase in ('initial', 'final'):
            path = args.run_root / arm / f'cached-evaluation-{phase}' / 'predictions-activities.npz'
            with np.load(path, allow_pickle=False) as archive:
                data = {k: archive[k] for k in ('row', 'game_id', 'heldout', 'features', 'margins',
                    'trusted', 'unprotected', 'raw_probability', 'composed_probability')}
            if reference is None: reference = data
            for key in ('row', 'game_id', 'heldout', 'features', 'margins', 'trusted', 'unprotected'):
                if not np.array_equal(data[key], reference[key]): raise ValueError('Different cached scene selection: ' + key)
            f = data['features']; margin = data['margins'][:, 0]
            normal = (f[:, 21] > .5) & (f[:, 20] <= 0) & (f[:, 5] <= 0)
            hard = normal & data['unprotected'] & data['trusted'][:, 0] & (margin <= 0).any(-1) & (margin > 1).any(-1)
            for label, held in (('train', False), ('heldout', True)):
                selected = hard & (data['heldout'] == held)
                rows.append(dict(arm=arm, phase=phase, scope=label,
                    raw=metrics(data['raw_probability'][selected], (margin <= 0)[selected], data['game_id'][selected]),
                    composed=metrics(data['composed_probability'][selected], (margin <= 0)[selected], data['game_id'][selected])))
    report = dict(schema='continuous-cached-policy-threat-association-v1',
        packet_selection='same fixed critical/uniform scene sample, not natural-frequency play',
        comparison='exact random-permutation expectation of policy rows, globally and within game; preserves marginal actions',
        interpretation='Negative actual-minus-expectation supports state association on these labels; no native play, causal graph effect or stability claim',
        source_root=str(args.run_root.resolve()), script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        results=rows, Torch_imported='torch' in sys.modules, neural_forwards=0, optimizer_updates=0, native_actions=0)
    write_json(args.output, report)
    for row in rows:
        if row['scope'] == 'heldout':
            print({k: row[k] for k in ('arm', 'phase', 'scope')} | {'composed': row['composed']})


if __name__ == '__main__': main()
