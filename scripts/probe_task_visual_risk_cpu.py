"""Read-only linear geometry-risk decoder on committed task traces; no game/CUDA."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time

for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--rollouts', type=int, default=32)
    p.add_argument('--max-seconds', type=int, default=180)
    a = p.parse_args()
    if a.output.exists() or not 1 <= a.rollouts <= 64 or not 0 < a.max_seconds <= 600:
        raise ValueError('Fresh report and finite committed prefix/time required')
    import psutil
    current = psutil.Process()
    current.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    current.cpu_affinity(current.cpu_affinity()[-2:])
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from fly_touhou.thrl_dodge_diagnostics import _shared_binary
    from fly_touhou.thrl_study_io import read_shared_json
    from fly_touhou.thrl_durable_records import write_json
    started = time.monotonic()
    def guard():
        if time.monotonic() - started > a.max_seconds:
            raise TimeoutError('Finite CPU readout diagnosis time reached')
    status = read_shared_json(a.run / 'progress.json')
    if status['costs']['completed_rollouts'] < a.rollouts:
        raise ValueError('All chosen trace rollouts must already be committed')
    fields = ('encoded_visual', 'game_ids', 'actions', 'task_margins', 'task_trusted',
              'task_unprotected_mask', 'risk_logits')
    chunks = {key: [] for key in fields}
    pins = []
    # Training commits ordinal1 using trace version0. Never include the next
    # trace merely because it has already been written by a running collector.
    for number in range(a.rollouts):
        guard()
        path = a.run / 'traces' / f'rollout-{number:03}.npz'
        with _shared_binary(path) as stream:
            data = stream.read()
        digest = hashlib.sha256(data).hexdigest()
        trace = read_shared_json(path.with_suffix('.json'))
        if digest != trace['trace_sha256']:
            raise ValueError('Committed trace checksum differs')
        with np.load(io.BytesIO(data), allow_pickle=False) as values:
            for key in fields:
                value = values[key]
                chunks[key].append(value.reshape((-1, *value.shape[2:])))
        pins.append(dict(path=str(path), sha256=digest))
    x = {key: np.concatenate(values) for key, values in chunks.items()}
    eligible = x['task_unprotected_mask'] & (x['actions'] < 18) & x['task_trusted'][:, 0]
    margins = x['task_margins'][:, 0, :]
    known = eligible[:, None] & ((margins <= 0) | (margins > 1))
    y = margins <= 0
    held = (x['game_ids'] % 5) == 4
    train = ~held
    train_games = sorted(set(x['game_ids'][train].tolist()))
    held_games = sorted(set(x['game_ids'][held].tolist()))
    if not train_games or not held_games or set(train_games) & set(held_games):
        raise ValueError('Disjoint actual game-ID split required')
    mean = x['encoded_visual'][train & eligible].mean(0)
    scale = x['encoded_visual'][train & eligible].std(0).clip(1e-3)
    features = np.clip((x['encoded_visual'] - mean) / scale, -50, 50)
    prediction = np.zeros_like(margins, dtype=np.float64)
    actions = []
    for action in range(18):
        guard()
        mask = train & known[:, action]
        labels = y[mask, action]
        if len(np.unique(labels)) != 2:
            raise ValueError('Both classes required for each train-only action decoder')
        decoder = LogisticRegression(C=1., class_weight='balanced', max_iter=100,
                                     solver='lbfgs', random_state=0)
        decoder.fit(features[mask], labels)
        guard()
        prediction[:, action] = decoder.decision_function(features)
        actions.append(dict(action=action, train_rows=int(mask.sum()),
            train_positive=int(labels.sum()), iterations=int(decoder.n_iter_[0])))
    safe = margins > 1
    critical = held & eligible & safe.any(-1) & (~safe).any(-1)
    if not critical.any():
        raise ValueError('No held critical states for direction comparison')
    def report(scores):
        chosen = np.argmin(scores, axis=-1)
        unsafe = ~safe[np.arange(len(safe)), chosen]
        pairs = (known & y)[:, :, None] & (known & ~y)[:, None, :]
        pairs &= held[:, None, None]
        delta = scores[:, :, None] - scores[:, None, :]
        order = ((delta > 0) + .5 * (delta == 0))[pairs].mean() if pairs.any() else None
        return dict(critical_rows=int(critical.sum()), unsafe_count=int(unsafe[critical].sum()),
            unsafe_rate=float(unsafe[critical].mean()), within_state_risk_order=order,
            selected_actions=np.bincount(chosen[critical], minlength=18).tolist())
    train_critical = train & eligible & safe.any(-1) & (~safe).any(-1)
    prior = np.where(train_critical[:, None], ~safe, 0).sum(0) / max(1, train_critical.sum())
    constant = int(np.argmin(prior))
    optimistic = (~safe[critical]).mean(0)
    write_json(a.output, dict(schema='fly-task-visual384-linear-risk-diagnosis-v1',
        committed_rollouts=a.rollouts, train_game_ids=train_games, held_game_ids=held_games,
        split='game_ID modulo5 equals4 held; all preprocessing/model fitted on train only',
        decoder='18 independent linear logistic models; fixed balanced C1 max100 iterations',
        visual384_decoder=report(prediction), existing_online_DN_head=report(x['risk_logits']),
        train_fitted_constant_action=constant,
        train_fitted_constant_held_unsafe_rate=float((~safe[critical, constant]).mean()),
        optimistic_held_constant_action=int(np.argmin(optimistic)),
        optimistic_held_constant_unsafe_rate=float(optimistic.min()),
        fitted_actions=actions, sources=pins, elapsed_seconds=time.monotonic()-started,
        CUDA_initialized=False, native_games=0, actor_updates=0,
        conclusion_limits=['Geometry proxy labels, not native counterfactual truth',
            'Frozen visual mapping; small linear failure cannot prove information absent',
            'Existing online DN head has seen some held-game labels; its comparison is descriptive',
            'This is offline decoding, not deployment or a new actor',
            'No native avoidance improvement is established by this report']))
    print(json.dumps(dict(output=str(a.output), visual384=report(prediction),
        DN=report(x['risk_logits']), train_constant=constant,
        train_constant_unsafe=float((~safe[critical, constant]).mean()),
        optimistic_constant_unsafe=float(optimistic.min()), elapsed=time.monotonic()-started)))


if __name__ == '__main__':
    main()
