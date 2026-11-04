"""Fit small fixed CPU risk probes on paired frozen DN data, never a game policy."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-seconds', type=int, default=240)
    p.add_argument('--fit-cpu', action='store_true')
    a = p.parse_args()
    from fly_touhou.thrl_study_io import read_shared_json
    index = read_shared_json(a.capture/'index.json')
    if not index.get('complete') or a.output.exists():
        raise ValueError('Complete frozen capture and fresh output required')
    if not a.fit_cpu:
        print(json.dumps(dict(status='prepared_not_fitted', rows=index['rows'],
            native_games=0, actor_updates=0, Torch_imported=False)))
        return
    import psutil
    current = psutil.Process()
    current.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    current.cpu_affinity(current.cpu_affinity()[-2:])
    import numpy as np
    import torch
    from torch import nn
    from torch.nn import functional as F
    from sklearn.metrics import roc_auc_score
    from fly_touhou.thrl_durable_records import write_json
    torch.set_num_threads(2)
    assert not torch.cuda.is_initialized()
    begun = time.monotonic()
    def guard():
        if time.monotonic()-begun > a.max_seconds:
            raise TimeoutError('Finite CPU probe time reached; no actor/game was changed')
    keys = ('game', 'encoded_visual', 'DN_rate', 'DN_membrane', 'DN_recent_spikes',
            'margins', 'trusted', 'features', 'risk_logits', 'policy_logits')
    chunks = {key: [] for key in keys}
    for part in index['parts']:
        guard()
        path = a.capture/part['file']
        with path.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != part['sha256']:
                raise ValueError('Committed diagnostic part checksum differs')
        with np.load(path, allow_pickle=False) as values:
            for key in keys:
                chunks[key].append(values[key])
    data = {key: np.concatenate(parts) for key, parts in chunks.items()}
    f = data['features']
    normal = (f[:, 21] > .5) & (f[:, 20] <= 0) & (f[:, 5] <= 0)
    margins = data['margins'][:, 0]
    geometry = normal & data['trusted'][:, 0]
    known = geometry[:, None] & ((margins <= 0) | (margins > 1))
    target = margins <= 0
    train = (data['game'] % 5 != 4)
    held = ~train
    train_games, held_games = (sorted(set(data['game'][mask].tolist())) for mask in (train, held))
    if not train_games or not held_games or set(train_games) & set(held_games):
        raise ValueError('Disjoint nonempty game-ID split required')
    safe = margins > 1
    critical = geometry & safe.any(-1) & (~safe).any(-1)
    train_critical, held_critical = train & critical, held & critical
    def score(prediction):
        chosen = prediction.argmin(-1)
        unsafe = ~safe[np.arange(len(safe)), chosen]
        pairs = known[:, :, None] & target[:, :, None] & known[:, None, :] & ~target[:, None, :]
        pairs &= held[:, None, None]
        delta = prediction[:, :, None] - prediction[:, None, :]
        action_auc = []
        for action in range(18):
            mask = held & known[:, action]
            labels = target[mask, action]
            if len(np.unique(labels)) == 2:
                action_auc.append(dict(action=action, rows=int(mask.sum()), positive=int(labels.sum()),
                    AUC=float(roc_auc_score(labels, prediction[mask, action]))))
        return dict(held_critical_rows=int(held_critical.sum()),
            unsafe_count=int(unsafe[held_critical].sum()),
            unsafe_rate=float(unsafe[held_critical].mean()) if held_critical.any() else None,
            within_state_risk_order=float(((delta > 0)+.5*(delta == 0))[pairs].mean()) if pairs.any() else None,
            selected_directions=np.bincount(chosen[held_critical], minlength=18).tolist(),
            same_action_macro_AUC=float(np.mean([v['AUC'] for v in action_auc])) if action_auc else None,
            same_action_AUC=action_auc)
    recent = data['DN_recent_spikes'][:, ::-1, :]
    age = np.where(recent.any(1), recent.argmax(1), recent.shape[1]).astype(np.float32)/recent.shape[1]
    feature_sets = dict(visual_linear=(data['encoded_visual'], False),
        DN_rate_linear=(data['DN_rate'], False), DN_rate_MLP64=(data['DN_rate'], True),
        DN_rate_voltage_timing_MLP64=(np.concatenate((data['DN_rate'], data['DN_membrane'], age), -1), True))
    train_rows = np.flatnonzero(train & known.any(-1))
    positive = (known[train] & target[train]).sum(0)
    negative = (known[train] & ~target[train]).sum(0)
    # Per-action class balancing changes each output's intercept differently.
    # Preserve its raw discrimination metrics, and separately undo that prior
    # shift before comparing risk scores *between* different actions.
    prior_adjustment = np.where((positive > 0) & (negative > 0),
        np.log(np.maximum(positive, 1)/np.maximum(negative, 1)), 0.)
    class_weight = np.where(target, .5/np.maximum(positive, 1), .5/np.maximum(negative, 1))*known
    class_weight = class_weight.astype(np.float32)*max(1, len(train_rows))
    y = torch.from_numpy(target.astype(np.float32))
    weights = torch.from_numpy(class_weight)
    a.output.mkdir(parents=True, exist_ok=False)
    reports = {}
    for name, (features, nonlinear) in feature_sets.items():
        guard()
        mean = features[train_rows].mean(0)
        scale = features[train_rows].std(0)
        scale[scale < 1e-4] = 1.
        x = torch.from_numpy(np.clip((features-mean)/scale, -50, 50).astype(np.float32))
        torch.manual_seed(71921)
        model = (nn.Sequential(nn.Linear(x.shape[1], 64), nn.Tanh(), nn.Linear(64, 18))
                 if nonlinear else nn.Linear(x.shape[1], 18))
        optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=1e-4)
        generator = torch.Generator().manual_seed(21129)
        last_loss = None
        for epoch in range(40):
            guard()
            permutation = torch.randperm(len(train_rows), generator=generator).numpy()
            for start in range(0, len(train_rows), 256):
                rows = train_rows[permutation[start:start+256]]
                logits = model(x[rows])
                w = weights[rows]
                loss = (F.binary_cross_entropy_with_logits(logits, y[rows], reduction='none')*w).sum()/w.sum().clamp_min(1e-12)
                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
                last_loss = float(loss.detach())
        with torch.no_grad():
            prediction = model(x).numpy()
        reports[name] = dict(score(prediction), input_dim=x.shape[1], hidden=64 if nonlinear else 0,
            epochs=40, training_loss_last=last_loss, parameters=sum(p.numel() for p in model.parameters()))
        reports[name]['probe_Adam_steps'] = 40*((len(train_rows)+255)//256)
        reports[name]['prior_adjusted_direction_metrics'] = score(prediction+prior_adjustment)
        torch.save(dict(schema='frozen-DN-risk-CPU-probe-v1', role='diagnostic_only_not_deployed',
            feature_set=name, mean=torch.from_numpy(mean), scale=torch.from_numpy(scale),
            state=model.state_dict()), a.output/(name+'.pt'))
    constant = int((~safe[train_critical]).mean(0).argmin()) if train_critical.any() else None
    held_best = (~safe[held_critical]).mean(0) if held_critical.any() else None
    report = dict(schema='frozen-visual-versus-DN-risk-decoding-v1', reports=reports,
        train_game_ids=train_games, held_game_ids=held_games,
        split='gameID modulo5 equals4 held; fitted preprocessing/trainers use train only',
        train_selected_constant_direction=constant,
        constant_held_unsafe_rate=float((~safe[held_critical, constant]).mean()) if constant is not None and held_critical.any() else None,
        optimistic_held_constant_unsafe_rate=float(held_best.min()) if held_best is not None else None,
        existing_DN_head=score(data['risk_logits']),
        probe_per_action_logit_prior_adjustment=prior_adjustment.tolist(),
        primary_representation_metric='held same-action macro AUC; does not reward constant direction priors',
        training_positive_per_action=positive.tolist(), rows=len(f),
        capture_index_sha256=hashlib.sha256((a.capture/'index.json').read_bytes()).hexdigest(),
        source_capture=str(a.capture), elapsed_seconds=time.monotonic()-begun,
        capture_scope=index.get('scope', 'complete frozen diagnostic recording'),
        game_actions_sent=0, native_games_started=0, deployed_model_updates=0, CUDA_initialized=False,
        limits=['Geometry proxy labels, not native counterfactual truth',
            'Probe fitting cannot by itself establish native avoidance or clears',
            'Small-probe failure does not prove information is absent',
            'Held games are development/model-selection data, not final formal-test games'])
    write_json(a.output/'report.json', report)
    compact = {name: dict(same_action_macro_AUC=value['same_action_macro_AUC'],
        balanced_logit_unsafe_rate=value['unsafe_rate'],
        prior_adjusted_unsafe_rate=value['prior_adjusted_direction_metrics']['unsafe_rate'],
        prior_adjusted_within_state_order=value['prior_adjusted_direction_metrics']['within_state_risk_order'])
        for name, value in reports.items()}
    print(json.dumps(dict(output=str(a.output/'report.json'), reports=compact,
        constant_held_unsafe_rate=report['constant_held_unsafe_rate'], elapsed_seconds=report['elapsed_seconds'])))


if __name__ == '__main__':
    main()
