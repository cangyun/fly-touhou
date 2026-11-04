"""DN-only clearance/ranking calibration; source whole brain stays frozen."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--source-training', type=Path, required=True)
    p.add_argument('--source-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--calibration', choices=('margin', 'monotonic'), default='margin')
    a = p.parse_args()
    import psutil
    process = psutil.Process()
    process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    process.cpu_affinity(process.cpu_affinity()[-2:])
    import numpy as np
    import torch
    from torch import nn
    from torch.nn import functional as F
    from sklearn.metrics import roc_auc_score
    from fly_touhou.thrl_study_io import read_shared_json
    from fly_touhou.thrl_durable_records import write_json
    torch.set_num_threads(2)
    assert not torch.cuda.is_initialized()
    with a.source_training.open('rb') as stream:
        source_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    if source_sha != a.source_sha256 or a.output.exists():
        raise ValueError('Pinned source and fresh calibration output required')
    index = read_shared_json(a.capture/'index.json')
    if not index['complete']:
        raise ValueError('Complete read-only frozen activity capture required')
    keys = ('DN_rate', 'risk_logits', 'margins', 'trusted', 'features', 'game')
    values = {key: [] for key in keys}
    for part in index['parts']:
        path = a.capture/part['file']
        with path.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != part['sha256']:
                raise ValueError('Diagnostic packet checksum differs')
        with np.load(path, allow_pickle=False) as packet:
            for key in keys:
                values[key].append(packet[key])
    data = {key: np.concatenate(parts) for key, parts in values.items()}
    source = torch.load(a.source_training, map_location='cpu', weights_only=True)
    head = source['recovery']['model_payload']['DN_risk_payload']
    old_weight, old_bias = head['state']['risk.weight'], head['state']['risk.bias']
    rates = data['DN_rate']
    reconstructed = rates @ old_weight.numpy().T + old_bias.numpy()
    alignment_error = float(np.max(np.abs(reconstructed-data['risk_logits'])))
    if alignment_error > 1e-4:
        raise ValueError('Recorded DN rates do not reproduce this source head')
    f = data['features']; c = data['margins'][:, 0]
    valid = data['trusted'][:, 0] & (f[:, 21] > .5) & (f[:, 20] <= 0) & (f[:, 5] <= 0)
    train, held = data['game'] % 5 != 4, data['game'] % 5 == 4
    rows = np.flatnonzero(train & valid)
    mean, scale = rates[rows].mean(0), rates[rows].std(0)
    scale[scale < 1e-4] = 1.
    x = torch.from_numpy(((rates-mean)/scale).astype(np.float32))
    target_margin = torch.from_numpy((-(c-1.)/4.).astype(np.float32))
    positive = torch.from_numpy(c <= 0)
    known = torch.from_numpy((c <= 0) | (c > 1))
    safe = torch.from_numpy(c > 1)
    regression_weight = torch.from_numpy((1+5*(np.abs(c)<8)+3*(c<=0)).astype(np.float32))
    if a.calibration == 'monotonic':
        class Monotonic(nn.Module):
            def __init__(self):
                super().__init__()
                self.log_scale = nn.Parameter(torch.zeros(18))
                self.offset = nn.Parameter(torch.zeros(18))
            def forward(self, value):
                return value*self.log_scale.exp()+self.offset
        model = Monotonic()
        x = torch.from_numpy(reconstructed.astype(np.float32))
    else:
        model = nn.Linear(rates.shape[1], 18)
        with torch.no_grad():
            model.weight.copy_(old_weight*torch.from_numpy(scale)[None])
            model.bias.copy_(old_bias+old_weight@torch.from_numpy(mean))
    optimizer = torch.optim.Adam(model.parameters(), lr=.01 if a.calibration == 'monotonic' else .001,
                                 weight_decay=0. if a.calibration == 'monotonic' else 1e-4)
    generator = torch.Generator().manual_seed(21129)
    steps = 0
    for epoch in range(80):
        order = rows[torch.randperm(len(rows), generator=generator).numpy()]
        for start in range(0, len(order), 128):
            pick = order[start:start+128]
            pred = model(x[pick])
            weight = regression_weight[pick]
            margin_loss = (F.smooth_l1_loss(pred, target_margin[pick], reduction='none')*weight).sum()/weight.sum()
            bce = F.binary_cross_entropy_with_logits(pred, positive[pick].float(), reduction='none')
            pos, neg = positive[pick] & known[pick], ~positive[pick] & known[pick]
            binary_loss = .5*((bce*pos).sum()/pos.sum().clamp_min(1)+(bce*neg).sum()/neg.sum().clamp_min(1))
            pairs = positive[pick, :, None] & safe[pick, None, :]
            ranking = F.softplus(1+pred[:, None, :]-pred[:, :, None])
            counts = pairs.sum((1, 2))
            ranking_loss = ((ranking*pairs).sum((1, 2))/counts.clamp_min(1))
            ranking_loss = (ranking_loss*(counts>0)).sum()/(counts>0).sum().clamp_min(1)
            if a.calibration == 'monotonic':
                regularization = .01*((model.log_scale.exp()-1).square()+model.offset.square()).mean()
                loss = .2*binary_loss+.8*ranking_loss+regularization
            else:
                loss = margin_loss+.2*binary_loss+.2*ranking_loss
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); steps += 1
    with torch.no_grad():
        if a.calibration == 'monotonic':
            native_weight = old_weight*model.log_scale.exp()[:, None]
            native_bias = old_bias*model.log_scale.exp()+model.offset
        else:
            native_weight = model.weight/torch.from_numpy(scale)[None]
            native_bias = model.bias-native_weight@torch.from_numpy(mean)
        predicted = (torch.from_numpy(rates)@native_weight.T+native_bias).numpy()
        explicit = model(x).numpy()
    fold_error = float(np.max(np.abs(predicted-explicit)))
    def score(scores):
        critical = held & valid & (c>1).any(-1) & (c<=1).any(-1)
        selected = scores.argmin(-1)
        auc = []
        for action in range(18):
            mask = held & valid & known[:, action].numpy()
            labels = c[mask, action] <= 0
            if len(np.unique(labels)) == 2:
                auc.append(roc_auc_score(labels, scores[mask, action]))
        pairs = (c<=0)[:, :, None] & (c>1)[:, None, :] & (held & valid)[:, None, None]
        delta = scores[:, :, None]-scores[:, None, :]
        return dict(held_critical_rows=int(critical.sum()),
            unsafe_count=int((c[np.arange(len(c)), selected][critical]<=1).sum()),
            unsafe_rate=float((c[np.arange(len(c)), selected][critical]<=1).mean()),
            within_state_order=float(((delta>0)+.5*(delta==0))[pairs].mean()),
            same_action_macro_AUC=float(np.mean(auc)), score_min=float(scores.min()), score_max=float(scores.max()))
    a.output.mkdir(parents=True, exist_ok=False)
    head['state']['risk.weight'] = native_weight.detach().clone()
    head['state']['risk.bias'] = native_bias.detach().clone()
    torch.save(dict(schema='fly-DN-clearance-ranking-calibration-v1', source_task_sha256=source_sha,
        DN_risk_payload=head, geometry_used_for_execution=False,
        actor_or_brain_parameters_changed=False, normalization_folded_into_linear=True),
        a.output/'DN-risk-calibrated.pt')
    report = dict(schema='fly-DN-clearance-ranking-calibration-report-v1', source_task_sha256=source_sha,
        method=('positive per-action affine calibration; .2BCE+.8within-state ranking+.01identity regularization'
            if a.calibration == 'monotonic' else
            'existing DN-only linear head; negative(clearance-buffer1)/width4, weightedHuber+.2BCE+.2within-state ranking'),
        calibration=a.calibration, same_action_order_preserved_by_construction=a.calibration=='monotonic',
        fixed_epochs=80, CPU_Adam_steps=steps, training_rows=len(rows),
        train_games=sorted(set(data['game'][train].tolist())), held_games=sorted(set(data['game'][held].tolist())),
        original_head=score(data['risk_logits']), calibrated_head=score(predicted),
        source_head_DN_alignment_error=alignment_error, exact_linear_folding_max_error=fold_error,
        capture_index_sha256=hashlib.sha256((a.capture/'index.json').read_bytes()).hexdigest(),
        native_games=0, brain_optimizer_updates=0, CUDA_initialized=False,
        native_clear_achieved=False, diagnostic_geometry_is_native_counterfactual_truth=False)
    write_json(a.output/'report.json', report)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
