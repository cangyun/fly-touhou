"""Read existing student rollout evidence without running any neural model.

The geometric four-frame labels are diagnostic proxies, not native outcomes.
Rows come from changing student policies and are not a held-out learning test.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
from fly_touhou.thrl_study_io import read_shared_json
from fly_touhou.thrl_durable_records import write_json


def rate(n, d):
    return None if d == 0 else float(n / d)


def summarize(path):
    with np.load(path, allow_pickle=False) as z:
        f = z['features'].reshape(-1, 25)
        target = z['task_margins'][..., 0, :].reshape(-1, 18)
        rows = (z['task_trusted'][..., 0].reshape(-1)
            & z['task_unprotected_mask'].reshape(-1)
            & (f[:, 21] > .5) & (f[:, 20] <= 0) & (f[:, 5] <= 0))
        costs = z['predicted_costs'].reshape(-1, 18)
        probs = z['action_probs'].reshape(-1, 19)[:, :18]
        raw = np.clip(z['raw_logits'].reshape(-1, 19)[:, :18].astype(np.float64), -80, 80)
        raw -= raw.max(axis=-1, keepdims=True)
        raw_probs = np.exp(raw)
        raw_probs /= raw_probs.sum(axis=-1, keepdims=True)
        actions = z['actions'].reshape(-1)
        contacts = z['first_observed_contact'].reshape(-1)
        assert not np.any(actions >= 18), 'NoBomb movement-only trace required'
        assert costs.shape == target.shape == probs.shape
        assert np.isfinite(costs).all() and np.all(costs >= 0)
        assert np.isfinite(probs).all() and np.allclose(probs.sum(-1), 1., atol=1e-5)
        # margin=1-score/2 and cost=softplus(score). Classify in cost space;
        # this avoids an unstable inverse-softplus and uses no model forward.
        predicted_danger = costs >= np.logaddexp(0., 2.)
        truth_danger = target <= 0
        truth_safe_known = target > 1.
        valid = np.broadcast_to(rows[:, None], target.shape)
        mixed = rows & truth_danger.any(-1) & (~truth_danger).any(-1)
        index = np.arange(len(actions))
        actual_greedy = probs.argmax(-1)
        raw_greedy = raw_probs.argmax(-1)
        danger_n = int((valid & truth_danger).sum())
        safe_n = int((valid & truth_safe_known).sum())
        contact_n = int(contacts.sum())
        labelled_contact = contacts & rows
        risk_actual_contact_false_safe = contacts & ~predicted_danger[index, actions]
        report = dict(trace=str(path), original_rows=len(actions),
            labelled_normal_scenes=int(rows.sum()), labelled_mixed_scenes=int(mixed.sum()),
            danger_queries=danger_n, known_safe_queries=safe_n,
            risk_danger_recall=rate(int((valid & truth_danger & predicted_danger).sum()), danger_n),
            risk_false_positive_rate=rate(int((valid & truth_safe_known & predicted_danger).sum()), safe_n),
            mixed_raw_proxy_danger_mass=None if not mixed.any() else float((raw_probs * truth_danger)[mixed].sum(-1).mean()),
            mixed_composed_proxy_danger_mass=None if not mixed.any() else float((probs * truth_danger)[mixed].sum(-1).mean()),
            mixed_raw_greedy_proxy_danger=rate(int((mixed & truth_danger[index, raw_greedy]).sum()), int(mixed.sum())),
            mixed_composed_greedy_proxy_danger=rate(int((mixed & truth_danger[index, actual_greedy]).sum()), int(mixed.sum())),
            mixed_sampled_proxy_danger=rate(int((mixed & truth_danger[index, actions]).sum()), int(mixed.sum())),
            accepted_native_first_contacts=contact_n,
            accepted_contacts_predicted_safe=int(risk_actual_contact_false_safe.sum()),
            accepted_contacts_with_proxy_label=int(labelled_contact.sum()),
            labelled_accepted_contacts_proxy_unsafe=int((labelled_contact & truth_danger[index, actions]).sum()),
            actual_Bomb_actions=int((actions == 18).sum()))
    update_path = path.with_name('update-' + f'{int(path.stem.split("-")[-1]) + 1:06}' + '.json')
    if update_path.exists():
        learning = read_shared_json(update_path)['learning']
        keys = ('optimizer_updates', 'epochs_completed', 'first_gradient_groups', 'post_epoch')
        report['actual_update'] = {k: learning[k] for k in keys if k in learning}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Fresh evidence output required')
    reports = []
    for training in args.training:
        traces = sorted((training / 'traces').glob('rollout-*.npz'))
        reports.append(dict(training=str(training), rollouts=[summarize(p) for p in traces]))
    result = dict(schema='NoBomb-existing-rollout-risk-evidence-v1',
        UTC=datetime.now(timezone.utc).isoformat(), training=reports,
        scope='Changing on-policy student rows;not a held-out or paired policy evaluation. '
            '4f geometry is a proxy;accepted native first contacts are separately counted.',
        model_forwards=0, native_actions=0, optimizer_updates=0)
    args.output.mkdir(parents=True)
    write_json(args.output / 'report.json', result)
    print(json.dumps(dict(report=str(args.output / 'report.json'),
        rollouts=sum(len(x['rollouts']) for x in reports),
        model_forwards=0, native_actions=0, optimizer_updates=0)))


if __name__ == '__main__':
    main()
