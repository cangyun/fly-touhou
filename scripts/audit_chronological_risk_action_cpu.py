"""Read-only risk/actor decomposition on final-body chronological DN caches.

The raw-history cache is made by the learner, with a fixed final body. This
audit loads only CPU head modules. It never builds a brain, starts a game,
fits weights, exports a policy, or changes a live learner. Default is a plan.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'chronological-finalbody-risk-versus-actor-CPU-audit-v1'


def summarize(data, raw, cost, prediction, combined, *, kind=None):
    import numpy as np

    f = data['features']
    normal = data['unprotected_mask'] & (f[:, 21] > .5) & (f[:, 20] <= 0) & (f[:, 5] <= 0)
    if kind is not None:
        normal &= data['teacher_source'] == (kind == 'teacher')
    margin = data['margins']
    trusted = data['trusted']
    choices = dict(actor=raw[:, :18].argmax(-1), risk=cost.argmin(-1),
                   composed=combined[:, :18].argmax(-1))
    row = np.arange(len(f))
    m4 = margin[:, 0]
    known = normal & trusted[:, 0]
    bad = (m4 <= 0) & known[:, None]
    safe = (m4 > 1) & known[:, None]
    mixed = bad.any(-1) & safe.any(-1)
    danger = int(bad.sum())
    boundary = (abs(m4) < 8) & known[:, None]
    result = dict(rows=int(normal.sum()), known4_rows=int(known.sum()), mixed4_rows=int(mixed.sum()),
        dangerous_queries=danger,
        dangerous_recall=float(((prediction <= 0) & bad).sum() / danger) if danger else None,
        false_safe_fraction=float(((prediction > 1) & bad).sum() / danger) if danger else None,
        boundary_MAE=float(abs(prediction-m4)[boundary].mean()) if boundary.any() else None,
        views={})
    for name, choice in choices.items():
        selected = m4[row, choice]
        horizons = {}
        for h, frames in enumerate((4, 12, 24)):
            valid = normal & trusted[:, h]
            m = margin[:, h]
            current = m[row, choice]
            hmixed = valid & (m <= 0).any(-1) & (m > 1).any(-1)
            horizons[str(frames)] = dict(known_rows=int(valid.sum()), mixed_rows=int(hmixed.sum()),
                mixed_selected_danger_count=int((hmixed & (current <= 0)).sum()),
                mixed_selected_danger_fraction=float((current[hmixed] <= 0).mean()) if hmixed.any() else None)
        unsafe = mixed & (selected <= 0)
        recognized = prediction[row, choice] <= 0
        correct_safe_alternative = ((m4 > 1) & (prediction > 1)).any(-1)
        future = normal & trusted[:, 0] & trusted[:, 2]
        future &= (selected > 1) & (margin[:, 2] > 1).any(-1)
        result['views'][name] = dict(horizons=horizons,
            unsafe_mixed4_count=int(unsafe.sum()),
            unsafe_chosen_direction_was_not_predicted_dangerous=int((unsafe & ~recognized).sum()),
            unsafe_despite_predicted_danger_and_correct_safe_alternative=int((unsafe & recognized & correct_safe_alternative).sum()),
            unsafe_with_predicted_danger_but_no_correct_safe_alternative=int((unsafe & recognized & ~correct_safe_alternative).sum()),
            short_safe_with_some_24f_safe_alternative=int(future.sum()),
            short_safe_selected_24f_danger=int((future & (margin[row, 2, choice] <= 0)).sum()),
            selected_direction_counts=np.bincount(choice[normal], minlength=18).tolist())
    override = mixed & (m4[row, choices['composed']] <= 0)
    override &= (prediction[row, choices['composed']] <= 0) & ((m4 > 1) & (prediction > 1)).any(-1)
    result['example_cache_row_indices'] = np.flatnonzero(override)[:20].tolist()
    return result


def run(a):
    os.environ.update(CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                      OPENBLAS_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1')
    import psutil
    process = psutil.Process()
    process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    process.cpu_affinity(process.cpu_affinity()[-2:])
    import numpy as np
    import torch
    from torch import nn
    from torch.nn import functional as F
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    from fly_touhou.thrl_dodge_diagnostics import _shared_binary
    from fly_touhou.thrl_study_io import read_shared_json
    from fly_touhou.thrl_durable_records import write_json
    from fly_touhou.fly_dn_cost_native_policy import (
        compose_from_active_mask, normal_unprotected_from_features, source_policy_logits,
    )

    started = time.monotonic()
    cache_dir = a.run / 'final-body-DN-cache'
    commit = read_shared_json(cache_dir / 'COMMITTED.json')
    calibration = read_shared_json(a.run / 'actor-calibration-result.json')
    if (commit['schema'] != 'current-finalbody-fullhistory-DN-readout-cache-v1'
            or not commit['cached_DN_is_from_this_final_body']
            or not commit['history_started_at_actual_openings']
            or not calibration['body_and_risk_fixed_for_bounded_actor_fitting']):
        raise ValueError('Committed final-body history and bounded unchanged-body actor calibration required')
    if a.output.exists():
        raise FileExistsError('Fresh read-only audit output required')

    def load(path):
        with _shared_binary(path) as stream:
            return torch.load(stream, map_location='cpu', weights_only=True)

    def digest(path):
        with _shared_binary(path) as stream:
            return hashlib.file_digest(stream, 'sha256').hexdigest()

    before_path = a.run / 'risk-pretrained-parameters.pt'
    after_path = a.run / 'final-parameters.pt'
    before, after = load(before_path), load(after_path)
    if (before['state_dict'].keys() != after['state_dict'].keys()
            or not all(torch.equal(before['state_dict'][k], v) for k, v in after['state_dict'].items()
                       if not k.startswith('actor.'))):
        raise ValueError('Cache body/risk/critics changed between risk pretraining and final actor')
    for payload in (before, after):
        graph = payload['graph']
        if (graph['n_neurons'], graph['n_edges'], payload['core_K']) != (138639, 15091983, 8):
            raise ValueError('Complete same K8 connectome required')
        if (payload['teacher_at_inference'] or payload['D95_runtime_queries']
                or payload['risk_input'] != 'following DN only;LayerNorm'):
            raise ValueError('Expected exact DN-only LayerNorm risk subtype')

    actors = {}
    with torch.random.fork_rng(devices=[]):
        for name, payload in (('before_actor_calibration', before), ('after_actor_calibration', after)):
            actor = nn.Sequential(nn.LayerNorm(1299), nn.Linear(1299, 256), nn.SiLU(), nn.Linear(256, 19))
            actor.load_state_dict({k.removeprefix('actor.'): v for k, v in payload['state_dict'].items()
                                   if k.startswith('actor.')}, strict=True)
            actors[name] = actor.eval().requires_grad_(False)
        risk = nn.Sequential(nn.LayerNorm(1299), nn.Linear(1299, 128), nn.SiLU(), nn.Linear(128, 128),
                             nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 18))
        risk.load_state_dict({k.removeprefix('risk_head.layers.'): v for k, v in after['state_dict'].items()
                              if k.startswith('risk_head.layers.')}, strict=True)
        risk.eval().requires_grad_(False)
    sources = dict(before=dict(path=str(before_path), sha256=digest(before_path)),
                   after=dict(path=str(after_path), sha256=digest(after_path)),
                   committed_body_identity=commit['body_identity'], cache={})
    del before, after
    reports = {}
    with torch.no_grad():
        for role in ('train', 'heldout'):
            path = cache_dir / f'{role}.pt'
            data = load(path)
            if data['DN'].shape != (commit['rows'][role], 1299):
                raise ValueError('Exact committed final-body DN shape required')
            sources['cache'][role] = dict(path=str(path), sha256=digest(path), rows=len(data['DN']))
            raw_parts = {name: [] for name in actors}
            combined_parts = {name: [] for name in actors}
            costs, predictions = [], []
            for first in range(0, len(data['DN']), 1024):
                DN = data['DN'][first:first+1024]
                f = data['features'][first:first+1024]
                score = risk(DN)
                cost = F.softplus(score)
                costs.append(cost.numpy()); predictions.append((1.-.5*score).numpy())
                for name, actor in actors.items():
                    raw = actor(DN)
                    combined = compose_from_active_mask(raw, cost, normal_unprotected_from_features(f), beta=8., check_finite=False)
                    actual = source_policy_logits(combined, f, check_finite=False)
                    actual[:, 18] = -torch.inf
                    raw_parts[name].append(raw.numpy()); combined_parts[name].append(actual.numpy())
            npdata = {k: v.numpy() for k, v in data.items() if k != 'DN'}
            cost, prediction = np.concatenate(costs), np.concatenate(predictions)
            reports[role] = {}
            for name in actors:
                raw, combined = np.concatenate(raw_parts[name]), np.concatenate(combined_parts[name])
                reports[role][name] = {kind: summarize(npdata, raw, cost, prediction, combined,
                    kind=None if kind == 'all' else kind) for kind in ('all', 'teacher', 'own')}
    report = dict(schema=SCHEMA, status='completed', sources=sources, reports=reports,
        nonactor_parameters_bitwise_identical=True, beta=8., action_temperature=1., Bomb=False,
        scope='Same cached raw-history DN. Original geometry is a counterfactual proxy;only originally executed action outcomes are native facts. No new gameplay evidence.',
        horizons='4/12/24 original constant-candidate trajectory margins;24f danger is not proof of an unavoidable dead end.',
        classification='A chosen geometry-dangerous action is either not predicted dangerous,or predicted dangerous with/without a correctly predicted safe alternative.',
        CPU_only=True, CUDA_initialized=torch.cuda.is_initialized(), brain_forwards=0, native_actions=0,
        optimizer_updates=0, exported_policy=False, current_training_changed=False,
        elapsed_seconds=time.monotonic()-started)
    if report['CUDA_initialized']:
        raise RuntimeError('CPU-only audit initialized CUDA')
    a.output.mkdir(parents=True)
    write_json(a.output/'report.json', report)
    print(json.dumps(dict(output=str(a.output), elapsed_seconds=report['elapsed_seconds'],
        held_after=reports['heldout']['after_actor_calibration']['all']), ensure_ascii=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run-cpu', action='store_true')
    a = parser.parse_args(argv)
    if not a.run_cpu:
        print(json.dumps(dict(schema=SCHEMA, status='prepared_not_executed', run=str(a.run), output=str(a.output),
            requires='Completed final-body DN cache, actor calibration and final parameter point',
            CUDA=False, brain_or_native_or_optimizer=False)))
        return
    sys.path.insert(0, str(ROOT/'src'))
    run(a)


if __name__ == '__main__':
    main()
