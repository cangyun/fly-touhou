"""CPU-only audit of a frozen point-risk assumption in existing native tails.

No deployed policy is changed. A static, recorded-prefix alternative drops
unknown growth-motion point queries from the four-tick directional cost.
Its unexecuted actions have no measured native outcome.
"""
from pathlib import Path
import argparse
import hashlib
import io
import json
import os
import sys
import time


def known_point_margin(predicted, valid, growth):
    import torch
    if predicted.ndim != 3 or valid.shape != predicted.shape[::2] or growth.shape != valid.shape:
        raise ValueError('Matched batch/action/entity predictions and masks required')
    known = valid & ~growth
    return torch.where(known[:, None], predicted, torch.full_like(predicted, 64.)).amin(-1)


def inverse_softplus_cost(cost):
    import torch
    cost = cost.clamp_min(1e-30)
    return cost + torch.log(-torch.expm1(-cost))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(a):
    import numpy as np
    import psutil
    import torch
    sys.path[:0] = [str(a.runtime / 'src')]
    from fly_touhou.thrl_study_io import read_shared_json as read, utc_now
    from fly_touhou.atomic_json_io import atomic_write_json as write
    from fly_touhou.thrl_dodge_diagnostics import _shared_binary
    from fly_touhou.thrl_pair_policy import model_from_config
    from fly_touhou.thrl_pair_risk import observation_queries
    from fly_touhou.thrl_laser_risk import known_margin_costs
    from fly_touhou.thrl_policy_actions import policy_logits
    from fly_touhou.thrl_movement_teacher import MovementSafetyTeacher, normal_unprotected
    if (a.output.exists() or sha(a.runtime / 'source_manifest.json') !=
            'ec6461d83aa43243ea8d5f2d8a9520eb7223213c89bdee9ee6e5110b7f37aa70'):
        raise ValueError('Use a new output and the unchanged qualified recovery runtime')
    if psutil.virtual_memory().available < 2 * 1024**3:
        raise RuntimeError('Defer CPU audit until2GiB host reserve is available')
    psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
    torch.set_num_threads(1)
    report = read(a.report)
    if report['status'] != 'completed' or report['native_seed_split'] != 'development':
        raise ValueError('Use completed development evidence, never prospective formal tests')
    pin = report['pins']['trained']
    if pin['sha256'] != 'd47b65e29b439a8b6fb070abc3701bdbcd546364536d45b9233eab472cb83a69':
        raise ValueError('Actual selected pre-PPO BC required')
    checkpoint = Path(pin['copy'])
    if sha(checkpoint) != pin['sha256']:
        raise ValueError('BC checkpoint changed')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = model_from_config(**payload['model_config']).cpu().eval()
    model.load_state_dict(payload['model'], strict=True)
    if not all(torch.equal(v, payload['model'][k]) for k, v in model.state_dict().items()):
        raise ValueError('Actual loaded weights differ')
    with _shared_binary(a.current_checkpoint) as stream:
        current_raw = stream.read()
    current = torch.load(io.BytesIO(current_raw), map_location='cpu', weights_only=False)
    risk_keys = [k for k in payload['model'] if k.startswith(('risk.', 'laser_risk.'))]
    if len(risk_keys) != 18 or not all(torch.equal(current['model'][k], payload['model'][k]) for k in risk_keys):
        raise ValueError('Current frozen risk heads differ from the BC risk heads')
    del current
    selected = sorted((r for r in report['episode_results'] if r['policy'] == 'trained_greedy'), key=lambda r:r['game'])
    if len(selected) != 20 or [r['game'] for r in selected] != list(range(20)):
        raise ValueError('Use all20 recorded BC failures')
    a.output.mkdir(parents=True, exist_ok=False)
    write(a.output / 'launch.json', dict(pid=os.getpid(), create_time=psutil.Process().create_time(),
        utc=utc_now(), CPU_only=True, CUDA_used=False, optimizer_steps=0, native_worlds=0))
    (a.output / 'diagnostic_source.py').write_bytes(Path(__file__).read_bytes())
    teacher = MovementSafetyTeacher()
    rows = []
    started = time.monotonic()
    with torch.inference_mode():
        for case in selected:
            pointer = case['failure_trace']; receipt = read(pointer['receipt']); path = Path(pointer['arrays'])
            if (sha(path) != pointer['sha256'] or receipt['sha256'] != pointer['sha256']
                    or receipt['checkpoint_sha256'] != pin['sha256'] or not case['first_collision_observed']
                    or receipt.get('history_rows') != 128):
                raise ValueError('Native collision tail provenance differs')
            with np.load(path, allow_pickle=False) as bundle:
                f = torch.from_numpy(bundle['features'][-1:].copy())
                o = torch.from_numpy(bundle['objects'][-1:].copy())
                maps = bundle['last_preaction_maps'].copy()
                m = torch.from_numpy(maps[None])
                hidden = torch.from_numpy(bundle['hidden'][-1:].copy())[None]
                action = int(bundle['action'][-1])
                captured = torch.from_numpy(bundle['probabilities'][-1:].copy())
                before_frame = int(bundle['native_frame'][-1])
            if f.shape != (1,25) or o.shape != (1,96,29) or not normal_unprotected(f[0].numpy()):
                raise ValueError('Comparable normal,unprotected pre-action state required')
            base, _, _ = model.base.forward_step(f, m, hidden, objects=o)
            original_risk = model.precompute_frozen_risk(f, o)
            original = model._adjust(base, f, o, precomputed_risk=original_risk)
            original_prob = torch.softmax(policy_logits(original)[:, :18], -1)
            error = float((original_prob - captured).abs().max())
            if error > 3e-4 or int(original_prob.argmax(-1)) != action:
                raise ValueError('CPU replay did not reproduce the actual recorded BC action')
            query, valid = observation_queries(f, o[..., :21])
            predicted = model.risk(query)
            growth = valid & (o[...,20] > .5)
            all_margin = torch.where(valid[:,None], predicted, torch.full_like(predicted,64.)).amin(-1)
            alternative_margin = known_point_margin(predicted, valid, growth)
            laser, comfort = known_margin_costs(model.laser_risk, f, o,
                primary_buffer=model.buffer, primary_width=model.risk_width,
                comfort_buffer=model.laser_buffer, comfort_width=model.laser_risk_width)
            def combine(margin):
                point = torch.nn.functional.softplus((model.buffer - margin) / model.risk_width)
                return torch.maximum(point, laser) + model.laser_buffer_weight * comfort
            reconstructed = inverse_softplus_cost(combine(all_margin))
            parity = float((torch.nn.functional.softplus(reconstructed) - torch.nn.functional.softplus(original_risk)).abs().max())
            if parity > 1e-4:
                raise ValueError('Audit did not reproduce the deployed aggregation')
            alternative_risk = inverse_softplus_cost(combine(alternative_margin))
            alternative = model._adjust(base, f, o, precomputed_risk=alternative_risk)
            alternative_prob = torch.softmax(policy_logits(alternative)[:, :18], -1)
            point_cost_delta = (combine(all_margin) - combine(alternative_margin))[0]
            chosen_predictions = torch.where(valid[0], predicted[0,action], torch.full_like(predicted[0,action],64.))
            winner = int(chosen_predictions.argmin())
            row = dict(game=case['game'], native_seed=case['native_seed'], stage=case['stage'],
                actual_native_collision=True, before_native_frame=before_frame,
                after_native_frame=receipt['after_native_frame'], actual_action=action,
                CPU_replay_max_probability_error=error, aggregation_parity_error=parity,
                valid_point_tokens=int(valid.sum()), collidable_growth_tokens=int(growth.sum()),
                chosen_neural_point_margin=float(all_margin[0,action]),
                chosen_point_worst_token_growth=bool(valid[0,winner] and growth[0,winner]),
                chosen_known_point_margin=float(alternative_margin[0,action]),
                current_motion_teacher_abstention=teacher._uncertainty(f[0].numpy(), o[0].numpy(), maps, horizon=4),
                maximum_total_directional_cost_drop=float(point_cost_delta.max()),
                directional_cost_drop_range=float(point_cost_delta.max()-point_cost_delta.min()),
                actual_action_total_cost_drop=float(point_cost_delta[action]),
                alternative_action=int(alternative_prob.argmax(-1)),
                alternative_probability_on_actual_action=float(alternative_prob[0,action]),
                alternative_action_native_outcome_measured=False, source_receipt=pointer['receipt'])
            rows.append(row)
            write(a.output / 'progress.json', dict(status='CPU_readonly_diagnostic', completed=len(rows),
                total=20, native_worlds=0, optimizer_updates=0, wall_seconds=time.monotonic()-started))
    result = dict(schema='B1-pre-PPO-BC-point-growth-risk-static-audit-v1', status='completed', utc=utc_now(),
        source_runtime=str(a.runtime), source_manifest_sha256=sha(a.runtime/'source_manifest.json'),
        BC_checkpoint_sha256=pin['sha256'], all_loaded_BC_weights_verified=True,
        current_PPO_checkpoint_sha256=hashlib.sha256(current_raw).hexdigest(),
        current_frozen_risk_tensors_verified_identical=18, actual_native_BC_failure_cases=20,
        cases_with_collidable_growth=sum(r['collidable_growth_tokens']>0 for r in rows),
        cases_actual_action_worst_point_is_growth=sum(r['chosen_point_worst_token_growth'] for r in rows),
        cases_directional_cost_changed=sum(r['directional_cost_drop_range']>1e-5 for r in rows),
        cases_recorded_prefix_argmax_changed=sum(r['alternative_action']!=r['actual_action'] for r in rows),
        maximum_CPU_original_probability_replay_error=max(r['CPU_replay_max_probability_error'] for r in rows),
        rows=rows, scope='Static old BC native prefixes;current PPO base policy is different;no unexecuted native outcomes',
        inference='Unknown point motion uses a4-frame velocity forecast while unknown laser motion is filtered;static ranking sensitivity is not proof of avoidance improvement',
        CUDA_used=False, optimizer_updates=0, native_worlds_started=0, current_training_modified=False,
        deployment_forbidden=True, original_goal_achieved=False, wall_seconds=time.monotonic()-started)
    write(a.output/'report.json', result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    for name in ('runtime','report','current-checkpoint','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    run(parser.parse_args())
