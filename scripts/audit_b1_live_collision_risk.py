"""Read-only, CPU comparison of fixed B1 risk heads on live V5 observations.

The independent NumPy geometry is a current-motion proxy. Map-coverage and
motion uncertainty remain explicit; no original counterfactual is executed.
"""
from pathlib import Path
import argparse
import hashlib
import io
import json
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from audit_b1_live_laser_risk import capture_modalities
from fly_touhou.thrl_study_io import utc_now
from fly_touhou.atomic_json_io import atomic_write_json as write
from fly_touhou.thrl_dodge_diagnostics import _shared_binary
from fly_touhou.thrl_movement_teacher import MovementSafetyTeacher, normal_unprotected
from fly_touhou.thrl_laser_risk_labels import route_clearances


def modeled_rows(arrays):
    teacher = MovementSafetyTeacher()
    result = []
    for world, (f, o, maps) in enumerate(zip(arrays['features'], arrays['objects'], arrays['maps'])):
        origin = np.array([f[0]*368-184, f[1]*400+32])
        starts = np.tile(origin, (18, 1))
        no_laser = o.copy()
        no_laser[(o[:,3] > .5) | (o[:,4] > .5)] = 0
        ordinary = teacher.clearances(f, no_laser, starts, teacher.velocities, duration=4)
        laser = route_clearances(f, o, starts, teacher.velocities)
        combined = np.minimum(ordinary, laser)
        uncertainty = teacher._uncertainty(f, o, maps, horizon=4)
        vulnerable = bool(normal_unprotected(f))
        trusted = vulnerable and uncertainty is None
        result.append(dict(worker=world, stage=int(round(float(f[7])*6)),
            normal_unprotected=vulnerable, four_frame_uncertainty=uncertainty,
            trusted_current_motion_proxy=trusted, ordinary=ordinary, laser=laser, combined=combined,
            critical_with_buffered_alternative=bool(trusted and (combined <= 0).any() and (combined > 1).any())))
    return result


def run(args):
    if args.output.exists() or not 0 <= args.wait_seconds <= 600:
        raise ValueError('Use a fresh output and finite passive wait0..600seconds')
    started = time.monotonic()
    attempts = 0
    unstable = 0
    while True:
        try:
            arrays, transport = capture_modalities(args.ownership, include_maps=True)
        except RuntimeError as error:
            if 'repeat-stable transport' not in str(error) or time.monotonic()-started >= args.wait_seconds:
                raise
            unstable += 1
            time.sleep(.5)
            continue
        attempts += 1
        rows = modeled_rows(arrays)
        target_met = any(row['critical_with_buffered_alternative'] for row in rows)
        if target_met or time.monotonic()-started >= args.wait_seconds:
            break
        time.sleep(min(1., max(0., args.wait_seconds-(time.monotonic()-started))))
    transport.update(passive_capture_attempts=attempts, unstable_attempts=unstable,
                     wait_seconds=time.monotonic()-started, requested_sample_present=target_met)
    with _shared_binary(args.checkpoint) as stream:
        data = stream.read()
    import torch
    from torch.nn import functional as F
    from fly_touhou.thrl_inputs import SEPARATED_LASER_INPUT
    from fly_touhou.thrl_pair_risk import PairRiskNetwork, action_clearances as pair_clearances
    from fly_touhou.thrl_laser_risk import LaserRiskNetwork, action_clearances as laser_clearances, action_penalties
    torch.set_num_threads(2)
    payload = torch.load(io.BytesIO(data), map_location='cpu', weights_only=False)
    if payload['model_config']['input_schema'] != SEPARATED_LASER_INPUT:
        raise ValueError('Strict V5 checkpoint required')
    wrapper = payload['model_config']['pair_risk_wrapper']
    if not wrapper['frozen_readout']:
        raise ValueError('This audit requires both risk heads to be fixed')
    pair = PairRiskNetwork(wrapper['width']).eval().requires_grad_(False)
    laser = LaserRiskNetwork(wrapper['laser_width']).eval().requires_grad_(False)
    for network, prefix in [(pair, 'risk.'), (laser, 'laser_risk.')]:
        network.load_state_dict({k.removeprefix(prefix): v for k,v in payload['model'].items()
                                 if k.startswith(prefix)}, strict=True)
    f, o = torch.from_numpy(arrays['features']), torch.from_numpy(arrays['objects'])
    with torch.inference_mode():
        ordinary_prediction = pair_clearances(pair, f, o[...,:21])
        laser_prediction = laser_clearances(laser, f, o)
        costs = F.softplus((wrapper['buffer']-ordinary_prediction)/wrapper['risk_width'])
        costs += action_penalties(laser, f, o, buffer=wrapper['laser_buffer'], risk_width=wrapper['laser_risk_width'])
    ordinary_prediction = ordinary_prediction.numpy()
    laser_prediction = laser_prediction.numpy()
    costs = costs.numpy()
    for index, row in enumerate(rows):
        ordinary, beam, combined = (row.pop(key) for key in ['ordinary', 'laser', 'combined'])
        near = np.abs(ordinary) <= 8
        dangerous = ordinary <= 0
        best = int(np.argmin(costs[index]))
        row.update(ordinary_proxy_margins_clipped64=np.clip(ordinary,-64,64).tolist(),
            laser_proxy_margins_clipped64=np.clip(beam,-64,64).tolist(),
            combined_proxy_margins_clipped64=np.clip(combined,-64,64).tolist(),
            ordinary_predicted_margins=ordinary_prediction[index].tolist(),
            laser_predicted_margins=laser_prediction[index].tolist(),
            combined_learned_costs=costs[index].tolist(),
            ordinary_danger_action_pairs=int(dangerous.sum()),
            ordinary_danger_predicted_beyond_buffer=int((dangerous & (ordinary_prediction[index]>=wrapper['buffer'])).sum()),
            ordinary_boundary_pairs=int(near.sum()),
            ordinary_boundary_absolute_error=float(np.abs(ordinary_prediction[index]-ordinary)[near].sum()),
            pure_risk_preferred_action=best, preferred_combined_proxy_margin=float(np.clip(combined[best],-64,64)),
            actual_policy_action_evaluated=False)
    trusted = [row for row in rows if row['trusted_current_motion_proxy']]
    critical = [row for row in trusted if row['critical_with_buffered_alternative']]
    boundary = sum(row['ordinary_boundary_pairs'] for row in trusted)
    summary = dict(worlds=len(rows), trusted_current_motion_worlds=len(trusted),
        protected_worlds=sum(not row['normal_unprotected'] for row in rows),
        uncertainty_worlds=sum(row['four_frame_uncertainty'] is not None for row in rows),
        critical_with_buffered_alternative=len(critical),
        critical_risk_preference_has_proxy_collision=sum(row['preferred_combined_proxy_margin'] <= 0 for row in critical),
        ordinary_danger_action_pairs=sum(row['ordinary_danger_action_pairs'] for row in trusted),
        ordinary_danger_predicted_beyond_buffer=sum(row['ordinary_danger_predicted_beyond_buffer'] for row in trusted),
        ordinary_boundary_pairs=boundary,
        ordinary_boundary_MAE=sum(row['ordinary_boundary_absolute_error'] for row in trusted)/boundary if boundary else None)
    args.output.mkdir(parents=True)
    path = args.output/'observations.npz'
    np.savez_compressed(path, **arrays)
    write(args.output/'report.json', dict(schema='B1-passive-collision-risk-proxy-audit-v1',
        status='completed' if target_met else 'requested-sample-not-observed', utc=utc_now(), transport=transport,
        source_checkpoint=str(args.checkpoint), checkpoint_sha256=hashlib.sha256(data).hexdigest(),
        checkpoint_optimized_new_decisions=payload.get('candidate_optimized_decisions'),
        arrays=str(path), arrays_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        summary=summary, rows=rows, device='cpu', native_instances_started=0, native_calls=0,
        optimizer_updates=0, mappings_modified=False, fit_on_these_states=False,
        native_collision_labels=False, actual_policy_actions=False,
        scope='Same represented objects;current-motion proxy only;not original counterfactual outcomes or actual evasion'))
    print(json.dumps(summary))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    for key in ['ownership', 'checkpoint', 'output']:
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--wait-seconds', type=float, default=0.)
    run(parser.parse_args())
