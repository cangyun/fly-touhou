"""Passive V5 transport and frozen laser-readout diagnostic, on CPU only.

Borrow managed shared buffers without native calls, controls, or unlinking.
Compare the trained MLP with independent conservative capsule proxy labels;
neither predictions nor labels are claimed as original-game outcomes.
"""
from pathlib import Path
from multiprocessing.shared_memory import SharedMemory
import argparse
import hashlib
import io
import json
import sys
import time

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fly_touhou.thrl_study_io import read_shared_json as read, utc_now
from fly_touhou.atomic_json_io import atomic_write_json as write
from fly_touhou.thrl_dodge_diagnostics import _shared_binary


def capture_modalities(path, *, include_maps=False):
    ownership = read(path)
    owner = ownership['owner']
    if (ownership['status'] != 'active'
            or abs(psutil.Process(owner['pid']).create_time() - owner['create_time']) > .01):
        raise ValueError('An identity-pinned active managed vector is required')
    handles, views = {}, {}
    try:
        shapes = [('features', (25,)), ('objects', (96, 29))]
        if include_maps:
            shapes.append(('maps', (16, 92, 96)))
        for key, tail in shapes:
            descriptor = ownership['shared_memory'][key]
            shape = tuple(descriptor['shape'])
            if (descriptor['dtype'] != 'float32' or shape[1:] != tail
                    or not 1 <= shape[0] <= 30):
                raise ValueError('Only bounded V5 FP32 observations are supported')
            handles[key] = SharedMemory(name=descriptor['name'], create=False)
            views[key] = np.ndarray(shape, np.float32, buffer=handles[key].buf)
            views[key].setflags(write=False)
        if any(len(view) != len(views['features']) for view in views.values()):
            raise ValueError('Unpaired vector shapes')
        for attempt in range(8):
            arrays = {key: view.copy() for key, view in views.items()}
            if all(np.array_equal(arrays[key], view) for key, view in views.items()):
                break
        else:
            raise RuntimeError('No repeat-stable transport snapshot; do not label it')
        if abs(psutil.Process(owner['pid']).create_time() - owner['create_time']) > .01:
            raise ValueError('The borrowed owner changed during capture')
        if not all(np.isfinite(array).all() for array in arrays.values()):
            raise ValueError('Nonfinite source observations')
        return arrays, dict(owner=owner, ownership=str(path), utc=utc_now(),
            repeated_read_stable=True, attempts=attempt+1,
            atomic_native_frame_claimed=False)
    finally:
        views.clear()
        for handle in handles.values():
            handle.close()  # The vector owner alone controls mapping lifetime.


def capture(path):
    arrays, transport = capture_modalities(path)
    return arrays['features'], arrays['objects'], transport


def run(args):
    if args.output.exists():
        raise ValueError('Preserve previous diagnostics')
    if not 0 <= args.wait_laser_seconds <= 600:
        raise ValueError('Finite passive sampling is limited to ten minutes')
    from fly_touhou.thrl_laser_risk_labels import route_clearances
    from fly_touhou.thrl_movement_teacher import action_velocities
    velocities = action_velocities()
    started = time.monotonic()
    capture_attempts = 0
    target_met = False
    while True:
        features, objects, transport = capture(args.ownership)
        capture_attempts += 1
        known_now = ((objects[..., 0] > .5) & ((objects[..., 3] > .5) | (objects[..., 4] > .5))
                     & (objects[..., 13] > .5) & (objects[..., 26] > .5))
        target_met = bool(known_now.any())
        if target_met and args.require_dangerous_alternative:
            target_met = False
            for index in np.flatnonzero(known_now.any(-1)):
                f0 = features[index]
                if not (f0[21] > .5 and f0[20] <= 0 and f0[5] <= 0):
                    continue
                o0 = np.where(known_now[index, :, None], objects[index], 0)
                origin = np.array([f0[0]*368-184, f0[1]*400+32])
                margins = route_clearances(f0, o0, np.tile(origin, (18, 1)), velocities)
                if (margins <= 0).any() and (margins > 4).any():
                    target_met = True
                    break
        if target_met or time.monotonic()-started >= args.wait_laser_seconds:
            break
        time.sleep(min(1., max(0., args.wait_laser_seconds-(time.monotonic()-started))))
    transport.update(passive_capture_attempts=capture_attempts,
                     wait_seconds=time.monotonic()-started, requested_sample_present=target_met)
    with _shared_binary(args.checkpoint) as stream:
        data = stream.read()
    import torch
    from fly_touhou.thrl_inputs import SEPARATED_LASER_INPUT
    from fly_touhou.thrl_laser_risk import LaserRiskNetwork, action_clearances, action_penalties
    torch.set_num_threads(2)
    payload = torch.load(io.BytesIO(data), map_location='cpu', weights_only=False)
    configuration = payload['model_config']
    if configuration['input_schema'] != SEPARATED_LASER_INPUT:
        raise ValueError('Do not reinterpret an earlier checkpoint input schema')
    wrapper = configuration['pair_risk_wrapper']
    if not wrapper['frozen_readout']:
        raise ValueError('Diagnostic is scoped to the fixed current laser head')
    network = LaserRiskNetwork(wrapper['laser_width']).eval().requires_grad_(False)
    network.load_state_dict({k.removeprefix('laser_risk.'): v
        for k, v in payload['model'].items() if k.startswith('laser_risk.')}, strict=True)
    valid = ((objects[..., 0] > .5) & ((objects[..., 3] > .5) | (objects[..., 4] > .5))
             & (objects[..., 13] > .5))
    known = valid & (objects[..., 26] > .5)
    unknown = valid & ~known
    known_objects = np.where(known[..., None], objects, 0).astype(np.float32)
    unknown_objects = np.where(unknown[..., None], objects, 0).astype(np.float32)
    f = torch.from_numpy(features)
    with torch.inference_mode():
        predicted = action_clearances(network, f, torch.from_numpy(known_objects)).numpy()
        penalties = action_penalties(network, f, torch.from_numpy(known_objects),
            buffer=wrapper['laser_buffer'], risk_width=wrapper['laser_risk_width']).numpy()
        unknown_penalties = action_penalties(network, f, torch.from_numpy(unknown_objects),
            buffer=wrapper['laser_buffer'], risk_width=wrapper['laser_risk_width']).numpy()
    rows = []
    for world in range(len(features)):
        row = dict(worker=world, stage=int(round(float(features[world, 7])*6)),
            laser_segments=int(valid[world].sum()), known_segments=int(known[world].sum()),
            unknown_segments=int(unknown[world].sum()),
            normal_unprotected=bool(features[world, 21] > .5 and features[world, 20] <= 0
                                    and features[world, 5] <= 0),
            unknown_only_penalty_action_range=float(np.ptp(unknown_penalties[world])))
        if known[world].any():
            origin = np.array([features[world, 0]*368-184, features[world, 1]*400+32])
            gold = route_clearances(features[world], known_objects[world],
                np.tile(origin, (18, 1)), velocities)
            predicted_safe = predicted[world] >= wrapper['laser_buffer']
            dangerous = gold <= 0
            near = np.abs(gold) <= 16
            best = int(np.argmin(penalties[world]))
            row.update(predicted_min_margins=predicted[world].tolist(),
                proxy_min_margins=gold.tolist(), learned_relative_penalties=penalties[world].tolist(),
                modeled_danger_actions=int(dangerous.sum()),
                modeled_danger_predicted_beyond_buffer=int((dangerous & predicted_safe).sum()),
                modeled_buffered_actions=int((gold > wrapper['laser_buffer']).sum()),
                modeled_buffered_predicted_danger=int(((gold > wrapper['laser_buffer']) & ~predicted_safe).sum()),
                boundary_pairs=int(near.sum()), boundary_absolute_error=float(np.abs(predicted[world]-gold)[near].sum()),
                pure_laser_head_preferred_action=best,
                preferred_action_proxy_margin=float(gold[best]),
                laser_penalty_action_range=float(np.ptp(penalties[world])),
                actual_native_action_evaluated=False)
        rows.append(row)
    measured = [r for r in rows if r['known_segments'] and r['normal_unprotected']]
    danger = sum(r['modeled_danger_actions'] for r in measured)
    boundary = sum(r['boundary_pairs'] for r in measured)
    unknown_rows = [r for r in rows if r['unknown_segments']]
    summary = dict(worlds=len(rows), worlds_with_lasers=sum(r['laser_segments'] > 0 for r in rows),
        known_segments=int(known.sum()), unknown_segments=int(unknown.sum()),
        normal_unprotected_known_laser_worlds=len(measured),
        modeled_danger_action_pairs=danger,
        modeled_danger_predicted_beyond_buffer=sum(r['modeled_danger_predicted_beyond_buffer'] for r in measured),
        boundary_pairs=boundary,
        boundary_mean_absolute_error=sum(r['boundary_absolute_error'] for r in measured)/boundary if boundary else None,
        unknown_validation_worlds=len(unknown_rows),
        max_unknown_only_penalty_action_range=max(r['unknown_only_penalty_action_range'] for r in unknown_rows) if unknown_rows else None)
    if unknown_rows and summary['max_unknown_only_penalty_action_range'] > 1e-5:
        raise ValueError('Unknown endpoint motion created unjustified directional preferences')
    args.output.mkdir(parents=True)
    arrays = args.output / 'observations.npz'
    np.savez_compressed(arrays, features=features, objects=objects)
    write(args.output/'report.json', dict(schema='B1-passive-live-laser-risk-proxy-audit-v1',
        status='completed' if target_met else 'requested-sample-not-observed', utc=utc_now(), transport=transport,
        checkpoint=str(args.checkpoint), checkpoint_sha256=hashlib.sha256(data).hexdigest(),
        captured_arrays=str(arrays), arrays_sha256=hashlib.sha256(arrays.read_bytes()).hexdigest(),
        summary=summary, rows=rows, device='cpu', native_instances_started=0,
        native_calls=0, optimizer_updates=0, mappings_modified=False, fit_on_these_states=False,
        native_collision_labels=False, actual_policy_actions=False,
        claim='Readout coverage and conservative geometry proxy agreement only; not native evasion success'))
    print(json.dumps(summary))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    for key in ['ownership', 'checkpoint', 'output']:
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--wait-laser-seconds', type=float, default=0.)
    parser.add_argument('--require-dangerous-alternative', action='store_true')
    run(parser.parse_args())
