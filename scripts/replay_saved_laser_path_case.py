"""Replay a recorded contact's current laser shape under two player-path hints."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from fly_touhou.compact_spiking_sensation import point_and_laser_clearances
from fly_touhou.thrl_movement_teacher import MovementSafetyTeacher


def analyze(path: Path):
    torch.set_num_threads(2)
    event = torch.load(path,map_location='cpu',weights_only=False)
    summary, transition = event['summary'], event['physical_transition']
    if transition is None or not transition['geometry_replay_before']['available']:
        raise ValueError('This event has no saved pre-contact geometry')
    action = summary['last_action']
    if not 0 <= action < 18:
        raise ValueError('An executed movement is required')
    geometry = transition['geometry_replay_before']
    f = np.asarray(geometry['source_features25'],np.float32)
    objects = np.asarray(geometry['source_objects29'],np.float32)
    start = np.array([[f[0]*368-184,f[1]*400+32]])
    velocities = MovementSafetyTeacher().velocities
    velocity = velocities[[action % 9 if f[18] < 0 else action]].copy()
    duration = transition['native_frame_gap']
    if not 1 <= duration <= 24:
        raise ValueError('The recorded native action duration is outside the replay horizon')
    arguments=dict(anticipate_arming=geometry['forecast_native_arming'],
        growth_motion_envelope=geometry['forecast_growth_motion'],
        native_forecasts=geometry.get('source_native_forecasts'),max_laser_margin=64.)
    old = point_and_laser_clearances(f,objects,start,velocity,duration=duration,
        unknown_laser_player_motion=False,**arguments)[0]
    moving = {str(t):point_and_laser_clearances(f,objects,start,velocity,duration=t,
        unknown_laser_player_motion=True,**arguments)[0] for t in sorted({1,min(2,duration),duration})}
    endpoint = np.clip(start[0]+velocity[0]*duration,[-184,32],[184,432])
    actual = np.asarray(transition['player_after']['position'])
    value = lambda x: float(x) if math.isfinite(x) else None
    return dict(scope=('Saved pre-contact geometry only. The new calculation moves the player against '
        'the recorded laser shape; it does not reconstruct future laser motion or native node rectangles. '
        'No candidate is certified safe, and no new game or model action is executed.'),
        source=str(path),event=summary['event'],stage=summary['before']['stage'],
        stage_frame=summary['before']['stage_frame'],action=action,native_frames=duration,
        recorded_margin=summary['chosen_immediate_margin'],replayed_legacy_margin=value(old),
        conditional_player_path_margin_by_horizon={k:value(v) for k,v in moving.items()},
        recorded_unknown_laser_segments=summary['pre_action_uncertainty']['unknown_laser_segments'],
        actual_player_after=actual.tolist(),player_state_after=transition['player_after']['state'],
        hypothetical_no_contact_endpoint=endpoint.tolist(),
        actual_vs_no_contact_endpoint_difference_pixels=float(np.linalg.norm(endpoint-actual)),
        endpoint_scope='The no-contact endpoint assumes movement for the entire action; an observed contact can stop movement early.')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('event',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=analyze(args.event)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))
