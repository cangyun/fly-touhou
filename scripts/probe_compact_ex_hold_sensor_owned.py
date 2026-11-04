"""Compare EX-hold sensor inputs on one unchanged, already-running game state."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import numpy as np
import psutil

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.compact_spiking_sensation import CompactObservationEncoder
from fly_touhou.hazard_reader import FULL_GAME_SHA256,GeometryHazardReader
from fly_touhou.native import NativeError,ProcessMemory,Profile
from fly_touhou.thrl_study_io import read_shared_json
from fly_touhou.thrl_durable_records import write_json


def encoder():
    result=CompactObservationEncoder()
    result.native_motion_forecast=True
    result.native_curve_laser_geometry=True
    result.native_focus_gate=True
    result.unknown_laser_player_motion=True
    result.anticipate_arming=True
    result.growth_motion_envelope=True
    result.set_native_episode(1)
    return result


def without_hold_forecast(snapshot):
    metadata=snapshot.extra['hazard_metadata']
    bullets=[dict(row,native_ex_hold_forecast_enabled=False) for row in metadata['bullets']]
    extra=dict(snapshot.extra,hazard_metadata=dict(metadata,bullets=bullets))
    return replace(snapshot,extra=extra)


def summarize(result):
    bullets=result['source_native_forecasts']['bullets']
    held=[row['release'] for row in bullets if row['release'].get('mode')=='native_ex_position_hold']
    return dict(trusted=result['details']['trusted'].tolist(),
        unresolved=result['details']['uncertainty']['unresolved_release_points_4_12_24'],
        held_predictions=len(held),
        held_known_by_horizon={str(h):sum(len(x['known_by_frame'])>h and x['known_by_frame'][h]
                                           for x in held) for h in (4,12,24)})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-pid',type=int,required=True)
    parser.add_argument('--environment-config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seconds',type=float,default=30.)
    args=parser.parse_args()
    games=[p for p in psutil.Process(args.parent_pid).children(recursive=True) if p.name().lower()=='th14.exe']
    if len(games)!=1:raise ValueError('Expected one owned original game')
    profile=Profile.load(read_shared_json(args.environment_config)['profile'])
    profile.hooks=dict(profile.hooks or {},native_position_hold_forecast='th14-ex-position-hold-v1')
    memory=ProcessMemory(games[0].pid,writable=False,can_suspend=False)
    reader=object.__new__(GeometryHazardReader)
    reader.mem,reader.profile,reader.o=memory,profile,profile.offsets
    reader.digest,reader.verified_identity=FULL_GAME_SHA256,True
    deadline=time.monotonic()+args.seconds
    result=None;errors=[]
    try:
        while time.monotonic()<deadline and games[0].is_running():
            try:
                snapshot=reader.read(retries=1)
                bullets=snapshot.extra['hazard_metadata']['bullets']
                count=sum(row.get('native_ex_active_flags')==0x04000000 for row in bullets)
                if not count:time.sleep(.1);continue
                new_encoder=encoder();new_obs=new_encoder.encode(snapshot)
                old_encoder=encoder();old_obs=old_encoder.encode(without_hold_forecast(snapshot))
                if reader.global_frame()!=snapshot.frame:continue
                result=dict(status='observed',stage=snapshot.stage,stage_frame=snapshot.stage_frame,
                    frame=snapshot.frame,hold_bullets=count,
                    original=summarize(old_encoder.last_result),patched=summarize(new_encoder.last_result),
                    maximum_feature_change=float(np.max(np.abs(new_obs['features']-old_obs['features']))),
                    scope='One coherent read-only snapshot; no alternate actions or outcome comparison')
                break
            except (NativeError,ValueError,OSError) as exc:
                errors.append(str(exc));time.sleep(.1)
    finally:memory.close()
    if result is None:result=dict(status='no_coherent_hold_snapshot',errors=errors[-3:])
    write_json(args.output,result)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
