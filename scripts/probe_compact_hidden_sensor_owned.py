"""Try the opt-in sensor against one already-running original process."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import psutil
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.hazard_reader import FULL_GAME_SHA256,GeometryHazardReader
from fly_touhou.native import NativeError,ProcessMemory,Profile
from fly_touhou.compact_spiking_sensation import CompactObservationEncoder
from fly_touhou.thrl_study_io import read_shared_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-pid',type=int,required=True)
    p.add_argument('--environment-config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seconds',type=float,default=25)
    a=p.parse_args()
    games=[v for v in psutil.Process(a.parent_pid).children(recursive=True) if v.name().lower()=='th14.exe']
    if len(games)!=1:raise ValueError('Expected one already-running original game')
    profile=Profile.load(read_shared_json(a.environment_config)['profile'])
    memory=ProcessMemory(games[0].pid,writable=False,can_suspend=False)
    reader=object.__new__(GeometryHazardReader)
    reader.mem,reader.profile,reader.o=memory,profile,profile.offsets
    reader.digest,reader.verified_identity=FULL_GAME_SHA256,True
    def encoder_for_snapshot():
        encoder=CompactObservationEncoder();encoder.native_motion_forecast=True
        encoder.native_curve_laser_geometry=True;encoder.native_focus_gate=True
        encoder.unknown_laser_player_motion=True;encoder.anticipate_arming=True
        encoder.growth_motion_envelope=True;encoder.set_native_episode(1)
        return encoder
    deadline=time.monotonic()+a.seconds;errors=[];result=None
    try:
        while time.monotonic()<deadline and games[0].is_running():
            try:
                snapshot=reader.read(retries=1)
                encoder=encoder_for_snapshot()
                observation=encoder.encode(snapshot)
                original_resources={k:v for k,v in snapshot.extra['resources'].items()
                                    if not k.startswith('hidden_enemy_')}
                original=replace(snapshot,extra={**snapshot.extra,'resources':original_resources})
                old_encoder=encoder_for_snapshot()
                old_observation=old_encoder.encode(original)
                if reader.global_frame()!=snapshot.frame:continue
                resources=snapshot.extra['resources']
                encoded=encoder.last_result
                result=dict(status='observed',stage=snapshot.stage,frame=snapshot.frame,
                    stage_frame=snapshot.stage_frame,visible_enemies=len(resources['enemies']),
                    hidden_enemies=len(resources.get('hidden_enemy_contacts',[])),
                    hidden_geometry_unreadable=resources.get('hidden_enemy_unreadable_geometry'),
                    feature_shape=list(observation['features'].shape),
                    encoded_contact_bodies=len(encoded['source_native_forecasts']['enemies']),
                    encoded_hidden_contact_bodies=encoded['details']['uncertainty']['hidden_enemy_contact_bodies'],
                    trusted=encoded['details']['trusted'].tolist(),
                    previous_trusted=old_encoder.last_result['details']['trusted'].tolist(),
                    maximum_feature_change=float(np.max(np.abs(observation['features']-old_observation['features']))),
                    scope='Passive snapshot of one already-running game; no game inputs, pauses or writes')
                break
            except (NativeError,ValueError,OSError) as error:
                errors.append(str(error));time.sleep(.1)
    finally:memory.close()
    if result is None:
        result=dict(status='no_coherent_snapshot',errors=errors[-3:])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
