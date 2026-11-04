"""Observe held EX bullets in an already-running owned game, without control."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import time

import psutil

from fly_touhou.hazard_reader import VersionedHazardReader
from fly_touhou.native import NativeError, ProcessMemory, Profile
from fly_touhou.native_bullet_release import predict_position_hold_candidate
from fly_touhou.thrl_study_io import read_shared_json


def capture(ownership, profile_path, world, seconds, limit, *, parent_pid=None,
            stage=None, minimum_stage_frame=0, follow=False, max_snapshots=18):
    if parent_pid is None:
        entry=next(row for row in read_shared_json(ownership)['workers'] if row['index']==world)
        identity=entry['native_identity']
    else:
        # A single-game evaluator owns one native child rather than a vector
        # ownership file. Only observe that evaluator's actual game child.
        games=[p for p in psutil.Process(parent_pid).children(recursive=True)
               if p.name().lower()=='th14.exe']
        if len(games)!=1:raise ValueError('Exactly one owned original game child required')
        identity=dict(pid=games[0].pid,create_time=games[0].create_time())
    process=psutil.Process(identity['pid'])
    if abs(process.create_time()-identity['create_time'])>.01:
        raise ValueError('The owned game process has exited')
    profile=Profile.load(profile_path)
    profile.hooks=dict(profile.hooks or {},native_position_hold_forecast='th14-ex-position-hold-v1')
    memory=ProcessMemory(process.pid,writable=False,can_suspend=False)
    reader=object.__new__(VersionedHazardReader)
    reader.mem,reader.profile,reader.o=memory,profile,profile.offsets
    snapshots=[];seen=set();tracked=set();deadline=time.monotonic()+seconds
    try:
        while time.monotonic()<deadline and process.is_running() and len(snapshots)<max_snapshots:
            try:
                frame=reader.global_frame()
                if frame in seen:
                    time.sleep(.05);continue
                current_stage=reader.static('environment','stage')
                stage_frame=reader.stage_frame()
                if stage is not None and current_stage>stage:break
                if (stage is not None and current_stage<stage) or stage_frame<minimum_stage_frame:
                    time.sleep(.5);continue
                bullets=[]
                for bullet in reader._bullets():
                    tracked_bullet=follow and bullet.id in tracked
                    new_hold=(not follow or not tracked) and bullet.active and memory.integer(bullet.id+0xc04)==0x04000000
                    if tracked_bullet or new_hold:
                        bullets.append(reader._bullet_metadata(bullet))
                        if len(bullets)>=limit:break
                if reader.global_frame()!=frame:continue
                seen.add(frame)
                if bullets:
                    if follow and not tracked:tracked={b['id'] for b in bullets}
                    snapshots.append(dict(frame=frame,stage=current_stage,
                                          stage_frame=stage_frame,bullets=bullets))
            except (NativeError,OSError,ValueError):
                pass
            time.sleep(.05)
    finally:
        memory.close()
    counts=Counter();ids=set();intervals=set();maximum=0.;contradictions=[];boundary_maximum=0.
    for first in snapshots:
        for raw in first['bullets']:
            prediction=predict_position_hold_candidate(raw,24)
            if prediction is None:
                counts['unsupported_snapshots']+=1;continue
            counts['candidate_snapshots']+=1
            for second in snapshots:
                dt=second['frame']-first['frame']
                if not 1<=dt<=24 or not prediction['known_by_frame'][dt]:continue
                after=next((v for v in second['bullets'] if v['id']==raw['id']),None)
                if after is None or after['native_timer']!=raw['native_timer']+dt:continue
                error=math.dist(prediction['positions'][dt],after['native_position'])
                counts['matched_intervals']+=1;ids.add(raw['id']);intervals.add(dt);maximum=max(maximum,error)
                if dt==raw['native_ex_hold_remaining']:
                    counts['at_countdown_boundary']+=1;boundary_maximum=max(boundary_maximum,error)
                if error>.05:
                    counts['outside_0_05px']+=1
                    if len(contradictions)<5:
                        contradictions.append(dict(frame=first['frame'],dt=dt,error=error,before=raw,after=after))
    return dict(scope='Read-only samples of an already-running original game;no input,suspension,or game writes',
        world=world,game_pid=identity['pid'],snapshots=snapshots,summary=dict(**counts,distinct_bullets=len(ids),
            frame_intervals=sorted(intervals),maximum_error_pixels=maximum if ids else None,
            boundary_maximum_error_pixels=boundary_maximum if counts['at_countdown_boundary'] else None),
        contradictions=contradictions)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--ownership',type=Path)
    source.add_argument('--parent-pid',type=int)
    p.add_argument('--profile',type=Path,required=True)
    p.add_argument('--world',type=int,default=0)
    p.add_argument('--seconds',type=float,default=20.)
    p.add_argument('--limit',type=int,default=24)
    p.add_argument('--stage',type=int)
    p.add_argument('--minimum-stage-frame',type=int,default=0)
    p.add_argument('--follow',action='store_true')
    p.add_argument('--max-snapshots',type=int,default=18)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    result=capture(a.ownership,a.profile,a.world,a.seconds,a.limit,parent_pid=a.parent_pid,
                   stage=a.stage,minimum_stage_frame=a.minimum_stage_frame,
                   follow=a.follow,max_snapshots=a.max_snapshots)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(world=a.world,snapshots=len(result['snapshots']),**result['summary'])))


if __name__=='__main__':main()
