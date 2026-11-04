"""Read EX8 bullet populations in an owned game without controlling it."""
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
from fly_touhou.native_bullet_release import predict_bullet_release
from fly_touhou.thrl_study_io import read_shared_json


def capture(ownership: Path, profile_path: Path, world: int, seconds: float, limit: int, *, include_growing=False):
    entry = next(row for row in read_shared_json(ownership)['workers'] if row['index'] == world)
    identity = entry['native_identity']
    process = psutil.Process(identity['pid'])
    if abs(process.create_time()-identity['create_time']) > .01:
        raise ValueError('The owned game process has exited')
    profile = Profile.load(profile_path)
    memory = ProcessMemory(process.pid, writable=False, can_suspend=False)
    reader = object.__new__(VersionedHazardReader)
    reader.mem, reader.profile, reader.o = memory, profile, profile.offsets
    snapshots, seen = [], set()
    deadline = time.monotonic()+seconds
    try:
        while time.monotonic() < deadline and process.is_running() and len(snapshots) < 12:
            try:
                frame = reader.global_frame()
                if frame in seen:
                    time.sleep(.05)
                    continue
                player = reader.pointer('player', 'player_pointer')
                position = memory.pair(player+profile.offsets['player']['zPlayer_pos'])
                candidates = [bullet for bullet in reader._bullets() if (bullet.active or
                              (include_growing and memory.integer(bullet.id+0xc0e,size=2)==2))
                              and -184 <= bullet.x <= 184 and 32 <= bullet.y <= 432
                              and memory.integer(bullet.id+0xc04) == 8]
                candidates.sort(key=lambda bullet: math.hypot(bullet.x-position[0],bullet.y-position[1]))
                raw = [reader._bullet_metadata(bullet) for bullet in candidates[:limit]]
                stage_frame = reader.stage_frame()
                if reader.global_frame() != frame:
                    continue
                seen.add(frame)
                if raw:
                    snapshots.append(dict(frame=frame,stage_frame=stage_frame,bullets=raw))
            except (NativeError, OSError, ValueError):
                pass  # Discard a partial or changing-frame observation.
            time.sleep(.05)
    finally:
        memory.close()
    counts, ids, examples = Counter(), set(), []
    intervals = set()
    maximum_error = 0.
    for first in snapshots:
        for raw in first['bullets']:
            if raw['native_state']==2:
                from fly_touhou.native_ex8_state2_candidate import predict_state2_ex8
                prediction = predict_state2_ex8(raw,24)
            else:
                prediction = predict_bullet_release(raw,24)
            if prediction is None or prediction['mode'] not in ('native_ex8_polar_hint','native_ex8_state2_candidate'):
                continue
            for second in snapshots:
                dt = second['frame']-first['frame']
                if not 1 <= dt <= prediction['conditional_until']:
                    continue
                after = next((value for value in second['bullets'] if value['id'] == raw['id']),None)
                if (after is None or after['native_state'] not in (1,2)
                        or after['native_ex_active_flags'] != 8 or after['native_timer'] != raw['native_timer']+dt):
                    continue
                if (raw['native_state']==1 and (after['native_state']!=1
                        or after['native_ex8_tick']!=raw['native_ex8_tick']+dt)):
                    continue
                a,b=prediction['bounds'][dt];actual=after['native_position']
                dx,dy=b[0]-a[0],b[1]-a[1];denominator=dx*dx+dy*dy
                t=min(1.,max(0.,((actual[0]-a[0])*dx+(actual[1]-a[1])*dy)/denominator)) if denominator else 0.
                radius=prediction.get('extra_radii',[0.]*25)[dt]
                error=max(0.,math.hypot(actual[0]-a[0]-t*dx,actual[1]-a[1]-t*dy)-radius)
                counts['matched_intervals'] += 1
                counts['starting_state_'+str(raw['native_state'])] += 1
                ids.add(raw['id']);intervals.add(dt);maximum_error=max(maximum_error,error)
                if error>.05:
                    counts['over_0_05px'] += 1
                    if len(examples)<4:
                        examples.append(dict(frame=first['frame'],dt=dt,error=error,before=raw,after=after))
    return dict(scope=('Same-frame, read-only original game observations; overlapping intervals '
                       'from the same bullet are not independent trials.'),world=world,
        snapshots=snapshots,summary=dict(**counts,distinct_bullets=len(ids),intervals=sorted(intervals),
        maximum_error_pixels=maximum_error if ids else None),contradictions=examples)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ownership',type=Path,required=True)
    parser.add_argument('--profile',type=Path,required=True)
    parser.add_argument('--world',type=int,required=True)
    parser.add_argument('--seconds',type=float,default=15.)
    parser.add_argument('--limit',type=int,default=16)
    parser.add_argument('--include-growing',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=capture(args.ownership,args.profile,args.world,args.seconds,args.limit,
                   include_growing=args.include_growing)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(world=args.world,snapshots=len(result['snapshots']),**result['summary'])))
