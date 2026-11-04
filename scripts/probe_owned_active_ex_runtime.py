"""Read active EX bullet metadata from a currently owned game, without control."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import struct
import time

import psutil

from fly_touhou.hazard_reader import VersionedHazardReader
from fly_touhou.native import NativeError, ProcessMemory, Profile
from fly_touhou.native_bullet_release import predict_bullet_release
from fly_touhou.thrl_study_io import read_shared_json


def probe(ownership: Path, profile_path: Path, world: int, seconds: float, *, unsupported=False, flag=None):
    profile = Profile.load(profile_path)
    entry = next(row for row in read_shared_json(ownership)['workers'] if row['index'] == world)
    identity = entry['native_identity']
    process = psutil.Process(identity['pid'])
    if abs(process.create_time() - identity['create_time']) > .01:
        raise ValueError('The owned game process has exited')
    memory = ProcessMemory(process.pid, writable=False, can_suspend=False)
    reader = object.__new__(VersionedHazardReader)
    reader.mem, reader.profile, reader.o = memory, profile, profile.offsets
    samples, selected, seen = [], None, set()
    until = time.monotonic() + seconds
    try:
        while time.monotonic() < until and process.is_running():
            try:
                frame = reader.global_frame()
                if frame in seen:
                    time.sleep(.05)
                    continue
                bullets = reader._bullets()
                if selected is None:
                    def matches(bullet):
                        if not bullet.active and memory.integer(bullet.id + 0xc0e, size=2) != 2:
                            return False
                        flags = memory.integer(bullet.id + 0xc04)
                        if flag is not None:
                            return flags == flag
                        return (flags != 0 and flags not in (1, 4, 0x200000, 0x80000000)
                                if unsupported else flags == 4)
                    player = reader.pointer('player', 'player_pointer')
                    player_xy = memory.pair(player + profile.offsets['player']['zPlayer_pos'])
                    candidates = [bullet for bullet in bullets if matches(bullet)
                                  and -184 <= bullet.x <= 184 and 32 <= bullet.y <= 432]
                    if candidates:
                        selected = min(candidates, key=lambda bullet: math.hypot(
                            bullet.x-player_xy[0], bullet.y-player_xy[1])).id
                target = next((bullet for bullet in bullets if bullet.id == selected), None)
                if target is None or (not target.active and memory.integer(target.id+0xc0e, size=2) != 2):
                    selected = None
                    time.sleep(.05)
                    continue
                raw = reader._bullet_metadata(target)
                if raw.get('native_ex_active_flags', 0) & 8:
                    tick, elapsed = struct.unpack('<if', memory.read(target.id + 0xfd4, 8))
                    speed_delta, angle_delta = struct.unpack('<ff', memory.read(target.id + 0xfe4, 8))
                    duration = memory.integer(target.id + 0x1004, signed=True)
                    raw.update(native_ex8_tick=tick, native_ex8_time=elapsed,
                               native_ex8_speed_delta=speed_delta,
                               native_ex8_angle_delta=angle_delta, native_ex8_duration=duration)
                stage_frame = reader.stage_frame()
                if reader.global_frame() != frame:
                    continue
                seen.add(frame)
                samples.append(dict(frame=frame, stage_frame=stage_frame, raw=raw))
                if len(samples) >= 10:
                    break
            except (NativeError, OSError, ValueError):
                # A live game can advance during a read; discard that sample.
                pass
            time.sleep(.05)
    finally:
        memory.close()
    counts = Counter()
    comparisons = []
    for first in samples:
        prediction = predict_bullet_release(first['raw'], horizon=24)
        counts['source_' + str(first['raw'].get('native_ex_acceleration_source'))] += 1
        counts['reason_' + prediction['reason']] += 1
        for second in samples:
            dt = second['frame'] - first['frame']
            if not 1 <= dt <= 24 or second['raw']['id'] != first['raw']['id']:
                continue
            if not prediction['known_by_frame'][dt]:
                counts['unknown_pairs'] += 1
                continue
            counts['known_pairs'] += 1
            a, b = prediction['bounds'][dt]
            point = second['raw']['native_position']
            dx, dy = b[0]-a[0], b[1]-a[1]
            denominator = dx*dx + dy*dy
            t = min(1., max(0., ((point[0]-a[0])*dx + (point[1]-a[1])*dy)/denominator)) if denominator else 0.
            error = max(0., math.hypot(point[0]-a[0]-t*dx, point[1]-a[1]-t*dy)
                        - prediction.get('extra_radii', [0.]*25)[dt])
            if error > .05:
                counts['outside_envelope'] += 1
                if len(comparisons) < 5:
                    comparisons.append(dict(start_frame=first['frame'], elapsed=dt, error=error,
                                            first=first['raw'], observed=second['raw']))
    return dict(scope='Read-only active owned original game; no input, suspension or writes to game.',
                world=world, samples=samples, comparisons=counts, contradictions=comparisons)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ownership', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--world', type=int, required=True)
    parser.add_argument('--seconds', type=float, default=35.)
    parser.add_argument('--unsupported', action='store_true',
                        help='Read an active EX family not yet modeled by the predictor')
    parser.add_argument('--flag', type=lambda value: int(value, 0),
                        help='Read one exact native EX active-mask value')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = probe(args.ownership, args.profile, args.world, args.seconds,
                   unsupported=args.unsupported, flag=args.flag)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(world=result['world'], samples=len(result['samples']),
                          comparisons=result['comparisons']), ensure_ascii=False))
