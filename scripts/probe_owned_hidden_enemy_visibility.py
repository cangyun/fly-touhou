"""Observe hidden and visible original enemies in an existing private game."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from fly_touhou.native import NativeError, NativeReader, ProcessMemory, Profile
from fly_touhou.native_enemy_contact import read_enemy_contact
from fly_touhou.thrl_study_io import read_shared_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-pid', type=int, required=True)
    p.add_argument('--environment-config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seconds', type=float, default=45)
    p.add_argument('--stage', type=int, default=4)
    p.add_argument('--minimum-stage-frame',type=int,default=0)
    p.add_argument('--max-snapshots',type=int,default=0)
    a = p.parse_args()
    children = [x for x in psutil.Process(a.parent_pid).children(recursive=True)
                if x.name().lower() == 'th14.exe']
    if len(children) != 1:
        raise ValueError('Expected one original-game child of this evaluator')
    process = children[0]
    profile = Profile.load(Path(read_shared_json(a.environment_config)['profile']))
    memory = ProcessMemory(process.pid, writable=False, can_suspend=False)
    reader = object.__new__(NativeReader)
    reader.mem, reader.profile, reader.o = memory, profile, profile.offsets
    start = time.monotonic(); frames = []; previous = None; changes = []; seen = set()
    try:
        while time.monotonic()-start < a.seconds and process.is_running():
            try:
                frame = reader.global_frame()
                if frame in seen:
                    time.sleep(.1); continue
                stage = reader.static('environment', 'stage')
                if stage > a.stage:
                    break
                if stage != a.stage:
                    time.sleep(.5); continue
                if reader.stage_frame() < a.minimum_stage_frame:
                    time.sleep(.5); continue
                pstate = reader._player()
                if pstate is None:
                    time.sleep(.1); continue
                o = reader.o['enemies']
                manager = reader.pointer('enemies', 'enemy_manager_pointer')
                identities = list(reader._entries(memory.integer(manager+o['zEnemyManager_list']), limit=2048))
                enemies = {};spawn_sources=[]
                for identity in identities:
                    d = identity + o['zEnemy_data']
                    flags = memory.integer(d+o['zEnemyData_flags'], size=8)
                    boss = bool(flags & o['zEnemyFlags_is_boss'])
                    animation = memory.integer(d+o['zEnemyData_anm_vm_id'])
                    x, y = memory.pair(d+o['zEnemyData_pos'])
                    hx, hy = memory.pair(d+o['zEnemyData_hitbox'])
                    if not all(math.isfinite(v) for v in (x,y,hx,hy)):
                        continue
                    visible = boss or bool(animation)
                    distance = math.hypot(x-pstate.x, y-pstate.y)
                    row = dict(id=identity, visible=visible, animation=animation, flags=flags,
                               position=[x,y], size=[hx,hy], distance=distance)
                    if not visible or boss:
                        contact = read_enemy_contact(memory, identity, d, image_base=memory.base,
                                                     horizon=4, capture_spawns=True)
                        instructions=[item for thread in contact['ecl_threads']
                                      for item in thread['spawn_instructions']]
                        if instructions:
                            spawn_sources.append(dict(id=identity,visible=visible,position=[x,y],
                                                      instructions=instructions))
                    if not visible and distance < 96:
                        row['contactable_next4'] = [tick['contactable'] for tick in contact['forecast']['ticks']]
                        row['contact_timer'] = contact['contact_timer']['integer']
                    enemies[identity] = row
                if reader.global_frame() != frame:
                    continue
                if previous is not None:
                    for identity, row in enemies.items():
                        prior = previous['enemies'].get(identity)
                        if row['distance'] < 128 and (prior is None or prior['visible'] != row['visible']):
                            changes.append(dict(frame=frame, prior_frame=previous['frame'],
                                                before=prior, after=row))
                hidden = [row for row in enemies.values() if not row['visible']]
                near_hidden = sorted((row for row in hidden if row['distance'] < 96),
                                     key=lambda row: row['distance'])[:12]
                frames.append(dict(frame=frame, stage_frame=reader.stage_frame(),
                                   player=[pstate.x,pstate.y], state=pstate.state,
                                   enemy_count=len(enemies), visible_count=len(enemies)-len(hidden),
                                   hidden_count=len(hidden), near_hidden=near_hidden,
                                   observed_spawn_instructions=spawn_sources))
                previous = dict(frame=frame, enemies=enemies)
                seen.add(frame)
                if a.max_snapshots and len(frames)>=a.max_snapshots:
                    break
            except (NativeError,OSError,ValueError):
                pass
            time.sleep(.1)
    finally:
        memory.close()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(scope='Read-only observation of an existing game; no input, suspension or game writes',
                  game_pid=process.pid, frames=frames, near_visibility_changes=changes)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(frames=len(frames),
        hidden_near_samples=sum(len(row['near_hidden']) for row in frames),
        visibility_changes=len(changes))))


if __name__ == '__main__':
    main()
