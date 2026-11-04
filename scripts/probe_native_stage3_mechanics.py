"""Focused original-game motion/contact/release capture; no learning or input UI."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]


def finite(value):
    if isinstance(value, dict): return {str(k): finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [finite(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def run(config_path, output):
    from fly_touhou.compact_full_story import story_kwargs, factory
    from fly_touhou.native_enemy_motion import read_enemy_motion, predict_enemy_positions
    from fly_touhou.native_enemy_contact import read_enemy_contact
    from fly_touhou.thrl_durable_records import write_json
    output.mkdir(parents=True, exist_ok=True)
    os.environ['FLY_TOUHOU_DESKTOP_NAME'] = f'fly-touhou-job-stage3-mechanics-{os.getpid()}'
    config = json.loads(config_path.read_text(encoding='utf-8-sig'))
    options = story_kwargs(config, output, 1, 'train', teacher=True)[0]
    options['private_appdata'] = str(Path(os.environ['TEMP']) / f'fly-stage3probe-{os.getpid()}')
    env = factory(options)
    started = time.monotonic()
    records = samples = 0
    try:
        with (output / 'stage3.jsonl').open('w', encoding='utf-8') as stream:
            for episode in range(3):
                _, info = env.reset(seed=202610340 + episode)
                stage3_start = None
                while True:
                    s = env.snapshot
                    if s.stage == 3 and s.player is not None:
                        if stage3_start is None: stage3_start = s.frame
                        memory = env.controller._memory
                        enemies = []
                        for e in s.extra['resources']['enemies']:
                            if not e['boss']: continue
                            address = e['id'] + env.controller.profile.offsets['enemies']['zEnemy_data']
                            motion = read_enemy_motion(memory, address)
                            enemies.append(dict(resource=e, motion=motion,
                                predicted_motion=predict_enemy_positions(motion),
                                contact=read_enemy_contact(memory, e['id'], address,
                                    image_base=memory.base)))
                        bullets = s.extra['hazard_metadata']['bullets']
                        growth = [b for b in bullets if b['native_state'] == 2 or b.get('native_ex_active_flags', 0) & 1]
                        record = dict(episode=episode, frame=s.frame, stage=s.stage,
                            stage_frame=s.stage_frame, player=dict(x=s.player.x, y=s.player.y,
                                state=s.player.state, protection=s.player.invulnerable_frames),
                            enemies=enemies, growth=growth,
                            tracked_bullets=[dict(id=b.id,x=b.x,y=b.y,vx=b.vx,vy=b.vy,age=b.age,
                                active=b.active) for b in s.bullets],
                            teacher_action=env._teacher_label['action'])
                        stream.write(json.dumps(finite(record), ensure_ascii=False, allow_nan=False)+'\n')
                        samples += 1
                        if samples % 32 == 0: stream.flush()
                    action = int(env._teacher_label['action'])
                    # Probe-only finer sampling. Training stays at four native frames.
                    env.frame_skip = 2 if action == 18 else 1 if s.stage == 3 else 4
                    _, _, done, cut, info = env.step(action)
                    records += 1
                    if records % 128 == 0 or done:
                        write_json(output / 'status.json', dict(phase='sampling', pid=os.getpid(),
                            episode=episode, stage=env.snapshot.stage, frame=env.snapshot.frame,
                            stage_frame=env.snapshot.stage_frame, decisions=records,
                            stage3_samples=samples, seconds=time.monotonic()-started))
                    if done or cut: break
                    if stage3_start is not None and (env.snapshot.stage != 3 or env.snapshot.frame-stage3_start >= 18000):
                        write_json(output / 'result.json', dict(complete=True, stage3_samples=samples,
                            decisions=records, seconds=time.monotonic()-started,
                            next_stage=env.snapshot.stage, scope='native mechanics capture; CPU teacher controls; not fly evaluation'))
                        return
                if samples:
                    write_json(output / 'result.json', dict(complete=True, stage3_samples=samples,
                        decisions=records, seconds=time.monotonic()-started,
                        terminal=info.get('native_terminal_kind'),
                        scope='native mechanics capture; CPU teacher controls; not fly evaluation'))
                    return
        write_json(output / 'result.json', dict(complete=False, reason='teacher did not reach stage3',
            stage3_samples=samples, decisions=records, seconds=time.monotonic()-started))
    except BaseException as error:
        write_json(output / 'failure.json', dict(type=type(error).__name__, message=str(error),
            stage3_samples=samples, decisions=records))
        raise
    finally:
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.output)
