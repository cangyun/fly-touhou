"""Record clean original-game route intervals across all six stages.

The rule teacher may use resources between intervals to reach later patterns.
Only unprotected, no-contact, no-Bomb intervals enter the atlas. This is not
student evaluation, PPO sampling, or evidence of a full NBNM clear.
"""
from pathlib import Path
import argparse
import concurrent.futures
import json
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]


def collect(job):
    spec, stage, seed, folder = job
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    from fly_touhou.nbnm_run import environment, runtime
    from fly_touhou.compact_full_story import CompactFullStoryEnv, story_kwargs
    from fly_touhou.thrl_native_env import THRLNativeEnv
    from fly_touhou.nbnm_stage_practice import bind_controller
    from fly_touhou.nbnm_spell_practice import copy_score
    from fly_touhou.nbnm_route_atlas import phase_progress, route_windows
    from fly_touhou.thrl_durable_records import write_json
    from fly_touhou.thrl_job_paths import cleanup_native_temp
    from fly_touhou.resource_teacher_v10 import REVISION
    from fly_touhou.thrl_training_mode import seed_split
    if seed_split(seed) != 'train':
        raise ValueError('Atlas collection only accepts training seeds')
    temporary = runtime(folder)

    class RouteEnv(CompactFullStoryEnv):
        def _create_controller(self):
            controller = super()._create_controller()
            return bind_controller(controller, dict(stage=stage)) if stage else controller
        def _validate_initial_run(self):
            self._verify_native_run(self.snapshot)
        def _verify_native_run(self, snapshot):
            if not stage:
                return super()._verify_native_run(snapshot)
            if (snapshot.extra.get('run_mode') != 1 or snapshot.stage != stage
                    or snapshot.difficulty != 3 or snapshot.character != 0 or snapshot.subshot != 0):
                raise ValueError('Expected original Lunatic ReimuA Stage Practice')
        def _native_end(self, snapshot):
            if not stage:
                return super()._native_end(snapshot)
            if snapshot.extra.get('native_gameover') or snapshot.lives < 0:
                return 1
            return 2 if snapshot.pause == 1 or snapshot.screen != 7 else 0

    cfg = environment(spec, evaluation=True)
    cfg.pop('nbnm', None)
    options = story_kwargs(cfg, folder, 1, 'train', teacher=True)[0]
    options.pop('online_teacher_revision', None)
    env = RouteEnv(**options, teacher_revision=REVISION)
    env.native_seed = seed
    env.private_appdata = folder/'appdata'
    copy_score(spec['nbnm']['score_template'], env.private_appdata)
    rows = []
    started = time.monotonic()
    try:
        _, info = THRLNativeEnv.reset(env)
        info = env._label_current(env._info())
        for decision in range(18000):
            snapshot = env.snapshot
            phase, clock = phase_progress(snapshot)
            f = env.encoder.last_result['features']
            action = int(info['teacher_label']['action'])
            events = snapshot.extra['thrl_events']
            row = dict(phase=phase, clock=clock, frame=snapshot.frame,
                position=[snapshot.player.x, snapshot.player.y], power=float(f[2]),
                signature=f[18:72].tolist(), native_seed=seed, clean=False)
            _, _, done, truncated, info = env.step(action)
            following = env.snapshot
            after = following.extra['thrl_events']
            row['clean'] = bool(action < 18 and snapshot.player.state == following.player.state == 1
                and snapshot.player.invulnerable_frames == following.player.invulnerable_frames == 0
                and following.frame-snapshot.frame == 4
                and events['misses'] == after['misses'] and events['bombs_used'] == after['bombs_used'])
            rows.append(row)
            if decision % 64 == 0:
                write_json(folder/'progress.json', dict(decisions=decision+1, stage=following.stage,
                    stage_frame=following.stage_frame, phase=phase, teacher_only=True,
                    misses=after['misses'], Bombs=after['bombs_used']))
            if done or truncated:
                break
        records = route_windows(rows)
        write_json(folder/'route-records.json', records)
        report = dict(stage=stage or 'full_story', seed=seed, decisions=len(rows),
            clean_route_rows=len(records), phases=sorted({r['phase'] for r in records}),
            elapsed_seconds=time.monotonic()-started, teacher_only=True,
            full_game_NBNM_evidence=False, network_training=False,
            rule_teacher_misses=after['misses'], rule_teacher_Bombs=after['bombs_used'])
        report.update(terminated=bool(done), truncated=bool(truncated),
            decision_limit_reached=not (done or truncated),
            final_stage=following.stage, final_stage_frame=following.stage_frame,
            final_phase=phase_progress(following)[0])
        write_json(folder/'report.json', report)
        return report
    except BaseException:
        write_json(folder/'failure.json', dict(error=traceback.format_exc()))
        raise
    finally:
        env.close()
        cleanup_native_temp(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stages', type=int, nargs='+', default=[1,2,3,4,5,6])
    parser.add_argument('--seeds', type=int, nargs='+', default=[50000,50001])
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    from fly_touhou.nbnm_run import read
    from fly_touhou.nbnm_route_atlas import write_atlas
    from fly_touhou.thrl_durable_records import write_json
    args.output.mkdir(parents=True, exist_ok=True)
    spec = read(args.config)
    jobs = [(spec, stage, seed, str(args.output/f's{stage}-train-{seed}'))
            for stage in args.stages for seed in args.seeds]
    reports = []
    todo = []
    for job in jobs:
        path = Path(job[3])/'report.json'
        if path.exists():
            reports.append(read(path))
        else:
            todo.append(job)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in concurrent.futures.as_completed([pool.submit(collect, job) for job in todo]):
            reports.append(future.result())
            write_json(args.output/'progress.json', dict(completed=len(reports), total=len(jobs),
                teacher_only=True, network_training=False))
    records = [row for job in jobs for row in read(Path(job[3])/'route-records.json')]
    manifest = write_atlas(records, args.output, sources=reports)
    write_json(args.output/'report.json', dict(status='complete', rows=manifest['rows'],
        phase_count=len(manifest['phases']), sources=reports, network_training=False))


if __name__ == '__main__':
    main()
