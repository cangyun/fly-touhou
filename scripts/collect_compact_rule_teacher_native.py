"""Record one rule-teacher full game through the repaired compact sensor.

The rule teacher selects original-game actions.  No fly model is evaluated and
no PPO samples are produced.  A private headless original instance is used.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]


def collect(env, output, *, chunk_rows=256, progress=lambda **kw: None):
    import numpy as np
    from fly_touhou.thrl_durable_records import write_json
    obs, info = env.reset()
    buffer = []
    chunks = []
    decisions = 0
    start = time.monotonic()

    def flush():
        if not buffer:
            return
        arrays = {key: np.asarray([row[key] for row in buffer]) for key in buffer[0]}
        path = output / f'part-{len(chunks):05}.npz'
        with path.open('wb') as stream:
            np.savez_compressed(stream, **arrays)
        chunks.append(dict(file=path.name, rows=len(buffer),
            first_stage=int(arrays['stage'][0]), last_stage=int(arrays['stage'][-1]),
            first_decision=decisions-len(buffer)))
        buffer.clear()
        write_json(output/'index.json', dict(schema='native-compact129-rule-teacher-episode-v1',
            status='in_progress', chunks=chunks, rows=decisions, teacher_controlled=True, PPO_samples=0))

    while True:
        features = obs['features'].copy()
        label, target = info['teacher_label'], info['compact_targets']
        action = int(label['action'])
        before = info
        obs, _, done, truncated, info = env.step(action)
        if truncated:
            raise RuntimeError('Rule teacher game was truncated, not naturally ended')
        elapsed = int(info['episode_frames'] - before['episode_frames'])
        hit = bool(info['misses'] > before['misses'] or
                   (obs['features'][17] > .5 and features[17] < .5))
        outcome = int(hit) if features[14]>.5 and action < 18 else -1
        margins = np.asarray(target['margins'], np.float32)
        trusted = np.asarray(target['trusted'], bool).copy()
        from fly_touhou.compact_native_contact_targets import native_exposure_mask
        if outcome >= 0 and native_exposure_mask(features,target['normal']):
            chosen = margins[0, action]
            if (outcome == 1 and 0<elapsed<=4 and chosen > 0) or (outcome == 0 and elapsed==4 and chosen < 0):
                trusted[:] = False  # The executed native transition vetoes a conflicting forecast.
        from fly_touhou.compact_partial_risk import veto_selected_clear
        partial,_=veto_selected_clear(np.asarray(target.get('partial_point_danger',np.zeros((3,18),bool))),
            np.asarray(action),np.asarray(outcome),np.asarray(target['normal']),np.asarray(elapsed))
        buffer.append(dict(features=features, targets=np.asarray(label['probabilities'], np.float32),
            actions=action, episode_starts=(decisions == 0),
            teacher_guidance_weight=float(label.get('guidance_weight', 1.)),
            margins=margins, trusted=trusted, unprotected_mask=bool(target['normal']),
            partial_point_danger=partial,
            outcome=outcome, native_action_frames=elapsed, stage=int(before['stage']),
            stage_frame=int(before['stage_frame']), before_episode_frames=int(before['episode_frames']),
            misses_before=int(before['misses']), bombs_used_before=int(before['bombs_used'])))
        decisions += 1
        if len(buffer) >= chunk_rows or done:
            flush()
        if decisions % 64 == 0 or done:
            progress(phase='teacher_playing_original', decisions=decisions, stage=int(info['stage']),
                stage_frame=int(info['stage_frame']), misses=int(info['misses']),
                bombs_used=int(info['bombs_used']), seconds=time.monotonic()-start)
        if done:
            break
    summary = info.get('episode_summary', {})
    from fly_touhou.compact_native_forecast import native_repair_revisions
    from fly_touhou.compact_teacher_support import REVISION as support_revision
    profile=getattr(getattr(env,'controller',None),'profile',None)
    wall_revision=(getattr(profile,'hooks',None) or {}).get('native_wall_reactive_forecast')
    result = dict(schema='native-compact129-rule-teacher-episode-v1', status='completed',
        chunks=chunks, rows=decisions, teacher='ResourceV10', teacher_controlled=True,
        full_native_support_action_adjustment=bool(getattr(env,'full_native_teacher_support',False)),
        resource_reference_revision='native-positive-route-resource-reference-v1',
        partial_point_danger_revision=target.get('partial_point_danger_revision'),
        native_repair_revisions=native_repair_revisions(reactive_orbs=wall_revision),
        native_wall_reactive_forecast=wall_revision,
        teacher_support_revision=support_revision,
        PPO_samples=0, episode_summary=summary,
        sensor='Current compact129 native motion, focus and curve laser settings',
        native_risk_semantics='Forecasts with trust flags;only executed movement outcomes are native collision facts',
        seconds=time.monotonic()-start)
    write_json(output/'index.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--environment-config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=202611047)
    p.add_argument('--legacy-resource-teacher',action='store_true',
        help='Explicit historical comparison only;by default use repaired current-native teacher labels')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
        FLY_TOUHOU_DESKTOP_NAME='fly-touhou-job-rule-data-'+str(os.getpid()))
    from fly_touhou.thrl_study_io import read_shared_json
    from fly_touhou.thrl_durable_records import write_json
    from fly_touhou.thrl_job_paths import configure_native_temp
    env = None
    native_temp = None
    def progress(**values):
        write_json(a.output/'progress.json', dict(PID=os.getpid(), **values))
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        progress(phase='loading_environment_modules')
        # Report an import stall as an actionable traceback while leaving the
        # independent learner alone. This is closed once loading succeeds.
        import faulthandler
        with (a.output/'startup-trace.log').open('w', encoding='utf-8') as trace:
            faulthandler.dump_traceback_later(12, repeat=True, file=trace)
            try:
                from fly_touhou.compact_full_story import factory, story_kwargs
            finally:
                faulthandler.cancel_dump_traceback_later()
        progress(phase='preparing_private_environment')
        native_temp=configure_native_temp(a.output)
        progress(phase='reading_native_environment_config')
        cfg = read_shared_json(a.environment_config)
        cfg['seed'] = a.seed
        cfg['compact_full_native_teacher_support'] = not a.legacy_resource_teacher
        progress(phase='preparing_original_arguments')
        options = story_kwargs(cfg, a.output, 1, 'train', teacher=True)[0]
        progress(phase='starting_private_original')
        env = factory(options)
        result = collect(env, a.output, progress=progress)
        write_json(a.output/'result.json', result)
        progress(phase='completed', decisions=result['rows'], outcome=result['episode_summary'].get('outcome'))
    except BaseException as exc:
        write_json(a.output/'failure.json', dict(error=repr(exc), traceback=traceback.format_exc()))
        progress(phase='failed', error=repr(exc))
        raise
    finally:
        if env is not None:
            env.close()
        if native_temp is not None:
            from fly_touhou.thrl_job_paths import cleanup_native_temp
            write_json(a.output/'native-temp-cleanup.json',cleanup_native_temp(native_temp))


if __name__ == '__main__':
    main()
