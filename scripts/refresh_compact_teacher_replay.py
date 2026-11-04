"""Replay a training demonstration with current sensing and late-stage guidance.

Recorded actions retain the known original trajectory. The current rule teacher
labels stages five and six, without taking control. Earlier stages retain only
measured behavior and freshly encoded risk labels; old forecasts are not copied.
This is a refresh of an existing demonstration, not an independent clear or PPO.
"""
from pathlib import Path
import argparse
from collections import Counter, defaultdict
import os
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]


class RecordedTeacherEnv:
    def __init__(self, env, source, index):
        from replay_compact_teacher_wall import recorded_rows
        self.env = env
        self.rows = iter(recorded_rows(source, index))
        self.index = index
        self.current = None
        self.decisions = 0
        self.stats = defaultdict(Counter)
        self.teacher = env._teacher
        env._teacher = None
        env._teacher_label = None

    def __getattr__(self, name):
        return getattr(self.env, name)

    def _decorate(self, info):
        import numpy as np
        from replay_compact_teacher_wall import check_prefix
        check_prefix(self.current, info)
        stage = int(info['stage'])
        action = int(self.current['actions'])
        if stage >= 5:
            if self.env._teacher is None:
                self.env._teacher = self.teacher
                self.teacher.reset()
                info = self.env._label_current(info)
            label = dict(info['teacher_label'])
            self.stats[stage]['new_rule_disagrees_with_recorded_action'] += int(label['action'] != action)
            self.stats[stage]['refreshed_rule_rows'] += 1
            route=label.get('native_route_guidance',{})
            self.stats[stage]['multiturn_route_queries'] += int('candidates' in route)
            self.stats[stage]['multiturn_route_guidance_rows'] += int(route.get('applied',False))
            self.stats[stage]['multi_direction_rows'] += int(
                label.get('guidance_weight', 1.) > 0 and
                np.count_nonzero(np.asarray(label['probabilities'])[:18] >= .001) > 1)
        else:
            label = dict(probabilities=np.zeros(19, np.float32), guidance_weight=0.)
            self.stats[stage]['observed_behavior_only_rows'] += 1
        # Copy only the public label. Keep the rule teacher's recommendation and
        # internal state intact; the collector must record the executed action.
        label['action'] = action
        self.before_frames = int(info['episode_frames'])
        return dict(info, teacher_label=label)

    def reset(self):
        obs, info = self.env.reset()
        self.current = next(self.rows)
        return obs, self._decorate(info)

    def step(self, action):
        if int(action) != int(self.current['actions']):
            raise ValueError('Collector did not use the recorded action')
        obs, reward, done, truncated, info = self.env.step(int(action))
        elapsed = int(info['episode_frames']) - self.before_frames
        if truncated or elapsed != int(self.current['native_action_frames']):
            raise ValueError('Recorded action native duration diverged')
        self.decisions += 1
        if done:
            actual = info.get('episode_summary', {})
            if (self.decisions != self.index['rows'] or
                    actual.get('outcome') != self.index['episode_summary']['outcome']):
                raise ValueError('Recorded native trajectory did not reproduce its full outcome')
        else:
            self.current = next(self.rows)
            info = self._decorate(info)
        return obs, reward, done, truncated, info


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--environment-config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
        FLY_TOUHOU_DESKTOP_NAME='fly-touhou-job-rule-refresh-' + str(os.getpid()))
    from fly_touhou.thrl_durable_records import write_json
    from fly_touhou.thrl_study_io import read_shared_json
    from fly_touhou.thrl_job_paths import configure_native_temp, cleanup_native_temp
    import psutil
    process = psutil.Process()
    process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    process.cpu_affinity([14, 15])
    started = time.monotonic()
    env = None
    native_temp = None
    def progress(**values):
        write_json(a.output / 'progress.json', dict(PID=os.getpid(),
            create_time=process.create_time(), PPO_samples=0, **values))
    try:
        progress(phase='loading_environment')
        from fly_touhou.compact_full_story import factory, story_kwargs
        from collect_compact_rule_teacher_native import collect
        from replay_compact_teacher_wall import FixedRecordedSeed
        index = read_shared_json(a.source / 'index.json')
        expected = index['episode_summary']
        if expected['native_seed_split'] != 'train':
            raise ValueError('Do not convert held-out demonstrations into training data')
        native_temp = configure_native_temp(a.output)
        cfg = read_shared_json(a.environment_config)
        cfg['compact_full_native_teacher_support'] = True
        options = story_kwargs(cfg, a.output, 1, 'train', teacher=True)[0]
        progress(phase='starting_private_original')
        env = factory(options)
        env._full_story_rng = FixedRecordedSeed(expected['full_story_native_seed'])
        wrapped = RecordedTeacherEnv(env, a.source, index)
        result = collect(wrapped, a.output, progress=progress)
        result.update(recorded_source=str(a.source), new_independent_clear=False,
            action_source='Recorded training demonstration;current teacher supplies labels only',
            refreshed_distribution_stages=[5, 6],
            early_guidance='Measured behavior only;no old probabilities or forecasts copied',
            refreshed_risk_stages=[1, 2, 3, 4, 5, 6],
            stage_refresh_statistics=dict(wrapped.stats),
            native_wall_reactive_forecast='th14-distance-reactive-orbs-v1',
            elapsed_seconds=time.monotonic()-started)
        write_json(a.output / 'index.json', result)
        write_json(a.output / 'result.json', result)
        progress(phase='completed', decisions=result['rows'],
            outcome=result['episode_summary'].get('outcome'), new_independent_clear=False)
    except BaseException as exc:
        write_json(a.output / 'failure.json', dict(error=repr(exc), traceback=traceback.format_exc()))
        progress(phase='failed', error=repr(exc))
        raise
    finally:
        if env is not None:
            env.close()
        if native_temp is not None:
            write_json(a.output / 'native-temp-cleanup.json', cleanup_native_temp(native_temp))


if __name__ == '__main__':
    main()
