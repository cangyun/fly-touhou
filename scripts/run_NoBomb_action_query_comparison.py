"""One finite owned control -> action-query comparison, without model polling.

No retry, new automation, teacher takeover or full-story training is started.
The existing control owns its worlds until it and its evaluation have exited.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from fly_touhou.thrl_durable_records import write_json
from fly_touhou.thrl_study_io import read_shared_json


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--specification', type=Path, required=True)
    args = parser.parse_args()
    spec = read_shared_json(args.specification)
    control = Path(spec['pipeline_control']).resolve()
    if (control / 'status.json').exists():
        raise FileExistsError('One finite dispatch only; no duplicate controller')
    import psutil
    me = psutil.Process()
    me.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    state = dict(schema='NoBomb-control-action-query-finite-pipeline-v1',
        PID=me.pid, create_time=me.create_time(), started_UTC=datetime.now(timezone.utc).isoformat(),
        phase='waiting_owned_control', automatic_retry=False,
        full_story_PPO_resumed=False, maximum_native_worlds=30)

    def publish(phase, **values):
        state.update(phase=phase, UTC=datetime.now(timezone.utc).isoformat(), **values)
        write_json(control / 'status.json', state)

    def guard():
        if (control / 'STOP').exists():
            raise RuntimeError('Operator STOP: no next training stage')
        if time.time() > spec['deadline_unix']:
            raise RuntimeError('Finite comparison wall budget expired; no retry')

    def wait_owned(pid, created):
        while True:
            guard()
            try:
                owner = psutil.Process(pid)
                if abs(owner.create_time() - created) > .01:
                    return  # Original owner exited; never wait on reused PID.
                if not owner.is_running():
                    return
            except psutil.NoSuchProcess:
                return
            time.sleep(5)

    def update_goal(values):
        path = ROOT / 'artifacts/engineering/B1-autonomous-dodge-20261024/goal_state.json'
        current = read_shared_json(path)
        current.update(values, goal_achieved=False, user_input_required=False,
            updated_UTC=datetime.now(timezone.utc).isoformat())
        write_json(path, current)

    try:
        publish('waiting_owned_control', existing_control=spec['existing_control_output'],
            owned_control_PID=spec['existing_control_PID'],
            owned_control_create_time=spec['existing_control_create_time'])
        wait_owned(spec['existing_control_PID'], spec['existing_control_create_time'])
        from summarize_object_avoidance import summarize, summarize_matched_candidates
        old = Path(spec['existing_control_output'])
        old_launch = read_shared_json(old / 'launch.json')
        if (old_launch['PID'] != spec['existing_control_PID']
                or abs(old_launch['create_time'] - spec['existing_control_create_time']) > .01):
            raise ValueError('Completed control owner differs from dispatch')
        publish('summarizing_control')
        review = summarize(old, Path(spec['baselines']), Path(spec['control_review']),
            parent_evaluation=Path(spec['parent_evaluation']))
        update_goal(dict(avoidance_control_review=review))
        if review['preregistered_near_D95_development_screen']:
            publish('control_requires_independent_confirmation', control_review=spec['control_review'],
                candidate_dispatched=False)
            return

        guard()
        candidate_spec = spec['candidate_specification']
        for filename, digest in spec['candidate_code_SHA256'].items():
            if sha(ROOT / filename) != digest:
                raise ValueError('Prepared candidate code changed: ' + filename)
        if sha(candidate_spec['source']) != candidate_spec['source_sha256']:
            raise ValueError('Complete common source checkpoint changed')
        if any((p.info['name'] or '').lower() == spec['native_process_name'].lower()
                for p in psutil.process_iter(['name'])):
            raise RuntimeError('Native worlds remain owned; no competing launch')
        candidate_control = Path(candidate_spec['control']).resolve()
        candidate_output = Path(candidate_spec['output']).resolve()
        candidate_control.relative_to(ROOT / 'artifacts')
        candidate_output.relative_to(Path('./outputs').resolve())
        if candidate_control.exists() or candidate_output.exists():
            raise FileExistsError('Fresh candidate control and output required')
        candidate_control.mkdir(parents=True)
        write_json(candidate_control / 'launch-specification.json', candidate_spec)
        command = [str(ROOT / '.venv/Scripts/python.exe'), '-B',
            str(ROOT / 'scripts/run_training_with_native_cache_cleanup.py'),
            '--specification', str(candidate_control / 'launch-specification.json')]
        publish('dispatching_action_query', candidate_control=str(candidate_control),
            candidate_output=str(candidate_output))
        with (candidate_control / 'supervisor-launch.stdout.log').open('xb') as stdout, \
                (candidate_control / 'supervisor-launch.stderr.log').open('xb') as stderr:
            process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=stdout, stderr=stderr, creationflags=subprocess.CREATE_NO_WINDOW)
        receipt = dict(launcher_PID=process.pid,
            launcher_create_time=psutil.Process(process.pid).create_time(),
            control=str(candidate_control), output=str(candidate_output), automatic_retry=False)
        write_json(candidate_control / 'dispatch.json', receipt)
        startup_deadline = time.monotonic() + 180
        while not (candidate_output / 'launch.json').exists():
            guard()
            if process.poll() is not None or time.monotonic() > startup_deadline:
                raise RuntimeError('Candidate has no actual learner receipt; inspect retained logs')
            time.sleep(1)
        launch = read_shared_json(candidate_output / 'launch.json')
        publish('action_query_running', candidate_actual_PID=launch['PID'],
            candidate_actual_create_time=launch['create_time'])
        # The existing service is already prepared to parse this new family.
        # Only its source configuration changes, preserving port/log history.
        monitor_path = Path(spec['monitor_configuration'])
        monitor = read_shared_json(monitor_path)
        history = monitor.setdefault('historical_fly_algorithm_sources', {})
        history.update(monitor.get('fly_algorithm_sources', {}))
        monitor['fly_algorithm_sources'] = {spec['candidate_monitor_key']: str(candidate_output)}
        monitor['updated_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(monitor_path, monitor)
        write_json(control / 'monitor-hot-source.json', dict(configuration=str(monitor_path),
            active_source=str(candidate_output), actual_PID=launch['PID'],
            actual_create_time=launch['create_time'], new_monitor_started=False))
        update_goal(dict(current_continuous_training=dict(output=str(candidate_output),
            control=str(candidate_control), actual_PID=launch['PID'],
            actual_create_time=launch['create_time'], source=candidate_spec['source'],
            source_SHA=candidate_spec['source_sha256'], student_decisions=30720,
            full_story_PPO_resumed=False, mode='NoBomb action-query matched candidate'),
            avoidance_finite_pipeline=dict(control=str(control), status='action_query_running')))
        wait_owned(launch['PID'], launch['create_time'])
        while process.poll() is None:
            guard()
            time.sleep(2)
        if process.returncode:
            raise RuntimeError('Candidate failed; no retry; inspect its failure.json')
        publish('summarizing_action_query')
        candidate_review = summarize(candidate_output, Path(spec['baselines']),
            Path(spec['candidate_review']), parent_evaluation=Path(spec['parent_evaluation']))
        comparison = summarize_matched_candidates(old, candidate_output, Path(spec['matched_review']))
        write_json(control / 'comparison-dispatch-receipt.json',
            dict(common_source_SHA=candidate_spec['source_sha256'],
                verified_dispatched_training_seed=spec['matched_training_seed'],
                verified_dispatched_action_seed=spec['matched_action_seed'],
                verified_dispatched_minibatch_seed=spec['matched_minibatch_seed'],
                explanation='These fields were checked in both exact dispatch argv; '
                    'do not rewrite fields absent from historical launch.json.'))
        update_goal(dict(avoidance_action_query_review=candidate_review,
            avoidance_matched_comparison=comparison,
            avoidance_finite_pipeline=dict(control=str(control), status='comparison_completed'),
            avoidance_next_step='Review frozen same-parent comparison; keep full-story PPO stopped. '
                'Independent new20 confirmation only after avoidance approaches full D95.'))
        publish('comparison_completed', control_review=spec['control_review'],
            candidate_review=spec['candidate_review'], matched_review=spec['matched_review'],
            full_story_PPO_resumed=False)
    except BaseException as error:
        publish('failed_or_operator_stopped', error=repr(error), automatic_retry=False)
        raise


if __name__ == '__main__':
    main()
