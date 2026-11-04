"""Play a completed teacher-pilot candidate on a previous opening seed."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import time
import traceback

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from fly_touhou.thrl_durable_records import write_json
from fly_touhou.thrl_study_io import read_shared_json


def process_alive(pid, created):
    try:
        p = psutil.Process(pid)
        return p.is_running() and abs(p.create_time()-created) < .01
    except psutil.NoSuchProcess:
        return False
    except psutil.AccessDenied:
        # An unreadable process is not evidence that the pilot has exited.
        return True


def stage_means(rows):
    result = {}
    for stage in (3, 4):
        values = [r['loss'] for r in rows if r['native_rule_stage'] == stage]
        result[str(stage)] = sum(values)/len(values) if values else None
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('pilot', 'baseline', 'connectome', 'environment-config', 'kernel-cache', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--pilot-pid', type=int)
    p.add_argument('--pilot-created', type=float)
    p.add_argument('--baseline-pid', type=int, required=True)
    p.add_argument('--baseline-created', type=float, required=True)
    p.add_argument('--pilot-kind', choices=('rule', 'bomb'), default='rule')
    p.add_argument('--seed-index', type=int, default=0,
        help='Select a reset seed from the baseline development episode plan')
    p.add_argument('--wait-for-pid', type=int,
        help='Wait for an existing candidate game to finish before starting this one')
    p.add_argument('--wait-for-created', type=float)
    p.add_argument('--compare-single-result', type=Path,
        help='Also compare with an earlier frozen game of this same candidate and opening seed')
    a = p.parse_args()
    if (a.pilot_pid is None) != (a.pilot_created is None):
        p.error('Both pilot PID and creation time are required when waiting')
    if (a.wait_for_pid is None) != (a.wait_for_created is None):
        p.error('Both existing-game PID and creation time are required')
    a.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    def status(phase, **data):
        write_json(a.output/'status.json', dict(phase=phase, seconds=time.monotonic()-start, **data))

    try:
        status('waiting_for_teacher_pilot')
        while not (a.pilot/'report.json').exists():
            if (a.pilot/'failure.json').exists():
                raise RuntimeError('Teacher pilot failed: '+str(read_shared_json(a.pilot/'failure.json')['error']))
            if a.pilot_pid is None:
                raise RuntimeError('Pilot report is not complete and no live pilot identity was supplied')
            if not process_alive(a.pilot_pid, a.pilot_created):
                raise RuntimeError('Teacher pilot exited without a completed report')
            time.sleep(15)
        pilot = read_shared_json(a.pilot/'report.json')
        if a.pilot_kind == 'bomb':
            if not pilot.get('candidate_selected'):
                status('no_candidate_selected', model_replaced=False)
                return
            before, after = pilot['baseline_validation'], pilot['selected_validation']
            checkpoint = a.pilot/'model-parameters.pt'
        else:
            before = stage_means(pilot['heldout_before']['rows'])
            after = stage_means(pilot['heldout_after']['rows'])
            checkpoint = a.pilot/'candidate-parameters.pt'
        plan = read_shared_json(a.baseline/'episode_plan.json')
        if not 0 <= a.seed_index < len(plan['seeds']):
            raise ValueError('Seed index is outside the baseline episode plan')
        reset_seed = int(plan['seeds'][a.seed_index])
        while a.wait_for_pid is not None and process_alive(a.wait_for_pid, a.wait_for_created):
            status('waiting_for_prior_candidate_game', prior_PID=a.wait_for_pid,
                   reset_seed=reset_seed, teacher_loss_before=before, teacher_loss_after=after)
            if (a.output/'STOP').exists():
                raise InterruptedError('Follow-up stopped before starting the next game')
            time.sleep(15)
        argv = [sys.executable, '-B', str(ROOT/'scripts/evaluate_compact_single_native.py'),
                '--checkpoint', str(checkpoint), '--connectome', str(a.connectome),
                '--environment-config', str(a.environment_config), '--output', str(a.output/'candidate'),
                '--kernel-cache', str(a.kernel_cache), '--explicit-reset-seed', str(reset_seed)]
        with (a.output/'candidate-stdout.log').open('wb') as stdout, (a.output/'candidate-stderr.log').open('wb') as stderr:
            flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            child = subprocess.Popen(argv, cwd=ROOT, stdin=subprocess.DEVNULL,
                                     stdout=stdout, stderr=stderr, creationflags=flags)
            status('candidate_frozen_original_game', PID=child.pid, reset_seed=reset_seed,
                   teacher_loss_before=before, teacher_loss_after=after)
            child.wait()
            if child.returncode:
                raise RuntimeError('Candidate original-game evaluator exited '+str(child.returncode))
        candidate = read_shared_json(a.output/'candidate'/'result.json')
        status('waiting_for_baseline_report', candidate=candidate['episode_summary'], reset_seed=reset_seed)
        while not (a.baseline/'report.json').exists():
            if not process_alive(a.baseline_pid, a.baseline_created):
                raise RuntimeError('Baseline training process exited before producing its frozen report')
            time.sleep(15)
        baseline = read_shared_json(a.baseline/'report.json')
        matches = [r for r in baseline['episode_results'] if r['evaluation_seed'] == reset_seed]
        if len(matches) != 1:
            raise ValueError('Baseline does not contain exactly one matching reset seed')
        parent = matches[0]
        comparison = dict(status='completed', pilot_kind=a.pilot_kind,
            auxiliary_before=before, auxiliary_after=after,
            baseline=parent, candidate=candidate, reset_seed=reset_seed,
            matched_initial_native_seed=(parent['full_story_native_seed'] == candidate['native_seed']),
            model_replaced=False, training_updates_in_original_game=0,
            scope='One frozen development game per model with the same opening seed; trajectories and RNG consumption can diverge.')
        if a.compare_single_result is not None:
            previous=read_shared_json(a.compare_single_result)
            if (previous['native_seed']!=candidate['native_seed'] or
                    previous.get('explicit_reset_seed')!=candidate.get('explicit_reset_seed') or
                    Path(previous['checkpoint']).resolve()!=Path(candidate['checkpoint']).resolve()):
                raise ValueError('Sensor comparison requires the same checkpoint and original opening seed')
            comparison['same_candidate_previous_sensor']=previous
            comparison['sensor_comparison_scope']=('One game per sensor with matching checkpoint and opening seed;'
                'policy trajectories and subsequent RNG consumption can differ.')
        write_json(a.output/'report.json', comparison)
        status('completed', report=str(a.output/'report.json'),
               matched_initial_native_seed=comparison['matched_initial_native_seed'], model_replaced=False)
    except BaseException as exc:
        write_json(a.output/'failure.json', dict(error=repr(exc), traceback=traceback.format_exc()))
        status('failed', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
