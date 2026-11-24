"""Run a finite student continuation, close its games, then evaluate its final weights."""
from pathlib import Path
import argparse
import os
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from fly_touhou.thrl_study_io import read_shared_json as read
from fly_touhou.thrl_durable_records import write_json


def position_comparison(baseline, candidate):
    result = {}
    for stage in map(str, range(1, 7)):
        old = baseline.get('stages', {}).get(stage, {})
        new = candidate.get('stages', {}).get(stage, {})
        result[stage] = {key: dict(before=old.get(key), after=new.get(key)) for key in
            ('upper_fraction', 'unprotected_upper_fraction', 'mean_y',
             'misses', 'upper_misses', 'contacts_per_10000_frames', 'bombs')}
    return result


def operator_boundary_result(train):
    """Evaluate a saved, deliberately completed batch without starting training."""
    train = Path(train)
    stopped = read(train / 'failure.json')
    if (not stopped.get('operator_requested_stop')
            or stopped.get('error') != "RuntimeError('Operator STOP_AFTER_CYCLE')"
            or stopped.get('checkpoint_save_error')):
        raise RuntimeError('No completed operator-stop checkpoint is available')
    progress = read(train / 'progress.json')
    coverage = read(train / 'stage-coverage.json')
    result = dict(status='operator_stopped_after_completed_cycle',
        final_parameters=stopped['operator_stop_checkpoint'],
        completed_cycle=progress['cycle'], frozen_evaluation_deferred=True,
        student_actions_this_job=progress['student_actions_this_job'],
        PPO_updates_this_job=progress['PPO_updates_this_job'],
        PPO_updates=stopped['PPO_updates'], new_student_actions=stopped['new_student_actions'],
        stage_coverage=coverage, requested_native_game_round_complete=coverage['game_target_met'],
        training_paused=True, automatic_training_resume=False,
        remaining_first_games_aborted_by_operator=True)
    write_json(train / 'operator-stop-result.json', result)
    write_json(train / 'progress.json', dict(progress,
        phase='operator_stopped_after_completed_cycle', completed_cycle=progress['cycle'],
        training_paused=True, automatic_training_resume=False))
    return result


def run_plan(plan, control, *, register_monitor, evaluate_operator_stop=False):
    control = Path(control)
    train = Path(plan['training_output'])
    evaluation = Path(plan['evaluation_output'])
    started = time.time()
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1',
        TEMP=plan['temporary_directory'], TMP=plan['temporary_directory'])
    Path(plan['temporary_directory']).mkdir(parents=True, exist_ok=True)
    flags = (subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP) if os.name == 'nt' else 0

    def status(phase, **values):
        write_json(control / 'status.json', dict(phase=phase, updated_at=time.time(),
            started_at=started, training_output=str(train), evaluation_output=str(evaluation),
            **values))

    def run_child(kind, command, output):
        with (control / (kind + '-stdout.log')).open('ab') as stdout, \
                (control / (kind + '-stderr.log')).open('ab') as stderr:
            child = subprocess.Popen(command, cwd=plan.get('project_directory', str(ROOT)),
                env=env, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, creationflags=flags)
            import psutil
            receipt = dict(PID=child.pid, create_time=psutil.Process(child.pid).create_time(),
                command=command, output=str(output),
                identity_note='Dispatch process; the Python workload records its own PID/create_time in launch.json')
            write_json(control / (kind + '-dispatch.json'), receipt)
            status(kind + '_running', dispatch=receipt, training_paused=(kind == 'evaluation'))
            code = child.wait()
        receipt.update(returncode=code, exited_at=time.time())
        write_json(control / (kind + '-dispatch.json'), receipt)
        if code:
            failure = read(output / 'failure.json') if (output / 'failure.json').exists() else None
            raise RuntimeError(f'{kind} exited with code {code}: {failure}')

    try:
        if evaluate_operator_stop:
            result = operator_boundary_result(train)
        else:
            run_child('training', plan['training_command'], train)
            result = read(train / 'result.json')
        if not result.get('frozen_evaluation_deferred'):
            raise RuntimeError('Training did not finish at the deferred frozen-evaluation boundary')
        final = Path(result['final_parameters'])
        if not final.is_file():
            raise FileNotFoundError(final)
        cleanup = read(train / 'native-training' / 'cleanup.json') if (train / 'native-training' / 'cleanup.json').exists() else {}
        if cleanup.get('errors'):
            raise RuntimeError(f'Training game cleanup failed: {cleanup}')
        command = list(plan['evaluation_command'])
        command[command.index('--checkpoint') + 1] = str(final)
        write_json(control / 'evaluation-input.json', dict(checkpoint=str(final),
            source_cycle=read(train / 'progress.json').get('completed_cycle'),
            training_result=result, native_cleanup=cleanup, command=command))
        write_json(control / 'evaluation-monitor-registration.json', register_monitor(evaluation))
        run_child('evaluation', command, evaluation)
        report = read(evaluation / 'development' / 'report.json')
        old = read(Path(plan['baseline_output']) / 'development' / 'report.json')
        if report.get('status') != 'completed' or report.get('episodes') != plan['evaluation_episodes']:
            raise RuntimeError('Frozen evaluation did not complete all planned games')
        comparison = dict(status='completed', before=dict(episodes=old['episodes'], clears=old['clears'],
            stage_counts=old['stage_counts']), after=dict(episodes=report['episodes'], clears=report['clears'],
            stage_counts=report['stage_counts']), training=result, final_parameters=str(final),
            baseline_report=str(Path(plan['baseline_output']) / 'development' / 'report.json'),
            new_report=str(evaluation / 'development' / 'report.json'),
            training_paused=True, automatic_training_resume=False,
            scope='Paired development opening seeds; action-dependent native RNG may subsequently diverge; not a held-out test')
        comparison['positions'] = position_comparison(
            read(Path(plan['baseline_output']) / 'development' / 'position-metrics.json'),
            read(evaluation / 'development' / 'position-metrics.json'))
        write_json(control / 'comparison.json', comparison)
        status('completed_training_and_frozen_evaluation', training_paused=True,
            automatic_training_resume=False, clears=report['clears'], episodes=report['episodes'],
            comparison=str(control / 'comparison.json'))
        return comparison
    except BaseException as error:
        failure = dict(error=repr(error), traceback=traceback.format_exc(),
            no_automatic_retry=True, automatic_training_resume=False)
        write_json(control / 'failure.json', failure)
        status('failed', **failure)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--evaluate-operator-stop', action='store_true',
        help='Only evaluate the checkpoint saved after an operator-requested batch stop; never launch training.')
    args = parser.parse_args()
    import psutil
    from train_compact_story_ppo import register_tensorboard_run
    control = args.plan.resolve().parent
    proc = psutil.Process()
    write_json(control / 'watcher.json', dict(PID=proc.pid, create_time=proc.create_time(),
        plan=str(args.plan), created_at=time.time()))
    run_plan(read(args.plan), control, register_monitor=register_tensorboard_run,
        evaluate_operator_stop=args.evaluate_operator_stop)


if __name__ == '__main__':
    main()
