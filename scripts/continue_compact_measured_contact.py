"""One explicit corrective handoff after an owned complete-cycle stop.

Keeps global native evidence separate from a new job's local best. No retry,
parallel learner, desktop input, or forced process termination is introduced.
"""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fly_touhou.thrl_study_io import read_shared_json as read
from fly_touhou.thrl_durable_records import write_json
from fly_touhou.compact_native_budget import compare, metrics


def alive(identity):
    import psutil
    try:
        process = psutil.Process(identity['PID'])
        return process.is_running() and abs(process.create_time() - identity['create_time']) < .01
    except psutil.NoSuchProcess:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    import psutil
    plan = read(args.plan)
    control = args.plan.parent
    current = Path(plan['current_job'])
    previous = Path(plan['previous_output'])
    output = Path(plan['output'])
    identity = plan['previous_executor']
    me = psutil.Process()
    state = dict(PID=me.pid, create_time=me.create_time(), plan=str(args.plan))

    def status(phase, **values):
        state.update(phase=phase, **values)
        write_json(control / 'status.json', state)

    def authorized():
        j = read(current)
        if ((control / 'STOP').exists() or j.get('scheduling_suspended')
                or j.get('pause_after_current_test')):
            raise InterruptedError('Continuation suspended; no next learner launched')
        active = j.get('actual_executor')
        if active and active != identity and alive(active):
            raise InterruptedError('Another active executor owns the task; no parallel learner launched')

    try:
        authorized()
        metrics(read(plan['incumbent_report']))
        rows = plan['workers'] * plan['steps']
        if (plan['initial_new_decisions'] % rows or plan['maximum_new_decisions'] % rows
                or not 0 < plan['initial_new_decisions'] <= plan['maximum_new_decisions']):
            raise ValueError('Whole-rollout finite decision allowances required')
        owned = []
        if alive(identity):
            for child in psutil.Process(identity['PID']).children(recursive=True):
                try:
                    owned.append(dict(PID=child.pid, create_time=child.create_time()))
                except psutil.NoSuchProcess:
                    pass
        status('waiting_for_prior_complete_cycle', previous_executor=identity)
        started = time.monotonic()
        while alive(identity):
            authorized()
            if time.monotonic() - started > 10800:
                raise TimeoutError('Owned cycle wait exceeded three hours; no restart')
            time.sleep(2)
        authorized()
        previous_controller=plan.get('previous_controller')
        if previous_controller:
            deadline=time.monotonic()+60
            while alive(previous_controller):
                if time.monotonic()>deadline:
                    raise TimeoutError('Prior controller is still finalizing the frozen report')
                time.sleep(1)
        if plan.get('stop_for_qualifying_frozen_clears'):
            report=read(plan['previous_completed_frozen_report'])
            if report.get('clears',0)>=plan['stop_for_qualifying_frozen_clears']:
                status('qualifying_checkpoint_needs_independent_evaluation',
                       report=plan['previous_completed_frozen_report'],clears=report['clears'],goal_complete=False)
                return
        failure = read(previous / 'failure.json')
        progress = read(previous / 'progress.json')
        # The last progress callback can precede completed_cycle assignment.
        # The saved optimizer checkpoint records the completed boundary itself.
        stopped_training=None
        stopped_costs=None
        completed_cycle=progress.get('completed_cycle',progress.get('cycle'))
        if plan.get('use_operator_stop_checkpoint') and failure.get('operator_requested_stop'):
            import torch
            stopped_training=read(previous/'operator-stop-training-path.json')['path']
            checkpoint_state=torch.load(stopped_training,map_location='cpu',weights_only=False)
            completed_cycle=checkpoint_state['completed_cycle']
            stopped_costs=checkpoint_state['model_payload']['costs']
            del checkpoint_state
        if (not failure.get('operator_requested_stop') or failure.get('checkpoint_save_error') is not None
                or not failure.get('operator_stop_checkpoint')
                or completed_cycle != plan['previous_stop_cycle']):
            raise ValueError('The prior complete-cycle checkpoint was not preserved')
        if (not plan.get('previous_evaluation_follows_cycle')
                and progress.get('learning', {}).get('update') != plan['updates_per_cycle']):
            raise ValueError('The prior PPO cycle did not finish all scheduled updates')
        cleanup = read(previous / 'native-training/cleanup.json')
        if cleanup.get('errors'):
            raise ValueError('Prior native cleanup reported errors')
        deadline = time.monotonic() + 60
        while any(alive(item) for item in owned):
            if time.monotonic() > deadline:
                raise TimeoutError('Prior owned descendants remain live; no next learner')
            time.sleep(1)
        spent = int(progress['student_actions_this_job'])
        updates = int(progress['PPO_updates_this_job'])
        paid = dict(plan['paid_total_before_previous'])
        paid['PPO_updates'] += updates
        paid['new_student_actions'] += spent
        prepared = dict(plan, status='bound_after_prior_native_release',
                        previous_paid_student_decisions=spent, previous_paid_PPO_updates=updates,
                        paid_total_costs_before_job=paid, previous_cleanup=cleanup,
                        prior_owned_descendants_released=True)
        write_json(control / 'prepared.json', prepared)
        authorized()
        if output.exists():
            raise FileExistsError('Preserve existing output; do not dispatch a duplicate')
        argv = list(plan['argv'])
        if plan.get('use_operator_stop_checkpoint'):
            for name,value in (('--checkpoint',failure['operator_stop_checkpoint']),
                               ('--resume-training',stopped_training)):
                argv[argv.index(name)+1]=str(value)
        if plan.get('incumbent_from_current') and '--budget-benefit-reference' in argv:
            # Compare any extension with the best frozen model available at
            # handoff, including the evaluation that just finished. Otherwise
            # the next job could get credit for an older job's improvement.
            reference=read(current).get('best_verified_native_report') or plan['incumbent_report']
            argv[argv.index('--budget-benefit-reference')+1]=str(reference)
        with (control / 'training-stdout.log').open('xb') as out, (control / 'training-stderr.log').open('xb') as err:
            child = subprocess.Popen(argv, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                     creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP)
        status(plan.get('active_stage','starting_measured_contact_T05_PPO'), launcher_PID=child.pid, output=str(output))
        bound = False
        nodes_mtime = None
        observed = set()

        def sync_native_nodes():
            nodes_path = output / 'evaluations.json'
            if not bound or not nodes_path.exists():
                return
            for node in read(nodes_path)['nodes']:
                if node['cycle'] in observed:
                    continue
                j = read(current)
                decision = compare(node['report'], read(j['best_verified_native_report']))
                report_path = output / f"development-cycle{node['cycle']:03}/report.json"
                write_json(control / f"global-native-comparison-cycle{node['cycle']:03}.json", decision)
                j['latest_completed_frozen_report'] = str(report_path)
                if decision['improved']:
                    j['best_verified_parameters'] = node['checkpoint']
                    j['best_verified_native_report'] = str(report_path)
                write_json(current, j)
                observed.add(node['cycle'])

        while child.poll() is None:
            launch_path = output / 'launch.json'
            if not bound and launch_path.exists() and (output / 'progress.json').exists():
                launch = read(launch_path)
                actual = {key: launch[key] for key in ('PID', 'create_time')}
                if not alive(actual):
                    time.sleep(1)
                    continue
                dispatch = control / 'training-dispatch.json'
                write_json(dispatch, dict(argv=argv, output=str(output), actual_executor=actual,
                                          launcher_PID=child.pid, controller=str(args.plan)))
                j = read(current)
                j.setdefault('historical_jobs', []).append(dict(output=str(previous), actual_executor=identity,
                    status=plan.get('previous_history_status','controlled_stop_after_complete_cycle_for_measured_contact_T05'),
                    new_student_actions=spent, PPO_updates=updates,
                    operator_stop_checkpoint=failure['operator_stop_checkpoint'],
                    frozen_report=plan['previous_completed_frozen_report'], all_models_tapes_logs_preserved=True))
                incumbent_checkpoint=(j.get('best_verified_parameters') if plan.get('incumbent_from_current') else None) or plan['incumbent_checkpoint']
                incumbent_report=(j.get('best_verified_native_report') if plan.get('incumbent_from_current') else None) or plan['incumbent_report']
                j.update(active_output=str(output), actual_executor=actual, dispatch=str(dispatch),
                    active_stage=plan.get('active_stage','measured_native_contact_T05_full_story_PPO'), source_costs=stopped_costs or plan['source_costs'],
                    paid_total_costs_before_job=paid, initial_new_decisions=plan['initial_new_decisions'],
                    planned_new_decisions=plan['maximum_new_decisions'],
                    planned_new_Adam_updates=plan['maximum_new_decisions'] // rows * plan['updates_per_cycle'],
                    conditional_extension_decisions=plan['maximum_new_decisions'] - plan['initial_new_decisions'],
                    budget_accounting='new finite corrective pilot; all prior paid work retained separately',
                    evaluation_every_cycles=plan['evaluation_every_cycles'], native_budget_flat_patience=2,
                    best_verified_parameters=incumbent_checkpoint, best_verified_native_report=incumbent_report,
                    latest_completed_frozen_report=plan['previous_completed_frozen_report'],
                    source_binding_proof=plan['source_binding_proof'], next_training_source_is_provisional=False,
                    next_training_preparation=str(control / 'prepared.json'), controlled_handoff_completed=True,
                    latest_progress=read(output / 'progress.json'), last_verified_progress=read(output / 'progress.json'),
                    movement_sampling_temperature=.5, native_contact_risk_calibration=True,
                    native_motion_forecast=plan.get('native_motion_forecast',False),
                    interface_repair_in_progress=False,goal_complete=False)
                write_json(current, j)
                monitor = Path(plan['tensorboard_configuration'])
                cfg = read(monitor)
                cfg.setdefault('historical_fly_algorithm_sources', {}).update(cfg.get('fly_algorithm_sources', {}))
                cfg['fly_algorithm_sources'] = {plan.get('tensorboard_series','compact_SNN_measured_contact_T05_PPO'): str(output)}
                write_json(monitor, cfg)
                status('running_measured_contact_T05_PPO', actual_executor=actual,
                       initial_new_decisions=plan['initial_new_decisions'], maximum_new_decisions=plan['maximum_new_decisions'])
                bound = True
            nodes_path = output / 'evaluations.json'
            if bound and nodes_path.exists() and nodes_path.stat().st_mtime_ns != nodes_mtime:
                nodes_mtime = nodes_path.stat().st_mtime_ns
                sync_native_nodes()
            time.sleep(1)
        sync_native_nodes()
        if child.returncode:
            terminal = read(output / 'failure.json') if (output / 'failure.json').exists() else {}
            if terminal.get('operator_requested_stop'):
                status('controlled_stop_after_complete_checkpoint', goal_complete=False)
                return
            raise RuntimeError(f'Corrective learner exited {child.returncode}; no automatic retry')
        status('completed_finite_corrective_job', goal_complete=False)
    except InterruptedError as error:
        status('stopped_before_launch', reason=str(error), automatic_retry=False, goal_complete=False)
    except BaseException as error:
        status('failed_no_automatic_retry', error=repr(error), traceback=traceback.format_exc(), goal_complete=False)
        raise


if __name__ == '__main__':
    main()
