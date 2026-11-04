"""Run one patched-interface comparison before the next Adam update.

The current collection finishes and all16 original worlds remain parked.
Reuse the learner/model/GPU; no second learner or polling process is started.
"""
from datetime import datetime, timezone
from pathlib import Path
import os
import sys
import time
import traceback


def _main_frame():
    for frame in sys._current_frames().values():
        while frame is not None:
            if frame.f_code.co_name == 'main' and Path(frame.f_code.co_filename).name == 'train_compact_story_ppo.py':
                return frame
            frame = frame.f_back
    raise RuntimeError('The original learner main frame is unavailable')


def arm(plan_path):
    import psutil
    from fly_touhou.thrl_study_io import read_shared_json
    from fly_touhou.thrl_durable_records import write_json
    from fly_touhou.compact_native_forecast import native_repair_revisions

    plan_path = Path(plan_path)
    plan = read_shared_json(plan_path)
    if os.getpid() != plan['pid'] or abs(psutil.Process().create_time() - plan['creation_time']) > .01:
        raise ValueError('Apply only to the current owned learner')
    frame = _main_frame()
    local = frame.f_locals
    if local['updates'] != plan['PPO_updates']:
        raise ValueError('The model already changed; this would not compare the same weights')
    optimizer = local['optimizer']
    if getattr(optimizer, '_pending_interface_comparison', None):
        return dict(status='already_queued')
    original = optimizer.sampled_update_snapshot

    def evaluate_before_snapshot():
        optimizer.sampled_update_snapshot = original
        optimizer._pending_interface_comparison = None
        state = frame.f_locals
        if state['updates'] != plan['PPO_updates']:
            raise RuntimeError('Unexpected optimizer update before the queued frozen comparison')
        from fly_touhou.compact_full_story import evaluate
        output = Path(plan['evaluation_output'])
        write_json(plan_path.parent / 'patched-comparison-status.json', dict(
            status='running', utc=datetime.now(timezone.utc).isoformat(),
            checkpoint=plan['checkpoint'], PPO_updates=state['updates'],
            source_revision=native_repair_revisions(), output=str(output)))
        # The caller takes this optimizer snapshot only after collection has
        # drained its native commands, before timing or executing the update.
        if state['collector'].vector._inflight:
            raise RuntimeError('Native collection did not drain before its update boundary')
        report = None
        try:
            report = evaluate(state['model'], state['cfg'], output,
                episodes=4, seed=plan['evaluation_seed'], policy_seed=plan['policy_seed'],
                action_selection=plan['action_selection'], evaluation_split='development',
                worker_cpus=state['cpus'], guard=state['guard'],
                progress=lambda info: state['progress']('frozen_full_story_development',
                    cycle=87, comparison='same_weights_patched_interface',
                    evaluation=dict(info, comparison='patched_v3', trained_checkpoint_cycle=87)))
        except Exception as error:
            failure = dict(status='comparison_failed_training_data_preserved',
                error=repr(error), traceback=traceback.format_exc(), output=str(output))
            write_json(plan_path.parent / 'patched-comparison-failure.json', failure)
            write_json(plan_path.parent / 'patched-comparison-status.json', failure)
        if report is None:
            _await_analysis(state, plan, plan_path, comparison_failed=True)
            return original()
        write_json(output / 'paired-comparison.json', dict(
            baseline_report=plan['baseline_report'], checkpoint=plan['checkpoint'],
            source_cycle=87, same_optimizer_updates=state['updates'],
            baseline_opening_seeds=plan['opening_seeds'],
            evaluation_seed=plan['evaluation_seed'], policy_seed=plan['policy_seed'],
            source_revision=native_repair_revisions(), report=report,
            scope='Four paired-opening development games with the same weights and decoder; '
                  'later trajectories/RNG consumption can differ. No evaluation labels enter training.',
            model_weights_replaced=False, optimizer_reset=False, training_worlds_reset=False))
        write_json(plan_path.parent / 'patched-comparison-status.json', dict(
            status='completed', output=str(output), clears=report['clears'],
            stage_counts=report['stage_counts'], PPO_updates=state['updates']))
        _await_analysis(state, plan, plan_path, comparison_failed=False)
        state['progress']('full_story_PPO_update', cycle=state['cycle'],
                          preceding_patched_comparison=str(output / 'report.json'))
        return original()

    optimizer.sampled_update_snapshot = evaluate_before_snapshot
    optimizer._pending_interface_comparison = str(plan_path)
    receipt = dict(status='queued_before_next_Adam_update', pid=os.getpid(),
        PPO_updates=local['updates'], source_cycle=87, current_cycle=local['cycle'],
        evaluation_output=plan['evaluation_output'], episodes=4,
        hold_parameter_updates_after_comparison=bool(plan.get('hold_for_analysis')),
        current_collection_interrupted=False, game_resets=0, new_learner_process=False)
    write_json(plan_path.parent / 'patched-comparison-status.json', receipt)
    return receipt


def _await_analysis(state, plan, plan_path, *, comparison_failed):
    """Keep the collected rollout and worlds in the same learner until resumed."""
    if not plan.get('hold_for_analysis'):
        return
    from fly_touhou.thrl_study_io import read_shared_json
    from fly_touhou.thrl_durable_records import write_json
    resolution = plan_path.parent / 'patched-comparison-resolution.json'
    status = dict(status='waiting_for_stage6_analysis_before_parameter_updates',
        comparison_failed=comparison_failed, PPO_updates=state['updates'],
        preserved_rollout_cycle=state['cycle'], source_checkpoint=plan['checkpoint'],
        resolution_path=str(resolution), training_worlds_reset=False)
    write_json(plan_path.parent / 'patched-comparison-hold.json', status)
    last = 0.
    while True:
        state['guard']()
        if resolution.exists():
            decision = read_shared_json(resolution)
            if (decision.get('action') == 'resume_updates'
                    and decision.get('PPO_updates') == state['updates']):
                write_json(plan_path.parent / 'patched-comparison-hold.json',
                    dict(status, status='resuming_preserved_rollout', decision=decision))
                break
        now = time.monotonic()
        if now - last >= 15:
            state['progress']('stage6_analysis_before_parameter_updates',
                cycle=state['cycle'], frozen_checkpoint_cycle=87,
                comparison_output=plan['evaluation_output'], comparison_failed=comparison_failed)
            last = now
        time.sleep(1.)
