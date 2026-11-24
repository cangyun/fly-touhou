"""Explicit input, guidance and event-objective migrations at PPO boundaries."""
from copy import deepcopy


def validate_resume(saved, spec):
    old = saved['spec']
    for field in ('tasks', 'world_groups', 'segment_decisions'):
        if old[field] != spec[field]:
            raise ValueError('Resume requires the same '+field)
    if old['nbnm'] == spec['nbnm']:
        return dict(route_prior_added=False)
    before, after = deepcopy(old['nbnm']), deepcopy(spec['nbnm'])
    reward_change = None
    pace_target_changed = False
    if before.get('objective') != after.get('objective'):
        from dataclasses import asdict
        from ..nbnm_objective import Objective
        previous = asdict(Objective(**before.get('objective', {})))
        requested = asdict(Objective(**after.get('objective', {})))
        old_pace = previous.pop('completion_pace_weight')
        new_pace = requested.pop('completion_pace_weight')
        old_calm = previous.pop('calm_cost_limit')
        new_calm = requested.pop('calm_cost_limit')
        old_events = (previous.pop('progress_revision'), previous.pop('progress_weight'))
        new_events = (requested.pop('progress_revision'), requested.pop('progress_weight'))
        add_events = old_events[0] is None and new_events[0] == 'native-events-v1'
        extras_changed = (old_pace, old_calm) != (new_pace, new_calm)
        if (previous != requested or saved.get('in_progress') is not None
                or not (add_events or (old_events == new_events and extras_changed))):
            raise ValueError('Reward changes require a committed segment and compatible success and event objectives')
        pace_target_changed = bool(old_pace) != bool(new_pace)
        reward_change = dict(before=before['objective'], after=after['objective'])
        before['objective'] = deepcopy(after['objective'])
    # Teacher distributions and their coefficients are recomputed on NEW
    # student trajectories. They do not alter observation columns, rewards,
    # native worlds, carried brain states or existing optimizer moments.
    changes={}
    old_corridor=before.pop('corridor_risk',None)
    new_corridor=after.pop('corridor_risk',None)
    if old_corridor!=new_corridor:
        from ..nbnm_corridor import validate
        if not validate(new_corridor) or (old_corridor and (old_corridor['revision']!=new_corridor['revision']
                or (old_corridor.get('witness_weight') and not new_corridor.get('witness_weight')))):
            raise ValueError('Preserve the existing corridor readout when resuming')
        changes['corridor_risk']=dict(before=old_corridor,after=new_corridor)
    old_confirmation=before.pop('event_confirmation_revision',None)
    new_confirmation=after.pop('event_confirmation_revision',None)
    if old_confirmation!=new_confirmation:
        from ..nbnm_event_progress import CONFIRMATION_REVISION
        if old_confirmation is not None or new_confirmation!=CONFIRMATION_REVISION:
            raise ValueError('Unsupported native event confirmation migration')
        changes['event_confirmation_revision']=dict(before=old_confirmation,after=new_confirmation)
    old_geometry=before.pop('native_rule_geometry',None);new_geometry=after.pop('native_rule_geometry',None)
    if old_geometry!=new_geometry:
        from ..nbnm_geometry_revision import validate
        if old_geometry is not None or not validate(new_geometry):
            raise ValueError('Only explicit legacy to live-EX geometry migration is implemented')
        changes['native_rule_geometry']=dict(before=old_geometry,after=new_geometry)
    old_risk=before.pop('risk_target_revision',None);new_risk=after.pop('risk_target_revision',None)
    if old_risk!=new_risk:
        from ..nbnm_continuation_risk import REVISION
        from ..nbnm_policy_continuation_risk import REVISION as POLICY_REVISION
        if (old_risk,new_risk) not in ((None,REVISION),(None,POLICY_REVISION),(REVISION,POLICY_REVISION)):
            raise ValueError('Unsupported continuation-risk migration')
        changes['risk_target_revision']=dict(before=old_risk,after=new_risk)
    for key in ('route_preference_revision','future_guidance_coefficient','rule_guidance_coefficient','demonstration_route_teacher','live_route_planner','movement_risk_horizon_weights','future_reference','wall_exit_guidance','teacher_outcome_revision','native_warning48','continuation_supervision','timed_corridor_supervision'):
        a,b=before.pop(key,None),after.pop(key,None)
        if key=='native_warning48' and a and (not b or a['revision']!=b['revision']):
            raise ValueError('Resume must preserve already trained warning weights and Adam moments')
        if a!=b:changes[key]=dict(before=a,after=b)
    if changes or reward_change:
        if before!=after or saved.get('in_progress') is not None:
            raise ValueError('Guidance/geometry can change only at a committed segment with unchanged other inputs/tasks/rewards')
        return dict(route_prior_added=False,guidance_changed=changes,
            event_progress_added=bool(reward_change and add_events), reward_changed=reward_change,
            completion_pace_target_changed=pace_target_changed,
            existing_brain_and_Adam_preserved=not pace_target_changed and not (reward_change and add_events),
            existing_policy_brain_Adam_preserved=True,saved_gameplay_and_recurrent_states_preserved=True,
            added_feature_columns=126 if new_corridor and not old_corridor else 0)
    old_future = before.pop('future_rules', {})
    new_future = after.pop('future_rules', {})
    if old_future != new_future:
        if (before != after or old_future.get('enabled') or not new_future.get('enabled')
                or after.get('route_prior',{}).get('enabled') or saved.get('in_progress') is not None):
            raise ValueError('Future rules require a committed segment and unchanged NBNM objective/tasks')
        return dict(route_prior_added=False, future_rules_added=True,
            existing_brain_and_Adam_preserved=True, new_sensory_weights='zero initialized',
            added_feature_columns=144, saved_gameplay_and_recurrent_states_preserved=True)
    old_prior = before.pop('route_prior', {})
    new_prior = after.pop('route_prior', {})
    if (before != after or old_prior.get('enabled') or not new_prior.get('enabled')
            or saved.get('in_progress') is not None):
        raise ValueError('A route prior can only be added to a committed segment with the same NBNM objective/tasks')
    return dict(route_prior_added=True, existing_brain_and_Adam_preserved=True,
        new_sensory_weights='zero initialized', added_feature_columns=108,
        saved_gameplay_and_recurrent_states_preserved=True)


def preflight(checkpoint, requested, objective_patch=None, guidance_patch=None):
    """Check the actual durable cursor and the pending on-policy tape."""
    from pathlib import Path
    from ..nbnm_run import load_cpu_checkpoint
    from ..nbnm_pending_guidance import for_saved_rollout
    saved = load_cpu_checkpoint(checkpoint)
    spec = requested
    if objective_patch:
        spec, _ = objective_for_saved_rollout(spec, saved)
    if guidance_patch:
        spec, _ = for_saved_rollout(spec, saved)
    migration = validate_resume(saved, spec)
    pending = saved['in_progress']
    pending_updates = 0
    if pending:
        tape = Path(pending['rollout'])
        rollout = load_cpu_checkpoint(tape)
        rows = int(rollout['valid_mask'].sum()) if 'valid_mask' in rollout else rollout['actions'].numel()
        if rows != pending['collection']['student_actions'] or rows % 256:
            raise ValueError('Saved training cursor and real PPO rows differ')
        pending_updates = rows // 256 - len(pending['logs'])
        if pending_updates < 0:
            raise ValueError('Saved PPO cursor exceeds its rollout')
    return dict(checkpoint=str(checkpoint), cycle=saved['cycle'], completed_decisions=saved['completed_decisions'],
        optimizer_updates=saved['optimizer_updates'], pending_rollout=pending['rollout'] if pending else None,
        pending_updates=pending_updates, saved_worlds=len((saved.get('native_continuation') or {}).get('worlds', [])),
        migration=migration)


def pending_objective(config_path):
    from pathlib import Path
    from ..thrl_study_io import read_shared_json
    path = Path(config_path).with_suffix('.next-resume.json')
    if not path.exists():
        return None
    patch = read_shared_json(path)
    if patch.get('schema') != 'nbnm-next-resume-objective-v1':
        raise ValueError('Unknown next-resume objective patch')
    fields = patch['objective']
    if set(fields) != {'progress_revision', 'progress_weight'} or fields['progress_revision'] != 'native-events-v1':
        raise ValueError('Next-resume patch only enables native event progress')
    return patch


def with_pending_objective(spec, patch):
    result = deepcopy(spec)
    if patch is not None:
        result['nbnm']['objective'].update(patch['objective'])
        for entry in result.get('task_entries', {}).values():
            if 'objective' in entry:
                entry['objective'].update(patch['objective'])
    return result


def objective_for_saved_rollout(spec, saved):
    """Finish an old on-policy tape under its original critic/actor objective."""
    if saved.get('in_progress') is None or saved['spec']['nbnm']['objective'] == spec['nbnm']['objective']:
        return spec, False
    result = deepcopy(spec)
    result['nbnm']['objective'] = deepcopy(saved['spec']['nbnm']['objective'])
    for key, entry in result.get('task_entries', {}).items():
        old = saved['spec'].get('task_entries', {}).get(key, {})
        if 'objective' in old:
            entry['objective'] = deepcopy(old['objective'])
        else:
            entry.pop('objective', None)
    return result, True


def reset_critic_output(model, optimizer, index):
    """Reset one output of the independent or packed value heads."""
    import torch
    from torch import nn
    from ..fly_gru_aligned_model import ThreeDNValueCritics
    if isinstance(model.critic, ThreeDNValueCritics):
        layer = model.critic.heads[index]
        row = slice(None)
        layout = 'three-independent-DN-critics'
    else:
        layer = [m for m in model.critic.modules() if isinstance(m, nn.Linear)][-1]
        if layer.out_features != 3:
            raise ValueError('The event objective retains three critic outputs')
        row = index
        layout = 'packed-three-output-layer'
    with torch.no_grad():
        for parameter in (layer.weight, layer.bias):
            if parameter is None:
                continue
            parameter[row].zero_()
            for value in optimizer.state.get(parameter, {}).values():
                if torch.is_tensor(value) and value.shape == parameter.shape:
                    value[row].zero_()
    return dict(output_row=index, critic_layout=layout, brain_movement_risk_preserved=True,
        other_critics_preserved=True, optimizer_step_preserved=True)


def reset_event_critic(model, optimizer):
    return dict(reset_critic_output(model, optimizer, 2), first_two_critics_preserved=True,
        only_third_row_Adam_moments_cleared=True)


def record_objective_activation(config_path, spec, patch, output, completed, updates, migration):
    from pathlib import Path
    from ..thrl_study_io import read_shared_json
    from ..thrl_durable_records import write_json
    # Persist only the authorized objective patch, preserving any other launch edits.
    current = with_pending_objective(read_shared_json(config_path), patch)
    write_json(config_path, current)
    write_json(Path(output) / 'configuration.json', spec)
    write_json(Path(output) / 'event-reward-activation.json', dict(
        schema='nbnm-event-reward-activation-v1', revision=patch['objective']['progress_revision'],
        at_decisions=completed, at_optimizer_updates=updates, objective=spec['nbnm']['objective'],
        migration=migration, old_rollout_relabelled=False, native_history_credit_replayed=False))
