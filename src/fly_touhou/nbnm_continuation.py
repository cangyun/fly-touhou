"""Durable native continuation using executed actions, not edited game memory.

Rebuild sensor/teacher history in CPU workers and restore the saved SNN state.
Replayed actions are never new PPO samples, episodes, or optimizer updates.
"""
from copy import deepcopy
import json
import time
import numpy as np

SCHEMA='th14-nbnm-live-world-v1'


def semantic(snapshot):
    from .native_replay import semantic_state
    from .nbnm_replay_state import gameplay_state
    return gameplay_state(json.loads(json.dumps(semantic_state(snapshot))))


def export_world(env):
    if env._needs_reset:raise RuntimeError('Checkpoint an acknowledged live world after its terminal reset')
    rotation=env._task_rotation
    guide=getattr(env,'_demonstration_guide',None)
    return dict(schema=SCHEMA,native_seed=env.native_seed,
        task=deepcopy(env.nbnm['task']),task_key=env.nbnm.get('task_key'),
        task_cursor=rotation.last_cursor if rotation else None,
        next_task_cursor=rotation.cursor if rotation else None,
        selected_entry=deepcopy(getattr(env,'_selected_entry',None)),
        native_episode=env.episode_id,full_story_episode=env._full_story_episode,
        bank_rng=deepcopy(env._bank_rng.bit_generator.state),
        actions=list(env._continuation_actions),landing=semantic(env.snapshot),
        features=env.encoder.last_result['features'].copy(),
        corridor_risk=deepcopy(env.encoder.corridor_risk),
        native_rule_geometry=deepcopy(env.nbnm.get('native_rule_geometry')),
        demonstration_reference=guide.export_reference() if guide is not None else None,
        task_report=env.task_tracker.report(env.snapshot,terminal=0),
        task_returns=env._task_returns.copy(),
        calm_cost_spent=getattr(env,'_calm_cost_spent',0.),
        reward_progress_revision=env.objective.progress_revision,
        event_confirmation_revision=env.nbnm.get('event_confirmation_revision'),
        event_reward_activation_progress=getattr(env,'_event_reward_activation_progress',0.))


def restore_world(env, capsule):
    if capsule.get('schema')!=SCHEMA:raise ValueError('Unknown native continuation capsule')
    began=time.monotonic()
    if env._task_rotation is not None:env._task_rotation.cursor=capsule['task_cursor']
    # Keep native episode identity consistent with the saved recurrent history.
    env._full_story_episode=capsule['full_story_episode']-1
    env.episode_id=capsule['native_episode']-1
    options=dict(native_seed=capsule['native_seed'],_continuation_entry=capsule['selected_entry'])
    # Restore all sensor and stateful rule-teacher history. Pure multileg
    # preference searches for already-executed actions are unused and can be
    # skipped; compute the final current-state label once before resuming.
    env._replaying_saved_actions=True
    capture_prefix=bool(env.nbnm.get('collect_route_samples'))
    # A 281-column learner never had a corridor or its carried path. Preserve
    # its native/brain history; first construct the new 126 columns at landing.
    # An existing 407-column checkpoint still replays its full corridor history.
    old_corridor_input=(not capture_prefix and bool(env.nbnm.get('corridor_risk'))
        and len(capsule['features'])==281)
    # A 407-column checkpoint can have a different planner with the same
    # layout. Reproduce the observations that its saved brain actually saw.
    requested_corridor=env.encoder.corridor_risk
    saved_corridor=capsule.get('corridor_risk')
    changed_planner=bool(saved_corridor and requested_corridor
        and saved_corridor!=requested_corridor and len(capsule['features'])==407)
    if changed_planner:
        env.encoder.corridor_risk=saved_corridor
    if old_corridor_input:
        env.encoder.replaying_before_corridor_migration=True
    prefix_features=[];prefix_actions=[]
    try:
        observation,info=env.reset(options=options)
        if capture_prefix:
            original_prefix=info.get('policy_burn_in') or {}
            prefix_features.extend(original_prefix.get('features',[]))
            prefix_actions.extend(original_prefix.get('actions',[]))
        if env.nbnm['task']!=capsule['task']:raise ValueError('Continuation task changed')
        for action in capsule['actions']:
            if capture_prefix:
                prefix_features.append(observation['features'].copy());prefix_actions.append(action)
            observation,_,done,truncated,info=env.step(action)
            if done or truncated:raise RuntimeError('Saved live trajectory terminated during native restoration')
    finally:
        env._replaying_saved_actions=False
        env.encoder.corridor_risk=requested_corridor
        if old_corridor_input:
            env.encoder.replaying_before_corridor_migration=False
    if old_corridor_input:
        observation=env.encoder.finish_corridor_migration()
        info=dict(info,**env._info())
    from .nbnm_replay_state import practice_variation, upgrade_continuation_interrupts
    expected=upgrade_continuation_interrupts(capsule['landing'])
    differences=practice_variation(expected,semantic(env.snapshot))
    if differences:
        from .thrl_durable_records import write_json
        actual=semantic(env.snapshot)
        write_json(env.private_appdata/'continuation-difference.json',dict(
            expected={k:expected[k] for k in differences},
            actual={k:actual[k] for k in differences}))
        raise RuntimeError('Continuation gameplay differs: '+','.join(differences))
    # Bomb inventory is masked out by NBNM; every policy input must still agree.
    old_features=capsule['features']
    new_features=observation['features']
    prior_migration=(len(old_features)==281 and len(new_features)==389
        and env.nbnm.get('route_prior',{}).get('enabled'))
    future_migration=(len(old_features)==281 and len(new_features)==425
        and env.nbnm.get('future_rules',{}).get('enabled'))
    corridor_migration=(len(old_features)==281 and len(new_features)==407 and bool(env.nbnm.get('corridor_risk')))
    from .nbnm_geometry_revision import verify_restored_features
    if changed_planner:
        env.encoder.corridor_risk=saved_corridor
    try:
        geometry_migration=verify_restored_features(env,capsule,observation)
    finally:
        env.encoder.corridor_risk=requested_corridor
    if geometry_migration is None:
        np.testing.assert_allclose(new_features[:281] if prior_migration or future_migration or corridor_migration else new_features,
            old_features,rtol=1e-5,atol=1e-6)
    changed_corridor_columns=0
    if changed_planner:
        # The old carried path is only a candidate. The new planner
        # revalidates it from the actual position before making it available.
        observation=env.encoder.finish_corridor_migration()
        changed_corridor_columns=int(np.count_nonzero(
            np.abs(observation['features'][281:]-new_features[281:])>1e-6))
        info=dict(info,**env._info())
    if changed_planner or old_corridor_input:
        env._last_measurable_observation=env._copy_observation(observation)
    reference_restored=False
    if env.nbnm.get('teacher_mode')=='nbnm-route-v1' and getattr(env,'_teacher_label',None) is not None:
        from .nbnm_demonstration_routes import guide_for_environment
        guide=guide_for_environment(env)
        if guide is not None:
            reference_restored=guide.restore_reference(capsule.get('demonstration_reference'))
        # Restore the complete current teacher distribution only at the
        # landing. The executed prefix needed only its stateful updates.
        info=env._label_current(info)
    reward_migration = restore_reward_accounting(env, capsule)
    env._bank_rng.bit_generator.state=deepcopy(capsule['bank_rng'])
    if env._task_rotation is not None:env._task_rotation.cursor=capsule['next_task_cursor']
    info.pop('policy_burn_in',None)
    if capture_prefix:
        info['policy_burn_in']=dict(features=prefix_features,actions=prefix_actions)
    info['native_continuation']=dict(status='matched_native_gameplay_and_policy_input',
        replayed_actions=len(capsule['actions']),seconds=time.monotonic()-began,
        added_route_prior_columns=108 if prior_migration else 0,
        added_future_rule_columns=144 if future_migration else 0,
        added_corridor_columns=126 if corridor_migration else 0,
        historical_corridors_absent_from_checkpoint_skipped=old_corridor_input,
        saved_corridor_profile_reproduced=changed_planner,
        refreshed_corridor_columns=changed_corridor_columns,
        historical_multileg_preferences_skipped=True,
        demonstration_reference_restored=reference_restored,
        reward_migration=reward_migration,geometry_migration=geometry_migration,
        new_PPO_samples=0,teacher_executed_actions=0,gameplay_memory_edited=False)
    from .thrl_durable_records import write_json
    write_json(env.private_appdata/'continuation.json',info['native_continuation'])
    return observation,info


def checkpoint_continuation(state):
    """Recover the original planner of legacy capsules from their own run spec."""
    capsule=state.get('native_continuation')
    if capsule is None:
        return None
    spec=state['spec']
    worlds=[]
    for world in capsule['worlds']:
        if 'corridor_risk' not in world and len(world['features'])==407:
            task_settings=spec.get('task_entries',{}).get(world.get('task_key'),{})
            original=task_settings.get('corridor_risk',spec['nbnm'].get('corridor_risk'))
            world=dict(world,corridor_risk=deepcopy(original))
        worlds.append(world)
    return dict(capsule,worlds=worlds)


def restore_reward_accounting(env, capsule):
    """Replayed historical events rebuild the ledger but earn no new PPO credit."""
    spent = float(capsule.get('calm_cost_spent', 0.))
    env._calm_cost_spent = spent
    motion = getattr(env, '_calm_movement', None)
    if motion is not None:
        motion.spent = spent
    # Old actions reproduced the event ledger. Their already-accounted cost is
    # restored once without issuing rewards for historical actions.
    env._task_returns[2] -= spent
    before = capsule.get('reward_progress_revision')
    after = env.objective.progress_revision
    old_confirmation = capsule.get('event_confirmation_revision')
    new_confirmation = getattr(env, 'nbnm', {}).get('event_confirmation_revision')
    if before == after and after and old_confirmation != new_confirmation:
        from .nbnm_event_progress import CONFIRMATION_REVISION
        if old_confirmation is not None or new_confirmation != CONFIRMATION_REVISION:
            raise ValueError('Unsupported event confirmation migration')
        np.testing.assert_allclose(env._task_returns[:2], capsule['task_returns'][:2], rtol=1e-5, atol=1e-6)
        # Newly visible historical completions rebuild the de-duplication ledger
        # but are not replayed as new rewards or PPO trajectories.
        baseline = float(env._task_returns[2] - capsule['task_returns'][2])
        env._event_reward_activation_progress = baseline
        env._task_returns[:] = capsule['task_returns']
        migration = dict(before=old_confirmation, after=new_confirmation,
            confirmation_only=True, historical_events_rewarded=False,
            historical_progress_baseline=baseline, existing_game_position_preserved=True)
    elif before == after:
        baseline = float(capsule.get('event_reward_activation_progress', 0.))
        env._event_reward_activation_progress = baseline
        if after:
            env._task_returns[2] -= baseline
        np.testing.assert_allclose(env._task_returns, capsule['task_returns'], rtol=1e-5, atol=1e-6)
        migration = None
    elif before is None and after == 'native-events-v1':
        np.testing.assert_allclose(env._task_returns[:2], capsule['task_returns'][:2], rtol=1e-5, atol=1e-6)
        env._event_reward_activation_progress = float(env._task_returns[2])
        env._task_returns[2] = 0.
        migration = dict(before=before, after=after, historical_events_rewarded=False,
            historical_progress=env._event_reward_activation_progress,
            old_HP_return=float(capsule['task_returns'][2]), existing_game_position_preserved=True)
    else:
        raise ValueError('Unsupported native continuation reward migration')
    env._episode_returns[:] = env._task_returns
    env._episode_raw_returns[:] = env._task_returns
    return migration


def export_collector(collector):
    if collector.vector._inflight:raise RuntimeError('Save worlds only at an acknowledged update boundary')
    messages=collector.vector._exchange(list(range(collector.workers)),'checkpoint')
    return dict(schema='nbnm-live-collector-v1',worlds=[m['continuation'] for m in messages],
        brain=[x.detach().cpu().clone() for x in collector.state.tensors()],
        starts=collector.starts.copy(),actions=collector.actions,completed=collector.completed,
        coverage=deepcopy(collector.coverage),policy_version=collector.policy_version,
        adaptive_course=(collector.adaptive_course.state_dict() if getattr(collector,'adaptive_course',None) else None),
        adaptive_stage_chain=(collector.adaptive_stage_chain.state_dict() if getattr(collector,'adaptive_stage_chain',None) else None))


def restore_collector(collector, capsule, model):
    from .spiking_engineered_control import SpikingState
    if capsule.get('schema')!='nbnm-live-collector-v1':raise ValueError('Unknown collector continuation')
    collector.state=SpikingState(*(v.to(model.device) for v in capsule['brain']))
    collector.starts=np.asarray(capsule['starts'],dtype=bool).copy()
    collector.actions=capsule['actions'];collector.completed=capsule['completed']
    collector.coverage=capsule['coverage'];collector.policy_version=capsule['policy_version']
