"""Pure sparse-snapshot collision telemetry; never advances or reads a game.

A state4 observation proves a collision window was visible at that checkpoint.
Neither equal interval endpoints nor stable inventory prove the intervening
native frames were safe. Invulnerability causes (Bomb/respawn/etc.) are unknown.
"""
from __future__ import annotations

from copy import deepcopy

SCHEMA = 'th14-sparse-collision-telemetry-v1'
SAMPLE_SCHEMA = 'th14-sparse-collision-sample-v1'
PROTECTION_CLASSES = ('protected','unprotected','collision-pending','nonplay','unknown')


def _get(value,key,default=None):
    return value.get(key,default) if isinstance(value,dict) else getattr(value,key,default)


def _integer(value,*,minimum=0):
    return value if type(value) is int and value >= minimum else None


def snapshot_collision_sample(snapshot, *, episode_key, decision_index=None, dropped_before=0):
    """Extract only present fields. Counters require explicit observed=True.

    `episode_key` is supplied by the caller's verified ownership identity, e.g.
    run/worker/PID/creation-time. It is not guessed from a stage or inventory.
    Missing counters remain None; lower bomb/life inventory never fills them.
    """
    if not isinstance(episode_key,(str,int)) or isinstance(episode_key,bool):
        raise ValueError('Explicit scalar source episode identity required')
    if type(dropped_before) is not int or dropped_before < 0:
        raise ValueError('dropped_before must be a nonnegative integer')
    if decision_index is not None and _integer(decision_index) is None:
        raise ValueError('Decision index must be observed or None')
    player=_get(snapshot,'player');extra=_get(snapshot,'extra',{})
    events=_get(extra,'thrl_events',{})
    counters_observed=_get(events,'observed') is True
    return dict(schema=SAMPLE_SCHEMA,episode_key=episode_key,
        frame=_integer(_get(snapshot,'frame')),stage=_integer(_get(snapshot,'stage')),
        stage_frame=_integer(_get(snapshot,'stage_frame')),
        consistent=_get(snapshot,'consistent') is True,
        screen=_integer(_get(snapshot,'screen')),dialogue=_get(snapshot,'dialogue'),
        player_present=player is not None,
        player_state=_integer(_get(player,'state')),
        invulnerable_frames=_integer(_get(player,'invulnerable_frames')),
        deathbomb_timer=_integer(_get(player,'deathbomb_timer')),
        bomb_stock=_integer(_get(snapshot,'bombs')),
        misses=_integer(_get(events,'misses')) if counters_observed else None,
        bombs_used=_integer(_get(events,'bombs_used')) if counters_observed else None,
        decision_index=decision_index,dropped_before=dropped_before)


def new_collision_telemetry(*, expected_frame_stride=None):
    if expected_frame_stride is not None and (
            type(expected_frame_stride) is not int or expected_frame_stride < 1):
        raise ValueError('Expected stride must be a positive integer or None for sparse sidecar polling')
    return dict(schema=SCHEMA,expected_frame_stride=expected_frame_stride,
        accepted_samples=0,duplicate_samples=0,dropped_samples=0,drop_reasons={},
        episode_segments=0,current_episode_key=None,last=None,last_definite_player_state=None,
        observed_run_active=False,continuity_broken=True,
        state4_samples=0,state4_observation_runs=0,state4_event_lower_bound=0,
        usable_state4_samples=0,timer1_state4_samples=0,
        first_observed_collision=None,current_first_observed_collision=None,
        completed_episode_first_collisions=[],
        current_state4_lower_bound=0,current_miss_increases=0,closed_collision_lower_bound=0,
        miss_counter_increases=0,bomb_counter_increases=0,
        miss_counter_intervals=0,bomb_counter_intervals=0,counter_regressions=0,
        current_counter_regressions=0,miss_counter_observed_samples=0,bomb_counter_observed_samples=0,
        protection_samples={key:0 for key in PROTECTION_CLASSES},
        native_frame_span=0,unsampled_native_frame_checkpoints=0,
        intervals=0,multiple_frame_intervals=0,max_frame_interval=0,unexpected_stride_intervals=0,
        current_clean_span=None,longest_observed_clean_span=None)


def _check_state(state):
    if not isinstance(state,dict) or state.get('schema') != SCHEMA:
        raise ValueError('Versioned collision telemetry state required')


def _drop_mutable(state,reason,count=1):
    state['dropped_samples']+=count
    # A small fixed vocabulary keeps arbitrary source errors out of retained
    # telemetry and bounds pure-state copying regardless of capture duration.
    if reason not in ('inconsistent','missing-frame','frame-regression','conflicting-same-frame',
                      'read-error','missing-owned-process','source-reported-drop'):
        reason='other'
    state['drop_reasons'][reason]=state['drop_reasons'].get(reason,0)+count
    state['continuity_broken']=True;state['observed_run_active']=False
    state['current_clean_span']=None


def record_collision_drop(state, *, reason='inconsistent', count=1):
    _check_state(state)
    if type(count) is not int or count < 1:raise ValueError('Positive dropped sample count required')
    result=deepcopy(state);_drop_mutable(result,reason,count);return result


def _protection(sample):
    if sample.get('screen') != 7 or sample.get('dialogue') is True:
        return 'nonplay' if sample.get('screen') is not None else 'unknown'
    if sample.get('dialogue') is not False or not sample.get('player_present'):
        return 'unknown'
    state=sample.get('player_state');timer=sample.get('invulnerable_frames')
    if state == 4:return 'collision-pending'
    if timer is None:return 'unknown'
    if timer > 0:return 'protected'
    return 'unprotected' if state == 1 else 'unknown'


def _collision_bound(state):
    # Miss increases may duplicate visible state4 collisions. max, not sum.
    # A reset/nonmonotonic counter cannot safely strengthen that lower bound.
    current=state['current_state4_lower_bound']
    if state['current_counter_regressions']==0:
        current=max(current,state['current_miss_increases'])
    return state['closed_collision_lower_bound']+current


def observe_collision_sample(state, sample):
    """Return a new bounded aggregate; neither argument is mutated.

    Duplicate same-frame snapshots do not overweight protection fractions.
    A changed source episode starts a new segment. A frame regression without
    a changed identity is rejected, not silently treated as a fresh game.
    """
    _check_state(state)
    if not isinstance(sample,dict) or sample.get('schema') != SAMPLE_SCHEMA:
        raise ValueError('Use snapshot_collision_sample to supply a versioned observation')
    if not isinstance(sample.get('episode_key'),(str,int)) or isinstance(sample.get('episode_key'),bool):
        raise ValueError('Explicit source episode identity required')
    result=deepcopy(state);sample=deepcopy(sample)
    lost=sample.get('dropped_before',0)
    if type(lost) is not int or lost < 0:raise ValueError('Invalid dropped_before')
    if lost:_drop_mutable(result,'source-reported-drop',lost)
    if sample.get('consistent') is not True:
        _drop_mutable(result,'inconsistent');return result
    if _integer(sample.get('frame')) is None:
        _drop_mutable(result,'missing-frame');return result
    new_episode=result['current_episode_key'] != sample['episode_key']
    if new_episode:
        if result['last'] is not None:
            result['closed_collision_lower_bound']=_collision_bound(result)
            result['completed_episode_first_collisions'].append(dict(
                episode_key=result['current_episode_key'],first_observed_collision=result['current_first_observed_collision']))
            result['completed_episode_first_collisions']=result['completed_episode_first_collisions'][-32:]
        result.update(current_episode_key=sample['episode_key'],last=None,
            last_definite_player_state=None,observed_run_active=False,continuity_broken=True,
            current_first_observed_collision=None,current_state4_lower_bound=0,
            current_miss_increases=0,current_counter_regressions=0,current_clean_span=None)
        result['episode_segments']+=1
    previous=result['last'];delta=None
    if previous is not None:
        delta=sample['frame']-previous['frame']
        if delta < 0:
            _drop_mutable(result,'frame-regression');return result
        if delta == 0:
            compared=('stage','stage_frame','player_present','player_state','invulnerable_frames',
                      'deathbomb_timer','bomb_stock','misses','bombs_used','screen','dialogue')
            if any(sample.get(key)!=previous.get(key) for key in compared):
                _drop_mutable(result,'conflicting-same-frame')
            else:result['duplicate_samples']+=1
            return result
        result['native_frame_span']+=delta
        result['unsampled_native_frame_checkpoints']+=max(0,delta-1)
        result['intervals']+=1;result['multiple_frame_intervals']+=int(delta>1)
        result['max_frame_interval']=max(result['max_frame_interval'],delta)
        stride=result['expected_frame_stride']
        if stride is not None and delta>stride:
            result['unexpected_stride_intervals']+=1
            result['continuity_broken']=True;result['observed_run_active']=False
            result['current_clean_span']=None
    result['accepted_samples']+=1
    for counter in ('misses','bombs_used'):
        label='miss' if counter=='misses' else 'bomb'
        if _integer(sample.get(counter)) is not None:result[label+'_counter_observed_samples']+=1
    increments={'misses':None,'bombs_used':None}
    counter_regressed=False
    if previous is not None:
        for counter in increments:
            old=_integer(previous.get(counter));new=_integer(sample.get(counter))
            if old is None or new is None:continue
            label='miss' if counter=='misses' else 'bomb'
            if new < old:
                result['counter_regressions']+=1;result['current_counter_regressions']+=1
                counter_regressed=True;continue
            increments[counter]=new-old
            result[label+'_counter_intervals']+=1
            result[label+'_counter_increases']+=new-old
            if counter=='misses':result['current_miss_increases']+=new-old
    if counter_regressed:
        result['continuity_broken']=True;result['observed_run_active']=False;result['current_clean_span']=None
    active_play=sample.get('screen')==7 and sample.get('dialogue') is False
    state4=active_play and sample.get('player_present') and sample.get('player_state')==4
    if state4:
        result['state4_samples']+=1
        if not result['observed_run_active']:result['state4_observation_runs']+=1
        # A drop can split one sampled run into two; it alone cannot prove two
        # native collisions. Retain the last definite state across unknowns.
        if result['last_definite_player_state'] != 4:
            result['state4_event_lower_bound']+=1;result['current_state4_lower_bound']+=1
        timer=_integer(sample.get('deathbomb_timer'));stock=_integer(sample.get('bomb_stock'))
        usable=timer is not None and timer>0 and stock is not None and stock>0
        result['usable_state4_samples']+=int(usable)
        result['timer1_state4_samples']+=int(usable and timer==1)
    result['observed_run_active']=bool(state4)
    if active_play and sample.get('player_present') and _integer(sample.get('player_state')) is not None:
        result['last_definite_player_state']=sample['player_state']
    evidence=None
    if state4:evidence='observed-state4-checkpoint'
    elif increments['misses'] is not None and increments['misses']>0:evidence='observed-miss-counter-increase'
    if evidence is not None:
        record=dict(episode_key=sample['episode_key'],observation_frame=sample['frame'],
            stage=sample.get('stage'),stage_frame=sample.get('stage_frame'),evidence=evidence,
            true_collision_frame=None,native_start_coverage_verified=False,
            previous_observation_frame=None if previous is None else previous['frame'])
        if result['first_observed_collision'] is None:result['first_observed_collision']=record
        if result['current_first_observed_collision'] is None:result['current_first_observed_collision']=record
    protection=_protection(sample);result['protection_samples'][protection]+=1
    clean=(protection=='unprotected' and not state4
           and not any(value is not None and value>0 for value in increments.values()))
    if clean:
        span=result['current_clean_span']
        if span is None or result['continuity_broken']:
            span=dict(episode_key=sample['episode_key'],first_observation_frame=sample['frame'],
                last_observation_frame=sample['frame'],span_frames=0,observations=1,
                unsampled_native_frame_checkpoints=0,miss_counter_intervals=0,bomb_counter_intervals=0)
        else:
            span['last_observation_frame']=sample['frame'];span['span_frames']+=delta
            span['observations']+=1;span['unsampled_native_frame_checkpoints']+=max(0,delta-1)
            span['miss_counter_intervals']+=int(increments['misses'] is not None)
            span['bomb_counter_intervals']+=int(increments['bombs_used'] is not None)
        result['current_clean_span']=span
        longest=result['longest_observed_clean_span']
        if longest is None or (span['span_frames'],span['observations'])>(longest['span_frames'],longest['observations']):
            result['longest_observed_clean_span']=deepcopy(span)
    else:result['current_clean_span']=None
    result['last']=sample;result['continuity_broken']=False
    return result


def collision_telemetry_summary(state):
    _check_state(state)
    classes=state['protection_samples'];classified=classes['protected']+classes['unprotected']
    span=deepcopy(state['longest_observed_clean_span'])
    if span is not None:
        intervals=max(0,span['observations']-1)
        span.update(scope='clean-observed-checkpoints-only-not-proven-continuous-dodging',
            true_uninterrupted_collision_and_bomb_free_duration_frames=None,
            all_miss_counter_intervals_observed=intervals>0 and span['miss_counter_intervals']==intervals,
            all_bomb_counter_intervals_observed=intervals>0 and span['bomb_counter_intervals']==intervals,
            unseen_between_snapshot_collisions_or_protection_unknown=True)
    return dict(schema=SCHEMA,scope='read-only-consistent-snapshot-telemetry-not-native-frame-truth',
        expected_frame_stride=state['expected_frame_stride'],accepted_samples=state['accepted_samples'],
        duplicate_samples=state['duplicate_samples'],dropped_samples=state['dropped_samples'],
        drop_reasons=deepcopy(state['drop_reasons']),episode_segments=state['episode_segments'],
        current_episode_key=state['current_episode_key'],
        first_observed_collision=deepcopy(state['first_observed_collision']),
        current_first_observed_collision=deepcopy(state['current_first_observed_collision']),
        true_first_native_collision_frame=None,
        completed_episode_first_collisions=deepcopy(state['completed_episode_first_collisions']),
        collision=dict(state4_observations=state['state4_samples'],
            state4_observation_runs=state['state4_observation_runs'],
            observation_runs_are_native_window_count=False,
            visible_state4_collision_events_lower_bound=state['state4_event_lower_bound'],
            total_collision_events_lower_bound=_collision_bound(state),
            usable_state4_observations=state['usable_state4_samples'],
            timer1_usable_observations=state['timer1_state4_samples'],
            true_native_collision_count=None,unobserved_collisions_unknown=True),
        counters=dict(miss_increases=(state['miss_counter_increases'] if state['miss_counter_intervals'] else None),
            bomb_increases=(state['bomb_counter_increases'] if state['bomb_counter_intervals'] else None),
            miss_observed_samples=state['miss_counter_observed_samples'],
            bomb_observed_samples=state['bomb_counter_observed_samples'],
            miss_comparable_intervals=state['miss_counter_intervals'],bomb_comparable_intervals=state['bomb_counter_intervals'],
            counter_regressions=state['counter_regressions'],inventory_deltas_used_as_events=False),
        protection=dict(observation_counts=deepcopy(classes),
            protected_fraction_of_classifiable_snapshots=classes['protected']/classified if classified else None,
            classifiable_snapshots=classified,unclassified_snapshots=state['accepted_samples']-classified,
            native_protected_duration_frames=None,native_unprotected_duration_frames=None,
            protection_cause='unknown; invulnerability may be Bomb, respawn or another native protection',
            weighting='unique observed frame checkpoints, not elapsed native time'),
        coverage=dict(native_frame_span=state['native_frame_span'],
            unsampled_native_frame_checkpoints=state['unsampled_native_frame_checkpoints'],
            observation_intervals=state['intervals'],multiple_frame_intervals=state['multiple_frame_intervals'],
            max_frame_interval=state['max_frame_interval'],unexpected_stride_intervals=state['unexpected_stride_intervals'],
            complete_native_frame_coverage=False,observed_from_native_episode_start=False),
        longest_observed_clean_span=span,
        longest_true_continuous_unassisted_dodge_frames=None,
        training_modified=False,learned_dodging_claimed=False)
