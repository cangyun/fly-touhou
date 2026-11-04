"""Native pre/post contact measurements, stored only after the executed action.

These records never enter policy inputs, teacher labels or PPO targets. The
two observed endpoints do not identify the actual collision object or prove
that any measured past velocity will continue in the future.
"""
import math
import numpy as np


def recorded_policy_context(previous,current,before,*,risk_scale=1.,history=32):
    """Small pre-action history from already recorded student decisions.

    No model call, teacher query or game step. Keep the pre-outcome trust mask,
    because training can veto that mask after observing the actual collision.
    This diagnostic never enters the policy or optimizer.
    """
    episode=int(current['recorded_native_episode'])
    records=[r for r in previous[-(history-1):]
             if int(r['recorded_native_episode'])==episode]+[current]
    def numbers(value):
        a=np.asarray(value,dtype=float)
        return np.where(np.isfinite(a),a,None).tolist()
    rows=[]
    for r in records:
        rows.append(dict(context=dict(stage=int(r['recorded_native_stage']),
                episode_frames=int(r['recorded_episode_frames']),native_episode=episode),
            action=int(r['actions']),normal=bool(r['unprotected_mask']),
            features=numbers(r['features']),logits=numbers(r['recorded_policy_logits']),
            behavior_logits=numbers(r['recorded_behavior_policy_logits']),
            horizon_scores=numbers(r['recorded_neural_horizon_scores']),
            route_scores=numbers(r['recorded_route_scores']),margins=numbers(r['margins']),
            trusted=np.asarray(r['recorded_pre_action_trusted'],bool).tolist(),
            teacher_probabilities=numbers(r['rule_probabilities']),teacher_weight=float(r['rule_weights'])))
    rows[-1]['context'].update({k:before.get(k) for k in ('lives','bombs','misses','bombs_used')})
    return dict(schema='compact-live-training-contact-policy-v1',neural_risk_scale=float(risk_scale),
        pre_action_tail=rows,history_limit_decisions=history,
        scope='Changing-policy training;original pre-action scores. Forecast alternatives are not native counterfactual outcomes.',
        model_calls=0,game_steps=0,teacher_queries=0)


def _geometry_replay(snapshot, encoding):
    """Archive the actual encoder geometry, including observed laser motion.

    Contact telemetry is optional: older/unmatched encoders produce an
    explicit unavailable record rather than interrupting a native game or
    supplying a mismatched scene for a later replay.
    """
    if (encoding.get('frame'), encoding.get('stage')) != (snapshot.frame, snapshot.stage):
        return dict(available=False, reason='encoder_frame_or_stage_mismatch')
    if not all(k in encoding for k in ('source_features25', 'source_objects29', 'features')):
        return dict(available=False, reason='complete_geometry_not_retained_by_this_encoder')
    f = np.asarray(encoding['source_features25'])
    o = np.asarray(encoding['source_objects29'])
    x = np.asarray(encoding['features'])
    if (f.shape != (25,) or o.ndim != 2 or o.shape[1] != 29
            or not all(np.isfinite(a).all() for a in (f, o, x))):
        return dict(available=False, reason='geometry_not_finite_or_not25_by_Nx29')
    details = encoding['details']
    uncertainty = details.get('uncertainty', {})
    return dict(available=True, schema='compact-native-contact-geometry25-Nx29-v1',
        frame=snapshot.frame, stage=snapshot.stage,
        source_features25=f.tolist(), source_objects29=o.tolist(),
        encoded_features=x.tolist(), object_rows=len(o),
        laser_segment_rows=details['laser_segments'],
        full_observed_hazards=details['full_observed_hazards'],
        forecast_native_arming=bool(uncertainty.get('pending_native_arming_forecast')),
        forecast_growth_motion=bool(uncertainty.get('growth_motion_modeled')),
        source_native_forecasts=encoding.get('source_native_forecasts'),
        margins=np.asarray(details['margins']).tolist(),
        trusted=np.asarray(details['trusted']).tolist(),
        policy_features_changed=False, teacher_labels_changed=False,
        scope='Actual current encoder geometry;observed motion is not future native trajectory truth')


def _body(snapshot,prepared):
    p=snapshot.player;g=prepared['player_geometry']
    return dict(position=[p.x,p.y],state=p.state,invulnerability=p.invulnerable_frames,
        deathbomb_timer=p.deathbomb_timer,circle_component=float(g['circle_component']),
        half_extents=[float(x) for x in g['bounds_half_extents']],
        offset=[float(x) for x in g['bounds_offset']])


def _points(snapshot,prepared):
    result={}
    for b in snapshot.bullets:
        g=prepared['geometry'][b.id];raw=prepared['raw_bullets'][b.id]
        if not g['visible']:continue
        result[b.id]=dict(identity=b.id,kind=b.kind,position=[b.x,b.y],stored_velocity=[b.vx,b.vy],
            width=g['width'],height=g['height'],circle=g['circle'],collidable=g['collidable'],
            state=raw['native_state'],age=raw['native_timer'],flags=raw['native_flags'],
            native_speed=raw['native_speed'],native_angle=raw['native_angle'],
            native_animation_gate=raw['native_animation_gate'])
        result[b.id].update({key:raw[key] for key in (
            'native_ex_active_flags','native_ex_index','native_ex_current_record_hex',
            'native_ex_wrap_count','native_ex_wrap_limit','native_ex_wrap_sides','native_ex_wrap_sprite_size',
            'native_ex_speed_boost_tick','native_ex_speed_boost_time',
            'native_ex_acceleration_tick','native_ex_acceleration_time','native_ex_acceleration',
            'native_ex_acceleration_duration','native_ex_acceleration_source','native_ex_speed_delta','native_ex_wait_remaining',
            'native_ex_wait_time','native_ex_hold_remaining','native_ex_hold_time','native_ex_hold_rate',
            'native_ex_hold_forecast_enabled',
            'native_ex_wait_rate','native_motion_time_rate','native_ex_program_start_index','native_ex_program_prefix_hex',
            'native_ex_after_cancel_opcode','native_ex_after_cancel_condition',
            'native_ex8_time_rate',
            'native_ex8_tick','native_ex8_time','native_ex8_speed_delta',
            'native_ex8_angle_delta','native_ex8_duration',
            'native_ex16_tick','native_ex16_time','native_ex16_time_rate','native_ex16_duration',
            'native_ex16_repeats','native_ex16_completed','native_ex16_turn_mode',
            'native_ex16_turn_speed','native_ex16_turn_angle',
            'native_ex8_polar4_forecast_enabled') if key in raw})
    return result


def _unresolved_point_ids(prepared, encoding, *, frame, stage):
    """Map current four-frame unknown forecasts to the original bullet IDs."""
    if not encoding or (encoding.get('frame'),encoding.get('stage'))!=(frame,stage):return {}
    geometry_rows=[r for r in prepared['candidates'] if r[3] in ('bullet','enemy','boss')]
    found={}
    for row in (encoding.get('source_native_forecasts') or {}).get('bullets',()):
        index=row['object_index'];release=row['release'];coverage=release.get('known_by_frame',())
        if (len(coverage)>4 and all(coverage[1:5])) or not 0<=index<len(geometry_rows):continue
        target=geometry_rows[index]
        if target[3]=='bullet':found[target[4][1]]=release.get('reason')
    return found


def native_phase_summary(snapshot):
    """Existing same-frame ECL/HP evidence, for diagnostics only."""
    roles=snapshot.extra.get('thrl_events',{}).get('ecl_roles',{})
    resources=snapshot.extra.get('resources',{})
    return dict(stage=snapshot.stage,stage_frame=snapshot.stage_frame,
        role_observed=roles.get('observed',False),
        main_bosses=[{k:row.get(k) for k in ('id','ecl_sub_name','hp','role')}
                     for row in roles.get('main_bosses',())],
        callbacks=[dict(enemy_id=row['id'],callback=contact['special_tick_callback'])
            for row in (*resources.get('enemies',()),*resources.get('hidden_enemy_contacts',()))
            if (contact:=row.get('native_contact',{})).get('special_tick_callback')])


def contact_transition(before,after,*,before_encoding=None,after_encoding=None,teacher_label=None):
    if before is None or after is None or before.player is None or after.player is None:return None
    entered=after.player.state==4 and before.player.state!=4
    missed=after.extra.get('thrl_events',{}).get('misses',0)>before.extra.get('thrl_events',{}).get('misses',0)
    if not entered and not missed:return None
    from .thrl_geometry_observation import shared_object_candidates
    old=shared_object_candidates(before,hp_independent_contacts=True)
    new=shared_object_candidates(after,hp_independent_contacts=True)
    points=_points(before,old);following=_points(after,new);dt=after.frame-before.frame
    distance=lambda q:math.hypot(q['position'][0]-before.player.x,q['position'][1]-before.player.y)-max(q['width'],q['height'])/2
    ranked=sorted(points.values(),key=lambda q:(distance(q),q['identity']))
    # All hazards still participate in the policy encoder. This bound applies
    # only to the human-readable event record, with growth examples retained.
    selected=ranked[:12]
    selected_ids={q['identity'] for q in selected}
    selected.extend(q for q in [q for q in ranked if q['state']==2][:4] if q['identity'] not in selected_ids)
    # An unread EX command anywhere currently makes the four-frame forecast
    # unknown. Preserve a few nearest such points even if dense bullets pushed
    # them outside the twelve nearest; this is telemetry only.
    unknown=_unresolved_point_ids(old,before_encoding,frame=before.frame,stage=before.stage)
    selected_ids={q['identity'] for q in selected}
    additions=[q for q in ranked if q['identity'] in unknown and q['identity'] not in selected_ids][:8]
    selected.extend(additions)
    rows=[]
    for point in selected:
        q=dict(point);n=following.get(point['identity'])
        if point['identity'] in unknown:q['forecast4_unknown_reason']=unknown[point['identity']]
        paired=(n is not None and before.stage==after.stage and dt>0 and n['age']-q['age']==dt and n['kind']==q['kind'])
        q['observed_pair_usable']=paired
        if paired:
            q.update(after=n,observed_center_velocity=[(y-x)/dt for x,y in zip(q['position'],n['position'])],
                observed_width_change_per_frame=(n['width']-q['width'])/dt,
                observed_height_change_per_frame=(n['height']-q['height'])/dt)
        rows.append(q)
    enemies=[dict(kind=row[3],identity=list(row[4]),token=list(row[2]))
             for row in sorted(old['candidates'],key=lambda row:row[0]) if row[3] in ('enemy','boss')][:8]
    result=dict(schema='compact-SNN-contact-transition-observation-v1',before_frame=before.frame,after_frame=after.frame,
        native_frame_gap=dt,stage=before.stage,entered_contact_window=entered,native_miss_increment=missed,
        player_before=_body(before,old),player_after=_body(after,new),
        native_phase_before=native_phase_summary(before),native_phase_after=native_phase_summary(after),
        all_visible_points_before=len(points),all_visible_growth_points_before=sum(q['state']==2 for q in points.values()),
        unresolved_forecast_points_before=len(unknown),recorded_unresolved_points=sum(q['identity'] in unknown for q in selected),
        recorded_near_points=rows,near_enemy_tokens=enemies,laser_count_before=len(before.lasers),
        policy_features_changed=False,teacher_labels_changed=False,actual_collision_identity_known=False)
    if before_encoding is not None:
        result['geometry_replay_before'] = _geometry_replay(before, before_encoding)
    if after_encoding is not None:
        result['geometry_replay_after'] = _geometry_replay(after, after_encoding)
    if teacher_label is not None:
        result['teacher_guidance_before'] = {key: teacher_label.get(key) for key in
            ('frame', 'stage', 'native_episode', 'action', 'guidance_weight',
             'guidance_weight_reason', 'soft_route_support', 'full_native_support_attempt')}
    return result
