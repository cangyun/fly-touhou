"""Reachable rolling corridors for the student's current position and legal speeds.

Each first command has its own beam. Every reported endpoint has one executable
chain of four-frame commands. Search failures and predicted contacts have
distinct evidence codes. Future spawns outside the observed scene remain unknown.
"""
import numpy as np
from .native_bullet_release import share_release_sweeps
from .native_ex_conditional import share_waiting_aim
from .native_projection_cache import share_native_projections

REVISION = 'student-reachable-corridors-v1'
SCHEMA = 'connectome-nbnm-route407-corridor-v1'
HORIZONS = (4, 12, 24, 48, 96)
LEGS = (4, 12, 24, 48, 72, 96)
NAMES = ('margin48', 'margin96', 'search48', 'search96', 'turn_dx', 'turn_dy', 'confidence')
WIDTH = len(NAMES)
FEATURE_WIDTH = 281 + 18 * WIDTH


def validate(config):
    if config is None:
        return None
    permitted={'revision','actor_weight','loss_weight','beam','witness_weight','corner_reserve_stages',
        'keep_course','keep_course_weight','keep_course_stages','keep_course_spell_numbers','course_port_gain','onward_stages',
        'wide_beam_stages','reactive_retry_stages','reactive_retry_boss_cards','refinement_phases','endpoint_risk','refine_uncertain_turns',
        'revalidated_course_only','refine_early_turns','order_supported_exits','refinement_first_boss_spell',
        'anticipate_line_births','anticipate_stationary_volleys','anticipate_rotating_rings','rotating_ring_arc',
        'learn_native_exits','align_pending_clearances','prefer_full_horizon','refine_reactive_turns',
        'issun_lure','issun_lure_scope','issun_lure_tail_turns','issun_guidance_weight',
        'road_exit_supervision_stages','exit_supervision','exit_supervision_48_fallback',
        'native_emission_recheck_limit','local_turn_limit','short_risk_priority'}
    if config.get('revision') != REVISION or set(config) - permitted:
        raise ValueError('Unknown student corridor configuration')
    for key in ('actor_weight', 'loss_weight'):
        value = config[key]
        if not np.isfinite(value) or not 0 < value <= 8:
            raise ValueError('Positive bounded corridor weight required: ' + key)
    if type(config.get('beam',1)) is not int or config.get('beam', 1) not in (1, 2):
        raise ValueError('Use one or two spatially distinct witnesses per first move')
    recheck = config.get('native_emission_recheck_limit', 0)
    if type(recheck) is not int or not 0 <= recheck <= 4:
        raise ValueError('Native emission rechecks accept zero to four starting courses')
    local_turns = config.get('local_turn_limit', 0)
    if type(local_turns) is not int or not 0 <= local_turns <= 2:
        raise ValueError('Local turn refinement accepts zero to two starting courses')
    if 'witness_weight' in config and (not np.isfinite(config['witness_weight']) or not 0<config['witness_weight']<=12):
        raise ValueError('Use a finite positive learned witness-logit weight up to 12')
    priority = config.get("short_risk_priority")
    if priority is not None:
        fields = {"trigger_risk", "alternative_risk", "remaining_corridor_weight"}
        if (not isinstance(priority, dict) or set(priority) != fields
                or not 0 <= priority["alternative_risk"] < priority["trigger_risk"] <= 1
                or not 0 < priority["remaining_corridor_weight"] < 1):
            raise ValueError("Specify two ordered short-risk thresholds and a positive corridor weight")
    for key in ('corner_reserve_stages','keep_course_stages','onward_stages','wide_beam_stages',
                'reactive_retry_stages','road_exit_supervision_stages'):
        stages=config.get(key,[])
        if not isinstance(stages,(tuple,list)) or any(type(s) is not int or s not in range(1,7) for s in stages):
            raise ValueError(f'{key} must use original Story stages 1..6')
    if type(config.get('exit_supervision', False)) is not bool:
        raise ValueError('exit_supervision must be a boolean')
    if config.get('exit_supervision') and config.get('road_exit_supervision_stages'):
        raise ValueError('Choose general exit supervision or the legacy road scope')
    fallback = config.get('exit_supervision_48_fallback', False)
    if type(fallback) is not bool or (fallback and not config.get('exit_supervision')):
        raise ValueError('The 48-frame fallback requires general exit supervision')
    if type(config.get('keep_course',False)) is not bool:
        raise ValueError('keep_course must be a boolean')
    if 'revalidated_course_only' in config and (
            type(config['revalidated_course_only']) is not bool or not config.get('keep_course')):
        raise ValueError('Revalidated-course scoring requires an enabled continuation readout')
    if type(config.get('refine_uncertain_turns',False)) is not bool:
        raise ValueError('refine_uncertain_turns must be a boolean')
    guidance = config.get('issun_guidance_weight', 0.)
    if not np.isfinite(guidance) or not 0 <= guidance <= .15:
        raise ValueError('Use a finite Wall of Issun supervision weight between 0 and 0.15')
    if type(config.get('issun_lure',False)) is not bool:
        raise ValueError('issun_lure must be a boolean')
    if 'issun_lure_scope' in config and (not config.get('issun_lure') or
            config['issun_lure_scope'] not in ('all','constrained')):
        raise ValueError('A Wall of Issun lure needs an enabled scope of all or constrained')
    if 'issun_lure_tail_turns' in config and (type(config['issun_lure_tail_turns']) is not bool or
            (config['issun_lure_tail_turns'] and not config.get('issun_lure'))):
        raise ValueError('Wall of Issun tail turns require an enabled lure')
    if type(config.get('anticipate_line_births',False)) is not bool:
        raise ValueError('anticipate_line_births must be a boolean')
    if type(config.get('anticipate_stationary_volleys',False)) is not bool:
        raise ValueError('anticipate_stationary_volleys must be a boolean')
    if type(config.get('anticipate_rotating_rings',False)) is not bool:
        raise ValueError('anticipate_rotating_rings must be a boolean')
    if 'rotating_ring_arc' in config and (type(config['rotating_ring_arc']) is not bool
            or (config['rotating_ring_arc'] and not config.get('anticipate_rotating_rings'))):
        raise ValueError('Arc geometry requires rotating-ring forecasts')
    if type(config.get('align_pending_clearances',False)) is not bool:
        raise ValueError('align_pending_clearances must be a boolean')
    if 'learn_native_exits' in config and (type(config['learn_native_exits']) is not bool
            or (config['learn_native_exits'] and not config.get('witness_weight'))):
        raise ValueError('Original successful continuations supervise the corridor witness readout')
    if 'prefer_full_horizon' in config and (type(config['prefer_full_horizon']) is not bool
            or (config['prefer_full_horizon'] and not config.get('learn_native_exits'))):
        raise ValueError('Full-horizon preferences must also preserve original surviving deviations')
    if type(config.get('order_supported_exits',False)) is not bool:
        raise ValueError('order_supported_exits must be a boolean')
    if 'refine_early_turns' in config and (type(config['refine_early_turns']) is not bool
            or not (config.get('refine_uncertain_turns') or config.get('refine_reactive_turns'))):
        raise ValueError('Early-turn search supplements the uncertain-turn refinement')
    if config.get('refinement_phases','all') not in ('all','boss_spells'):
        raise ValueError('Use all phases or original main-boss spells for corridor refinements')
    if 'reactive_retry_boss_cards' in config:
        cards=config['reactive_retry_boss_cards']
        if (config.get('refinement_phases')!='boss_spells' or not config.get('reactive_retry_stages')
                or not isinstance(cards,(tuple,list)) or not cards
                or any(type(card) is not int or card<1 for card in cards)):
            raise ValueError('Reactive retry spell numbers require main-boss spell scope and original stages')
    if 'refine_reactive_turns' in config and (type(config['refine_reactive_turns']) is not bool
            or (config['refine_reactive_turns'] and not config.get('reactive_retry_stages'))):
        raise ValueError('Scoped late-turn search requires a reactive retry stage')
    if 'refinement_first_boss_spell' in config and (config.get('refinement_phases')!='boss_spells'
            or type(config['refinement_first_boss_spell']) is not int or config['refinement_first_boss_spell']<1):
        raise ValueError('A phase-scoped refinement starts at a positive original main-boss spell number')
    if 'endpoint_risk' in config:
        endpoint=config['endpoint_risk']
        if (not isinstance(endpoint,dict) or not {'weight','stages'}.issubset(endpoint)
                or set(endpoint)-{'weight','stages','paired_only'}):
            raise ValueError('A learned endpoint-risk correction needs a weight and its applicable stages')
        if type(endpoint.get('paired_only',False)) is not bool:
            raise ValueError('paired_only must be a boolean')
        if not np.isfinite(endpoint['weight']) or not 0<endpoint['weight']<=8:
            raise ValueError('Use a positive endpoint-risk weight up to eight')
        stages=endpoint['stages']
        if not isinstance(stages,(list,tuple)) or not stages or any(type(s) is not int or s not in range(1,7) for s in stages):
            raise ValueError('Specify original stages for the endpoint-risk correction')
    if 'keep_course_spell_numbers' in config:
        numbers=config['keep_course_spell_numbers']
        if (not config.get('keep_course_stages') or not isinstance(numbers,(list,tuple))
                or not numbers or any(type(s) is not int or not 1<=s<=32 for s in numbers)):
            raise ValueError('A spell-specific course readout needs original stages and encoded spell numbers')
    if any(key in config for key in ('keep_course_weight','keep_course_stages','keep_course_spell_numbers','course_port_gain')):
        weight=config.get('keep_course_weight',0.)
        if not config.get('keep_course') or not np.isfinite(weight) or not 0<=weight<=8:
            raise ValueError('Course readout requires keep_course and a bounded finite weight')
        gain=config.get('course_port_gain',0.)
        if not np.isfinite(gain) or not 0<=gain<=2:
            raise ValueError('Use a finite continuation sensory gain in [0,2]')
    return config


def native_window_covered(encoding, scene, horizon):
    from .compact_native_forecast import enemy_forecast_uncertainty

    projection = scene.get('emitter_projection_horizon')
    boundary = scene.get('pending_emitter_unknown_from_tick')
    if (projection is not None and projection < horizon) or (boundary is not None and boundary <= horizon):
        return False
    for row in (scene.get('line_geometry') or {}).get('lines', ()):
        known = row['forecast'].get('known_by_frame', ())
        if len(known) <= horizon or not all(known[:horizon + 1]):
            return False
    return not enemy_forecast_uncertainty(encoding['source_features25'], encoding['source_objects29'],
                                          scene, horizon)


def _tail_scene(encoding, end):
    """Extend available programs once per scene. Retain raw geometry for uncovered tails."""
    from .native_bullet_release import predict_bullet_release
    from .native_ex_conditional import needs_player_path
    from .native_curve_motion_paths import motion_covered
    fc = encoding.get('source_native_forecasts') or {}
    result = dict(fc, bullets=[], enemies=[],
        reactive_bullets=[r for r in fc.get('reactive_bullets',()) if r['supported']])
    confidence = .75 if encoding['details']['trusted'][0] else .25
    if fc.get('pending_stationary_volleys') is not None:
        confidence=.25  # The verified stationary window ends before later EX motion.
    if fc.get('pending_rotating_rings') is not None:
        confidence=.25  # Rotation is bounded; other emitter families remain conditional.
    for row in fc.get('bullets', ()):
        release = row['release']
        raw = row.get('program_raw')
        point=row.get('reactive_point')
        if point is not None and len(release['bounds'])<=end:
            from .native_wall_geometry import _untriggered_prefix
            extension=_untriggered_prefix(point,fc['reactive_context'],end,_clearance_arrays=True)
            if extension is not None:
                release=extension
            elif (point['native']['native_state']==1 and not point['native']['native_flags'] & 0x200
                    and point['native'].get('native_motion_time_rate',1.)==1.):
                # Out of reach through frame 24 does not imply out of reach at
                # frame 96. Restore the observed callback on this candidate's path.
                result['reactive_bullets'].append(dict(object_index=row['object_index'],point=point,supported=True))
                continue
        if len(release['bounds']) <= end and raw:
            release = predict_bullet_release(raw, end, _clearance_arrays=True)
        if len(release['bounds']) <= end:
            confidence = .25
            continue
        dependent = row.get('candidate_dependent', False)
        if not all(release['known_by_frame'][:end + 1]):
            dependent = bool(raw and needs_player_path(raw))
            confidence = .25
        result['bullets'].append(dict(row, release=release, candidate_dependent=dependent))
    for row in fc.get('enemies', ()):
        ticks = row['contact']['forecast']['ticks']
        if len(ticks) >= end and len(row['motion']['positions']) > end:
            result['enemies'].append(row)
        else:
            confidence = .25
    lines = fc.get('line_geometry') or {}
    if lines:
        if lines.get('lines') and lines.get('horizon', 24) < end and fc.get('line_program_source'):
            from .native_line_laser_paths import native_line_geometry
            raw,indices=fc['line_program_source']
            lines=native_line_geometry(raw,indices,horizon=end)
        kept = [row for row in lines['lines'] if len(row['forecast']['origin']) > end]
        if len(kept) < len(lines['lines']):
            confidence = .25
        result['line_geometry'] = dict(lines, lines=kept) if kept else None
    if any(not r['supported'] for r in fc.get('reactive_bullets', ())):
        # Keep unsupported objects in the ordinary measured-geometry path.
        confidence = .25
    if (fc.get('hidden_enemy_unreadable_geometry') or
            any(c['width'] > 0 and not motion_covered(c, end)
                for c in (fc.get('curve_geometry') or {}).get('curves', ()))):
        confidence = .25
    if not np.asarray(encoding['details']['trusted']).all():
        confidence=.25
    captured_lines={row['object_index'] for row in (result.get('line_geometry') or {}).get('lines',())}
    for index,token in enumerate(encoding['source_objects29']):
        if (token[3]>.5 or token[4]>.5) and index not in captured_lines:
            confidence=.25
            break
    if confidence >= .5 and 'emitter_projection_horizon' in fc and not native_window_covered(encoding, result, end):
        confidence = .25
    return result, confidence


def _reserve(xy):
    return (np.minimum(184 - abs(xy[:, 0]), 48) / 48
            + np.minimum(np.minimum(xy[:, 1] - 32, 432 - xy[:, 1]), 64) / 64)


def _room_to_turn(xy):
    """Lateral or vertical distance available to escape a two-wall corner."""
    p=np.asarray(xy)
    side=184-np.abs(p[...,0])
    vertical=np.minimum(p[...,1]-32,432-p[...,1])
    return np.maximum(side,vertical).clip(0.)


def commands4(plan_result, action):
    """Return only an executable witness for this first action."""
    refined=plan_result.get('refined_courses',{}).get(action)
    if refined is not None:
        return refined.copy()
    retained=plan_result.get('retained')
    if retained is not None and retained[0]==action:
        return retained
    for i in reversed(range(len(LEGS))):
        row=plan_result['commands'][i,action,:i+1]
        if (row>=0).all():
            return np.repeat(row,np.diff((0,*LEGS[:i+1]))//4)
    return None


def carry_after(encoding, found, action, frame):
    if found is None or action not in range(18):
        return None
    course=commands4(found,action)
    from .thrl_movement_teacher import action_velocities
    f=encoding['source_features25']
    origin=np.array([f[0]*368-184,f[1]*400+32])
    speeds=np.trunc(action_velocities()*128)/128
    def anchor(sequence):
        if sequence is None or len(sequence)<2:
            return None
        first=int(sequence[0])
        speed=speeds[first%9 if f[18]<0 else first]
        expected=np.clip(origin+4*speed,[-184,32],[184,432])
        return dict(frame=frame+4,position=expected,course=sequence[1:].copy(),stage=encoding.get('stage'))
    primary=anchor(course)
    retained=found.get('retained')
    retained_lure=found.get('retained_lure',False)
    if primary is not None and (any(p['action']==action for p in found.get('issun_lure',{}).get('published',()))
            or (retained_lure and np.array_equal(course,retained))):
        primary['issun_lure']=True
    if retained is not None and len(retained)>1 and retained[0]!=action:
        alternate=anchor(retained)
        if retained_lure:alternate['issun_lure']=True
        if primary is None:
            return alternate
        primary['alternate']=alternate
    return primary


def pack_course(found, action):
    """A fixed-width trace of the plan actually available for the chosen move."""
    course=np.full(24,-1,np.int8)
    if found is not None and 0<=action<18:
        row=commands4(found,action)
        if row is not None:
            course[:len(row)]=row
    return course


def _onward_exits(f,objects,tail,paths,v,options):
    from .native_wall_geometry import player_paths
    from .compact_spiking_sensation import point_and_laser_clearances
    parents=np.repeat(np.arange(len(paths)),len(options))
    turns=np.tile(options,len(paths))
    endings=paths[parents,-1]
    extended=player_paths(endings,v[turns],ticks=120,prefix=paths[parents],turn_tick=96)
    gap=point_and_laser_clearances(f,objects,endings,v[turns],duration=24,initial_time=96,
        native_forecasts=tail,anticipate_arming=True,growth_motion_envelope=True,
        unknown_laser_player_motion=True,max_laser_margin=64.,player_path=extended,linear_hint=True)
    if np.isnan(gap).any() or np.isneginf(gap).any() or tail.get('nonfinite_clearance_fallbacks'):
        return None
    gaps=np.minimum(gap,64.).reshape(len(paths),-1)
    best=gaps.argmax(1)
    return gaps[np.arange(len(paths)),best],options[best]


def _revalidate(encoding, tail, v, origin, course, *, endpoint_room=None, onward_exit=None,
                clearance_floor=1.):
    """Recheck every swept segment, including new bullets, from the actual position."""
    from .native_wall_geometry import player_paths
    from .compact_spiking_sensation import point_and_laser_clearances
    fc=encoding.get('source_native_forecasts') or {}
    path=np.asarray(origin,np.float32)[None,None]
    limit=len(course)*4
    gaps={}
    gap=64.
    begin=0
    while begin<limit:
        a=course[begin//4]
        end=begin+4
        while end<limit and end not in HORIZONS and course[end//4]==a:
            end+=4
        start=path[:,-1]
        path=player_paths(start,v[a:a+1],ticks=end,prefix=path,turn_tick=begin)
        scene=fc if end<=24 else tail
        value=point_and_laser_clearances(encoding['source_features25'],encoding['source_objects29'],
            start,v[a:a+1],duration=end-begin,initial_time=begin,native_forecasts=scene,
            anticipate_arming=True,growth_motion_envelope=True,unknown_laser_player_motion=True,
            max_laser_margin=64.,player_path=path,linear_hint=end>24)[0]
        if not value > clearance_floor or scene.get('nonfinite_clearance_fallbacks'):
            break
        if endpoint_room is not None and end==96:
            value=min(value,1.+float(_room_to_turn(path[0,-1])))
            endpoint_room.append(float(_room_to_turn(path[0,-1])))
        gap=min(gap,value)
        if end in (48, 96, limit):
            gaps[end] = gap
        begin=end
    if onward_exit is not None and 96 in gaps:
        options=np.sort(np.unique(v,axis=0,return_index=True)[1])
        extra=_onward_exits(encoding['source_features25'],encoding['source_objects29'],tail,path,v,options)
        if extra is not None:onward_exit.append((float(extra[0][0]),int(extra[1][0])))
    return gaps


def _rejoin_courses(origin, anchor, old, v, options, immediate):
    """Reach a future point of the previous suffix with one or two legal inputs."""
    viable=options[immediate[options]>1]
    if not len(viable) or np.linalg.norm(origin-anchor)>8*np.linalg.norm(v,axis=1).max()+1.5:
        return []
    position=np.asarray(origin)[None]
    target=np.asarray(anchor)
    keys=np.empty((1,0),int)
    results=[]
    for depth in (1,2):
        actions=viable if depth==1 else options
        position=np.clip(position[:,None]+4*v[actions],[-184,32],[184,432]).reshape(-1,2)
        keys=np.column_stack((np.repeat(keys,len(actions),axis=0),np.tile(actions,len(keys))))
        target=np.clip(target+4*v[old[depth-1]],[-184,32],[184,432])
        distance=np.linalg.norm(position-target,axis=-1)
        for index in np.argsort(distance,kind='stable'):
            if distance[index]>1.5:
                break
            key=keys[index]
            if any(candidate[0][0]==key[0] for candidate in results):
                continue
            candidate=old.copy();candidate[:depth]=key
            results.append((candidate,dict(frames=4*depth,endpoint_error=float(distance[index]))))
            if len(results)==2:
                return results
    return results


def local_turn_grid(reached, horizon):
    endpoints = tuple(tick for tick in LEGS if reached < tick <= horizon)
    midpoint = reached + (endpoints[0] - reached) // 8 * 4
    return (midpoint, *endpoints)


def _refine_missing_96(encoding,tail,origin,v,options,confidence,corner_reserve,result,*,early=False,
                       retained_prefix=None, horizon=96):
    """Recover a locally mistimed turn on the legal four-frame input grid."""
    cues=result['features']
    if tail.get('nonfinite_clearance_fallbacks') or (retained_prefix is None and np.any(cues[:,3]>0)):
        return
    prefixes=[]
    if retained_prefix is not None:
        prefixes.append((int(retained_prefix[0]), retained_prefix))
    elif early:
        # A late turn cannot recover a corridor that required changing course
        # at 8, 16 or 20 frames. Begin with each distinct legal displacement.
        for action in options[result['margins'][0,options]>1]:
            prefixes.append((int(action),np.array([action],dtype=np.int8)))
    else:
        candidates=np.flatnonzero((cues[:,2]>0) & (cues[:,3]<=0))
        for action in candidates[np.argsort(-cues[candidates,0],kind='stable')]:
            course=commands4(result,action)
            if course is None or len(course)<12:
                continue
            prefix=course[:12]
            if any(np.array_equal(v[prefix],v[other]) for _,other in prefixes):
                continue
            prefixes.append((int(action),prefix.copy()))
            if len(prefixes)==3:
                break
    if not prefixes:
        return
    from .native_wall_geometry import player_paths
    from .compact_spiking_sensation import point_and_laser_clearances
    count=len(prefixes)
    courses=np.full((count,24),-1,np.int8)
    paths=np.broadcast_to(np.asarray(origin,np.float32),(count,1,2)).copy()
    initial=len(prefixes[0][1])
    courses[:,:initial]=np.stack([prefix for _,prefix in prefixes])
    for step in range(initial):
        paths=player_paths(paths[:,-1],v[courses[:,step]],ticks=(step+1)*4,prefix=paths,turn_tick=step*4)
    if retained_prefix is not None:
        reached = initial * 4
        gaps = result['margins'][LEGS.index(reached), courses[:,0]].copy()
        ends = local_turn_grid(reached, horizon)
    elif early:
        gaps=result['margins'][0,courses[:,0]].copy()
        ends=(8,12,16,20,24,32,40,48,60,72,84,96)
    else:
        normalized=cues[courses[:,0],0].astype(float)
        gaps=8.*normalized/(1.-normalized)
        ends=(60,72,84,96)
    begin=initial*4
    margin48=None
    def publish(course,horizon,margin,short_gap=None,raw_margin=None):
        first=int(course[0])
        index=1 if horizon==96 else 0
        aliases = (first,) if retained_prefix is not None else np.flatnonzero(np.all(v==v[first],axis=1))
        for action in aliases:
            witness=course[:horizon//4].copy()
            witness[0]=action
            result.setdefault('refined_courses',{})[int(action)]=witness
            cues[action,index]=margin/(8.+margin)
            cues[action,index+2]=1.
            if short_gap is not None:
                cues[action,0]=short_gap/(8.+short_gap)
                cues[action,2]=1.
                result['margins'][3,action]=short_gap
            cues[action,4:6]=v[course[1]]/np.linalg.norm(v,axis=1).max()
            cues[action,6]=confidence
            result['margins'][5 if horizon==96 else 3,action]=margin if raw_margin is None else raw_margin
    for end in ends:
        parents=np.repeat(np.arange(len(paths)),len(options))
        turns=np.tile(options,len(paths))
        start=paths[parents,-1]
        full=player_paths(start,v[turns],ticks=end,prefix=paths[parents],turn_tick=begin)
        scene=(encoding.get('source_native_forecasts') or {}) if end<=24 else tail
        clearance=point_and_laser_clearances(encoding['source_features25'],encoding['source_objects29'],
            start,v[turns],duration=end-begin,initial_time=begin,
            native_forecasts=scene,
            anticipate_arming=True,growth_motion_envelope=True,unknown_laser_player_motion=True,
            max_laser_margin=64.,player_path=full,linear_hint=end>24)
        key='refined_early_segments' if early else 'refined_turn_segments'
        result[key]=result.get(key,0)+len(parents)
        if scene.get('nonfinite_clearance_fallbacks') or np.isnan(clearance).any() or np.isneginf(clearance).any():
            return
        gap=np.minimum(gaps[parents],np.minimum(clearance,64.))
        possible=np.flatnonzero(gap>1)
        if not len(possible):
            return
        rank=gap+.4*_reserve(full[:,-1])
        if corner_reserve:
            rank=np.minimum(gap,1.+_room_to_turn(full[:,-1]))+.4*_reserve(full[:,-1])
        ordered=possible[np.argsort(-rank[possible],kind='stable')]
        kept=[ordered[0]]
        if end<horizon:
            separated=ordered
            for _ in range(2 if early else 1):
                separated=separated[np.linalg.norm(full[separated,-1]-full[kept[-1],-1],axis=-1)>=8.]
                if not len(separated):
                    break
                kept.append(separated[0])
        kept=np.asarray(kept)
        courses=courses[parents[kept]].copy()
        courses[:,begin//4:end//4]=turns[kept,None]
        paths=full[kept]
        gaps=gap[kept]
        if margin48 is not None:
            margin48=margin48[parents[kept]]
        if end==48:
            margin48=gaps.copy()
            if retained_prefix is None and not np.any(cues[:,2]>0):
                publish(courses[0],48,float(gaps[0]))
        begin=end
    course=courses[0]
    margin=float(gaps[0])
    if corner_reserve:
        margin=min(margin,1.+float(_room_to_turn(paths[0,-1])))
    short_gap = float(margin48[0]) if horizon == 96 and margin48 is not None else None
    publish(course,horizon,margin,short_gap,raw_margin=float(gaps[0]))


def refine_local_turns(encoding, result, limit):
    """Extend a few existing safe prefixes at the midpoint of the missing interval."""
    from time import perf_counter
    from .nbnm_exit48 import _positions

    cues = result['features']
    margins = result['margins']
    measured = result.get('refined_courses', {})
    candidates = []
    for action in np.flatnonzero((cues[:,6] > 0) & (cues[:,3] <= 0) & (margins[0] > 1.)):
        if action in measured:
            continue
        prefix = commands4(result, int(action))
        reached = 0 if prefix is None else len(prefix) * 4
        if reached not in (12, 24, 48, 72):
            continue
        horizon = 48 if reached < 48 else 96
        if cues[action, 2 + int(horizon == 96)] > 0:
            continue
        gap = margins[LEGS.index(reached), action]
        if np.isfinite(gap) and gap > 1.:
            candidates.append((reached, float(gap), int(action), prefix, horizon))
    if not candidates:
        return

    started = perf_counter()
    tail, confidence = _tail_scene(encoding, LEGS[-1])
    covered = {horizon: native_window_covered(encoding, tail, horizon) for horizon in (48, 96)}
    origin, velocities, _ = _positions(encoding)
    _, indices = np.unique(velocities, axis=0, return_index=True)
    options = np.sort(indices)
    attempted, added = [], []
    for _, _, action, prefix, horizon in sorted(candidates, key=lambda row: row[:2], reverse=True):
        if not covered[horizon]:
            continue
        attempted.append(dict(action=action, through=horizon))
        _refine_missing_96(encoding, tail, origin, velocities, options, confidence, False, result,
                           retained_prefix=prefix, horizon=horizon)
        if cues[action, 2 + int(horizon == 96)] > 0:
            added.append(action)
        if len(attempted) == limit:
            break
    result['local_turn_refinement'] = dict(attempted=attempted, added=added,
                                           covered_horizons=[h for h, known in covered.items() if known],
                                           seconds=perf_counter() - started)


@share_native_projections()
@share_release_sweeps()
@share_waiting_aim()
def _issun_guidance(encoding, tail, origin, velocities, options, confidence, result, carry, query):
    from .nbnm_issun_corridor import MIN_GUIDED_MARGIN, add_lure_corridors, constrained_exits, guided_moves

    if not encoding['details']['trusted'][0]:
        return
    # Forecast diagnostics can accumulate flags. Keep them local to the lesson.
    lesson_encoding = dict(encoding, source_native_forecasts=(encoding.get('source_native_forecasts') or {}).copy())
    lesson_tail = tail.copy()
    if (carry and encoding.get('frame') == carry['frame'] and encoding.get('stage') == carry['stage']
            and np.linalg.norm(origin - carry['position']) < .15):
        course = carry['course']
        first = int(course[0])
        if result['margins'][0, first] >= MIN_GUIDED_MARGIN:
            gaps = _revalidate(lesson_encoding, lesson_tail, velocities, origin, course,
                               clearance_floor=MIN_GUIDED_MARGIN)
            if len(course) * 4 in gaps:
                moves = np.zeros(18, bool)
                moves[first] = True
                result.update(lure_guidance=moves, lure_guidance_courses={first: course},
                              lure_lesson=dict(remaining_frames=len(course) * 4))
                return
    if not query and not constrained_exits(result['features']):
        return
    if (lesson_tail.get('nonfinite_clearance_fallbacks')
            or lesson_encoding['source_native_forecasts'].get('nonfinite_clearance_fallbacks')):
        return
    lesson = dict(features=result['features'].copy(), margins=result['margins'].copy())
    add_lure_corridors(lesson_encoding, lesson_tail, origin, velocities, options, confidence, lesson,
                       proactive=True)
    moves = guided_moves(lesson)
    if moves.any():
        courses = {int(a): lesson['refined_courses'][a] for a in np.flatnonzero(moves)}
        result.update(lure_guidance=moves, lure_guidance_courses=courses,
                      lure_lesson=lesson['issun_lure'])


def plan(encoding, *, beam=1, carry=None, corner_reserve=False, onward=False, reactive_retry=False,
         refine_uncertain_turns=False,refine_early_turns=False,issun_lure=False,issun_lure_tail_turns=False,
         teach_lure=False, lure_carry=None, query_lure=True, verified_tail=None):
    from .native_wall_geometry import player_paths
    from .compact_spiking_sensation import point_and_laser_clearances
    from .thrl_movement_teacher import action_velocities
    d = encoding['details']
    features = np.zeros((18, WIDTH), np.float32)
    margins = np.full((len(LEGS), 18), np.nan, np.float32)
    commands = np.full((len(LEGS), 18, len(LEGS)), -1, np.int8)
    room96=np.full(18,np.nan)
    result = dict(features=features, margins=margins, commands=commands, legs=LEGS, segments=0)
    fc = encoding.get('source_native_forecasts') or {}
    if (not d['normal'] or not d.get('full_observed_hazards') or not
            (d['entities'] or fc.get('pending_line_births') or fc.get('pending_point_birth_bounds')
             or fc.get('pending_stationary_volleys') is not None
             or fc.get('pending_rotating_rings') is not None)):
        return result
    f = encoding['source_features25']
    objects = encoding['source_objects29']
    if fc.get('nonfinite_clearance_fallbacks'):
        return result
    # An emitter can aim down either continuation after the same first
    # move. Keep a second spatially distinct path while a line birth is due.
    projected = 'emitter_projection_horizon' in fc
    if fc.get('pending_line_births'):
        beam=2
    v = np.trunc(action_velocities() * 128) / 128
    if f[18] < 0:
        v[9:] = v[:9]
    # Keep a single search per physical first step (focus-idle is a duplicate).
    _, indices, inverse = np.unique(v, axis=0, return_index=True, return_inverse=True)
    options = np.sort(indices)
    canonical = indices[inverse]
    origin = np.asarray((fc.get('reactive_context') or {}).get('player',
                        [f[0] * 368 - 184, f[1] * 400 + 32]))
    immediate = np.asarray(d.get('unclipped_margins', d['margins']))[0]
    partial = np.asarray(d.get('partial_point_danger', np.zeros((3, 18), bool)))[0]
    contact_confirmed = bool(d['trusted'][0]) and not d.get('conditional_emitter_birth_4', False)
    blocked = (immediate <= 0) & (contact_confirmed | partial)
    if fc.get('pending_line_births'):
        from .native_pending_lines import line_clearances
        root_paths = player_paths(np.broadcast_to(origin, v.shape), v, ticks=4)
        line_gaps = line_clearances(
            fc['pending_line_births'], root_paths, 0, 4, fc['pending_line_half_extents'])
        if not projected or contact_confirmed:
            blocked |= line_gaps <= 0
        immediate = np.minimum(immediate, line_gaps)
    features[blocked, 2:4] = -1
    features[blocked, 6] = 1
    margins[0] = np.clip(immediate, -64, 64)
    roots = options[immediate[options] > 1]
    if not len(roots):
        return result
    if verified_tail is None:
        tail, confidence = _tail_scene(encoding, 120 if onward else LEGS[-1])
        legs = LEGS
        if projected and confidence < .5:
            if onward:
                onward = False
                tail, confidence = _tail_scene(encoding, LEGS[-1])
            if confidence < .5:
                tail, confidence = _tail_scene(encoding, 48)
                legs = LEGS[:4] if confidence >= .5 else LEGS[:3]
    else:
        tail, confidence = verified_tail, .75
        legs = LEGS[:4]
    origin_batch = np.broadcast_to(origin, (len(roots), 2))
    paths = player_paths(origin_batch, v[roots], ticks=4)
    gaps = np.clip(immediate[roots], 0, 64)
    history_gaps=np.full((len(roots),len(LEGS)),np.nan,np.float32)
    history_gaps[:,0]=gaps
    seq = np.full((len(roots), len(LEGS)), -1, np.int8)
    seq[:, 0] = roots
    commands[0, roots] = seq
    starts = roots.copy()
    if onward:
        onward_margins=np.full(18,np.nan)
        onward_commands=np.full(18,-1,np.int8)
        onward_evaluated=np.zeros(18,bool)
    movement_scale = np.linalg.norm(v, axis=-1).max()
    for i, (begin, end) in enumerate(zip(legs, legs[1:]), 1):
        if not len(starts):
            break
        parents = np.repeat(np.arange(len(starts)), len(options))
        actions = np.tile(options, len(starts))
        pos = paths[parents, -1]
        speed = v[actions]
        fc_i = fc if end <= 24 else tail
        # Candidate-dependent programs must see the entire turning trajectory.
        conditional = bool(fc_i.get('reactive_bullets') or any(
            r.get('candidate_dependent') for r in fc_i.get('bullets', ())))
        if conditional:
            # Rigid geometry proposes two branches per parent. Both are then
            # checked with their OWN original aim/orb rules before acceptance.
            proposal = point_and_laser_clearances(f, objects, pos, speed, duration=end-begin,
                initial_time=begin, unknown_laser_player_motion=True, linear_hint=True, max_laser_margin=64.)
            ends = np.clip(pos + (end-begin)*speed, [-184,32], [184,432])
            rank = np.minimum(proposal, 24) + .75*_reserve(ends)
            if corner_reserve:
                rank=np.minimum(np.minimum(proposal,24),1.+_room_to_turn(ends))+.75*_reserve(ends)
            proposal_order = np.argsort(-rank.reshape(len(starts), -1), kind='stable')
            ids = proposal_order[:, :max(2, beam)].ravel()
            offsets = np.repeat(np.arange(len(starts))*len(options), max(2, beam))
            chosen = ids + offsets
            parents, actions, pos, speed = (a[chosen] for a in (parents, actions, pos, speed))
        full = player_paths(pos, speed, ticks=end, prefix=paths[parents], turn_tick=begin)
        gap = point_and_laser_clearances(f, objects, pos, speed, duration=end-begin,
            initial_time=begin, native_forecasts=fc_i, anticipate_arming=True,
            growth_motion_envelope=True, unknown_laser_player_motion=True,
            max_laser_margin=64., player_path=full, linear_hint=end>24)
        if conditional and reactive_retry and not fc_i.get('nonfinite_clearance_fallbacks') and not np.isnan(gap).any():
            # A rigid approximation can rank the correct move poorly, e.g. an
            # orb stops only if THIS branch approaches it. Recover a lost first
            # move before calling its exits unknown. Most rows need no retry.
            lost_first = ~np.isin(starts, starts[parents[gap > 1]])
            remaining = proposal_order[lost_first, max(2, beam):]
            if remaining.size:
                more_parents = np.repeat(np.flatnonzero(lost_first), remaining.shape[1])
                more_actions = options[remaining.ravel()]
                more_pos = paths[more_parents, -1]
                more_speed = v[more_actions]
                more_paths = player_paths(more_pos, more_speed, ticks=end,
                    prefix=paths[more_parents], turn_tick=begin)
                more_gap = point_and_laser_clearances(f, objects, more_pos, more_speed,
                    duration=end-begin, initial_time=begin, native_forecasts=fc_i,
                    anticipate_arming=True, growth_motion_envelope=True, unknown_laser_player_motion=True,
                    max_laser_margin=64., player_path=more_paths, linear_hint=end>24)
                parents = np.concatenate((parents, more_parents))
                actions = np.concatenate((actions, more_actions))
                full = np.concatenate((full, more_paths))
                gap = np.concatenate((gap, more_gap))
                result['reactive_retry_segments'] = result.get('reactive_retry_segments', 0) + len(more_gap)
        if fc_i.get('nonfinite_clearance_fallbacks') or np.isnan(gap).any():
            features[:] = 0
            commands[:] = -1
            margins[:] = np.nan
            return result
        result['segments'] += len(gap)
        bottleneck = np.minimum(np.minimum(gap, 64), gaps[parents])
        # Wall reserve only breaks close choices. Never buy it with a predicted collision.
        rank = bottleneck + .4 * _reserve(full[:, -1]) + .04 * (actions == seq[parents, i-1])
        if corner_reserve:
            rank=np.minimum(bottleneck,1.+_room_to_turn(full[:,-1]))+.4*_reserve(full[:,-1])+.04*(actions==seq[parents,i-1])
        root = starts[parents]
        if onward and end==96:
            extensions={}
            # Two alternatives per first move are enough for a bounded onward
            # test. Preserve spatially different endpoints where available.
            finalists=[]
            for first in np.unique(root):
                ids=np.flatnonzero((root==first)&(bottleneck>1))
                if not len(ids):continue
                order=ids[np.argsort(-rank[ids],kind='stable')]
                finalists.append(order[0])
                separated=order[np.linalg.norm(full[order,-1]-full[order[0],-1],axis=-1)>=8]
                if len(separated):finalists.append(separated[0])
            if finalists:
                extra=_onward_exits(f,objects,tail,full[finalists],v,options)
                if extra is not None:
                    extensions={parent:(gap,turn) for parent,gap,turn in zip(finalists,*extra)}
                    for parent,(gap,_) in extensions.items():
                        if gap>1:
                            rank[parent]+=65.+min(gap,bottleneck[parent],24.)
        keep = []
        for a in np.unique(root):
            ids = np.flatnonzero(root == a)
            viable = ids[bottleneck[ids] > 1]
            by_score = viable[np.argsort(-rank[viable], kind='stable')]
            if not len(by_score):
                # The finite search ran out of witnesses. This is censored,
                # never a native collision target or proof of impossibility.
                continue
            selected = by_score[0]
            witness=seq[parents[selected]].copy()
            witness[i]=actions[selected]
            # Shorter horizons describe prefixes of the same longest witness
            # even when a second beam wins on a later turn.
            for k in range(i+1):
                commands[k,a,:k+1]=witness[:k+1]
            margins[:i,a]=history_gaps[parents[selected],:i]
            margins[i,a]=bottleneck[selected]
            if corner_reserve and end==96:
                room96[a]=_room_to_turn(full[selected,-1])
            if onward and end==96 and selected in extensions:
                onward_margins[a],onward_commands[a]=extensions[selected]
                onward_evaluated[a]=True
            keep.append(selected)
            if beam > 1:
                distinct = by_score[np.linalg.norm(full[by_score,-1]-full[selected,-1],axis=-1) >= 8]
                if len(distinct):
                    keep.append(distinct[0])
        keep = np.asarray(keep, int)
        paths = full[keep]
        seq = seq[parents[keep]].copy()
        seq[:, i] = actions[keep]
        history_gaps=history_gaps[parents[keep]].copy()
        history_gaps[:,i]=bottleneck[keep]
        starts, gaps = root[keep], bottleneck[keep]
    for a in range(18):
        root = canonical[a]
        for i in range(len(LEGS)):
            if commands[i, root, 0] >= 0:
                commands[i, a] = commands[i, root]
                commands[i, a, 0] = a
                margins[i, a] = margins[i, root]
        if root not in roots:
            continue
        features[a, 6] = confidence
        for j, i in enumerate((3, 5)):
            if projected and LEGS[i] > legs[-1]:
                continue  # Uncovered rather than a failed search.
            gap = margins[i, a]
            if np.isfinite(gap):
                features[a, j] = gap / (8 + gap)
                features[a, 2+j] = 1
            else:
                features[a, 2+j] = -.5
        for i in (5, 3, 2, 1):
            if commands[i, a, 1] >= 0:
                features[a, 4:6] = v[commands[i, a, 1]] / movement_scale
                break
    if (carry and (not projected or legs[-1] >= 48) and
            encoding.get('frame')==carry['frame'] and encoding.get('stage')==carry['stage']):
        limit = legs[-1] // 4 if projected else 24
        for anchor in (carry,carry.get('alternate')):
            if anchor is None or len(result.get('retained',()))==limit:
                break
            old=anchor['course'][:limit]
            course=np.pad(old,(0,limit-len(old)),mode='edge')
            if np.linalg.norm(origin-anchor['position'])<.15:
                proposals=[(course,None)]
            else:
                proposals=_rejoin_courses(origin,anchor['position'],course,v,options,immediate)
            for course,rejoin in proposals:
                a=int(course[0])
                if immediate[a]<=1:
                    continue
                # Check the bridge and the complete remaining suffix against
                # current objects. Proximity to an old waypoint has no safety credit.
                room=[] if corner_reserve else None
                exit_tail=[] if onward else None
                verified=(_revalidate(encoding,tail,v,origin,course,endpoint_room=room,onward_exit=exit_tail) if (corner_reserve or onward)
                          else _revalidate(encoding,tail,v,origin,course))
                tail_turn=None
                carried_lure=bool(issun_lure and anchor.get('issun_lure') and rejoin is None)
                if issun_lure_tail_turns and carried_lure and 48 in verified and 96 not in verified:
                    from .nbnm_issun_corridor import extend_lure_course
                    extended=extend_lure_course(encoding,tail,origin,v,options,old)
                    if extended is not None:
                        new_room=[] if corner_reserve else None
                        new_exit=[] if onward else None
                        checked=_revalidate(encoding,tail,v,origin,extended,endpoint_room=new_room,onward_exit=new_exit)
                        if 96 in checked:
                            tail_turn=dict(first_action=int(extended[0]),previous_tail=int(course[-1]),
                                           new_tail=int(extended[-1]),minimum_clearance=float(checked[96]))
                            course,verified,room,exit_tail=extended,checked,new_room,new_exit
                if not verified:
                    continue
                horizon=max(verified)
                previous=result.get('retained')
                if previous is not None and len(previous)*4>=horizon:
                    continue
                new_margin=margins[5 if horizon==96 else 3,a]
                longer_new=horizon<96 and features[a,3]>0
                narrower=np.isfinite(new_margin) and verified[horizon]<min(4.,.5*new_margin)
                if longer_new or narrower:
                    continue
                if onward and horizon==96 and onward_margins[canonical[a]]>1 and exit_tail and exit_tail[0][0]<=1:
                    continue
                if previous is not None:
                    old_action=int(previous[0])
                    features[old_action]=result.pop('_before_retained')
                    result.pop('rejoined',None)
                result['_before_retained']=features[a].copy()
                for h,gap in verified.items():
                    j=0 if h==48 else 1
                    features[a,j]=gap/(8+gap)
                    features[a,j+2]=.75
                features[a,4:6]=v[course[1]]/movement_scale
                result['retained']=course[:horizon//4].copy()
                if carried_lure:result['retained_lure']=True
                else:result.pop('retained_lure',None)
                if tail_turn is not None:result['issun_lure_tail_turn']=tail_turn
                else:result.pop('issun_lure_tail_turn',None)
                if corner_reserve and horizon==96:
                    room96[canonical[a]]=room[-1]
                if onward and horizon==96:
                    onward_evaluated[canonical[a]]=bool(exit_tail)
                    if exit_tail:onward_margins[canonical[a]],onward_commands[canonical[a]]=exit_tail[0]
                if rejoin is not None:
                    result['rejoined']=rejoin
                break
            if tail.get('nonfinite_clearance_fallbacks') or fc.get('nonfinite_clearance_fallbacks'):
                features[:]=0
                result.pop('retained',None)
                result.pop('retained_lure',None)
                result.pop('issun_lure_tail_turn',None)
                result.pop('rejoined',None)
                break
    result.pop('_before_retained',None)
    if corner_reserve:
        # Retain evidence of every viable direction, including a necessary
        # corner route. When a wider onward exit exists, distinguish it on
        # the existing margin ports. Never turn an unknown route into a witness.
        room=room96[canonical]
        support=(features[:,3]>0)&np.isfinite(room)
        if support.any():
            for a in np.flatnonzero(support):
                reserve=1.+room[a]
                features[a,1]=min(features[a,1],reserve/(8.+reserve))
            if np.any(support & (room>=24)):
                features[support & (room<12),3]*=.25
    if onward:
        measured=onward_evaluated[canonical] & (features[:,3]>0)
        onward_gap=onward_margins[canonical]
        for a in np.flatnonzero(measured):
            reserve=max(1.,float(onward_gap[a]))
            features[a,1]=min(features[a,1],reserve/(8.+reserve))
        if np.any(measured & (onward_gap>1)):
            features[measured & (onward_gap<=1),3]*=.25
        result['onward_margin']=np.where(measured,onward_gap,np.nan)
        result['onward_command']=np.where(measured & (onward_gap>1),onward_commands[canonical],-1)
    long_covered = not projected or legs[-1] == LEGS[-1]
    if refine_uncertain_turns and long_covered:
        _refine_missing_96(encoding,tail,origin,v,options,confidence,corner_reserve,result)
        if refine_early_turns and not np.any(features[:,3]>0):
            _refine_missing_96(encoding,tail,origin,v,options,confidence,corner_reserve,result,early=True)
    if long_covered and not tail.get('nonfinite_clearance_fallbacks') and not fc.get('nonfinite_clearance_fallbacks'):
        if issun_lure:
            from .nbnm_issun_corridor import add_lure_corridors, constrained_exits
            if issun_lure != 'constrained' or constrained_exits(features):
                add_lure_corridors(encoding,tail,origin,v,options,confidence,result)
        if teach_lure:
            _issun_guidance(encoding, tail, origin, v, options, confidence, result, lure_carry, query_lure)
    return result

