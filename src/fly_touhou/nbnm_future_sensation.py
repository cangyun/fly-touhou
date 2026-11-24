"""Named future-rule inputs, kept separate from observed geometry and labels.

These are conditional forecasts: enemy survival and incomplete native callbacks
remain explicit. No forecast directly replaces the selected neural action.
"""
import math
import struct
import time
import numpy as np

WIDTH=8
NAMES=('future_margin4','future_margin12','future_margin24','future_margin60',
       'emission_proximity','earliest_emission','motion_coverage','rule_coverage')
SCHEMA='connectome-nbnm-route425-future-v1'


def _capsule_distance(point,first,last):
    d=last-first;den=(d*d).sum(-1)
    fraction=np.divide(((point-first)*d).sum(-1),den,out=np.zeros_like(den),where=den>1e-12)
    return np.linalg.norm(point-first-np.clip(fraction,0,1)[...,None]*d,axis=-1)


def _accelerating_birth_clearance(event, path, radius, begin, end):
    """Enclose supported unit-time EX2 followed by EX4 from the muzzle."""
    from .native_bullet_release import predict_bullet_release
    from .native_enemy_ecl import f32

    shooter = event['shooter']
    program = shooter['ex']
    if len(program) < 2 or not all(program[:2]) or any(program[2:]):
        return None
    animation, acceleration = program[:2]
    duration = acceleration['ints'][0]
    magnitude, heading = acceleration['floats'][:2]
    if (animation['command'] != 2 or animation['condition'] or animation['ints'][0] != 1
            or acceleration['command'] != 4 or acceleration['condition']
            or not event.get('unit_point_motion') or shooter['radius'] != 0.
            or duration < end - begin or end - begin > 120
            or not all(math.isfinite(v) for v in (magnitude, heading))
            or magnitude <= -990. or abs(heading) > 2*math.pi
            or acceleration['ints'][1] > -990 or acceleration['floats'][2] > -990.):
        return None
    delta = [f32(magnitude * math.cos(heading)), f32(magnitude * math.sin(heading))]
    age = np.arange(1, end - begin + 1, dtype=float)
    gaps = np.full(len(age), np.inf)
    for _, speed, vx, vy in event['vectors']:
        velocity = [vx, vy]
        if not all(math.isfinite(v) for v in (speed, vx, vy, *event['position'])):
            return None
        origin = [f32(p - 4 * v) for p, v in zip(event['position'], velocity)]
        raw = dict(
            native_position=origin, native_velocity=velocity,
            native_state=2, native_flags=0x12, native_ex_active_flags=4,
            native_ex_index=2, native_ex_current_record_hex='00' * 44,
            native_animation_gate=0, native_motion_time_rate=1.,
            native_ex_acceleration_tick=0, native_ex_acceleration_duration=duration,
            native_ex_acceleration_time=0., native_ex_acceleration_source=4,
            native_ex_acceleration=delta,
        )
        motion = predict_bullet_release(raw, len(age))
        if not all(motion['known_by_frame']):
            return None
        bounds = np.asarray(motion['bounds'][1:])
        width = np.asarray(motion['extra_radii'][1:])
        padding = 16 * np.finfo(np.float32).eps * (age + 1) * (
            np.abs(bounds).max(axis=(1, 2)) + abs(speed) + age * abs(magnitude) + 2)
        clearance = _capsule_distance(path[begin:end], bounds[:, 0], bounds[:, 1]) - width - padding
        gaps = np.minimum(gaps, clearance - radius)
    return gaps


def emission_clearances(events,path,radii,player_radius,horizon, *, resolved_rng=False,
                        player_half_extents=None, circular_point_styles=(), clearance_floor=None):
    """A conservative new-bullet prefix; unsupported EX/laser tails are unknown.

    EX2 sets state2 and subtracts four velocities at creation (4181f1..418275).
    During its animation it advances at half speed, then full speed. The capsule
    encloses every possible release time. This is a possible-threat envelope,
    not an exact collision or a negative-safe training label. Callers that
    have resolved all gameplay draws for this path may supply resolved_rng.
    """
    gaps=np.full(horizon+1,np.inf);supported=0;total=0;unknown=[];next_emission=horizon+1
    clears=sorted(e['tick'] for e in events if e['kind']=='clear_projectiles')
    for event in events:
        tick=event['tick'];kind=event['kind']
        if tick>horizon or kind in ('enemy','clear_projectiles'):continue
        next_emission=min(next_emission,tick)
        s=event['shooter'];total+=1
        end=min([horizon+1]+[c for c in clears if c>tick])
        commands=[x for x in s['ex'] if x is not None]
        if kind=='line_laser' and not commands:
            # 428b79/43af48 initialize this state;43be20 evolves it. The
            # full-width capsule deliberately encloses native end trimming
            # and narrower collision width. Not a measured collision label.
            from .native_line_laser_paths import predict_native_line
            angle=s['angle'];axis=np.array([math.cos(angle),math.sin(angle)])
            initial,maximum,limit,width=s['laser']
            origin=np.asarray(event['position'])+axis*s['radius']
            line=predict_native_line(dict(id=0,origin=origin.tolist(),velocity=(axis*s['speed']).tolist(),
                angle=angle,length=initial,max_length=maximum,travel_limit=limit,width=width,
                speed=s['speed'],travelled=.01 if initial>maximum else 0.),horizon=horizon-tick+1)
            # Each native tick retains both positions around its update.
            count = end - tick
            if (player_half_extents is not None and event.get('unit_line_motion')
                    and (resolved_rng or not event.get('rng_conditional', True))):
                from .native_line_laser_paths import native_line_contact_clearances
                margin = native_line_contact_clearances(
                    line['origin'][1:count + 1], line['length'][1:count + 1], width, angle,
                    path[tick:end, None, :], player_half_extents)[:, 0]
                valid = np.asarray(line['active'][1:count + 1])
                gaps[tick:end] = np.minimum(gaps[tick:end], np.where(valid, margin, np.inf))
                supported+=1
                continue
            starts = np.asarray(line['origin'][:count + 1])
            ends = np.asarray(line['endpoint'][:count + 1])
            active = np.asarray(line['active'][:count + 1])
            players = path[tick:end]
            before = _capsule_distance(players, starts[:-1], ends[:-1])
            after = _capsule_distance(players, starts[1:], ends[1:])
            gaps_to_line = np.minimum(before, after) - width * .5 - player_radius - .1
            gaps[tick:end] = np.minimum(gaps[tick:end],
                np.where(active[:-1] | active[1:], gaps_to_line, np.inf))
            supported+=1;continue
        if kind!='bullet':
            unknown.append(dict(tick=tick,kind=kind,reason='new_laser_lifecycle_not_validated'))
            continue
        unsupported=[x['command'] for x in commands if x['command'] not in (2,2048,1048576,33554432,4194304)]
        if unsupported:
            radius = radii.get(s['style'])
            if unsupported == [4] and radius is not None:
                modeled = _accelerating_birth_clearance(event, path, radius + player_radius + .1, tick, end)
                if modeled is not None:
                    gaps[tick:end] = np.minimum(gaps[tick:end], modeled)
                    supported += 1
                    continue
            unknown.append(dict(tick=tick,kind=kind,commands=unsupported,reason='new_EX_dynamics_not_validated'))
            continue
        radius=radii.get(s['style'])
        if radius is None:
            unknown.append(dict(tick=tick,kind=kind,reason='missing_native_bullet_radius'));continue
        # EX400000 writes both interpolation endpoints, not just a target.
        # For bounded non-overshooting modes the largest endpoint encloses
        # every possible within-tick interpolation order, including state2.
        scale=1.
        invalid_scale=False
        for command in commands:
            if command['command']!=4194304:continue
            first,last=command['floats'][:2];mode=command['ints'][1]
            if mode not in (*range(7),*range(9,17)) or min(first,last)<0:
                invalid_scale=True;break
            scale=max(scale,first,last)
        if invalid_scale:
            unknown.append(dict(tick=tick,kind=kind,reason='unbounded_or_special_scale_mode'));continue
        radius*=scale
        vectors=np.asarray(event['vectors'],float)
        if not len(vectors):supported+=1;continue
        v=vectors[:,2:];axis=np.column_stack((np.cos(vectors[:,0]),np.sin(vectors[:,0])))
        origin=np.asarray(event['position'])[None]+axis*s['radius']
        growing=any(x['command']==2 for x in commands)
        release = event.get('fresh_ex2_release_age')
        if release and s['radius'] == 0. and (resolved_rng or not event.get('rng_conditional', True)):
            from .native_bullet_release import fresh_ex2_centers
            centers = fresh_ex2_centers(event['position'],v,end-tick,release_age=release)
            if s['style'] in circular_point_styles:
                # From the supported native circular hit test, at each game tick.
                margin = np.linalg.norm(path[tick:end,None,:]-centers,axis=-1).min(axis=1)
                extent = math.hypot(radius,player_radius)
            else:
                initial = np.asarray(event['position'],dtype=np.float32) - 4. * v.astype(np.float32)
                before = np.concatenate((initial[None],centers[:-1]))
                margin = _capsule_distance(path[tick:end,None,:],before,centers).min(axis=1)
                extent = radius+player_radius
            gaps[tick:end] = np.minimum(gaps[tick:end], margin-extent-.1)
            supported+=1
            continue
        if growing:origin=origin-4*v
        for frame in range(tick,end):
            # Include both native within-tick movement order possibilities.
            age=frame-tick
            first=origin+v*(.5*age if growing else age)
            last=origin+v*(age+1)
            gap=_capsule_distance(path[frame],first,last).min()-radius-player_radius-.1
            gaps[frame]=min(gaps[frame],float(gap))
        supported+=1
    margins=[float(np.min(gaps[:min(h,horizon)+1])) for h in (4,12,24,60)]
    horizon_margin = float(gaps.min())
    result = dict(margins=margins, horizon_margin=horizon_margin if math.isfinite(horizon_margin) else None,
                  motion_coverage=supported/total if total else 1.,
                  events=total,earliest=next_emission,unknown=unknown,
                  labels_are_exact=False,envelope='possible_release_times_scale_extrema_and_full_laser_width')
    if clearance_floor is not None:
        low_margin = np.flatnonzero(gaps <= clearance_floor)
        result['first_low_margin_tick'] = int(low_margin[0]) if len(low_margin) else None
        result['frame_margins'] = gaps
    return result


class FutureRuleSensor:
    def __init__(self,config):
        from .nbnm_rule_catalog import SCHEMA as CATALOG_SCHEMA
        import json
        from pathlib import Path
        self.config=dict(config);self.reader=None;self.controller=None;self.radii={}
        self.preloaded={};self.preload_status=None;self.preload_stage=None
        # Load the complete rule set once per worker, not trajectory samples.
        self.catalog=json.loads(Path(config['catalog']).read_text(encoding='utf-8'))
        if self.catalog.get('schema')!=CATALOG_SCHEMA:raise ValueError('Original rule catalogue required')
        self.horizon=int(config.get('horizon',60))
        if not 24<=self.horizon<=240:raise ValueError('Use24..240 original-frame forecast horizons')

    def reset(self):
        self.preloaded={};self.preload_status=None;self.preload_stage=None

    def bind(self,reader):
        self.reader=reader
        self.preloaded={};self.preload_stage=None;self.preload_status=None
        from .nbnm_rule_inventory import build_inventory
        styles=build_inventory(self.catalog)['all_literal_styles']
        self.radii={s:struct.unpack('<f',reader.mem.read(reader.mem.base+0xd1e90+s*0x114+0x104,4))[0] for s in styles}

    def query(self,snapshot,*,source_features=None):
        from .native_shot_rules import capture_enemy_states,forecast_emissions,preload_programs
        from .thrl_movement_teacher import MovementSafetyTeacher
        if self.controller is not None and self.controller.reader is not self.reader:
            self.bind(self.controller.reader)
        if self.reader is None:raise RuntimeError('Bind future rules to the paused original game reader')
        start=time.perf_counter();v=MovementSafetyTeacher().velocities.copy()
        player_radius=snapshot.player.radius
        if source_features is not None:
            f=np.asarray(source_features,float)
            if f.shape!=(25,):raise ValueError('Original observed player geometry requires25 source features')
            if f[18]<0:v[9:]=v[:9] # Original game temporarily ignores Shift.
            player_radius=max(player_radius,abs(f[13])*64+np.linalg.norm(f[16:18]*64),np.linalg.norm(f[14:16]*64))
        paths=np.clip(np.array([snapshot.player.x,snapshot.player.y])[None,None,:]+
            v[:,None,:]*np.arange(self.horizon+1)[None,:,None],[-184,32],[184,432])
        states=capture_enemy_states(self.reader,snapshot);rng=snapshot.extra['native_rng_state']
        # Table addresses can change at a stage load or a new native run.
        manager=self.reader.pointer('enemies','enemy_manager_pointer')
        key=(snapshot.stage,manager)
        if states and self.preload_stage!=key:
            self.preloaded,self.preload_status=preload_programs(self.reader.mem,states,self.catalog,snapshot.stage)
            self.preload_stage=key
        forecasts=forecast_emissions(self.reader.mem,states,paths,seed=rng['gameplay_seed'],
            counter=rng['gameplay_counter'],horizon=self.horizon,preloaded_instructions=self.preloaded)
        x=np.zeros((18,WIDTH),np.float32);rows=[]
        half_extents = (snapshot.extra.get('native_line_lasers') or {}).get('player_half_extents')
        for index,(path,forecast) in enumerate(zip(paths,forecasts)):
            row=emission_clearances(forecast['events'],path,self.radii,player_radius,self.horizon,
                                    player_half_extents=half_extents)
            margins=np.asarray(row['margins'])
            if np.isnan(margins).any() or np.isneginf(margins).any():
                raise ValueError('Non-finite future-rule margin; refusing a silent zero replacement')
            finite=np.where(np.isposinf(margins),1e6,margins)
            x[index,:4]=finite/(32+abs(finite))
            # An empty supported prefix with unknown emissions is not an
            # all-safe forecast. Keep absent positive evidence neutral.
            if row['unknown'] or forecast['unknown']:
                x[index,:4]=np.minimum(x[index,:4],0.)
            x[index,4]=row['events']/(row['events']+8)
            x[index,5]=min(row['earliest'],self.horizon+1)/(self.horizon+1)
            x[index,6]=row['motion_coverage'];x[index,7]=float(forecast['rule_complete'])
            rows.append(dict(row,margins=[float(v) if math.isfinite(v) else None for v in margins],
                no_supported_threat_in_horizon=np.isposinf(margins).tolist(),rule_unknown=forecast['unknown'],
                rng_schedule_complete=forecast['global_rng_schedule_complete']))
        return x,dict(revision=SCHEMA,horizon=self.horizon,candidates=rows,
            seconds=time.perf_counter()-start,active_enemies=len(states),
            full_future_information=False,teacher_routes_used=False,
            all_rules_preloaded=True,preload=self.preload_status,exact_collision_labels=False)


def guide_future(probabilities,safe,detail,*,weight=.25):
    """Soft preference among already viable actions; no invented safe labels."""
    p=np.asarray(probabilities,float).copy()
    if detail is None:return p
    rows=detail['candidates'];score=np.zeros(18)
    for i,r in enumerate(rows):
        if r['events'] and r['motion_coverage']>0:
            margin=np.asarray([np.inf if x is None else x for x in r['margins']],float)
            score[i]=r['motion_coverage']*np.mean(np.minimum(0.,np.tanh((np.clip(margin,-256,256)-1)/16)))
    if not np.any(safe) or np.ptp(score[safe])<.05:return p
    q=np.where(safe,p*np.exp(weight*(score-score.max())),0)
    return q/q.sum() if q.sum()>0 else p
