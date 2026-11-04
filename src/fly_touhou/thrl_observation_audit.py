"""Pure, passive audit of current separated observations; never changes inputs.

Counts measure field samples, not distinct bullets across time. Normalization
references are not asserted physical maxima. Events missing from a passive
reader remain missing, never synthetic zero. No process/Torch/GPU operations.
"""
from __future__ import annotations

import math
from collections import Counter
import numpy as np

from .thrl_geometry_observation import (GLOBAL_FIELDS,OBJECT_FIELDS,
    shared_object_candidates,select_object_candidates,select_nonitem_fill_candidates)
from .thrl_inputs import (SEPARATED_SCHEMAS,SEPARATED_TIMERS_INPUT,
    SEPARATED_CONTACT_INPUT,SEPARATED_LASER_INPUT,PRIORITY_INPUT_SCHEMAS,SEPARATED_TIMER_SCHEMAS)
from .thrl_timer_observation import STATE_FIELDS

SCHEMA='th14-passive-observation-audit-v1'


def _finite(value):
    return isinstance(value,(int,float,np.number)) and not isinstance(value,(bool,np.bool_)) and math.isfinite(value)


def _field(raw,scale=1.,*,offset=0.,lower=0.,upper=1.,normalized=None):
    known=_finite(raw)
    value=(float(normalized) if normalized is not None else (float(raw)+offset)/scale) if known else None
    return dict(raw=float(raw) if known else None,normalization_divisor=scale,
        normalization_offset=offset,normalized_before_clip=value,bounds=[lower,upper],
        below=bool(value is not None and value<lower),above=bool(value is not None and value>upper),
        boundary=bool(value is not None and value in (lower,upper)),missing=not known)


def _columns(rows,names,lower=-1.,upper=1.):
    values=np.asarray(rows,dtype=float).reshape(-1,len(names))
    return {name:dict(samples=len(values),below=int(np.count_nonzero(values[:,i]<lower)),
        above=int(np.count_nonzero(values[:,i]>upper)),
        boundary=int(np.count_nonzero((values[:,i]==lower)|(values[:,i]==upper))),
        normalized_min=float(values[:,i].min()) if len(values) else None,
        normalized_max=float(values[:,i].max()) if len(values) else None,
        bounds=[lower,upper]) for i,name in enumerate(names)}


def _globals(snapshot,player,input_schema):
    p=snapshot.player;events=snapshot.extra.get('thrl_events',{})
    observed=events.get('schema')=='th14-thrl-events-v1' and events.get('observed') is True
    def count(name):
        value=events.get(name)
        return value if observed and type(value) is int and value>=0 else None
    values=[_field(p.x,368,offset=184),_field(p.y,400,offset=-32),
        _field(snapshot.power,400),_field(snapshot.lives,8),_field(snapshot.bombs,8),
        _field(int(p.invulnerable_frames>0)),_field(snapshot.character,2),
        _field(snapshot.stage,6),_field(snapshot.difficulty,3),_field(snapshot.graze,1000),
        _field(snapshot.score,100000000),_field(count('misses'),10),_field(count('point_items'),100),
        _field(player['circle_component'],64,lower=-1),
        *(_field(v,64,lower=-1) for v in player['bounds_half_extents']),
        *(_field(v,64,lower=-1) for v in player['bounds_offset']),
        _field(int(p.focused)),_field(p.deathbomb_timer,60,lower=-1)]
    result=dict(zip(GLOBAL_FIELDS,values))
    if input_schema in SEPARATED_TIMER_SCHEMAS:
        result.pop('deathbomb_timer')
        result['deathbomb_remaining_8frames']=_field(p.deathbomb_timer,8)
        frames=max(float(p.invulnerable_frames),0.)
        result['invulnerability_remaining_bounded']=_field(p.invulnerable_frames,
            normalized=frames/(frames+60))
        result['invulnerability_remaining_bounded']['transform']='max(t,0)/(max(t,0)+60); no hard upper clip'
        for state,name in zip((1,2,3,4),STATE_FIELDS):result[name]=_field(int(p.state==state))
    return result,observed


def _danger(row,player,horizon,margin):
    token=row[2]
    if not token[13]:return False
    if len(token)==29 and (token[3]>.5 or token[4]>.5):
        from .thrl_laser_observation import unbounded
        center=unbounded(token[6:8],[384,448]);half=unbounded(token[21:23],[384,448])
        v=unbounded(token[8:10],12);dv=unbounded(token[23:25],12)
        width=unbounded(token[10],128);rate=unbounded(token[25],12)
        times=np.r_[np.arange(horizon),horizon] if token[26]>.5 else np.asarray([0.])
        aa=center-half+(v-dv)*times[:,None];bb=center+half+(v+dv)*times[:,None];d=bb-aa
        offset=player['bounds_offset'];fraction=np.clip(((offset-aa)*d).sum(-1)/np.maximum((d*d).sum(-1),1e-12),0,1)
        radius=max(player['circle_component']+np.linalg.norm(offset),np.linalg.norm(player['bounds_half_extents']))+.05
        gap=np.linalg.norm(aa+fraction[:,None]*d-offset,axis=-1)-np.maximum(0,width+rate*times)-radius
        return bool(gap.min()<=margin)
    dx,dy,vx,vy=token[6]*384,token[7]*448,token[8]*12,token[9]*12
    hx,hy=token[10]*128,token[11]*128
    if token[12]:
        speed=vx*vx+vy*vy
        t=float(np.clip(-(dx*vx+dy*vy)/speed,0,horizon)) if speed>1e-12 else 0.
        clearance=math.hypot(dx+vx*t,dy+vy*t)-math.hypot(hx,player['circle_component'])
    else:
        from .native_geometry import swept_rectangle_clearance
        clearance=swept_rectangle_clearance(np.asarray((dx,dy))-player['bounds_offset'],
            (vx,vy),np.asarray((hx,hy))+player['bounds_half_extents'],0,horizon)
    return bool(clearance<=margin)


def audit_snapshot(snapshot,*,input_schema=SEPARATED_TIMERS_INPUT,danger_distance=32.,horizon_frames=60.,laser_motion=None):
    """Audit an already measured same-frame Snapshot without event fabrication.

    Top-K uses the exact shared production selector. Danger is a stationary
    player/linear-object screening heuristic, not future script prediction or
    a measured collision probability. Laser samples count as tokens, not beams.
    """
    if input_schema not in SEPARATED_SCHEMAS:
        raise ValueError('This audit requires an explicit semantic-separated input schema')
    if not _finite(danger_distance) or danger_distance<0 or not _finite(horizon_frames) or horizon_frames<0:
        raise ValueError('Finite nonnegative audit horizon and distance required')
    prepared=shared_object_candidates(snapshot,hp_independent_contacts=input_schema in (SEPARATED_CONTACT_INPUT,SEPARATED_LASER_INPUT))
    candidates=prepared['candidates']
    field_names=OBJECT_FIELDS
    if input_schema==SEPARATED_LASER_INPUT:
        from .thrl_laser_observation import object_candidates,LASER_FIELDS
        if laser_motion is None:raise ValueError('V5 audit requires the actual same-frame encoder laser-motion context')
        candidates=object_candidates(snapshot,prepared,laser_motion)
        field_names=(*OBJECT_FIELDS,*LASER_FIELDS)
    selector=(select_nonitem_fill_candidates if input_schema in PRIORITY_INPUT_SCHEMAS
        else select_object_candidates)
    selected=selector(candidates)
    chosen={row[4] for row in selected};player=prepared['player_geometry']
    fields,events=_globals(snapshot,player,input_schema)
    groups={}
    omitted=[]
    for kind in ('bullet','boss','enemy','item','laser'):
        population=[row for row in candidates if row[3]==kind]
        kept=[row for row in population if row[4] in chosen]
        danger=[row for row in population if _danger(row,player,horizon_frames,danger_distance)]
        covered=[row for row in danger if row[4] in chosen]
        missed=[row for row in danger if row[4] not in chosen]
        groups[kind]=dict(candidate_tokens=len(population),selected_tokens=len(kept),
            candidate_entities=len({row[4][:2] for row in population}),
            selected_entities=len({row[4][:2] for row in kept}),
            danger_tokens=len(danger),selected_danger_tokens=len(covered),
            omitted_danger_tokens=len(missed),danger_coverage=len(covered)/len(danger) if danger else None)
        omitted.extend(dict(kind=kind,identity=list(row[4]),selection_rank=float(row[0])) for row in missed[:16])
    raw=prepared['raw_bullets'];geometry=prepared['geometry']
    bullet_population=dict(snapshot_bullets=len(snapshot.bullets),
        visible=sum(bool(g['visible']) for g in geometry.values()),
        collidable=sum(bool(g['visible'] and g['collidable']) for g in geometry.values()),
        growth_state2=sum(row['native_state']==2 for row in raw.values()),
        growth_not_yet_collidable=sum(row['native_state']==2 and row['native_timer']<8
            for row in raw.values()),
        intangible=sum(bool(b.intangible) for b in snapshot.bullets))
    map_velocities=[];outside=0
    for b in snapshot.bullets:
        if geometry[b.id]['visible'] and geometry[b.id]['collidable']:
            map_velocities.append((b.vx/64,b.vy/64))
            outside+=int(not(-192<=b.x<=192 and 0<=b.y<=448))
    return dict(schema=SCHEMA,input_schema=input_schema,frame=snapshot.frame,stage_frame=snapshot.stage_frame,
        stage=snapshot.stage,player_state=snapshot.player.state,events_observed=events,
        global_fields=fields,missing_global_fields=[name for name,value in fields.items() if value['missing']],
        object_fields_all=_columns([row[2] for row in candidates],field_names),
        object_fields_selected=_columns([row[2] for row in selected],field_names),
        selection=dict(candidate_tokens=len(candidates),selected_tokens=len(selected),capacity=96,
            groups=groups,bullet_population=bullet_population,omitted_danger_examples=omitted[:32],
            rank_horizon_frames=60.,danger_horizon_frames=float(horizon_frames),danger_distance=float(danger_distance),
            method='exact production top-K; diagnostic danger uses current linear velocity and stationary player',
            laser_tokens_are_samples_not_beams=input_schema!=SEPARATED_LASER_INPUT,
            laser_tokens_are_complete_segments=input_schema==SEPARATED_LASER_INPUT),
        map_input=dict(bullet_velocity_before_spatial_averaging=_columns(map_velocities,('vx_div64','vy_div64')),
            off_field_collidable_bullets=outside,top_k_applies=False,
            map_clip_count_available=False,
            note='Entity speed overflow is not proof of clipped cell averages. Occupancy=count/4 caps at1; exact equality is not strict overflow. No event-dependent map encoding performed.'),
        timers=dict(invulnerability_frames=snapshot.player.invulnerable_frames,deathbomb_remaining=snapshot.player.deathbomb_timer,
            state=snapshot.player.state,deathbomb_valid=bool(0<=snapshot.player.deathbomb_timer<=8 and
                (snapshot.player.state==4 or snapshot.player.deathbomb_timer==0))),
        limitations=['Sampled snapshots do not measure actual policy causality or future collisions.',
            'Top-K object omission does not remove the object from the full-field map.',
            'Linear danger screening is diagnostic; laser samples and rectangle clearance are conservative approximations.',
            'Map cell saturation and opposing-velocity cancellation require separate cell-level evidence.'])


def aggregate_audits(records):
    """Combine JSON records by sample counts; never turn missing readings into0."""
    result=dict(schema=SCHEMA+'-aggregate',snapshots=0,stages={},input_schemas=[],global_fields={},
        object_fields_all={},object_fields_selected={},selection_groups={},missing_events_snapshots=0)
    schemas=set();stages=Counter()
    for record in records:
        if record.get('schema')!=SCHEMA:raise ValueError('Unknown observation audit schema')
        result['snapshots']+=1;stages[str(record['stage'])]+=1;schemas.add(record['input_schema'])
        result['missing_events_snapshots']+=int(not record['events_observed'])
        for name,value in record['global_fields'].items():
            total=result['global_fields'].setdefault(name,dict(samples=0,missing=0,below=0,above=0,boundary=0))
            total['samples']+=int(not value['missing']);total['missing']+=int(value['missing'])
            for key in ('below','above','boundary'):total[key]+=int(value[key])
        for field in ('object_fields_all','object_fields_selected'):
            for name,value in record[field].items():
                total=result[field].setdefault(name,dict(samples=0,below=0,above=0,boundary=0))
                for key in total:total[key]+=value[key]
        for name,value in record['selection']['groups'].items():
            total=result['selection_groups'].setdefault(name,dict(candidate_tokens=0,selected_tokens=0,
                danger_tokens=0,selected_danger_tokens=0,omitted_danger_tokens=0))
            for key in total:total[key]+=value[key]
    result.update(stages=dict(stages),input_schemas=sorted(schemas),
        scope='snapshot/token observations; repeated objects across frames are not distinct entities')
    for value in result['selection_groups'].values():
        value['danger_coverage']=value['selected_danger_tokens']/value['danger_tokens'] if value['danger_tokens'] else None
    return result
