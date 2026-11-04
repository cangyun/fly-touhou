"""V5: independent globals/map and full laser-segment entity tokens.

Historical V1..V4 encodings stay unchanged. A line is one segment, curves use
their current adjacent nodes with explicitly unknown motion. Endpoint finite
differences are observations, never guaranteed future motion/collision flags.
"""
from __future__ import annotations
import math
import numpy as np
from .thrl_inputs import SEPARATED_CONTACT_INPUT,SEPARATED_LASER_INPUT
from .thrl_geometry_observation import (OBJECT_FIELDS,shared_object_candidates,
    select_nonitem_fill_candidates)
from .thrl_timer_observation import THRLTimerObservationEncoder,timer_features
from .thrl_observation import THRLObservationEncoder
from .thrl_line_laser_motion import ObservedLineLaserTracker

LASER_FIELDS=('segment_half_x_bounded','segment_half_y_bounded',
    'segment_half_vx_bounded','segment_half_vy_bounded','half_width_rate_bounded',
    'endpoint_motion_observed','native_laser_state_bounded','native_laser_iframes_bounded')
OBJECT_DIM=29


def bounded(value,scale):
    value=np.asarray(value,dtype=np.float64)
    result=(value/(np.abs(value)+scale)).astype(np.float32)
    if not np.isfinite(result).all() or np.any(np.abs(result)>=1):
        raise ValueError('Laser measurement exceeds invertible FP32 bounded transport')
    return result


def unbounded(value,scale):
    value=np.asarray(value,dtype=np.float64)
    if not np.isfinite(value).all() or np.any(np.abs(value)>=1):
        raise ValueError('Invalid bounded laser measurement')
    return value*scale/(1-np.abs(value))


def segment_candidates(snapshot,measurement,*,exclude_curve_ids=()):
    """One token per line/curve edge; no sparse circle approximation of a beam."""
    if (measurement.frame,measurement.stage)!=(snapshot.frame,snapshot.stage):
        raise ValueError('Same-frame measured laser motion required')
    measured={row.identity:row for row in measurement.lines};p=snapshot.player
    rows=[]
    for laser in snapshot.lasers:
        if laser.width<=0 or (laser.nodes and laser.id in exclude_curve_ids):continue
        points,_=THRLObservationEncoder._polyline(laser)
        motion=measured.get(laser.id)
        known=motion is not None and motion.motion_observed and not laser.nodes
        va=np.asarray(motion.origin_velocity if known else (0.,0.))
        vb=np.asarray(motion.endpoint_velocity if known else (0.,0.))
        width_rate=motion.width_rate/2 if known else 0.
        single_node=len(laser.nodes)==1
        segments=[(points[0],points[0])] if single_node else zip(points[:-1],points[1:])
        for index,(a,b) in enumerate(segments):
            if np.array_equal(a,b) and not single_node:continue
            midpoint=(a+b)/2;half=(b-a)/2;velocity=(va+vb)/2;half_velocity=(vb-va)/2
            delta=midpoint-np.asarray((p.x,p.y))
            token=np.zeros(OBJECT_DIM,np.float32)
            token[0]=1;token[4 if laser.nodes else 3]=1
            # Laser-only bounded transforms prevent silently clipping long beams
            # or high endpoint rates. All non-laser first21 fields remain V4.
            token[6:8]=bounded(delta,[384.,448.]);token[8:10]=bounded(velocity,12.)
            token[10]=bounded(laser.width/2,128.);token[12:14]=1
            token[16]=np.clip(laser.kind/3,0,1);token[18]=min(laser.age/120,1.)
            token[21:23]=bounded(half,[384.,448.]);token[23:25]=bounded(half_velocity,12.)
            token[25]=bounded(width_rate,12.);token[26]=known
            token[27]=-1 if laser.native_state is None else bounded(laser.native_state,8.)
            token[28]=-1 if laser.native_iframes is None else bounded(laser.native_iframes,60.)
            ranks=[]
            for t in (0.,4.):
                aa=a+va*t;bb=b+vb*t;d=bb-aa
                fraction=np.clip(np.dot(np.asarray((p.x,p.y))-aa,d)/max(float(d@d),1e-12),0,1)
                ranks.append(float(np.linalg.norm(aa+fraction*d-(p.x,p.y))))
            rank=max(0.,min(ranks)-laser.width/2)+.1*ranks[0]
            rows.append((rank,tuple(token),token,'laser',('laser-segment',laser.id,index)))
    return rows


def object_candidates(snapshot,prepared,motion):
    candidates=[]
    for rank,tie,token,group,identity in prepared['candidates']:
        if group=='laser':continue
        token=[*np.clip(token,-1,1),*([0.]*8)]
        candidates.append((rank,(*tie,*([0.]*8)),token,group,identity))
    candidates.extend(segment_candidates(snapshot,motion))
    return candidates


class THRLLaserObservationEncoder:
    schema=SEPARATED_LASER_INPUT
    feature_shape=(25,);object_shape=(96,OBJECT_DIM);map_shape=(16,92,96)

    def __init__(self,*,require_events=True):
        self.reference=THRLTimerObservationEncoder(SEPARATED_CONTACT_INPUT,require_events=require_events)
        self.feature_names=self.reference.feature_names
        self.prefix=THRLObservationEncoder(require_events=require_events)
        self.tracker=ObservedLineLaserTracker();self.native_episode=0;self.last_motion=None

    def set_native_episode(self,episode):
        if type(episode) is not int or episode<0:raise ValueError('Measured native episode required')
        if episode!=self.native_episode:
            self.tracker.reset();self.last_motion=None;self.native_episode=episode

    def reset(self):
        self.tracker.reset();self.last_motion=None

    def specification(self):
        spec=self.reference.specification()
        for name in ('sole_change_from_object_priority_v3','object_fields_unchanged',
                     'object_selection_quotas_unchanged','numeric_feature_count_unchanged'):
            spec.pop(name,None)
        spec.update(schema=self.schema,source_observation_schema=SEPARATED_CONTACT_INPUT,
            checkpoint_compatible_with_contact_v4=False,native_collision_state_verified=False,
            future_motion_guaranteed=False,legacy_V4_encoder_unchanged=True)
        spec['objects'].update(shape=[96,OBJECT_DIM],fields=[*OBJECT_FIELDS,*LASER_FIELDS],
            laser_geometry='one complete line segment; each current curve edge is a segment; no sampled point circles',
            laser_allocation='same16 token reserve now counts segments; full laser map has no top-k limit',
            laser_transforms='laser dx/dy,center velocity,halfwidth and new segment fields: signed x/(abs(x)+scale); nonlaser first21 unchanged',
            laser_scales=dict(position=[384,448],velocity=12,halfwidth=128,halfvector=[384,448],
                halfvector_velocity=12,halfwidth_rate=12,native_state=8,native_iframes=60),
            unknown_laser_motion='column26=0; velocity placeholders never mean observed stationary; curve correspondence unverified',
            native_laser_flags='current measured fields only; not inferred collision gates',
            laser_presence='retain positive-width current geometry even when legacy active=False;column13 means conservative potential hazard,not verified native collision')
        spec['mapping']['object_motion']='line origin/endpoint finite differences with episode,stage,identity,age and1..8frame continuity; curves unknown'
        spec['mapping']['laser_velocity']='map points interpolate measured endpoint velocities; unknown map velocities remain0 and unknown is explicit in segment tokens'
        return spec

    def encode(self,snapshot):
        prepared=shared_object_candidates(snapshot,hp_independent_contacts=True)
        raw=snapshot.extra['hazard_metadata'].get('lasers')
        if not isinstance(raw,list) or len(raw)!=len(snapshot.lasers) or any(not isinstance(row,dict) or 'id' not in row for row in raw):
            raise ValueError('Same-frame full native laser metadata required')
        rows={row['id']:row for row in raw}
        if len(rows)!=len(raw) or set(rows)!={laser.id for laser in snapshot.lasers}:
            raise ValueError('Native laser identities do not match this snapshot')
        for laser in snapshot.lasers:
            row=rows[laser.id]
            timer=row.get('native_timer')
            if type(timer) is not int:raise ValueError('Measured native laser age required')
            if (row.get('native_state')!=laser.native_state or row.get('native_iframes')!=laser.native_iframes
                    or row.get('native_type')!=laser.kind or timer&0xffffffff!=laser.age
                    or tuple(row.get('native_position',()))!=(laser.x,laser.y)
                    or any(row.get('native_'+name)!=getattr(laser,name) for name in ('angle','length','width','speed'))):
                raise ValueError('Measured native laser geometry/flags differ from this frame')
        motion=self.tracker.observe(snapshot,native_episode=self.native_episode)
        self.last_motion=motion
        observation=self.prefix.encode(snapshot,_geometry=prepared['geometry'],_hp_independent_contacts=True,
            _laser_motion={row.identity:row for row in motion.lines})
        base=observation['features'];p=snapshot.player;g=prepared['player_geometry']
        values=timer_features(p)
        features=np.asarray([base[0],base[1],base[8],base[9],base[10],base[11],base[12],base[13],
            base[16],*base[32:36],g['circle_component']/64,*g['bounds_half_extents']/64,
            *g['bounds_offset']/64,float(p.focused),values[1],values[0],*values[2:]],np.float32)
        features=np.clip(features,-1,1)
        candidates=object_candidates(snapshot,prepared,motion)
        objects=np.zeros(self.object_shape,np.float32)
        for i,row in enumerate(select_nonitem_fill_candidates(candidates)):objects[i]=row[2]
        result=dict(features=features,objects=objects,maps=observation['maps'])
        if any(not np.isfinite(v).all() or np.abs(v).max()>1 for v in result.values()):
            raise ValueError('V5 finite/range contract failed')
        return result
