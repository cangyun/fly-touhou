"""Bounded measured laser transitions, never a policy input or action rule."""
from collections import Counter
from pathlib import Path
import json,math
from .atomic_json_io import atomic_write_json

SCHEMA='th14-measured-laser-motion-pairs-v1'


def geometry(laser):
    return dict(id=laser.id,kind=laser.kind,x=laser.x,y=laser.y,angle=laser.angle,
        length=laser.length,width=laser.width,speed=laser.speed,age=laser.age,
        conservative_active=laser.active,native_state=laser.native_state,native_iframes=laser.native_iframes,
        node_count=len(laser.nodes),node_samples=[list(laser.nodes[i]) for i in
            sorted({0,len(laser.nodes)//2,len(laser.nodes)-1})] if laser.nodes else [])


def measured_pairs(before,after):
    dt=after.frame-before.frame
    if before.stage!=after.stage or dt!=4:return []
    old={x.id:x for x in before.lasers};rows=[]
    for b in after.lasers:
        a=old.get(b.id)
        if a is None or a.kind!=b.kind:continue
        if not all(math.isfinite(v) for x in (a,b) for v in
            (x.x,x.y,x.angle,x.length,x.width,x.speed)):continue
        angle=math.atan2(math.sin(b.angle-a.angle),math.cos(b.angle-a.angle))
        origin=(b.x-a.x,b.y-a.y);length=b.length-a.length;width=b.width-a.width
        pattern='+'.join(key for key,yes in [('origin_translation',math.hypot(*origin)>1e-4),
            ('length_change',abs(length)>1e-4),('angle_change',abs(angle)>1e-5),
            ('width_change',abs(width)>1e-4)] if yes) or 'measured_stationary'
        row=dict(schema=SCHEMA,before_frame=before.frame,after_frame=after.frame,stage=after.stage,
            stage_frame=after.stage_frame,elapsed_frames=dt,before=geometry(a),after=geometry(b),
            pattern=pattern,age_continuous=b.age-a.age==dt,
            measured_origin_velocity=[v/dt for v in origin],measured_length_rate=length/dt,
            measured_angle_rate=angle/dt,measured_width_rate=width/dt,
            native_collision_state_known=False,curve_node_identity_verified=False,
            future_motion_guaranteed=False,policy_input_changed=False)
        if not a.nodes and not b.nodes:
            row['measured_endpoint_velocity']=[
                ((b.x+b.length*math.cos(b.angle))-(a.x+a.length*math.cos(a.angle)))/dt,
                ((b.y+b.length*math.sin(b.angle))-(a.y+a.length*math.sin(a.angle)))/dt]
        rows.append(row)
    return rows


class LaserMotionAudit:
    def __init__(self,path,*,max_rows=512,rows_per_pattern=8):
        if not 1<=max_rows<=4096 or not 1<=rows_per_pattern<=64:raise ValueError('Bounded laser telemetry required')
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.max_rows=max_rows;self.rows_per_pattern=rows_per_pattern
        self.observed=Counter();self.saved=Counter();self.steps=0;self.written=0

    def record(self,before,after,*,episode):
        self.steps+=1
        for row in measured_pairs(before,after):
            key=f"{row['after']['kind']}:{row['before']['native_state']}:{row['after']['native_state']}:{row['pattern']}"
            self.observed[key]+=1
            if self.written>=self.max_rows or self.saved[key]>=self.rows_per_pattern:continue
            row['native_episode']=episode
            with self.path.open('a',encoding='utf-8') as stream:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
            self.written+=1;self.saved[key]+=1
        if self.steps%1024==0:self.close()

    def close(self):
        atomic_write_json(self.path.with_suffix('.summary.json'),dict(schema=SCHEMA,policy_steps=self.steps,
            measured_pairs_by_pattern=dict(self.observed),saved_by_pattern=dict(self.saved),rows=self.written,
            max_rows=self.max_rows,policy_input_changed=False,actions_changed=False,new_native_instances=0))
