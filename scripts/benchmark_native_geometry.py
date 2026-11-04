"""Measure native read latency and clearly labelled synthetic many-box cost."""
from __future__ import annotations
import argparse
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.geometry_environment import GeometryHazardResourceCampaignEnv
from fly_touhou.hazard_reader import VersionedHazardReader
from fly_touhou.hazard_observation import HazardObservationEncoder
from fly_touhou.native import sha256_file
from fly_touhou.native_hazard_planner import GeometryNativeHazardPlanner
from fly_touhou.resource_teacher_v5 import ResourceTeacherV5


def timings(values):
    return {'n':len(values),'median_ms':float(np.median(values))*1000,
            'p95_ms':float(np.quantile(values,.95))*1000,'max_ms':max(values)*1000}


def make_fixture(snapshot,count,rectangle):
    """Derived computation fixture, never a claimed original-game scene."""
    original=snapshot.bullets[0]
    extra=deepcopy(snapshot.extra)
    extra['resources']['items']=[]
    extra['resources']['enemies']=[]
    extra['hazard_metadata']['lasers']=[]
    old=extra['hazard_metadata']['bullets'][0]
    flags=0x42 if rectangle else 0x52
    bullets=[];rows=[];geometry=[]
    for n in range(count):
        x=float(n%40*9-175); y=float(n//40%25*16+24)
        b=replace(original,id=0x1000000+n*0x2000,x=x,y=y,vx=.5,vy=1.,radius=4.,active=True)
        bullets.append(b)
        row={**old,'id':b.id,'native_state':1,'native_flags':flags,'native_position':[x,y],
             'native_velocity':[.5,1.],'native_speed':float(np.hypot(.5,1.)),
             'native_hitbox_value':4.,'native_scale':2.}
        rows.append(row)
        geometry.append({'id':b.id,'native_flags':flags,'native_scale':2.,
            'native_hitbox_value':4.,'native_height_value':20. if rectangle else None})
    extra['hazard_metadata']['bullets']=rows
    extra['native_collision_geometry']['bullets']=geometry
    return replace(snapshot,bullets=tuple(bullets),enemies=(),lasers=(),extra=extra)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe',type=Path,required=True)
    p.add_argument('--profile',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    opt=p.parse_args()
    opt.output.mkdir(parents=True,exist_ok=False)
    env=GeometryHazardResourceCampaignEnv(opt.exe,opt.profile,difficulty=3,max_episode_frames=400,
        private_appdata=opt.output/'userdata',dialogue_skip=True,timeout=30,
        window_title='原作尺寸读取耗时验证 · 非通关验收')
    report={'schema':'original-th14-geometry-timing-v1','status':'running',
        'desktop_input':False,'control_intervention':False,'exe_sha256':sha256_file(opt.exe),
        'source_sha256':sha256_file(__file__),'normal_native_reader_samples':[],
        'derived_computation_fixtures':[],'excluded_from_training':True}
    try:
        env.reset()
        report['pid']=env.pid
        legacy=VersionedHazardReader(env.controller._memory,env.controller.profile)
        reader=env.controller.reader
        teacher=ResourceTeacherV5(GeometryNativeHazardPlanner(env.encoder.planner.config))
        fixture_source=None
        for step in range(70):
            start=time.perf_counter();a=legacy.read();old=time.perf_counter()-start
            start=time.perf_counter();b=reader.read();new=time.perf_counter()-start
            np.testing.assert_array_equal(HazardObservationEncoder().encode(a),HazardObservationEncoder().encode(b))
            start=time.perf_counter();action=int(teacher.probabilities(b).argmax());decision=time.perf_counter()-start
            report['normal_native_reader_samples'].append({'step':step,'bullets':len(b.bullets),
                'v4_read_seconds':old,'v5_read_seconds':new,'v5_decision_seconds':decision,
                'v3_vector_identical':True})
            if b.bullets:fixture_source=b
            _,_,done,truncated,_=env.step(action)
            if done or truncated:break
        report['native_frames']=env.episode_frames
        env.close()
        for key in ('v4_read_seconds','v5_read_seconds','v5_decision_seconds'):
            report[key+'_summary']=timings([r[key] for r in report['normal_native_reader_samples']])
        if fixture_source is None:raise RuntimeError('No natural bullet was available for a labelled fixture seed')
        for count in (300,1000):
            for rectangle in (False,True):
                sample=make_fixture(fixture_source,count,rectangle)
                durations=[]
                for _ in range(7):
                    teacher.reset()
                    start=time.perf_counter();teacher.probabilities(sample);durations.append(time.perf_counter()-start)
                report['derived_computation_fixtures'].append({'scope':'synthetic layout for computation cost; not native scenes, not parallel games, not training data',
                    'bullets':count,'shape':'rectangle' if rectangle else 'circle',
                    'operation':'complete V5 teacher decision including preparation and 18 movement candidates',
                    'timing':timings(durations)})
        report['status']='passed'
    except Exception as error:
        report.update(status='failed',error=repr(error));raise
    finally:
        env.close()
        (opt.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({k:v for k,v in report.items() if k!='normal_native_reader_samples'},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
