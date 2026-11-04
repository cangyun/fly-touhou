"""Compare saved original-game scenes against the pre-optimization geometry."""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np
from fly_touhou import thrl_movement_teacher as movement
from fly_touhou.compact_spiking_sensation import encode_geometry


def run(snapshot, evidence):
    spec=importlib.util.spec_from_file_location('fly_touhou._previous_movement_teacher',snapshot)
    previous=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=previous
    spec.loader.exec_module(previous)
    old,new=previous.swept_clearances,movement.swept_clearances
    rng=np.random.default_rng(20261031)
    comparisons=0
    for kind in ('circle','rectangle','mixed'):
        for _ in range(24):
            starts=rng.uniform([-184,32],[184,432],size=(18,2))
            if _%3==0:starts[:2]=[[184,432],[-184,32]]
            velocities=rng.uniform(-4.5,4.5,(18,2))
            positions=rng.uniform([-184,32],[184,432],size=(30,2))
            motions=rng.uniform(-6,6,(30,2))
            circles=np.ones(30,bool) if kind=='circle' else np.zeros(30,bool) if kind=='rectangle' else rng.random(30)<.5
            radii=rng.uniform(.1,12,30);extents=rng.uniform(.1,12,(30,2));offset=rng.uniform(-3,3,2)
            for per_object in (False,True):
                args=(starts,velocities,positions,motions,circles,radii,extents,offset)
                opts=dict(initial_time=float(_%4),duration=float((0,4,12,24)[_%4]),per_object=per_object)
                np.testing.assert_array_equal(new(*args,**opts),old(*args,**opts))
                comparisons+=1
    records=[json.loads(line)['transition']['geometry_replay_before']
        for line in evidence.read_text(encoding='utf-8').splitlines() if line.strip()]
    timings={'previous':[],'optimized':[]};scenes=[]
    try:
        for row in records:
            if not row.get('available'):continue
            f=np.asarray(row['source_features25'],np.float32)
            o=np.asarray(row['source_objects29'],np.float32)
            options=dict(complete=True,anticipate_arming=row['forecast_native_arming'],
                growth_motion_envelope=row['forecast_growth_motion'],
                native_forecasts=row['source_native_forecasts'],unknown_laser_player_motion=True)
            results={}
            for repeat in range(4):
                for name,implementation in (('previous',old),('optimized',new)):
                    movement.swept_clearances=implementation
                    begin=time.perf_counter()
                    x,detail=encode_geometry(f,o,**options)
                    elapsed=time.perf_counter()-begin
                    if repeat:timings[name].append(elapsed)
                    results[name]=(x,detail)
            a,b=results['previous'],results['optimized']
            np.testing.assert_array_equal(a[0],b[0])
            np.testing.assert_array_equal(a[1]['margins'],b[1]['margins'])
            np.testing.assert_array_equal(a[1]['trusted'],b[1]['trusted'])
            stored=np.asarray(row['encoded_features'],np.float32)[:111]
            np.testing.assert_array_equal(b[0],stored)
            scenes.append(dict(frame=row['frame'],stage=row['stage'],objects=len(o),
                exact_feature_match=True,exact_margin_and_trust_match=True))
    finally:
        movement.swept_clearances=new
    medians={name:statistics.median(values) for name,values in timings.items()}
    return dict(scope=('Saved original contact states, plus synthetic edge cases; no new game or learner. '
        'Timing is local single-process CPU and does not directly predict whole-training throughput.'),
        synthetic_exact_comparisons=comparisons,scenes=scenes,median_seconds_per_scene=medians,
        median_time_ratio=medians['optimized']/medians['previous'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot',type=Path,required=True)
    p.add_argument('--evidence',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=run(a.snapshot,a.evidence)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))
