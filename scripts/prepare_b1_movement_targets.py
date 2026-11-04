"""Dense, explicitly model-predicted action geometry from sealed native states.

No simulated native outcomes, game launches, action overrides or gradients.
Uncertain scenes retain abstention. Targets complement resource-teacher CE.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib,json,os
from pathlib import Path
import sys,time,traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.atomic_json_io import atomic_write_json as write
from fly_touhou.thrl_study_io import utc_now,read_shared_json as read
_arrays=None;_teacher=None;_horizon_specific=False


def initialize(folder,horizon_specific=False):
    global _arrays,_teacher,_horizon_specific
    _horizon_specific=horizon_specific
    from fly_touhou.thrl_movement_teacher import MovementSafetyTeacher
    _arrays={k:np.load(Path(folder)/'demonstrations'/f'{k}.npy',mmap_mode='r') for k in
        ['features','objects','maps','actions','dones','native_episodes','before_frames','after_observation_frames']}
    _teacher=MovementSafetyTeacher()


def chunk(indices):
    from fly_touhou.thrl_movement_teacher import normal_unprotected,swept_clearances
    t=_arrays['features'].shape[0];reasons=Counter();rows=[];facts=[];contradictions=[]
    for index in indices:
        world,frame=divmod(index,t)
        f,o,m=(_arrays[k][frame,world] for k in ('features','objects','maps'))
        if not normal_unprotected(f):reasons['protected_or_non_normal']+=1;continue
        action=int(_arrays['actions'][frame,world]);outcome=-1
        if (frame+1<t and 0<=action<18 and not _arrays['dones'][frame,world]
                and _arrays['native_episodes'][frame+1,world]==_arrays['native_episodes'][frame,world]
                and _arrays['before_frames'][frame+1,world]-_arrays['before_frames'][frame,world]==4
                and _arrays['after_observation_frames'][frame,world]==_arrays['before_frames'][frame+1,world]):
            after=_arrays['features'][frame+1,world]
            if after[7]==f[7]:
                if after[24]>.5 or after[11]>f[11]:outcome=1
                elif normal_unprotected(after) and after[11]==f[11]:outcome=0
        if outcome>=0:facts.append((index,action,outcome))
        uncertainty=[_teacher._uncertainty(f,o,m,horizon=n) for n in (4,12,24)] if _horizon_specific else [_teacher._uncertainty(f,o,m)]*3
        trusted=np.asarray([r is None for r in uncertainty],dtype=bool)
        if not trusted[0]:reasons[uncertainty[0]]+=1;continue
        scene=_teacher._scene(f,o);origin,*geometry=scene
        starts=np.repeat(origin[None],18,axis=0)
        margins=np.stack([swept_clearances(starts,_teacher.velocities,*geometry,duration=n) for n in (4,12,24)])
        if np.isnan(margins).any():raise FloatingPointError('NaN in modeled margins')
        if ((outcome==1 and margins[0,action]>0) or (outcome==0 and margins[0,action]<0)):
            reasons['native_geometry_contradiction']+=1
            contradictions.append(dict(index=index,world=world,time=frame,action=action,outcome=outcome,
                predicted_margin=float(margins[0,action]),native_before_frame=int(_arrays['before_frames'][frame,world])))
            continue
        rows.append((index,np.clip(margins,-64,64).astype(np.float32),trusted))
        reasons['trusted_current_motion']+=1
    return rows,dict(reasons),facts,contradictions


def run(a):
    a.output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    f=np.load(a.demonstrations/'demonstrations/features.npy',mmap_mode='r');t,w=f.shape[:2];n=t*w
    margins=np.zeros((n,3,18),np.float32);valid=np.zeros(n,bool);trusted_horizons=np.zeros((n,3),bool);reasons=Counter();done=0
    actions=np.full(n,-1,np.int64);outcomes=np.full(n,-1,np.int8);contradictions=[]
    indices=[list(range(s,min(n,s+128))) for s in range(0,n,128)]
    with ProcessPoolExecutor(max_workers=a.workers,initializer=initialize,initargs=(str(a.demonstrations),a.horizon_specific)) as pool:
        pending={pool.submit(chunk,ix):len(ix) for ix in indices}
        for future in as_completed(pending):
            rows,counts,facts,conflicts=future.result();reasons.update(counts);done+=pending[future]
            contradictions.extend(conflicts)
            for i,row,trust in rows:margins[i]=row;valid[i]=True;trusted_horizons[i]=trust
            for i,action,outcome in facts:actions[i]=action;outcomes[i]=outcome
            if done%1024<128:
                write(a.output/'progress.json',dict(status='preparing',states=done,target_states=n,valid=int(valid.sum()),
                    reasons=dict(reasons),wall_seconds=time.monotonic()-started))
    np.savez(a.output/'targets.npz',margins=margins,valid=valid,worlds=np.array(w),frames=np.array(t),
             actions=actions,outcomes=outcomes,trusted_horizons=trusted_horizons)
    result=dict(status='completed',utc=utc_now(),states=n,valid=int(valid.sum()),reasons=dict(reasons),
        horizons=[4,12,24],array_order='world-major, then time; margins[state,horizon,source_action0..17]',
        horizon_specific_abstention=a.horizon_specific,trusted_horizon_states=trusted_horizons.sum(0).tolist(),
        target_semantics='current-motion continuous swept geometry; not measured native counterfactual outcomes',
        near12_states=int((valid&(margins[:,1].min(-1)<24)).sum()),
        dangerous12_action_pairs=int(((margins[:,1]<1)&valid[:,None]).sum()),
        measured_four_tick_movements=int((outcomes>=0).sum()),observed_collision_actions=int((outcomes==1).sum()),
        geometry_native_contradictions=sorted(contradictions,key=lambda x:x['index']),
        source_demonstrations=str(a.demonstrations),source_manifest_sha256=hashlib.sha256((a.demonstrations/'demonstrations.json').read_bytes()).hexdigest(),
        wall_seconds=time.monotonic()-started,game_instances_started=0,optimizer_updates=0)
    write(a.output/'report.json',result);print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--demonstrations',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=6)
    p.add_argument('--horizon-specific',action='store_true')
    a=p.parse_args()
    try:run(a)
    except BaseException:
        if a.output.is_dir():write(a.output/'failure.json',dict(traceback=traceback.format_exc()))
        raise
