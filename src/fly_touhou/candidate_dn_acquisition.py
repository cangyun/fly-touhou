"""Actual student-controlled training worlds; mine selected false-safe queries."""
from pathlib import Path
import copy,time
import numpy as np
import torch
from .compact_spiking_native import CompactContactEnv
from .compact_spiking_run import kwargs
from .thrl_durable_records import write_json


class LabelContactEnv(CompactContactEnv):
    def _info(self,elapsed=0):
        info=super()._info(elapsed)
        r=getattr(getattr(self,'encoder',None),'last_result',None)
        if r is not None and self.snapshot is not None and (r['frame'],r['stage'])==(self.snapshot.frame,self.snapshot.stage):
            d=r['details']
            info['compact_observation_frame']=self.snapshot.frame
            info['compact_targets']=dict(margins=d['margins'].tolist(),trusted=d['trusted'].tolist(),normal=d['normal'],
                frame=r['frame'],native_episode=int(self.episode_id))
        return info


def factory(options):
    options=dict(options);options.pop('online_teacher_revision',None)
    return LabelContactEnv(**options)


@torch.no_grad()
def acquire(model,config,output,*,worker_cpus=None,workers=16,steps=512,seed=202610319,guard=lambda:None,progress=lambda x:None):
    from . import thrl_vector
    from .owned_worker_affinity import spread_owned_workers
    import os,psutil
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    cfg=copy.deepcopy(config);cfg['seed']=seed
    old=thrl_vector.worker_factory;vector=None;rng=np.random.default_rng(seed);begun=time.monotonic()
    recorded=[[] for _ in range(workers)];count=hits=0
    try:
        thrl_vector.worker_factory=factory
        vector=thrl_vector.THRLNativeVector(kwargs(cfg,output,workers,'train'),output/'workers',timeout=180)
        if worker_cpus:
            p=psutil.Process(os.getpid());write_json(output/'affinity.json',spread_owned_workers(p.pid,p.create_time(),worker_cpus))
        obs,infos=vector.reset();states=[model.initial_state(1) for _ in range(workers)]
        for t in range(steps):
            guard();features=obs['features'].copy();before=infos
            out,states,_=model.infer_candidates(features,states,return_all_DN=True)
            actions=out['logits'].argmax(-1).cpu().numpy()
            all_DN=out['candidate_DN'].cpu().numpy();all_scores=out['horizon_scores'].cpu().numpy()
            samples=[]
            for w in range(workers):
                label=before[w]['compact_targets']
                if (label['frame'],label['native_episode'])!=(before[w]['compact_observation_frame'],before[w]['native_episode']):
                    raise RuntimeError('Original pre-action label clock differs')
                m=np.asarray(label['margins'],np.float32);trust=np.asarray(label['trusted'],bool)
                danger=np.flatnonzero(m[0]<=0);boundary=np.flatnonzero(abs(m[0])<8)
                queries=np.array([actions[w],rng.choice(danger) if len(danger) else rng.integers(18),
                    rng.choice(boundary) if len(boundary) else rng.integers(18),rng.integers(18)])
                samples.append(dict(DN=all_DN[w,queries],margins=m[:,queries].T,
                    mask=np.broadcast_to(trust & label['normal'],(4,3)).copy(),before_scores=all_scores[w,queries],
                    queries=queries,actual_action=np.full(4,actions[w]),features=np.repeat(features[w:w+1],4,axis=0),
                    native_frame=np.full(4,label['frame']),native_episode=np.full(4,label['native_episode'])))
            after,_,done,truncated,infos=vector.step(actions.tolist());count+=workers
            if np.asarray(truncated).any():raise RuntimeError('Recording budget must not truncate native episodes')
            for w,info in enumerate(infos):
                if info['policy_action']!=int(actions[w]) or info['bombs_used']!=0:raise RuntimeError('Executed student action differs')
                hit=bool(info.get('avoidance_episode',{}).get('first_collision_observed'));hits+=hit
                gap=float(before[w]['compact_targets']['margins'][0][actions[w]])
                if (hit and gap>0) or (not hit and gap<0):samples[w]['mask'][:]=False
                samples[w]['actual_native_contact']=np.full(4,hit)
                recorded[w].append(samples[w])
            obs=after;ended=np.flatnonzero(done).tolist()
            if ended:
                fresh=vector.reset_at(ended);obs=vector.observation
                for w,info in zip(ended,fresh):infos[w]=info;states[w]=model.initial_state(1)
            if (t+1)%16==0:progress(dict(student_actions=count,target=workers*steps,native_contacts=hits,elapsed_seconds=time.monotonic()-begun))
        records=[]
        for w,rows in enumerate(recorded):
            arrays={k:np.concatenate([row[k] for row in rows]) for k in rows[0]};path=output/f'world-{w:02}.npz'
            with path.open('xb') as f:np.savez_compressed(f,**arrays)
            records.append(dict(path=str(path),fold='heldout' if w%5==0 else 'train',world=w,queries=len(arrays['DN'])))
        manifest=dict(status='completed',records=records,student_actions=count,native_contacts=hits,teacher_actions=0,
            optimizer_updates=0,source='same original pre-action state and same action-producing SNN forward',
            labels='provisional all-hazard geometric forecasts;contradictory actual outcomes veto labels',
            new_training_RNG_seed=seed,recording_budget_is_not_episode_cap=True,query_sampling='selected plus danger,boundary,uniform')
        write_json(output/'manifest.json',manifest);return manifest
    finally:
        thrl_vector.worker_factory=old
        if vector is not None:
            vector.close();write_json(output/'cleanup.json',dict(errors=vector.close_errors))
            if vector.close_errors:raise RuntimeError('DN acquisition cleanup unresolved')
