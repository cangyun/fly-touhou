"""Brief actor alignment to FINAL teacher policy with calibrated risk held fixed.

The fixed current-model DN cache is only an interface-calibration stage.
Internal/sensory training must resume before the active goal is claimed.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
SCHEMA='current-DN-final-policy-actor-alignment-v1'


def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ('source-training','cache-source','training-raw-index','heldout-raw-index',
                 'connectome','environment-config','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--updates',type=int,default=2000)
    p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--seed',type=int,default=2026102941)
    p.add_argument('--evaluation-seed',type=int,default=2027437901)
    p.add_argument('--max-seconds',type=int,default=7200)
    p.add_argument('--include-native-openings',action='store_true')
    p.add_argument('--preserve-native-opening-control',action='store_true')
    a=p.parse_args()
    import psutil
    from fly_touhou.thrl_study_io import read_shared_json as read
    from fly_touhou.thrl_durable_records import write_json,append_jsonl
    current=psutil.Process();cpus=current.cpu_affinity();current.cpu_affinity(cpus[-2:])
    current.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    os.environ.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        CUBLAS_WORKSPACE_CONFIG=':4096:8')
    a.output.mkdir(parents=True,exist_ok=False);started=time.monotonic();costs={}
    launch=dict(schema=SCHEMA,PID=current.pid,create_time=current.create_time(),argv=sys.argv,status='running',
        started_UTC=datetime.now(timezone.utc).isoformat(),source_training=str(a.source_training),
        temporary_fixed_body_and_risk=True,raw_actor_coefficient=0.,teacher_supervised_policy='final composed distribution',
        include_native_openings=a.include_native_openings,
        preserve_native_opening_control=a.preserve_native_opening_control,
        permanent_freeze=False,app_goal_remains_active=True,CUDA_started=False,native_started=False)
    write_json(a.output/'launch.json',launch)
    def guard():
        if (a.output/'STOP').exists():raise RuntimeError('Operator STOP')
        if time.monotonic()-started>a.max_seconds:raise RuntimeError('Finite actor-alignment wall budget reached')
    def progress(phase,**extra):
        write_json(a.output/'progress.json',dict(schema=SCHEMA,phase=phase,costs=costs,
            elapsed_seconds=time.monotonic()-started,app_goal_remains_paused=False,**extra))
    try:
        import numpy as np
        import torch
        from fly_touhou.thrl_dodge_diagnostics import _shared_binary
        from fly_touhou.continuous_guided_dagger import build_guided_model,guided_loss,parameter_payload
        from fly_touhou.continuous_cached_actor_alignment import joined_actor_batch,native_opening_records,encode_native_opening_batch
        from fly_touhou.guided_optimizer_io import groups_from_signature,parameter_signature
        torch.set_num_threads(2);torch.manual_seed(a.seed);rng=np.random.default_rng(a.seed)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        environment=read(a.environment_config);exe=Path(environment['game_exe']).name.lower()
        if any((s.info['name'] or '').lower()==exe for s in psutil.process_iter(['name'])):
            raise RuntimeError('Existing original-game job;do not overlap actor alignment')
        with _shared_binary(a.source_training) as stream:
            source_sha=hashlib.file_digest(stream,'sha256').hexdigest();stream.seek(0)
            parent=torch.load(stream,map_location='cpu',weights_only=True)
        payload=parent['model_payload'];costs.update(payload['counters'])
        model=build_guided_model(payload,a.connectome,device='cuda:0').eval().requires_grad_(False)
        cache_index=read(a.cache_source/'cache-index.json');cache_launch=read(a.cache_source/'launch.json')
        with _shared_binary(cache_launch['source']) as stream:cache_parent=torch.load(stream,map_location='cpu',weights_only=True)
        if any(not torch.equal(v,cache_parent['state_dict'][k]) for k,v in payload['state_dict'].items() if not k.startswith('risk_head.')):
            raise ValueError('Current body/actor must match the feature-cache source before alignment')
        del cache_parent
        launch.update(CUDA_started=True,graph_neurons=model.core.n_neurons,graph_edges=model.core.n_edges,
            source_training_SHA256=source_sha,feature_cache=str(a.cache_source))
        write_json(a.output/'launch.json',launch)
        datasets={};evidence={}
        for role,index in (('train',a.training_raw_index),('heldout',a.heldout_raw_index)):
            records=read(index)['records'];files=cache_index['paths'][role]
            if len(files)!=(len(records)+15)//16:raise ValueError('Exact ordered raw/cache batch count required')
            parts=[]
            for i,path in enumerate(files):
                guard();parts.append(joined_actor_batch(path,records[16*i:16*i+16]))
                progress('joining_original_teacher_labels',role=role,batches_done=i+1,batches_total=len(files))
            if a.include_native_openings:
                openings=native_opening_records(records)
                if not openings:raise ValueError('Actual native opening teacher prefixes required in both splits')
                for first in range(0,len(openings),16):
                    parts.append(encode_native_opening_batch(model,openings[first:first+16],guard))
                    progress('encoding_previously_omitted_native_openings',role=role,records_done=min(first+16,len(openings)),records_total=len(openings))
            datasets[role]={k:torch.as_tensor(np.concatenate([r[k] for r in parts]),device=model.device) for k in parts[0]}
            d=datasets[role]
            with torch.no_grad():
                scores=model.risk_head.layers(d['DN1299']);d['fixed_risk_cost']=torch.nn.functional.softplus(scores)
                d['fixed_predicted_margin']=1.-.5*scores
            evidence[role]=dict(rows=len(d['features']),raw_index=str(index),raw_records=len(records),
                label_join='T16 learning rows16:32,time-major;features,margins,trust,actions,outcomes all checked equal',
                expert_rows=int(d['expert_mask'].sum()),actual_native_opening_rows=int(d['native_opening_mask'].sum()),
                opening_labels='unaltered original ResourceV10 full distributions;exact current-brain reset state')
        write_json(a.output/'data-source.json',dict(evidence=evidence,cache_source_SHA256=cache_index['source_sha256'],
            model_body_frozen_for_brief_interface_alignment=True,no_development_native_observations_used_for_training=True))
        model.actor.train().requires_grad_(True)
        optimizer=torch.optim.Adam(model.actor.parameters(),lr=a.lr,eps=1e-8)
        train=datasets['train'];normal=train['unprotected_mask'];trusted=train['trusted'][:,0]&normal
        mixed=trusted&(train['margins'][:,0,:]>1).any(-1)&(train['margins'][:,0,:]<=0).any(-1)
        hit=normal&(train['outcomes']==1)
        informative=(train['rule_weights']>0)|trusted|hit
        pools=[torch.nonzero(x&informative).reshape(-1).cpu().numpy() for x in
            (train['expert_mask'],mixed,hit,informative)]
        if not len(pools[-1]):raise ValueError('No valid final-policy teacher rows')
        if a.include_native_openings:
            opening_pool=torch.nonzero(train['native_opening_mask']&informative).reshape(-1).cpu().numpy()
            if not len(opening_pool):raise ValueError('Actual opening teacher labels must be informative')
            pools.append(opening_pool)

        def forward_loss(d):
            raw=model.actor(d['DN1299'])
            combined=model.policy_logits(model.compose(raw,d['fixed_risk_cost'],d['features']),d['features'])
            valid=torch.ones(len(raw),device=model.device,dtype=torch.bool)
            loss,metrics=guided_loss(raw,combined,d['fixed_predicted_margin'],d['features'],d,valid,
                native_hit_actor=True,actual_policy_priority=True,raw_actor_coefficient=0.)
            return loss,metrics,combined

        @torch.no_grad()
        def assess(d):
            loss,metrics,combined=forward_loss(d)
            selected=d['margins'][:,0,:].gather(-1,combined[:,:18].argmax(-1)[:,None]).squeeze(-1)
            opening=d['native_opening_mask'];teacher_best=d['rule_probabilities'][:,:18].argmax(-1)
            opening_correct=((combined[:,:18].argmax(-1)==teacher_best)&opening).sum()
            return dict(loss=float(loss),final_CE=float(metrics['composed_CE']),
                native_opening_rows=int(opening.sum()),native_opening_teacher_top1_agreement=float(opening_correct/opening.sum().clamp_min(1)),
                raw_CE_diagnostic_not_optimized=float(metrics['actor_CE']),safe_set=float(metrics['safe_set']),
                mixed_scenes=int(metrics['mixed_rows']),teacher_rows=int(metrics['teacher_rows']),
                final_unsafe=int(metrics['composed_unsafe']),
                final_unsafe_fraction=float(metrics['composed_unsafe']/metrics['mixed_rows'].clamp_min(1)))

        best=None
        for step in range(a.updates+1):
            guard()
            if step%250==0 or step==a.updates:
                held=assess(datasets['heldout'])
                key=(held['final_unsafe_fraction'],held['final_CE'])
                if a.preserve_native_opening_control:
                    if not a.include_native_openings:raise ValueError('Opening selection requires explicit real opening validation')
                    key=(held['final_unsafe_fraction'],1.-held['native_opening_teacher_top1_agreement'],held['final_CE'])
                snapshots=a.output/'actor-candidates';snapshots.mkdir(exist_ok=True)
                torch.save(dict(schema=SCHEMA+'-actor-candidate',step=step,heldout=held,source_training_SHA256=source_sha,
                    actor_state_dict={k:v.detach().cpu().clone() for k,v in model.actor.state_dict().items()},
                    optimizer=optimizer.state_dict(),full_brain_checkpoint=False),snapshots/f'step-{step:04}.pt')
                if best is None or key<best['key']:
                    best=dict(key=key,step=step,heldout=held,weights={k:v.detach().cpu().clone() for k,v in model.actor.state_dict().items()},
                        optimizer=copy.deepcopy(optimizer.state_dict()))
                row=dict(step=step,heldout=held,elapsed_seconds=time.monotonic()-started)
                append_jsonl(a.output/'learning.jsonl',row);progress('cached_DN_final_actor_alignment',actor_step=step,heldout=held)
            if step==a.updates:break
            ids=[]
            for pool in pools:ids.extend(rng.choice(pool if len(pool) else pools[-1],size=64,replace=True).tolist())
            ids=torch.as_tensor(ids,device=model.device);batch={k:v.index_select(0,ids) for k,v in train.items()}
            optimizer.zero_grad(set_to_none=True);loss,_,_=forward_loss(batch)
            if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite final-policy alignment loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.actor.parameters(),5.);optimizer.step()
        model.actor.load_state_dict(best['weights']);optimizer.load_state_dict(best['optimizer'])
        if any(not torch.equal(v.detach().cpu(),payload['state_dict'][k]) for k,v in model.state_dict().items() if not k.startswith('actor.')):
            raise RuntimeError('The fixed current brain or calibrated risk changed during actor-only alignment')
        costs.update(actor_final_policy_Adam_paid=a.updates,selected_actor_final_policy_Adam=best['step'],
            brain_updates_in_this_phase=0)
        selected=parameter_payload(model,payload['base_source'],payload['base_sha256'],payload['calibration'],costs)
        path=a.output/'aligned-parameters.pt'
        with path.open('wb') as stream:torch.save(selected,stream)
        groups=groups_from_signature(model,parent['optimizer'],parent['optimizer_parameter_signature'])
        full_optimizer=torch.optim.Adam(groups);full_optimizer.load_state_dict(parent['optimizer'])
        for parameter in model.actor.parameters():full_optimizer.state[parameter]=copy.deepcopy(optimizer.state[parameter])
        with (a.output/'aligned-training.pt').open('wb') as stream:
            torch.save(dict(schema=parent['schema'],model_payload=selected,optimizer=full_optimizer.state_dict(),
                optimizer_parameter_signature=parameter_signature(model,full_optimizer),costs=costs,
                numpy_RNG=parent['numpy_RNG'],torch_RNG=parent['torch_RNG'],CUDA_RNG=parent['CUDA_RNG'],
                neural_midgame_resume=False,body_and_risk_Adam_preserved=True,selected_actor_Adam_replaced=True),stream)
        write_json(a.output/'alignment-result.json',dict(best_step=best['step'],heldout=best['heldout'],
            raw_actor_loss_coefficient=0.,final_policy_loss_coefficient=1.,risk_and_body_changed=False,
            temporary_alignment_only=True,goal_achieved=False))
        del datasets,parent,payload,selected,full_optimizer,optimizer;gc.collect();torch.cuda.empty_cache()
        model.eval().requires_grad_(False)
        from fly_touhou.continuous_no_bomb_first_contact import make_no_bomb_policy,evaluate_no_bomb
        from fly_touhou.owned_worker_affinity import worker_cpu_pool
        from fly_touhou.thrl_job_paths import configure_native_temp
        os.environ['FLY_TOUHOU_DESKTOP_NAME']='fly-touhou-job-final-actor-alignment-'+str(current.pid)
        configure_native_temp(a.output)
        with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        policy=make_no_bomb_policy(torch,model,family='deep',episodes=20,movement_readout='composed')
        launch['native_started']=True;write_json(a.output/'launch.json',launch)
        report=evaluate_no_bomb(torch,policy,environment,a.output/'frozen-final-policy20',
            checkpoint_sha256=digest,episodes=20,workers=20,seed=a.evaluation_seed,check_stop=guard,
            worker_cpus=worker_cpu_pool(cpus),progress=lambda row:progress('frozen_final_policy20',evaluation=row))
        write_json(a.output/'result.json',dict(schema=SCHEMA,status='completed',costs=costs,report=report,
            alignment=dict(best_step=best['step'],heldout=best['heldout']),goal_achieved=False,
            internally_trainable_joint_phase_still_required=True,full_game_training_started=False,
            app_goal_remains_active=True,elapsed_seconds=time.monotonic()-started))
        launch['status']='completed';write_json(a.output/'launch.json',launch);progress('actor_alignment_experiment_completed')
    except BaseException as error:
        write_json(a.output/'failure.json',dict(schema=SCHEMA,error=repr(error),traceback=traceback.format_exc(),
            costs=costs,automatic_retry=False,app_goal_remains_active=True,elapsed_seconds=time.monotonic()-started))
        raise


if __name__=='__main__':main()
