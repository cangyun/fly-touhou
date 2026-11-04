"""Finite diagnostic: current fullbrain features,three risk decoders,NoBomb20.

The brain and its OWN learned sensory encoder are held fixed only for this
readout-identification phase. A positive result does not satisfy the active
goal until subsequent internally trainable joint training retains avoidance.
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
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
SCHEMA='current-fullbrain-three-coordinate-risk-readout-calibration-v1'


def main():
    p=argparse.ArgumentParser(__doc__)
    for key in ('source','source-training','replay-index','expert-data','connectome','environment-config','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--source-sha256',required=True)
    p.add_argument('--updates',type=int,default=2000)
    p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--seed',type=int,default=2026102933)
    p.add_argument('--evaluation-seed',type=int,default=2027435901)
    p.add_argument('--max-seconds',type=int,default=10800)
    a=p.parse_args()
    import psutil
    from fly_touhou.thrl_study_io import read_shared_json as read
    from fly_touhou.thrl_durable_records import write_json,append_jsonl
    process=psutil.Process();cpus=process.cpu_affinity();process.cpu_affinity(cpus[-2:])
    process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    os.environ.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        CUBLAS_WORKSPACE_CONFIG=':4096:8')
    a.output.mkdir(parents=True,exist_ok=False);started=time.monotonic();costs={}
    launch=dict(schema=SCHEMA,PID=process.pid,create_time=process.create_time(),argv=sys.argv,status='running',
        started_UTC=datetime.now(timezone.utc).isoformat(),source=str(a.source),source_sha256=a.source_sha256,
        diagnostic_only=True,brain_and_own_sensory_frozen_for_decoder_identification=True,
        permanent_freeze=False,app_goal_remains_active=True,full_game_training_started=False,
        views=['encoded384','port_normalized384','DN1299'],production_readout='DN1299 only',
        input_probes_never_deployed=True,teacher_actions_at_inference=0,CUDA_started=False,native_started=False)
    write_json(a.output/'launch.json',launch)
    def guard():
        if (a.output/'STOP').exists():raise RuntimeError('Operator STOP')
        if time.monotonic()-started>a.max_seconds:raise RuntimeError('Finite calibration wall budget reached')
    def progress(phase,**extra):
        write_json(a.output/'progress.json',dict(schema=SCHEMA,phase=phase,costs=costs,
            elapsed_seconds=time.monotonic()-started,app_goal_remains_paused=False,**extra))
    try:
        import numpy as np
        import torch
        from torch.nn import functional as F
        from fly_touhou.thrl_dodge_diagnostics import _shared_binary
        from fly_touhou.continuous_guided_dagger import build_guided_model,SequenceReplay,LayerNormalizedRisk,parameter_payload
        from fly_touhou.continuous_guided_replay import add_expert_sequences
        from fly_touhou.continuous_connectome_training import reset_activity
        from fly_touhou.continuous_signed_clearance import signed_clearance4f_terms
        from fly_touhou.guided_optimizer_io import groups_from_signature,parameter_signature
        torch.set_num_threads(2);rng=np.random.default_rng(a.seed)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        environment=read(a.environment_config)
        exe=Path(environment['game_exe']).name.lower()
        if any((q.info['name'] or '').lower()==exe for q in psutil.process_iter(['name'])):
            raise RuntimeError('An original-game job exists;do not overlap calibration')
        with _shared_binary(a.source) as stream:
            digest=hashlib.file_digest(stream,'sha256').hexdigest();stream.seek(0)
            if digest!=a.source_sha256:raise ValueError('Exact frozen source SHA required')
            payload=torch.load(stream,map_location='cpu',weights_only=True)
        with _shared_binary(a.source_training) as stream:parent=torch.load(stream,map_location='cpu',weights_only=True)
        if any(not torch.equal(v,parent['model_payload']['state_dict'][k]) for k,v in payload['state_dict'].items()):
            raise ValueError('Parent Adam checkpoint must contain the exact selected model weights')
        costs.update(payload['counters']);model=build_guided_model(payload,a.connectome,device='cuda:0').eval().requires_grad_(False)
        launch.update(CUDA_started=True,graph_neurons=model.core.n_neurons,graph_edges=model.core.n_edges)
        write_json(a.output/'launch.json',launch)
        train_records=read(a.replay_index)['records']
        held=SequenceReplay(a.output/'heldout-raw')
        progress('preparing_whole_teacher_heldout')
        held_evidence=add_expert_sequences(held,a.expert_data,rng,split_role='heldout',check_stop=guard,
            progress=lambda row:progress('preparing_whole_teacher_heldout',data=row))
        write_json(a.output/'heldout-source.json',held_evidence)
        if any(r['metadata'].get('whole_game_split')=='heldout' for r in train_records):
            raise ValueError('Held-out teacher trajectories must not enter the training cache')
        expert_seeds={r['metadata'].get('native_seed') for r in train_records if r['origin']=='expert'}
        if expert_seeds&set(held_evidence['split']['heldout_native_seeds']):
            raise ValueError('Current expert-training seeds overlap the held-out teacher records')
        source_roles={'train':train_records,'heldout':held.records};paths={}
        for role,records in source_roles.items():
            folder=a.output/('features-'+role);folder.mkdir()
            paths[role]=[]
            for first in range(0,len(records),16):
                guard();chosen=records[first:first+16];arrays=[]
                for record in chosen:
                    with np.load(record['path'],allow_pickle=False) as source:
                        arrays.append({k:source[k] for k in ('features','maps','objects','episode_starts',
                            'margins','trusted','unprotected_mask','actions','outcomes')})
                batch={k:np.stack([r[k] for r in arrays],axis=1) for k in arrays[0]}
                b=len(chosen);before=model.initial_state(16);saved=[]
                with torch.no_grad():
                    encoded=model.encode_raw(batch['features'],batch['maps'],objects=batch['objects'])
                    normalized=model.core.input_encoder.normalized(encoded)
                    for t in range(32):
                        guard();starts=torch.zeros(16,dtype=torch.bool,device=model.device)
                        starts[:b]=torch.as_tensor(batch['episode_starts'][t],device=model.device)
                        before=reset_activity(before,starts);z=torch.zeros(16,384,device=model.device);z[:b]=encoded[t]
                        DN,before=model.core(z,before)
                        if t>=16:saved.append(DN[:b].cpu().numpy())
                fields={k:v[16:].reshape(-1,*v.shape[2:]) for k,v in batch.items() if k not in ('maps','objects','episode_starts')}
                fields.update(encoded384=encoded[16:].cpu().numpy().reshape(-1,384),
                    port_normalized384=normalized[16:].cpu().numpy().reshape(-1,384),
                    DN1299=np.stack(saved).reshape(-1,len(model.core.output_indices)))
                path=folder/f'batch-{first//16:04}.npz';np.savez_compressed(path,**fields);paths[role].append(str(path))
                progress('caching_current_frozen_fullbrain',cache_role=role,records_done=min(first+16,len(records)),records_total=len(records))
                del batch,arrays,encoded,normalized,saved;gc.collect()
        write_json(a.output/'cache-index.json',dict(schema=SCHEMA,source_sha256=digest,paths=paths,
            raw_observations_reencoded_with_current_model=True,archived_old_DN_used=False,
            cached_DN_for_frozen_decoder_diagnostic_only=True,burn_in=16,canonical_brain_batch=16,
            heldout_whole_teacher_games=held_evidence['split']['heldout_native_seeds']))
        dataset={}
        for role,files in paths.items():
            all_rows=[]
            for path in files:
                with np.load(path,allow_pickle=False) as source:all_rows.append({k:source[k] for k in source.files})
            dataset[role]={k:torch.as_tensor(np.concatenate([d[k] for d in all_rows]),device=model.device) for k in all_rows[0]}
        views=('encoded384','port_normalized384','DN1299')
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(a.seed)
            heads={name:LayerNormalizedRisk(width=dataset['train'][name].shape[-1]).to(model.device) for name in views[:2]}
        heads['DN1299']=model.risk_head
        for head in heads.values():head.train().requires_grad_(True)
        optimizers={name:torch.optim.Adam(head.parameters(),lr=a.lr,eps=1e-8) for name,head in heads.items()}
        data=dataset['train'];normal=data['unprotected_mask'].bool();trusted=data['trusted'][:,0].bool()&normal
        mixed=trusted&(data['margins'][:,0,:]>1).any(-1)&(data['margins'][:,0,:]<=0).any(-1)
        contact=normal&(data['outcomes']==1)
        pools=[torch.nonzero(mask).reshape(-1).cpu().numpy() for mask in (mixed,contact,trusted|contact)]
        if not len(pools[-1]):raise ValueError('No trustworthy training risk targets')
        best={};curve=[]
        def loss_and_metrics(head,d,name):
            pred=1.-.5*head.layers(d[name]);valid=torch.ones(len(pred),device=model.device,dtype=torch.bool)
            physical=signed_clearance4f_terms(pred,d,d['features'],valid)
            eligible=d['trusted'][:,0]&d['unprotected_mask'];margin=d['margins'][:,0,:]
            safe=margin>1;danger=margin<=0;pairs=safe[:,:,None]&danger[:,None,:]&eligible[:,None,None]
            gaps=pred[:,:,None]-pred[:,None,:]
            rank=torch.where(pairs,F.softplus(1.-gaps),0.).sum()/pairs.sum().clamp_min(1)
            hit=d['unprotected_mask']&(d['outcomes']==1);actions=d['actions'].clamp(0,17)
            selected=pred.gather(-1,actions[:,None]).squeeze(-1)
            contact_loss=torch.where(hit,F.softplus(selected-.5),0.).sum()/hit.sum().clamp_min(1)
            return physical['loss']+.2*rank+.5*contact_loss,pred
        @torch.no_grad()
        def assess(head,d,name):
            losses=[];predictions=[]
            for first in range(0,len(d['features']),512):
                part={k:v[first:first+512] for k,v in d.items()};loss,pred=loss_and_metrics(head,part,name)
                losses.append((float(loss),len(pred)));predictions.append(pred)
            pred=torch.cat(predictions);margin=d['margins'][:,0,:];eligible=d['trusted'][:,0]&d['unprotected_mask']
            safe=margin>1;danger=margin<=0;critical=eligible&safe.any(-1)&danger.any(-1)
            chosen=pred.argmax(-1);selected=margin.gather(-1,chosen[:,None]).squeeze(-1)
            unsafe=float(((selected<=0)&critical).sum()/critical.sum().clamp_min(1))
            pairs=danger[:,:,None]&safe[:,None,:]&critical[:,None,None]
            difference=pred[:,None,:]-pred[:,:,None]
            auc=((difference>0).float()+.5*(difference==0).float())
            auc=float(torch.where(pairs,auc,0.).sum()/pairs.sum().clamp_min(1))
            return dict(loss=sum(v*n for v,n in losses)/sum(n for v,n in losses),rows=len(pred),
                mixed_scenes=int(critical.sum()),selected_proxy_unsafe_fraction=unsafe,pair_ranking_AUC=auc)
        for step in range(a.updates+1):
            guard()
            if step%250==0 or step==a.updates:
                node={}
                for name,head in heads.items():
                    metrics=assess(head,dataset['heldout'],name);node[name]=metrics
                    key=(metrics['selected_proxy_unsafe_fraction'],metrics['loss'])
                    if name not in best or key<best[name]['key']:
                        best[name]=dict(key=key,step=step,metrics=metrics,
                            weights={k:v.detach().cpu().clone() for k,v in head.state_dict().items()},
                            optimizer=copy.deepcopy(optimizers[name].state_dict()))
                row=dict(step=step,heldout=node,elapsed_seconds=time.monotonic()-started)
                append_jsonl(a.output/'learning.jsonl',row);curve.append(row)
                progress('frozen_body_readout_calibration',calibration_step=step,heldout=node)
            if step==a.updates:break
            ids=[]
            for pool,count in zip(pools,(128,64,64)):
                pool=pool if len(pool) else pools[-1];ids.extend(rng.choice(pool,size=count,replace=True).tolist())
            chosen=torch.as_tensor(ids,device=model.device);batch={k:v.index_select(0,chosen) for k,v in data.items()}
            for name,head in heads.items():
                optimizer=optimizers[name];optimizer.zero_grad(set_to_none=True)
                loss,_=loss_and_metrics(head,batch,name)
                if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite readout calibration')
                loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),5.);optimizer.step()
        selected=best['DN1299'];model.risk_head.load_state_dict(selected['weights'])
        optimizers['DN1299'].load_state_dict(selected['optimizer'])
        if any(not torch.equal(v.detach().cpu(),payload['state_dict'][k]) for k,v in model.state_dict().items()
                if not k.startswith('risk_head.')):
            raise RuntimeError('Decoder-identification stage unexpectedly changed the body or actor')
        costs.update(DN_readout_calibration_Adam_paid=a.updates,selected_DN_readout_calibration_Adam=selected['step'],
            diagnostic_input_probe_Adam_paid=2*a.updates,brain_updates_in_this_phase=0)
        calibrated=parameter_payload(model,payload['base_source'],payload['base_sha256'],payload['calibration'],costs)
        parameter_path=a.output/'calibrated-parameters.pt'
        with parameter_path.open('wb') as stream:torch.save(calibrated,stream)
        groups=groups_from_signature(model,parent['optimizer'],parent['optimizer_parameter_signature'])
        full_optimizer=torch.optim.Adam(groups);full_optimizer.load_state_dict(parent['optimizer'])
        for parameter in model.risk_head.parameters():full_optimizer.state[parameter]=copy.deepcopy(optimizers['DN1299'].state[parameter])
        with (a.output/'calibrated-training.pt').open('wb') as stream:
            torch.save(dict(schema=parent['schema'],model_payload=calibrated,optimizer=full_optimizer.state_dict(),
                optimizer_parameter_signature=parameter_signature(model,full_optimizer),costs=costs,
                numpy_RNG=parent['numpy_RNG'],torch_RNG=parent['torch_RNG'],CUDA_RNG=parent['CUDA_RNG'],
                neural_midgame_resume=False,body_Adam_states_restored_from_parent=True,
                risk_Adam_states_from_selected_calibration=True,temporary_frozen_body_diagnostic=True),stream)
        summary={name:dict(step=v['step'],heldout=v['metrics']) for name,v in best.items()}
        write_json(a.output/'decoder-comparison.json',dict(schema=SCHEMA,results=summary,source_sha256=digest,
            validation='six original teacher records excluded from current raw replay and earlier direct-teacher optimization',
            heldout_proxy_is_not_native_avoidance=True,input_probe_weights_never_deployed=True,
            temporary_freeze_purpose='identify sufficient risk information at the current interfaces;not final learning protocol',
            goal_achieved=False))
        del dataset,heads,optimizers,full_optimizer,parent,calibrated,payload;gc.collect();torch.cuda.empty_cache()
        model.eval().requires_grad_(False)
        from fly_touhou.thrl_job_paths import configure_native_temp
        from fly_touhou.owned_worker_affinity import worker_cpu_pool
        from fly_touhou.continuous_no_bomb_first_contact import evaluate_no_bomb,make_no_bomb_policy
        os.environ['FLY_TOUHOU_DESKTOP_NAME']='fly-touhou-job-DN-calibration-'+str(process.pid)
        configure_native_temp(a.output)
        with parameter_path.open('rb') as stream:checkpoint_digest=hashlib.file_digest(stream,'sha256').hexdigest()
        policy=make_no_bomb_policy(torch,model,family='deep',episodes=20,movement_readout='risk_only')
        launch['native_started']=True;write_json(a.output/'launch.json',launch)
        report=evaluate_no_bomb(torch,policy,environment,a.output/'frozen-DN-risk-only20',
            checkpoint_sha256=checkpoint_digest,seed=a.evaluation_seed,episodes=20,workers=20,
            check_stop=guard,worker_cpus=worker_cpu_pool(cpus),
            progress=lambda row:progress('frozen_calibrated_DN_evaluation',evaluation=row))
        write_json(a.output/'result.json',dict(schema=SCHEMA,status='completed',costs=costs,
            decoder_comparison=summary,native_report=report,goal_achieved=False,
            full_joint_training_and_independent_confirmation_still_required=True,temporary_frozen_body_diagnostic=True,
            full_game_training_started=False,app_goal_remains_active=True,elapsed_seconds=time.monotonic()-started))
        launch['status']='completed';write_json(a.output/'launch.json',launch);progress('diagnostic_completed')
    except BaseException as error:
        write_json(a.output/'failure.json',dict(schema=SCHEMA,error=repr(error),traceback=traceback.format_exc(),
            costs=costs,automatic_retry=False,app_goal_remains_active=True,elapsed_seconds=time.monotonic()-started))
        raise


if __name__=='__main__':main()
