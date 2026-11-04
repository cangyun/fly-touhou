"""B1 CUDA imitation calibration from sealed, provenance-bound native data.

No native control or teacher takeover; preserves the B1 observation/core. The
output is a zero-PPO checkpoint, never a claim of learned native avoidance.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from copy import deepcopy
from dataclasses import asdict

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.atomic_json_io import atomic_write_json as write
from fly_touhou.thrl_study_io import read_shared_json as read,utc_now


def run(args):
    import numpy as np
    import torch
    from fly_touhou.thrl_model import MOActorCritic
    from fly_touhou.thrl_policy_actions import teacher_cross_entropy,policy_logits
    from fly_touhou.thrl_teacher_training import grouped_teacher_split,TeacherPretrainingConfig
    from fly_touhou.thrl_control import ControlCenter
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    torch.set_num_threads(2);random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    args.output.mkdir(parents=True,exist_ok=False)
    import psutil
    identity=dict(pid=os.getpid(),create_time=psutil.Process().create_time())
    write(args.output/'launch.json',dict(**identity,utc=utc_now(),CUDA_required=True,desktop_input=False))
    if args.template_config:
        template=read(args.template_config)
    else:
        ctl=ControlCenter(ROOT/'artifacts/engineering/training-control-center-v1')
        template=read(ctl.config['template_configs']['B1']['path'])
    initial=args.warm_checkpoint or Path(template['warm_start_checkpoint'])
    initial_sha=hashlib.sha256(initial.read_bytes()).hexdigest()
    if args.warm_checkpoint is None and initial_sha!=template['warm_start_sha256']:
        raise ValueError('B1 common initializer changed')
    from fly_touhou.thrl_demonstration_sources import demonstration_sources
    source_folders=[args.demonstrations,*args.additional_demonstrations]
    arrays,sources,source_schema=demonstration_sources(source_folders);source=sources[0]
    if source_schema!=template['input_schema']:
        raise ValueError('Teacher input schema differs from B1')
    payload=torch.load(initial,map_location='cpu',weights_only=False)
    model_config=dict(payload['model_config'])
    if model_config.get('object_geometry_embedding'):args.geometry_embedding=True
    if args.geometry_embedding:model_config['object_geometry_embedding']=True
    model=MOActorCritic(**model_config).cuda()
    initial_state=deepcopy(payload['model'])
    if args.geometry_embedding:
        key='object_attention.object_encoder.0.weight'
        old=initial_state[key];new=torch.zeros_like(model.state_dict()[key],device='cpu')
        if old.shape[1] not in (20,28) or new.shape[1]!=28:raise ValueError('Unexpected B1 geometry migration')
        new[:,:old.shape[1]]=old;initial_state[key]=new
        payload.update(model=initial_state,model_config=model.model_config())
        template['model_overrides']['object_geometry_embedding']=True
        torch.save(payload,args.output/'geometry_initial.pt')
    model.load_state_dict(initial_state,strict=True)
    if not model.is_recurrent or model.attention!='both':raise ValueError('B1 GRU plus both attention required')
    template.update(warm_start_checkpoint=str(initial),warm_start_sha256=initial_sha,
        old_checkpoints_used=bool(args.warm_checkpoint),new_initialization=not bool(args.warm_checkpoint))
    t,w=arrays['features'].shape[:2]
    cfg=TeacherPretrainingConfig(samples=t*w,workers=w,epochs=args.epochs,seq_len=args.seq_len,
        seqs_per_batch=args.batch_sequences,validation_enabled=True,validation_seed=args.seed,
        learning_rate=args.learning_rate,mirror_probability=args.mirror_probability,
        context_mode='episode-refresh' if args.continuous_context else 'chunk-zero',max_grad_norm=1.)
    split=grouped_teacher_split(arrays['native_seeds'],arrays['native_episodes'],cfg)
    if not split['available']:raise ValueError('Native-seed holdout unavailable')
    write(args.output/'split.json',split)
    started=time.monotonic();device=torch.device('cuda:0')
    data={}
    for k in ('features','objects','maps','targets','dones'):
        a=arrays[k];cache_dtype=(torch.float16 if k=='maps' and args.map_cache_dtype=='float16' else torch.bool if k=='dones' else torch.float32)
        if k=='maps' and args.map_cache_device=='mmap':
            from fly_touhou.thrl_map_replay_cache import build_map_cache,close_map_sources,open_verified_cache
            dtype=np.float16 if args.map_cache_dtype=='float16' else np.float32
            if args.existing_map_cache:
                if not args.existing_map_cache_sha256:raise ValueError('Explicit replay cache identity required')
                cache=open_verified_cache(args.existing_map_cache,a.shape,dtype=dtype,sha256=args.existing_map_cache_sha256)
            else:
                cache=build_map_cache(a,args.output/'map-replay-cache.dat',dtype=dtype,
                    release_source_views=True,
                    on_progress=lambda row:write(args.output/'map_cache_progress.json',dict(status='preparing',**row)))
            data[k]=torch.from_numpy(cache);close_map_sources(a)
            continue
        cache_device='cpu' if k=='maps' and args.map_cache_device=='cpu' else device
        data[k]=torch.empty((w*t,*a.shape[2:]),dtype=cache_dtype,device=cache_device)
        for world in range(w):
            data[k][world*t:(world+1)*t].copy_(torch.from_numpy(a[:,world].copy()))
    movement_margins=movement_trusted=movement_actions=movement_outcomes=None
    if args.movement_coefficient:
        if args.movement_targets is None:raise ValueError('Explicit modeled movement target provenance required')
        margins=[];trust=[];fact_actions=[];fact_outcomes=[];target_paths=[args.movement_targets,*args.additional_movement_targets]
        if len(target_paths)!=len(source_folders):raise ValueError('One movement target bundle per native pool required')
        for target_path,report in zip(target_paths,sources):
            with np.load(target_path) as bundle:
                if int(bundle['frames'])!=t or len(bundle['valid'])!=report['samples']:raise ValueError('Movement targets belong to another native dataset')
                margins.append(bundle['margins'])
                if args.safety_mode=='receding-step-v2' and 'trusted_horizons' not in bundle:
                    raise ValueError('Receding-step loss requires explicit per-horizon target provenance')
                trust.append(bundle['trusted_horizons'] if args.safety_mode=='receding-step-v2' else bundle['valid'])
                fact_actions.append(bundle['actions'] if 'actions' in bundle else np.full(report['samples'],-1,np.int64))
                fact_outcomes.append(bundle['outcomes'] if 'outcomes' in bundle else np.full(report['samples'],-1,np.int8))
        movement_margins=torch.tensor(np.concatenate(margins),device=device)
        movement_trusted=torch.tensor(np.concatenate(trust),device=device)
        movement_actions=torch.tensor(np.concatenate(fact_actions),device=device)
        movement_outcomes=torch.tensor(np.concatenate(fact_outcomes),device=device)
    for name,p in model.named_parameters():p.requires_grad_(not name.startswith(('critic_trunks.','critic_heads.')))
    trainable=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.Adam(trainable,lr=args.learning_rate)
    train=list(map(tuple,split['train']));heldout=list(map(tuple,split['heldout']))
    rng=np.random.default_rng(args.seed);steps=0;examples=0;history=[];best=float('inf');stale=0
    context_cache={}
    def refresh_context(mirrored=False):
        if not args.continuous_context:return
        from fly_touhou.thrl_bc_context import recurrent_context_cache
        def fetch(world,start,stop):return {k:(data[k][world*t+start:world*t+stop].to(device=device,dtype=torch.float32) if k=='maps' else data[k][world*t+start:world*t+stop]) for k in data}
        context_cache[mirrored]=recurrent_context_cache(model,fetch,w,t,mirrored=mirrored)
    def forward(desc):
        ix=torch.tensor([world*t+start for world,start in desc],device=device)[:,None]+torch.arange(args.seq_len,device=device)[None]
        f,o,y=(data[k][ix] for k in ('features','objects','targets'))
        map_ix=ix.cpu() if data['maps'].device.type=='cpu' else ix
        m=data['maps'][map_ix].to(device=device,dtype=torch.float32)
        mirror_selected=torch.zeros(len(desc),device=device,dtype=torch.bool)
        if model.training and args.mirror_probability:
            from fly_touhou.thrl_augmentation import horizontal_mirror
            mirror_selected=torch.rand(len(desc),device=device)<args.mirror_probability
            f,m,o,y=horizontal_mirror(f,m,o,y,mirror_selected)
        hidden=torch.zeros(1,len(desc),model.gru_hidden_size,device=device)
        if context_cache:
            hidden=context_cache[False][ix[:,0]].unsqueeze(0)
            if mirror_selected.any():hidden=torch.where(mirror_selected[None,:,None],context_cache[True][ix[:,0]][None],hidden)
        logits,_=model.forward_sequence(f,m,
            hidden,data['dones'][ix],objects=o)
        margins=trusted=actual_actions=actual_outcomes=None
        if movement_margins is not None:
            margins=movement_margins[ix]
            trusted=movement_trusted[ix]
            actual_actions=movement_actions[ix];actual_outcomes=movement_outcomes[ix]
            if mirror_selected.any():
                from fly_touhou.thrl_augmentation import mirror_action_indices
                margins=margins.clone()
                margins[mirror_selected]=margins[mirror_selected][...,mirror_action_indices()[:18]]
                # Mirrored states are augmentation, not actually observed
                # native transitions. Keep hard collision facts on raw rows.
                actual_outcomes=actual_outcomes.clone();actual_outcomes[mirror_selected]=-1
        return policy_logits(logits),y.reshape(-1,19),margins,trusted,actual_actions,actual_outcomes
    def evaluate(desc):
        model.eval();ce=entropy=correct=dense_sum=unsafe4_sum=unsafe12_sum=collision_sum=0.;n=dense_n=critical4_n=critical12_n=collision_n=0
        with torch.inference_mode():
            for start in range(0,len(desc),args.batch_sequences):
                x,y,margins,trusted,actual_actions,actual_outcomes=forward(desc[start:start+args.batch_sequences]);legal=torch.isfinite(x)
                y=y*legal;y=y/y.sum(-1,keepdim=True);logs=x.log_softmax(-1).masked_fill(~legal,0.)
                ce+=float(-(y*logs).sum());entropy+=float(-(y*y.clamp_min(1e-20).log()).sum())
                correct+=int((x.argmax(-1)==y.argmax(-1)).sum());n+=len(x)
                if margins is not None:
                    from fly_touhou.thrl_dodge_supervision import dense_safety_loss,receding_step_safety_loss
                    loss_function=receding_step_safety_loss if args.safety_mode=='receding-step-v2' else dense_safety_loss
                    dense,detail=loss_function(x,margins,trusted);dense_n+=detail['states'];dense_sum+=float(dense)*detail['states']
                    a,b=detail.get('critical4_states',0),detail.get('critical12_states',0)
                    critical4_n+=a;critical12_n+=b
                    unsafe4_sum+=(detail.get('predicted_unsafe4_mass') or 0)*a
                    unsafe12_sum+=(detail.get('predicted_unsafe12_mass') or 0)*b
                    from fly_touhou.thrl_movement_learning import collision_action_loss
                    native,nc=collision_action_loss(x,actual_actions,actual_outcomes)
                    collision_sum+=float(native)*nc;collision_n+=nc
        result=dict(teacher_KL=(ce-entropy)/n,soft_CE=ce/n,argmax_agreement=correct/n,samples=n)
        if movement_margins is not None:
            result.update(dense_loss=dense_sum/max(1,dense_n),dense_states=dense_n,critical4_states=critical4_n,critical12_states=critical12_n,
                predicted_unsafe4_mass=unsafe4_sum/max(1,critical4_n),predicted_unsafe12_mass=unsafe12_sum/max(1,critical12_n),
                native_collision_actions=collision_n,native_collision_loss=collision_sum/max(1,collision_n),
                objective=args.teacher_weight*ce/n+args.movement_coefficient*dense_sum/max(1,dense_n)
                    +args.native_collision_coefficient*collision_sum/max(1,collision_n))
        return result
    def save(epoch,metrics):
        nonlocal best
        score=metrics['heldout'].get('objective',metrics['heldout']['teacher_KL'])
        if score>=best:return
        best=score
        state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        p=deepcopy(payload);p.update(model=state,model_config=model.model_config(),created_utc=utc_now(),
            decisions=0,candidate_optimized_decisions=0,rollout_updates=0,adam_minibatch_steps=0,
            optimizer_transaction_complete=True,
            teacher_pretrain_updates=steps,teacher_samples=t*w,
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),
            dodge_calibration=dict(epoch=epoch,BC_adam=steps,BC_examples=examples,heldout=metrics['heldout'],
                source_demonstrations=str(args.demonstrations),source_schema=source['schema'],
                input_schema=template['input_schema'],optimizer='Adam; critic frozen',
                original_B1_architecture=not args.geometry_embedding,
                B1_core_and_observation_retained=True,derived_object_geometry=args.geometry_embedding,
                native_performance_evaluated=False),optimizer_bc=optimizer.state_dict())
        temporary=args.output/'.best.tmp';torch.save(p,temporary);os.replace(temporary,args.output/'best.pt')
        write(args.output/'best.json',dict(epoch=epoch,teacher_KL=metrics['heldout']['teacher_KL'],selection_score=best,BC_adam=steps,BC_examples=examples,
            checkpoint=str(args.output/'best.pt'),sha256=hashlib.sha256((args.output/'best.pt').read_bytes()).hexdigest()))
    plan=dict(schema='b1-native-teacher-calibration-v1',utc=utc_now(),device=str(device),seed=args.seed,
        model_config=model.model_config(),parameter_counts=model.parameter_counts(),initial_checkpoint=str(initial),
        source_demonstrations=str(args.demonstrations),teacher_frames=t*4,teacher_seeds=split['groups'],
        epochs=args.epochs,learning_rate=args.learning_rate,seq_len=args.seq_len,batch_sequences=args.batch_sequences,
        source_native_games_reused=True,game_instances_started=0,PPO_updates=0,teacher_execution_override=False,
        source_folders=list(map(str,source_folders)),map_cache_dtype=args.map_cache_dtype,map_cache_device=args.map_cache_device,
        safety_loss_mode=args.safety_mode,movement_coefficient=args.movement_coefficient,
        teacher_CE_weight=args.teacher_weight,native_collision_coefficient=args.native_collision_coefficient,
        movement_target_bundles=list(map(str,[args.movement_targets,*args.additional_movement_targets])),
        external_map_cache=str(args.existing_map_cache) if args.existing_map_cache else None,
        external_map_cache_sha256=args.existing_map_cache_sha256,
        map_model_precision='float32; float16 cache is quantized then converted before encoding',
        initial_checkpoint_sha256=initial_sha,inherited_BC_updates=payload.get('teacher_pretrain_updates',0))
    write(args.output/'plan.json',plan)
    write(args.output/'study_config.json',{**template,'workers':30,'teacher_pretraining':asdict(cfg)})
    def publish_progress(metrics,status='training'):
        progress=dict(status=status,stage='teacher_pretraining',**metrics,best_selection_score=best,
            teacher_epoch=metrics['epoch'],teacher_epochs=args.epochs,teacher_samples=t*w,target_teacher_samples=t*w,
            teacher_pretrain_updates=steps,teacher_examples_seen=examples,teacher_validation=metrics['heldout'],workers=30)
        write(args.output/'progress.json',progress)
        write(args.output/'study_status.json',dict(status='teacher_pretraining' if status=='training' else status,
            branch='B1',current_run=str(args.output),current_progress=progress,updated_utc=utc_now(),
            parameter_counts=model.parameter_counts(),device='cuda:0'))
    refresh_context()
    metrics=dict(epoch=0,train=evaluate(train),heldout=evaluate(heldout));initial_metrics=deepcopy(metrics);save(0,metrics)
    print(json.dumps(metrics),flush=True)
    for epoch in range(1,args.epochs+1):
        if (args.output/'STOP').exists():break
        refresh_context()
        if args.mirror_probability:refresh_context(True)
        model.train();rng.shuffle(train);loss_sum=0.;count=0
        for start in range(0,len(train),args.batch_sequences):
            x,y,margins,trusted,actual_actions,actual_outcomes=forward(train[start:start+args.batch_sequences]);loss=args.teacher_weight*teacher_cross_entropy(x,y)
            if margins is not None:
                from fly_touhou.thrl_dodge_supervision import dense_safety_loss,receding_step_safety_loss
                loss_function=receding_step_safety_loss if args.safety_mode=='receding-step-v2' else dense_safety_loss
                auxiliary,_=loss_function(x,margins,trusted)
                loss=loss+args.movement_coefficient*auxiliary
                if args.native_collision_coefficient:
                    from fly_touhou.thrl_movement_learning import collision_action_loss
                    native,_=collision_action_loss(x,actual_actions,actual_outcomes)
                    loss=loss+args.native_collision_coefficient*native
            optimizer.zero_grad(set_to_none=True);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(trainable,cfg.max_grad_norm,error_if_nonfinite=True);optimizer.step()
            steps+=1;examples+=len(x);loss_sum+=float(loss.detach())*len(x);count+=len(x)
        if epoch%args.evaluate_every==0 or epoch==args.epochs or epoch==1:
            refresh_context()
            metrics=dict(epoch=epoch,train=evaluate(train),heldout=evaluate(heldout),
                steps=steps,examples=examples,epoch_loss=loss_sum/count,wall_seconds=time.monotonic()-started,
                cuda_allocated=torch.cuda.memory_allocated(),cuda_peak=torch.cuda.max_memory_allocated())
            before=best;save(epoch,metrics);history.append(metrics)
            stale=0 if best<before-.005 else stale+1
            publish_progress(metrics)
            with (args.output/'history.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(metrics)+'\n')
            print(json.dumps(metrics),flush=True)
            if args.patience and stale>=args.patience:break
    paid_epoch=epoch;best_record=read(args.output/'best.json')
    sealed=torch.load(args.output/'best.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(sealed['model'])
    refresh_context()
    best_train,best_heldout=evaluate(train),evaluate(heldout)
    metadata=dict(validation_enabled=True,split=split,fit_claimed=False,
        fit_status='fit-diagnostic-passed' if best_heldout['teacher_KL']<=cfg.validation_max_kl else 'underfit-or-unhelpful',
        epochs=paid_epoch,best_epoch=best_record['epoch'],restored_best_epoch=best_record['epoch'],
        restored_best_BC_optimizer=True,stopped_reason=('heldout-objective-patience' if args.movement_coefficient else 'heldout-KL-patience') if paid_epoch<args.epochs else 'maximum-epochs',
        BC_adam_minibatch_steps=steps,examples_seen=examples,unique_teacher_samples=t*w,
        optimizer_parameter_names=[n for n,p in model.named_parameters() if p.requires_grad],
        final_train=best_train,final_heldout=best_heldout,initial_train=initial_metrics['train'],initial_heldout=initial_metrics['heldout'],
        counters_include_paid_updates_after_best=True,augmentation_probability=args.mirror_probability,
        critic_parameter_updates=0,chunk_initial_hidden=cfg.context_mode,
        source_folders=list(map(str,source_folders)),map_cache_dtype=args.map_cache_dtype,map_cache_device=args.map_cache_device,
        inherited_BC_updates=payload.get('teacher_pretrain_updates',0),
        movement_supervision=dict(coefficient=args.movement_coefficient,teacher_CE_weight=args.teacher_weight,
            safety_loss_mode=args.safety_mode,
            native_collision_coefficient=args.native_collision_coefficient,native_facts_on_mirrored_rows=False,
            targets=str(args.movement_targets),execution_teacher_or_physics=False,
            targets_are_modeled_not_native_counterfactual_outcomes=True),
        selection_criterion='teacher CE plus dense modeled action safety' if args.movement_coefficient else 'heldout teacher KL',
        native_performance_evaluated=False,fit_claim_scope='supervised diagnostic only')
    write(args.output/'teacher_fit_report.json',{**metadata,'epoch_history':history})
    write(args.output/'teacher_validation_split.json',split)
    sealed.update(teacher_pretrain_updates=steps,teacher_samples=t*w,teacher_pretraining={'pretraining':metadata},
        optimizer_bc_parameter_names=metadata['optimizer_parameter_names'],config={**template,'teacher_pretraining':asdict(cfg)})
    checkpoint=args.output/'teacher_pretrained.pt';torch.save(sealed,checkpoint)
    from fly_touhou.thrl_teacher_training import validated_teacher_result_fields
    result=dict(status='completed',steps=steps,examples=examples,best=best_record,
        final=metrics,wall_seconds=time.monotonic()-started,PPO_updates=0,native_performance_evaluated=False,
        checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        teacher_samples=t*w,teacher_actions_executed=sum(r.get('teacher_actions_executed',r['samples']) for r in sources),
        student_actions_executed=sum(r.get('student_actions_executed',0) for r in sources),
        teacher_labels_queried=t*w,
        teacher_collection_mode='teacher-labelled-student-state-mixture-v1' if args.additional_demonstrations else 'teacher-prefix-v1',
        source_collection_manifests=[dict(path=str(f/'demonstrations.json'),sha256=hashlib.sha256((f/'demonstrations.json').read_bytes()).hexdigest()) for f in source_folders],
        BC_adam_minibatch_steps=steps,
        inherited_PPO_decisions=0,inherited_PPO_rollout_updates=0,inherited_PPO_adam_minibatch_steps=0,
        **validated_teacher_result_fields(template,cfg,metadata,args.output))
    write(args.output/'result.json',result);publish_progress(metrics,'completed');return result


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--demonstrations',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--epochs',type=int,default=120)
    p.add_argument('--learning-rate',type=float,default=.0003);p.add_argument('--seq-len',type=int,default=16)
    p.add_argument('--batch-sequences',type=int,default=16);p.add_argument('--evaluate-every',type=int,default=5)
    p.add_argument('--seed',type=int,default=11)
    p.add_argument('--mirror-probability',type=float,default=0.)
    p.add_argument('--patience',type=int,default=0)
    p.add_argument('--continuous-context',action='store_true')
    p.add_argument('--movement-targets',type=Path)
    p.add_argument('--movement-coefficient',type=float,default=0.)
    p.add_argument('--teacher-weight',type=float,default=1.)
    p.add_argument('--safety-mode',choices=['legacy-straight-v1','receding-step-v2'],default='legacy-straight-v1')
    p.add_argument('--map-cache-device',choices=['cpu','cuda','mmap'],default='cuda')
    p.add_argument('--template-config',type=Path)
    p.add_argument('--existing-map-cache',type=Path)
    p.add_argument('--existing-map-cache-sha256')
    p.add_argument('--geometry-embedding',action='store_true')
    p.add_argument('--additional-demonstrations',type=Path,action='append',default=[])
    p.add_argument('--additional-movement-targets',type=Path,action='append',default=[])
    p.add_argument('--warm-checkpoint',type=Path)
    p.add_argument('--map-cache-dtype',choices=['float32','float16'],default='float32')
    p.add_argument('--native-collision-coefficient',type=float,default=.02)
    a=p.parse_args()
    try:run(a)
    except BaseException:
        import traceback
        if a.output.is_dir():write(a.output/'failure.json',dict(utc=utc_now(),traceback=traceback.format_exc()))
        raise
