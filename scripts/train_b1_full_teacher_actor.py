"""CUDA-only temporary actor warmup from sealed full natural teacher games.

Uses one shared B1, inventory-legal executed-action CE and actual full-prefix
recurrent context. No native game, policy takeover, PPO step or permanent
encoder freeze. A following full-policy optimization phase remains required.
"""
from pathlib import Path
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
import random
import sys
import time
import traceback
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.atomic_json_io import atomic_write_json as write
from fly_touhou.thrl_study_io import read_shared_json as read,utc_now


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def imitation_batch_loss(logits,actions,legal):
    import torch
    from fly_touhou.thrl_policy_actions import policy_logits
    clean=policy_logits(logits)
    target=torch.where(legal,actions,torch.zeros_like(actions))
    losses=torch.nn.functional.cross_entropy(clean,target,reduction='none')
    return (losses*legal).sum()/legal.sum().clamp_min(1),clean,target


@contextmanager
def evaluation_scope(model):
    flags=[(p,p.requires_grad) for p in model.parameters()]
    modes=[(m,m.training) for m in model.modules()]
    model.requires_grad_(False);model.eval()
    try:yield
    finally:
        for p,flag in flags:p.requires_grad_(flag)
        for module,mode in modes:module.training=mode


def run(args):
    import numpy as np,torch,psutil
    from fly_touhou.thrl_pair_policy import model_from_config
    from fly_touhou.thrl_cached_actor_fit import begin_cached_actor_phase,forward_cached_actor
    from fly_touhou.thrl_full_teacher_data import FullTeacherDataset
    from fly_touhou.thrl_teacher_latents import TeacherLatentCache
    from fly_touhou.thrl_policy_actions import policy_logits
    if not (ROOT/'source_manifest.json').is_file():raise RuntimeError('Freeze and qualify this fitter before real training')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required;no CPU training fallback')
    if not 4<=args.minimum_epochs<=args.epochs<=100 or args.patience<1 or args.batch_sequences<1:
        raise ValueError('Explicit finite epochs,validation patience and sequence batch required')
    if sha(args.parent_checkpoint)!=args.parent_sha256:raise ValueError('Warm initializer differs')
    qualification=read(args.cuda_qualification)
    if (qualification.get('status')!='passed' or qualification.get('device')!='cuda'
            or qualification.get('checkpoint_sha256')!=args.parent_sha256
            or qualification['source_hashes']['thrl_cached_actor_fit.py']!=sha(ROOT/'src/fly_touhou/thrl_cached_actor_fit.py')):
        raise ValueError('Pinned actual-CUDA cache/gradient qualification required')
    dataset=FullTeacherDataset(args.demonstrations)
    qualification_only=bool(getattr(args,'qualification_only',False))
    if qualification_only:
        if dataset.manifest.get('numerical_contract_fixture_only') is not True:
            raise ValueError('Scratch qualification must use an explicitly marked numerical fixture')
    else:
        if dataset.manifest.get('numerical_contract_fixture_only'):
            raise ValueError('Synthetic fixtures are forbidden in deployable policy fitting')
        origin=Path(dataset.manifest['source_runtime'])/'source_manifest.json'
        if sha(origin)!=dataset.manifest['source_manifest_sha256']:
            raise ValueError('Native teacher collection source provenance differs')
    split=dataset.split(fraction=.2,seed=args.seed)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.65)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    payload=torch.load(args.parent_checkpoint,map_location='cpu',weights_only=False)
    model=model_from_config(**payload['model_config']).cuda();model.load_state_dict(payload['model'],strict=True)
    model.gru.flatten_parameters();scope=begin_cached_actor_phase(model)
    args.output.mkdir(parents=True,exist_ok=False);args.output_created=True
    write(args.output/'launch.json',dict(pid=os.getpid(),create_time=psutil.Process().create_time(),utc=utc_now(),
        CUDA_required=True,device=torch.cuda.get_device_name(),native_worlds=0,desktop_input=False))
    config=dict(payload['config'],source_runtime=str(ROOT),source_manifest_sha256=sha(ROOT/'source_manifest.json'),
        neural_training_role='temporary-full-teacher-actor-warmup',teacher_pretraining=True,
        parent_checkpoint=str(args.parent_checkpoint),parent_sha256=args.parent_sha256,
        warm_start_checkpoint=str(args.parent_checkpoint),warm_start_sha256=args.parent_sha256,
        teacher_dataset=str(args.demonstrations),teacher_dataset_sha256=dataset.manifest_sha256,
        BC_epochs=args.epochs,BC_sequence_length=args.seq_len,BC_batch_sequences=args.batch_sequences,
        BC_learning_rate=args.learning_rate,PPO_updates=0,native_worlds=0,full_policy_phase_still_required=True)
    config.update(numerical_qualification_only=qualification_only,deployment_forbidden=qualification_only)
    write(args.output/'study_config.json',config);write(args.output/'config.json',config);write(args.output/'split.json',split)
    started=time.monotonic();paid_steps=examples_seen=epoch=0;history=[];best=best_epoch=best_steps=None
    last=0.
    def stop():
        if (args.output/'STOP').exists():raise RuntimeError('Operator STOP;preserve optimizer and selected checkpoints')
    def progress(row,force=False):
        nonlocal last
        now=time.monotonic()
        if not force and now-last<5:return
        state=dict(status='teacher_pretraining',stage='full_teacher_actor_warmup',updated_utc=utc_now(),
            wall_seconds=now-started,device='cuda:0',CUDA_required=True,epoch=epoch,target_epochs=args.epochs,
            BC_paid_optimizer_steps=paid_steps,BC_examples_seen=examples_seen,unique_native_teacher_rows=dataset.rows,
            trainable_parameters=scope['trainable_parameters'],total_parameters=scope['original_full_model_parameters'],
            native_worlds=0,PPO_updates=0,cuda_allocated_mb=torch.cuda.memory_allocated()/1024**2,
            cuda_reserved_mb=torch.cuda.memory_reserved()/1024**2)
        details=dict(row)
        if 'wall_seconds' in details:details['phase_wall_seconds']=details.pop('wall_seconds')
        state.update(details)
        write(args.output/'progress.json',state)
        write(args.output/'study_status.json',dict(status=state['status'],current_run=str(args.output),current_progress=state,updated_utc=utc_now()))
        last=now
    cache=TeacherLatentCache(torch,model,dataset,on_progress=lambda row:progress(row,True),check_stop=stop)
    train,excluded_train=cache.valid_descriptors(split['train'],length=args.seq_len)
    heldout,excluded_heldout=cache.valid_descriptors(split['heldout'],length=args.seq_len)
    write(args.output/'data_report.json',dict(cache=cache.stats,split=split,
        train_chunks=len(train),heldout_chunks=len(heldout),excluded_empty_legal_label_chunks=excluded_train+excluded_heldout,
        omitted_short_tail_rows=dataset.rows-(len(train)+len(heldout)+excluded_train+excluded_heldout)*args.seq_len,
        original_archived_rows_not_deleted=True,geometry_abstention_does_not_discard_actual_imitation_labels=True))
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=args.learning_rate)
    rng=np.random.default_rng(args.seed)
    def forward(descriptors):
        batch=cache.batch(descriptors,length=args.seq_len)
        logits=forward_cached_actor(model,**{k:batch[k] for k in ('encoded','features','risk','initial_hidden','dones')})
        legal=batch['legal']
        loss,clean,target=imitation_batch_loss(logits,batch['actions'],legal)
        return loss,clean,target,legal
    def evaluate(descriptors):
        count=correct=clamped=0;total=0.
        with evaluation_scope(model),torch.no_grad():
            for first in range(0,len(descriptors),args.batch_sequences):
                stop();loss,logits,target,legal=forward(descriptors[first:first+args.batch_sequences])
                n=int(legal.sum());count+=n;total+=float(loss)*n
                correct+=int(((logits.argmax(-1)==target)&legal).sum())
                clamped+=int(((logits.gather(1,target[:,None]).squeeze(1).abs()>=80)&legal).sum())
        if not count:raise ValueError('Validation split contains no legal actual teacher actions')
        return dict(hard_teacher_CE=total/count,executed_action_top1_agreement=correct/count,legal_labels=count,
                    teacher_action_at_policy_logit_limit_fraction=clamped/count,not_native_performance=True)
    def save(path,*,selected_epoch,selected_steps,selected_metrics):
        state=dict(schema='B1-full-teacher-temporary-actor-warmup-v1',created_utc=utc_now(),
            model_config=model.model_config(),model={k:v.detach().cpu() for k,v in model.state_dict().items()},
            BC_optimizer=optimizer.state_dict(),config=config,scope=scope,
            parent_checkpoint=str(args.parent_checkpoint),parent_sha256=args.parent_sha256,
            teacher_manifest=str(dataset.manifest_path),teacher_manifest_sha256=dataset.manifest_sha256,
            selected_epoch=selected_epoch,selected_BC_optimizer_steps=selected_steps,paid_BC_optimizer_steps=paid_steps,
            BC_examples_seen=examples_seen,selected_heldout=selected_metrics,PPO_updates=0,
            decisions=0,candidate_optimized_decisions=0,full_policy_optimization_still_required=True,
            inherited_parent_lifetime_optimized_decisions=payload.get('lifetime_decisions'),
            PPO_optimizer_missing_intentionally=True,new_PPO_optimizer_required_after_BC=True,
            numerical_qualification_only=qualification_only,deployment_forbidden=qualification_only,
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=rng.bit_generator.state)
        temporary=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
        torch.save(state,temporary);os.replace(temporary,path)
    cache.refresh_contexts();initial_train=evaluate(train);initial_heldout=evaluate(heldout)
    best=initial_heldout['hard_teacher_CE'];best_epoch=best_steps=0;stale=0
    save(args.output/'best_actor_warmup.pt',selected_epoch=0,selected_steps=0,selected_metrics=initial_heldout)
    write(args.output/'initial_fit.json',dict(train=initial_train,heldout=initial_heldout,scope=scope,teacher_outcomes=dataset.manifest['outcomes']))
    for epoch in range(1,args.epochs+1):
        stop();cache.assert_valid();cache.refresh_contexts()
        order=list(train);rng.shuffle(order);model.train()
        model.base.feature_proj.eval();model.base.map_encoder.eval();model.base.object_attention.eval()
        sum_loss=count=0
        for first in range(0,len(order),args.batch_sequences):
            stop();loss,logits,target,legal=forward(order[first:first+args.batch_sequences])
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite full-teacher actor loss')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
            optimizer.step();paid_steps+=1;n=int(legal.sum());examples_seen+=n
            sum_loss+=float(loss.detach())*n;count+=n
            progress(dict(phase='fitting_actor',latest_training_CE=float(loss.detach())))
        cache.assert_valid();cache.refresh_contexts();train_metrics=evaluate(train);heldout_metrics=evaluate(heldout)
        row=dict(epoch=epoch,training_minibatch_CE=sum_loss/count,train=train_metrics,heldout=heldout_metrics,
                 paid_BC_optimizer_steps=paid_steps,BC_examples_seen=examples_seen)
        history.append(row)
        with (args.output/'fit_epochs.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(row)+'\n')
        if heldout_metrics['hard_teacher_CE']<best:
            significant=best-heldout_metrics['hard_teacher_CE']>=.001
            best=heldout_metrics['hard_teacher_CE'];best_epoch=epoch;best_steps=paid_steps
            save(args.output/'best_actor_warmup.pt',selected_epoch=epoch,selected_steps=paid_steps,selected_metrics=heldout_metrics)
            stale=0 if significant else stale+1
        else:stale+=1
        write(args.output/'fit_report.json',dict(status='fitting',epoch_history=history,initial_heldout=initial_heldout,
            best_epoch=best_epoch,best_heldout_CE=best,paid_BC_optimizer_steps=paid_steps,selected_BC_optimizer_steps=best_steps))
        progress(dict(phase='validated_epoch',train=train_metrics,heldout=heldout_metrics,best_epoch=best_epoch),True)
        if epoch>=args.minimum_epochs and stale>=args.patience:break
    selected=torch.load(args.output/'best_actor_warmup.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(selected['model'],strict=True);optimizer.load_state_dict(selected['BC_optimizer'])
    cache.rebind_after_best_restore();cache.refresh_contexts();final_heldout=evaluate(heldout)
    save(args.output/'actor_warmup.pt',selected_epoch=best_epoch,selected_steps=best_steps,selected_metrics=final_heldout)
    result=dict(status='completed',schema='B1-full-natural-teacher-actor-warmup-result-v1',utc=utc_now(),
        policy=str(args.output/'actor_warmup.pt'),policy_sha256=sha(args.output/'actor_warmup.pt'),
        original_teacher_rows=dataset.rows,teacher_complete_episodes=len(dataset.records),teacher_outcomes=dataset.manifest['outcomes'],
        epochs_executed=epoch,selected_epoch=best_epoch,paid_BC_optimizer_steps=paid_steps,
        selected_BC_optimizer_steps=best_steps,BC_examples_seen=examples_seen,initial_heldout=initial_heldout,
        selected_heldout=final_heldout,CUDA=True,actual_device_name=torch.cuda.get_device_name(),
        trainable_parameters=scope['trainable_parameters'],total_parameters=scope['original_full_model_parameters'],
        original_encoder_risk_and_critic_unchanged=True,PPO_updates=0,native_worlds_started=0,
        full_policy_optimization_still_required=True,native_efficacy_not_yet_tested=True,avoidance_goal_achieved=False,
        numerical_qualification_only=qualification_only,deployment_forbidden=qualification_only,
        wall_seconds=time.monotonic()-started)
    write(args.output/'result.json',result)
    progress(dict(status='completed',phase='actor_warmup_complete',selected_heldout=final_heldout,full_policy_optimization_still_required=True),True)
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for name in ('parent-checkpoint','cuda-qualification','demonstrations','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--parent-sha256',required=True);p.add_argument('--epochs',type=int,default=12)
    p.add_argument('--minimum-epochs',type=int,default=4);p.add_argument('--patience',type=int,default=4)
    p.add_argument('--seq-len',type=int,default=16);p.add_argument('--batch-sequences',type=int,default=64)
    p.add_argument('--learning-rate',type=float,default=1e-4);p.add_argument('--seed',type=int,default=31)
    args=p.parse_args()
    try:run(args)
    except BaseException:
        if getattr(args,'output_created',False):write(args.output/'failure.json',dict(utc=utc_now(),traceback=traceback.format_exc(),no_automatic_restart=True))
        raise
