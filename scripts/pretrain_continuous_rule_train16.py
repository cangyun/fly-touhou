"""Prepare, or explicitly execute, one complete16-train/2-fit-held epoch.

Default: JSON metadata only. --execute-cuda is never implied by a prepared
plan. No native game, teacher, PPO, reward/return/value target is created.
"""
from __future__ import annotations
import argparse
import copy
from dataclasses import asdict
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
SCHEMA='continuous-complete-rule-episode-pretraining-entry-v1'
PROTOCOL='continuous-complete-rule-train16-v1'


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('training-source','manifest','connectome','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--training-source-sha256',required=True)
    p.add_argument('--manifest-sha256',required=True)
    p.add_argument('--max-seconds',type=int,required=True)
    p.add_argument('--resume-data-state',action='store_true',help='Same unfinished16-lane epoch and exact data binding only')
    p.add_argument('--execute-cuda',action='store_true')
    a=p.parse_args(argv)
    if a.max_seconds<1:p.error('Finite positive wall budget required')
    for name in ('training_source_sha256','manifest_sha256'):
        s=getattr(a,name)
        if len(s)!=64 or any(c not in '0123456789abcdef' for c in s):p.error('Explicit artifact SHA256 required:'+name)
    return a


def plan(a):
    from fly_touhou.continuous_rule_train16_pretraining import inspect_train16
    data=inspect_train16(a.manifest,expected_sha256=a.manifest_sha256)
    train=[s for s in data['specs'] if s['role']=='train']
    return dict(schema=SCHEMA,protocol=PROTOCOL,status='prepared_not_executed',
        source=str(a.training_source),source_sha256=a.training_source_sha256,output=str(a.output),data=data,
        execution_ready=data['ready'],Torch_imported='torch' in sys.modules,NumPy_imported='numpy' in sys.modules,
        epoch=dict(epochs=1,real_train_lanes=16,physical_neural_width=16,initial_padding_lanes=0,
            naturally_finished_lanes='inactive valid=False until remaining original episodes finish;no repeat/fake rows',
            K=8,BPTT=16,source_rows=data['train_rows'],max_lane_rows=max(s['rows'] for s in train),
            source_order='every original row0 to natural terminal;no stage/Miss reset',
            carry='own activity;detach each16;preceding-theta approximation after updates'),
        data_state=dict(mode='same_unfinished_epoch_resume' if a.resume_data_state else 'new_dataset_epoch_from_row0',
            resume_explicit=a.resume_data_state,default_ignores_parent_6_or16_lane_activity_and_cursors=True,
            weights_Adam_RNG_inherited=True,arbitrary_late_row_zero_start=False),
        objective=dict(signed4f_raw_coefficient=1.,online_PPO_coefficient_not_used=.05,
            five_guidance='original coefficients/functions;new10 real4/12/24,old8 only4f',
            outcomes='unknown=-1;no invented observed collisions',source_rule='unchanged p/weight;world000 weight0',
            PPO=False,rewards=False,values=False,critic_update=False),
        optimizer=dict(stage_LR=.0003,global_grad_clip=.5,source_moments_preserved=True,
            critic18_parameters_moments_group_preserved=True,critic_grad='always None'),
        checkpoint=dict(every_completed_Adam=128,atomic_replace=True,activity_shape='16 x all138639',
            full_dynamic_Adam_RNG=True,explicit_resume_flag_required=True),
        fit_held=dict(episodes=2,rows=data['held_rows'],final_theta_own_row0_history=True,
            reuse='evaluate_fit_held:margin<=.5 classification;separate conditional/unconditional danger and Bomb',
            independent_native_test=False,checkpoint_selection=False),
        max_seconds=a.max_seconds,RAM_or_VRAM_caps=None,FP32=True,TF32=False,
        CUDA_started=False,native_actions=0,teacher_queries=0,core_forwards=0,optimizer_instances=0)


def execute(a,prepared):
    if 'torch' in sys.modules:raise RuntimeError('Fresh executor required;CVD before Torch')
    if not prepared['execution_ready']:raise RuntimeError('Complete18-episode dataset commit required')
    if a.output.exists():raise FileExistsError('Fresh output required;no implicit overwrite or resume')
    import psutil
    current=psutil.Process();allowed=current.cpu_affinity()
    current.cpu_affinity([18,19] if {18,19}<=set(allowed) else allowed[-2:])
    if os.name=='nt':current.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    os.environ.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    a.output.mkdir(parents=True);begun=time.monotonic()
    from fly_touhou.thrl_durable_records import write_json
    write_json(a.output/'launch.json',dict(prepared,PID=os.getpid(),create_time=current.create_time(),
        started_UTC=datetime.now(timezone.utc).isoformat(),CPU_affinity=current.cpu_affinity()))
    model=optimizer=activity=None;costs={};cursors=[];counter_base={};algorithm=provenance=None
    transaction_complete=True;last_checkpoint=None;last_saved_boundary=None
    def guard():
        if (a.output/'STOP').exists():raise RuntimeError('Operator STOP')
        if time.monotonic()-begun>a.max_seconds:raise RuntimeError('Finite supervised-epoch wall budget reached')
    def progress(phase,**kwargs):
        write_json(a.output/'progress.json',dict(schema=SCHEMA,protocol=PROTOCOL,phase=phase,
            costs=copy.deepcopy(costs),cursors=list(cursors),elapsed_seconds=time.monotonic()-begun,**kwargs))
    def save(tag,*,epoch_complete=False):
        nonlocal last_checkpoint,last_saved_boundary
        if critic_identity(model,optimizer)!=critic_before:raise RuntimeError('Critic parameter/Adam identity changed')
        counters=copy.deepcopy(counter_base)
        counters['complete_rule_train16_supervised_Adam']=counters.get('complete_rule_train16_supervised_Adam',0)+costs['optimizer_updates']
        payload=checkpoint_train16(model,optimizer,algorithm=algorithm,counters=counters,provenance=provenance,
            binding=binding,cursors=cursors,activity=activity,costs=costs,counter_base=counter_base,epoch_complete=epoch_complete)
        path=a.output/f'{tag}-training.pt';atomic_torch_save(path,payload);last_checkpoint=str(path)
        last_saved_boundary=(costs['optimizer_updates'],tuple(cursors))
        return payload
    try:
        import torch
        torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(False)
        from fly_touhou.continuous_rule_train16_pretraining import (Train16RuleEpisodeReader,
            assemble_train16_window,supervised_forward_window,scalar_terms,construct_source,critic_identity,
            evaluate_fit_held,data_binding,requested_data_state,checkpoint_train16,atomic_torch_save,sha)
        from fly_touhou.fly_gru_aligned_guidance import DEFAULT_CONFIG
        from fly_touhou.whole_brain_ppo import clip_gradients_fp64
        guard();source_sha=sha(a.training_source)
        if source_sha!=a.training_source_sha256:raise ValueError('Explicit source checkpoint SHA differs')
        source=torch.load(a.training_source,map_location='cpu',weights_only=True)
        model,optimizer,inherited,normalized=construct_source(source,a.connectome)
        specs=prepared['data']['specs'];train=[s for s in specs if s['role']=='train']
        held=[s for s in specs if s['role']=='readout_fit_held']
        readers=[Train16RuleEpisodeReader(s,check_stop=guard) for s in train]
        held_readers=[Train16RuleEpisodeReader(s,check_stop=guard) for s in held]
        for reader in (*readers,*held_readers):
            if (reader.identity['actual_perception_source']!=source['model_payload']['perception_provenance']
                    or reader.identity['actual_statistics']!=source['model_payload']['statistics_provenance']):
                raise ValueError('Actual frozen visual frontend/statistics differs from checkpoint')
        binding=data_binding(prepared['data'])
        defaults=dict(optimizer_updates=0,optimizer_step_attempts=0,backward_passes=0,source_rows_consumed=0,
            training_real_rows=0,completed_windows=0,state_only_windows=0,forward_real_rows=0,
            forward_physical_lanes=0,checkpoint_recompute_physical_lanes=0,risk_valid_scene_updates=0,
            actual_native_actions=0,teacher_queries=0)
        resume=requested_data_state(source,binding,resume_data_state=a.resume_data_state)
        parent_state=source.get('offline_resume') or {}
        if resume is None:
            activity=model.initial_state(16).activity;cursors=[0]*16;costs=defaults;counter_base=copy.deepcopy(inherited)
        else:
            activity=resume['activity'].to(model.device);cursors=list(resume['cursors'])
            costs={**defaults,**resume['costs']};counter_base=copy.deepcopy(resume['counter_base'])
            if activity.shape!=(16,model.core.n_neurons) or activity.dtype!=torch.float32 or not bool(torch.isfinite(activity).all()):
                raise ValueError('Complete own16-lane activity required for exact saved data cursor')
        critic_before=critic_identity(model,optimizer)
        algorithm=copy.deepcopy(source['algorithm'])
        algorithm.update(phase='complete_rule_episode_supervised_pretraining',
            complete_rule_pretraining=dict(schema=SCHEMA,protocol=PROTOCOL,data_binding=binding,epochs=1,
                BPTT=16,K=8,physical_width=16,train_lanes=16,learning_rate=.0003,global_clip=.5,
                risk_coefficient=1.,online_PPO_risk_coefficient=.05,guidance=asdict(DEFAULT_CONFIG),
                old8_future_targets_absent=True,new10_actual_horizons=[4,12,24],
                collision_targets_absent=True,critic_update=False,
                data_state_mode=prepared['data_state']['mode']))
        algorithm['optimizer_group_learning_rates']={g.get('name',str(i)):g['lr'] for i,g in enumerate(optimizer.param_groups)}
        provenance=dict(parent=dict(path=str(a.training_source),sha256=source_sha,counters=inherited,
                algorithm=copy.deepcopy(source['algorithm']),parent_data_state_schema=parent_state.get('schema'),
                parent_epoch_complete=parent_state.get('epoch_complete')),
            data_binding=binding,data_state_mode=prepared['data_state']['mode'],
            source_behavior='original ResourceV10 trajectories;no neural teacher execution',
            entry_sha256=sha(__file__),helper_sha256=sha(ROOT/'src/fly_touhou/continuous_rule_train16_pretraining.py'),
            reused_helper_sha256=sha(ROOT/'src/fly_touhou/continuous_rule_episode_pretraining.py'),
            normalized_risk=normalized,normalization_spec=copy.deepcopy(source['model_payload'].get('risk_input_normalization')),
            critic_initial_identity=critic_before,RNG_and_dynamic_source_Adam_restored=True,
            states_cross_Miss_and_stage=True,post_update_history_is_detached_preceding_theta_approximation=True,
            parent_activity_and_cursors_ignored_unless_explicit_resume=not a.resume_data_state,native_midgame_resume=False)
        del source,parent_state,resume
        parameters=[p for p in model.parameters() if p.requires_grad]
        model.train();progress('training_complete_histories')
        while any(c<r.rows for c,r in zip(cursors,readers)):
            guard();data,used=assemble_train16_window(readers,cursors);optimizer.zero_grad(set_to_none=True)
            result=supervised_forward_window(model,data,activity,costs,use_checkpoint=True);terms=scalar_terms(result)
            if not bool(torch.isfinite(result['loss'])):raise FloatingPointError('Nonfinite full-brain supervised loss')
            labelled=terms['risk_valid_scenes']>0 or any(v>0 for v in terms['guidance_counts'].values())
            if labelled:
                guard();result['loss'].backward();costs['backward_passes']+=1
                if any(p.grad is not None for p in model.critic.parameters()):raise RuntimeError('Critic gradient must remain None')
                terms['preclip_global_grad_norm']=clip_gradients_fp64(parameters,.5);guard()
                transaction_complete=False;costs['optimizer_step_attempts']+=1
                optimizer.step();costs['optimizer_updates']+=1
                if any(not bool(torch.isfinite(p).all()) for p in parameters):raise FloatingPointError('Nonfinite learned parameter')
                transaction_complete=True;costs['training_real_rows']+=sum(used)
                costs['risk_valid_scene_updates']+=int(terms['risk_valid_scenes'])
            else:costs['state_only_windows']+=1
            activity=result['following_activity'].detach();cursors=[c+n for c,n in zip(cursors,used)]
            costs['source_rows_consumed']+=sum(used);costs['completed_windows']+=1
            with (a.output/'history.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(dict(window=costs['completed_windows'],Adam=costs['optimizer_updates'],
                    cursors=cursors,source_rows=sum(used),physical_rows=len(data['valid_mask'])*16,
                    labelled=labelled,terms=terms),allow_nan=False)+'\n')
            if labelled and costs['optimizer_updates']%128==0:save('last-completed')
            if costs['completed_windows']==1 or costs['completed_windows']%16==0:
                progress('training_complete_histories',learning=terms)
            del result,data
        final=save('final',epoch_complete=True)
        atomic_torch_save(a.output/'final-parameters.pt',final['model_payload']);del final
        progress('frozen_fit_held_own_row0_replay')
        fit=evaluate_fit_held(model,held_readers,check_stop=guard,
            progress=lambda row:progress('frozen_fit_held_own_row0_replay',held=row))
        write_json(a.output/'fit-held-final.json',fit)
        if critic_identity(model,optimizer)!=critic_before:raise RuntimeError('Critic changed in offline stage')
        write_json(a.output/'result.json',dict(schema=SCHEMA,protocol=PROTOCOL,status='completed',costs=costs,
            complete_epoch=True,source_cursors=dict(zip((s['episode'] for s in train),cursors)),
            train_rows=prepared['data']['train_rows'],fit_held=fit,algorithm=algorithm,provenance=provenance,
            critic_parameters_and_Adam_bitwise_preserved=True,critic_gradient_always_None=True,
            final_training=dict(path=str(a.output/'final-training.pt'),sha256=sha(a.output/'final-training.pt')),
            final_parameters=dict(path=str(a.output/'final-parameters.pt'),sha256=sha(a.output/'final-parameters.pt')),
            elapsed_seconds=time.monotonic()-begun,native_development=None,native_actions=0,PPO_updates=0))
        progress('completed')
    except BaseException as error:
        checkpoint_error=None
        if (model is not None and optimizer is not None and algorithm is not None and transaction_complete
                and last_saved_boundary!=(costs['optimizer_updates'],tuple(cursors))):
            try:save('stopped-after-completed-window',epoch_complete=all(c==r.rows for c,r in zip(cursors,readers)))
            except BaseException as second:checkpoint_error=repr(second)
        write_json(a.output/'failure.json',dict(schema=SCHEMA,protocol=PROTOCOL,error=repr(error),costs=costs,
            cursors=cursors,optimizer_transaction_complete=transaction_complete,
            latest_recoverable_checkpoint=last_checkpoint,checkpoint_error=checkpoint_error,
            pending_window_not_committed=True,native_actions=0,automatic_retry=False));raise
    finally:
        write_json(a.output/'training-cleanup.json',dict(training_closed=True,close_errors=[],native_worlds_created=0))


def main(argv=None):
    a=parse_args(argv);prepared=plan(a)
    if not a.execute_cuda:print(json.dumps(prepared,ensure_ascii=False,indent=2));return 0
    execute(a,prepared);return 0


if __name__=='__main__':raise SystemExit(main())
