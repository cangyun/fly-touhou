"""One complete six-lane whole-brain supervised epoch;NoTorch plan by default.

Only --execute-cuda constructs the graph/model/optimizer. No native,teacher,
PPO,return or value target is created. Fixed held episodes are replayed from
their own row0 with the final theta after training,not the training activity.
"""
from __future__ import annotations
import argparse
import copy
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
SCHEMA='continuous-complete-rule-episode-pretraining-entry-v1'


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('training-source','manifest','visual-manifest','connectome','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--training-source-sha256',required=True)
    p.add_argument('--manifest-sha256',required=True)
    p.add_argument('--visual-manifest-sha256',required=True)
    p.add_argument('--max-seconds',type=int,required=True)
    p.add_argument('--execute-cuda',action='store_true')
    a=p.parse_args(argv)
    if a.max_seconds<1:p.error('Positive finite wall budget required')
    for name in ('training_source_sha256','manifest_sha256','visual_manifest_sha256'):
        value=getattr(a,name)
        if len(value)!=64 or any(c not in '0123456789abcdef' for c in value):p.error('Explicit artifact SHA256 required:'+name)
    return a


def plan(a):
    from fly_touhou.continuous_rule_episode_pretraining import inspect_fixed8
    data=inspect_fixed8(a.manifest,a.visual_manifest,expected_sha256=a.manifest_sha256,
        expected_visual_sha256=a.visual_manifest_sha256)
    train=[s for s in data['specs'] if s['role']=='train']
    return dict(schema=SCHEMA,status='prepared_not_executed',Torch_imported='torch' in sys.modules,
        NumPy_imported='numpy' in sys.modules,source=str(a.training_source),source_sha256=a.training_source_sha256,
        output=str(a.output),data=data,execution_ready=data['ready'],
        epoch=dict(epochs=1,real_train_lanes=6,physical_neural_width=16,padding_lanes=10,K=8,BPTT=16,
            source_rows=sum(s['rows'] for s in train),max_lane_rows=max(s['rows'] for s in train),
            source_order='each original row0->natural terminal;no stage curriculum or truncated prefixes',
            activity='own continuous activity across Miss/stage/packets;detach each16;no teacher state',
            post_update_history='carried detached preceding-theta activity is a finite-history approximation'),
        objective=dict(signed4f_raw_coefficient=1.,online_PPO_signed_coefficient_not_used=.05,
            risk='original signed query-mixture Huber(pred/8,y/8,beta.1)+.2knownBCE;not aggregate-cost or ranking',
            five_guidance='original coefficients/formulas;future/collision count0 because labels absent',
            rule='original source_rule_targets/weight;world000 missing =>zero weight',
            measured_deathbomb='original V5 timer2..8/state4/stock mask',
            PPO=False,rewards=False,values=False,critics_trainable=False),
        optimizer=dict(stage_LR=.0003,global_grad_clip=.5,source_moments_preserved=True,
            critic_parameters_moments_and_group_preserved=True,critic_grad='always None',
            empty_label_window='advance full history only;no Adam;not row filtering'),
        checkpoint=dict(every_completed_Adam=128,atomic_replace=True,final=True,
            activity_shape='6 x all138639',source_cursors=True,RNG=True,
            resume='own saved complete state+cursor only;no arbitrary late-row zero start'),
        held=dict(episodes=2,rows=data['held_rows'],final_theta_own_row0_history=True,
            fit_held_not_independent_native=True,checkpoint_selection=False),
        raw_or_explicit_normalized_risk_source=True,FP32=True,TF32=False,
        max_seconds=a.max_seconds,RAM_or_VRAM_caps=None,CUDA_started=False,
        native_actions=0,teacher_queries=0,core_forwards=0,optimizer_instances=0)


def execute(a,prepared):
    if 'torch' in sys.modules:raise RuntimeError('Fresh executor required;CVD before Torch')
    if not prepared['execution_ready']:raise RuntimeError('Every visual history must have a real complete COMMITTED marker')
    if a.output.exists():raise FileExistsError('Fresh output required;resume with a saved checkpoint into a fresh directory')
    import psutil
    current=psutil.Process();allowed=current.cpu_affinity()
    current.cpu_affinity([18,19] if {18,19}<=set(allowed) else allowed[-2:])
    if os.name=='nt':current.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    os.environ.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    a.output.mkdir(parents=True);begun=time.monotonic()
    from fly_touhou.thrl_durable_records import write_json
    write_json(a.output/'launch.json',dict(prepared,PID=os.getpid(),create_time=current.create_time(),
        started_UTC=datetime.now(timezone.utc).isoformat(),CPU_affinity=current.cpu_affinity()))
    model=optimizer=activity=None;counts={};cursors=[];counter_base={};algorithm=provenance=None
    transaction_complete=True;last_checkpoint=None;last_saved_boundary=None
    def guard():
        if (a.output/'STOP').exists():raise RuntimeError('Operator STOP')
        if time.monotonic()-begun>a.max_seconds:raise RuntimeError('Finite supervised-epoch wall budget reached')
    def progress(phase,**kwargs):
        write_json(a.output/'progress.json',dict(schema=SCHEMA,phase=phase,costs=copy.deepcopy(counts),
            cursors=copy.deepcopy(cursors),elapsed_seconds=time.monotonic()-begun,**kwargs))
    def atomic_torch(path,payload):
        temporary=path.with_suffix(path.suffix+'.tmp')
        with temporary.open('wb') as stream:torch.save(payload,stream);stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path)
    def save(tag,*,epoch_complete=False):
        nonlocal last_checkpoint,last_saved_boundary
        if critic_identity(model,optimizer)!=critic_before:raise RuntimeError('Critic parameter/Adam identity changed')
        counters=copy.deepcopy(counter_base)
        counters['complete_rule_supervised_Adam']=counters.get('complete_rule_supervised_Adam',0)+counts['optimizer_updates']
        payload=training_checkpoint_payload(model,optimizer,algorithm=algorithm,counters=counters,provenance=provenance)
        if normalized:tag_normalized_payload(payload,normalization_spec)
        payload['offline_resume']=dict(schema='continuous-sixlane-complete-history-resume-v1',
            data_binding=data_binding,cursors=list(cursors),activity=activity.detach().cpu().clone(),
            costs=copy.deepcopy(counts),counter_base=copy.deepcopy(counter_base),epoch_complete=epoch_complete,
            history='own row0-origin state;TBPTT16 detached preceding-theta carry;not arbitrary zero reset',
            native_mid_game_resume=False)
        payload['recurrent_activity_saved']=True;payload['offline_activity_only']=True
        path=a.output/f'{tag}-training.pt';atomic_torch(path,payload);last_checkpoint=str(path)
        last_saved_boundary=(counts['optimizer_updates'],tuple(cursors))
        return payload
    try:
        import torch
        torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(False)
        from fly_touhou.continuous_rule_episode_pretraining import (RuleEpisodeReader,assemble_window,
            supervised_forward_window,scalar_terms,construct_source,critic_identity,evaluate_fit_held)
        from fly_touhou.continuous_connectome_training import training_checkpoint_payload
        from fly_touhou.continuous_normalized_risk_coordinate import tag_normalized_payload
        from fly_touhou.fly_gru_aligned_guidance import DEFAULT_CONFIG
        from fly_touhou.whole_brain_ppo import clip_gradients_fp64
        guard();source_sha=sha(a.training_source)
        if source_sha!=a.training_source_sha256:raise ValueError('Caller source checkpoint bytes differ')
        source=torch.load(a.training_source,map_location='cpu',weights_only=True)
        model,optimizer,inherited,normalized=construct_source(source,a.connectome)
        normalization_spec=source['model_payload'].get('risk_input_normalization')
        train_specs=[s for s in prepared['data']['specs'] if s['role']=='train']
        held_specs=[s for s in prepared['data']['specs'] if s['role']=='readout_fit_held']
        train_readers=[RuleEpisodeReader(s,check_stop=guard) for s in train_specs]
        held_readers=[RuleEpisodeReader(s,check_stop=guard) for s in held_specs]
        for reader in (*train_readers,*held_readers):
            if (reader.identity['actual_perception_source']!=source['model_payload']['perception_provenance']
                    or reader.identity['actual_statistics']!=source['model_payload']['statistics_provenance']):
                raise ValueError('Encoded visual frontend differs from exact frozen source perception/statistics')
        data_binding=dict(manifest_sha256=prepared['data']['manifest_sha256'],
            visual_manifest_sha256=prepared['data']['visual_manifest_sha256'],
            train=[{k:s[k] for k in ('episode','rows','source_index_sha256','label_index_sha256','visual_index_sha256')} for s in train_specs],
            held=[{k:s[k] for k in ('episode','rows','source_index_sha256','label_index_sha256','visual_index_sha256')} for s in held_specs])
        defaults=dict(optimizer_updates=0,optimizer_step_attempts=0,backward_passes=0,
            source_rows_consumed=0,training_real_rows=0,completed_windows=0,state_only_windows=0,
            forward_real_rows=0,forward_physical_lanes=0,checkpoint_recompute_physical_lanes=0,
            risk_valid_scene_updates=0,actual_native_actions=0,teacher_queries=0)
        resume=source.get('offline_resume')
        if resume is not None:
            if resume['schema']!='continuous-sixlane-complete-history-resume-v1' or resume['data_binding']!=data_binding:
                raise ValueError('Pure offline resume must use its identical complete source histories')
            activity=resume['activity'].to(model.device);cursors=list(resume['cursors'])
            counts={**defaults,**resume['costs']};counter_base=copy.deepcopy(resume['counter_base'])
            if activity.shape!=(6,model.core.n_neurons) or activity.dtype!=torch.float32 or not bool(torch.isfinite(activity).all()):
                raise ValueError('Own full six-lane continuous state required for late cursors')
            if any(type(c) is not int or not 0<=c<=r.rows for c,r in zip(cursors,train_readers)) or len(cursors)!=6:
                raise ValueError('Saved exact per-source cursors required')
        else:
            activity=model.initial_state(6).activity;cursors=[0]*6;counts=defaults;counter_base=copy.deepcopy(inherited)
        critic_before=critic_identity(model,optimizer)
        algorithm=copy.deepcopy(source['algorithm'])
        algorithm.update(phase='complete_rule_episode_supervised_pretraining',
            complete_rule_pretraining=dict(schema=SCHEMA,data_binding=data_binding,epochs=1,BPTT=16,K=8,
                physical_width=16,train_lanes=6,learning_rate=.0003,global_clip=.5,
                risk_coefficient=1.,online_PPO_risk_coefficient=.05,guidance=asdict(DEFAULT_CONFIG),
                future_and_collision_targets_absent=True,critic_update=False,
                optimizer_group_learning_rates={g.get('name',str(i)):g['lr'] for i,g in enumerate(optimizer.param_groups)}))
        algorithm['optimizer_group_learning_rates']={g.get('name',str(i)):g['lr'] for i,g in enumerate(optimizer.param_groups)}
        provenance=dict(parent=dict(path=str(a.training_source),sha256=source_sha,counters=inherited,
                algorithm=copy.deepcopy(source['algorithm'])),
            data_binding=data_binding,source_behavior='original ResourceV10 trajectories;no neural teacher execution',
            entry_sha256=sha(__file__),helper_sha256=sha(ROOT/'src/fly_touhou/continuous_rule_episode_pretraining.py'),
            normalized_risk=normalized,critic_initial_identity=critic_before,
            RNG_and_dynamic_source_Adam_restored=True,states_cross_Miss_and_stage=True,
            post_update_history_is_detached_preceding_theta_approximation=True,native_midgame_resume=False)
        del source
        parameters=[p for p in model.parameters() if p.requires_grad]
        model.train();progress('training_complete_histories')
        while any(c<r.rows for c,r in zip(cursors,train_readers)):
            guard();data,used=assemble_window(train_readers,cursors)
            optimizer.zero_grad(set_to_none=True)
            result=supervised_forward_window(model,data,activity,counts,use_checkpoint=True)
            terms=scalar_terms(result)
            if not bool(torch.isfinite(result['loss'])):raise FloatingPointError('Nonfinite supervised full-brain loss')
            labelled=terms['risk_valid_scenes']>0 or any(v>0 for v in terms['guidance_counts'].values())
            if labelled:
                guard();result['loss'].backward();counts['backward_passes']+=1
                if any(p.grad is not None for p in model.critic.parameters()):raise RuntimeError('Critic gradient must remain None')
                norm=clip_gradients_fp64(parameters,.5);guard()
                transaction_complete=False;counts['optimizer_step_attempts']+=1
                optimizer.step();counts['optimizer_updates']+=1
                if any(not bool(torch.isfinite(p).all()) for p in parameters):raise FloatingPointError('Nonfinite learned parameter')
                transaction_complete=True
                counts['training_real_rows']+=sum(used);counts['risk_valid_scene_updates']+=int(terms['risk_valid_scenes'])
                terms['preclip_global_grad_norm']=norm
            else:counts['state_only_windows']+=1
            # Commit only a completed update/state-only window. An interrupted
            # pending window never advances saved cursors or saved activity.
            activity=result['following_activity'].detach();cursors=[c+n for c,n in zip(cursors,used)]
            counts['source_rows_consumed']+=sum(used);counts['completed_windows']+=1
            with (a.output/'history.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(dict(window=counts['completed_windows'],Adam=counts['optimizer_updates'],
                    cursors=cursors,source_rows=sum(used),labelled=labelled,terms=terms),allow_nan=False)+'\n')
            if labelled and counts['optimizer_updates']%128==0:save('last-completed')
            if counts['completed_windows']==1 or counts['completed_windows']%16==0:progress('training_complete_histories',learning=terms)
            del result,data
        final=save('final',epoch_complete=True)
        atomic_torch(a.output/'final-parameters.pt',final['model_payload']);del final
        progress('frozen_fit_held_own_row0_replay')
        held=evaluate_fit_held(model,held_readers,check_stop=guard,
            progress=lambda row:progress('frozen_fit_held_own_row0_replay',held=row))
        write_json(a.output/'fit-held-final.json',held)
        if critic_identity(model,optimizer)!=critic_before:raise RuntimeError('Critic changed during offline stage')
        write_json(a.output/'result.json',dict(schema=SCHEMA,status='completed',costs=counts,
            complete_epoch=True,source_cursors=dict(zip((s['episode'] for s in train_specs),cursors)),
            train_rows=prepared['data']['train_rows'],fit_held=held,algorithm=algorithm,provenance=provenance,
            critic_parameters_and_Adam_bitwise_preserved=True,critic_gradient_always_None=True,
            final_training=dict(path=str(a.output/'final-training.pt'),sha256=sha(a.output/'final-training.pt')),
            final_parameters=dict(path=str(a.output/'final-parameters.pt'),sha256=sha(a.output/'final-parameters.pt')),
            elapsed_seconds=time.monotonic()-begun,native_development=None,native_actions=0,PPO_updates=0))
        progress('completed')
    except BaseException as error:
        checkpoint_error=None
        if (model is not None and optimizer is not None and algorithm is not None and transaction_complete
                and last_saved_boundary!=(counts['optimizer_updates'],tuple(cursors))):
            try:save('stopped-after-completed-window',epoch_complete=all(c==r.rows for c,r in zip(cursors,train_readers)))
            except BaseException as second:checkpoint_error=repr(second)
        write_json(a.output/'failure.json',dict(schema=SCHEMA,error=repr(error),costs=counts,cursors=cursors,
            optimizer_transaction_complete=transaction_complete,latest_recoverable_checkpoint=last_checkpoint,
            checkpoint_error=checkpoint_error,pending_window_not_committed=True,
            native_actions=0,automatic_retry=False));raise
    finally:
        write_json(a.output/'training-cleanup.json',dict(training_closed=True,close_errors=[],native_worlds_created=0))
        model=optimizer=activity=None


def main(argv=None):
    a=parse_args(argv);prepared=plan(a)
    if not a.execute_cuda:print(json.dumps(prepared,ensure_ascii=False,indent=2));return 0
    execute(a,prepared);return 0


if __name__=='__main__':raise SystemExit(main())
