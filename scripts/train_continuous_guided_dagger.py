"""Finite direct-teacher avoidance experiment; does not resume the app goal."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime,timezone
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
SCHEMA='continuous-calibrated-fullbrain-direct-teacher-DAgger-v1'


def arguments(argv=None):
    p=argparse.ArgumentParser(__doc__)
    for key in ('source','connectome','environment-config','expert-data','initial-student-trace','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--source-sha256',required=True)
    p.add_argument('--resume-checkpoint',type=Path)
    p.add_argument('--resume-sha256')
    p.add_argument('--resume-replay-index',type=Path)
    p.add_argument('--focused-supervision',action='store_true')
    p.add_argument('--actual-policy-priority',action='store_true')
    p.add_argument('--raw-actor-coefficient',type=float,default=None)
    p.add_argument('--native-opening-learning-probability',type=float,default=0.)
    p.add_argument('--attention-teaching-coefficient',type=float,default=0.)
    p.add_argument('--attention-query-lr',type=float,default=0.)
    p.add_argument('--joint-attention-preparation-updates',type=int,default=0)
    p.add_argument('--action-selection',choices=('categorical','student-greedy24-explore6'),default='categorical')
    p.add_argument('--goal-active',action='store_true')
    p.add_argument('--seed',type=int,default=2026102911)
    p.add_argument('--warmup-updates',type=int,default=240)
    p.add_argument('--stable-head-updates',type=int,default=48)
    p.add_argument('--rounds',type=int,default=8)
    p.add_argument('--evaluation-every-new-rounds',type=int,default=4)
    p.add_argument('--updates-per-round',type=int,default=90)
    p.add_argument('--max-seconds',type=int,default=28800)
    p.add_argument('--execute',action='store_true')
    return p.parse_args(argv)


def private_desktop_name(pid):
    # Match the existing IsolatedDesktopProcess shared-desktop contract.
    return 'fly-touhou-job-guided-dagger-'+str(int(pid))


def execute(a):
    import psutil
    from fly_touhou.thrl_durable_records import write_json,append_jsonl
    current=psutil.Process();a.output.mkdir(parents=True,exist_ok=False)
    config=json.loads(a.environment_config.read_text(encoding='utf8'))
    exe=Path(config['game_exe']).name.lower()
    if any((p.info['name'] or '').lower()==exe for p in psutil.process_iter(['name'])):
        raise RuntimeError('An existing original-game job is running; do not overlap it')
    from fly_touhou.owned_worker_affinity import worker_cpu_pool,spread_owned_workers
    worker_cpus=worker_cpu_pool(current.cpu_affinity())
    current.cpu_affinity(current.cpu_affinity()[-2:]);current.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    os.environ.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        FLY_TOUHOU_DESKTOP_NAME=private_desktop_name(os.getpid()),CUBLAS_WORKSPACE_CONFIG=':4096:8')
    from fly_touhou.thrl_job_paths import configure_native_temp
    configure_native_temp(a.output)
    begun=time.monotonic();costs=dict(supervised_Adam=0,student_actions=0,completed_rounds=0,PPO_updates=0)
    vector=collector=model=optimizer=None;calibration={};base_source=a.source;last_native_live=0.;previous_raw_tail=None
    launch=dict(schema=SCHEMA,status='running',PID=current.pid,create_time=current.create_time(),
        started_UTC=datetime.now(timezone.utc).isoformat(),argv=sys.argv,
        app_goal_remains_paused=not a.goal_active,experiment_authorized_by_current_user=True,
        action_selection=a.action_selection,focused_supervision=a.focused_supervision,
        actual_policy_priority=a.actual_policy_priority,
        raw_actor_coefficient=a.raw_actor_coefficient,
        native_opening_learning_probability=a.native_opening_learning_probability,
        attention_teaching_coefficient=a.attention_teaching_coefficient,attention_query_lr=a.attention_query_lr,
        training_worlds=30,max_native_worlds=30,student_decision_budget=a.rounds*7680,
        supervised_Adam_budget=a.warmup_updates+a.rounds*a.updates_per_round+a.joint_attention_preparation_updates,
        teacher='original ResourceV10 + all18 geometry4/12/24;no D95 action imitation',
        native_frame_cap=None,Bomb_forbidden=True,RAM_or_VRAM_caps=None,
        evaluation=f'20 greedy NoBomb games after every{a.evaluation_every_new_rounds} new rounds or end;new20 confirmation only if >=8/20 stage1 clean',
        full_game_PPO_resume=False,CUDA_started=False,native_started=False)
    write_json(a.output/'launch.json',launch)

    def guard():
        nonlocal last_native_live
        if (a.output/'STOP').exists():raise RuntimeError('Operator STOP')
        if time.monotonic()-begun>a.max_seconds:raise RuntimeError('Finite experiment wall budget reached')
        if shutil.disk_usage(a.output).free<10*1024**3:raise RuntimeError('Output disk reserve10GiB reached')
        if collector is not None and time.monotonic()-last_native_live>=5:
            write_json(a.output/'native-live.json',dict(acknowledged_student_actions=collector.acknowledged_actions,
                pending_actions=collector.pending_actions,elapsed_seconds=time.monotonic()-begun,
                world_count=30,actual_current_games=[{k:info.get(k) for k in ('pid','native_episode','stage','stage_frame','episode_frames')}
                    for info in (collector.infos or [])]))
            last_native_live=time.monotonic()

    def progress(phase,**extra):
        row=dict(schema=SCHEMA,phase=phase,costs=dict(costs),elapsed_seconds=time.monotonic()-begun,
            app_goal_remains_paused=not a.goal_active,**extra)
        write_json(a.output/'progress.json',row)

    def save(tag):
        from fly_touhou.continuous_guided_dagger import parameter_payload,sha
        payload=parameter_payload(model,base_source,a.source_sha256,calibration,costs)
        path=a.output/(tag+'-parameters.pt');temp=path.with_suffix('.pt.tmp')
        with temp.open('wb') as stream:torch.save(payload,stream);stream.flush();os.fsync(stream.fileno())
        os.replace(temp,path)
        if tag in ('initial','warmup','latest','completed-round','final'):
            path=a.output/(tag+'-training.pt');temp=path.with_suffix('.pt.tmp')
            with temp.open('wb') as stream:
                from fly_touhou.guided_optimizer_io import parameter_signature
                torch.save(dict(schema=SCHEMA+'-training',model_payload=payload,optimizer=optimizer.state_dict(),
                    optimizer_parameter_signature=parameter_signature(model,optimizer),
                    costs=dict(costs),numpy_RNG=rng.bit_generator.state,torch_RNG=torch.get_rng_state(),
                    CUDA_RNG=torch.cuda.get_rng_state_all(),neural_midgame_resume=False),stream)
                stream.flush();os.fsync(stream.fileno())
            os.replace(temp,path)
        return a.output/(tag+'-parameters.pt')

    def close_training():
        nonlocal vector,collector,previous_raw_tail
        if vector is not None:
            vector.close()
            errors=getattr(vector,'close_errors',[])
            write_json(a.output/f'cleanup-after-round{costs["completed_rounds"]:02}.json',dict(errors=errors))
            vector=collector=None;previous_raw_tail=None
            if errors:raise RuntimeError('Native world cleanup incomplete:'+repr(errors))

    def start_training():
        nonlocal vector,collector
        from fly_touhou.thrl_vector import THRLNativeVector
        from fly_touhou.continuous_object_avoidance_ppo import make_object_avoidance_PPO_vector
        from fly_touhou.fly_visual_online_collector_continuation import MovementDiagnosticsVector
        from fly_touhou.continuous_full_sensory_ppo import FullSensoryPPOPool
        from train_fly_visual_long_continuous import staggered_reset_vector
        cfg=copy.deepcopy(config);cfg['seed']=a.seed+costs['completed_rounds']
        vector=MovementDiagnosticsVector(make_object_avoidance_PPO_vector(cfg,
            a.output/f'native-train-{costs["completed_rounds"]:02}',vector_class=staggered_reset_vector(THRLNativeVector)))
        write_json(a.output/f'worker-affinity-{costs["completed_rounds"]:02}.json',
            spread_owned_workers(current.pid,current.create_time(),worker_cpus))
        collector=FullSensoryPPOPool(model,vector,reference_refiner=None,geometry_label_budget=2048,action_seed=a.seed+7,
            action_selection=a.action_selection)
        progress('starting_private_native_worlds')
        collector.start();launch['native_started']=True;write_json(a.output/'launch.json',launch)

    def train_updates(count,*,round_index,warmup=False,attention_preparation=False):
        from fly_touhou.continuous_guided_dagger import supervised_sequence
        from fly_touhou.whole_brain_ppo import clip_gradients_fp64
        model.train().requires_grad_(True)
        for _ in range(count):
            guard();data,mix=replay.sample(rng,current_round=round_index,warmup=warmup)
            stable=warmup and costs['supervised_Adam']<a.stable_head_updates
            optimizer.zero_grad(set_to_none=True)
            offsets=None
            if a.native_opening_learning_probability>0:
                starts=data['episode_starts'][0].astype(bool)
                offsets=np.where(starts&(rng.random(len(starts))<a.native_opening_learning_probability),0,16)
            loss,metrics=supervised_sequence(model,data,stable_body=stable,native_hit_actor=a.focused_supervision,
                actual_policy_priority=a.actual_policy_priority,
                raw_actor_coefficient=a.raw_actor_coefficient,
                attention_coefficient=a.attention_teaching_coefficient,learning_offsets=offsets,check_stop=guard)
            if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite direct-teacher loss')
            loss.backward()
            if costs['supervised_Adam'] in (0,a.stable_head_updates):
                gradients={g['name']:dict(gradient_tensors=sum(p.grad is not None for p in g['params']),
                    tensors=len(g['params']),norm=sum(float(p.grad.detach().double().square().sum())
                    for p in g['params'] if p.grad is not None)**.5) for g in optimizer.param_groups}
                write_json(a.output/f'gradient-path-{costs["supervised_Adam"]:06}.json',gradients)
            norm=clip_gradients_fp64(list(model.parameters()),5.)
            optimizer.step();costs['supervised_Adam']+=1
            if not stable:costs['brain_updates_in_this_phase']=costs.get('brain_updates_in_this_phase',0)+1
            if attention_preparation:costs['attention_preparation_Adam']=costs.get('attention_preparation_Adam',0)+1
            numbers={k:float(v.detach()) for k,v in metrics.items()}
            if a.attention_teaching_coefficient>0:
                numbers['attention_query_norm']=float(model.sensory_adapter.object_attention.object_pool_query.detach().norm())
            row=dict(step=costs['supervised_Adam'],phase='head_calibration' if stable else 'expert_warmup' if warmup else 'DAgger_joint',
                round=round_index,total_loss=float(loss.detach()),gradient_norm=norm,sample_mix=mix,
                elapsed_seconds=time.monotonic()-begun,**numbers)
            append_jsonl(a.output/'learning.jsonl',row)
            progress(row['phase'],learning=row,replay_sequences=len(replay.records))
            if costs['supervised_Adam']%30==0:save('latest')
            del data,loss,metrics

    def evaluate(tag,seed):
        close_training();guard();optimizer.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
        path=save(tag);model.eval().requires_grad_(False)
        from fly_touhou.continuous_no_bomb_first_contact import evaluate_no_bomb,make_no_bomb_policy
        policy=make_no_bomb_policy(torch,model,family='deep',episodes=20)
        progress('frozen_NoBomb_evaluation',evaluation_tag=tag)
        result=evaluate_no_bomb(torch,policy,config,a.output/tag,checkpoint_sha256=sha(path),
            episodes=20,workers=20,seed=seed,check_stop=guard,worker_cpus=worker_cpus,
            progress=lambda row:progress('frozen_NoBomb_evaluation',evaluation_tag=tag,evaluation=row))
        model.requires_grad_(True).train();return result

    try:
        import numpy as np
        import torch
        from fly_touhou.continuous_guided_dagger import (sha,install_guided_interfaces,SequenceReplay)
        from fly_touhou.continuous_full_pair_geometry_model import build_full_pair_model
        from fly_touhou.continuous_guided_replay import add_expert_sequences,add_initial_student_sequences,add_student_rollout
        torch.set_num_threads(2);torch.manual_seed(a.seed);rng=np.random.default_rng(a.seed)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        if sha(a.source)!=a.source_sha256:raise ValueError('Source parameter checkpoint SHA differs')
        replay=SequenceReplay(a.output/'raw-replay')
        if a.focused_supervision:
            from fly_touhou.continuous_guided_focused_replay import FocusedSequenceReplay,add_focused_student_rollout
            replay=FocusedSequenceReplay(a.output/'raw-replay')
        if a.attention_teaching_coefficient>0:
            from fly_touhou.continuous_guided_attention_replay import AttentionSequenceReplay
            replay=AttentionSequenceReplay(a.output/'raw-replay')
        resume=None
        if a.resume_checkpoint:
            progress('restoring_saved_warmup_and_raw_replay')
            if sha(a.resume_checkpoint)!=a.resume_sha256:raise ValueError('Resume checkpoint SHA differs')
            with a.resume_checkpoint.open('rb') as stream:resume=torch.load(stream,map_location='cpu',weights_only=True)
            payload=resume['model_payload'] if resume.get('schema')==SCHEMA+'-training' else resume
            if payload.get('schema')!=SCHEMA+'-parameters':raise ValueError('Independent guided model checkpoint required')
            from fly_touhou.continuous_guided_dagger import build_guided_model
            model=build_guided_model(payload,a.connectome,device='cuda:0')
            costs.update(payload['counters']);calibration=copy.deepcopy(payload['calibration'])
            costs['brain_updates_in_this_phase']=0
            index=json.loads(a.resume_replay_index.read_text(encoding='utf8'))
            replay.records=copy.deepcopy(index['records']);replay.commit()
            write_json(a.output/'resume.json',dict(source=str(a.resume_checkpoint),sha256=a.resume_sha256,
                inherited_costs=dict(costs),reused_raw_replay_index=str(a.resume_replay_index),
                reused_sequences=len(replay.records),weights_restored=True,interfaces_reinitialized=False,
                recalibrated=False,optimizer_restored='optimizer' in resume,
                optimizer_transition='restore complete Adam' if 'optimizer' in resume else 'fresh Adam at DAgger phase;old warmup did not save optimizer',
                native_midgame_resume=False))
        else:
            progress('preparing_persistent_expert_replay')
            expert=add_expert_sequences(replay,a.expert_data,rng,check_stop=guard,
                progress=lambda row:progress('preparing_persistent_expert_replay',data=row))
            write_json(a.output/'expert-source.json',expert)
            old_student=add_initial_student_sequences(replay,a.initial_student_trace,rng,check_stop=guard)
            write_json(a.output/'initial-student-source.json',old_student)
            progress('loading_complete_connectome_on_CUDA')
            with a.source.open('rb') as stream:source=torch.load(stream,map_location='cpu',weights_only=True)
            model=build_full_pair_model(source,a.connectome,device='cuda:0');del source
            install_guided_interfaces(model)
        launch['CUDA_started']=True
        launch.update(graph_neurons=model.core.n_neurons,graph_edges=model.core.n_edges,input_ports=len(model.core.input_indices),
            output_ports=len(model.core.output_indices),source=str(a.source),source_sha256=a.source_sha256)
        write_json(a.output/'launch.json',launch)
        if resume is None:
            progress('calibrating_training_mixture_ports')
            calibration_rows=[]
            for origin in ('expert','student'):
                records=[r for r in replay.records if r['origin']==origin]
                selected=rng.choice(len(records),size=min(16,len(records)),replace=False)
                with torch.no_grad():
                    for i in selected:
                        guard()
                        with np.load(records[int(i)]['path']) as data:
                            encoded=model.encode_raw(data['features'],data['maps'],objects=data['objects'])
                        calibration_rows.append(encoded.detach())
            calibration=model.core.input_encoder.calibrate(torch.cat(calibration_rows))
            calibration.update(training_origins=['original_teacher_train_prefixes','prior_student_training_states'],
                development_or_test_observations_used=False,old_fixed_coordinates_retained_before_new_normalization=True)
            del calibration_rows
        write_json(a.output/'port-calibration.json',calibration)
        groups=model.optimizer_groups(input_lr=1e-4,brain_lr=3e-5,actor_lr=3e-4,critic_lr=3e-5,
            risk_lr=3e-4,sensory_lr=1e-4,query_lr=1e-4,cnn_numeric_lr=1e-4,geometry_lr=1e-4)
        if resume is not None and resume.get('optimizer_parameter_signature') is not None:
            from fly_touhou.guided_optimizer_io import groups_from_signature
            groups=groups_from_signature(model,resume['optimizer'],resume['optimizer_parameter_signature'])
        optimizer=torch.optim.Adam(groups,eps=1e-8)
        if resume is not None and 'optimizer' in resume:
            optimizer.load_state_dict(resume['optimizer']);rng.bit_generator.state=resume['numpy_RNG']
            torch.set_rng_state(resume['torch_RNG']);torch.cuda.set_rng_state_all(resume['CUDA_RNG'])
        if a.attention_query_lr>0:
            from fly_touhou.guided_optimizer_io import separate_attention_query
            write_json(a.output/'attention-query-transition.json',separate_attention_query(optimizer,
                model.sensory_adapter.object_attention.object_pool_query,lr=a.attention_query_lr))
        write_json(a.output/'algorithm.json',dict(schema=SCHEMA,
            optimizer='restored complete Adam' if resume is not None and 'optimizer' in resume else 'fresh Adam for supervised phase',
            optimizer_groups=[{k:g[k] for k in ('name','lr')} for g in optimizer.param_groups],
            actor_and_risk_joint=True,brain_and_all_sensory_trainable_after_brief_head_calibration=True,
            warmup_Adam=a.warmup_updates,stable_body_head_Adam=a.stable_head_updates,
            BPTT=16,raw_prefix_burn_in=16,raw_prefix_state_is_approximate=True,
            native_opening_learning_probability=a.native_opening_learning_probability,
            opening_history='exact reset;zero burn-in only for selected genuine game openings',
            sample_mix=dict(expert=.2,recent_student=.4,historical_hard=.4),
            historical_data_used_for_PPO=False,PPO_updates=0,value_updates=0,three_critics_retained=True,
            inference_path='raw -> sensory384 -> all8053VP -> fullgraph -> DN1299 -> actor/risk',
            teacher_targets='full18;trusted4/12/24;rule distribution;actual chosen contact fact',
            initial_actor_final_and_risk_readout_reinitialized=resume is None,no_D95_targets=True,
            focused_supervision=a.focused_supervision,native_hit_actor_loss_coefficient=.1 if a.focused_supervision else 0.,
            actual_policy_priority=a.actual_policy_priority,
            raw_actor_coefficient=(.25 if a.actual_policy_priority else 1.) if a.raw_actor_coefficient is None else a.raw_actor_coefficient,
            composed_actor_coefficient=1. if a.actual_policy_priority else .25,
            action_selection=a.action_selection,deterministic_data_never_PPO=True,
            attention_teacher=dict(coefficient=a.attention_teaching_coefficient,query_lr=a.attention_query_lr,
                joint_preparation_updates=a.joint_attention_preparation_updates,
                targets='per-direction per-object four-frame signed clearance;critical query min<=8;soft T2',
                teacher_labels_are_runtime_inputs=False,point_only_unknown_and_laser_scenes_abstain=True,
                teacher_fitted_query_injected=False),
            native_real_decision_frames=4,native_frame_cap=None,headless_focus='existing private non-input desktop'))
        save('initial');train_updates(max(0,a.warmup_updates-costs['supervised_Adam']),round_index=-1,warmup=True);save('warmup')
        remaining_preparation=max(0,a.joint_attention_preparation_updates-costs.get('attention_preparation_Adam',0))
        if remaining_preparation:
            train_updates(remaining_preparation,round_index=costs['completed_rounds']-1,attention_preparation=True)
            save('completed-round')
        evaluations=[];confirmation=None;success=False;phase_start_round=costs['completed_rounds']
        for round_index in range(costs['completed_rounds'],a.rounds):
            guard()
            if vector is None:start_training()
            progress('collecting_student_own_actions',round=round_index)
            rollout=collector.collect(steps=256,policy_version=round_index,check_stop=guard)
            costs['student_actions']+=int(rollout.actions.numel())
            progress('aggregating_student_danger_sequences',round=round_index)
            if a.focused_supervision:
                replay_receipt,previous_raw_tail=add_focused_student_rollout(replay,rollout,rng,
                    round_index=round_index,previous=previous_raw_tail,check_stop=guard)
            else:replay_receipt=add_student_rollout(replay,rollout,rng,round_index=round_index,check_stop=guard)
            write_json(a.output/f'round-{round_index+1:02}-collection.json',dict(replay=replay_receipt,
                completed_training_games=rollout.completed_episodes,collection=rollout.metadata,
                training_games_are_not_frozen_evaluation=True))
            del rollout;gc.collect()
            train_updates(a.updates_per_round,round_index=round_index)
            costs['completed_rounds']+=1;save('latest');save('completed-round')
            if (costs['completed_rounds']-phase_start_round)%a.evaluation_every_new_rounds==0 or costs['completed_rounds']==a.rounds:
                tag=f'development-round{costs["completed_rounds"]:02}'
                report=evaluate(tag,2027432901+costs['completed_rounds'])
                evaluations.append(dict(tag=tag,report=report))
                if report['first_stage_without_observed_contact_count']>=8:
                    confirmation=evaluate('confirmation20',2028432901+costs['completed_rounds'])
                    success=confirmation['first_stage_without_observed_contact_count']>=8
                    break
        close_training();save('final')
        result=dict(schema=SCHEMA,status='completed',costs=costs,evaluations=evaluations,confirmation=confirmation,
            basic_avoidance_target_met=success,app_goal_remains_paused=not a.goal_active,automatic_full_game_goal_resume=False,
            next_action='report then discuss restoring full goal' if success else 'report failure and discuss;no automatic extra training',
            elapsed_seconds=time.monotonic()-begun,calibration=calibration)
        write_json(a.output/'result.json',result);progress('experiment_completed',basic_avoidance_target_met=success)
    except BaseException as error:
        write_json(a.output/'failure.json',dict(schema=SCHEMA,status='failed',error=repr(error),traceback=traceback.format_exc(),
            costs=costs,app_goal_remains_paused=not a.goal_active,automatic_retry=False,elapsed_seconds=time.monotonic()-begun))
        raise
    finally:close_training()


def main(argv=None):
    a=arguments(argv)
    if not (0<=a.stable_head_updates<=a.warmup_updates and 1<=a.rounds<=32 and 1<=a.evaluation_every_new_rounds<=32
            and a.updates_per_round>0 and a.max_seconds>0):raise ValueError('Finite explicit experimental budget required')
    if a.resume_checkpoint and (not a.resume_sha256 or not a.resume_replay_index):
        raise ValueError('Resume checkpoint requires exact SHA and persistent raw replay index')
    if (not 0<=a.native_opening_learning_probability<=1. or not 0<=a.attention_teaching_coefficient<=1. or not 0<=a.attention_query_lr<=.01
            or not 0<=a.joint_attention_preparation_updates<=1000):
        raise ValueError('Explicit finite attention teaching configuration required')
    if a.attention_teaching_coefficient>0 and (not a.focused_supervision or not a.attention_query_lr>0):
        raise ValueError('Attention teaching requires focused replay and an explicit query learning rate')
    if not a.execute:
        print(json.dumps(dict(schema=SCHEMA,configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
            native_started=False,CUDA_started=False,app_goal_resumed=False),ensure_ascii=False,indent=2));return
    execute(a)


if __name__=='__main__':main()
