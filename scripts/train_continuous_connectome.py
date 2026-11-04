"""Bounded cached-teacher warmup -> autonomous continuous-connectome PPO30.

Default is a NoTorch plan. Only the explicit execution flag starts the new
continuous family; this entry neither resumes LIF nor controls another job.
"""
from __future__ import annotations
import argparse
import copy
from dataclasses import asdict
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
SCHEMA='continuous-connectome-warmup-PPO30-entry-v1'
D95_SHA='d95fe710068483fc81d8525d89bccf72e61d209c7e1bb442ea66d022b9c1d06c'


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('connectome','d95-checkpoint','environment-config','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--statistics',type=Path,default=Path('./outputs/distillation/D95-frozen-visual384-prefix2048-v1/train-statistics.npz'))
    p.add_argument('--warmup-trace',type=Path,action='append',default=[])
    p.add_argument('--warmup-updates',type=int,default=128)
    p.add_argument('--warmup-epochs',type=int,default=1)
    p.add_argument('--warmup-learning-rate',type=float,default=3e-4)
    p.add_argument('--cached-evaluation',action='store_true')
    p.add_argument('--cached-evaluation-max-rows',type=int,default=61440)
    p.add_argument('--decisions',type=int,default=983040)
    p.add_argument('--fixed-brain',action='store_true')
    p.add_argument('--training-source',type=Path)
    p.add_argument('--training-source-sha256')
    p.add_argument('--seed',type=int,default=613711)
    p.add_argument('--action-seed',type=int,default=971137)
    p.add_argument('--geometry-label-budget',type=int,default=2048)
    p.add_argument('--max-seconds',type=int,required=True)
    p.add_argument('--development-episodes',type=int,choices=(0,20),default=20)
    p.add_argument('--risk-supervision',choices=('relative-cost','signed-clearance4f'),default=None,
        help='Default relative-cost for fresh runs;omission retains an explicit source algorithm mode')
    p.add_argument('--freeze-after-first-training-clear',action='store_true')
    p.add_argument('--execute-cuda',action='store_true')
    a=p.parse_args(argv)
    if (a.decisions<0 or a.decisions%7680 or a.warmup_updates<0 or a.warmup_epochs<1 or a.max_seconds<1
            or not math.isfinite(a.warmup_learning_rate) or a.warmup_learning_rate<=0 or a.cached_evaluation_max_rows<1
            or not 1<=a.geometry_label_budget<=2048
            or any(not 0<=x<2**31 for x in (a.seed,a.action_seed))
            or (a.training_source is None)!=(a.training_source_sha256 is None)):
        p.error('Explicit finite warmup/PPO budgets,source identity and natural30-world rows required')
    if a.execute_cuda and a.warmup_updates and not a.warmup_trace:p.error('Warmup needs ordered actual cached traces from new-game row0')
    return a


def resolve_risk_supervision(requested,source_algorithm=None):
    mode=requested if requested is not None else (source_algorithm or {}).get('risk_supervision','relative-cost')
    if mode not in ('relative-cost','signed-clearance4f'):raise ValueError('Unsupported source risk supervision')
    return mode


def first_full_training_clear(completed_episodes,environment):
    """Only a development checkpoint trigger;never a frozen/stable success claim."""
    if environment.get('training_mode')!='full-story-v1' or environment.get('continues_allowed') is not False:
        return None
    for episode in completed_episodes:
        summary=episode.get('summary',{})
        history=summary.get('stage_history',[])
        if (summary.get('outcome')!='clear' or not history
                or any(type(stage) is not int for stage in history) or max(history)<6):continue
        source='summary.continues';measured=summary.get('continues')
        if measured is None:
            source='summary.endpoints.end.continues'
            measured=summary.get('endpoints',{}).get('end',{}).get('continues')
        if measured is not None and measured!=0:continue
        evidence=(dict(source=source,value=measured) if measured is not None else
            dict(source='full-story environment protocol',continues_allowed=False,
                episode_continues_measurement_available=False))
        return dict(reason='训练通关触发开发验证',episode=copy.deepcopy(episode),max_stage=max(history),
            no_continue_evidence=evidence,training_episode_is_frozen_evaluation=False,formal_stability_success=False)
    return None


def plan(a):
    risk_mode=resolve_risk_supervision(a.risk_supervision)
    return dict(schema=SCHEMA,status='prepared_not_executed',Torch_imported='torch' in sys.modules,
        family='continuous signed complete-connectome RNN;not spiking',K=8,
        full_graph_neurons=138639,full_graph_edges=15091983,input_ports='all annotated VP8053',DN_ports=1299,
        trainable_internal_brain=not a.fixed_brain,fixed_brain_control_preserves_input_and_heads=True,
        warmup=dict(ordered_trace_paths=[str(p) for p in a.warmup_trace],max_Adam=a.warmup_updates,
            epochs=a.warmup_epochs,learning_rate=a.warmup_learning_rate,
            real_forward_rows_per_update=480,forward_rows_at128_updates=61440,heldout_rule='game_id%5==0',
            objective='original five guidance + relative D95cost;noPPO/value',
            own_continuous_state_from_actual_newgame=True,teacher_hidden_used=False),
        PPO=dict(decisions=a.decisions,rollouts=a.decisions//7680,worlds=30,rollout_steps=256,
            BPTT=16,neural_batch=16,minibatch_true_rows=256,microbatches_per_Adam=1,
            PPO_loss_scale=1.,epochs_max=3,Adam_max=a.decisions//7680*90,
            gamma=.999,GAE_lambda=.995,clip=.2,value_clip=.2,value_coefficient=.5,
            critic_shared_gradient=.1,entropy_coefficient=.0001,global_grad_clip=.5,target_KL=.03,
            learning_rate=3e-5,three_values=['survival','fight','resource']),
        action_count=19,action_temperature=1.,reward='th14-completion-reward-v3',
        native_caps=None,continues=False,teacher_actions_executed=0,
        activity_reset='new game only;notMiss/stage/update',source=str(a.training_source) if a.training_source else None,
        checkpoint_compatibility='typed structure/shape/graphhash + fulloptimizer/RNG;sourceSHA provenance',
        cached_evaluation=a.cached_evaluation,cached_evaluation_max_rows=a.cached_evaluation_max_rows,
        FP32=True,TF32=False,deterministic_algorithms=False,
        maximum_seconds=a.max_seconds,frozen_development_games=a.development_episodes,
        frozen_development_seed=2027431601,independent_stability_test=False,
        risk_supervision=risk_mode,risk_coefficient=.05 if risk_mode=='signed-clearance4f' else 1.,
        omitted_risk_mode_retains_source_mode_on_execution=True,
        freeze_after_first_training_clear=a.freeze_after_first_training_clear,
        CUDA_started=False,native_started=False)


def execute(a):
    if a.output.exists():raise FileExistsError('Fresh bounded output required;no implicit retry')
    if 'torch' in sys.modules:raise RuntimeError('Fresh explicit executor required')
    if sha(a.d95_checkpoint)!=D95_SHA:raise ValueError('The fixed D95 sensory/label source differs')
    from fly_touhou.thrl_study_io import read_shared_json
    environment=read_shared_json(a.environment_config)
    if (environment.get('reward_schema')!='th14-completion-reward-v3'
            or environment.get('training_mode')!='full-story-v1' or environment.get('headless') is not True
            or environment.get('development_objective')=='avoidance-first-v1'):
        raise ValueError('Original private natural full-story completionV3 environment required')
    import psutil
    current=psutil.Process();native_name=Path(environment['game_exe']).name.lower()
    if any((p.info['name'] or '').lower()==native_name for p in psutil.process_iter(['name'])):
        raise RuntimeError('Existing original games;this entry does not control another job')
    from fly_touhou.owned_worker_affinity import worker_cpu_pool,spread_owned_workers
    allowed=current.cpu_affinity();worker_cpus=worker_cpu_pool(allowed)
    current.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);current.cpu_affinity(allowed[-2:])
    os.environ.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',FLY_TOUHOU_DESKTOP_NAME='fly-touhou-job-continuous-'+str(os.getpid()))
    from fly_touhou.thrl_durable_records import write_json
    from fly_touhou.thrl_job_paths import configure_native_temp
    a.output.mkdir(parents=True);(a.output/'traces').mkdir();configure_native_temp(a.output)
    launch=dict(plan(a),PID=os.getpid(),create_time=current.create_time(),started_UTC=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
        environment_config_sha256=sha(a.environment_config),source_SHA_is_provenance_only=True)
    write_json(a.output/'launch.json',launch)
    started=time.monotonic();model=optimizer=collector=vector=None;costs=dict(warmup_Adam=0,PPO_Adam=0,student_actions=0,completed_rollouts=0)
    latest={};cleanup_errors=[];closed=False;algorithm=provenance=None
    transaction={'complete':True};phase={'name':'warmup_Adam'};training_stop=None
    def progress(row):
        latest.clear();latest.update(row)
        write_json(a.output/'progress.json',dict(row,schema=SCHEMA,trainable_brain=not a.fixed_brain,
            costs=copy.deepcopy(costs),elapsed_seconds=time.monotonic()-started))
    def guard():
        if (a.output/'STOP').exists():raise RuntimeError('OperatorSTOP')
        if time.monotonic()-started>a.max_seconds:raise RuntimeError('Finite wall budget reached')
    def save(tag):
        from fly_touhou.continuous_connectome_training import training_checkpoint_payload
        payload=training_checkpoint_payload(model,optimizer,algorithm=algorithm,counters=costs,provenance=provenance)
        for suffix,value in (('training',payload),('parameters',model.parameter_checkpoint_payload())):
            target=a.output/f'{tag}-{suffix}.pt';temporary=target.with_suffix('.pt.tmp')
            with temporary.open('wb') as stream:torch.save(value,stream);stream.flush();os.fsync(stream.fileno())
            os.replace(temporary,target)
    def close_training():
        nonlocal closed
        if not closed:
            closed=True
            if vector is not None:
                try:vector.close()
                except BaseException as error:cleanup_errors.append(repr(error))
                cleanup_errors.extend(str(x) for x in getattr(vector,'close_errors',[]))
        write_json(a.output/'training-cleanup.json',dict(close_errors=cleanup_errors,worlds=30,training_closed=closed))
        return cleanup_errors
    try:
        import torch
        torch.set_num_threads(2);torch.manual_seed(a.seed)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(False)
        from fly_touhou.continuous_connectome import ContinuousConnectomeCore
        from fly_touhou.continuous_connectome_model import ContinuousConnectomeActorCritic
        from fly_touhou.continuous_connectome_training import (configure_learning,learning_parameters,
            pretrain_cached_teacher_rollouts,update_continuous_pool,restore_training_checkpoint,transition_learning_rate)
        from fly_touhou.fly_frozen_v5_perception import FrozenV5Perception
        from fly_touhou.fly_visual_distillation import FixedStandardizedVisual,load_visual_statistics
        from fly_touhou.fly_gru_aligned_ppo import aligned_ppo_config
        from fly_touhou.fly_gru_aligned_guidance import DEFAULT_CONFIG
        from fly_touhou.fly_gru_risk26_contract import ContinuousCostConfig
        perception=FrozenV5Perception.from_d95_checkpoint(a.d95_checkpoint)
        mean,std,statistics=load_visual_statistics(a.statistics)
        adapter=FixedStandardizedVisual(perception,mean,std,statistics)
        core=ContinuousConnectomeCore.from_full_connectome(a.connectome,K=8,
            trainable_brain=not a.fixed_brain,device='cpu',seed=a.seed)
        model=ContinuousConnectomeActorCritic(core,adapter,seed=a.seed).to('cuda:0')
        scope=configure_learning(model,trainable_brain=not a.fixed_brain)
        cfg=aligned_ppo_config()
        risk_mode=resolve_risk_supervision(a.risk_supervision)
        risk_config=ContinuousCostConfig(coefficient=.05 if risk_mode=='signed-clearance4f' else 1.)
        initial_lr=a.warmup_learning_rate if a.warmup_updates else cfg.learning_rate
        optimizer=torch.optim.Adam(model.optimizer_groups(input_lr=initial_lr,brain_lr=initial_lr,
            actor_lr=initial_lr,critic_lr=initial_lr,risk_lr=initial_lr),lr=initial_lr)
        def before_step(*unused):transaction['complete']=False
        def after_step(*unused):
            costs[phase['name']]+=1;transaction['complete']=True
        optimizer.register_step_pre_hook(before_step);optimizer.register_step_post_hook(after_step)
        algorithm=dict(schema=SCHEMA,core=model.core.config(),PPO=asdict(cfg),guidance=asdict(DEFAULT_CONFIG),
            continuous_cost=asdict(risk_config),brain_BPTT=16,neural_width=16,minibatch_true_rows=256,
            PPO_loss_scale=1.,fixed_brain=a.fixed_brain,warmup_max_updates=a.warmup_updates,
            phase='cached_teacher_warmup' if a.warmup_updates else 'autonomous_PPO',
            actor_policy='DN raw + DN cost beta8;inventory legal T1',
            freeze_after_first_training_clear=a.freeze_after_first_training_clear,
            optimizer_group_learning_rates={group['name']:group['lr'] for group in optimizer.param_groups})
        source_modules=('continuous_connectome.py','continuous_sparse.py','continuous_connectome_model.py',
            'continuous_connectome_collector.py','continuous_connectome_training.py')
        provenance=dict(entry_sha256=sha(__file__),module_sha256={name:sha(ROOT/'src/fly_touhou'/name) for name in source_modules},
            D95_checkpoint_sha256=D95_SHA,statistics=statistics,scope=scope,source_SHA_is_load_gate=False,
            source_native_state_resumed=False)
        if a.training_source:
            if sha(a.training_source)!=a.training_source_sha256:raise ValueError('Explicit source checkpoint identity differs')
            source=torch.load(a.training_source,map_location='cpu',weights_only=True)
            inherited=restore_training_checkpoint(model,optimizer,source,restore_rng=True)
            if model.risk_head.beta!=8.:raise ValueError('Source cost composition beta differs from the declared beta8')
            source_algorithm=source.get('algorithm',{})
            risk_mode=resolve_risk_supervision(a.risk_supervision,source_algorithm)
            risk_config=(ContinuousCostConfig(coefficient=.05) if risk_mode=='signed-clearance4f' else
                ContinuousCostConfig(**source_algorithm.get('continuous_cost',asdict(ContinuousCostConfig()))))
            provenance['parent']=dict(path=str(a.training_source),sha256=a.training_source_sha256,counters=inherited,
                algorithm=copy.deepcopy(source_algorithm))
            if 'risk_pretraining_protocol' in source_algorithm:
                algorithm['risk_pretraining_protocol']=copy.deepcopy(source_algorithm['risk_pretraining_protocol'])
            del source
        algorithm.update(risk_supervision=risk_mode,continuous_cost=asdict(risk_config))
        if risk_mode=='signed-clearance4f':
            if a.warmup_updates:raise ValueError('Signed risk pretraining has its own exact-query entry;set --warmup-updates0 for this PPO entry')
            from fly_touhou.continuous_signed_clearance import protocol
            algorithm['risk_supervision_protocol']=protocol()
            provenance['module_sha256']['continuous_signed_clearance.py']=sha(ROOT/'src/fly_touhou/continuous_signed_clearance.py')
        launch.update(risk_supervision=risk_mode,risk_coefficient=risk_config.coefficient)
        write_json(a.output/'launch.json',launch)
        provenance['initial_phase_learning_rate']=transition_learning_rate(optimizer,initial_lr,phase=algorithm['phase'])
        write_json(a.output/'initialization.json',dict(algorithm=algorithm,provenance=provenance))
        save('initial')
        if a.cached_evaluation:
            from fly_touhou.continuous_connectome_cached_evaluation import evaluate_cached_continuous
            progress(dict(phase='cached_evaluation_initial'))
            initial_evaluation=evaluate_cached_continuous(model,a.warmup_trace,a.output/'cached-evaluation-initial',
                check_stop=guard,max_rows=a.cached_evaluation_max_rows)
            write_json(a.output/'cached-evaluation-initial.json',initial_evaluation)
        if a.warmup_updates:
            warm=pretrain_cached_teacher_rollouts(model,optimizer,a.warmup_trace,max_updates=a.warmup_updates,
                epochs=a.warmup_epochs,check_stop=guard,cost_config=risk_config,
                progress=lambda record:progress(dict(phase='cached_teacher_warmup',learning=record)))
            write_json(a.output/'warmup-result.json',warm);save('warmup-completed');gc.collect()
        if a.decisions:
            algorithm['phase']='autonomous_PPO'
            provenance['warmup_to_PPO_learning_rate']=transition_learning_rate(optimizer,cfg.learning_rate,phase='autonomous_PPO')
            algorithm['optimizer_group_learning_rates']={group['name']:group['lr'] for group in optimizer.param_groups}
            write_json(a.output/'warmup-to-PPO.json',dict(same_model_and_Adam=True,optimizer_reset=False,
                cached_own_activity_not_mistaken_for_native_state=True,new_native_worlds=30,costs=copy.deepcopy(costs)))
            from fly_touhou.continuous_connectome_collector import ContinuousConnectomeOnlinePool,record_continuous_activity
            from fly_touhou.fly_d95_reference_refinement import D95RiskPenalty
            from fly_touhou.fly_task_reference_pool import make_task_reference_vector
            from fly_touhou.fly_visual_online_collector_continuation import MovementDiagnosticsVector
            from fly_touhou.thrl_vector import THRLNativeVector
            from train_fly_visual_long_continuous import staggered_reset_vector
            penalty=D95RiskPenalty.from_checkpoint(a.d95_checkpoint,device=model.device)
            vector=MovementDiagnosticsVector(make_task_reference_vector(environment,a.output/'training',
                vector_class=staggered_reset_vector(THRLNativeVector)))
            write_json(a.output/'worker-affinity-before-reset.json',spread_owned_workers(os.getpid(),current.create_time(),worker_cpus))
            collector=ContinuousConnectomeOnlinePool(model,vector,reference_refiner=penalty,
                geometry_label_budget=a.geometry_label_budget,action_seed=a.action_seed)
            collector.start();phase['name']='PPO_Adam'
            for version in range(a.decisions//7680):
                guard();progress(dict(phase='collecting',rollout=version))
                rollout=collector.collect(steps=256,policy_version=version,check_stop=guard)
                costs['student_actions']+=int(rollout.actions.numel())
                import numpy as np
                names=('encoded_visual','features','legal_actions','actions','old_log_probs','values',
                    'reward_vectors','bootstrap_values','terminated','truncated','episode_starts','game_ids',
                    'rule_probabilities','rule_weights','deathbomb_mask','d95_risk_costs',
                    'd95_risk_queried_mask','d95_risk_valid_mask')
                arrays={key:getattr(rollout,key).detach().cpu().numpy() for key in names}
                arrays.update({'task_'+key:np.asarray(getattr(rollout.task_targets,key))
                    for key in ('margins','trusted','outcomes','unprotected_mask')})
                trace_path=a.output/'traces'/f'rollout-{version:03}.npz';temporary=trace_path.with_suffix('.npz.tmp')
                with temporary.open('wb') as stream:np.savez(stream,**arrays);stream.flush();os.fsync(stream.fileno())
                os.replace(temporary,trace_path)
                write_json(trace_path.with_suffix('.json'),dict(trace_sha256=sha(trace_path),
                    metadata=rollout.metadata,completed_episodes=rollout.completed_episodes,
                    raw_maps_objects_or_old_neural_state_saved=False))
                try:activity=record_continuous_activity(model,rollout,a.output/'neural-diagnostics'/f'rollout-{version:06}')
                except (ValueError,OSError,RuntimeError) as error:
                    activity=dict(status='diagnostic_unavailable',error=repr(error),training_rejected=False)
                progress(dict(phase='updating',rollout=version))
                learning=update_continuous_pool(model,optimizer,rollout,cfg=cfg,check_stop=guard,minibatch_seed=a.seed+version,
                    cost_config=risk_config,risk_supervision=risk_mode)
                costs['completed_rollouts']+=1
                write_json(a.output/'traces'/f'update-{version+1:06}.json',dict(learning=learning,activity=activity,
                    completed_episodes=rollout.completed_episodes,costs=copy.deepcopy(costs)))
                save('last-completed');progress(dict(phase='updated',rollout=version))
                if a.freeze_after_first_training_clear:
                    trigger=first_full_training_clear(rollout.completed_episodes,environment)
                    if trigger is not None:
                        training_stop=dict(trigger,completed_rollout=version+1,paid_counters=copy.deepcopy(costs),
                            checkpoint_saved_before_trigger=str(a.output/'last-completed-training.pt'),
                            criterion='observation for early frozen development;not final acceptance')
                        provenance['training_stop']=copy.deepcopy(training_stop)
                        write_json(a.output/'training-clear-trigger.json',training_stop)
                        progress(dict(phase='training_clear_triggered',training_stop=training_stop))
                        del rollout,learning;gc.collect();break
                del rollout,learning;gc.collect()
        save('final')
        if close_training():raise RuntimeError('Native training cleanup incomplete')
        collector=vector=None;gc.collect();development=None
        if a.cached_evaluation:
            progress(dict(phase='cached_evaluation_final'))
            final_evaluation=evaluate_cached_continuous(model,a.warmup_trace,a.output/'cached-evaluation-final',
                check_stop=guard,max_rows=a.cached_evaluation_max_rows)
            write_json(a.output/'cached-evaluation-final.json',final_evaluation)
        if a.development_episodes:
            from fly_touhou.continuous_connectome_collector import evaluate_continuous_native_pool
            model.eval().requires_grad_(False);progress(dict(phase='frozen_development'))
            development=evaluate_continuous_native_pool(model,environment,a.output/'development',
                checkpoint_sha256=sha(a.output/'final-parameters.pt'),episodes=20,native_workers=20,
                neural_batch_size=16,seed=2027431601,action_selection='greedy',check_stop=guard,worker_cpus=worker_cpus,
                progress=lambda row:progress(dict(phase='frozen_development',evaluation=row)))
        write_json(a.output/'result.json',dict(schema=SCHEMA,status='completed',costs=costs,development=development,
            algorithm=algorithm,provenance=provenance,trainable_brain=not a.fixed_brain,
            training_stop=training_stop,
            elapsed_seconds=time.monotonic()-started,formal_stability_test=False))
    except BaseException as error:
        recovery_error=None
        if model is not None and optimizer is not None and algorithm is not None and transaction['complete']:
            try:save('stopped-after-update')
            except BaseException as second:recovery_error=repr(second)
        write_json(a.output/'failure.json',dict(schema=SCHEMA,error=repr(error),costs=costs,trainable_brain=not a.fixed_brain,
            update_diagnostics=getattr(error,'continuous_update_diagnostics',None),
            optimizer_transaction_complete=transaction['complete'],checkpoint_error=recovery_error,no_automatic_retry=True))
        raise
    finally:close_training()


def main(argv=None):
    a=parse_args(argv)
    if not a.execute_cuda:print(json.dumps(plan(a),ensure_ascii=False));return
    execute(a)


if __name__=='__main__':main()
