"""Prepared deep-aware continuous PPO30 continuation with explicit policy-cost routing.

Default NoTorch plan. The explicit CUDA path reuses the existing native pool,
three-value PPO updater,guidance,physical signed4f and frozen20 evaluator.
No current process/source is modified and no offline activity enters new games.
"""
from __future__ import annotations
import argparse
import copy
from dataclasses import asdict
from datetime import datetime,timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
SCHEMA='continuous-deep128x3-PPO30-cost-routing-entry-v1'
D95=Path('./outputs/engineering/B1-autonomous-dodge-20261024/full-geometry-point-risk-calibration-v1/candidate_frozen.pt')


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('training-source','environment-config','connectome','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--training-source-sha256',required=True)
    p.add_argument('--policy-cost-gradient',choices=('shared','supervised-only'),required=True)
    p.add_argument('--decisions',type=int,required=True)
    p.add_argument('--max-seconds',type=int,required=True)
    p.add_argument('--d95-checkpoint',type=Path,default=D95)
    p.add_argument('--seed',type=int,default=613711)
    p.add_argument('--action-seed',type=int,default=971137)
    p.add_argument('--geometry-label-budget',type=int,default=2048)
    p.add_argument('--development-episodes',type=int,choices=(0,20),default=20)
    p.add_argument('--freeze-after-first-training-clear',action='store_true')
    p.add_argument('--execute-cuda',action='store_true')
    a=p.parse_args(argv)
    if (a.decisions<7680 or a.decisions%7680 or a.max_seconds<1 or not 1<=a.geometry_label_budget<=2048
            or any(not 0<=v<2**31 for v in (a.seed,a.action_seed))
            or len(a.training_source_sha256)!=64 or any(c not in '0123456789abcdef' for c in a.training_source_sha256)):
        p.error('Finite natural30x256 budget,seeds,geometry budget and explicit artifact SHA required')
    return a


def plan(a):
    environment=json.loads(a.environment_config.read_text('utf-8'))
    if (environment.get('training_mode')!='full-story-v1' or environment.get('headless') is not True
            or environment.get('reward_schema')!='th14-completion-reward-v3'
            or environment.get('continues_allowed') is not False or environment.get('first_hit_terminal') is not False
            or environment.get('native_frame_horizon') is not None):
        raise ValueError('Original natural full-story V3 source19 environment required')
    shared=a.policy_cost_gradient=='shared'
    return dict(schema=SCHEMA,status='prepared_not_executed',Torch_imported='torch' in sys.modules,
        source_checkpoint_read=False,array_payloads_read=0,
        source=str(a.training_source),source_sha256=a.training_source_sha256,output=str(a.output),
        environment_config=str(a.environment_config),environment_config_sha256=sha(a.environment_config),
        policy_cost_gradient=a.policy_cost_gradient,full_parameter_PPO_gradient=shared,
        semi_gradient_policy_and_guidance=not shared,
        forward_and_frozen_policy_changed=False,new_parameters_or_inputs=0,
        initial_risk_Jacobian_reduction_claim=False,unique_failure_cause_claim=False,
        source_types='explicit deep128x3 full training/routing subtype;original64 raw/normalized delegate to original restore',
        learnable_scopes=['complete core','VP input','actor','three critics','risk'],perception_frozen=True,
        source_Adam_and_RNG='all restored;critic explicitly reenabled;no moments reset',
        source_offline_activity_used=False,native_midgame_resume=False,
        PPO=dict(decisions=a.decisions,rollouts=a.decisions//7680,worlds=30,steps_per_world=256,
            physical_sampling_groups=[16,14],K=8,BPTT=16,microbatch_real_rows=256,
            epochs_max=3,Adam_max=a.decisions//7680*90,action_count=19,action_temperature=1.,
            gamma=.999,GAE_lambda=.995,clip=.2,value_clip=.2,value_coefficient=.5,
            three_values=['survival','fight','resource'],critic_shared_gradient=.1,
            entropy=.0001,learning_rate=3e-5,global_clip=.5,target_KL=.03,
            old_logp_absolute_tolerance=.0001),
        guidance='unchanged original five formulas/coefficients;no new risk ranking or rewards',
        physical_risk='signed4f original query-mixture Huber+.2BCE;coefficient.05;full risk/core gradients',
        geometry_label_budget=a.geometry_label_budget,reward='original V3 three-component reward',
        native_episode_end='natural GameOver or six-stage clear;no first-hit/stage/frame termination',
        brain_reset='actual new game only;not Miss/stage/update',
        freeze_after_first_training_clear=a.freeze_after_first_training_clear,
        frozen_development_games=a.development_episodes,frozen_development_seed=2027431601,
        formal_stability_success=False,FP32=True,TF32=False,RAM_or_VRAM_caps=None,
        maximum_seconds=a.max_seconds,CUDA_started=False,native_started=False,optimizer_instances=0)


def execute(a,prepared):
    if 'torch' in sys.modules:raise RuntimeError('Fresh explicit executor required')
    if a.output.exists():raise FileExistsError('Fresh output required;no automatic retry')
    environment=json.loads(a.environment_config.read_text('utf-8'))
    import psutil
    current=psutil.Process();native_name=Path(environment['game_exe']).name.lower()
    if any((p.info['name'] or '').lower()==native_name for p in psutil.process_iter(['name'])):
        raise RuntimeError('Existing original worlds;do not overlap or control another job')
    from fly_touhou.owned_worker_affinity import worker_cpu_pool,spread_owned_workers
    worker_cpus=worker_cpu_pool(current.cpu_affinity())
    current.cpu_affinity(current.cpu_affinity()[-2:]);current.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    os.environ.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',FLY_TOUHOU_DESKTOP_NAME='fly-touhou-job-continuous-routing-'+str(os.getpid()))
    from fly_touhou.thrl_durable_records import write_json
    from fly_touhou.thrl_job_paths import configure_native_temp
    a.output.mkdir(parents=True);(a.output/'traces').mkdir();configure_native_temp(a.output)
    begun=time.monotonic();costs=dict(PPO_Adam=0,student_actions=0,completed_rollouts=0,warmup_Adam=0)
    model=optimizer=collector=vector=penalty=None;algorithm=provenance=None;closed=False;cleanup_errors=[];deep=False
    transaction=dict(complete=True);training_stop=None;last_checkpoint=None
    launch=dict(prepared,status='running',PID=os.getpid(),create_time=current.create_time(),
        started_UTC=datetime.now(timezone.utc).isoformat())
    write_json(a.output/'launch.json',launch)
    def guard():
        if (a.output/'STOP').exists():raise RuntimeError('Operator STOP')
        if time.monotonic()-begun>a.max_seconds:raise RuntimeError('Finite wall budget reached')
    def progress(phase,**kwargs):
        write_json(a.output/'progress.json',dict(schema=SCHEMA,phase=phase,costs=copy.deepcopy(costs),
            elapsed_seconds=time.monotonic()-begun,policy_cost_gradient=a.policy_cost_gradient,
            full_parameter_PPO_gradient=a.policy_cost_gradient=='shared',**kwargs))
    def save(tag):
        nonlocal last_checkpoint
        payload=training_checkpoint_payload(model,optimizer,algorithm=algorithm,counters=costs,provenance=provenance)
        if deep:tag_deep_payload(payload,normalization_spec)
        elif normalized:tag_normalized_payload(payload,normalization_spec)
        # Parameters retain the existing inference subtype. The new TRAINING
        # subtype stops an old runner silently ignoring protected routing.
        payload['schema']=DEEP_ROUTING_TRAINING_SCHEMA if deep else ROUTING_TRAINING_SCHEMA
        payload['policy_cost_gradient_protocol']=routing_protocol(a.policy_cost_gradient)
        for suffix,value in (('training',payload),('parameters',payload['model_payload'])):
            path=a.output/f'{tag}-{suffix}.pt';temporary=path.with_suffix('.pt.tmp')
            with temporary.open('wb') as stream:torch.save(value,stream);stream.flush();os.fsync(stream.fileno())
            os.replace(temporary,path)
        last_checkpoint=str(a.output/f'{tag}-training.pt')
    def close_training():
        nonlocal closed
        if not closed:
            closed=True
            if vector is not None:
                try:vector.close()
                except BaseException as error:cleanup_errors.append(repr(error))
                cleanup_errors.extend(str(x) for x in getattr(vector,'close_errors',[]))
        write_json(a.output/'training-cleanup.json',dict(training_closed=True,worlds=30,close_errors=cleanup_errors))
        return cleanup_errors
    try:
        import torch
        torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(False)
        from fly_touhou.continuous_policy_cost_routing import (routing_protocol,
            TRAINING_SCHEMA as ROUTING_TRAINING_SCHEMA)
        from fly_touhou.continuous_deep_policy_cost_routing import (restore_deep_aware_routed_source,
            TRAINING_SCHEMA as DEEP_ROUTING_TRAINING_SCHEMA)
        from fly_touhou.continuous_deep_risk_readout import (tag_deep_payload,
            PARAMETER_SCHEMA as deep_parameter_schema,TOPOLOGY)
        from fly_touhou.continuous_normalized_risk_coordinate import (tag_normalized_payload,
            PARAMETER_SCHEMA as normalization_parameter_schema,BASE_PARAMETER_SCHEMA as raw_parameter_schema)
        from fly_touhou.continuous_connectome_training import (training_checkpoint_payload,
            update_continuous_pool,transition_learning_rate,OLD_LOGP_ABS_TOLERANCE)
        from fly_touhou.fly_gru_aligned_ppo import aligned_ppo_config
        from fly_touhou.fly_gru_aligned_guidance import DEFAULT_CONFIG
        from fly_touhou.fly_gru_risk26_contract import ContinuousCostConfig
        from fly_touhou.continuous_signed_clearance import protocol as signed_protocol
        guard()
        if sha(a.training_source)!=a.training_source_sha256:raise ValueError('Explicit complete source checkpoint identity differs')
        source=torch.load(a.training_source,map_location='cpu',weights_only=True)
        deep=source['model_payload'].get('schema')==deep_parameter_schema
        model,optimizer,inherited,normalized=restore_deep_aware_routed_source(source,a.connectome,
            policy_cost_gradient=a.policy_cost_gradient)
        normalization_spec=source['model_payload'].get('risk_input_normalization')
        cfg=aligned_ppo_config();risk_config=ContinuousCostConfig(coefficient=.05)
        rate_transition=transition_learning_rate(optimizer,cfg.learning_rate,phase='autonomous_PPO_after_full_rule_pretraining')
        source_algorithm=copy.deepcopy(source.get('algorithm',{}))
        algorithm=dict(schema=SCHEMA,phase='autonomous_PPO',core=model.core.config(),
            PPO=asdict(cfg),guidance=asdict(DEFAULT_CONFIG),
            continuous_cost=asdict(risk_config),risk_supervision='signed-clearance4f',
            risk_supervision_protocol=signed_protocol(),brain_BPTT=16,neural_width=16,minibatch_true_rows=256,
            PPO_loss_scale=1.,fixed_brain=False,policy_cost_gradient=a.policy_cost_gradient,
            full_parameter_PPO_gradient=a.policy_cost_gradient=='shared',
            policy_cost_gradient_protocol=routing_protocol(a.policy_cost_gradient),
            actor_policy='following-DN raw + following-DN cost beta8;stock legal T1',
            optimizer_group_learning_rates={g['name']:g['lr'] for g in optimizer.param_groups},
            freeze_after_first_training_clear=a.freeze_after_first_training_clear,
            initial_old_logp_absolute_tolerance=OLD_LOGP_ABS_TOLERANCE)
        if normalized:algorithm['risk_input_coordinates']=copy.deepcopy(normalization_spec)
        if deep:algorithm['risk_topology']=copy.deepcopy(TOPOLOGY)
        if 'risk_pretraining_protocol' in source_algorithm:
            algorithm['risk_pretraining_protocol']=copy.deepcopy(source_algorithm['risk_pretraining_protocol'])
        source_files=('continuous_deep_policy_cost_routing.py','continuous_deep_risk_readout.py',
            'continuous_policy_cost_routing.py','continuous_normalized_risk_coordinate.py',
            'continuous_connectome.py','continuous_sparse.py','continuous_connectome_model.py',
            'continuous_connectome_collector.py','continuous_connectome_training.py','continuous_signed_clearance.py')
        provenance=dict(parent=dict(path=str(a.training_source),sha256=a.training_source_sha256,
                counters=inherited,algorithm=source_algorithm),
            entry_sha256=sha(__file__),module_sha256={name:sha(ROOT/'src/fly_touhou'/name) for name in source_files},
            source_code_SHA_is_compatibility_gate=False,source_artifact_checksum_verified=True,
            source_Adam_and_global_RNG_fully_restored=True,
            critic_parameters_reenabled=True,critic_Adam_reset=False,
            source_offline_activity_discarded_for_new_native_games=bool(source.get('offline_resume')),
            native_midgame_resume=False,learning_rate_transition=rate_transition,
            gradient_routing_transition=dict(source_mode=source_algorithm.get('policy_cost_gradient','unmarked source gradient'),
                target_mode=a.policy_cost_gradient,optimizer_moments_reset=False),
            no_new_actor_or_risk_or_input_parameters=True,normalized_risk=normalized,
            deep128x3=deep,risk_parameter_tensors=len(list(model.risk_head.parameters())),
            risk_main_Adam_states_restored=sum(p in optimizer.state for p in model.risk_head.parameters()),
            all_risk_Adam_states_retained_without_reset=True)
        del source
        def pre_step(*unused):transaction['complete']=False
        def post_step(*unused):costs['PPO_Adam']+=1;transaction['complete']=True
        optimizer.register_step_pre_hook(pre_step);optimizer.register_step_post_hook(post_step)
        launch.update(Torch_imported=True,source_checkpoint_read=True,CUDA_started=True,optimizer_instances=1,
            actual_initial_old_logp_absolute_tolerance=OLD_LOGP_ABS_TOLERANCE)
        write_json(a.output/'launch.json',launch)
        write_json(a.output/'initialization.json',dict(algorithm=algorithm,provenance=provenance))
        save('initial');progress('creating_natural_worlds')
        from fly_touhou.continuous_connectome_collector import (ContinuousConnectomeOnlinePool,
            record_continuous_activity,evaluate_continuous_native_pool)
        from fly_touhou.fly_d95_reference_refinement import D95RiskPenalty
        from fly_touhou.fly_task_reference_pool import make_task_reference_vector
        from fly_touhou.fly_visual_online_collector_continuation import MovementDiagnosticsVector
        from fly_touhou.thrl_vector import THRLNativeVector
        from train_fly_visual_long_continuous import staggered_reset_vector
        from train_continuous_connectome import first_full_training_clear
        # The frozen labels-only risk factory must not consume restored RNG.
        with torch.random.fork_rng(devices=[0]):penalty=D95RiskPenalty.from_checkpoint(a.d95_checkpoint,device=model.device)
        vector=MovementDiagnosticsVector(make_task_reference_vector(environment,a.output/'training',
            vector_class=staggered_reset_vector(THRLNativeVector)))
        write_json(a.output/'worker-affinity-before-reset.json',spread_owned_workers(os.getpid(),current.create_time(),worker_cpus))
        collector=ContinuousConnectomeOnlinePool(model,vector,reference_refiner=penalty,
            geometry_label_budget=a.geometry_label_budget,action_seed=a.action_seed)
        collector.start()
        launch.update(native_started=True,actual_training_worlds=30)
        write_json(a.output/'launch.json',launch)
        for version in range(a.decisions//7680):
            guard();progress('collecting',rollout=version)
            rollout=collector.collect(steps=256,policy_version=version,check_stop=guard)
            costs['student_actions']+=int(rollout.actions.numel())
            import numpy as np
            names=('encoded_visual','features','legal_actions','actions','old_log_probs','values',
                'reward_vectors','reward_raw_vectors','rewards','bootstrap_values','terminated','truncated',
                'episode_starts','game_ids','native_episodes','episode_steps',
                'rule_probabilities','rule_weights','deathbomb_mask','d95_risk_costs',
                'd95_risk_queried_mask','d95_risk_valid_mask','raw_logits','composed_logits',
                'predicted_costs','action_probs')
            arrays={k:getattr(rollout,k).detach().cpu().numpy() for k in names}
            arrays.update({'task_'+k:np.asarray(getattr(rollout.task_targets,k)) for k in ('margins','trusted','outcomes','unprotected_mask')})
            path=a.output/'traces'/f'rollout-{version:03}.npz';temporary=path.with_suffix('.npz.tmp')
            with temporary.open('wb') as stream:np.savez(stream,**arrays);stream.flush();os.fsync(stream.fileno())
            os.replace(temporary,path)
            write_json(path.with_suffix('.json'),dict(trace_sha256=sha(path),metadata=rollout.metadata,
                completed_episodes=rollout.completed_episodes,policy_cost_gradient=a.policy_cost_gradient,
                raw_maps_objects_or_old_neural_state_saved=False))
            del arrays
            try:activity=record_continuous_activity(model,rollout,a.output/'neural-diagnostics'/f'rollout-{version:06}')
            except (ValueError,OSError,RuntimeError) as e:activity=dict(status='diagnostic_unavailable',error=repr(e),training_rejected=False)
            progress('updating',rollout=version)
            learning=update_continuous_pool(model,optimizer,rollout,cfg=cfg,check_stop=guard,
                minibatch_seed=a.seed+version,cost_config=risk_config,risk_supervision='signed-clearance4f')
            costs['completed_rollouts']+=1
            write_json(a.output/'traces'/f'update-{version+1:06}.json',dict(learning=learning,activity=activity,
                completed_episodes=rollout.completed_episodes,costs=copy.deepcopy(costs),
                policy_cost_gradient=a.policy_cost_gradient,full_parameter_PPO_gradient=a.policy_cost_gradient=='shared'))
            save('last-completed');progress('updated',rollout=version,learning=learning)
            if a.freeze_after_first_training_clear:
                trigger=first_full_training_clear(rollout.completed_episodes,environment)
                if trigger is not None:
                    training_stop=dict(trigger,completed_rollout=version+1,paid_counters=copy.deepcopy(costs),
                        checkpoint_saved_before_trigger=last_checkpoint)
                    provenance['training_stop']=copy.deepcopy(training_stop)
                    write_json(a.output/'training-clear-trigger.json',training_stop)
                    del rollout,learning;gc.collect();break
            del rollout,learning;gc.collect()
        save('final')
        if close_training():raise RuntimeError('Native training cleanup incomplete')
        collector=vector=penalty=None;gc.collect();development=None
        if a.development_episodes:
            model.eval().requires_grad_(False);progress('frozen_development')
            development=evaluate_continuous_native_pool(model,environment,a.output/'development',
                checkpoint_sha256=sha(a.output/'final-parameters.pt'),episodes=20,native_workers=20,
                neural_batch_size=16,seed=2027431601,action_selection='greedy',check_stop=guard,worker_cpus=worker_cpus,
                progress=lambda row:progress('frozen_development',evaluation=row))
            parameter_schema=(deep_parameter_schema if deep else
                (normalization_parameter_schema if normalized else raw_parameter_schema))
            development['source_parameter_schema']=parameter_schema
            development['episode_protocol']['source_schema']=parameter_schema
            development['continuous_connectome_policy']['schema']=parameter_schema
            write_json(a.output/'development'/'report.json',development)
        write_json(a.output/'result.json',dict(schema=SCHEMA,status='completed',costs=costs,development=development,
            algorithm=algorithm,provenance=provenance,training_stop=training_stop,
            policy_cost_gradient=a.policy_cost_gradient,full_parameter_PPO_gradient=a.policy_cost_gradient=='shared',
            elapsed_seconds=time.monotonic()-begun,formal_stability_test=False))
    except BaseException as error:
        checkpoint_error=None
        if model is not None and optimizer is not None and algorithm is not None and transaction['complete']:
            try:save('stopped-after-update')
            except BaseException as second:checkpoint_error=repr(second)
        write_json(a.output/'failure.json',dict(schema=SCHEMA,error=repr(error),costs=costs,
            update_diagnostics=getattr(error,'continuous_update_diagnostics',None),
            optimizer_transaction_complete=transaction['complete'],latest_recoverable_checkpoint=last_checkpoint,
            checkpoint_error=checkpoint_error,automatic_retry=False));raise
    finally:close_training()


def main(argv=None):
    a=parse_args(argv);prepared=plan(a)
    if not a.execute_cuda:print(json.dumps(prepared,ensure_ascii=False,indent=2));return 0
    execute(a,prepared);return 0


if __name__=='__main__':raise SystemExit(main())
