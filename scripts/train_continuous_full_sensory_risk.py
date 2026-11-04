"""Prepared student-visited NoBomb risk-only stage;default is a NoTorch plan.

Restore a genuine Full/FullPair training source. Optimize only the original
unscaled signed4f physical loss,never PPO/guidance/value/ranking. No automatic
PPO continuation,service or monitor/config change is part of this entry.
"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
SCHEMA='continuous-full-sensory-risk-only-NoBomb-stage-entry-v1'
PATCH_VERSION='dedicated-risk-v1-entry-v2'


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def cumulative_counters(inherited,stage_costs):
    """Retain all source counters;only add the three costs paid by this stage."""
    result=copy.deepcopy(inherited)
    for key in ('supervised_Adam','student_actions','completed_rollouts'):
        previous=result.get(key,0);added=stage_costs[key]
        if type(previous) is not int or type(added) is not int or min(previous,added)<0:
            raise ValueError('Exact nonnegative cumulative stage counters required:'+key)
        result[key]=previous+added
    return result


def arguments(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('training-source','environment-config','connectome','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--training-source-sha256',required=True)
    p.add_argument('--source-kind',choices=('full','full-pair'),required=True)
    p.add_argument('--optimizer-mode',choices=('preserve','dedicated-risk-v1'),default='dedicated-risk-v1')
    p.add_argument('--max-grad-norm',type=float,choices=(.5,5.),default=None)
    p.add_argument('--decisions',type=int,default=61440)
    p.add_argument('--epochs',type=int,choices=(3,),default=3)
    p.add_argument('--max-seconds',type=int,required=True)
    p.add_argument('--seed',type=int,default=613727)
    p.add_argument('--action-seed',type=int,default=971159)
    p.add_argument('--training-seed',type=int,default=2026102805)
    p.add_argument('--evaluation-seed',type=int,default=2027431601)
    p.add_argument('--evaluation-episodes',type=int,default=20)
    p.add_argument('--evaluation-workers',type=int,default=20)
    p.add_argument('--geometry-label-budget',type=int,default=2048)
    p.add_argument('--min-disk-free-gib',type=float,default=4.)
    p.add_argument('--execute',action='store_true')
    a=p.parse_args(argv)
    if a.max_grad_norm is None:a.max_grad_norm=5. if a.optimizer_mode=='dedicated-risk-v1' else .5
    if (a.decisions<7680 or a.decisions%7680 or a.max_seconds<1
            or not 1<=a.geometry_label_budget<=2048
            or not 1<=a.evaluation_episodes<=20 or not 1<=a.evaluation_workers<=min(20,a.evaluation_episodes)
            or not 0<=a.min_disk_free_gib<float('inf')
            or any(not 0<=n<2**31 for n in (a.seed,a.action_seed,a.training_seed,a.evaluation_seed))
            or len(a.training_source_sha256)!=64 or any(c not in '0123456789abcdef' for c in a.training_source_sha256)):
        p.error('Exact source SHA,finite30x256/full three-epoch budget and explicit finite controls required')
    return a


def plan(a):
    c=json.loads(a.environment_config.read_text(encoding='utf-8'))
    if (c.get('training_mode')!='full-story-v1' or c.get('headless') is not True
            or c.get('input_schema')!='semantic-separated-observed-laser-segments-v5'
            or c.get('reward_schema')!='th14-completion-reward-v3' or c.get('continues_allowed') is not False
            or c.get('native_frame_horizon') is not None):
        raise ValueError('Existing original private V5/noContinue/native-unbounded configuration required')
    return dict(schema=SCHEMA,patch_version=PATCH_VERSION,status='prepared_not_executed',Torch_imported='torch' in sys.modules,
        training_source=str(a.training_source),training_source_sha256=a.training_source_sha256,
        source_kind=a.source_kind,source_checkpoint_deserialized=False,source_SHA_verified=False,
        output=str(a.output),environment_config=str(a.environment_config),environment_config_sha256=sha(a.environment_config),
        mode='student-visited NoBomb first-native-contact risk-only supervised stage;NOT PPO/fullstory training',
        loss=dict(name='original signed_clearance4f',field='loss',coefficient=1.,PPO=False,guide=False,value=False,ranking=False,GAE=False),
        epochs=3,minibatches_per_epoch=30,Adam_max_per_rollout=90,supervised_Adam_max=a.decisions//7680*90,
        empty_label_minibatches='explicitly skipped;no fabricated paid Adam',KL_gate=False,
        student_decisions=a.decisions,rollouts=a.decisions//7680,worlds=30,steps_per_world=256,
        neural_groups=[16,14],K=8,brain_BPTT=16,canonical_frontend_batch=30,
        perception_branches=dict(map=True,spatial=True,numeric=True,object=True,query=True,pair_geometry=a.source_kind=='full-pair'),
        optimizer_mode=a.optimizer_mode,max_grad_norm=a.max_grad_norm,
        optimizer_transition=dict(mode=a.optimizer_mode,status='planned_not_applied',
            source_complete_named_Adam_restored_first=True,optimizer_object_reused=True,
            loss_path_Adam='clear complete per-parameter Adam state including step' if a.optimizer_mode=='dedicated-risk-v1' else 'preserve source state',
            learning_rates=dict(risk=.001,input_brain_sensory=.0003) if a.optimizer_mode=='dedicated-risk-v1' else 'preserve source group settings',
            actor_critic_groups_and_Adam='preserve exactly',parameters_flags_and_RNG='preserve exactly'),
        learning_rate='dedicated risk1e-3;other loss-path3e-4;actor/critic unchanged' if a.optimizer_mode=='dedicated-risk-v1'
            else 'preserve every saved source group setting;no stage LR override',
        actor_and_three_critics='retained/trainable flags unchanged;gradNone;parameters and named Adam must remain exact',
        trainable_loss_path='raw sensing -> complete graph -> DN -> risk signed4f',
        raw_replay='FP32 f25/maps16x92x96/objects96x29;281/384 evidence only',
        raw_map_archive='existing bounded ZIP1 writer;no re-encoding/quantization;atomic index',
        action_support='student T1 categorical movement0..17;Bomb18 forbidden',
        endpoint='first measured native state4/Miss or natural clear;not fabricated native GameOver',
        native_frame_cap=None,teacher_actions_executed=0,D95_model_queries=0,
        PPO_updates=0,value_updates=0,automatic_PPO_resume=False,
        training_seed=a.training_seed,evaluation_seed=a.evaluation_seed,
        frozen_evaluation_episodes=a.evaluation_episodes,frozen_evaluation_workers=a.evaluation_workers,
        max_seconds=a.max_seconds,min_disk_free_gib=a.min_disk_free_gib,RAM_or_VRAM_caps=None,
        CUDA_started=False,native_started=False)


def execute(a,prepared):
    if 'torch' in sys.modules:raise RuntimeError('Fresh executor required;Torch only after explicit --execute')
    if a.output.exists():raise FileExistsError('Fresh output required;no auto retry/overwrite')
    if sha(a.training_source)!=a.training_source_sha256:raise ValueError('Frozen complete source SHA differs')
    c=json.loads(a.environment_config.read_text(encoding='utf-8'))
    import psutil
    current=psutil.Process()
    if any((p.info['name'] or '').lower()==Path(c['game_exe']).name.lower() for p in psutil.process_iter(['name'])):
        raise RuntimeError('Existing native job;do not overlap or take over worlds')
    from fly_touhou.owned_worker_affinity import worker_cpu_pool,spread_owned_workers
    worker_cpus=worker_cpu_pool(current.cpu_affinity())
    current.cpu_affinity(current.cpu_affinity()[-2:]);current.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    os.environ.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',FLY_TOUHOU_DESKTOP_NAME='fly-touhou-job-continuous-risk-'+str(os.getpid()))
    from fly_touhou.thrl_durable_records import write_json
    from fly_touhou.thrl_job_paths import configure_native_temp
    a.output.mkdir(parents=True);(a.output/'traces').mkdir();configure_native_temp(a.output)
    begun=time.monotonic();costs=dict(supervised_Adam=0,student_actions=0,completed_rollouts=0)
    model=optimizer=vector=collector=None
    algorithm=provenance=checkpoint_saver=None;actual_transition=None
    transaction=dict(complete=True);last_checkpoint=None;closed=False
    actor_critic_before=None;inherited={}
    launch=dict(prepared,status='running',PID=os.getpid(),create_time=current.create_time(),
        started_UTC=datetime.now(timezone.utc).isoformat(),entry_sha256=sha(__file__))
    write_json(a.output/'launch.json',launch)

    def guard():
        if (a.output/'STOP').exists():raise RuntimeError('Operator STOP')
        if time.monotonic()-begun>a.max_seconds:raise RuntimeError('Finite wall budget;unfinished evaluation is not a score')
        if shutil.disk_usage(a.output).free<a.min_disk_free_gib*1024**3:raise RuntimeError('Output disk reserve reached')

    def progress(phase,**kw):
        write_json(a.output/'progress.json',dict(schema=SCHEMA,phase=phase,costs=copy.deepcopy(costs),
            elapsed_seconds=time.monotonic()-begun,PPO_updates=0,value_updates=0,teacher_actions_executed=0,**kw))

    def save(tag):
        nonlocal last_checkpoint
        payload=checkpoint_saver(model,optimizer,algorithm=algorithm,
            counters=cumulative_counters(inherited,costs),provenance=provenance)
        for suffix,value in (('training',payload),('parameters',payload['model_payload'])):
            path=a.output/f'{tag}-{suffix}.pt';temp=path.with_suffix('.pt.tmp')
            with temp.open('wb') as stream:torch.save(value,stream);stream.flush();os.fsync(stream.fileno())
            os.replace(temp,path)
        last_checkpoint=str(a.output/f'{tag}-training.pt')

    def close_training():
        nonlocal closed
        errors=[]
        if not closed:
            closed=True
            if vector is not None:
                try:vector.close()
                except BaseException as error:errors.append(repr(error))
                errors.extend(str(x) for x in getattr(vector,'close_errors',[]))
            write_json(a.output/'training-cleanup.json',dict(training_closed=True,
                worlds=30 if vector is not None else 0,close_errors=errors))
        return errors

    try:
        import numpy as np
        import torch
        torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(False)
        from fly_touhou.thrl_dodge_diagnostics import _shared_binary
        from fly_touhou.continuous_full_sensory_ppo import FullSensoryPPOPool,write_raw_map_archive
        from fly_touhou.continuous_full_sensory_risk_pretraining import update_full_sensory_risk,_nonupdated_identity
        from fly_touhou.continuous_risk_optimizer_transition import transition_risk_stage_optimizer
        from fly_touhou.continuous_object_avoidance_ppo import make_object_avoidance_PPO_vector
        if a.source_kind=='full':
            from fly_touhou.continuous_full_sensory_model import restore_full_sensory_training_source as restore_source
            from fly_touhou.continuous_full_sensory_model import full_sensory_training_checkpoint_payload as checkpoint_saver
        else:
            from fly_touhou.continuous_full_pair_geometry_model import restore_full_pair_training_source as restore_source
            from fly_touhou.continuous_full_pair_geometry_model import full_pair_training_checkpoint_payload as checkpoint_saver
        guard()
        with _shared_binary(a.training_source) as stream:source=torch.load(stream,map_location='cpu',weights_only=True)
        model,optimizer,inherited=restore_source(source,a.connectome)
        actor_critic_before=_nonupdated_identity(model,optimizer)
        actual_transition=transition_risk_stage_optimizer(model,optimizer,mode=a.optimizer_mode)
        if actor_critic_before!=_nonupdated_identity(model,optimizer):
            raise RuntimeError('Optimizer transition changed actor/threecritic parameters or named Adam')
        write_json(a.output/'optimizer-transition.json',actual_transition)
        algorithm=dict(prepared,status='executing',source_SHA_verified=True,Torch_imported=True,
            source_checkpoint_deserialized=True,CUDA_started=True,
            source_named_Adam_and_RNG_restored=True,actor_critic_updated=False,
            optimizer_transition=actual_transition,
            policy_cost_gradient=model.policy_cost_gradient,
            effective_optimizer_groups=[{key:group[key] for key in ('name','lr','eps')} for group in optimizer.param_groups])
        provenance=dict(parent=dict(path=str(a.training_source),sha256=a.training_source_sha256,
            counters=inherited,algorithm=copy.deepcopy(source.get('algorithm',{}))),
            native_midgame_resume=False,learning_rates_and_moments_preserved=a.optimizer_mode=='preserve',
            complete_source_restored_before_explicit_phase_transition=True,
            optimizer_transition=actual_transition,
            updater='continuous_full_sensory_risk_pretraining.update_full_sensory_risk',
            source_parameter_schema=model.configuration()['schema'],module_sha256={name:sha(ROOT/'src/fly_touhou'/name) for name in (
                'continuous_full_sensory_risk_pretraining.py','continuous_full_sensory_ppo.py','continuous_full_sensory_model.py',
                'continuous_full_pair_geometry_model.py','continuous_signed_clearance.py','continuous_avoidance_ppo_env.py',
                'continuous_risk_optimizer_transition.py')})
        del source
        write_json(a.output/'actor-critic-before.json',actor_critic_before)
        excluded=tuple(model.actor.parameters())+tuple(model.critic.parameters())
        def pre_step(*unused):
            if any(p.grad is not None for p in excluded):raise RuntimeError('Risk stage must not step any actor/critic gradient')
            transaction['complete']=False
        def post_step(*unused):
            costs['supervised_Adam']+=1;transaction['complete']=True
            if costs['supervised_Adam']==1 or costs['supervised_Adam']%10==0:
                progress('updating_risk_only',rollout=costs['completed_rollouts'])
        optimizer.register_step_pre_hook(pre_step);optimizer.register_step_post_hook(post_step)
        save('initial')
        write_json(a.output/'initialization.json',dict(algorithm=algorithm,provenance=provenance))
        launch.update(Torch_imported=True,CUDA_started=True,source_SHA_verified=True,source_checkpoint_deserialized=True,
            optimizer_instances=1,algorithm=algorithm,actual_transition=actual_transition)
        write_json(a.output/'launch.json',launch)
        from fly_touhou.thrl_vector import THRLNativeVector
        from fly_touhou.fly_visual_online_collector_continuation import MovementDiagnosticsVector
        from train_fly_visual_long_continuous import staggered_reset_vector
        training_cfg=copy.deepcopy(c);training_cfg['seed']=a.training_seed
        progress('creating_student_worlds')
        vector=MovementDiagnosticsVector(make_object_avoidance_PPO_vector(training_cfg,a.output/'training',
            vector_class=staggered_reset_vector(THRLNativeVector)))
        write_json(a.output/'worker-affinity-before-reset.json',spread_owned_workers(os.getpid(),current.create_time(),worker_cpus))
        collector=FullSensoryPPOPool(model,vector,reference_refiner=None,
            geometry_label_budget=a.geometry_label_budget,action_seed=a.action_seed)
        collector.start();launch.update(native_started=True,actual_training_worlds=30);write_json(a.output/'launch.json',launch)
        for version in range(a.decisions//7680):
            guard();progress('collecting_student_risk_states',rollout=version)
            rollout=collector.collect(steps=256,policy_version=version,check_stop=guard)
            costs['student_actions']+=int(rollout.actions.numel())
            archive=write_raw_map_archive(rollout.raw_maps,a.output/'traces'/f'rollout-{version:03}-raw-maps.zip',check_stop=guard)
            names=('static_visual','objects','encoded_visual','features','DN_features','raw_logits','composed_logits','predicted_costs',
                'legal_actions','actions','old_log_probs','action_probs','values','bootstrap_values','reward_vectors',
                'task_reward_vectors','native_reward_vectors','native_reward_raw_vectors','first_observed_contact',
                'task_terminated','native_terminated','terminated','truncated','episode_starts','game_ids','native_episodes','episode_steps',
                'rule_probabilities','rule_weights','source_rule_weights','rule_excluded_no_legal_mass')
            arrays={key:getattr(rollout,key).detach().cpu().numpy() for key in names if hasattr(rollout,key)}
            arrays.update({'task_'+key:np.asarray(getattr(rollout.task_targets,key)) for key in ('margins','trusted','outcomes','unprotected_mask')})
            path=a.output/'traces'/f'rollout-{version:03}.npz';temp=path.with_suffix('.npz.tmp')
            with temp.open('wb') as stream:np.savez_compressed(stream,**arrays);stream.flush();os.fsync(stream.fileno())
            os.replace(temp,path)
            write_json(path.with_suffix('.json'),dict(entry_schema=SCHEMA,trace_sha256=sha(path),metadata=rollout.metadata,
                optimization_scope='risk-only physical signed4f coefficient1;collector PPO-shaped values/rewards are evidence only',
                PPO_updates=0,value_updates=0,teacher_actions_executed=0,completed_episodes=rollout.completed_episodes,
                task_reward_components=rollout.reward_components,native_reward_components=rollout.native_reward_components,
                raw_map_archive=archive,raw_perception_recomputed=True))
            del arrays
            progress('updating_risk_only',rollout=version)
            paid_before=costs['supervised_Adam']
            learning=update_full_sensory_risk(model,optimizer,rollout,epochs=a.epochs,
                minibatch_seed=a.seed+version,check_stop=guard,use_checkpoint=True,max_grad_norm=a.max_grad_norm)
            if costs['supervised_Adam']-paid_before!=learning['optimizer_updates']:
                raise RuntimeError('Paid Adam post-hook and updater accounting differ')
            costs['completed_rollouts']+=1
            write_json(a.output/'traces'/f'update-{version+1:06}.json',dict(schema=SCHEMA,learning=learning,
                costs=copy.deepcopy(costs),completed_episodes=rollout.completed_episodes))
            save('last-completed')
            progress('risk_rollout_completed',rollout=version,
                actor_critic_parameters_and_Adam_exact=learning['actor_critic_parameters_and_Adam_exact'],
                empty_label_minibatches=learning['empty_label_minibatches'])
            del rollout,learning
        if close_training():raise RuntimeError('Student-world cleanup incomplete')
        collector=vector=None
        actor_critic_after=_nonupdated_identity(model,optimizer)
        write_json(a.output/'actor-critic-after.json',actor_critic_after)
        if actor_critic_before!=actor_critic_after:raise RuntimeError('Stage changed actor/threecritic parameters or named Adam')
        identity_files=dict(before=dict(path=str(a.output/'actor-critic-before.json'),sha256=sha(a.output/'actor-critic-before.json')),
            after=dict(path=str(a.output/'actor-critic-after.json'),sha256=sha(a.output/'actor-critic-after.json')))
        save('final')
        # Release the last update's zero/unused grad buffers before frozen eval;
        # never clear CUDA allocator behind an active collector or mid-update.
        optimizer.zero_grad(set_to_none=True)
        import gc
        gc.collect();torch.cuda.empty_cache()
        model.eval().requires_grad_(False)
        from fly_touhou.continuous_no_bomb_first_contact import make_no_bomb_policy,evaluate_no_bomb
        policy=make_no_bomb_policy(torch,model,family='deep',episodes=a.evaluation_episodes)
        versions=tuple(p._version for p in model.parameters())
        def frozen_guard():
            guard()
            if tuple(p._version for p in model.parameters())!=versions:raise RuntimeError('Frozen evaluation parameters changed')
        progress('frozen_NoBomb_development',actor_critic_parameters_and_Adam_exact=True)
        report=evaluate_no_bomb(torch,policy,c,a.output/'development',checkpoint_sha256=sha(a.output/'final-parameters.pt'),
            episodes=a.evaluation_episodes,workers=a.evaluation_workers,seed=a.evaluation_seed,check_stop=frozen_guard,
            worker_cpus=worker_cpus,progress=lambda row:progress('frozen_NoBomb_development',evaluation=row))
        frozen_guard();report.update(source_family=a.source_kind,source_parameter_schema=model.configuration()['schema'],risk_only_stage=True)
        write_json(a.output/'development/report.json',report)
        write_json(a.output/'result.json',dict(schema=SCHEMA,status='completed',algorithm=algorithm,provenance=provenance,
            patch_version=PATCH_VERSION,optimizer_mode=a.optimizer_mode,max_grad_norm=a.max_grad_norm,actual_transition=actual_transition,
            costs=costs,cumulative_counters=cumulative_counters(inherited,costs),
            PPO_updates=0,value_updates=0,teacher_actions_executed=0,
            actor_critic_parameters_and_Adam_exact=True,actor_critic_identity_files=identity_files,
            development=report,automatic_PPO_resume=False,formal_stability_test=False,elapsed_seconds=time.monotonic()-begun))
    except BaseException as error:
        checkpoint_error=None
        if model is not None and optimizer is not None and algorithm is not None and transaction['complete']:
            try:save('stopped-after-update')
            except BaseException as second:checkpoint_error=repr(second)
        write_json(a.output/'failure.json',dict(schema=SCHEMA,status='failed',error=repr(error),costs=costs,
            patch_version=PATCH_VERSION,optimizer_mode=a.optimizer_mode,max_grad_norm=a.max_grad_norm,actual_transition=actual_transition,
            cumulative_counters=cumulative_counters(inherited,costs),
            PPO_updates=0,value_updates=0,teacher_actions_executed=0,
            update_diagnostics=getattr(error,'full_sensory_risk_update_diagnostics',None),
            optimizer_transaction_complete=transaction['complete'],latest_recoverable_checkpoint=last_checkpoint,
            checkpoint_error=checkpoint_error,actual_acknowledged_student_actions=None if collector is None else collector.acknowledged_actions,
            pending_student_actions=None if collector is None else collector.pending_actions,automatic_retry=False))
        raise
    finally:
        close_training()


def main(argv=None):
    a=arguments(argv);prepared=plan(a)
    if not a.execute:print(json.dumps(prepared,ensure_ascii=False,indent=2));return 0
    execute(a,prepared);return 0


if __name__=='__main__':raise SystemExit(main())
