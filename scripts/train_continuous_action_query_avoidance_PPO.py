"""Action-conditioned sensory-query comparison, with unchanged NoBomb PPO.

Only the object frontend gains a zero-initialized action-query residual.
All decisions still traverse the complete connectome and descending neurons.
Source migration is explicit; no execution occurs in default plan mode.
"""
from __future__ import annotations
import argparse
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
SCHEMA = 'continuous-action-query-object-NoBomb-survival-PPO-entry-v1'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('training-source','environment-config','connectome','output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--training-source-sha256', required=True)
    p.add_argument('--source-kind', choices=('migrate-object','resume-action-query'), required=True)
    p.add_argument('--decisions', type=int, required=True)
    p.add_argument('--max-seconds', type=int, required=True)
    p.add_argument('--learning-rate', type=float, default=3e-5)
    p.add_argument('--seed', type=int, default=613719)
    p.add_argument('--action-seed', type=int, default=971151)
    p.add_argument('--training-seed', type=int, default=2026102801)
    p.add_argument('--evaluation-seed', type=int, default=2027431601)
    p.add_argument('--geometry-label-budget', type=int, default=2048)
    p.add_argument('--execute-cuda', action='store_true')
    a = p.parse_args(argv)
    if (a.decisions < 7680 or a.decisions % 7680 or a.max_seconds < 1
            or not math.isfinite(a.learning_rate) or a.learning_rate <= 0
            or not 1 <= a.geometry_label_budget <= 2048
            or any(not 0 <= x < 2**31 for x in (a.seed,a.action_seed,a.training_seed,a.evaluation_seed))
            or len(a.training_source_sha256) != 64
            or any(x not in '0123456789abcdef' for x in a.training_source_sha256)):
        p.error('Complete source SHA and finite original30x256 budget required')
    return a


def plan(a):
    c = json.loads(a.environment_config.read_text('utf-8'))
    if (c.get('training_mode') != 'full-story-v1' or c.get('headless') is not True
            or c.get('input_schema') != 'semantic-separated-observed-laser-segments-v5'
            or c.get('reward_schema') != 'th14-completion-reward-v3'
            or c.get('continues_allowed') is not False or c.get('native_frame_horizon') is not None):
        raise ValueError('Same original private V5 backend and unbounded original opening required')
    return dict(schema=SCHEMA, status='prepared_not_executed', Torch_imported='torch' in sys.modules,
        source_checkpoint_deserialized=False, training_source=str(a.training_source),
        training_source_sha256=a.training_source_sha256, output=str(a.output),
        environment_config=str(a.environment_config), environment_config_sha256=sha(a.environment_config),
        mode='avoidance-only PPO;full-story Bomb PPO remains stopped',
        source_kind=a.source_kind,
        sensory_change='18 fixed movement-conditioned queries;zero128 residual;no risk/teacher input',
        new_action_query_parameters=186624,
        sensory_width_preserved=384,full_graph_and_DN_readout_preserved=True,
        student_decisions=a.decisions, rollouts=a.decisions//7680, worlds=30,
        steps_per_world=256, neural_groups=[16,14], K=8, brain_BPTT=16,
        PPO=dict(epochs_max=3, Adam_max=a.decisions//7680*90, gamma=.999,
            GAE_lambda=.995, clip=.2, value_clip=.2, value_coefficient=.5,
            critic_shared_gradient=.1, objective_weights=[1.,0.,0.],
            entropy=1e-4, learning_rate=a.learning_rate, grad_clip=.5, target_KL=.03),
        action_support='source19 with Bomb18 masked;movement0..17 only,T1 categorical',
        task_end='first accepted native state4/Miss or natural clear;not fabricated native GameOver',
        task_reward='native v3 vulnerable time+.001/4frames+corner;one total-2 at first contact;natural clear+5',
        other_targets='native fight/resource retained for three critics;actor advantage survival only',
        auxiliary_value_gradients_can_affect_brain=True,
        guidance='unchanged safe/future/hit/rule;deathbomb0;signed4f.05',
        all_three_value_heads_retained_and_updated=True, source_named_Adam_and_RNG_restored=True,
        policy_cost_gradient='preserve source mode;explicitly report semi-gradient if supervised-only',
        teacher_actions=0, safety_takeovers=0, D95_model_or_risk_queries=0,
        trainable=['object attention','action-query residual','VP input','full continuous graph','DN actor','DN risk','three DN critics'],
        training_seed=a.training_seed, evaluation_seed=a.evaluation_seed,
        evaluation_environment_config_unchanged=True, native_frame_cap=None,
        frozen_evaluation_episodes=20, frozen_evaluation_workers=20,
        maximum_seconds=a.max_seconds, RAM_or_VRAM_caps=None, CUDA_started=False, native_started=False)


def execute(a, prepared):
    if 'torch' in sys.modules:
        raise RuntimeError('Fresh executor required')
    if a.output.exists():
        raise FileExistsError('Fresh output required;no automatic retry')
    if sha(a.training_source) != a.training_source_sha256:
        raise ValueError('Complete training source identity differs')
    c = json.loads(a.environment_config.read_text('utf-8'))
    import psutil
    current = psutil.Process()
    if any((p.info['name'] or '').lower() == Path(c['game_exe']).name.lower()
            for p in psutil.process_iter(['name'])):
        raise RuntimeError('Another native job owns the original worlds')
    from fly_touhou.owned_worker_affinity import worker_cpu_pool, spread_owned_workers
    worker_cpus = worker_cpu_pool(current.cpu_affinity())
    current.cpu_affinity(current.cpu_affinity()[-2:])
    current.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    os.environ.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
        OPENBLAS_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8',
        FLY_TOUHOU_DESKTOP_NAME='fly-touhou-job-continuous-'+str(os.getpid()))
    from fly_touhou.thrl_durable_records import write_json
    from fly_touhou.thrl_job_paths import configure_native_temp
    a.output.mkdir(parents=True)
    (a.output/'traces').mkdir()
    configure_native_temp(a.output)
    begun = time.monotonic()
    costs = dict(PPO_Adam=0,student_actions=0,completed_rollouts=0)
    model = optimizer = vector = collector = None
    algorithm = provenance = None
    closed = False
    transaction = dict(complete=True)
    last_checkpoint = None
    launch = dict(prepared,status='running',PID=os.getpid(),create_time=current.create_time(),
        started_UTC=datetime.now(timezone.utc).isoformat(),entry_sha256=sha(__file__))
    write_json(a.output/'launch.json',launch)

    def guard():
        if (a.output/'STOP').exists():
            raise RuntimeError('Operator STOP')
        if time.monotonic()-begun > a.max_seconds:
            raise RuntimeError('Finite wall budget;unfinished evaluation is not a score')

    def progress(phase, **kw):
        write_json(a.output/'progress.json',dict(schema=SCHEMA,phase=phase,costs=copy.deepcopy(costs),
            elapsed_seconds=time.monotonic()-begun,full_story_PPO_resumed=False,**kw))

    def save(tag):
        nonlocal last_checkpoint
        p = action_query_training_checkpoint_payload(model,optimizer,algorithm=algorithm,
            counters=costs,provenance=provenance)
        for suffix,value in (('training',p),('parameters',p['model_payload'])):
            path = a.output/f'{tag}-{suffix}.pt'
            tmp = path.with_suffix('.pt.tmp')
            with tmp.open('wb') as stream:
                torch.save(value,stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp,path)
        last_checkpoint = str(a.output/f'{tag}-training.pt')

    def close_training():
        nonlocal closed
        errors = []
        if not closed:
            closed = True
            if vector is not None:
                try:
                    vector.close()
                except BaseException as error:
                    errors.append(repr(error))
                errors.extend(str(x) for x in getattr(vector,'close_errors',[]))
            write_json(a.output/'training-cleanup.json',dict(training_closed=True,
                worlds=30 if vector is not None else 0,close_errors=errors))
        return errors

    try:
        import numpy as np
        import torch
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(False)
        from fly_touhou.continuous_action_query_learning_model import (
            migrate_object_training_source,restore_action_query_training_source,
            action_query_training_checkpoint_payload)
        from fly_touhou.continuous_object_avoidance_ppo import (
            ObjectAvoidancePPOPool,make_object_avoidance_PPO_vector,update_object_avoidance_PPO)
        from fly_touhou.fly_gru_aligned_ppo import aligned_ppo_config
        from fly_touhou.thrl_dodge_diagnostics import _shared_binary
        guard()
        with _shared_binary(a.training_source) as stream:
            source = torch.load(stream,map_location='cpu',weights_only=True)
        if a.source_kind == 'migrate-object':
            model,optimizer,inherited,migration = migrate_object_training_source(
                source,a.connectome,query_lr=a.learning_rate)
        else:
            model,optimizer,inherited = restore_action_query_training_source(source,a.connectome)
            migration = dict(already_typed_action_query_source=True,source_RNG_restored=True)
        ppo_cfg = aligned_ppo_config(learning_rate=a.learning_rate,objective_weights=(1.,0.,0.))
        rates_before = {g['name']:g['lr'] for g in optimizer.param_groups}
        for group in optimizer.param_groups:
            group['lr'] = a.learning_rate
        algorithm = dict(prepared,phase='avoidance_only_PPO',PPO=asdict(ppo_cfg),
            policy_cost_gradient=model.policy_cost_gradient,
            full_parameter_PPO_gradient=model.policy_cost_gradient=='shared',
            three_critics_reenabled_with_retained_parameters_and_Adam=True)
        provenance = dict(parent=dict(path=str(a.training_source),sha256=a.training_source_sha256,
            counters=inherited,algorithm=copy.deepcopy(source.get('algorithm',{}))),
            source_named_Adam_and_global_RNG_restored=True,native_midgame_resume=False,
            sensory_action_query_migration=migration,
            teacher_actions=0,policy_cost_gradient_preserved=True,
            learning_rate_transition=dict(previous=rates_before,
                current={g['name']:g['lr'] for g in optimizer.param_groups},moments_reset=False),
            entry_sha256=sha(__file__),module_sha256={name:sha(ROOT/'src/fly_touhou'/name) for name in (
                'continuous_object_avoidance_ppo.py','continuous_avoidance_ppo_env.py',
                'continuous_object_avoidance.py','continuous_object_learning_model.py',
                'continuous_object_sensory.py','continuous_action_query_learning_model.py',
                'action_query_residual_objects.py')})
        del source
        def pre_step(*unused):
            transaction['complete'] = False
        def post_step(*unused):
            costs['PPO_Adam'] += 1
            transaction['complete'] = True
            if costs['PPO_Adam'] == 1 or costs['PPO_Adam'] % 10 == 0:
                progress('updating_avoidance_PPO',rollout=costs['completed_rollouts'])
        optimizer.register_step_pre_hook(pre_step)
        optimizer.register_step_post_hook(post_step)
        save('initial')
        launch.update(Torch_imported=True,CUDA_started=True,source_checkpoint_read=True,
            optimizer_instances=1,policy_cost_gradient=model.policy_cost_gradient)
        write_json(a.output/'launch.json',launch)
        write_json(a.output/'initialization.json',dict(algorithm=algorithm,provenance=provenance))
        from fly_touhou.thrl_vector import THRLNativeVector
        from fly_touhou.fly_visual_online_collector_continuation import MovementDiagnosticsVector
        from train_fly_visual_long_continuous import staggered_reset_vector
        training_cfg = copy.deepcopy(c)
        training_cfg['seed'] = a.training_seed
        progress('creating_avoidance_PPO_worlds')
        vector = MovementDiagnosticsVector(make_object_avoidance_PPO_vector(training_cfg,a.output/'training',
            vector_class=staggered_reset_vector(THRLNativeVector)))
        write_json(a.output/'worker-affinity-before-reset.json',
            spread_owned_workers(os.getpid(),current.create_time(),worker_cpus))
        collector = ObjectAvoidancePPOPool(model,vector,reference_refiner=None,
            geometry_label_budget=a.geometry_label_budget,action_seed=a.action_seed)
        collector.start()
        launch.update(Torch_imported=True,CUDA_started=True,native_started=True,
            actual_training_worlds=30,optimizer_instances=1,policy_cost_gradient=model.policy_cost_gradient)
        write_json(a.output/'launch.json',launch)
        for version in range(a.decisions//7680):
            guard()
            progress('collecting_avoidance_PPO',rollout=version)
            rollout = collector.collect(steps=256,policy_version=version,check_stop=guard)
            costs['student_actions'] += int(rollout.actions.numel())
            names = ('static_visual','objects','encoded_visual','features','DN_features',
                'raw_logits','composed_logits','predicted_costs','legal_actions','actions',
                'old_log_probs','action_probs','values','bootstrap_values','reward_vectors',
                'task_reward_vectors','native_reward_vectors','task_terminated','native_terminated',
                'native_reward_raw_vectors','first_observed_contact',
                'terminated','truncated','episode_starts','game_ids','native_episodes','episode_steps',
                'rule_probabilities','rule_weights','source_rule_weights','rule_excluded_no_legal_mass')
            arrays = {k:getattr(rollout,k).detach().cpu().numpy() for k in names if hasattr(rollout,k)}
            arrays.update({'task_'+k:np.asarray(getattr(rollout.task_targets,k)) for k in
                ('margins','trusted','outcomes','unprotected_mask')})
            path = a.output/'traces'/f'rollout-{version:03}.npz'
            with path.with_suffix('.npz.tmp').open('wb') as stream:
                np.savez_compressed(stream,**arrays)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(path.with_suffix('.npz.tmp'),path)
            write_json(path.with_suffix('.json'),dict(trace_sha256=sha(path),metadata=rollout.metadata,
                completed_episodes=rollout.completed_episodes,
                task_reward_components=rollout.reward_components,
                native_reward_components=rollout.native_reward_components,
                actor_objective_weights=[1.,0.,0.],full_story_PPO=False))
            del arrays
            progress('updating_avoidance_PPO',rollout=version)
            learning = update_object_avoidance_PPO(model,optimizer,rollout,cfg=ppo_cfg,
                minibatch_seed=a.seed+version,check_stop=guard)
            costs['completed_rollouts'] += 1
            write_json(a.output/'traces'/f'update-{version+1:06}.json',dict(learning=learning,
                completed_episodes=rollout.completed_episodes,costs=copy.deepcopy(costs)))
            save('last-completed')
            progress('avoidance_PPO_updated',rollout=version,learning=learning)
            del rollout,learning
        if close_training():
            raise RuntimeError('Native avoidance PPO cleanup incomplete')
        collector = vector = None
        save('final')
        model.eval().requires_grad_(False)
        from fly_touhou.continuous_no_bomb_first_contact import make_no_bomb_policy,evaluate_no_bomb
        policy = make_no_bomb_policy(torch,model,family='deep',episodes=20)
        versions = tuple(p._version for p in model.parameters())
        def frozen_guard():
            guard()
            if tuple(p._version for p in model.parameters()) != versions:
                raise RuntimeError('Frozen development weights changed')
        progress('frozen_NoBomb_development')
        report = evaluate_no_bomb(torch,policy,c,a.output/'development',
            checkpoint_sha256=sha(a.output/'final-parameters.pt'),episodes=20,workers=20,
            seed=a.evaluation_seed,check_stop=frozen_guard,worker_cpus=worker_cpus,
            progress=lambda row:progress('frozen_NoBomb_development',evaluation=row))
        frozen_guard()
        report.update(source_family='action-query-object-learning',source_parameter_schema=model.configuration()['schema'])
        write_json(a.output/'development/report.json',report)
        write_json(a.output/'result.json',dict(schema=SCHEMA,status='completed',costs=costs,
            development=report,algorithm=algorithm,provenance=provenance,
            full_story_PPO_resumed=False,formal_stability_test=False,
            elapsed_seconds=time.monotonic()-begun))
    except BaseException as error:
        checkpoint_error = None
        if model is not None and optimizer is not None and algorithm is not None and transaction['complete']:
            try:
                save('stopped-after-update')
            except BaseException as second:
                checkpoint_error = repr(second)
        write_json(a.output/'failure.json',dict(schema=SCHEMA,error=repr(error),costs=costs,
            update_diagnostics=getattr(error,'continuous_update_diagnostics',None),
            optimizer_transaction_complete=transaction['complete'],
            actual_acknowledged_student_actions=None if collector is None else collector.acknowledged_actions,
            pending_student_actions=None if collector is None else collector.pending_actions,
            latest_recoverable_checkpoint=last_checkpoint,checkpoint_error=checkpoint_error,
            automatic_retry=False))
        raise
    finally:
        close_training()


def main(argv=None):
    a = arguments(argv)
    prepared = plan(a)
    if not a.execute_cuda:
        print(json.dumps(prepared,ensure_ascii=False,indent=2))
        return 0
    execute(a,prepared)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
