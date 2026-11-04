"""Student-operated NoBomb avoidance guidance, followed by frozen native20.

This is a supervised avoidance phase, not PPO. Three existing critics and
their Adam history are retained without value/return updates. No native or
Torch work occurs unless --execute-cuda is supplied.
"""
from __future__ import annotations
import argparse
import copy
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
SCHEMA = 'continuous-object-NoBomb-guidance-entry-v1'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('training-source', 'environment-config', 'connectome', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--training-source-sha256', required=True)
    p.add_argument('--source-kind', choices=('deep', 'object'), required=True)
    p.add_argument('--source-policy-cost-gradient', choices=('shared', 'supervised-only'), required=True)
    p.add_argument('--decisions', type=int, required=True)
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--learning-rate', type=float, default=3e-5)
    p.add_argument('--geometry-label-budget', type=int, default=2048)
    p.add_argument('--seed', type=int, default=613711)
    p.add_argument('--action-seed', type=int, default=971137)
    p.add_argument('--training-seed', type=int)
    p.add_argument('--evaluation-seed', type=int, default=2027431601)
    p.add_argument('--max-seconds', type=int, required=True)
    p.add_argument('--execute-cuda', action='store_true')
    a = p.parse_args(argv)
    import math
    if (a.decisions < 7680 or a.decisions % 7680 or not 1 <= a.epochs <= 3
            or not math.isfinite(a.learning_rate) or a.learning_rate <= 0
            or not 1 <= a.geometry_label_budget <= 2048 or a.max_seconds < 1
            or any(not 0 <= v < 2**31 for v in (a.seed, a.action_seed, a.evaluation_seed))
            or (a.training_seed is not None and not 0 <= a.training_seed < 2**31)
            or len(a.training_source_sha256) != 64
            or any(c not in '0123456789abcdef' for c in a.training_source_sha256)):
        p.error('Explicit full checkpoint SHA, finite natural30x256 budget and positive learning controls required')
    return a


def plan(a):
    cfg = json.loads(a.environment_config.read_text('utf-8'))
    if (cfg.get('training_mode') != 'full-story-v1' or cfg.get('headless') is not True
            or cfg.get('reward_schema') != 'th14-completion-reward-v3'
            or cfg.get('input_schema') != 'semantic-separated-observed-laser-segments-v5'
            or cfg.get('continues_allowed') is not False or cfg.get('native_frame_horizon') is not None):
        raise ValueError('Same private original-game V5 environment required')
    return dict(schema=SCHEMA, status='prepared_not_executed', Torch_imported='torch' in sys.modules,
        training_source=str(a.training_source), training_source_sha256=a.training_source_sha256,
        source_kind=a.source_kind, source_policy_cost_gradient=a.source_policy_cost_gradient,
        training_seed=cfg.get('seed') if a.training_seed is None else a.training_seed,
        training_seed_override_only=a.training_seed is not None,
        evaluation_environment_seed_configuration_unchanged=True,
        target_policy_cost_gradient='supervised-only', output=str(a.output),
        environment_config=str(a.environment_config), environment_config_sha256=sha(a.environment_config),
        objective='student-operated NoBomb avoidance guidance;NOT PPO',
        PPO_updates=0, GAE_or_returns_computed=False, value_updates=0,
        critic_parameters_and_named_Adam_retained=True,
        student_decisions=a.decisions, worlds=30, steps_per_rollout=256,
        supervised_Adam_max=a.decisions//7680 * a.epochs * 30, epochs=a.epochs,
        geometry_label_budget=a.geometry_label_budget,
        learning_rate=a.learning_rate, critic_LR_changed=False,
        trainable=['object attention', 'VP input', 'full signed directed brain', 'DN actor', 'DN risk'],
        frozen=['CNN/numeric projection', 'DN risk normalization'],
        graph=dict(neurons=138639, edges=15091983, VP_ports=8053, DN_ports=1299, K=8, BPTT=16),
        dynamics='continuous ANN activity;positive learnable edge gains;not LIF or membrane recordings',
        guidance='original safe/future/hit/rule formulas;deathbomb coefficient zero;physical signed4f coefficient.05',
        behavior='student T1 categorical over actual movement0..17;teacher labels only',
        endpoint='avoidance TASK ends on first accepted native state4/Miss or natural clear;not native GameOver',
        native_frame_cap=None, stage_curriculum=False, task_end_reward_added=False,
        reward='existing v3 recorded as diagnostics only;not optimized in this phase',
        native_resets='fresh train-partition original openings at task contact;brain resets only with fresh game',
        encoded384_recomputed_for_learning=True, cached_trainable_outputs_used_for_gradient=False,
        native_workers_closed_before_frozen_evaluation=True,
        frozen_evaluation=dict(episodes=20, workers=20, seed=a.evaluation_seed,
            Bomb=False, first_native_contact=True, frame_cap=None, greedy=True),
        maximum_seconds=a.max_seconds, RAM_or_VRAM_caps=None, CUDA_started=False, native_started=False)


def execute(a, prepared):
    if 'torch' in sys.modules:
        raise RuntimeError('Fresh executor required')
    if a.output.exists():
        raise FileExistsError('Fresh output required;no auto retry')
    if sha(a.training_source) != a.training_source_sha256:
        raise ValueError('Explicit complete source checkpoint identity differs')
    cfg = json.loads(a.environment_config.read_text('utf-8'))
    import psutil
    current = psutil.Process()
    native = Path(cfg['game_exe']).name.lower()
    if any((p.info['name'] or '').lower() == native for p in psutil.process_iter(['name'])):
        raise RuntimeError('Another native job owns the original worlds')
    from fly_touhou.owned_worker_affinity import worker_cpu_pool, spread_owned_workers
    worker_cpus = worker_cpu_pool(current.cpu_affinity())
    current.cpu_affinity(current.cpu_affinity()[-2:])
    current.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    os.environ.update(CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
        OPENBLAS_NUM_THREADS='2', CUBLAS_WORKSPACE_CONFIG=':4096:8',
        FLY_TOUHOU_DESKTOP_NAME='fly-touhou-job-continuous-'+str(os.getpid()))
    from fly_touhou.thrl_durable_records import write_json
    from fly_touhou.thrl_job_paths import configure_native_temp
    a.output.mkdir(parents=True)
    (a.output/'traces').mkdir()
    configure_native_temp(a.output)
    begun = time.monotonic()
    costs = dict(supervision_Adam=0, PPO_Adam=0, student_actions=0, completed_rollouts=0)
    model = optimizer = vector = collector = None
    algorithm = provenance = None
    transaction = dict(complete=True)
    closed = False
    last_checkpoint = None
    launch = dict(prepared, status='running', PID=os.getpid(), create_time=current.create_time(),
        started_UTC=datetime.now(timezone.utc).isoformat(), entry_sha256=sha(__file__))
    write_json(a.output/'launch.json', launch)

    def guard():
        if (a.output/'STOP').exists():
            raise RuntimeError('Operator STOP')
        if time.monotonic()-begun > a.max_seconds:
            raise RuntimeError('Finite wall budget reached;unfinished evaluation is not a score')

    def progress(phase, **kw):
        write_json(a.output/'progress.json', dict(schema=SCHEMA, phase=phase, costs=copy.deepcopy(costs),
            elapsed_seconds=time.monotonic()-begun, PPO_updates=0, value_updates=0, **kw))

    def save(tag):
        nonlocal last_checkpoint
        payload = object_training_checkpoint_payload(model, optimizer, algorithm=algorithm,
            counters=costs, provenance=provenance)
        for suffix, value in (('training', payload), ('parameters', payload['model_payload'])):
            path = a.output/f'{tag}-{suffix}.pt'
            temporary = path.with_suffix('.pt.tmp')
            with temporary.open('wb') as stream:
                torch.save(value, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
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
                errors.extend(str(x) for x in getattr(vector, 'close_errors', []))
            write_json(a.output/'training-cleanup.json', dict(training_closed=True, worlds=30, close_errors=errors))
        return errors

    def critic_fingerprint():
        digest = hashlib.sha256()
        for name, p in model.critic.named_parameters():
            digest.update(name.encode())
            digest.update(p.detach().cpu().contiguous().numpy().tobytes())
            for key, value in sorted(optimizer.state.get(p, {}).items()):
                digest.update(str(key).encode())
                if torch.is_tensor(value):
                    digest.update(value.detach().cpu().contiguous().numpy().tobytes())
                else:
                    digest.update(repr(value).encode())
        return digest.hexdigest()

    try:
        import numpy as np
        import torch
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(False)
        from fly_touhou.continuous_object_learning_model import (
            migrate_deep_training_source, restore_object_training_source, object_training_checkpoint_payload)
        from fly_touhou.continuous_object_avoidance import (
            ObjectAvoidancePool, update_object_avoidance, make_object_avoidance_vector)
        from fly_touhou.fly_gru_aligned_guidance import DEFAULT_CONFIG
        from fly_touhou.thrl_dodge_diagnostics import _shared_binary
        guard()
        with _shared_binary(a.training_source) as stream:
            source = torch.load(stream, map_location='cpu', weights_only=True)
        if a.source_kind == 'deep':
            model, optimizer, inherited, migration = migrate_deep_training_source(source, a.connectome,
                sensory_lr=a.learning_rate, source_policy_cost_gradient=a.source_policy_cost_gradient,
                policy_cost_gradient='supervised-only')
        else:
            model, optimizer, inherited = restore_object_training_source(source, a.connectome)
            if model.policy_cost_gradient != 'supervised-only':
                raise ValueError('Resume the explicit avoidance guidance routing')
            migration = dict(new_parameters=0, full_named_Adam_and_RNG_restored=True)
        source_algorithm = copy.deepcopy(source.get('algorithm', {}))
        del source
        rates_before = {g['name']:g['lr'] for g in optimizer.param_groups}
        for group in optimizer.param_groups:
            if group['name'] != 'critic':
                group['lr'] = a.learning_rate
        guide = replace(DEFAULT_CONFIG, deathbomb_coefficient=0.)
        algorithm = dict(prepared, phase='NoBomb_avoidance_guidance', guidance=asdict(guide),
            policy_cost_gradient='supervised-only', physical_signed4f_coefficient=.05,
            optimizer_group_learning_rates={g['name']:g['lr'] for g in optimizer.param_groups})
        provenance = dict(parent=dict(path=str(a.training_source), sha256=a.training_source_sha256,
            inherited_counters=inherited, algorithm=source_algorithm), migration=migration,
            group_learning_rate_transition=dict(previous=rates_before,
                current=algorithm['optimizer_group_learning_rates'], moments_reset=False),
            source_code_SHA_is_compatibility_gate=False, source_artifact_checksum_verified=True,
            native_midgame_resume=False, full_story_resume=False, teacher_actions=0,
            entry_sha256=sha(__file__), module_sha256={name:sha(ROOT/'src/fly_touhou'/name) for name in (
                'continuous_object_avoidance.py', 'continuous_avoidance_task_env.py',
                'continuous_object_learning_model.py', 'continuous_object_sensory.py')})
        before_critic = critic_fingerprint()
        def pre_step(*unused):
            if any(p.grad is not None for p in model.critic.parameters()):
                raise RuntimeError('Avoidance guidance must not update the three critics')
            transaction['complete'] = False
        def post_step(*unused):
            costs['supervision_Adam'] += 1
            transaction['complete'] = True
            if costs['supervision_Adam'] == 1 or costs['supervision_Adam'] % 10 == 0:
                progress('updating_avoidance_guidance', rollout=costs['completed_rollouts'])
        optimizer.register_step_pre_hook(pre_step)
        optimizer.register_step_post_hook(post_step)
        save('initial')
        launch.update(Torch_imported=True, CUDA_started=True, source_checkpoint_read=True,
            actual_trainable_object_tensors=54, optimizer_instances=1)
        write_json(a.output/'launch.json', launch)
        write_json(a.output/'initialization.json', dict(algorithm=algorithm, provenance=provenance))
        from fly_touhou import thrl_vector
        from fly_touhou.fly_visual_online_collector_continuation import MovementDiagnosticsVector
        from train_fly_visual_long_continuous import staggered_reset_vector
        training_cfg = copy.deepcopy(cfg)
        if a.training_seed is not None:
            training_cfg['seed'] = a.training_seed
        try:
            progress('creating_avoidance_worlds')
            vector = MovementDiagnosticsVector(make_object_avoidance_vector(training_cfg, a.output/'training',
                vector_class=staggered_reset_vector(thrl_vector.THRLNativeVector)))
            write_json(a.output/'worker-affinity-before-reset.json',
                spread_owned_workers(os.getpid(), current.create_time(), worker_cpus))
            collector = ObjectAvoidancePool(model, vector, reference_refiner=None,
                geometry_label_budget=a.geometry_label_budget, action_seed=a.action_seed)
            collector.start()
            launch.update(native_started=True, actual_training_worlds=30)
            write_json(a.output/'launch.json', launch)
            for version in range(a.decisions//7680):
                guard()
                progress('collecting_avoidance', rollout=version)
                rollout = collector.collect(steps=256, policy_version=version, check_stop=guard)
                costs['student_actions'] += int(rollout.actions.numel())
                names = ('static_visual', 'objects', 'features', 'actions', 'old_log_probs',
                    'encoded_visual', 'DN_features',
                    'raw_logits', 'composed_logits', 'predicted_costs', 'action_probs', 'legal_actions',
                    'episode_starts', 'game_ids', 'native_episodes', 'episode_steps',
                    'rule_probabilities', 'rule_weights', 'source_rule_weights',
                    'rule_excluded_no_legal_mass', 'task_terminated', 'native_terminated',
                    'native_reward_vectors', 'truncated')
                arrays = {k:getattr(rollout,k).detach().cpu().numpy() for k in names if hasattr(rollout,k)}
                arrays.update({'task_'+k:np.asarray(getattr(rollout.task_targets,k)) for k in
                    ('margins', 'trusted', 'outcomes', 'unprotected_mask')})
                path = a.output/'traces'/f'rollout-{version:03}.npz'
                with path.with_suffix('.npz.tmp').open('wb') as stream:
                    np.savez_compressed(stream, **arrays)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(path.with_suffix('.npz.tmp'), path)
                write_json(path.with_suffix('.json'), dict(trace_sha256=sha(path), metadata=rollout.metadata,
                    completed_episodes=rollout.completed_episodes, optimizer_updates=0,
                    sampling='actual student NoBomb T1 categorical;not PPO'))
                del arrays
                progress('updating_avoidance_guidance', rollout=version)
                learning = update_object_avoidance(model, optimizer, rollout, epochs=a.epochs,
                    minibatch_seed=a.seed+version, check_stop=guard)
                costs['completed_rollouts'] += 1
                write_json(a.output/'traces'/f'update-{version+1:06}.json', dict(learning=learning,
                    completed_episodes=rollout.completed_episodes, costs=copy.deepcopy(costs)))
                save('last-completed')
                progress('avoidance_guidance_updated', rollout=version, learning=learning)
                del rollout, learning
        finally:
            errors = close_training()
        if errors:
            raise RuntimeError('Native avoidance training cleanup incomplete')
        if critic_fingerprint() != before_critic:
            raise RuntimeError('Critic parameters or Adam history changed during guidance-only stage')
        provenance['three_critic_parameters_and_Adam_bitwise_retained'] = True
        save('final')
        collector = vector = None
        model.eval().requires_grad_(False)
        progress('frozen_NoBomb_development')
        from fly_touhou.continuous_no_bomb_first_contact import make_no_bomb_policy, evaluate_no_bomb
        policy = make_no_bomb_policy(torch, model, family='deep', episodes=20)
        frozen_versions = tuple(p._version for p in model.parameters())
        def frozen_guard():
            guard()
            if tuple(p._version for p in model.parameters()) != frozen_versions:
                raise RuntimeError('Frozen development weights changed')
        report = evaluate_no_bomb(torch, policy, cfg, a.output/'development',
            checkpoint_sha256=sha(a.output/'final-parameters.pt'), episodes=20, workers=20,
            seed=a.evaluation_seed, check_stop=frozen_guard, worker_cpus=worker_cpus,
            progress=lambda row:progress('frozen_NoBomb_development', evaluation=row))
        frozen_guard()
        write_json(a.output/'result.json', dict(schema=SCHEMA, status='completed', costs=costs,
            development=report, algorithm=algorithm, provenance=provenance,
            elapsed_seconds=time.monotonic()-begun, PPO_updates=0, value_updates=0,
            formal_stability_test=False))
    except BaseException as error:
        checkpoint_error = None
        if model is not None and optimizer is not None and algorithm is not None and transaction['complete']:
            try:
                save('stopped-after-update')
            except BaseException as second:
                checkpoint_error = repr(second)
        write_json(a.output/'failure.json', dict(schema=SCHEMA, error=repr(error), costs=costs,
            actual_acknowledged_student_actions=None if collector is None else collector.acknowledged_actions,
            pending_student_actions=None if collector is None else collector.pending_actions,
            update_diagnostics=getattr(error, 'continuous_update_diagnostics', None),
            optimizer_transaction_complete=transaction['complete'],
            latest_recoverable_checkpoint=last_checkpoint, checkpoint_error=checkpoint_error,
            automatic_retry=False))
        raise
    finally:
        close_training()


def main(argv=None):
    a = arguments(argv)
    prepared = plan(a)
    if not a.execute_cuda:
        print(json.dumps(prepared, ensure_ascii=False, indent=2))
        return 0
    execute(a, prepared)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
