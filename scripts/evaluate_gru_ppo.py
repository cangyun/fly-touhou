"""Independent fixed-checkpoint M4 evaluation, with no training or selection.

seal/aggregate/--validate-only perform file and CPU bookkeeping only. Only
the evaluate subcommand without --validate-only starts CUDA and owned native
games. No command changes a checkpoint or a training configuration.
"""
from __future__ import annotations

import argparse
from functools import partial
from importlib.metadata import distribution, version
import json
from pathlib import Path
import signal
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

import numpy as np

from fly_touhou.atomic_json_io import atomic_write_json
from fly_touhou.avoidance_rewards import (SURVIVAL01, SURVIVAL_REPORT_SCHEMA, plan_mode,
    specification as reward_specification)
from fly_touhou.engineering_contract import file_record, write_json_new
from fly_touhou.gru_ppo_evaluation import (
    EvaluationStopped, FixedQuotaVecMixin, POLICIES, REPORT_SCHEMA, SEEDS, aggregate_reports,
    audit_native_logs, coverage_diagnostics, evaluate_vector_group, load_plan, read_json, seal_plan, verify_record,
)


def atomic_json(path, value):
    return atomic_write_json(path, value)


def sources():
    # NativeAvoidanceEnv imports the reader/controller lazily; protecting just
    # the neural policy misses those executable Python and injected JS sources.
    package = ROOT/'src/fly_touhou'
    dependencies = {
        'stable-baselines3': ('stable_baselines3/common/base_class.py',
            'stable_baselines3/common/vec_env/base_vec_env.py',
            'stable_baselines3/common/vec_env/subproc_vec_env.py'),
        'sb3-contrib': ('sb3_contrib/common/recurrent/policies.py',
            'sb3_contrib/ppo_recurrent/ppo_recurrent.py')}
    return [Path(__file__), *sorted(path for path in package.rglob('*')
        if path.is_file() and path.suffix in ('.py', '.js')),
        *(Path(distribution(name).locate_file(path)) for name, paths in dependencies.items() for path in paths)]


def runtime_dependencies():
    return {name: version(name) for name in ('torch', 'numpy', 'stable-baselines3',
        'sb3-contrib', 'gymnasium', 'psutil', 'frida')}


def verify_runtime_sources(plan):
    committed = {Path(record['path']).resolve(): record for record in plan['runtime_sources']}
    actual = {path.resolve(): file_record(path) for path in sources()}
    if committed != actual:
        raise ValueError('Evaluation must execute the complete precommitted Python/JS source tree')
    if plan.get('runtime_dependencies') != runtime_dependencies():
        raise ValueError('Evaluation runtime library versions differ from the sealed plan')


def seal(args):
    plan = seal_plan(seed=args.seed, protocol=args.base_protocol, first_run=args.first_run,
        trained_run=args.trained_run, checkpoint=args.checkpoint, last_checkpoint=args.last_checkpoint,
        selection_evidence=args.selection_evidence, selection_rule=args.selection_rule,
        workers=args.workers, source_paths=sources(), dependency_versions=runtime_dependencies())
    write_json_new(args.output, plan)
    return {'status': 'evaluation_plan_sealed_no_games_started', 'plan': file_record(args.output),
            'planned_counted_episodes': plan['planned_counted_episodes'], 'formal_M4_accepted': False}


def check_loaded_model(model, plan, item):
    from fly_touhou.standard_gru_ppo import NativeGRUPolicy, optimizer_steps
    if (not isinstance(model.policy, NativeGRUPolicy) or model.seed != plan['seed']
            or model.num_timesteps != item['decisions'] or optimizer_steps(model) != item['optimizer_steps']
            or str(model.device) != 'cuda' and not str(model.device).startswith('cuda:')
            or model.observation_space.shape != (1912,) or model.action_space.n != 18
            or model.policy.lstm_actor.hidden_size != 128 or model.policy.lstm_critic.hidden_size != 128
            or tuple(model.policy.features_extractor.bounds) != (-184., 184., 32., 432.)
            or len(model.policy.features_extractor.feature_names) != 1912
            or model.policy.features_extractor.feature_names[-1] != 'task_progress'):
        raise ValueError('Loaded model does not implement the sealed ordinary-GRU M4 contract')


def evaluate(args):
    plan = load_plan(args.plan)
    reward_mode = plan_mode(plan)
    report_schema = SURVIVAL_REPORT_SCHEMA if reward_mode == SURVIVAL01 else REPORT_SCHEMA
    verify_runtime_sources(plan)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json_new(output/'config.json', {'schema': report_schema, 'plan': file_record(args.plan),
        'plan_contents': plan, 'wall_cap_seconds': args.wall_cap_seconds,
        'validation_only': args.validate_only, 'training_started': False})
    if args.validate_only:
        result = {'status': 'validated_only', 'original_game_started': False, 'CUDA_started': False,
                  'formal_M4_accepted': False}
        write_json_new(output/'report.json', result)
        return result, 0
    # Heavy runtime imports occur only after explicitly choosing real evaluation.
    import torch
    from sb3_contrib import RecurrentPPO
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from fly_touhou.gru_imitation import state_subset_sha256
    from fly_touhou.standard_gru_ppo import NativeAvoidanceEnv, optimizer_steps
    class EvaluationVecEnv(FixedQuotaVecMixin, SubprocVecEnv):
        pass
    if not torch.cuda.is_available():
        raise RuntimeError('M4 model inference requires CUDA; no CPU fallback')
    torch.set_num_threads(2)
    torch.manual_seed(plan['evaluation_seed'])
    torch.cuda.manual_seed_all(plan['evaluation_seed'])
    started, run_id = time.monotonic(), uuid.uuid4().hex
    groups, exceptions, model_checks = {}, [], {}
    interrupted = [False]
    previous = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.signal(signum, lambda *_: interrupted.__setitem__(0, True))
    def stopping():
        return interrupted[0] or (output/'STOP').exists() or (output.parent/'STOP').exists()
    status, code = 'finished', 0
    current, stream = None, None
    episode_files = {}
    try:
        with (output/'events.jsonl').open('x', encoding='utf-8', buffering=1) as events:
            progress_counts = {}
            def event(kind, **fields):
                if kind == 'episode_finished':
                    stream.write(json.dumps(fields['row'], ensure_ascii=False, allow_nan=False)+'\n')
                elif kind == 'group_progress':
                    marker = (fields['episodes'], fields['extra_completed_episodes'])
                    if progress_counts.get(fields['policy_kind']) != marker:
                        progress_counts[fields['policy_kind']] = marker
                        atomic_json(output/'progress.json', dict(fields, status='evaluating',
                            wall_seconds=time.monotonic()-started))
                else:
                    events.write(json.dumps(dict(kind=kind, **fields), ensure_ascii=False, allow_nan=False)+'\n')
            for current in plan['execution_order']:
                if stopping():
                    raise EvaluationStopped('STOP before the next evaluation policy')
                model = None
                item = None
                if current not in ('static', 'random'):
                    item = plan['checkpoints']['last_trained' if current == 'trained_last' else current]
                    model = RecurrentPPO.load(item['model']['path'], device='cuda')
                    check_loaded_model(model, plan, item)
                    model.policy.set_training_mode(False)
                    before = (state_subset_sha256(model.policy, private_critic=False),
                              state_subset_sha256(model.policy, private_critic=True), optimizer_steps(model))
                    def predict(observations, state, starts):
                        with torch.no_grad():
                            return model.predict(observations, state=state, episode_start=starts, deterministic=True)
                elif current == 'static':
                    def predict(observations, state, starts):
                        return np.zeros(plan['workers'], dtype=np.int64), None
                else:
                    generators = [np.random.default_rng(plan['evaluation_seed']+10000+worker)
                                  for worker in range(plan['workers'])]
                    def predict(observations, state, starts):
                        return np.asarray([rng.integers(0, 18) for rng in generators], dtype=np.int64), None
                def env_factory():
                    return EvaluationVecEnv([partial(NativeAvoidanceEnv, plan['protocol']['path'],
                        str(output/current/'native'/f'worker{worker}'), worker, 'formal_test',
                        task_reward=reward_mode, append_task_time=True, capture_death_windows=True)
                        for worker in range(plan['workers'])], start_method='spawn')
                event('group_started', policy_kind=current,
                      model=item['model'] if item is not None else None)
                log_path = output/f'{current}_episodes.jsonl'
                try:
                    with log_path.open('x', encoding='utf-8', buffering=1) as stream:
                        group = evaluate_vector_group(env_factory, predict, policy_kind=current,
                            workers=plan['workers'], run_id=run_id, stop_requested=stopping,
                            deadline=started+args.wall_cap_seconds, event=event,
                            native_output=output/current/'native', reward_mode=reward_mode)
                        groups[current] = group
                finally:
                    if log_path.is_file():
                        episode_files[current] = file_record(log_path)
                if model is not None:
                    after = (state_subset_sha256(model.policy, private_critic=False),
                             state_subset_sha256(model.policy, private_critic=True), optimizer_steps(model))
                    model_checks[current] = {'unchanged': before == after,
                        'actor_sha256': before[0], 'private_critic_sha256': before[1],
                        'optimizer_steps_before': before[2], 'optimizer_steps_after': after[2]}
                    if before != after:
                        raise ValueError('An evaluation changed model tensors or optimizer counters')
                event('group_finished', policy_kind=current, episodes=group['episodes'],
                      successes=group['successes'], extra_completed_episodes=group['extra_completed_episodes'])
                del model
        load_plan(args.plan)  # Inputs and the committed source must still match.
        verify_runtime_sources(plan)
    except BaseException as error:
        if hasattr(error, 'partial_evaluation_group') and current is not None:
            groups[current] = error.partial_evaluation_group
        status = ('operator_stop' if stopping() else 'wall_cap_stop') if isinstance(error, EvaluationStopped) else 'failed'
        code = 2 if isinstance(error, EvaluationStopped) else 1
        exceptions.append({'kind': type(error).__name__, 'reason': str(error),
            'traceback': traceback.format_exc(), 'system_fault_counted_as_hit': False})
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    result = {'schema': report_schema, 'status': status, 'run_id': run_id, 'seed': plan['seed'],
        'device': 'cuda', 'plan': file_record(args.plan), 'groups': groups, 'episode_files': episode_files,
        'all_models_unchanged': status == 'finished' and all(value['unchanged'] for value in model_checks.values()),
        'model_checks': model_checks, 'exceptions': exceptions, 'PPO_updates_during_evaluation': 0,
        'teacher_actions': 0, 'safety_takeovers': 0, 'bomb_actions': 0,
        'deploy_action_rule': 'argmax', 'formal_test_results_used_for_selection': False,
        'formal_M4_accepted': False, 'formal_M4_requires_three_actual_training_seeds': list(SEEDS),
        'native_game_rng_controlled': False, 'wall_seconds': time.monotonic()-started,
        'all_completed_extra_episodes_retained': True,
        'coverage_diagnostics': coverage_diagnostics([dict(row, policy_kind=kind, seed=plan['seed'])
            for kind, group in groups.items()
            for row in group.get('native_log_audit', {}).get('episode_fingerprints', [])]),
        'last_checkpoint_role': 'secondary_only' if 'trained_last' in plan['policies'] else 'same_as_primary'}
    if reward_mode == SURVIVAL01:
        result.update(reward_mode=SURVIVAL01, reward_specification=reward_specification(SURVIVAL01))
    write_json_new(output/'report.json', result)
    return {'status': status, 'report': file_record(output/'report.json'), 'formal_M4_accepted': False}, code


def aggregate(args):
    items, evidence = [], []
    plans = {load_plan(path)['seed']: (path, load_plan(path)) for path in args.plans}
    if len(plans) != 3:
        raise ValueError('Exactly three distinct sealed seed plans are required')
    seen_seeds = set()
    for path in args.reports:
        report = read_json(path)
        seed = report.get('seed')
        if seed not in plans or seed in seen_seeds:
            raise ValueError('Reports must cover distinct actual seeds11/23/37')
        seen_seeds.add(seed)
        plan_path, plan = plans[seed]
        if report.get('plan') != file_record(plan_path):
            raise ValueError('Evaluation report refers to another sealed checkpoint list')
        for kind, record in report.get('episode_files', {}).items():
            stream = verify_record(record)
            logged = [json.loads(line) for line in stream.read_text(encoding='utf-8').splitlines() if line]
            group = report['groups'][kind]
            declared = group['episode_results']+group.get('extra_episode_results', [])
            if ({row['evaluation_episode_id']: row for row in logged}
                    != {row['evaluation_episode_id']: row for row in declared} or len(logged) != len(declared)):
                raise ValueError('Completed or extra native episodes were omitted from the report')
            for native_record in group.get('native_log_audit', {}).get('files', []):
                verify_record(native_record)
            actual_native_audit = audit_native_logs(Path(path).resolve().parent/kind/'native',
                group, require_complete=True, reward_mode=plan_mode(plan))
            if actual_native_audit != group.get('native_log_audit'):
                raise ValueError('Native raw-log reconciliation differs from the declared episode/RNG/fingerprint audit')
        if set(report.get('episode_files', {})) != set(plan['policies']):
            raise ValueError('Every completed policy requires its retained full episode stream')
        items.append((plan, report))
        evidence.append({'plan': file_record(plan_path), 'report': file_record(path)})
    result = aggregate_reports(items)
    result['evidence'] = evidence
    result['secondary_last_checkpoint_reports'] = {
        str(plan['seed']): report['groups'].get('trained_last', {'same_as_primary_trained': True})
        for plan, report in items}
    write_json_new(args.output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)
    seal_parser = subparsers.add_parser('seal', help='Pin actual checkpoints and method before formal tests; CPU only')
    seal_parser.add_argument('--seed', type=int, choices=SEEDS, required=True)
    seal_parser.add_argument('--base-protocol', type=Path, required=True)
    seal_parser.add_argument('--first-run', type=Path, required=True)
    seal_parser.add_argument('--trained-run', type=Path, required=True)
    seal_parser.add_argument('--checkpoint', type=Path, required=True)
    seal_parser.add_argument('--last-checkpoint', type=Path, required=True)
    seal_parser.add_argument('--selection-evidence', type=Path, required=True)
    seal_parser.add_argument('--selection-rule', choices=('last', 'development_success_then_mean_first_hit_then_earlier'), required=True)
    seal_parser.add_argument('--workers', type=int, choices=(1, 2), default=2)
    seal_parser.add_argument('--output', type=Path, required=True)
    evaluation_parser = subparsers.add_parser('evaluate', help='Run the precommitted native test; no learner updates')
    evaluation_parser.add_argument('--plan', type=Path, required=True)
    evaluation_parser.add_argument('--output', type=Path, required=True)
    evaluation_parser.add_argument('--wall-cap-seconds', type=float, default=7200.)
    evaluation_parser.add_argument('--validate-only', action='store_true')
    aggregate_parser = subparsers.add_parser('aggregate', help='CPU-only formal three-seed summary')
    aggregate_parser.add_argument('--plans', type=Path, nargs=3, required=True)
    aggregate_parser.add_argument('--reports', type=Path, nargs=3, required=True)
    aggregate_parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'seal':
        result, code = seal(args), 0
    elif args.command == 'aggregate':
        result, code = aggregate(args), 0
    else:
        if args.wall_cap_seconds <= 0:
            parser.error('The real evaluation must have a positive declared wall cap')
        result, code = evaluate(args)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
