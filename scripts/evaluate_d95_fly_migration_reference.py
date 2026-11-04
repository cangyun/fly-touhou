"""One D95/source19 migration reference; reuse the frozen final-PPO evaluator.

Default inspect hashes files and validates completed metadata without Torch,
checkpoint deserialization or native launch. Only explicit CUDA modes load D95.
No model, reward, game speed, evaluator, optimizer or controller is implemented here.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import importlib
import json
import os
import sys
import traceback

PLAN_SCHEMA = 'D95-fly-migration-paired20-reference-plan-v1'
RESULT_SCHEMA = 'D95-fly-migration-paired20-reference-result-v1'
CHECKPOINT_SHA = 'd95fe710068483fc81d8525d89bccf72e61d209c7e1bb442ea66d022b9c1d06c'
RUNTIME_SHA = '16a4602ab8c150c66820ed354a598824654cf3fa15e8b3b8e716188d9227c385'
INPUT_SCHEMA = 'semantic-separated-observed-laser-segments-v5'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def pin(path, expected):
    if sha(path) != expected:
        raise ValueError('Pinned file changed: ' + str(path))


def validate_protocol(plan):
    if (plan.get('schema') != PLAN_SCHEMA or plan.get('checkpoint_sha256') != CHECKPOINT_SHA
            or plan.get('source_manifest_sha256') != RUNTIME_SHA or plan.get('runtime_files') != 502
            or plan.get('episodes') != 20 or plan.get('workers') != 20
            or plan.get('maximum_original_worlds') != 30 or plan.get('new_optimizer_updates') != 0
            or plan.get('policy_actions') != 19 or plan.get('autonomous_Bomb') is not True
            or plan.get('continues') != 0 or plan.get('native_frame_horizon', 'missing') is not None
            or plan.get('action_selection') != 'greedy' or plan.get('action_temperature') != .25
            or plan.get('native_seed_split') != 'development' or plan.get('study_seed') != 2026102601
            or plan.get('evaluation_seed') != 2027431401
            or type(plan.get('capture_failure_traces')) is not bool
            or plan.get('trace_history') != 128):
        raise ValueError('Require exactly one D95,20 matched development games,greedy T.25,source19 and no optimization')
    seeds = plan.get('expected_native_seeds', [])
    if (len(seeds) != 20 or len(set(seeds)) != 20
            or any(type(seed) is not int or not 0 <= seed < 65536 or seed % 10 not in (6, 7) for seed in seeds)):
        raise ValueError('Require the twenty original development seeds')
    jobs = plan.get('episode_jobs', [])
    if (len(jobs) != 20 or any(job.get('game') != i
            or any(type(job.get(key)) is not int or not 0 <= job[key] < 2**32
                   for key in ('environment_seed', 'action_seed')) for i, job in enumerate(jobs))):
        raise ValueError('Require all original episode,environment and action seed identities')


def verify_runtime(plan):
    runtime = Path(plan['source_runtime']).resolve()
    manifest_path = runtime / 'source_manifest.json'
    pin(manifest_path, plan['source_manifest_sha256'])
    files = read(manifest_path)['files']
    if len(files) != plan['runtime_files']:
        raise ValueError('Frozen runtime file count changed')
    seen = set()
    for item in files:
        path = (runtime / item['path']).resolve()
        if not path.is_relative_to(runtime) or path in seen:
            raise ValueError('Unsafe or duplicate frozen runtime file')
        seen.add(path)
        pin(path, item['sha256'])
    return runtime, len(files)


def require_frozen_module(module, runtime):
    if not getattr(module, '__file__', None) or not Path(module.__file__).resolve().is_relative_to(runtime):
        raise ValueError('Imported module is outside the qualified frozen runtime: ' + module.__name__)


def install_runtime(runtime):
    for name, module in tuple(sys.modules.items()):
        if name == 'fly_touhou' or name.startswith('fly_touhou.') or name == 'train_thrl_native':
            require_frozen_module(module, runtime)
    paths = [str(runtime / 'src'), str(runtime / 'scripts')]
    sys.path[:0] = paths
    os.environ['PYTHONPATH'] = os.pathsep.join(paths)


def validate_baseline(plan, report, episode_plan):
    rows = report.get('episode_results', [])
    if (report.get('episodes') != 20 or report.get('clears') != 8 or len(rows) != 20
            or report.get('policy_frozen') is not True or report.get('PPO_updates') != 0
            or any(report.get(key) != 0 for key in ('continues', 'teacher_actions', 'safety_takeovers'))
            or report.get('action_selection') != 'greedy' or report.get('action_temperature') != .25
            or report.get('native_seed_split') != 'development'
            or episode_plan.get('workers') != 20 or episode_plan.get('action_selection') != 'greedy'
            or episode_plan.get('action_temperature') != .25
            or episode_plan.get('RNG_protocol') != 'episode-seeded-argmax-v1'
            or episode_plan.get('jobs') != plan['episode_jobs']):
        raise ValueError('Reuse only the sealed complete greedy T.25 PPO20 report')
    for index, (row, job, native) in enumerate(zip(rows, plan['episode_jobs'], plan['expected_native_seeds'])):
        if (row.get('evaluation_game') != index + 1 or job.get('game') != index
                or row.get('evaluation_seed') != job.get('environment_seed')
                or row.get('action_seed') != job.get('action_seed') or row.get('native_seed') != native
                or row.get('outcome') not in ('clear', 'gameover') or row.get('full_story_episode') is not True
                or row.get('action_temperature') != .25
                or type(row.get('evaluation_decisions')) is not int or row['evaluation_decisions'] < 1):
            raise ValueError('PPO original episode identity or true terminal changed')
    if sum(row['outcome'] == 'clear' for row in rows) != 8:
        raise ValueError('PPO clear count differs from actual complete rows')


def inspect(plan):
    validate_protocol(plan)
    pin(__file__, plan['entry_sha256'])
    runtime, count = verify_runtime(plan)
    for field in ('checkpoint', 'environment_config', 'ppo_report', 'ppo_episode_plan',
                  'ppo_native_cleanup', 'ppo_training_cleanup', 'ppo_study_result', 'ppo_launch'):
        pin(plan[field], plan[field + '_sha256'])
    config = read(plan['environment_config'])
    if (Path(config['source_runtime']).resolve() != runtime
            or config.get('source_manifest_sha256') != RUNTIME_SHA
            or config.get('seed') != plan['study_seed'] or config.get('input_schema') != INPUT_SCHEMA
            or config.get('training_mode') != 'full-story-v1' or config.get('policy_action_count') != 19
            or config.get('autonomous_bomb') is not True or config.get('headless') is not True
            or config.get('reward_schema') != 'th14-completion-reward-v3'
            or config.get('parameters', {}).get('action_temperature') != .25
            or config.get('full_story_evaluation_backend') != 'full-story-ready-parallel-v1'
            or config.get('full_story_evaluation_workers') != 20
            or config.get('full_story_evaluation_action_selection') != 'greedy'):
        raise ValueError('Reuse the final PPO original environment and evaluation protocol exactly')
    pin(config['game_exe'], plan['game_exe_sha256'])
    pin(config['profile'], plan['profile_sha256'])
    baseline = read(plan['ppo_report'])
    validate_baseline(plan, baseline, read(plan['ppo_episode_plan']))
    if (read(plan['ppo_native_cleanup']) != dict(close_errors=[], workers=20)
            or read(plan['ppo_training_cleanup']).get('close_errors') != []
            or read(plan['ppo_study_result']).get('status') != 'finished'):
        raise ValueError('The actual PPO training/evaluation predecessor must be complete and closed')
    if Path(plan['output']).exists():
        raise FileExistsError('Never reuse a reference output directory')
    install_runtime(runtime)
    import numpy as np
    training_mode = importlib.import_module('fly_touhou.thrl_training_mode')
    require_frozen_module(training_mode, runtime)
    generated, native = [], []
    for game in range(20):
        environment_seed, action_seed = map(int, np.random.SeedSequence([plan['evaluation_seed'], game]).generate_state(2))
        generated.append(dict(game=game, environment_seed=environment_seed, action_seed=action_seed))
        native.append(training_mode.draw_native_seed(np.random.default_rng(environment_seed), 'development'))
    if generated != plan['episode_jobs'] or native != plan['expected_native_seeds']:
        raise ValueError('Evaluation seed does not reproduce the actual matched PPO native cohort')
    return dict(status='CPU-inspect-passed', source_runtime=str(runtime), verified_runtime_files=count,
                checkpoint_sha256=CHECKPOINT_SHA, checkpoint_deserialized=False,
                checkpoint_schema_to_verify_in_explicit_CUDA_mode='B1-risk-calibrated-frozen-candidate-v1',
                episodes_new=20, PPO_episodes_reused=20, PPO_clears_reused=8,
                evaluation_seed=plan['evaluation_seed'], native_seeds=native,
                native_worlds_started=0, optimizer_updates=0,
                Torch_imported=any(name == 'torch' or name.startswith('torch.') for name in sys.modules))


def execute_cuda(plan, qualification, *, qualify_only=False):
    """Control layer only; model loading and complete-game evaluation are frozen imports."""
    runtime = Path(plan['source_runtime']).resolve()
    import psutil
    previous = read(plan['ppo_launch'])
    try:
        process = psutil.Process(previous['pid'])
        if abs(process.create_time() - previous['create_time']) < .01 and process.is_running():
            raise RuntimeError('The actual PPO predecessor has not exited')
    except psutil.NoSuchProcess:
        pass
    if any((p.info['name'] or '').lower() == 'th14.exe' for p in psutil.process_iter(['name'])):
        raise RuntimeError('Another original pool exists; leave it untouched')
    if psutil.virtual_memory().available < 16 * 1024 ** 3:
        raise RuntimeError('Require16GiB available physical RAM before this20-world pool')
    import torch
    source_io = importlib.import_module('fly_touhou.thrl_parameter_transfer')
    factory = importlib.import_module('fly_touhou.thrl_pair_policy')
    evaluator = importlib.import_module('train_thrl_native')
    comparison = importlib.import_module('evaluate_b1_full_story_comparison')
    for module in (source_io, factory, evaluator, comparison):
        require_frozen_module(module, runtime)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA inference required; no CPU inference fallback')
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.3)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if torch.cuda.mem_get_info()[0] < 2 * 1024 ** 3:
        raise RuntimeError('Require2GiB free CUDA memory before native launch')
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    payload = source_io.read_evaluation_source(torch, plan['checkpoint'], CHECKPOINT_SHA)
    model = factory.model_from_config(**payload['model_config']).cuda()
    model.load_state_dict(payload['model'], strict=True)
    model.eval()
    model.requires_grad_(False)
    def exact_weights():
        return all(torch.equal(value.detach().cpu(), payload['model'][key]) for key, value in model.state_dict().items())
    if not exact_weights():
        raise ValueError('Actual CUDA D95 state differs from the qualified source')
    config = dict(read(plan['environment_config']), workers=20, PPO_training_active=False,
                  evaluation_only_policy_role='D95 fly migration teacher reference',
                  evaluation_checkpoint_sha256=CHECKPOINT_SHA)
    if plan['capture_failure_traces']:
        config['full_story_trace'] = dict(checkpoint_sha256=CHECKPOINT_SHA, history=128)
    else:
        config.pop('full_story_trace', None)
    output = Path(plan['output'])
    kwargs = evaluator.make_env_kwargs(config, output / 'games', 20, 'development')
    if any(k.get('policy_action_count') != 19 or k.get('frame_skip') != 4
           or k.get('wall_time_limit_seconds') is not None or 'online_teacher_revision' in k for k in kwargs):
        raise ValueError('Frozen source19 evaluator must have no teacher execution or external horizon')
    qualification = dict(qualification, status='CUDA-source-qualified', checkpoint_deserialized=True,
                         source_checkpoint_schema=payload['schema'], all_state_tensors_exact=True,
                         source_state_tensors=len(payload['model']), optimizer_created=False,
                         CUDA=True, native_worlds_started=0, Torch_imported=True)
    if qualify_only:
        return qualification
    output.mkdir(parents=True, exist_ok=False)
    from fly_touhou.thrl_durable_records import write_json as write
    from fly_touhou.thrl_job_paths import configure_native_temp
    identity = dict(pid=os.getpid(), create_time=psutil.Process().create_time(),
                    utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    entry_sha256=sha(__file__), source_manifest_sha256=RUNTIME_SHA,
                    checkpoint_sha256=CHECKPOINT_SHA, optimizer_updates=0, native_worlds_max=20,
                    desktop_input=False, job_type='D95-fly-migration-reference-evaluation-only-v1')
    write(output / 'launch.json', identity)
    write(output / 'plan.json', plan)
    write(output / 'qualification.json', qualification)
    write(output / 'study_config.json', config)
    os.environ['FLY_TOUHOU_DESKTOP_NAME'] = 'fly-touhou-job-D95-fly-reference-' + str(os.getpid())
    configure_native_temp(output)
    def stop():
        if (output / 'STOP').exists():
            raise RuntimeError('Operator STOP; preserve completed games without automatic restart')
    def publish(row):
        write(output / 'progress.json', dict(status='evaluation', job_type=identity['job_type'],
              optimizer_updates=0, policy_checkpoint_sha256=CHECKPOINT_SHA, evaluation_progress=row))
    try:
        result = evaluator.evaluate(torch, model, config, output / 'games', episodes=20,
                    seed=plan['evaluation_seed'], check_stop=stop, progress=publish, native_split='development')
        actual = [r['native_seed'] for r in result['episode_results']]
        if actual != plan['expected_native_seeds'] or not exact_weights():
            raise ValueError('Executed cohort or frozen D95 tensors changed')
        if read(output / 'games/native_cleanup.json') != dict(close_errors=[], workers=20):
            raise RuntimeError('Owned complete-game pool cleanup was not successful')
        pin(plan['checkpoint'], CHECKPOINT_SHA)
        baseline = read(plan['ppo_report'])
        paired = comparison.paired_summary(dict(PPO_reused=baseline, D95=result), 20, baseline='PPO_reused')
        report = dict(schema=RESULT_SCHEMA, status='completed', job_type=identity['job_type'],
            D95_report=str(output / 'games/report.json'), D95_report_sha256=sha(output / 'games/report.json'),
            D95_checkpoint_sha256=CHECKPOINT_SHA, reused_PPO_report=plan['ppo_report'],
            reused_PPO_report_sha256=plan['ppo_report_sha256'], paired=paired,
            newly_executed_native_games=20, reused_PPO_native_games=20, optimizer_updates=0,
            policy_actions=19, autonomous_Bomb=True, continues=0, native_frame_horizon=None,
            action_selection='greedy', action_temperature=.25, native_seed_split='development',
            formal_test=False, formal_acceptance=False, fly_student_evaluated=False,
            source_manifest_sha256=RUNTIME_SHA, launch=identity)
        write(output / 'report.json', report)
        publish(dict(status='completed', completed=20, target=20, native_worlds=0))
        return report
    except BaseException:
        write(output / 'failure.json', dict(status='failed_or_stopped', traceback=traceback.format_exc(),
              optimizer_updates=0, partial_results_are_not_a_score=True, no_automatic_restart=True))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--mode', choices=('inspect', 'qualify-cuda', 'evaluate-cuda'), default='inspect')
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args(argv)
    if args.plan is None:
        parser.print_help()
        return 0
    if args.receipt and args.receipt.exists():
        raise FileExistsError('Never reuse a preparation receipt')
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[key] = '2'
    if args.mode == 'inspect':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        import psutil
        if sys.platform == 'win32':
            psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    plan = read(args.plan)
    result = inspect(plan)
    if args.mode != 'inspect':
        result = execute_cuda(plan, result, qualify_only=args.mode == 'qualify-cuda')
    if args.receipt:
        with args.receipt.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
    print(json.dumps({key:result.get(key) for key in ('status','native_worlds_started','optimizer_updates','Torch_imported')}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
