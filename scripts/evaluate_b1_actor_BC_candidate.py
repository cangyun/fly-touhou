"""Inspect/export the new BC schema on CPU; native execution is explicit.

No old risk-candidate, cached-warmup or PPO-resume schema is accepted. An
export plan cannot launch a game. Native execution requires a separate plan
pinning an already exported evaluation artifact and an additive frozen runtime.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
import uuid

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from fly_touhou import thrl_actor_bc_evaluation as adapter
from fly_touhou.thrl_study_io import read_shared_json


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('x', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_plan(plan):
    if plan.get('schema') == adapter.EXPORT_PLAN_SCHEMA:
        if set(plan) != {'schema', 'BC_checkpoint', 'selection'}:
            raise ValueError('An export plan contains only the selected BC checkpoint and selection references')
        return plan
    required = {'schema', 'candidate', 'evaluation_runtime', 'evaluation_manifest_sha256',
        'environment_config', 'seed', 'expected_native_seeds', 'workers', 'episodes',
        'max_native_worlds', 'main_gate'}
    if (plan.get('schema') != adapter.NATIVE_PLAN_SCHEMA or not required.issubset(plan)
            or set(plan) - (required | {'max_seconds'})
            or type(plan['workers']) is not int or not 1 <= plan['workers'] <= 20
            or type(plan['episodes']) is not int or plan['episodes'] != 20
            or type(plan['max_native_worlds']) is not int or plan['max_native_worlds'] != 30
            or plan['expected_native_seeds'] != adapter.development_seeds(plan['seed'])):
        raise ValueError('Use an explicit frozen20-game development plan within the30-world ceiling')
    gate = plan['main_gate']
    if (not isinstance(gate, dict) or set(gate) != {'output', 'pid', 'create_time'}
            or not isinstance(gate['output'], str) or type(gate['pid']) is not int or gate['pid'] < 1
            or isinstance(gate['create_time'], bool) or not isinstance(gate['create_time'], (int, float))
            or not math.isfinite(gate['create_time']) or gate['create_time'] <= 0):
        raise ValueError('An explicit predecessor launch output/PID/creation time is required')
    seconds = plan.get('max_seconds', 4 * 3600)
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not 0 < seconds <= 4 * 3600:
        raise ValueError('Finite evaluation wall budget must be at most4hours;partial games are never scored')
    return {**plan, 'max_seconds': float(seconds)}


def verify_evaluation_context(plan, payload, runtime=ROOT):
    runtime = Path(runtime).resolve()
    if Path(plan['evaluation_runtime']).resolve() != runtime:
        raise ValueError('Execute native development only from its explicitly frozen evaluation runtime')
    entries = adapter.verify_manifest(runtime, plan['evaluation_manifest_sha256'])
    required = ('src/fly_touhou/thrl_actor_bc_evaluation.py',
                'scripts/evaluate_b1_actor_BC_candidate.py')
    if any(name not in entries for name in required):
        raise ValueError('Independent evaluation adapter or entry is not frozen')
    qualified = payload['provenance']['source_BC_training_context']['source_runtime_files']
    if any(entries.get(name) != value for name, value in qualified.items()):
        raise ValueError('Evaluation runtime must retain every qualified BC runtime file unchanged')
    if (Path(adapter.__file__).resolve() != runtime / required[0]
            or Path(__file__).resolve() != runtime / required[1]):
        raise ValueError('The imported evaluation adapter or entry belongs to another source tree')
    environment = adapter.check_reference(plan['environment_config'])
    cfg = adapter.development_config(adapter.read_json(environment['path']), plan['seed'])
    adapter.reference(cfg['game_exe'], cfg['game_exe_sha256'])
    adapter.reference(cfg['profile'], cfg['profile_sha256'])
    if 'native/th14_fake_d3d9.dll' not in entries:
        raise ValueError('The qualified headless COM library is missing')
    return cfg


def predecessor_alive(gate, psutil):
    # launch.json is read with Windows share-delete semantics even though it
    # should be immutable;no live writer is denied an atomic replacement.
    launch = read_shared_json(Path(gate['output']) / 'launch.json')
    if launch.get('pid') != gate['pid'] or launch.get('create_time') != gate['create_time']:
        raise RuntimeError('Pinned predecessor launch identity differs')
    try:
        process = psutil.Process(gate['pid'])
        if abs(process.create_time() - gate['create_time']) > .01:
            raise RuntimeError('Predecessor PID was reused;stop this evaluation for an explicit new plan')
        return process.is_running()
    except psutil.NoSuchProcess:
        return False
    except psutil.AccessDenied as error:
        raise RuntimeError('Predecessor identity cannot be verified') from error


def native_world_count(psutil):
    try:
        return sum((p.info['name'] or '').lower() == 'th14.exe'
                   for p in psutil.process_iter(['name']))
    except psutil.AccessDenied as error:
        raise RuntimeError('Native-world capacity cannot be verified') from error


class NativeGuard:
    def __init__(self, plan, output, psutil):
        self.plan, self.output, self.psutil = plan, Path(output), psutil
        self.started = time.monotonic()
        self.last_capacity_check = -float('inf')

    def check(self, *, startup=False):
        if (self.output / 'STOP').exists() or time.monotonic() - self.started > self.plan['max_seconds']:
            raise RuntimeError('Stop this finite BC evaluation;preserve complete rows and mark partial games unscored')
        now = time.monotonic()
        if startup or now - self.last_capacity_check >= 1.:
            adapter.capacity_gate(native_world_count(self.psutil), self.plan['workers'], startup=startup,
                main_identity_alive=predecessor_alive(self.plan['main_gate'], self.psutil))
            self.last_capacity_check = now


def model_matches(torch, model, payload):
    if model.model_config() != payload['model_config'] or model.input_schema != adapter.INPUT_SCHEMA:
        raise ValueError('Constructed evaluation model configuration/input differs')
    current = model.state_dict()
    if (set(current) != set(payload['model'])
            or any(v.dtype != payload['model'][n].dtype or v.shape != payload['model'][n].shape
                or not torch.equal(v.detach().cpu(), payload['model'][n]) for n, v in current.items())):
        raise ValueError('Frozen evaluation NN tensors differ')
    scope = payload['provenance']['scope']
    if set(dict(model.named_parameters())) != set(scope['trainable_parameter_names'] + scope['frozen_parameter_names']):
        raise ValueError('Actual NN parameter/buffer identities differ from the BC scope')


def execute_native(torch, plan, payload, cfg, output):
    import psutil
    from fly_touhou.thrl_pair_policy import model_from_config
    from fly_touhou.thrl_full_story_evaluation import evaluate
    from fly_touhou.thrl_training_mode import episode_protocol
    from fly_touhou.thrl_job_paths import configure_native_temp
    from train_thrl_native import make_env_kwargs, development_summary
    guard = NativeGuard(plan, output, psutil)
    guard.check(startup=True)  # before any model construction/CUDA/native work
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if torch.is_autocast_enabled() or torch.is_autocast_enabled('cpu'):
        raise ValueError('Frozen development requires strict FP32 without autocast')
    if not torch.cuda.is_available():
        raise RuntimeError('Explicit native execution requires CUDA;no CPU inference fallback')
    torch.cuda.set_per_process_memory_fraction(.25, 0)
    configure_native_temp(output)
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    os.environ['FLY_TOUHOU_DESKTOP_NAME'] = 'fly-touhou-job-B1-actor-BC-dev-' + str(os.getpid())
    candidate_sha = plan['candidate']['sha256']
    cfg = {**cfg, 'workers': plan['workers'], 'evaluation_checkpoint_sha256': candidate_sha}
    write_json(output / 'study_config.json', cfg)
    model = model_from_config(**payload['model_config']).cpu()
    model.load_state_dict(payload['model'], strict=True)
    model_matches(torch, model, payload)
    model.requires_grad_(False)
    model.eval()
    guard.check(startup=True)
    model = model.cuda()
    model_matches(torch, model, payload)
    destination = output / 'full_story_development'
    def publish(row):
        progress = dict(status='development', updated_utc=adapter.utc(), native_evaluation_executed=True,
            candidate_sha256=candidate_sha, action_temperature=1., action_selection='greedy',
            optimizer_updates=0, evaluation_progress=row)
        write_json(output / 'progress.json', progress)
        print(json.dumps(progress, ensure_ascii=False), flush=True)
    publish(dict(completed=0, target=20))
    result = adapter.evaluate_development(torch, model, candidate_sha,
        make_env_kwargs(cfg, destination, plan['workers'], 'development'), destination,
        seed=plan['seed'], expected_native_seeds=plan['expected_native_seeds'],
        check_stop=guard.check, summary=development_summary, evaluator=evaluate,
        progress=publish, episode_protocol=episode_protocol(cfg))
    model_matches(torch, model, payload)
    report = dict(result, schema='B1-actor-path-BC-frozen-full-story-development-result-v1',
        status='completed', created_utc=adapter.utc(), evaluation_source=plan['candidate'],
        candidate_provenance=payload['provenance'], evaluation_contract=payload['evaluation_contract'],
        native_games=20, new_optimizer_updates=0, formal_test=False,
        independent_stability_acceptance=False, not_teacher_demonstration=True,
        demonstration_catalogue_eligible=False, numerical_PPO_resume=False)
    write_json(output / 'report.json', report)
    return report


def run(args):
    plan = validate_plan(adapter.read_json(args.plan))
    export_plan = plan['schema'] == adapter.EXPORT_PLAN_SCHEMA
    if (args.execute_native and export_plan) or (args.export_candidate and not export_plan):
        raise ValueError('CPU export and native development require distinct plan schemas')
    if (args.execute_native or args.export_candidate) and args.output is None:
        raise ValueError('An explicit fresh output directory is required')
    # This process-only setting must precede Torch import on the default CPU
    # path. On Windows an empty value removes the variable and exposes CUDA.
    if not args.execute_native:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    import psutil
    import torch
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if export_plan:
        bc, provenance = adapter.load_selected_bc(torch, plan['BC_checkpoint'], plan['selection'])
        result = dict(schema='B1-actor-path-BC-CPU-inspection-v1', native_worlds_started=0,
            model_constructed=False, CUDA_initialized=torch.cuda.is_initialized(),
            evaluation_schema=adapter.SCHEMA, scope=provenance['scope'], cost=provenance['cost'],
            selected=provenance['selection']['selected'], source_D95=provenance['source_D95'])
        del bc
    else:
        payload = adapter.read_candidate(torch, plan['candidate'])
        cfg = verify_evaluation_context(plan, payload)
        result = adapter.candidate_summary(payload)
        result.update(CUDA_initialized=torch.cuda.is_initialized(), model_constructed=False)
    if args.output is None:
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return result
    args.output.mkdir(parents=True, exist_ok=False)
    args.created_output = args.output
    write_json(args.output / 'launch.json', dict(pid=os.getpid(), create_time=psutil.Process().create_time(),
        utc=adapter.utc(), mode='native-development' if args.execute_native else
        'CPU-export' if args.export_candidate else 'CPU-inspect', native_worlds_started=0,
        optimizer_updates=0, private_desktop=bool(args.execute_native), precision=adapter.PRECISION))
    write_json(args.output / 'plan.json', plan)
    if args.export_candidate:
        artifact, summary = adapter.export_candidate(torch, plan['BC_checkpoint'], plan['selection'],
            args.output / 'candidate_for_development.pt')
        result.update(candidate=artifact, candidate_summary=summary)
    elif args.execute_native:
        return execute_native(torch, plan, payload, cfg, args.output)
    write_json(args.output / 'inspection.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--export-candidate', action='store_true', help='CPU export only;never starts native worlds')
    modes.add_argument('--execute-native', action='store_true', help='Explicitly execute the pinned frozen development plan')
    return parser.parse_args(argv)


if __name__ == '__main__':
    arguments = parse_args()
    try:
        run(arguments)
    except BaseException:
        if getattr(arguments, 'created_output', None):
            write_json(arguments.created_output / 'failure.json', dict(created_utc=adapter.utc(),
                traceback=traceback.format_exc(), optimizer_updates=0,
                partial_games_are_not_a_score=True, automatic_restart=False))
        raise
