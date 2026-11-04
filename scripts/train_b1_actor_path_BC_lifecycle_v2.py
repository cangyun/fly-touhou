"""Independent finite BC lifecycle v2;default metadata-only CPU inspect.

Unchanged raw stream/helper/CE/accumulation/held evaluation from v1. Only the
main lifecycle gate and safe partial preservation differ. No resume path.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import os
from pathlib import Path
import sys
import traceback

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'))
import train_b1_actor_path_BC_contrast as base
from fly_touhou import thrl_actor_bc_lifecycle as lifecycle

CONFIG_SCHEMA = 'B1-raw-actor-path-BC-offline-lifecycle-config-v2'
BASE_RUNNER_SHA = '6dff7fb599d7e864df18a99fd99b40bb63b2422ff00c52f250baddcefa16e999'


def validate_config(config):
    required = {'schema', 'BC', 'lifecycle', 'main_contract'}
    if (not required.issubset(config) or set(config)-required-{'arms'} or config.get('schema') != CONFIG_SCHEMA
            or config.get('lifecycle') != lifecycle.LIFECYCLE):
        raise ValueError('Explicit lifecycle-v2 wrapper config and unchanged finite recipe required')
    bc = base.validate_config(config['BC'])
    contract = config['main_contract']
    required = {'output', 'pid', 'create_time', 'launch_sha256', 'study_config_sha256',
                'control_directory', 'background_spec_sha256'}
    if (not isinstance(contract, dict) or set(contract) != required
            or bc.get('main_gate') != {k: contract[k] for k in ('output', 'pid', 'create_time')}
            or any(not isinstance(contract[k], str) or len(contract[k]) != 64
                   for k in ('launch_sha256', 'study_config_sha256', 'background_spec_sha256'))):
        raise ValueError('The BC gate must bind the actual main launch/config/background specification')
    arms = config.get('arms', list(base.ARMS))
    if (type(arms) is not list or not arms or any(type(arm) is not str or arm not in base.ARMS for arm in arms)
            or len(set(arms)) != len(arms)
            or arms != [arm for arm in base.ARMS if arm in arms]):
        raise ValueError('Explicit scopes must be a nonempty ordered subset of the original two arms')
    return {**deepcopy(config), 'BC': bc, 'arms': list(arms)}


def save_whole(torch, path, model, optimizer, *, outer, provenance, **kwargs):
    base.save_checkpoint(torch, path, model, optimizer, config=outer['BC'], **kwargs)
    base.write_json(path.with_name(path.name+'.lifecycle-v2.json'), dict(
        schema='B1-complete-BC-checkpoint-lifecycle-provenance-v2', checkpoint=str(path),
        checkpoint_sha256=base.sha(path), producer=provenance,
        BC_checkpoint_schema=base.CHECKPOINT_SCHEMA, lifecycle_change_only=True,
        recurrent_numerical_recipe_changed=False, optimizer_inherited=False))


def execute(torch, outer, output, dataset, split, guard, state, report, provenance):
    """Small owning control layer;the numerical v1 functions are called directly."""
    from fly_touhou.thrl_pair_policy import model_from_config
    from fly_touhou.thrl_parameter_transfer import read_evaluation_source
    from fly_touhou.thrl_full_actor_path_bc import begin_actor_path_bc
    config = outer['BC']
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if torch.is_autocast_enabled() or torch.is_autocast_enabled('cpu'):
        raise ValueError('Strict FP32 without enclosing autocast required')
    guard.before_gpu()
    if not torch.cuda.is_available():
        raise lifecycle.LifecycleStop('CUDA required;no CPU training fallback')
    torch.cuda.set_per_process_memory_fraction(.25, 0)
    state['actual_GPU_training_started'] = True
    source = read_evaluation_source(torch, config['source_checkpoint'], base.D95_SHA)
    source_digest = base.state_digest(source['model'])
    baseline = None
    for arm in outer['arms']:
        state.update(current_arm=arm, stage='initializing_arm')
        guard.check()
        model = model_from_config(**source['model_config']).cpu()
        model.load_state_dict(source['model'], strict=True)
        if base.state_digest(model.state_dict()) != source_digest:
            raise ValueError('Both fresh arms must inherit all same-source D95 tensors exactly')
        scope = begin_actor_path_bc(model, arm)
        if scope['trainable_parameters'] != (3804212 if arm == 'full_actor_path' else 897043):
            raise ValueError('Qualified production scope count differs')
        guard.before_gpu()
        model = model.cuda()
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=3e-5)
        folder = output/arm
        folder.mkdir(exist_ok=False)
        cost = dict(paid_BC_optimizer_steps=0, selected_BC_optimizer_steps=None, paid_BC_training_labels=0,
            pending_uncommitted_legal_labels=0, BC_optimizer_step_attempts=0, optimizer_transaction_complete=True,
            complete_train_epoch=False, native_worlds_started=0, PPO_optimizer_steps=0)
        partial = lifecycle.PartialWriter(torch, model, optimizer, folder, config=config, arm=arm,
            scope=scope, cost=cost, provenance=provenance, every=100)
        state['arms'][arm] = dict(status='initializing', scope=scope, cost=cost, partial_checkpoints=partial.committed)
        report(True)
        guard.before_gpu()
        save_whole(torch, folder/'initial.pt', model, optimizer, outer=outer, provenance=provenance,
            arm=arm, scope=scope, cost=cost, phase='initial-untrained')
        baseline_reused = baseline is not None
        if not baseline_reused:
            state['arms'][arm]['status'] = state['stage'] = 'baseline_heldout'
            baseline = base.evaluate(torch, model, dataset, split['heldout'], guard, cost, 'baseline_heldout', report)
            baseline['initial_model_state_digest'] = source_digest
            base.write_json(output/'shared_initial_heldout.json', baseline)
        elif source_digest != baseline['initial_model_state_digest']:
            raise ValueError('Initial heldout baseline cannot be reused for different tensors')
        base.write_json(folder/'initial_heldout.json', {**baseline,
            'reused_identical_initial_baseline': baseline_reused, 'actual_initial_state_digest': source_digest})
        state['arms'][arm]['status'] = state['stage'] = 'training_one_pass'
        state.pop('evaluation_progress', None)
        report(True)
        # v1 calls guard.before_gpu at the next raw block after its completed
        # Adam+sync+zero_grad+cost update. Pending-gradient boundaries skip.
        guard.on_safe_boundary = partial.maybe_save
        try:
            base.train_one_pass(torch, model, optimizer, dataset, split['train'], guard, cost, report)
        finally:
            guard.on_safe_boundary = None
        guard.before_gpu()
        current = model.state_dict()
        trainable = set(scope['trainable_parameter_names'])
        if any(not torch.equal(value.detach().cpu(), source['model'][name])
               for name, value in current.items() if name not in trainable):
            raise ValueError('A frozen encoder,critic,risk or persistent buffer changed')
        save_whole(torch, folder/'final.pt', model, optimizer, outer=outer, provenance=provenance,
            arm=arm, scope=scope, cost=cost, phase='final-one-raw-epoch')
        state['arms'][arm]['status'] = state['stage'] = 'final_heldout'
        final = base.evaluate(torch, model, dataset, split['heldout'], guard, cost, 'final_heldout', report)
        base.write_json(folder/'final_heldout.json', final)
        selected = base.choose_initial_or_final(baseline, final)
        cost['selected_BC_optimizer_steps'] = cost['paid_BC_optimizer_steps'] if selected == 'final' else 0
        guard.before_gpu()
        if selected == 'initial':
            model.load_state_dict(source['model'], strict=True)
            optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=3e-5)
        save_whole(torch, folder/'selected.pt', model, optimizer, outer=outer, provenance=provenance,
            arm=arm, scope=scope, cost=cost, phase='selected-'+selected,
            metrics=final if selected == 'final' else baseline, model_steps=cost['selected_BC_optimizer_steps'])
        selection = dict(selected=selected, selected_checkpoint=str(folder/'selected.pt'),
            selected_checkpoint_sha256=base.sha(folder/'selected.pt'), selected_from=str(folder/(selected+'.pt')),
            selected_from_sha256=base.sha(folder/(selected+'.pt')), cost=deepcopy(cost),
            initial_heldout=baseline, final_heldout=final, selected_by='lower complete heldout hard CE;initial wins ties',
            native_performance=False, deployment_forbidden=True,
            new_PPO_requires_fresh_Adam=True, critic_uncalibrated_for_new_policy=True)
        base.write_json(folder/'selection.json', selection)
        state['arms'][arm].update(status='completed', selection=selection)
        report(True)
        del current, optimizer, model, partial
        dataset.close_caches()
    state.update(status='completed_finite_BC_contrast_lifecycle_v2', stage='finished', native_performance=False,
        completed_arms=[arm for arm, row in state['arms'].items() if row['status']=='completed'],
        both_scopes_executed=all(arm in state['arms'] for arm in base.ARMS),
        deployment_forbidden=True, automatic_restart=False, mid_epoch_resume_supported=False)
    report(True)


def run(args):
    import psutil
    import shutil
    outer = validate_config(base.read_json(args.config))
    config = outer['BC']
    if args.output.exists():
        raise ValueError('Use a fresh output;no numerical or partial resume exists')
    if base.sha(ROOT/'scripts/train_b1_actor_path_BC_contrast.py') != BASE_RUNNER_SHA:
        raise ValueError('The numerical v1 runner bytes differ')
    args.output.mkdir(parents=True, exist_ok=False)
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    dataset = None
    state = dict(schema='B1-raw-actor-path-BC-lifecycle-result-v2', status='inspecting', arms={},
        requested_arms=outer['arms'],
        actual_original_worlds_started=0, production_training_requested=bool(args.execute_cuda),
        baseline_reused_from_prior_job=False, prior_optimizer_or_hidden_restored=False)
    provenance = dict(schema=lifecycle.SCHEMA, entry=str(Path(__file__).resolve()),
        entry_sha256=base.sha(__file__), lifecycle_module_sha256=base.sha(lifecycle.__file__),
        v1_numerical_runner_sha256=BASE_RUNNER_SHA, outer_config=deepcopy(outer),
        new_job_paid_counters_start_at_zero=True, partials_are_not_resume_or_deployment=True)
    try:
        def resource_check(*, startup, elapsed, stop_exists):
            return base.resource_decision(psutil.virtual_memory().available, shutil.disk_usage(args.output.parent).free,
                elapsed, config['max_seconds'], stop_exists, startup=startup)
        guard = lifecycle.LifecycleGuard(config, outer['main_contract'], args.output, resource_check)
        guard.check(startup=True)
        base.write_json(args.output/'launch.json', dict(pid=os.getpid(), create_time=psutil.Process().create_time(),
            utc=base.utc(), mode='lifecycle-v2-execute-CUDA' if args.execute_cuda else 'lifecycle-v2-CPU-inspect',
            execute_cuda=bool(args.execute_cuda), automatic_restart=False, native_worlds_started=0,
            main_process_control_modified=False, numerical_resume=False))
        dataset, split, pins = base.inspect_sources(config, args.output)
        manifest = base.read_json(ROOT/'source_manifest.json')
        entries = {item['path']:item['sha256'] for item in manifest['files']}
        if ('src/fly_touhou/thrl_actor_bc_lifecycle.py' not in pins
                or entries.get('scripts/train_b1_actor_path_BC_lifecycle_v2.py') != base.sha(__file__)):
            raise ValueError('The new lifecycle module must be frozen and qualification source-pinned')
        base.write_json(args.output/'config.json', outer)
        base.write_json(args.output/'lifecycle_provenance.json', provenance)
        if args.execute_cuda:
            base.load_qualification(config, pins, required=True)
            guard.check(startup=True)
            publisher = base.ProgressPublisher(state, guard, args.output)
            def report(force=False, *, evaluation=None):
                state['verified_main_normal_completion'] = guard.last_completion
                publisher(force, evaluation=evaluation)
            guard.on_wait = publisher.waiting
            state.update(status='qualified_cuda_execution_lifecycle_v2', stage='waiting_for_first_GPU_block')
            report(True)
            import torch
            execute(torch, outer, args.output, dataset, split, guard, state, report, provenance)
        else:
            state.update(status='CPU_inspect_only', actual_GPU_training_started=False)
        base.write_json(args.output/'result.json', state)
        return 0
    except BaseException as error:
        stopped = isinstance(error, (base.BCStop, lifecycle.LifecycleStop))
        state.update(status='stopped' if stopped else 'failed', reason=str(error), traceback=traceback.format_exc(),
            automatic_restart=False, numerical_resume=False, mid_epoch_resume_supported=False,
            no_exception_path_GPU_checkpoint_attempt=True, committed_checkpoints_preserved=True,
            actual_BC_optimizer_steps=sum(row['cost']['paid_BC_optimizer_steps'] for row in state['arms'].values()),
            cost_is_lower_bound_if_an_inflight_GPU_operation_failed=True)
        base.write_json(args.output/'failure.json', state)
        return 2 if stopped else 1
    finally:
        if dataset is not None:
            dataset.close_caches()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute-cuda', action='store_true')
    return parser.parse_args(argv)


if __name__ == '__main__':
    raise SystemExit(run(parse_args()))
