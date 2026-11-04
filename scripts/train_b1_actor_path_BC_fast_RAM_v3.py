"""Additive fast physical-RAM entry; unchanged lifecycle-v2 numerical recipe.

Only this owning run layer is copied. Configuration, CLI, save/fit/heldout and
the complete lifecycle/GPU ordering remain the original v2 implementation.
"""
from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import sys
import traceback

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'))
import train_b1_actor_path_BC_lifecycle_v2 as v2

base = v2.base
lifecycle = v2.lifecycle
validate_config = v2.validate_config
execute = v2.execute
parse_args = v2.parse_args
BASE_RUNNER_SHA = v2.BASE_RUNNER_SHA
RAM_HELPER_PATH = 'src/fly_touhou/windows_physical_memory.py'
RAM_BACKEND = ('kernel32.GlobalMemoryStatusEx.ullAvailPhys' if os.name == 'nt'
               else 'psutil.virtual_memory.available-non-Windows-fallback')


def run(args):
    import psutil
    import shutil
    from fly_touhou.windows_physical_memory import available_physical_memory_bytes
    outer = validate_config(base.read_json(args.config))
    config = outer['BC']
    if args.output.exists():
        raise ValueError('Use a fresh output;no numerical or partial resume exists')
    if base.sha(ROOT/'scripts/train_b1_actor_path_BC_contrast.py') != BASE_RUNNER_SHA:
        raise ValueError('The numerical v1 runner bytes differ')
    args.output.mkdir(parents=True, exist_ok=False)
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    dataset = None
    ram_monitor = dict(backend=RAM_BACKEND, helper_path=RAM_HELPER_PATH,
        helper_source_sha256=base.sha(ROOT/RAM_HELPER_PATH), fresh_measurement_per_check=True)
    state = dict(schema='B1-raw-actor-path-BC-lifecycle-result-v2', status='inspecting', arms={},
        requested_arms=outer['arms'],
        actual_original_worlds_started=0, production_training_requested=bool(args.execute_cuda),
        baseline_reused_from_prior_job=False, prior_optimizer_or_hidden_restored=False,
        RAM_monitor=deepcopy(ram_monitor))
    provenance = dict(schema=lifecycle.SCHEMA, entry=str(Path(__file__).resolve()),
        entry_sha256=base.sha(__file__), lifecycle_module_sha256=base.sha(lifecycle.__file__),
        v1_numerical_runner_sha256=BASE_RUNNER_SHA, outer_config=deepcopy(outer),
        new_job_paid_counters_start_at_zero=True, partials_are_not_resume_or_deployment=True,
        RAM_monitor=deepcopy(ram_monitor), reused_v2_entry_sha256=base.sha(v2.__file__))
    try:
        def resource_check(*, startup, elapsed, stop_exists):
            return base.resource_decision(available_physical_memory_bytes(), shutil.disk_usage(args.output.parent).free,
                elapsed, config['max_seconds'], stop_exists, startup=startup)
        guard = lifecycle.LifecycleGuard(config, outer['main_contract'], args.output, resource_check)
        guard.check(startup=True)
        base.write_json(args.output/'launch.json', dict(pid=os.getpid(), create_time=psutil.Process().create_time(),
            utc=base.utc(), mode='lifecycle-v3-fastRAM-execute-CUDA' if args.execute_cuda else 'lifecycle-v3-fastRAM-CPU-inspect',
            execute_cuda=bool(args.execute_cuda), automatic_restart=False, native_worlds_started=0,
            main_process_control_modified=False, numerical_resume=False, RAM_monitor=deepcopy(ram_monitor)))
        dataset, split, pins = base.inspect_sources(config, args.output)
        manifest = base.read_json(ROOT/'source_manifest.json')
        entries = {item['path']:item['sha256'] for item in manifest['files']}
        if ('src/fly_touhou/thrl_actor_bc_lifecycle.py' not in pins
                or entries.get('scripts/train_b1_actor_path_BC_fast_RAM_v3.py') != base.sha(__file__)
                or pins.get(RAM_HELPER_PATH) != ram_monitor['helper_source_sha256']):
            raise ValueError('The lifecycle/fastRAM entry and RAM helper must be frozen and qualification source-pinned')
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


if __name__ == '__main__':
    raise SystemExit(run(parse_args()))
