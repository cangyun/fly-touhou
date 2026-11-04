"""Finite neural-component timing on 128 verified source19 rows; zero training."""
import argparse
from contextlib import closing
import importlib.util
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '2'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
from fly_touhou.fly_cached_distillation import (
    CachedFitConfig, FitGuard, atomic_json, file_sha, import_reader_preserving_CVD, process_identity,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-cuda', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a new output, preserving prior measurements')
    args.output.mkdir(parents=True)
    report = dict(schema='fly-source19-cached-capture128-component-benchmark-v1', status='inspect_only',
                  script_sha256=file_sha(__file__), data_rows=128, total_forward_budget=264,
                  eager_setup_forwards=4, capture_setup_forwards=4, optimizer_steps=0, backward_calls=0,
                  native_worlds=0, data_forwards_completed=0, setup_forwards_completed=0,
                  comparison_scope='all128logits+last_complete_state;not_all_intermediate_states',
                  timing_scope='resident GPU Phi, full brain calls and per-row guard; excludes reader/metrics/checkpoint/loss/backward')
    if not args.execute_cuda:
        atomic_json(args.output / 'inspect.json', report)
        return 0
    guard = FitGuard(args.output, CachedFitConfig(max_seconds=180., max_updates=0, max_model_decisions=264))
    try:
        guard.check('before_reader')
        module, sequence = import_reader_preserving_CVD()
        if os.environ.get('CUDA_VISIBLE_DEVICES') in (None, '', '-1'):
            raise ValueError('Explicit caller CUDA required')
        reader = module.FullEpisodeSensoryReader(
            './outputs/distillation/B1-d95-v5-sensory9721-resumable-cache-v1',
            './outputs/distillation/B1-d95-final19-labels-v1', guard)
        with closing(reader.iter_episode('world000', max_rows=128, audit_dir=args.output/'reader-audit')) as stream:
            rows = list(stream)
        if len(rows) != 128 or [r.metadata.source_row for r in rows] != list(range(128)):
            raise ValueError('Expected exact contiguous128 row0 prefix')
        import numpy as np
        import torch
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.set_per_process_memory_fraction(.2, 0)
        if torch.cuda.mem_get_info(0)[0] < 6*1024**3:
            raise ValueError('Require6GiB free VRAM')
        torch.cuda.reset_peak_memory_stats(0)
        spec = importlib.util.spec_from_file_location('qualified_source19_transfer', ROOT/'scripts/qualify_fly_cached_backend_and_capture.py')
        qualified = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(qualified)
        qualified.verify_sources()
        from fly_touhou.fly_distillation_student import FlyDistillationStudent
        from fly_touhou.whole_brain import BrainInterfaces
        checkpoint = Path('./outputs/distillation/D95-fly-Phi9721-continuous2048-v1/last-completed-parameters.pt')
        if file_sha(checkpoint) != 'd3045ef45b082eac006a84ceed644f490299c9bb9ad7029f808b1749bb7e79d2':
            raise ValueError('Immutable512 source differs')
        source = FlyDistillationStudent.from_parameter_checkpoint(checkpoint, ROOT/'data/connectome', device='cpu')
        config = source.brain.config()
        interface = BrainInterfaces(tuple(config['input_indices']), tuple(config['output_indices']), dict(source.interface_provenance))
        options = {key:config[key] for key in ('substeps','dt_ms','readout_tau_ms','max_input_rate_hz','rng_seed',
            'encoder_init_scale','surrogate_scale','surrogate_slope_per_mv','parameters')}
        cached = FlyDistillationStudent(source.brain.graph, interface, device='cpu',
            propagation_backend='coo-cusparse-cached', recurrent_gain_limit=0., require_full_graph=True, **options)
        report['transfer'] = qualified.restricted_IO_transfer(source, cached)
        cached.to('cuda:0')
        report.update(process=process_identity(), CVD_import_sequence=sequence,
                      source_checkpoint_sha256=file_sha(checkpoint), status='executing')
        phi = torch.tensor(np.stack([r.phi for r in rows]), device='cuda:0')
        with torch.no_grad():
            for _ in range(4):
                guard.check('eager_setup')
                cached.brain(phi[:1], cached.initial_state(1, rng_seed=613711))
                report['setup_forwards_completed'] += 1

            def phase(name):
                state = cached.initial_state(1, rng_seed=613711)
                outputs = []
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                wall = time.perf_counter()
                begin.record()
                for i in range(128):
                    guard.check(name)
                    logits, state = cached.brain(phi[i:i+1], state)
                    report['data_forwards_completed'] += 1
                    outputs.append(logits)
                end.record()
                torch.cuda.synchronize()
                seconds = time.perf_counter()-wall
                result = dict(wall_seconds=seconds, GPU_event_milliseconds=begin.elapsed_time(end),
                              decisions=128, decisions_per_second=128/seconds)
                return result, torch.cat(outputs).cpu(), qualified.snapshot_state(state)

            eager, eager_logits, eager_state = phase('cached_eager')
            setup_begin = time.perf_counter()
            cached.brain.enable_cuda_capture(1)
            torch.cuda.synchronize()
            report['capture_setup_seconds'] = time.perf_counter()-setup_begin
            report['setup_forwards_completed'] += 4
            captured, capture_logits, capture_state = phase('cached_capture')
        report.update(eager=eager, capture=captured,
            component_wall_speed_ratio=eager['wall_seconds']/captured['wall_seconds'],
            logits_comparison=qualified.compare_tensor(capture_logits,eager_logits,hard=False,rtol=1e-5,atol=1e-5),
            last_state_comparison=qualified.compare_state(capture_state,eager_state,1e-5,1e-5))
        report['status'] = ('passed_component_measurement' if report['logits_comparison']['passed']
                            and report['last_state_comparison']['passed'] else 'numerical_difference_do_not_enable')
        report['peak_GPU_allocated_bytes'] = torch.cuda.max_memory_allocated()
        report['peak_GPU_reserved_bytes'] = torch.cuda.max_memory_reserved()
        cached.brain.clear_cuda_capture()
        cached.brain.clear_cached_propagation()
        report['resources'] = guard.report()
        atomic_json(args.output/'result.json', report)
        return 0 if report['status']=='passed_component_measurement' else 2
    except BaseException as error:
        report.update(status='failed', reason=str(error), resources=guard.report())
        atomic_json(args.output/'failure.json', report)
        raise


if __name__=='__main__':
    raise SystemExit(main())
