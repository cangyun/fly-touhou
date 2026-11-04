"""Prepare, then explicitly execute one bounded final-theta DN risk64 calibration.

NoTorch plan by default. No native game, teacher query or PPO step is created.
The eight source-run trajectories are calibration data already seen by PPO.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
SCHEMA = 'continuous-risk64-standardized-calibration-entry-v1'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('training-source', 'connectome', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--training-source-sha256', help='Optional caller-supplied artifact identity; not a source-code gate')
    p.add_argument('--max-seconds', type=int, required=True)
    p.add_argument('--seed', type=int, default=972137)
    p.add_argument('--cost-centering', choices=('min', 'mean', 'target-only'), default='min')
    p.add_argument('--execute-cuda', action='store_true')
    a = p.parse_args(argv)
    if a.max_seconds < 1 or not 0 <= a.seed < 2**31:
        p.error('Positive finite wall budget and explicit private seed required')
    if a.training_source_sha256 is not None and (len(a.training_source_sha256) != 64
            or any(c not in '0123456789abcdef' for c in a.training_source_sha256)):
        p.error('Optional source identity must be lowercase SHA256')
    return a


def plan(a):
    return dict(schema=SCHEMA, status='prepared_not_executed', Torch_imported='torch' in sys.modules,
        source=str(a.training_source), connectome=str(a.connectome), output=str(a.output),
        source_SHA_is_compatibility_gate=False, caller_source_identity=a.training_source_sha256,
        replay=dict(trace_paths=[str(a.training_source.parent/'traces'/f'rollout-{i:03}.npz') for i in range(8)],
            final_frozen_weights=True, starts_at_actual_row0=True, worlds=30, neural_batch=16,
            physical_groups=[16,14], K=8, real_rows=61440, physical_neural_lane_forwards=65536,
            old_DN_or_hidden_used=False, outputs='every following raw DN1299 + original cost18/query/valid/game/row'),
        calibration=dict(architecture='existing raw-DN->64 SiLU->18 softplus; no new runtime layer',
            updates=2000, batch=1024, replacement_sampling=True, Adam_lr=.0003, clip_norm=.5,
            centering=a.cost_centering, scale=8., SmoothL1_beta=.1,
            split='game_id%5==0 calibration-held; these trajectories already seen by PPO',
            statistic_population='train & queried & postfact-valid only',
            selection='fixed final update2000; no held early-stop or model selection'),
        fold=dict(runtime_input='original raw DN', initial='Wstd=Wraw*scale; bstd=braw+Wraw@mean',
            final='Wraw=Wstd/scale; braw=bstd-Wraw@mean', rounding='measured on all cached DN; no fixed 1e-5 rejection'),
        main_Adam=dict(restore='all source groups and states', reset='only risk4 after coordinate folding',
            other_parameters_and_moments='bitwise preserved', brain_remains_trainable=True),
        PPO_updates=0, native_actions=0, teacher_queries=0, max_seconds=a.max_seconds,
        RAM_or_VRAM_caps=None, automatic_retry=False, CUDA_started=False)


def _completed_predecessor(source, psutil):
    folder = source.parent
    result = json.loads((folder/'result.json').read_text('utf-8'))
    cleanup = json.loads((folder/'training-cleanup.json').read_text('utf-8'))
    if result.get('status') != 'completed' or cleanup.get('training_closed') is not True or cleanup.get('close_errors'):
        raise RuntimeError('Completed source run and cleanly closed native worlds required')
    launch = json.loads((folder/'launch.json').read_text('utf-8'))
    if 'PID' in launch and 'create_time' in launch:
        try:
            process = psutil.Process(launch['PID'])
            if abs(process.create_time()-launch['create_time']) < .01:
                raise RuntimeError('Source owner still alive; wait for its normal exit')
        except psutil.NoSuchProcess:
            pass
    return dict(result_sha256=sha(folder/'result.json'), cleanup_sha256=sha(folder/'training-cleanup.json'),
        source_owner=dict(PID=launch.get('PID'), create_time=launch.get('create_time')), exit_checked=True)


def execute(a):
    if 'torch' in sys.modules:
        raise RuntimeError('Fresh executor required; CUDA visibility must precede Torch import')
    if a.output.exists():
        raise FileExistsError('Fresh output required; no automatic retry or overwrite')
    import psutil
    process = psutil.Process()
    predecessor = _completed_predecessor(a.training_source, psutil)
    process.cpu_affinity(process.cpu_affinity()[-2:])
    if os.name == 'nt': process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    os.environ.update(CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    a.output.mkdir(parents=True)
    begun = time.monotonic()
    from fly_touhou.thrl_durable_records import write_json
    costs = dict(replay_real_rows=0, replay_core_lane_forwards=0, replay_padding_lane_forwards=0,
        calibration_Adam=0, calibration_backwards=0, sampled_training_rows=0,
        calibration_head_forward_rows=0, head_inference_rows=0,
        PPO_Adam=0, native_actions=0, teacher_queries=0)
    launch = dict(plan(a), PID=os.getpid(), create_time=process.create_time(),
        started_UTC=datetime.now(timezone.utc).isoformat(), predecessor=predecessor)
    write_json(a.output/'launch.json', launch)
    model = optimizer = work_head = None
    provenance = None
    def guard():
        if (a.output/'STOP').exists(): raise RuntimeError('Operator STOP')
        if time.monotonic()-begun > a.max_seconds: raise RuntimeError('Finite wall budget reached')
    def report(phase, **kwargs):
        write_json(a.output/'progress.json', dict(schema=SCHEMA, phase=phase,
            costs=copy.deepcopy(costs), elapsed_seconds=time.monotonic()-begun, **kwargs))
    def save_pt(path, value):
        temporary = path.with_suffix(path.suffix+'.tmp')
        with temporary.open('wb') as stream:
            torch.save(value, stream); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    try:
        source_sha = sha(a.training_source)
        if a.training_source_sha256 is not None and source_sha != a.training_source_sha256:
            raise ValueError('Caller-supplied source artifact identity differs')
        import numpy as np
        import torch
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(False)
        from fly_touhou.continuous_connectome import ContinuousConnectomeCore
        from fly_touhou.continuous_connectome_model import ContinuousConnectomeActorCritic
        from fly_touhou.continuous_connectome_training import restore_training_checkpoint, training_checkpoint_payload
        from fly_touhou.continuous_risk_calibration import (replay_final_DN, train_statistics,
            standardized_head, folded_head, head_predictions, prediction_error, fit_standardized_head,
            nonrisk_identity, publish_folded_head, calibration_loss)
        from fly_touhou.continuous_connectome_cached_evaluation import _row_cost_metrics, _cost_summary
        guard(); report('restoring_complete_source')
        source = torch.load(a.training_source, map_location='cpu', weights_only=True)
        config = source['model_payload']['configuration']['core']
        if config['K'] != 8 or config['trainable_brain'] is not True:
            raise ValueError('This candidate preserves the trainable continuous K8 family')
        core = ContinuousConnectomeCore.from_full_connectome(a.connectome, K=8, trainable_brain=True,
            device='cpu', seed=config['seed'], epsilon=config['epsilon'],
            edge_chunk_size=config['sparse']['edge_chunk_size'])
        model = ContinuousConnectomeActorCritic.from_parameter_payload(source['model_payload'], core).to('cuda:0')
        optimizer = torch.optim.Adam(model.optimizer_groups())
        inherited = restore_training_checkpoint(model, optimizer, source, restore_rng=True)
        source_algorithm = copy.deepcopy(source['algorithm'])
        provenance = dict(parent=dict(path=str(a.training_source), sha256=source_sha,
                counters=copy.deepcopy(inherited), algorithm=copy.deepcopy(source_algorithm)),
            predecessor=predecessor, source_SHA_is_load_gate=False,
            entry_sha256=sha(__file__), helper_sha256=sha(ROOT/'src/fly_touhou/continuous_risk_calibration.py'),
            process=dict(PID=os.getpid(), create_time=process.create_time()),
            device=dict(name=torch.cuda.get_device_name(0), total_memory=torch.cuda.get_device_properties(0).total_memory),
            FP32=True, TF32=False, artificial_memory_caps=None,
            native_state_or_trajectory_resumed=False, main_Adam_fully_restored_before_calibration=True)
        del source
        identity_before = nonrisk_identity(model, optimizer)
        report('replaying_final_theta_from_row0')
        def replay_progress(row):
            costs['replay_real_rows'] = row['costs']['real_rows']
            costs['replay_core_lane_forwards'] = row['costs']['core_lane_forwards']
            costs['replay_padding_lane_forwards'] = row['costs']['padding_lane_forwards']
            report('replaying_final_theta_from_row0', replay=row)
        replay = replay_final_DN(model,
            [a.training_source.parent/'traces'/f'rollout-{i:03}.npz' for i in range(8)],
            a.output/'final-theta-DN', check_stop=guard, progress=replay_progress)
        write_json(a.output/'final-theta-DN'/'report.json', replay)
        raw_DN = torch.from_numpy(np.load(replay['all_DN_path'], mmap_mode='c'))
        with np.load(replay['label_path'], allow_pickle=False) as archive:
            target = torch.from_numpy(archive['costs'].copy())
            valid = torch.from_numpy((archive['queried'] & archive['valid']).copy())
            held = torch.from_numpy((archive['game_id'] % 5 == 0).copy())
            games = archive['game_id'].copy()
        train_valid = ~held & valid
        if not bool((held & valid).any()):
            raise ValueError('No calibration-held valid cost scenes; do not invent a new split')
        mean, scale, statistics = train_statistics(raw_DN, train_valid)
        np.savez(a.output/'calibration-statistics.npz', mean=mean.numpy(), scale=scale.numpy(),
            train_valid_mask=train_valid.numpy(), calibration_held_mask=held.numpy())
        with torch.no_grad():
            initial_prediction = head_predictions(model.risk_head, raw_DN, check_stop=guard, paid=costs)
            work_head = standardized_head(model.risk_head, mean, scale)
            transformed = head_predictions(work_head, raw_DN, mean=mean, scale=scale, check_stop=guard, paid=costs)
        initial_error = prediction_error(transformed, initial_prediction)
        constant = (target[train_valid]-target[train_valid].amin(-1, keepdim=True)).mean(0)
        constant_rows = constant.expand_as(target)
        def metrics(prediction):
            rows = _row_cost_metrics(prediction.numpy(), target.numpy())
            constant_metrics = _row_cost_metrics(constant_rows.numpy(), target.numpy())
            result = {}
            for role, mask in (('calibration_train', train_valid), ('calibration_held', held & valid)):
                result[role] = dict(valid_rows=int(mask.sum()), games=sorted(set(games[mask.numpy()].tolist())),
                    loss=float(calibration_loss(prediction[mask], target[mask], centering=a.cost_centering)),
                    learned=_cost_summary(rows, mask.numpy()),
                    training_direction_constant=_cost_summary(constant_metrics, mask.numpy()))
            return result
        before = metrics(initial_prediction)
        write_json(a.output/'before.json', dict(statistics=statistics, metrics=before,
            initial_function_rounding=initial_error, not_native_or_end_to_end_heldout=True))
        report('risk64_standardized_calibration', statistics=statistics, before=before)
        def fit_progress(row):
            costs['calibration_Adam'] = row['optimizer_updates']
            costs['calibration_backwards'] = row['backward_passes']
            costs['sampled_training_rows'] = row['sampled_training_rows']
            costs['calibration_head_forward_rows'] = row['head_forward_rows']
            report('risk64_standardized_calibration', learning=row)
        learning = fit_standardized_head(work_head, raw_DN, target, train_valid, mean, scale,
            seed=a.seed, centering=a.cost_centering, check_stop=guard, progress=fit_progress)
        fit_progress(learning)
        standardized_prediction = head_predictions(work_head, raw_DN, mean=mean, scale=scale, check_stop=guard, paid=costs)
        folded = folded_head(work_head, mean, scale)
        folded_prediction = head_predictions(folded, raw_DN, check_stop=guard, paid=costs)
        fold_error = prediction_error(folded_prediction, standardized_prediction)
        report('publishing_folded_raw_DN_head', fold_rounding=fold_error)
        preservation = publish_folded_head(model, optimizer, folded, before_identity=identity_before)
        if not preservation['remaining_brain_trainable']:
            raise RuntimeError('The following joint stage must retain trainable core parameters')
        provenance['risk_calibration_phase'] = dict(protocol=plan(a), statistics=statistics,
            replay=replay, learning=learning, initial_function_rounding=initial_error,
            fold_rounding=fold_error, preservation=preservation,
            coordinate_Adam_transfer='none; only main risk4 invalid old moments removed',
            split_warning='PPO already saw these trajectories; only risk calibration uses the game split')
        counters = copy.deepcopy(inherited)
        counters['risk_calibration_Adam'] = counters.get('risk_calibration_Adam', 0)+learning['optimizer_updates']
        counters['risk_calibration_replay_rows'] = counters.get('risk_calibration_replay_rows', 0)+replay['costs']['real_rows']
        # PPO algorithm, all group LRs and every non-risk moment remain source-identical.
        save_pt(a.output/'final-training.pt', training_checkpoint_payload(model, optimizer,
            algorithm=source_algorithm, counters=counters, provenance=provenance))
        save_pt(a.output/'final-parameters.pt', model.parameter_checkpoint_payload())
        report('completed', learning=learning)
        write_json(a.output/'result.json', dict(schema=SCHEMA, status='completed',
            phase='risk64_calibration_folded_to_original_DN', costs=costs, inherited_counters=inherited,
            source_algorithm_preserved=True, main_Adam_preservation=preservation,
            statistics=statistics, initial_function_rounding=initial_error, fold_rounding=fold_error,
            before=before, after=metrics(folded_prediction), learning=learning,
            final_training=dict(path=str(a.output/'final-training.pt'), sha256=sha(a.output/'final-training.pt')),
            final_parameters=dict(path=str(a.output/'final-parameters.pt'), sha256=sha(a.output/'final-parameters.pt')),
            provenance=provenance, elapsed_seconds=time.monotonic()-begun,
            development=None, native_or_end_to_end_heldout_claim=False))
    except BaseException as error:
        learning = getattr(error, 'risk_calibration_diagnostics', None)
        if learning:
            costs.update(calibration_Adam=learning['optimizer_updates'],
                calibration_backwards=learning['backward_passes'], sampled_training_rows=learning['sampled_training_rows'],
                calibration_head_forward_rows=learning['head_forward_rows'])
        replay = getattr(error, 'continuous_risk_replay_diagnostics', None)
        if replay:
            costs.update(replay_real_rows=replay['real_rows'], replay_core_lane_forwards=replay['core_lane_forwards'],
                replay_padding_lane_forwards=replay['padding_lane_forwards'])
        recovery = None
        work = getattr(error, 'risk_calibration_work', None)
        if work is None and work_head is not None and costs['calibration_Adam']:
            work = dict(head={k:v.detach().cpu().clone() for k,v in work_head.state_dict().items()},
                optimizer=None, record=copy.deepcopy(costs),
                independent_optimizer_state_unavailable_after_completed_fit=True)
        if work is not None:
            try:
                work.update(schema='standardized-coordinate-risk64-work-only-v1',
                    mean=mean, scale=scale, source=str(a.training_source), centering=a.cost_centering,
                    raw_DN_runtime_checkpoint=False, automatic_resume=False)
                save_pt(a.output/'stopped-standardized-work.pt', work)
                recovery = str(a.output/'stopped-standardized-work.pt')
            except BaseException as saving_error: recovery = repr(saving_error)
        write_json(a.output/'failure.json', dict(schema=SCHEMA, error=repr(error), costs=costs,
            learning=learning, replay=replay, standardized_work_recovery=recovery,
            source_checkpoint_unchanged=True, native_actions=0, automatic_retry=False))
        raise
    finally:
        write_json(a.output/'training-cleanup.json', dict(training_closed=True, close_errors=[],
            native_worlds_created=0, cleanup='process-local model released on executor exit'))
        model = optimizer = work_head = None


def main(argv=None):
    a = parse_args(argv)
    if not a.execute_cuda:
        print(json.dumps(plan(a), ensure_ascii=False, indent=2)); return 0
    execute(a); return 0


if __name__ == '__main__':
    raise SystemExit(main())
