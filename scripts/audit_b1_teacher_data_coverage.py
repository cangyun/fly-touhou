"""Review sealed native evidence and lossless storage without running a policy.

Teacher test cases remain diagnostic evidence. No observations are exported
for fitting, no native worlds are opened, and no CUDA context is created.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import argparse
import hashlib
import io
import json
import sys
import time
import zlib

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fly_touhou.atomic_json_io import atomic_write_json as write
from fly_touhou.thrl_study_io import read_shared_json as read, utc_now


def source(path):
    return dict(path=str(path.resolve()),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def storage_check(trace_paths):
    observations = []
    trace_sources = []
    for path in trace_paths:
        receipt = read(path)
        arrays = Path(receipt.get('arrays_file', receipt.get('arrays', '')))
        if not arrays.is_file() or source(arrays)['sha256'] != receipt['sha256']:
            raise ValueError('Sealed trace identity differs: ' + str(path))
        with np.load(arrays, allow_pickle=False) as bundle:
            for index in (0, len(bundle['features']) - 1):
                row = {key: bundle[key][index].copy()
                       for key in ('features', 'objects', 'maps')}
                if any(value.dtype != np.float32 for value in row.values()):
                    raise ValueError('Only exact original FP32 observations qualify')
                observations.append(row)
        trace_sources.append(source(path))
    stacked = {key: np.stack([row[key] for row in observations])
               for key in observations[0]}
    stream = io.BytesIO()
    np.savez(stream, **stacked)
    raw = stream.getvalue()
    raw_bytes = sum(value.nbytes for value in stacked.values())
    times, encoded = [], None
    for _ in range(3):
        started = time.perf_counter()
        encoded = zlib.compress(raw, level=1)
        times.append(time.perf_counter() - started)
    started = time.perf_counter()
    with np.load(io.BytesIO(zlib.decompress(encoded)), allow_pickle=False) as bundle:
        if set(bundle.files) != set(stacked):
            raise ValueError('Lossless field names differ')
        for key, value in stacked.items():
            if (bundle[key].dtype != value.dtype or bundle[key].shape != value.shape
                    or bundle[key].tobytes() != value.tobytes()):
                raise ValueError('Compression changed original FP32 bits: ' + key)
    decode_seconds = time.perf_counter() - started
    return dict(observations=len(observations), raw_array_bytes=raw_bytes,
                compressed_bytes=len(encoded), compression='zlib-level1 over uncompressed NPZ',
                compression_fraction=len(encoded) / raw_bytes,
                encode_seconds_three_repeats=times, decode_verify_seconds=decode_seconds,
                all_original_FP32_bits_preserved=True, traces=trace_sources,
                sample_is_not_a_whole_story_storage_bound=True,
                observations_exported_for_training=0)


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    movement = read(args.movement_report)
    if (movement['status'] != 'completed'
            or read(args.movement_report.parent / 'cleanup.json') != dict(closed=True, close_errors=[])):
        raise ValueError('Completed closed movement comparison required')
    policies = ('trained_sampled', 'initial_sampled', 'uniform18',
                'trained_greedy', 'initial_greedy')
    groups = {name: [row for row in movement['episode_results'] if row['policy'] == name]
              for name in policies}
    if any(len(rows) != 20 or len({r['game'] for r in rows}) != 20 for rows in groups.values()):
        raise ValueError('Five complete 20-game conditions required')
    matched_seeds = {}
    for game in range(20):
        seeds = {r['native_seed'] for rows in groups.values() for r in rows if r['game'] == game}
        if len(seeds) != 1:
            raise ValueError('Actual original initial RNG seeds were not paired')
        matched_seeds[str(game)] = seeds.pop()
    if len(set(matched_seeds.values())) != 20:
        raise ValueError('Unique native initial seeds required')

    teacher = read(args.teacher_report)
    if (teacher['status'] != 'completed' or teacher['cap_native_frames'] is not None
            or teacher['neural_model_test'] is not False or teacher['optimizer_updates'] != 0
            or teacher['continues_allowed'] is not False or teacher['policy_actions'] != 19):
        raise ValueError('Natural full-story reference contract required')

    dataset = args.engineering / 'teacher30-v1/demonstrations.json'
    demo = read(dataset)
    endpoints = demo['world_endpoints']
    if demo['samples'] != sum(row['teacher_actions_executed'] for row in endpoints):
        raise ValueError('Teacher sample/executed-action count mismatch')
    fit = []
    for folder in ('calibration30-v2', 'calibration30-v3-context',
                   'calibration-v6-student-replay', 'calibration-v7-receding-step-mmap-v2'):
        path = args.engineering / folder / 'teacher_fit_report.json'
        result = read(path)
        fit.append(dict(**source(path), chunk_initial_hidden=result['chunk_initial_hidden'],
                        restored_best_epoch=result['restored_best_epoch'],
                        epochs=result['epochs'], fit_status=result['fit_status']))
    student_path = args.engineering / 'student-aggregation-v1/demonstrations.json'
    student = read(student_path)
    routes = read(args.route_report)
    if routes['source_checkpoint_sha256'] != movement['pins']['trained']['sha256']:
        raise ValueError('Earlier-route audit belongs to another frozen policy')
    result = dict(
        schema='B1-teacher-coverage-and-fresh-movement-review-v1', status='completed', utc=utc_now(),
        movement_source=source(args.movement_report), checkpoint_pins=movement['pins'],
        policies={name: dict(**movement['policies'][name],
                    stage_at_first_collision=dict(Counter(row['stage'] for row in groups[name])),
                    stage1_no_hit_to_stage2=sum(row['stage'] >= 2 for row in groups[name]))
                  for name in policies},
        actual_original_initial_seeds_by_game=matched_seeds,
        paired=movement['paired'], greedy_paired=movement['greedy_paired'],
        stable_improvement_established=False, full_story_test_completed=False,
        teacher_reference=dict(source=source(args.teacher_report), policies=teacher['policies'],
                neural_model=False, reference_version='engineered V7',
                paired_with_current_B1_cohort=False,
                current_V10_full_game_performance_established=False),
        initial_demonstrations=dict(source=source(dataset), samples=demo['samples'],
                native_frames=demo['native_frames'], complete_native_episodes=demo['complete_native_episodes'],
                endpoint_stage_counts=dict(Counter(row['stage'] for row in endpoints)),
                endpoint_frames=sorted({row['episode_frames'] for row in endpoints}),
                input_schema=demo['input_schema']),
        student_state_aggregation=dict(source=source(student_path), samples=student['samples'],
                complete_native_episodes=student['complete_native_episodes'], outcomes=student['outcomes'],
                Bomb_allowed=student['Bomb_allowed_in_collection'],
                maximum_reported_endpoint_frames=max(student['endpoint_frames']),
                six_stage_teacher_success_trajectories_supplied=False),
        BC_context_evidence=fit,
        BC_context_zero_chunk_issue_already_addressed_in_later_calibrations=True,
        complete_lineage_coverage_or_single_cause_claimed=False,
        online_PPO_guidance_is_not_the_same_as_complete_teacher_trajectory_BC=True,
        earlier_route_audit=dict(source=source(args.route_report), summary=routes['summary'],
                unexecuted_native_alternatives_not_validated=True),
        storage_check=storage_check(args.storage_trace),
        native_worlds_started=0, optimizer_updates=0, CUDA_initialized=False,
        active_training_or_test_modified=False, test_observations_used_for_fitting=False,
        prospective_next_method_if_full_tests_fail=[
            'Collect new training-seed natural full-story teacher trajectories in current V5 input',
            'Use bounded lossless storage and group validation by native seed/trajectory',
            'Retain recurrent context, actual native outcomes and separate unprotected movement metrics',
            'Measure frozen native transfer before increasing PPO budget or model capacity'])
    args.output.mkdir(parents=True)
    write(args.output / 'report.json', result)
    (args.output / 'audit_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(status=result['status'], storage_check={k:v for k,v in result['storage_check'].items()
                         if k != 'traces'}, earlier_routes=routes['summary'])))


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    for name in ('movement-report', 'teacher-report', 'route-report', 'engineering', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--storage-trace', type=Path, nargs='+', required=True)
    run(p.parse_args())
