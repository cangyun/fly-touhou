"""Round-trip sealed native FP32 inputs as non-training storage samples."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fly_touhou.atomic_json_io import atomic_write_json as write
from fly_touhou.thrl_study_io import read_shared_json as read, utc_now
from fly_touhou.thrl_teacher_archive import TeacherTrajectoryWriter, TeacherTrajectoryReader


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    report = read(args.report)
    if report['status'] != 'completed':
        raise ValueError('Use sealed native evidence only')
    selected = []
    for policy in ('trained_greedy', 'trained_sampled'):
        for stage in (1, 2):
            selected.extend([r for r in report['episode_results']
                             if r['policy'] == policy and r['stage'] == stage][:2])
    if len(selected) != 8:
        raise ValueError('Eight real first/second-stage cases required')
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    cases = []
    for case in selected:
        receipt_path = Path(case['failure_trace']['receipt'])
        receipt = read(receipt_path)
        arrays_path = Path(receipt['arrays_file'])
        if (sha(arrays_path) != receipt['sha256']
                or receipt['checkpoint_sha256'] != report['pins']['trained']['sha256']):
            raise ValueError('Native input provenance changed')
        with np.load(arrays_path, allow_pickle=False) as data:
            fields = {name: (data[name].dtype, data[name].shape[1:])
                      for name in ('features', 'objects', 'maps')}
            fields['dones'] = (np.bool_, ())
            rows = [{**{name: data[name][i].copy() for name in ('features', 'objects', 'maps')},
                     'dones': np.array(False, np.bool_)} for i in (0, len(data['features']) - 1)]
        folder = args.output / (case['policy'] + '-game' + str(case['game']))
        writer = TeacherTrajectoryWriter(folder, fields, chunk_rows=1, reserve_bytes=1024**3,
            metadata=dict(qualification_only=True, observations_are_not_contiguous=True,
                          native_episode_claimed=False, use_for_training=False,
                          source_receipt=str(receipt_path), source_receipt_sha256=sha(receipt_path)))
        for row in rows:
            writer.append(row)
        index = writer.close_prefix('Storage qualification samples;not a complete native episode or training data')
        try:
            TeacherTrajectoryReader(folder)
        except ValueError:
            pass
        else:
            raise ValueError('Qualification prefix was accepted as complete training trajectory')
        reader = TeacherTrajectoryReader(folder, require_complete=False, cache_bytes=0)
        restored = reader.fetch(0, 2)
        for name in fields:
            expected = np.stack([row[name] for row in rows])
            if (restored[name].dtype != expected.dtype or restored[name].shape != expected.shape
                    or restored[name].tobytes() != expected.tobytes()):
                raise ValueError('Real original input bits changed: ' + name)
        cases.append(dict(policy=case['policy'], game=case['game'], stage=case['stage'],
                          source_receipt=str(receipt_path), source_sha256=receipt['sha256'],
                          exact_original_observations_restored=2,
                          raw_array_bytes=sum(row[name].nbytes for row in rows for name in fields),
                          shard_bytes=sum(part['bytes'] for part in index['shards']),
                          natural_episode_complete=False))
    if 'torch' in sys.modules:
        raise RuntimeError('Storage qualification must not load Torch or CUDA')
    result = dict(schema='B1-native-FP32-teacher-shard-qualification-v1', utc=utc_now(),
                  status='passed', cases=cases,
                  original_FP32_observations_restored=sum(row['exact_original_observations_restored'] for row in cases),
                  all_original_bits_preserved=True, prefixes_rejected_as_completed_trajectories=True,
                  raw_array_bytes=sum(row['raw_array_bytes'] for row in cases),
                  shard_bytes=sum(row['shard_bytes'] for row in cases),
                  wall_seconds=time.perf_counter() - started,
                  source_report=str(args.report), source_report_sha256=sha(args.report),
                  storage_module_sha256=sha(ROOT/'src/fly_touhou/thrl_teacher_archive.py'),
                  script_sha256=sha(Path(__file__)),
                  native_worlds_started=0, optimizer_updates=0, CUDA_initialized=False,
                  active_evaluator_modified=False, observations_used_for_training=0,
                  full_story_collection_or_model_learning_not_yet_qualified=True)
    write(args.output/'report.json', result)
    (args.output/'qualification_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({key: value for key, value in result.items() if key != 'cases'}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    run(p.parse_args())
