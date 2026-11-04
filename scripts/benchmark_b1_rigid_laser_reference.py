"""Time unchanged V10 labels on sealed measured geometry, without game worlds."""
import argparse
import hashlib
import statistics
import sys
import time
from pathlib import Path


def run(args):
    sys.path.insert(0, str(args.runtime / 'src'))
    import numpy as np
    from fly_touhou.atomic_json_io import atomic_write_json as write
    from fly_touhou.resource_teacher_v10 import REVISION
    from fly_touhou.state import Snapshot
    from fly_touhou.thrl_native_env import SOURCE19_TO_NATIVE36
    from fly_touhou.thrl_study_io import read_shared_json as read, utc_now
    from fly_touhou.thrl_teacher import make_teacher

    cases = read(args.dataset)['cases']
    snapshots = [Snapshot.from_dict(raw) for case in cases
                 if case['name'].startswith('recorded-native-')
                 for raw in case['snapshots']]
    if not snapshots:
        raise ValueError('Measured native snapshot required')
    teacher = make_teacher(REVISION)
    rows = []
    for index, snapshot in enumerate(snapshots):
        wall, cpu, hashes = [], [], []
        for repeat in range(args.repeats + 1):
            teacher.reset()
            started_wall, started_cpu = time.perf_counter(), time.process_time()
            p = teacher.probabilities(snapshot)
            elapsed_cpu = time.process_time() - started_cpu
            elapsed_wall = time.perf_counter() - started_wall
            q = np.asarray(p)[list(SOURCE19_TO_NATIVE36)].astype(np.float32)
            hashes.append(hashlib.sha256(q.tobytes()).hexdigest())
            if repeat:
                wall.append(elapsed_wall)
                cpu.append(elapsed_cpu)
        if len(set(hashes)) != 1:
            raise ValueError('Reset repeated-state labels differed')
        rows.append(dict(snapshot=index, source19_FP32_sha256=hashes[0],
                         wall_seconds=wall, cpu_seconds=cpu,
                         median_wall_seconds=statistics.median(wall),
                         median_cpu_seconds=statistics.median(cpu)))
    write(args.output, dict(status='completed', utc=utc_now(),
          runtime=str(args.runtime), dataset=str(args.dataset),
          dataset_sha256=hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
          repeats=args.repeats, discarded_warmup_per_snapshot=1, rows=rows,
          process_startup_import_and_state_decoding_included=False,
          teacher_reset_before_each_call=True, native_worlds_started=0,
          optimizer_updates=0, full_training_speedup_claimed=False))
    print(str(args.output))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    for name in ('runtime', 'dataset', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=5)
    run(parser.parse_args())
