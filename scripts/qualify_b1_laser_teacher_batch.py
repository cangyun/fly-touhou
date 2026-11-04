"""Compare batched laser forecasts with the actual frozen V3 implementation.

Recorded native endpoint pairs supply geometry. Proposed player paths and
protection times are modeled qualification cases, not game outcomes. Timing
covers the isolated known-laser contribution, not end-to-end training.
"""
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch
import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from fly_touhou.atomic_json_io import atomic_write_json as write
from fly_touhou.thrl_study_io import read_shared_json as read, utc_now
from fly_touhou.thrl_dodge_diagnostics import _shared_binary
from fly_touhou.thrl_line_laser_motion import line_measurement
from fly_touhou.thrl_laser_teacher_batch import prepare_segments
from fly_touhou.resource_teacher_v9 import ObservedLaserGeometryPlanner
from fly_touhou.native_hazard_planner import GeometryNativeHazardPlanner
from fly_touhou.planner import PlannerConfig
from fly_touhou.state import Snapshot, Player
from audit_b1_line_laser_motion import native_line


def run(args):
    if args.output.exists():
        raise ValueError('Preserve qualification artifacts')
    source = args.reference_runtime/'src/fly_touhou/resource_teacher_v9.py'
    specification = importlib.util.spec_from_file_location('fly_touhou._frozen_laser_batch_reference', source)
    frozen = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(frozen)
    sensor = read(args.sensor_qualification)
    if sensor['status'] != 'qualified' or sensor['continuous_motion_pairs'] != 890:
        raise ValueError('Use the earlier sealed native-motion qualification cohort')
    rows = []
    for receipt in sensor['source']:
        with _shared_binary(Path(receipt['path'])) as stream:
            raw = stream.read(receipt['pinned_complete_bytes'])
        if hashlib.sha256(raw).hexdigest() != receipt['sha256']:
            raise ValueError('Recorded source prefix identity changed')
        for line in raw.splitlines():
            record = json.loads(line)
            if not record['age_continuous'] or record['before']['node_count'] or record['after']['node_count']:
                continue
            row = line_measurement(native_line(record['after']), native_line(record['before']),
                elapsed_native_frames=record['elapsed_frames'])
            if row.motion_observed:
                rows.append(row)
    if len(rows) != 890:
        raise ValueError('Pinned known-motion cohort differs')
    old = frozen.ObservedLaserGeometryPlanner(PlannerConfig())
    new = ObservedLaserGeometryPlanner(PlannerConfig())
    rng = np.random.default_rng(791103)
    count = 0
    max_error = 0.
    def no_nonlaser_forecast(self, snapshot, path):
        return math.inf
    # Both implementations get the same empty inherited contribution. This
    # deliberately isolates the changed known-laser computation.
    with patch.object(GeometryNativeHazardPlanner, 'path_clearance', no_nonlaser_forecast):
        for index in range(256):
            size = [0, 1, 4, 16, 32, 64][index % 6]
            selected = tuple(rows[i] for i in rng.integers(0, len(rows), size=size))
            protection = [0., .5, 4., 12.25, 200.][index % 5]
            snapshot = Snapshot(100, 100, Player(0., 368., invulnerable_frames=protection), stage=2)
            offset = rng.uniform(-3, 3, 2)
            radius = [1.05, 2., 12., 48.][index % 4]
            path = []
            for first, last in [(0., 4.), (4., 24.), (24.25, 60.75), (80., 160.)]:
                path.append((first, last, rng.uniform([-184, 32], [184, 432]), rng.uniform(-4.5, 4.5, 2)))
            for planner in (old, new):
                planner._observed_lines = selected
                planner._laser_player_offset = offset
                planner._laser_player_radius = radius
            new._observed_segment_arrays = prepare_segments(selected)
            before, after = old.path_clearance(snapshot, path), new.path_clearance(snapshot, path)
            if math.isinf(before) and math.isinf(after):
                error = 0.
            else:
                error = abs(before-after)
            max_error = max(max_error, error)
            if error > 1e-10 or (before > 0) != (after > 0) or (before > 1) != (after > 1):
                raise ValueError('Forecast/route-safety parity failed: '+repr((index, before, after)))
            count += 1
        # Warm both code paths on an explicit representative dense laser scene.
        selected = tuple(rows[i] for i in rng.integers(0, len(rows), size=64))
        snapshot = replace(snapshot, player=replace(snapshot.player, invulnerable_frames=0))
        path = [(0., 4., np.array([0., 368.]), np.array([0., 4.5])),
                (4., 24., np.array([0., 368.]), np.array([2., 0.]))]
        for planner in (old, new):
            planner._observed_lines = selected
            planner._laser_player_offset = np.zeros(2)
            planner._laser_player_radius = 2.
        new._observed_segment_arrays = prepare_segments(selected)
        for _ in range(3):
            old.path_clearance(snapshot, path)
            new.path_clearance(snapshot, path)
        timings = {}
        for label, planner in [('frozen_per_line', old), ('batched', new)]:
            values = []
            for _ in range(5):
                started = time.perf_counter()
                for _ in range(40):
                    planner.path_clearance(snapshot, path)
                values.append((time.perf_counter()-started)/40)
            timings[label] = float(np.median(values))
    result = dict(schema='B1-batched-known-laser-teacher-qualification-v1', status='qualified', utc=utc_now(),
        reference=str(source), reference_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        current_source_sha256=hashlib.sha256((ROOT/'src/fly_touhou/resource_teacher_v9.py').read_bytes()).hexdigest(),
        sensor_qualification=str(args.sensor_qualification), known_native_pairs=len(rows),
        modeled_path_cases=count, max_clearance_difference=max_error,
        protection_times=[0., .5, 4., 12.25, 200.], player_radii=[1.05, 2., 12., 48.],
        line_counts=[0, 1, 4, 16, 32, 64], fractional_and_zero_duration_endpoints=True,
        forecast_horizons_up_to=160., safe_zero_and_one_thresholds_identical=True,
        median_seconds_per_dense_scene=timings,
        isolated_known_laser_speed_ratio=timings['frozen_per_line']/timings['batched'],
        actual_frozen_source_used=True, nonlaser_contribution_isolated=True,
        training_speedup_claimed=False, native_evasion_claimed=False,
        current_running_frozen_job_modified=False, native_instances_started=0, optimizer_updates=0)
    args.output.mkdir(parents=True)
    write(args.output/'report.json', result)
    print(json.dumps({k: result[k] for k in ['status', 'known_native_pairs', 'modeled_path_cases',
        'max_clearance_difference', 'median_seconds_per_dense_scene', 'isolated_known_laser_speed_ratio']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    for key in ['reference-runtime', 'sensor-qualification', 'output']:
        parser.add_argument('--'+key, type=Path, required=True)
    run(parser.parse_args())
