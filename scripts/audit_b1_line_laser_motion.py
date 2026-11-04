"""Qualify endpoint measurements against bounded live native laser records.

Read-only source access with Windows write/delete sharing. This starts no
native process, imports no Torch and changes no running model or observation.
"""
from pathlib import Path
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from fly_touhou.atomic_json_io import atomic_write_json
from fly_touhou.state import Laser
from fly_touhou.thrl_dodge_diagnostics import _shared_binary
from fly_touhou.thrl_laser_motion_audit import SCHEMA as SOURCE_SCHEMA
from fly_touhou.thrl_line_laser_motion import line_endpoint, line_measurement
from fly_touhou.thrl_study_io import utc_now


def native_line(geometry):
    if geometry['node_count']:
        raise ValueError('Bounded curve samples do not establish complete node correspondence')
    return Laser(geometry['id'], geometry['kind'], geometry['x'], geometry['y'],
        geometry['angle'], geometry['length'], geometry['width'], geometry['conservative_active'],
        speed=geometry['speed'], age=geometry['age'], native_state=geometry['native_state'],
        native_iframes=geometry['native_iframes'])


def summarize_errors(rows):
    if not rows:
        return {'n': 0}
    return dict(n=len(rows),
        measured_motion_origin_error_mean=statistics.mean(x['measured'][0] for x in rows),
        measured_motion_endpoint_error_mean=statistics.mean(x['measured'][1] for x in rows),
        measured_motion_endpoint_error_max=max(x['measured'][1] for x in rows),
        zero_motion_origin_error_mean=statistics.mean(x['zero'][0] for x in rows),
        zero_motion_endpoint_error_mean=statistics.mean(x['zero'][1] for x in rows),
        nominal_speed_origin_error_mean=statistics.mean(x['nominal'][0] for x in rows),
        nominal_speed_endpoint_error_mean=statistics.mean(x['nominal'][1] for x in rows),
        pattern_changes=sum(x['pattern_change'] for x in rows))


def qualify(source):
    records = []
    sources = []
    excluded = Counter()
    by_pattern = defaultdict(list)
    for path in sorted(Path(source).glob('worker*.jsonl')):
        with _shared_binary(path) as stream:
            data = stream.read()
        read_length = len(data)
        complete_length = data.rfind(b'\n') + 1
        data = data[:complete_length]
        sources.append(dict(path=str(path.resolve()), read_bytes=read_length,
            complete_prefix_bytes=len(data), complete_prefix_sha256=hashlib.sha256(data).hexdigest(),
            ignored_partial_tail_bytes=read_length - len(data)))
        for raw in data.splitlines():
            row = json.loads(raw)
            if row['schema'] != SOURCE_SCHEMA:
                raise ValueError('Unexpected source motion-record contract')
            row['_worker'] = path.stem
            records.append(row)
            if row['before']['node_count'] or row['after']['node_count']:
                excluded['curve_node_correspondence_unknown'] += 1
                continue
            before, after = native_line(row['before']), native_line(row['after'])
            measured = line_measurement(after, before, elapsed_native_frames=row['elapsed_frames'])
            if measured.motion_observed != row['age_continuous']:
                raise ValueError('Telemetry continuity and independent measurement disagree')
            if not measured.motion_observed:
                excluded[measured.unknown_reason] += 1
                continue
            actual = (*measured.origin_velocity, *measured.endpoint_velocity, measured.width_rate)
            logged = (*row['measured_origin_velocity'], *row['measured_endpoint_velocity'], row['measured_width_rate'])
            if any(abs(a - b) > 1e-10 for a, b in zip(actual, logged)):
                raise ValueError('Independent endpoint measurement does not reproduce native record')
            row['_measurement'] = measured
            by_pattern[row['pattern']].append(row)
    if not sources or not records:
        raise ValueError('No complete native laser measurements available')

    def key(row, side):
        return (row['_worker'], row['native_episode'], row['stage'],
                row[side]['id'], row[side + '_frame'])

    preceding = {key(row, 'after'): row for row in records if '_measurement' in row}
    forecasts = defaultdict(list)
    unmatched = Counter()
    for following in records:
        previous = preceding.get(key(following, 'before'))
        if previous is None:
            unmatched['no_saved_immediately_preceding_pair'] += 1
            continue
        if '_measurement' not in following:
            unmatched['following_pair_not_continuous_line'] += 1
            continue
        if previous['after'] != following['before']:
            unmatched['shared_boundary_geometry_differs'] += 1
            continue
        a, b = native_line(previous['after']), native_line(following['after'])
        m = previous['_measurement']
        dt = following['elapsed_frames']
        ea, eb = line_endpoint(a), line_endpoint(b)
        origin_delta = (b.x - a.x, b.y - a.y)
        endpoint_delta = tuple(y - x for x, y in zip(ea, eb))
        nominal = (a.speed * math.cos(a.angle), a.speed * math.sin(a.angle))

        def residual(displacement, velocity):
            return math.hypot(*(x - dt * v for x, v in zip(displacement, velocity)))

        forecasts[previous['pattern']].append(dict(
            measured=(residual(origin_delta, m.origin_velocity), residual(endpoint_delta, m.endpoint_velocity)),
            zero=(math.hypot(*origin_delta), math.hypot(*endpoint_delta)),
            nominal=(residual(origin_delta, nominal), residual(endpoint_delta, nominal)),
            pattern_change=previous['pattern'] != following['pattern']))

    return dict(schema='B1-native-line-endpoint-measurement-qualification-v1', utc=utc_now(),
        source=str(Path(source).resolve()), sources=sources, native_rows=len(records),
        independently_reproduced_continuous_line_pairs=sum(map(len, by_pattern.values())),
        excluded_measurements=dict(excluded),
        measured_patterns={pattern:dict(n=len(rows),
            raw_states=dict(Counter(str(row['after']['native_state']) for row in rows)),
            nominal_speeds=sorted(set(row['after']['speed'] for row in rows)),
            mean_origin_speed=statistics.mean(math.hypot(*row['_measurement'].origin_velocity) for row in rows),
            mean_endpoint_speed=statistics.mean(math.hypot(*row['_measurement'].endpoint_velocity) for row in rows))
            for pattern, rows in by_pattern.items()},
        four_native_frame_forecast_by_preceding_pattern={k:summarize_errors(v) for k,v in forecasts.items()},
        forward_comparisons=sum(map(len, forecasts.values())), unmatched_forward_rows=dict(unmatched),
        limitations=[
            'The bounded recorder keeps early examples of each pattern; consecutive coverage is selective.',
            'Multiple beams/steps from the same worker are correlated, not independent game trials.',
            'Finite-difference velocities describe observed motion and do not guarantee the next step.',
            'Only exactly matching saved boundaries and continuous native ages enter forward comparisons.',
            'Raw native_state is not treated as a verified collision/lifecycle label.',
            'Curve-node correspondence, native collision outcomes and policy improvement are not verified.',
            'Scalar-speed and zero-motion errors are geometry baselines, not new native policies.'],
        native_instances_started=0, optimizer_updates=0, active_input_schema_changed=False,
        model_feature_encoding_registered=False, native_collision_state_verified=False,
        stable_avoidance_achieved=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve existing measurement evidence; use a new output path')
    result = qualify(args.source)
    atomic_write_json(args.output, result)
    print(json.dumps({k:result[k] for k in ('native_rows',
        'independently_reproduced_continuous_line_pairs', 'excluded_measurements',
        'measured_patterns', 'four_native_frame_forecast_by_preceding_pattern', 'forward_comparisons')},
        ensure_ascii=False))
