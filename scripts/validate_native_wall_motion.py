"""Compare the candidate with already-recorded, original-exe motion."""
import argparse
import json
from pathlib import Path
from collections import Counter
import numpy as np
from fly_touhou.thrl_dodge_diagnostics import _shared_binary
from fly_touhou.thrl_study_io import read_shared_json
from fly_touhou.native_wall_motion import parse_program, simulate


def read_lines(path):
    with _shared_binary(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def compare(pairs, programs, order):
    stats, skipped, flags, transitions = Counter(), Counter(), Counter(), Counter()
    examples = []
    for before, after in pairs:
        frames = after['stage_frame']-before['stage_frame']
        if frames not in (1, 4) or not before['bodies']:
            continue
        following = {point['id']: point for point in after['points']}
        body = before['bodies'][0]
        # Actual endpoints reconstruct held-input movement; clamped segments
        # are separately counted because linear interpolation is approximate.
        path = np.linspace(before['player'], after['player'], frames+1, dtype=np.float32)
        for point in before['points']:
            new = following.get(point['id'])
            if new is None or new['program_index'] != point['program_index']:
                continue
            try:
                result = simulate(point, programs[point['program_index']],
                    body['near_radius'], body['far_radius'], path, player_order=order)[-1]
            except ValueError as error:
                skipped[str(error)] += 1
                continue
            stats['matched_identity_intervals'] += 1
            error = float(np.max(np.abs(result['position']-new['native']['native_position'])))
            stats['maximum_position_error_pixels'] = max(stats['maximum_position_error_pixels'], error)
            flags[str(point['native']['native_ex_active_flags'])] += 1
            states_match = (result['marker'] == new['marker']
                and result['timer'] == new['reaction_timer']
                and result['active_flags'] == new['native']['native_ex_active_flags'])
            stats['state_matches'] += int(states_match)
            stats['position_errors_over_0_005_pixels'] += int(error > .005)
            if point['marker'] != new['marker']:
                transitions[f"{point['marker']}->{new['marker']}"] += 1
            if (error > .005 or not states_match) and len(examples) < 12:
                examples.append(dict(stage_frame=before['stage_frame'], point_id=point['id'],
                    position_error=error, old_marker=point['marker'], actual_marker=new['marker'],
                    predicted_marker=result['marker'], actual_flags=new['native']['native_ex_active_flags'],
                    predicted_flags=result['active_flags']))
    return dict(statistics=dict(stats), excluded=dict(skipped), active_flags=dict(flags),
                observed_switches=dict(transitions), mismatch_examples=examples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    programs = {row['index']: parse_program(row['program_hex'])
        for row in read_shared_json(args.capture/'wall-programs.json')['records']}
    singles = read_lines(args.capture/'wall-single-frame.jsonl')
    full = read_lines(args.capture/'wall-native.jsonl')
    pairs = [(a,b) for a,b in zip(full,full[1:]) if b['stage_frame']-a['stage_frame']==4]
    result = dict(kind='recorded_original_exe_conditional_motion_validation',
        note='Same teacher replay, not independent games; spawning state2 is excluded. This does not establish student success.',
        single_frame={order:compare([(r['before'],r['after']) for r in singles], programs,order)
            for order in ('before','after')},
        four_frames={order:compare(pairs,programs,order) for order in ('before','after')})
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__ == '__main__':
    main()
