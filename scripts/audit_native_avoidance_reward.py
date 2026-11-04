"""Counterfactual feedback audit on sealed original game trajectories, CPU only.

This does not replay a brain or change the recorded policy's actions. It checks
reward definitions, native metadata, bounded shaping and computational cost.
"""
import argparse
import gzip
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np  # noqa: E402
from fly_touhou.campaign_rewards import NativeLifeAvoidanceReward  # noqa: E402
from fly_touhou.experiment import file_sha256  # noqa: E402
from fly_touhou.native import Profile  # noqa: E402
from fly_touhou.planner import PlannerConfig  # noqa: E402
from fly_touhou.state import Snapshot  # noqa: E402


def audit(path, config):
    evaluator = NativeLifeAvoidanceReward(gamma=.995, planner_config=config)
    count = cache_reuses = dense = exposed = 0
    totals, latencies, examples = {}, [], []
    discounted_shaping = initial_potential = 0.
    previous_after = None
    last_phi = 0.
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            item = json.loads(line)
            if item.get('record_type') != 'transition':
                continue
            before, after = Snapshot.from_dict(item['snapshot']), Snapshot.from_dict(item['next_snapshot'])
            if previous_after is not None and before == previous_after:
                before = previous_after
                cache_reuses += 1
            start = time.perf_counter()
            reward, components, details = evaluator.evaluate(before, after, item['elapsed_frames'],
                                                              terminated=bool(item['terminated']))
            latencies.append((time.perf_counter() - start)*1000)
            shaping = components['avoidance_potential_difference']
            dense += abs(shaping) > 1e-5
            exposed += details['before'].get('hazard', 0.) >= .2
            if count == 0:
                initial_potential = details['before']['value']
            last_phi = details['after']['value']
            assert -1.1000001 <= details['before']['value'] <= 0.
            assert -1.1000001 <= last_phi <= 0.
            assert abs(shaping) <= .2750001
            assert components['actual_life_decrement'] == -5.*max(0, before.lives - after.lives)
            assert abs(sum(components.values()) - reward) <= 1e-12
            discounted_shaping += evaluator.gamma**count * shaping
            for name, value in components.items():
                totals[name] = totals.get(name, 0.) + value
            if len(examples) < 20 and (details['native_deathbomb_entry'] or details['life_decrements']
                                      or abs(shaping) >= .01):
                examples.append({'step': item.get('step', count), 'stage_frame': before.stage_frame,
                    'actual_action': item['action'], 'components': components, 'diagnostics': details})
            previous_after = after
            count += 1
    identity = -.25*initial_potential + .25*evaluator.gamma**count*last_phi
    assert abs(discounted_shaping - identity) <= 1e-10
    return {'path': str(path.resolve()), 'sha256': file_sha256(path), 'decisions': count,
            'same_snapshot_cache_reuses': cache_reuses, 'feedback_above_1e_minus5_decisions': dense,
            'hazard_potential_at_least_0_point2_decisions': exposed, 'components_sum': totals,
            'discounted_shaping_identity_error': abs(discounted_shaping - identity),
            'evaluation_ms': {'mean': float(np.mean(latencies)), 'p95': float(np.quantile(latencies,.95)),
                              'maximum': float(max(latencies))},
            'examples': examples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--traces', type=Path, nargs='+', required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve counterfactual feedback evidence')
    profile = Profile.load(args.profile)
    left, right, top, bottom = profile.bounds
    config = PlannerConfig(left=left, right=right, top=top, bottom=bottom,
                           focused_speed=profile.slow_speed, normal_speed=profile.fast_speed)
    report = {'schema': 'sealed-native-avoidance-feedback-audit-v1', 'status': 'passed',
        'game_started': False, 'brain_replayed': False, 'learning_performed': False,
        'scope': 'counterfactual rewards on actual recorded transitions, not a policy performance result',
        'script_sha256': file_sha256(__file__),
        'reward_source_sha256': file_sha256(ROOT/'src/fly_touhou/campaign_rewards.py'),
        'profile_sha256': file_sha256(args.profile),
        'specification': NativeLifeAvoidanceReward(gamma=.995, planner_config=config).specification(),
        'traces': [audit(path, config) for path in args.traces]}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'status': 'passed', 'traces': [{'decisions': row['decisions'],
        'dense_feedback_decisions': row['feedback_above_1e_minus5_decisions'],
        'actual_life_penalty': row['components_sum']['actual_life_decrement'],
        'feedback_ms': row['evaluation_ms']} for row in report['traces']]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
