"""Read-only Bomb economy audit and candidate reward calculation on sealed games.

No native game, full brain, CUDA allocation, action override or new learning.
"""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from fly_touhou.campaign_rewards import (NativeEmergencyBombReward,NativeGeometryEmergencyReward,
    EMERGENCY_BOMB_REWARD,GEOMETRY_EMERGENCY_REWARD)  # noqa: E402
from fly_touhou.experiment import file_sha256  # noqa: E402
from fly_touhou.native import Profile  # noqa: E402
from fly_touhou.planner import PlannerConfig  # noqa: E402
from fly_touhou.state import Snapshot  # noqa: E402


def audit(path, config, reward_profile=EMERGENCY_BOMB_REWARD):
    reward_type={EMERGENCY_BOMB_REWARD:NativeEmergencyBombReward,
                 GEOMETRY_EMERGENCY_REWARD:NativeGeometryEmergencyReward}[reward_profile]
    evaluator = reward_type(gamma=.995, planner_config=config)
    counts, totals, examples, timings = Counter(), Counter(), [], []
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if row.get('record_type') != 'transition':
                continue
            before, after = Snapshot.from_dict(row['snapshot']), Snapshot.from_dict(row['next_snapshot'])
            started = time.perf_counter()
            value, components, details = evaluator.evaluate(before, after, row['elapsed_frames'],
                terminated=bool(row['terminated']), action=row['action'])
            timings.append((time.perf_counter()-started)*1000.)
            evidence = details['bomb']
            counts['decisions'] += 1
            counts['requests'] += evidence['bomb_requested']
            counts['empty_requests'] += evidence['bomb_requested'] and before.bombs <= 0
            counts['requests_during_invulnerability'] += evidence['bomb_requested'] and evidence['invulnerable_before']
            counts['supported_spend_units'] += evidence['supported_spend_units']
            counts['supported_emergency_units'] += evidence['supported_spend_units'] if evidence['supported_spend_is_emergency'] else 0
            counts['observed_deathbomb_recoveries'] += evidence['deathbomb_recovery_observed']
            counts['unattributed_inventory_decrement'] += evidence['unattributed_inventory_decrement']
            counts['deathbomb_opportunities_with_stock'] += evidence['deathbomb_window_before'] and before.bombs > 0
            counts['requests_in_stocked_deathbomb_window'] += evidence['deathbomb_window_before'] and before.bombs > 0 and evidence['bomb_requested']
            assert components['survival'] == 0.
            assert components['bomb_inventory_spend'] <= 0.
            assert abs(sum(components.values()) - value) <= 1e-12
            totals.update(components)
            totals['removed_elapsed_survival_reward'] += details['removed_elapsed_survival_reward']
            if evidence['supported_spend_units'] or evidence['deathbomb_window_before'] or evidence['unattributed_inventory_decrement']:
                examples.append({'step': row.get('step'), 'stage_frame': before.stage_frame,
                    'actual_action': row['action'], 'bomb': evidence, 'spend_reward': components['bomb_inventory_spend']})
    counts['supported_nonemergency_units'] = counts['supported_spend_units']-counts['supported_emergency_units']
    return {'path': str(path.resolve()), 'sha256': file_sha256(path), 'counts': dict(counts),
            'counterfactual_reward_component_sums': dict(totals), 'events': examples,
            'feedback_ms': {'mean': sum(timings)/len(timings), 'max': max(timings)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--traces', nargs='+', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reward-profile',choices=(EMERGENCY_BOMB_REWARD,GEOMETRY_EMERGENCY_REWARD),
                        default=EMERGENCY_BOMB_REWARD,
                        help='Explicit feedback geometry for this read-only recomputation')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve earlier Bomb-economy evidence')
    profile = Profile.load(args.profile)
    left, right, top, bottom = profile.bounds
    config = PlannerConfig(left=left, right=right, top=top, bottom=bottom,
        focused_speed=profile.slow_speed, normal_speed=profile.fast_speed)
    reward_type={EMERGENCY_BOMB_REWARD:NativeEmergencyBombReward,
                 GEOMETRY_EMERGENCY_REWARD:NativeGeometryEmergencyReward}[args.reward_profile]
    report = {'schema':'sealed-native-bomb-economy-feedback-audit-v1', 'status':'passed',
        'game_started':False, 'brain_replayed':False, 'model_trained':False,
        'scope':'recorded requests and conservative inventory attribution, candidate rewards only; no new policy ability claim',
        'limits':['Pickups may mask actual consumption; inventory observations are not a direct Bomb-start hook.',
                  'Life/stage/screen changes with inventory decreases remain unattributed.',
                  'Emergency movement forecasts use current measured geometry and gates; future scripts are unknown.'],
        'script_sha256':file_sha256(__file__),
        'reward_source_sha256':file_sha256(ROOT/'src/fly_touhou/campaign_rewards.py'),
        'profile_sha256':file_sha256(args.profile),
        'specification':reward_type(gamma=.995, planner_config=config).specification(),
        'traces':[audit(path, config,args.reward_profile) for path in args.traces]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({'status':'passed','traces':[{'path':r['path'],'counts':r['counts']} for r in report['traces']]},ensure_ascii=False))


if __name__ == '__main__':
    main()
