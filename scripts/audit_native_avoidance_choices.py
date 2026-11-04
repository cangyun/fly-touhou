"""Compare recorded movement with short-horizon forecasts of alternative moves.

Read-only diagnosis: alternative actions are never sent to a game. Forecast
clearance is not a replayed native counterfactual or an exact collision oracle.
"""
from dataclasses import replace
import argparse
import gzip
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from fly_touhou.campaign import FULL_GAME_SHA256  # noqa: E402
from fly_touhou.experiment import file_sha256  # noqa: E402
from fly_touhou.motion_calibration import calibrated_planner  # noqa: E402
from fly_touhou.native_hazard_planner import NativeHazardPlanner  # noqa: E402
from fly_touhou.planner import PlannerConfig  # noqa: E402
from fly_touhou.state import Snapshot  # noqa: E402


def audit(path, config):
    planner = NativeHazardPlanner(config)
    vulnerable = hazards = useful_labels = avoidable = hits = avoidable_hits = 0
    stocked_bomb = 0
    examples, started = [], time.perf_counter()
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            item = json.loads(line)
            if item.get('record_type') != 'transition':
                continue
            before = Snapshot.from_dict(item['snapshot'])
            if before.player is None or before.player.state != 1 or before.player.invulnerable_frames > 0:
                continue
            assert (before.character, before.subshot, before.difficulty) == (0, 0, 3)
            vulnerable += 1
            requested_stocked_bomb = item['action'] >= 18 and before.bombs > 0
            stocked_bomb += requested_stocked_bomb
            if requested_stocked_bomb:
                continue
            metrics = [planner.action_metrics(before, action) for action in range(18)]
            clearances = [row.clearance for row in metrics]
            chosen = clearances[item['action'] % 18]
            best = max(clearances)
            if math.isinf(best):
                assert all(math.isinf(value) for value in clearances)
                continue
            hazards += 1
            # A direction label is informative only if forecasts distinguish
            # unsafe and comfortably clear movement under measured motion.
            label_useful = min(clearances) < 0 and best >= 5.
            useful_labels += label_useful
            bad_choice = chosen < 0 and best >= 5.
            avoidable += bad_choice
            after_player = item['next_snapshot'].get('player')
            hit = bool(after_player and after_player['state'] == 4)
            hits += hit
            avoidable_hits += hit and bad_choice
            if len(examples) < 30 and (bad_choice or hit):
                best_actions = [i for i, value in enumerate(clearances) if value >= 5.]
                examples.append({'step': item.get('step'), 'stage_frame': before.stage_frame,
                    'chosen_action': item['action'], 'movement_action': item['action'] % 18,
                    'chosen_forecast_clearance_pixels': chosen, 'best_forecast_clearance_pixels': best,
                    'alternative_actions_with_5px_forecast_margin': best_actions,
                    'native_hit_on_recorded_next_snapshot': hit,
                    'player': {'x': before.player.x, 'y': before.player.y}, 'bombs': before.bombs})
    return {'path': str(path.resolve()), 'sha256': file_sha256(path),
            'vulnerable_decisions': vulnerable, 'stocked_Bomb_requests_excluded': stocked_bomb,
            'forecast_hazard_movement_decisions': hazards,
            'states_with_predicted_collision_and_clear_alternative': useful_labels,
            'chosen_predicted_collision_while_clear_alternative_exists': avoidable,
            'native_next_snapshot_hits_in_evaluated_movement': hits,
            'such_hits_with_predicted_clear_alternative': avoidable_hits,
            'elapsed_CPU_wall_seconds': time.perf_counter()-started, 'examples': examples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--traces', type=Path, nargs='+', required=True)
    parser.add_argument('--motion-calibration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve earlier movement diagnoses')
    config, evidence = calibrated_planner(PlannerConfig(), args.motion_calibration, exe_sha256=FULL_GAME_SHA256)
    config = replace(config, horizon_frames=8.)
    report = {'schema': 'measured-motion-native-movement-choice-diagnosis-v1',
        'game_started': False, 'model_trained': False, 'alternative_native_actions_executed': False,
        'scope': 'Eight-frame constant-action forecasts, not a native future replay or proof of avoidability',
        'limitations': ['Actual decisions change every four native frames; holding one alternative for eight frames is a diagnostic assumption.',
            'Future attack-script acceleration, activation, new spawns and unverified laser transformations are unknown.',
            'Positive forecast clearance is not confirmed native safety; stocked Bomb decisions are excluded.'],
        'script_sha256': file_sha256(__file__), 'motion_calibration': evidence,
        'predictor': NativeHazardPlanner(config).specification(), 'traces': [audit(path, config) for path in args.traces]}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'traces': [{key: value for key, value in row.items() if key not in ('examples', 'sha256')}
                                for row in report['traces']]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
