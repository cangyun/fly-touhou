"""Copy already collected contact windows from the owned trainer, without stepping it."""
from datetime import datetime, timezone
from pathlib import Path
import os
import sys
import math


FIELDS = (
    'features', 'actions', 'outcomes', 'native_action_frames', 'unprotected_mask',
    'margins', 'recorded_pre_action_trusted', 'recorded_native_stage',
    'recorded_native_episode', 'recorded_episode_frames', 'rule_weights',
    'rule_probabilities', 'recorded_neural_horizon_scores', 'recorded_policy_logits',
    'recorded_behavior_policy_logits', 'recorded_route_scores', 'deathbomb_mask',
)


def json_value(value):
    if hasattr(value, 'tolist'):
        value = value.tolist()
    if isinstance(value, list):
        return [json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def capture(output, expected_pid):
    from fly_touhou.thrl_durable_records import write_json
    if os.getpid() != expected_pid:
        raise ValueError('This read-only capture belongs to a different trainer')
    active = None
    for frame in sys._current_frames().values():
        while frame is not None:
            if (frame.f_code.co_name == 'rollout'
                    and Path(frame.f_code.co_filename).name == 'compact_story_ready.py'):
                active = frame
                break
            frame = frame.f_back
        if active is not None:
            break
    if active is None:
        write_json(Path(output), dict(status='no_active_collection', PID=os.getpid()))
        return
    local = active.f_locals
    records = [list(world) for world in local['records']]
    collector = local['self']
    copied = []
    for world, rows in enumerate(records):
        for index, record in enumerate(rows):
            if int(record['outcomes']) != 1:
                continue
            first = max(0, index - 128)
            window = []
            for item in rows[first:min(len(rows), index + 4)]:
                snapshot = {}
                for key in FIELDS:
                    value = item[key]
                    snapshot[key] = json_value(value)
                window.append(snapshot)
            copied.append(dict(world=world, world_record=index, contact_index=index-first,
                               records=window))
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json(destination, dict(status='captured', captured_utc=datetime.now(timezone.utc).isoformat(),
        PID=os.getpid(), policy_version=collector.policy_version,
        total_student_decisions=collector.actions, per_world_records=[len(x) for x in records],
        movement_temperature=collector.movement_sampling_temperature,
        Bomb_temperature=collector.bomb_sampling_temperature,
        neural_risk_scale=local['model'].neural_risk_scale, contacts=copied,
        masked_nonfinite_logits='String -inf preserves masked actions in strict JSON.',
        native_actions_executed=0, live_parameters_changed=False,
        scope='Read-only copy of existing correlated training windows; not independent evaluation.'))
