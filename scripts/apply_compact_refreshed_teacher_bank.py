"""Install refreshed supervision while the existing learner is evaluating.

Keep the replay object's identity: the paused update already holds it as an
argument. Do not restart worlds, change parameters, or release the analysis hold.
"""
from pathlib import Path
from datetime import datetime, timezone
import os

from apply_compact_live_wall import frames, replace


def apply(plan_path):
    import psutil
    from fly_touhou.compact_native_rule_replay import NativeRuleTeacherReplay
    from fly_touhou.thrl_study_io import read_shared_json
    from fly_touhou.thrl_durable_records import write_json

    plan = read_shared_json(Path(plan_path))
    if os.getpid() != plan['pid'] or abs(psutil.Process().create_time()-plan['creation_time']) > .01:
        raise ValueError('Apply to the selected existing learner')
    stack = list(frames())
    if not any(f.f_code.co_name in ('evaluate', 'evaluate_before_snapshot') for f in stack):
        raise ValueError('Install only during the existing frozen comparison/probe')
    main = next(f for f in stack if f.f_code.co_name == 'main' and
                Path(f.f_code.co_filename).name == 'train_compact_story_ppo.py')
    state = main.f_locals
    replay = state['native_rule_replay']
    weights = dict(enumerate(plan['stage_weights'], 1))
    if replay.path == Path(plan['manifest']) and replay.stage_weights == weights:
        return dict(status='already_applied')

    replacement = NativeRuleTeacherReplay(plan['manifest'], stage_weights=weights)
    previous = replay.summary()
    old_identity = id(replay)
    replace('compact_native_rule_replay', 'summary', cls='NativeRuleTeacherReplay')
    replay.__dict__.clear()
    replay.__dict__.update(replacement.__dict__)
    options = state['native_teacher']
    options['manifest'] = Path(plan['manifest'])
    options['stage_weights'] = list(plan['stage_weights'])
    state['a'].native_rule_teacher_manifest = Path(plan['manifest'])
    state['a'].native_rule_teacher_stage_weights = list(plan['stage_weights'])
    checkpoint_fields = main.f_globals['native_teacher_checkpoint_fields'](options)
    state['launch'].update(checkpoint_fields)
    output = Path(state['a'].output)
    write_json(output/'native-rule-teacher-replay.json', replay.summary())
    write_json(output/'launch.json', state['launch'])
    result = dict(status='applied', PID=os.getpid(), utc=datetime.now(timezone.utc).isoformat(),
        previous=previous, current=replay.summary(), checkpoint_fields=checkpoint_fields,
        replay_identity_preserved=id(replay)==old_identity,
        PPO_updates=state['updates'], pending_cycle=state['cycle'],
        model_parameters_changed=False, optimizer_changed=False,
        collector_reset=False, PPO_sampling_changed=False, analysis_hold_released=False,
        teacher_data_only=True)
    write_json(Path(plan['receipt']), result)
    return result
