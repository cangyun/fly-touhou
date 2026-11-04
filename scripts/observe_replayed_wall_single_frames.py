"""Temporarily split recorded input spans into identical one-tick spans.

This records a bounded set of native wall transitions in the existing replay, with
the same input bits, total ticks, original game and event counters.
"""
from pathlib import Path
import os
import sys
import math


def arm(output, limit=64):
    import psutil
    from fly_touhou.thrl_study_io import read_shared_json
    from fly_touhou.thrl_durable_records import append_jsonl, write_json
    output = Path(output)
    owner = read_shared_json(output / 'ownership.json')
    if os.getpid() != owner['pid'] or abs(psutil.Process().create_time()-owner['create_time'])>.01:
        raise ValueError('Only the current recorded teacher replay is eligible')
    main = None
    for frame in sys._current_frames().values():
        while frame is not None:
            if frame.f_code.co_name == 'main' and Path(frame.f_code.co_filename).name == 'replay_compact_teacher_wall.py':
                main = frame
                break
            frame = frame.f_back
        if main is not None:
            break
    if main is None:
        raise RuntimeError('Replay loop unavailable')
    local = main.f_locals
    env, programs = local['env'], local['programs']
    controller = env.controller
    if getattr(controller, '_wall_single_frame_observer', False):
        return {'status': 'already_armed'}
    phase = main.f_globals['phase']
    capture = main.f_globals['capture_reactive']
    original = controller.advance
    count = 0
    measuring = False

    def advance(mask, ticks):
        nonlocal count, measuring
        before = env.snapshot
        contacts = phase(before)[1] if before.stage == 6 else []
        if not contacts or count >= limit:
            return original(mask, ticks)
        pre = capture(before, controller._memory, contacts, programs)
        if not measuring:
            near, far = pre['bodies'][0]['near_radius'], pre['bodies'][0]['far_radius']
            measuring = any(abs(math.dist(point['native']['native_position'], pre['player'])
                - (near if point['marker']==1 else far)) < 32 for point in pre['points'])
            if not measuring:
                return original(mask, ticks)
        for tick in range(ticks):
            if tick:
                before = after
                contacts = phase(before)[1]
            if count < limit and contacts:
                pre = capture(before, controller._memory, contacts, programs)
                after = original(mask, 1)
                post_contacts = phase(after)[1]
                post = capture(after, controller._memory, post_contacts, programs)
                append_jsonl(output / 'wall-single-frame.jsonl', dict(
                    ordinal=count, input_mask=int(mask), requested_ticks=1,
                    native_frame_gap=after.frame-before.frame, before=pre, after=post))
                count += 1
            else:
                after = original(mask, 1)
        if count >= limit:
            controller.advance = original
            controller._wall_single_frame_observer = False
            write_json(output / 'single-frame-observer.json', dict(status='completed',
                native_transitions=count, input_masks_changed=False, extra_native_ticks=0))
        return after

    controller.advance = advance
    controller._wall_single_frame_observer = True
    receipt = dict(status='armed', pid=os.getpid(), maximum_native_transitions=limit,
        input_masks_changed=False, extra_native_ticks=0, model_or_teacher_parameters_changed=False)
    write_json(output / 'single-frame-observer.json', receipt)
    return receipt
