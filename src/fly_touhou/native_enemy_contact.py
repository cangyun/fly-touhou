"""TH14 1.00b body-contact gates and shared near-future ECL changes.

Native evidence: 423d4d..423d81 gates body contact; 424810 runs movement,
ECL, the optional tick callback, contact, then decrements timer4000. 4115a0
uses float32 subtraction and truncation toward zero. 482380/483e70 execute
all ECL contexts; this module reads their bytecode, never executes game code.

This forecasts contact *availability*, not positions or collision outcomes.
Unimplemented control flow, variable contact arguments, and custom callbacks
produce unknowns, rather than a prediction of safety.

Circle contact uses r_enemy=Data[0x118]/2 and the exact effective player circle
component from 44efa0: dist_squared < r_enemy**2 + r_player**2. It is not the
usual sum-of-radii test. Rectangle geometry exposes measured angles and an
all-rotation bound; ANM animation and its later Data+120 copy are not forecast.
"""
from __future__ import annotations

import math
import struct


SCHEMA = 'th14-native-enemy-contact-v1'
CONTACT_MASK = 0x22 | 0x04000000
GLOBAL_SCALE_RVA = 0xd8f58
MAX_THREADS = 64
MAX_INSTRUCTIONS = 512


def _f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def _finite(value):
    return value if value is not None and math.isfinite(value) else None


def _normalized_angle(angle):
    """403590's float32 +/-pi normalization before the ANM anchor trig."""
    pi, two_pi, count = _f32(math.pi), _f32(2 * math.pi), 0
    while angle > pi:
        previous = count
        count += 1
        angle = _f32(angle - two_pi)
        if previous > 32:
            break
    while angle < -pi:
        previous = count
        count += 1
        angle = _f32(angle + two_pi)
        if previous > 32:
            break
    return angle


def rectangle_contact_geometry(width, height, rotation, anm_rotation_z):
    """Equivalent centered rectangle for native 423dad / 44f0b0.

    44f0b0 uses long-axis [0,height] and short-axis [-width/2,width/2].
    Data+120 rotates that long axis. ANM+50 only positions its starting anchor.
    The returned centered representation swaps extents to [height/2,width/2],
    so its rotation is exactly Data+120, not the sum of the two native angles.
    Missing ANM angle yields a conservative AABB over all anchor rotations.
    """
    if any(_finite(value) is None for value in (width, height)):
        return None
    half_width, half_height = _f32(width * .5), _f32(height * .5)
    if min(half_width, half_height) < 0:
        return None
    result = {'known': False, 'angle': _finite(rotation),
        'half_extents': [half_height, half_width], 'anchor_offset': None,
        'center_offset': None, 'aabb_center_offset': None,
        'aabb_half_extents': None, 'anm_rotation_z': _finite(anm_rotation_z),
        'all_rotations_aabb_half_extents': [
            half_height + math.hypot(height, half_width)] * 2,
        'all_rotations_aabb_center_offset': [0., 0.]}
    if _finite(rotation) is None:
        return result
    sin_theta, cos_theta = _f32(math.sin(rotation)), _f32(math.cos(rotation))
    long_center = [_f32(cos_theta * half_height), _f32(sin_theta * half_height)]
    aabb_half = [abs(cos_theta) * half_height + abs(sin_theta) * half_width,
                 abs(sin_theta) * half_height + abs(cos_theta) * half_width]
    if _finite(anm_rotation_z) is None:
        result.update(aabb_center_offset=long_center,
            aabb_half_extents=[side + half_height for side in aabb_half],
            aabb_scope='all possible ANM anchor rotations at the measured Data rotation')
        return result
    anchor_angle = _normalized_angle(_f32(anm_rotation_z + _f32(math.pi / 2)))
    anchor = [-_f32(_f32(math.sin(anchor_angle)) * half_height),
               _f32(_f32(math.cos(anchor_angle)) * half_height)]
    center = [_f32(anchor[i] + long_center[i]) for i in range(2)]
    result.update(known=True, anchor_offset=anchor, center_offset=center,
        aabb_center_offset=center, aabb_half_extents=aabb_half,
        aabb_scope='measured native rectangle')
    return result


def _read_animation(read, anm_id, image_base):
    """Mirror 47f0a0 VM lookup, including both slow-list heads."""
    result = {'id': anm_id, 'manager': None, 'address': None,
              'rotation_z': None, 'lookup': None}
    if not anm_id:
        return result
    manager_read = read(image_base + 0xf56cc, '<I')
    if manager_read is None or not manager_read[0]:
        return result
    manager = result['manager'] = manager_read[0]
    index = anm_id & 0x1fff
    vm = None
    if index != 0x1fff:
        candidate = manager + 0xdc + index * 0x5e4
        identity = read(candidate + 0x540, '<I')
        active = read(candidate + 0x5dc, '<B')
        if identity is not None and identity[0] == anm_id and active and active[0]:
            vm = candidate
            result['lookup'] = 'fast_array'
    else:
        for offset in (0xfe8208, 0xfe8210):
            head = read(manager + offset, '<I')
            node = None if head is None else head[0]
            seen = set()
            while node and node not in seen and len(seen) < 16384:
                seen.add(node)
                item = read(node, '<II')
                if item is None:
                    break
                candidate, node = item
                identity = read(candidate + 0x540, '<I')
                if identity is not None and identity[0] == anm_id:
                    vm = candidate
                    result['lookup'] = 'slow_list'
                    break
            if vm:
                break
    if vm:
        result['address'] = vm
        angle = read(vm + 0x50, '<f')
        result['rotation_z'] = None if angle is None else _finite(angle[0])
    return result


def _native_scale(raw, pointer, local, global_pointer, *, timer=False):
    if pointer == 0:
        return 1.0 if timer else None
    if raw is None:
        return None
    value = raw
    if pointer == global_pointer:
        if local is None:
            return None
        if local > 0:
            value = min(1.0, max(0.0, _f32(raw - _f32(local * raw))))
    # 4115a0 treats values strictly between these float32 bounds as one.
    if timer and _f32(.99) < value < _f32(1.01):
        return 1.0
    return value


def _read_thread(read, context, scale, horizon, order, image_base, *, skip_first_tick=False, capture_spawns=False):
    header = read(context, '<fI')
    rank = read(context + 0x101c, '<B')
    result = {'context': context, 'order': order, 'time': None,
              'instruction_pointer': None, 'rank_mask': None,
              'next_instruction': None, 'events': [], 'spawn_instructions': [],
              'unknown_from_tick': None, 'unmodeled': []}

    def unknown(tick, reason):
        result['unknown_from_tick'] = tick
        result['unmodeled'].append(reason)

    if header is None or rank is None or _finite(header[0]) is None:
        unknown(1, 'unreadable_ecl_context')
        return result
    clock, ip = header
    rank = rank[0]
    result.update(time=clock, instruction_pointer=ip, rank_mask=rank)
    if scale is None:
        unknown(1, 'unknown_ecl_timescale')
        return result
    cache = {}
    steps = 0
    event_order = 0

    for tick in range(1, horizon + 1):
        # New enemies may already have their Data update guard set. 424810
        # returns before ECL/contact/timers; 4229ac then clears the guard.
        if tick == 1 and skip_first_tick:
            continue
        while ip:
            if ip not in cache:
                raw = read(ip, '<ihHHBB')
                if raw is None:
                    unknown(tick, 'unreadable_ecl_instruction')
                    return result
                due, opcode, size, mask, command_rank, padding = raw
                cache[ip] = (due, opcode, size, mask, command_rank)
            due, opcode, size, mask, command_rank = cache[ip]
            if result['next_instruction'] is None:
                result['next_instruction'] = {'address': ip, 'time': due,
                    'opcode': opcode, 'size': size, 'parameter_mask': mask,
                    'rank_mask': command_rank}
            if clock < due:
                break
            steps += 1
            if steps > MAX_INSTRUCTIONS:
                unknown(tick, 'ecl_instruction_budget')
                return result
            if size < 0x10:
                unknown(tick, 'invalid_ecl_instruction_size')
                return result
            if not command_rank & rank:
                ip += size
                continue

            def literal(index, kind='i'):
                if mask & (1 << index) or size < 0x14 + 4 * index:
                    return None
                value = read(ip + 0x10 + 4 * index, '<' + kind)
                return None if value is None else value[0]

            if opcode == 1:  # Native context termination, not a sub return.
                ip = 0
                break
            if opcode == 12:  # Native jump reads both arguments directly.
                jump = read(ip + 0x10, '<ii') if size >= 0x18 else None
                if jump is None:
                    unknown(tick, 'unreadable_ecl_jump')
                    return result
                ip += jump[0]
                clock = _f32(float(jump[1]))
                continue
            if opcode in (23, 24):
                delay = literal(0, 'i' if opcode == 23 else 'f')
                if _finite(delay) is None:
                    unknown(tick, 'variable_or_unreadable_ecl_wait')
                    return result
                clock = _f32(clock - _f32(float(delay)))
            elif 10 <= opcode <= 21:
                unknown(tick, f'unmodeled_ecl_control_flow_{opcode}')
                return result
            elif opcode in (500, 501, 502, 503, 515, 541, 563, 564, 632, 634, 639):
                count = 2 if opcode in (500, 501) else 1
                kind = 'f' if opcode in (500, 501, 564) else 'i'
                args = [literal(i, kind) for i in range(count)]
                if any(_finite(value) is None for value in args):
                    # 500/515 only control receiving player shots, not body contact.
                    if opcode not in (500, 515):
                        unknown(tick, f'variable_or_unreadable_contact_opcode_{opcode}')
                        return result
                event = {'tick': tick, 'thread_order': order,
                    'instruction_order': event_order, 'address': ip,
                    'time': due, 'opcode': opcode, 'arguments': args}
                if opcode in (632, 634, 639) and args[0] is not None:
                    table_rva = 0xdb548 if opcode == 634 else 0xd4cec
                    entry = read(image_base + table_rva + args[0] * 4, '<I')
                    event['callback'] = None if entry is None else entry[0]
                    if entry is None:
                        unknown(tick, 'unreadable_ecl_callback')
                        return result
                result['events'].append(event)
                event_order += 1
            elif capture_spawns and opcode in (300,301,304,305,309,310,311,312):
                # Record the instruction bytes only; resolving string/variable
                # parameters and proving actual spawn geometry are separate.
                raw_args=read(ip+0x10,f'<{size-0x10}s') if size>0x10 else (b'',)
                result['spawn_instructions'].append(dict(tick=tick,opcode=opcode,
                    address=ip,parameter_mask=mask,
                    arguments_hex=None if raw_args is None else raw_args[0].hex()))
            # Ordinary arithmetic/shot/animation instructions do not directly
            # change the verified contact gates. Variable contact arguments and
            # branches are stopped above; geometry/movement is a separate model.
            ip += size
        clock = _f32(clock + scale)
    return result


def read_enemy_contact(memory, enemy_base, data_address, *,
                       image_base=0x400000, horizon=24, capture_spawns=False,
                       motion_state=None):
    """Read one same-tick enemy through the existing memory.read API.

    The caller owns checkpoint/frame consistency. Missing optional reads remain
    null and are listed, so archived or partial observations cannot become safe.
    """
    if not isinstance(horizon, int) or not 1 <= horizon <= 240:
        raise ValueError('Contact forecast horizon must be an integer in 1..240')
    missing = []

    def read(address, fmt):
        try:
            return struct.unpack(fmt, memory.read(address, struct.calcsize(fmt)))
        except Exception:
            missing.append({'address': address, 'format': fmt})
            return None

    def scalar(address, fmt):
        value = read(address, fmt)
        return None if value is None else value[0]

    flags = scalar(data_address + 0x4054, '<Q')
    timer = read(data_address + 0x4000, '<iifII')
    local = _finite(scalar(enemy_base + 0x14d4, '<f'))
    global_pointer = image_base + GLOBAL_SCALE_RVA
    timer_int = None if timer is None else timer[1]
    timer_float = None if timer is None else _finite(timer[2])
    timer_pointer = None if timer is None else timer[3]
    raw_scale = (None if timer_pointer is None else 1.0 if timer_pointer == 0
                 else _finite(scalar(timer_pointer, '<f')))
    timer_scale = _native_scale(raw_scale, timer_pointer, local, global_pointer, timer=True)
    ecl_pointer = scalar(data_address + 0x2c8, '<I')
    ecl_raw_scale = (None if not ecl_pointer
                     else _finite(scalar(ecl_pointer, '<f')))
    ecl_scale = _native_scale(ecl_raw_scale, ecl_pointer, local, global_pointer)
    global_scale = (raw_scale if timer_pointer == global_pointer else
                    ecl_raw_scale if ecl_pointer == global_pointer else
                    _finite(scalar(global_pointer, '<f')))
    default_timer_scale = _native_scale(global_scale, global_pointer, local,
                                        global_pointer, timer=True)
    dimensions = read(data_address + 0x118, '<fff')
    callback = scalar(data_address + 0x4118, '<I')
    tick_callback = scalar(data_address + 0x410c, '<I')
    result = {'schema': SCHEMA, 'image_base': image_base, 'enemy_base': enemy_base, 'data_address': data_address,
        'native_flags': flags, 'rectangle': None if flags is None else bool(flags & 0x1000),
        'data_update_skip_next_tick': None if flags is None else bool(flags & 0x40000),
        'local_slowdown': local, 'custom_contact_callback': callback,
        'special_tick_callback': tick_callback,
        'contact_timer': {'previous_integer': None if timer is None else timer[0],
            'integer': timer_int, 'value': timer_float, 'scale_pointer': timer_pointer,
            'raw_scale': raw_scale, 'effective_scale': timer_scale,
            'default_effective_scale': default_timer_scale,
            'flags': None if timer is None else timer[4]},
        'ecl_timescale': {'pointer': ecl_pointer, 'raw_scale': ecl_raw_scale,
                         'effective_scale': ecl_scale},
        'hitbox': {'width': None if dimensions is None else _finite(dimensions[0]),
            'height': None if dimensions is None else _finite(dimensions[1]),
            'rotation': None if dimensions is None else _finite(dimensions[2])},
        'ecl_threads': [], 'ecl_threads_complete': True, 'ecl_events': [],
        'ecl_forecast_horizon': horizon,
        'read_errors': missing}
    # Production observations pass the motion state from the same checkpoint.
    # Its ECL interpreter resolves stacks, branches and calls once, so a flag
    # transition cannot disagree with the accompanying movement trajectory.
    if motion_state is None:
        try:
            from .native_enemy_motion import read_enemy_motion
            motion_state = read_enemy_motion(memory, data_address, image_base=image_base)
        except (OSError, KeyError, ValueError, struct.error):
            # Archived contact-only fixtures lack the movement/context stack;
            # keep their literal reader, which explicitly stops at branches.
            # Live full-state readers all use the shared interpreter below.
            motion_state = None
    shared_ecl = (motion_state or {}).get('ecl_forecast')
    node = 0 if shared_ecl is not None else enemy_base + 0x11d0
    seen = set()
    while node and len(seen) < MAX_THREADS:
        if node in seen:
            result['ecl_threads_complete'] = False
            break
        seen.add(node)
        item = read(node, '<III')
        if item is None:
            result['ecl_threads_complete'] = False
            break
        context, next_node, previous_node = item
        if context:
            thread = _read_thread(read, context, ecl_scale, horizon,
                                  len(result['ecl_threads']), image_base,
                                  skip_first_tick=bool(flags is not None and flags & 0x40000),
                                  capture_spawns=capture_spawns)
            result['ecl_threads'].append(thread)
            result['ecl_events'].extend(thread['events'])
        node = next_node
    if node:
        result['ecl_threads_complete'] = False
    if shared_ecl is not None:
        from .native_enemy_ecl import CONTACT_OPS
        result['ecl_revision'] = shared_ecl['revision']
        result['tick_callback_effects_revision'] = shared_ecl.get('tick_callback_effects_revision')
        result['ecl_forecast_horizon'] = shared_ecl['horizon']
        result['ecl_threads_complete'] = not motion_state.get('ecl_context_list_truncated', False)
        result['ecl_threads'] = [dict(thread) for thread in shared_ecl['threads']]
        if not capture_spawns:
            for thread in result['ecl_threads']:
                thread['spawn_instructions'] = []
        result['ecl_events'] = [event for event in shared_ecl['events'] if event['opcode'] in CONTACT_OPS]
        # A capture/clock error can precede creation of the per-thread records.
        unknown = shared_ecl.get('contact_unknown_from_tick', shared_ecl.get('unknown_from_tick'))
        if unknown is not None and not any(t.get('unknown_from_tick') == unknown
                                          for t in result['ecl_threads']):
            result['ecl_threads'].append(dict(unknown_from_tick=unknown,
                unmodeled=[shared_ecl.get('contact_unknown_reason') or shared_ecl.get('unknown_reason', 'unknown_shared_ecl')], events=[]))
    result['ecl_events'].sort(key=lambda event: (event['tick'], event['thread_order'],
                                               event['instruction_order']))
    needs_animation = (flags is not None and bool(flags & 0x1000)) or any(
        event['opcode'] == 563 and event['arguments'][0] & 1
        for event in result['ecl_events'])
    result['animation'] = (_read_animation(read, scalar(data_address + 0x124, '<I'), image_base)
                           if needs_animation else None)
    animation_angle = (result['animation'] or {}).get('rotation_z')
    result['rectangle_geometry'] = (rectangle_contact_geometry(
        result['hitbox']['width'], result['hitbox']['height'],
        result['hitbox']['rotation'], animation_angle)
        if flags is not None and flags & 0x1000 else None)
    result['circle_radius'] = (None if dimensions is None else
                              _finite(_f32(dimensions[0] * .5)))
    result['forecast'] = predict_contact_ticks(result, horizon=horizon)
    return result


def predict_contact_ticks(contact, horizon=24):
    """Forecast next ticks 1..N: ECL, tick callback, contact, timer decrement.

    ``native_gate_open`` describes the verified native gates. ``contactable``
    also requires the default callback path; null means unsupported/unknown.
    Neither a true nor a false gate forecasts motion or a player's protection.
    """
    if not isinstance(horizon, int) or not 1 <= horizon <= 240:
        raise ValueError('Contact forecast horizon must be an integer in 1..240')
    flags = contact.get('native_flags')
    timer = contact.get('contact_timer', {})
    integer, value, scale = timer.get('integer'), timer.get('value'), timer.get('effective_scale')
    timer_flags = timer.get('flags')
    callback = contact.get('custom_contact_callback')
    tick_callback = contact.get('special_tick_callback')
    hitbox = dict(contact.get('hitbox', {}))
    persistent = []
    if not contact.get('ecl_threads_complete', False):
        persistent.append('incomplete_ecl_thread_list')
    threads = contact.get('ecl_threads', [])
    events = contact.get('ecl_events')
    if events is None:
        events = [event for thread in threads for event in thread.get('events', [])]
    events = sorted(events, key=lambda event: (event['tick'], event.get('thread_order', 0),
                                              event.get('instruction_order', 0)))
    by_tick = {}
    for event in events:
        by_tick.setdefault(event['tick'], []).append(event)
    ticks = []
    for tick in range(1, horizon + 1):
        skip_update = tick == 1 and flags is not None and bool(flags & 0x40000)
        if tick > contact.get('ecl_forecast_horizon', horizon):
            persistent.append('outside_read_ecl_horizon')
        for thread in threads:
            if thread.get('unknown_from_tick') == tick:
                persistent.extend(thread.get('unmodeled', ['unknown_ecl_thread']))
        applied = by_tick.get(tick, [])
        for event in applied:
            opcode, args = event['opcode'], event['arguments']
            if opcode in (502, 503) and flags is not None:
                argument = int(args[0]) & 0xffffffff
                low = flags & 0xffffffff
                low = (low | argument) if opcode == 502 else (low & ~argument)
                flags = (flags & ~0xffffffff) | low
            elif opcode == 541:
                integer = int(args[0])
                value = _f32(float(integer))
                # 408b00 sets the global scale pointer only on first Timer.set.
                if timer_flags is None:
                    scale = None
                elif not timer_flags & 1:
                    timer_flags |= 1
                    scale = timer.get('default_effective_scale')
            elif opcode == 501:
                hitbox.update(width=args[0], height=args[1])
            elif opcode == 563 and flags is not None:
                flags = (flags & ~0x1000) | ((int(args[0]) << 12) & 0x1000)
            elif opcode == 564:
                hitbox['rotation'] = args[0]
            elif opcode == 634:
                callback = event.get('callback')
            elif opcode in (632, 639):
                tick_callback = event.get('callback')
        from .native_enemy_ecl import variable_only_tick_callback, CALLBACK_EFFECTS_REVISION
        variable_only=(contact.get('tick_callback_effects_revision')==CALLBACK_EFFECTS_REVISION
                       and variable_only_tick_callback(tick_callback,contact.get('image_base',0x400000)))
        if not skip_update and not variable_only and (tick_callback is None or tick_callback != 0):
            persistent.append('unknown_tick_callback' if tick_callback is None
                              else 'unmodeled_tick_callback')
        unmodeled = list(dict.fromkeys(persistent))
        if flags is None:
            unmodeled.append('unknown_native_flags')
        if integer is None:
            unmodeled.append('unknown_contact_timer')
        gate = (False if skip_update else None if unmodeled else
                not bool(flags & CONTACT_MASK) and integer <= 0)
        contactable = gate
        if gate and callback != 0:
            contactable = None
            unmodeled.append('unknown_contact_callback' if callback is None
                              else 'unmodeled_contact_callback')
        rectangular = None if flags is None else bool(flags & 0x1000)
        rectangle = (rectangle_contact_geometry(hitbox.get('width'), hitbox.get('height'),
                     hitbox.get('rotation'), (contact.get('animation') or {}).get('rotation_z'))
                     if rectangular else None)
        shape_known = (bool(rectangle and rectangle['known']) if rectangular
                       else _finite(hitbox.get('width')) is not None)
        ticks.append({'tick': tick, 'native_flags': flags,
            'contact_timer_integer': integer, 'contact_timer_float': value,
            'data_update_skipped': skip_update,
            'native_gate_open': gate, 'contactable': contactable,
            'known': contactable is not None, 'geometry_known': shape_known,
            'geometry_forecast_complete': not bool(rectangular),
            'rectangle': rectangular, 'rectangle_geometry': rectangle,
            'hitbox': dict(hitbox), 'custom_contact_callback': callback,
            'special_tick_callback': tick_callback,
            'unmodeled': unmodeled, 'events': applied})
        if gate and callback != 0:
            # The native callback is arbitrary code; its effects beyond the
            # current contact check are unknown until the next measured frame.
            persistent.append('executed_unmodeled_contact_callback')
        # Native 424a12 decrements only if the current integer is positive.
        if skip_update:
            flags &= ~0x40000
        elif integer is not None and integer > 0:
            if value is None or scale is None:
                integer = value = None
            else:
                value = _f32(value - scale)
                integer = math.trunc(value)
    return {'schema': SCHEMA, 'horizon': horizon, 'ticks': ticks,
        'unmodeled_control_flow': any(any('control_flow' in reason for reason in row['unmodeled'])
                                     for row in ticks),
        'first_unknown_tick': next((row['tick'] for row in ticks if not row['known']), None),
        'next_contact_tick': next((row['tick'] for row in ticks if row['contactable'] is True), None),
        'scope': 'contact availability only; no motion or collision/protection forecast'}
