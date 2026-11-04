"""Bound the zero-speed HOLD/WAIT programs seen in TH14 stage three.

418140 resumes the 44-byte EX records. 0x2 changes the animation/state,
0x800 plays a positional sound, and 0x100000 changes ANM flags. None changes
the physical circle or the zero stored velocity. 419050 starts a WAIT;
419970 adds EX8's speed delta before constructing the polar velocity.
Animation delays may postpone that motion, so this is a displacement
envelope, not a claimed exact trajectory. Unparsed commands end coverage.
"""
from __future__ import annotations

import math
import struct

_NEUTRAL_OR_WAIT = {2, 0x800, 0x100000, 0x80000000}


def read_pending_metadata(memory, address, index):
    if not 0 <= index < 18:
        return {}
    program = memory.read(address + 0xc28 + index * 44, (18 - index) * 44)
    # Keep through the first substantive/unknown command, not the unused
    # rest of the allocation. An EX8 duration bounds the unread tail.
    end = 0
    for end in range(44, len(program) + 1, 44):
        if struct.unpack_from('<I', program, end - 12)[0] not in _NEUTRAL_OR_WAIT:
            break
    rate = struct.unpack('<f', memory.read(memory.base + 0xd8f58, 4))[0]
    if not math.isfinite(rate):
        return {}
    return dict(native_ex_program_start_index=index,
        native_ex_program_prefix_hex=program[:end].hex(), native_motion_time_rate=rate)


def predict_pending_zero_speed(raw, horizon):
    """Return a conservative circle-motion prefix, or defer to the old reader."""
    flags = raw.get('native_ex_active_flags')
    if (raw.get('native_state') != 1 or flags not in (0x04000000, 0x80000000)
            or not raw.get('native_flags', 0) & 0x10
            or raw.get('native_velocity') != [0., 0.] or raw.get('native_speed') != 0.
            or raw.get('native_motion_time_rate') != 1.):
        return None
    hold = flags == 0x04000000
    if hold and (not raw.get('native_ex_hold_forecast_enabled') or not raw.get('native_flags', 0) & 0x200):
        return None
    if not hold and raw.get('native_flags', 0) & 0x200:
        return None
    stem = 'native_ex_hold_' if hold else 'native_ex_wait_'
    remaining, clock, rate = (raw.get(stem + field) for field in ('remaining', 'time', 'rate'))
    if (type(remaining) is not int or remaining < 0 or clock is None
            or not math.isfinite(clock) or abs(clock - remaining) > 1e-5 or rate != 1.):
        return None
    index = raw.get('native_ex_index')
    if raw.get('native_ex_program_start_index') != index:
        return None
    try:
        program = bytes.fromhex(raw.get('native_ex_program_prefix_hex', ''))
        if (not program or len(program) % 44 or not isinstance(index, int)
                or not 0 <= index < 18 or len(program) > (18 - index) * 44
                or program[:44].hex() != raw.get('native_ex_current_record_hex')):
            return None
    except (ValueError, TypeError):
        return None
    position = raw.get('native_position')
    if not position or len(position) != 2 or not all(math.isfinite(x) for x in position):
        return None
    # Preserve the initial timer's already verified stationary prefix.
    # A newly set WAIT below may count its setup tick; ANM delays can only
    # postpone motion beyond this earliest-onset bound.
    delay = remaining
    limit, delta, polar_start, parsed = delay, 0., None, []
    for offset in range(0, len(program), 44):
        record = program[offset:offset + 44]
        opcode, condition = struct.unpack_from('<II', record, 32)
        parameter = struct.unpack_from('<i', record, 16)[0]
        if condition:  # A concurrent instruction need not wait for the timer.
            return None
        parsed.append(opcode)
        if opcode == 0:
            limit = horizon
            break
        if opcode == 2 and parameter == 1:
            continue
        if opcode == 0x800 or (opcode == 0x100000 and parameter in (0, 1)):
            continue
        if opcode == 0x80000000 and parameter > 0:
            delay += max(0, parameter - 1)
            limit = delay
            continue
        if opcode == 8 and parameter > 0:
            ds, da = struct.unpack_from('<2f', record)
            angle = raw.get('native_angle')
            if (angle is None or not all(math.isfinite(v) for v in (ds, da, angle))
                    or abs(da) > 2 * math.pi or abs(angle) > math.pi + 1e-5):
                return None
            delta, polar_start = abs(ds), delay
            limit = delay + max(0, parameter - 1)
            break
        break
    # Do not replace the already verified initial wait/hold prefix unless
    # reading the following program really extends it.
    if limit <= remaining:
        return None
    radii = []
    for frame in range(horizon + 1):
        steps = max(0, frame - polar_start) if polar_start is not None else 0
        radius = delta * steps * (steps + 1) * .5
        if radius:
            radius += 4 * 2**-23 * (max(abs(x) for x in position) + radius + 1) * (frame + 1)
        radii.append(radius)
    if not all(math.isfinite(r) for r in radii):
        return None
    return dict(mode='native_pending_zero_speed_motion_envelope',
        reason='bounded_pending_program' if limit >= horizon else 'unparsed_command_after_bounded_prefix',
        positions=[list(position) for _ in range(horizon + 1)],
        bounds=[[list(position), list(position)] for _ in range(horizon + 1)],
        extra_radii=radii, known_by_frame=[frame <= limit for frame in range(horizon + 1)],
        direction=[1., 0.], parsed_pending_opcodes=parsed, earliest_polar_motion_after=polar_start,
        motion_envelope_not_exact_trajectory=True)
