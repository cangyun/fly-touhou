"""TH14 1.00b enemy motion, decoded from the original x86 executable.

The stored final velocity is *before* the movement-limit clamp (423750), so
it must not be treated as the next displacement.  This module runs the native
absolute/relative motion and interpolators before composing and clamping.

Addresses in comments are preferred-image addresses, not process addresses.
Near-future ECL stack/control flow is read by native_enemy_ecl and its resolved
movement/contact events are shared with the contact reader. Remaining coverage
boundaries concern uncaptured external effects, not ordinary pushes or calls.
"""
from __future__ import annotations

import copy
import math
import struct


SCHEMA = "th14-native-enemy-motion-v1"
GAME_SPEED_RVA = 0xD8F58
MOVE_LIMIT_FLAG = 0x20000
ALREADY_UPDATED_FLAG = 0x40000
PLAYER_RELATIVE_FLAG = 0x4000000
SUPPORTED_MOTION_MODES = (0, 1, 2)
SUPPORTED_INTERPOLATION_MODES = tuple(range(18))


def _f32(value):
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def _add(a, b):
    return _f32(a + b)


def _mul(a, b):
    return _f32(a * b)


def _sub(a, b):
    return _f32(a - b)


def _vec(raw, offset, size=3):
    return list(struct.unpack_from("<" + "f" * size, raw, offset))


def _i32(raw, offset):
    return struct.unpack_from("<i", raw, offset)[0]


def _u32(raw, offset):
    return struct.unpack_from("<I", raw, offset)[0]


def _motion(raw, offset):
    fields = struct.unpack_from("<16fI", raw, offset)
    return {
        "position": list(fields[:3]), "secondary_position": list(fields[3:6]),
        "speed": fields[6], "angle": fields[7], "radius": fields[8],
        "radial_velocity": fields[9], "rotation": fields[10],
        "ellipse_scale": fields[11], "phase": fields[12],
        "velocity": list(fields[13:16]), "flags": fields[16],
        "mode": fields[16] & 0xF,
    }


def _interpolator(memory, raw, offset, dimension, speed_address, effective_speed):
    # Scalar: start/end/tangents/current. Vec3: current/start/end/tangents.
    # Vec2 has the scalar ordering. Timer is always 5 * dimension floats in.
    width = dimension * 4
    order = ("current", "start", "end", "control1", "control2") if dimension == 3 else (
        "start", "end", "control1", "control2", "current")
    result = {name: _vec(raw, offset + i * width, dimension) for i, name in enumerate(order)}
    timer = offset + 5 * width
    pointer = _u32(raw, timer + 12)
    if pointer == speed_address:
        timer_scale = effective_speed
    elif pointer:
        timer_scale = struct.unpack("<f", memory.read(pointer, 4))[0]
    else:
        timer_scale = None
    result.update(
        dimension=dimension, previous_timer=_i32(raw, timer),
        timer=_i32(raw, timer + 4), elapsed=_vec(raw, timer + 8, 1)[0],
        scale_pointer=pointer, timer_scale=timer_scale,
        timer_flags=_u32(raw, timer + 16), duration=_i32(raw, timer + 20),
    )
    if dimension == 3:
        result.update(axis_modes=[_i32(raw, timer + 24 + j * 4) for j in range(3)],
                      mode=_i32(raw, timer + 36), flags=_u32(raw, timer + 40))
    else:
        result.update(mode=_i32(raw, timer + 24), flags=0)
    return result


def _read_ecl_context(memory, address):
    raw = memory.read(address, 8)
    instruction = _u32(raw, 4)
    result = {"address": address, "time": _vec(raw, 0, 1)[0], "instruction": instruction}
    if instruction:
        header = memory.read(instruction, 16)
        result.update(instruction_time=_i32(header, 0),
                      opcode=struct.unpack_from("<h", header, 4)[0],
                      instruction_size=struct.unpack_from("<H", header, 6)[0],
                      parameter_mask=struct.unpack_from("<H", header, 8)[0],
                      difficulty_mask=header[10])
    return result


def read_enemy_motion(memory, data_address, *, image_base=None, read_ecl=True, max_ecl_contexts=64,
                      forecast_ecl=True, forecast_damage_bound=False):
    """Read native motion at the already-owned, paused game checkpoint.

    ``memory`` only needs ``read(address, size) -> bytes``. ``data_address`` is
    EnemyBase + 0x11f0, the same address used by the existing enemy reader.
    The caller owns snapshot coherence; this function neither resumes nor writes.
    """
    base = int(image_base if image_base is not None else getattr(memory, "base", 0x400000))
    address = int(data_address)
    raw = memory.read(address, 0x510)
    speed_address = base + GAME_SPEED_RVA
    global_speed = struct.unpack("<f", memory.read(speed_address, 4))[0]
    slowdown = _vec(raw, 0x2E4, 1)[0]  # EnemyBase+14d4; wrapper 424710.
    effective_speed = global_speed
    if slowdown > 0:
        effective_speed = min(1.0, max(0.0, _sub(global_speed, _mul(global_speed, slowdown))))
    flags = struct.unpack("<I", memory.read(address + 0x4054, 4))[0]
    bounds = struct.unpack("<4f", memory.read(address + 0x3F60, 16))
    owner = struct.unpack("<I", memory.read(address + 0x40EC, 4))[0]
    callback = struct.unpack("<I", memory.read(address + 0x410C, 4))[0]
    health_raw = memory.read(address + 0x3F74, 0x1C)
    interrupt_raw = memory.read(address + 0x406C, 0x80)
    interrupts = []
    for index in range(8):
        hp, timer, hp_sub, time_sub = struct.unpack_from("<4i", interrupt_raw, index * 0x10)
        interrupts.append(dict(index=index, hp=hp, time=timer, hp_sub=hp_sub, time_sub=time_sub))
    clock_pointer = _u32(raw, 0x2C8)
    ecl_dt = effective_speed if clock_pointer == speed_address else (
        struct.unpack("<f", memory.read(clock_pointer, 4))[0] if clock_pointer else None)
    interpolation = {}
    for name, offset, dimension in (
        ("absolute_position", 0x2F8, 3), ("relative_position", 0x360, 3),
        ("absolute_angle", 0x3C8, 1), ("absolute_speed", 0x3F8, 1),
        ("relative_angle", 0x428, 1), ("relative_speed", 0x458, 1),
        ("absolute_parameters", 0x488, 2), ("relative_parameters", 0x4CC, 2),
    ):
        interpolation[name] = _interpolator(memory, raw, offset, dimension, speed_address, effective_speed)
    state = {
        "schema": SCHEMA, "data_address": address, "image_base": base,
        "previous": _motion(raw, 0), "final": _motion(raw, 0x44),
        "absolute": _motion(raw, 0x88), "relative": _motion(raw, 0xCC),
        "flags": flags, "global_game_speed": global_speed,
        "local_slowdown": slowdown, "effective_game_speed": effective_speed,
        "movement_limit": {"enabled": bool(flags & MOVE_LIMIT_FLAG),
                           "center": list(bounds[:2]), "width": bounds[2], "height": bounds[3]},
        "interpolators": interpolation, "tick_callback": callback,
        "ecl_clock_pointer": clock_pointer, "ecl_time_step": ecl_dt,
        "owner_address": owner, "ecl_contexts": [], "ecl_read": bool(read_ecl),
        "health": {"hp": _i32(health_raw, 0), "hp_max": _i32(health_raw, 4),
                   "damage_accumulator": _i32(health_raw, 0x14),
                   "flags": _u32(health_raw, 0x18)},
        "interrupts": interrupts,
        "phase_timer": {"previous_timer": _i32(raw, 0x2BC),
                        "timer": _i32(raw, 0x2C0), "elapsed": _vec(raw, 0x2C4, 1)[0],
                        "scale_pointer": clock_pointer,
                        "timer_scale": ecl_dt if clock_pointer else None,
                        "flags": _u32(raw, 0x2CC)},
    }
    if forecast_damage_bound:
        from .native_enemy_damage import read_damage_bound
        state['damage_bound_enabled'] = True
        state['damage_bound'] = read_damage_bound(memory, address, base, flags)
    if read_ecl and owner:
        # 483e70 walks the native context list including the main context.
        node = owner + 0x11D0
        seen = set()
        for _ in range(max_ecl_contexts):
            if not node or node in seen:
                break
            seen.add(node)
            item = memory.read(node, 8)
            context, node = struct.unpack("<II", item)
            if context:
                state["ecl_contexts"].append(_read_ecl_context(memory, context))
        state["ecl_context_list_truncated"] = bool(node and node not in seen)
        if forecast_ecl:
            from .native_enemy_ecl import forecast_ecl as resolve_ecl
            state['ecl_forecast'] = resolve_ecl(memory, state)
    return state


def interpolation_factor(mode, elapsed, duration):
    """Native 4041a0 modes 0..17; special 7/8/17 handled by the interpolator."""
    if not duration:
        return 1.0
    t = _f32(elapsed / duration)
    if mode == 0 or mode in (7, 8, 17):
        return t
    if mode in (1, 2, 3):
        power = mode + 1
        result = t
        for _ in range(power - 1):
            result = _mul(result, t)
        return result
    if mode in (4, 5, 6):
        return _sub(1.0, interpolation_factor(mode - 3, _sub(1.0, t), 1))
    if mode in (9, 10, 11):
        doubled = _mul(t, 2.0)
        if doubled < 1:
            return _mul(0.5, interpolation_factor(mode - 8, doubled, 1))
        return _mul(0.5, _sub(2.0, interpolation_factor(mode - 8, _sub(2.0, doubled), 1)))
    if mode in (12, 13, 14):
        doubled = _mul(t, 2.0)
        if doubled < 1:
            return _mul(0.5, _sub(1.0, interpolation_factor(mode - 11, _sub(1.0, doubled), 1)))
        return _add(0.5, _mul(0.5, interpolation_factor(mode - 11, _sub(doubled, 1.0), 1)))
    if mode == 15:
        return 0.0
    if mode == 16:
        return 1.0
    raise NotImplementedError(f"native interpolation mode {mode}")


def _hermite(start, end, c1, c2, t):
    twice = _mul(t, 2.0)
    h00 = _mul(_add(twice, 1), _mul(_sub(t, 1), _sub(t, 1)))
    h01 = _mul(_sub(3, twice), _mul(t, t))
    h10 = _mul(_mul(_sub(1, t), _sub(1, t)), t)
    h11 = _mul(_mul(_sub(t, 1), t), t)
    return _add(_add(_add(_mul(end, h01), _mul(start, h00)), _mul(c1, h10)), _mul(c2, h11))


def _advance_interpolator(interp):
    # 414140 / 42cd40 / 42d370: timer update, endpoint, then mode.
    duration = interp["duration"]
    if duration <= 0:
        return interp["current"][:]
    interp["previous_timer"] = interp["timer"]
    scale = interp["timer_scale"]
    if scale is None or _f32(0.99) < scale < _f32(1.01):
        interp["timer"] += 1
        interp["elapsed"] = _add(interp["elapsed"], 1)
    else:
        interp["elapsed"] = _add(interp["elapsed"], scale)
        interp["timer"] = math.trunc(interp["elapsed"])
    if interp["timer"] >= duration:
        interp.update(timer=duration, previous_timer=duration - 1, elapsed=float(duration), duration=0)
        return interp["start"][:] if interp["mode"] in (7, 17) else interp["end"][:]
    modes = interp.get("axis_modes") if interp.get("flags", 0) & 1 else None
    result = []
    for axis in range(interp["dimension"]):
        mode = modes[axis] if modes is not None else interp["mode"]
        if mode == 7:
            interp["start"][axis] = _add(interp["start"][axis], interp["end"][axis])
            value = interp["start"][axis]
        elif mode == 17:
            interp["start"][axis] = _add(interp["start"][axis], interp["control2"][axis])
            interp["control2"][axis] = _add(interp["end"][axis], interp["control2"][axis])
            value = interp["start"][axis]
        elif mode == 8:
            value = _hermite(interp["start"][axis], interp["end"][axis],
                             interp["control1"][axis], interp["control2"][axis],
                             _f32(interp["elapsed"] / duration))
        else:
            factor = interpolation_factor(mode, interp["elapsed"], duration)
            value = _add(interp["start"][axis], _mul(_sub(interp["end"][axis], interp["start"][axis]), factor))
        result.append(value)
    interp["current"] = result
    return result[:]


def _wrap_angle(angle):
    pi, twice = _f32(math.pi), _f32(2 * math.pi)
    # Native functions cap their normalization loop at 34 iterations.
    for _ in range(34):
        if angle <= pi:
            break
        angle = _sub(angle, twice)
    for _ in range(34):
        if angle >= -pi:
            break
        angle = _add(angle, twice)
    return angle


def _quantize(value):
    # 403df0 -> 4a8560 is floor, including negative coordinates.
    return _f32(_f32(math.floor(_mul(value, 100.0))) / 100.0)


def _calculate_velocity(motion, dt):
    mode = motion["mode"]
    if mode == 0:
        speed = _mul(motion["speed"], dt)
        motion["velocity"][:2] = [_f32(math.cos(motion["angle"]) * speed),
                                   _f32(math.sin(motion["angle"]) * speed)]
    elif mode == 2:
        motion["radius"] = _add(motion["radius"], _mul(motion["radial_velocity"], dt))
        motion["angle"] = _wrap_angle(_add(motion["angle"], _mul(motion["speed"], dt)))
    elif mode != 1:
        raise NotImplementedError(f"native motion mode {mode}")


def _apply_motion(motion):
    mode = motion["mode"]
    if mode == 0:
        motion["position"] = [_add(p, v) for p, v in zip(motion["position"], motion["velocity"])]
    elif mode == 2:
        radius, angle = motion["radius"], motion["angle"]
        motion["position"] = [_add(motion["velocity"][0], _f32(math.cos(angle) * radius)),
                              _add(motion["velocity"][1], _f32(math.sin(angle) * radius)),
                              motion["velocity"][2]]
    elif mode != 1:
        raise NotImplementedError(f"native motion mode {mode}")
    motion["position"][:2] = [_quantize(p) for p in motion["position"][:2]]


def _step_motion(state):
    # A newly spawned enemy may already have run its Data tick. 424810 skips
    # that entire next manager visit; 4229ac then clears the marker. Position,
    # interpolators, ECL, callbacks and contact timers all remain unchanged.
    if state["flags"] & ALREADY_UPDATED_FLAG:
        state["flags"] &= ~ALREADY_UPDATED_FLAG
        return [0.0, 0.0, 0.0]
    if state["flags"] & PLAYER_RELATIVE_FLAG:
        raise NotImplementedError("player-relative native motion needs the candidate player path")
    dt = state["effective_game_speed"]
    interps = state["interpolators"]
    for name in ("absolute", "relative"):
        motion = state[name]
        if motion["mode"] not in SUPPORTED_MOTION_MODES:
            raise NotImplementedError(f"{name} native motion mode {motion['mode']}")
        angle = interps[name + "_angle"]
        if angle["duration"] > 0 and motion["mode"] not in (2, 3):
            motion["angle"] = _wrap_angle(_wrap_angle(_advance_interpolator(angle)[0]))
        speed = interps[name + "_speed"]
        if speed["duration"] > 0:
            motion["speed"] = _advance_interpolator(speed)[0]
        params = interps[name + "_parameters"]
        if params["duration"] > 0:
            motion["radius"], motion["radial_velocity"] = _advance_interpolator(params)
    # Native updates both deltas before applying either motion.
    for name in ("absolute", "relative"):
        motion, interp = state[name], interps[name + "_position"]
        if interp["duration"] > 0:
            target = _advance_interpolator(interp)
            motion["velocity"] = [_sub(p, old) for p, old in zip(target, motion["position"])]
        else:
            _calculate_velocity(motion, dt)
    _apply_motion(state["absolute"])
    _apply_motion(state["relative"])
    final = state["final"]
    old = final["position"][:]
    final["velocity"] = [_sub(_add(a, r), p) for a, r, p in zip(
        state["absolute"]["position"], state["relative"]["position"], old)]
    _apply_motion(final)
    limit = state["movement_limit"]
    if limit["enabled"]:
        for axis, size in enumerate((limit["width"], limit["height"])):
            half = _mul(size, 0.5)
            lower, upper = _sub(limit["center"][axis], half), _add(limit["center"][axis], half)
            if final["position"][axis] < lower:
                final["position"][axis] = lower
            elif final["position"][axis] > upper:
                final["position"][axis] = upper
        state["absolute"]["position"] = [_sub(p, r) for p, r in zip(
            final["position"], state["relative"]["position"])]
    # Deliberately preserve the pre-clamp stored velocity, exactly as 423750.
    return [_sub(p, previous) for p, previous in zip(final["position"], old)]


def _script_boundary(state, horizon):
    first_tick = 2 if state["flags"] & ALREADY_UPDATED_FLAG else 1
    if first_tick > horizon:
        return None, None
    if state.get("tick_callback"):
        from .native_enemy_ecl import variable_only_tick_callback, CALLBACK_EFFECTS_REVISION
        shared=state.get('ecl_forecast') or {}
        resolved=(shared.get('tick_callback_effects_revision')==CALLBACK_EFFECTS_REVISION
                  and variable_only_tick_callback(state['tick_callback'],state['image_base']))
        if not resolved:return first_tick, "native tick callback is not interpreted"
    if not state.get("ecl_read", False):
        return first_tick, "ECL contexts were not read"
    if state.get("ecl_context_list_truncated"):
        return first_tick, "ECL context list was truncated"
    forecast = state.get('ecl_forecast')
    if forecast is not None:
        due = forecast.get('unknown_from_tick')
        if due is not None and due <= horizon:
            return due, forecast.get('unknown_reason')
        if horizon > forecast.get('horizon', 0):
            return forecast.get('horizon', 0) + 1, 'outside resolved ECL horizon'
        return None, None
    dt = state.get("ecl_time_step")
    first, reason = None, None
    for context in state.get("ecl_contexts", ()):
        if not context.get("instruction"):
            continue
        delta = context["instruction_time"] - context["time"]
        if delta <= 0:
            due = first_tick
        elif dt is None:
            return first_tick, "ECL clock is unknown"
        elif dt <= 0:
            continue
        else:
            time, due = context["time"], first_tick
            while time < context["instruction_time"] and due <= horizon:
                time, due = _add(time, dt), due + 1
        if due <= horizon and (first is None or due < first):
            first, reason = due, f"uninterpreted ECL opcode {context['opcode']} at {context['instruction']:#x}"
    return first, reason


def _phase_boundary(state, horizon):
    """42bf70: HP interruptions and independently timed phase interruptions.

    Legacy snapshots retain an unknown bound. Opted-in snapshots include the
    original shot/Bomb cap and exceptional enemy damage sources; the first
    possibly reachable HP threshold is computed from those measured values.
    """
    first_tick = 2 if state["flags"] & ALREADY_UPDATED_FLAG else 1
    details = {"hp_interrupt": None, "time_interrupt": None,
               "damage_per_frame_bound": None,
               "damage_bound_source": "native damage cap has not been established"}
    bound = state.get('damage_bound') if state.get('damage_bound_enabled') else None
    if bound:
        details.update(damage_per_frame_bound=bound.get('per_frame'),
                       initial_extra_damage_bound=bound.get('initial_extra', 0),
                       damage_bound_source=bound['source'], damage_bound=bound)
    if first_tick > horizon:
        return None, None, details
    if "interrupts" not in state or "health" not in state or "phase_timer" not in state:
        return first_tick, "HP/time interruption state was not captured", details
    rows = state["interrupts"]
    # Native first pass stops at the first active HP entry, not the smallest HP.
    hp_row = next((r for r in rows if r["hp"] >= 0), None)
    hp_due, hp_reason = None, None
    if hp_row is not None and hp_row["hp_sub"]:
        margin = state["health"]["hp"] - hp_row["hp"]
        details["hp_interrupt"] = {**hp_row, "current_hp": state["health"]["hp"],
                                   "hp_margin": margin, "threshold_reached": margin <= 0}
        hp_due = first_tick
        hp_reason = ("native HP interruption threshold already reached" if margin <= 0 else
                     "armed HP interruption can be triggered by future shot damage; damage cap unknown")
        if margin > 0 and bound and bound.get('per_frame') is not None:
            cap, extra = bound['per_frame'], bound.get('initial_extra', 0)
            ticks = (max(1, math.ceil((margin - extra) / cap)) if cap > 0
                     else 1 if extra >= margin else None)
            hp_due = first_tick + ticks - 1 if ticks is not None else None
            details['hp_interrupt']['earliest_damage_bound_frame'] = hp_due
            if hp_due is not None and hp_due > horizon:
                hp_due = None
            hp_reason = 'native HP threshold first reachable under captured shot/Bomb damage cap'
    # Native second pass stops at the first active positive timeout.
    timed = next((r for r in rows if r["hp"] >= 0 and r["time"] > 0), None)
    time_due = None
    if timed is not None and timed["time_sub"]:
        timer = copy.deepcopy(state["phase_timer"])
        due = first_tick
        while timer["timer"] < timed["time"] and due <= horizon:
            scale = timer["timer_scale"]
            if scale is None or _f32(0.99) < scale < _f32(1.01):
                timer["timer"] += 1
                timer["elapsed"] = _add(timer["elapsed"], 1)
            else:
                timer["elapsed"] = _add(timer["elapsed"], scale)
                timer["timer"] = math.trunc(timer["elapsed"])
            due += 1
        if due <= horizon:
            time_due = due
        details["time_interrupt"] = {**timed, "current_timer": state["phase_timer"]["timer"],
                                     "due_frame": time_due}
    # A due script may replace HP, damage callbacks or interruption entries.
    # The normal ECL motion interpreter still runs, but the cap cannot certify
    # the old phase beyond such a write.
    effect = state.get('ecl_forecast', {}).get('damage_phase_change') if bound else None
    if effect and effect['tick'] <= horizon:
        due = effect['tick']
        if (hp_due is None or due <= hp_due) and (time_due is None or due <= time_due):
            return due, f"ECL opcode {effect['opcode']} may change HP, damage or phase interruption", details
    if hp_due is not None and (time_due is None or hp_due <= time_due):
        return hp_due, hp_reason, details
    if time_due is not None:
        return time_due, "native timed phase interruption changes the ECL program", details
    return None, None, details


def predict_enemy_positions(state, horizon=24):
    """Forecast up to 24 native frames from a captured motion state.

    ``positions[0]`` is the observed final position. Later positions are a
    conditional trajectory from the captured native movement program. A pending
    ECL instruction/callback is reported in ``unknown_from_frame``; positions
    at/after that boundary must not be described as a complete exact forecast.
    An unsupported movement/easing stops the trajectory instead of inventing
    stationary or finite-difference motion. No source state is mutated.
    """
    if not isinstance(horizon, int) or not 0 <= horizon <= 24:
        raise ValueError("native enemy prediction horizon must be an integer in [0, 24]")
    work = copy.deepcopy(state)
    positions = [work["final"]["position"][:]]
    velocities = [work["final"]["velocity"][:]]
    displacements = [[0.0, 0.0, 0.0]]
    boundary, reason = _script_boundary(work, horizon) if horizon else (None, None)
    phase_boundary, phase_reason, phase_details = _phase_boundary(work, horizon)
    if phase_boundary is not None and (boundary is None or phase_boundary < boundary):
        boundary, reason = phase_boundary, phase_reason
    supported = True
    motion_error = None
    events_by_frame = {}
    for event in work.get('ecl_forecast', {}).get('events', ()):
        events_by_frame.setdefault(event['tick'], []).append(event)
    for frame in range(1, horizon + 1):
        try:
            previous = work['final']['position'][:]
            displacement = _step_motion(work)
            for event in events_by_frame.get(frame, ()):
                from .native_enemy_ecl import apply_geometry_event
                apply_geometry_event(work, event)
            displacement = [_sub(p, old) for p, old in zip(work['final']['position'], previous)]
        except NotImplementedError as exc:
            supported, motion_error = False, str(exc)
            if boundary is None or frame < boundary:
                boundary, reason = frame, motion_error
            break
        positions.append(work["final"]["position"][:])
        velocities.append(work["final"]["velocity"][:])
        displacements.append(displacement)
    result = {"schema": SCHEMA, "horizon": horizon, "positions": positions,
            "displacements": displacements, "stored_velocities": velocities,
            "motion_supported": supported, "motion_error": motion_error,
            "unknown_from_frame": boundary, "unknown_reason": reason,
            "phase_unknown_from_frame": phase_boundary, "phase_coverage": phase_details,
            "damage_bound_enabled": bool(work.get('damage_bound_enabled')),
            "ecl_revision": work.get('ecl_forecast', {}).get('revision'),
            "resolved_ecl_events": sum(len(v) for k, v in events_by_frame.items() if k <= horizon),
            "forecast_complete": supported and boundary is None,
            "exact_until_frame": horizon if boundary is None else boundary - 1,
            "update_order": "motion; ECL; optional callback; contact; contact-timer decrement"}
    ecl = work.get('ecl_forecast', {})
    # A distant parent can create a contact body near the player. Keep the
    # known creation times; forecasting the parent alone does not cover it.
    result['pending_enemy_spawns'] = [dict(event) for event in ecl.get('spawn_instructions', ())
                                      if 1 <= event['tick'] <= horizon]
    if ecl.get('random_motion_events'):
        result['reachable_radius_by_frame'] = ecl['reachable_radius_by_frame'][:len(positions)]
        result['reachable_scope'] = ecl['reachable_scope']
        result['random_motion_events'] = ecl['random_motion_events']
    return result
