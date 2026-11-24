"""TH14 player fire, damage and drop forecasts from the paused original snapshot."""

import struct

import numpy as np

from .native_enemy_damage import (
    advance_known_player_shot, enemy_exits_view, ordinary_shot_culled,
    player_shot_overlaps_round_enemy, resolve_player_homing_target,
    read_player_shot_motions, read_player_damage_zones,
)
from .native_enemy_motion import predict_enemy_positions
from .native_pending_lines import forecast_line_emitter
from .native_player_firing import (
    advance_firing_clock, advance_player_options, advance_shot_drop_gauges,
    initial_player_shot, read_player_firing_state, scatter_drop_draws,
)
from .native_shot_rules import ReadCache


class UnsupportedNativeFire(ValueError):
    """The current original firing state is outside the modeled contract."""


def read_shot_render_state(memory, base, identity):
    from .native_enemy_contact import _read_animation

    def read(address, fmt):
        return struct.unpack(fmt, memory.read(address, struct.calcsize(fmt)))

    vm = _read_animation(read, identity, base)["address"]
    if vm is None:
        return None
    raw = memory.read(vm + 0x18, 0x80)
    flags, draw_flags = struct.unpack_from("<2I", raw)
    return dict(flags=flags, draw_flags=draw_flags,
                angle=struct.unpack_from("<f", raw, 0x50 - 0x18)[0],
                translation=list(struct.unpack_from("<3f", raw, 0x3c - 0x18)),
                scale=list(struct.unpack_from("<4f", raw, 0x60 - 0x18)),
                sprite_size=list(struct.unpack_from("<2f", raw, 0x80 - 0x18)),
                pivot=list(struct.unpack_from("<2f", raw, 0x90 - 0x18)),
                extra=list(read(vm + 0x4ec, "<3f")),
                parent=read(vm + 0x594, "<I")[0],
                center=list(read(vm + 0x59c, "<3f")))


def _native_damage_mode(memory):
    def read_flag(pointer, offset):
        address, = struct.unpack('<I', memory.read(memory.base + pointer, 4))
        value, = struct.unpack('<I', memory.read(address + offset, 4))
        return value

    return dict(suspended=bool(read_flag(0xdb550, 0x194)),
                resistance_enabled=read_flag(0xdb52c, 0x40) == 1,
                spell_slowdown=(read_flag(0xdb534, 0x7c) & 0x21) == 0x21)


def _health_damage(raw, target):
    """Native ordinary player mode uses this order after the shot cap."""
    modifiers = target.get('damage_modifiers')
    if modifiers is None:
        return raw
    if modifiers['suspended']:
        return 0
    scale = modifiers['resistance']
    if modifiers['resistance_enabled'] and scale < 1.:
        raw = int(np.float32(np.float32(raw) * np.float32(scale)))
    if modifiers['spell_slowdown']:
        raw //= 30
    return raw


def _apply_health_damage(damage, target):
    old_hp = target['hp']
    meter = target.get('health_conversion')
    if meter is None:
        target['hp'] -= damage
    else:
        # Native421480 caches the public HP after dividing the raw meter by
        # seven relative to an offset, with signed truncation toward zero.
        remaining, offset = meter
        remaining -= damage
        shifted = remaining - 7 * offset
        quotient = shifted // 7 if shifted >= 0 else -((-shifted) // 7)
        target['hp'] = offset + quotient
        target['health_conversion'] = (remaining, offset)
    return old_hp - target['hp']


def conditional_targets(reader, states, horizon, *, include_screen=False, include_drops=False,
                        include_damage_modifiers=False):
    """Read damage geometry and calculate the conditional path of each observed actor."""
    cache = ReadCache(reader.mem)
    instructions = {}
    targets = []
    mode = _native_damage_mode(reader.mem) if include_damage_modifiers else None
    for state in states:
        owner = state['owner_address']
        if not owner:
            continue
        dimensions = list(struct.unpack('<5f', reader.mem.read(state['data_address']+0x110,20)))
        key = reader.mem.integer(owner+0x11e4)
        projection, _, _, _ = forecast_line_emitter(cache,state,horizon,instructions)
        movement = predict_enemy_positions(dict(state,ecl_forecast=projection), horizon=horizon,
                                           include_damage_flags=True)
        # Incoming-damage geometry can change independently of contact geometry.
        damage_size_events = {e['tick']:e['arguments'] for e in projection['events'] if e['opcode'] == 500}
        damage_flags = movement['damage_flags_by_tick']
        damage_dimensions = []
        for tick in range(len(damage_flags)):
            if tick in damage_size_events:
                dimensions = [*damage_size_events[tick], *dimensions[2:]]
            damage_dimensions.append(dimensions)
        targets.append(dict(owner=owner,key=key,flags=state['flags'],dimensions=damage_dimensions[0],
                            hp=state['health']['hp'],position=state['final']['position'],
                            positions=movement['positions'],unknown_from=movement['unknown_from_frame'],
                            damage_flags_by_tick=damage_flags, damage_dimensions_by_tick=damage_dimensions))
        if mode is not None:
            resistance, = struct.unpack('<f', reader.mem.read(state['data_address'] + 0x4034, 4))
            if not np.isfinite(resistance) or resistance < 0.:
                raise UnsupportedNativeFire('Unknown original incoming-damage resistance')
            targets[-1]['damage_modifiers'] = dict(mode, resistance=resistance)
            meter, offset, _, health_flags = struct.unpack(
                '<3iI', reader.mem.read(state['data_address'] + 0x3f80, 16))
            if health_flags & 1:
                targets[-1]['health_conversion'] = (meter, offset)
        if include_drops:
            targets[-1]['drop_counts'] = list(struct.unpack('<12i', reader.mem.read(owner + 0x5184, 48)))
        if include_screen:
            targets[-1]['screen_size'] = list(struct.unpack(
                '<2f', reader.mem.read(state['data_address'] + 0x3f58, 8)))
    return targets


def targets_along_actor_tracks(targets, tracks, horizon):
    """Use path-dependent ECL positions, flags and damage dimensions for firing."""
    projected = []
    for target in targets:
        track = tracks.get(target['owner'])
        if not track or not track['samples']:
            projected.append(target)
            continue
        samples = track['samples'][:horizon + 1]
        positions = [frame['position'] for frame in samples]
        flags = [frame['flags'] for frame in samples]
        dimensions = target['dimensions']
        geometry = []
        resized = {event['tick']: event['arguments'] for event in track['events']
                   if event['opcode'] == 500}
        for index in range(len(samples)):
            if index in resized:
                dimensions = [*resized[index], *dimensions[2:]]
            geometry.append(dimensions)
        projected.append(dict(
            target, positions=positions, damage_flags_by_tick=flags,
            damage_dimensions_by_tick=geometry,
            unknown_from=len(samples) if len(samples) <= horizon else None,
        ))
    return projected


def predict_hits(shots, targets, horizon, *, frame_cap, damage_zones=(), allow_gohei=False,
                 game_speed=1., shot_schedule=None, rng_gauges=None, power_level=None):
    """Conditional damage from a paused state and an optional planned firing schedule."""
    radius = {z['slot']:z['circle_radius'] for z in damage_zones if z['active'] and z['circle_radius'] is not None}
    def supported_gohei(shot):
        return (allow_gohei and shot['tick_callback']==0x51cd0 and shot['hit_callback']==0x52050
                and shot['damage_slot'] in radius and game_speed==1.)
    active = [shot.copy() for shot in shots if shot['hit_callback']==0]
    active += [dict(s,circle_radius=radius[s['damage_slot']]) for s in shots if supported_gohei(s)]
    targets = [target.copy() for target in targets]
    removed = {}
    vanished_at_edge = {}
    rows = []
    next_birth_id = 256
    separate_draws = {}
    for tick in range(1,horizon+1):
        births, held = shot_schedule(tick, targets, active) if shot_schedule else ([], False)
        for shot in births:
            shot["slot"] = next_birth_id
            next_birth_id += 1
        active.extend(births)
        hits = []
        missing_motion = []
        gone = set()
        for shot in active:
            if shot_schedule and 'ordinary_sprite' in shot and ordinary_shot_culled(
                    shot, shot['ordinary_sprite'], shot['viewport']):
                gone.add(shot['slot'])
                continue
            if shot['tick_callback']==0x51cd0:
                if not _advance_gohei(shot, targets, game_speed, held=held):
                    gone.add(shot['slot'])
                continue
            target = resolve_player_homing_target(shot,targets)
            motion = advance_known_player_shot(shot,homing_target=target['position'] if target else None,game_speed=game_speed)
            if motion is None:
                missing_motion.append(shot['slot'])
                continue
            if shot['tick_callback']:
                shot.update(target=target['owner'] if target else 0,target_key=target['key'] if target else 0)
            shot['motion'] = motion
            shot['tick'] += 1
        active = [s for s in active if s['slot'] not in gone]
        alive = []
        for target in targets:
            positions = target['positions']
            if tick<len(positions):
                target['position'] = positions[tick]
            if 'damage_flags_by_tick' in target and tick < len(target['damage_flags_by_tick']):
                target['flags'] = target['damage_flags_by_tick'][tick]
                target['dimensions'] = target['damage_dimensions_by_tick'][tick]
            if shot_schedule and 'screen_size' in target and enemy_exits_view(target):
                vanished_at_edge[target['key']] = tick
                continue
            enemy_hits = [shot for shot in active if shot['slot'] not in missing_motion
                          and player_shot_overlaps_round_enemy(shot,target)]
            effects = [s for s in active if s['tick_callback'] == 0x51cd0 and _gohei_contact(s, target)]
            pulses = [s for s in effects if s['tick'] % 8 == 0]
            if rng_gauges is not None:
                impacts = []
                for shot in enemy_hits + effects:
                    slot = shot.get('damage_slot')
                    rod = shot['tick_callback'] == 0x51cd0
                    impacts.append((shot['slot'] if slot is None else slot, shot['shot_flags'],
                                    rod, rod and shot['tick'] % 8 == 0))
                impacts.sort()
                rng_gauges, draws = advance_shot_drop_gauges(
                    rng_gauges, (row[1:] for row in impacts), power_level)
                if draws:
                    separate_draws.setdefault(tick, {})[target['owner']] = draws
            for shot in pulses:
                shot['motion']['speed'] = 0.
                shot['steering_speed'] = -2. if shot['tick']>=30 else shot['steering_speed']*.5
                shot['recoil_ticks'] = 0
            hit_ids = {shot['slot'] for shot in enemy_hits}
            enemy_hits += pulses
            active = [shot for shot in active if shot['slot'] not in hit_ids]
            raw = sum(shot['damage'] for shot in enemy_hits)
            damage = _health_damage(min(frame_cap, raw), target)
            if damage:
                actual_damage = _apply_health_damage(damage, target)
                hits.append(dict(key=target['key'],damage=actual_damage,slots=[s['slot'] for s in enemy_hits]))
                if target['hp']<=0:
                    removed[target['key']] = tick
                    if rng_gauges is not None:
                        counts = separate_draws.setdefault(tick, {})
                        counts[target['owner']] = counts.get(target['owner'], 0) + scatter_drop_draws(target['drop_counts'])
                    continue
            alive.append(target)
        targets = alive
        row = dict(tick=tick, hits=hits, health={target['key']:target['hp'] for target in targets})
        if shot_schedule:
            row["births"] = [dict(definition=shot['definition'], slot=shot['slot'],
                                  position=shot['motion']['position'], target_key=shot['target_key'])
                             for shot in births]
            if rng_gauges is not None:
                row['drop_gauges'] = rng_gauges
            row["gohei"] = [dict(slot=shot['slot'], tick=shot["tick"], target_key=shot['target_key'],
                                 position=shot['motion']['position'], release_ticks=shot["release_ticks"])
                            for shot in active if shot['tick_callback']==0x51cd0]
        rows.append(row)
    return dict(rows=rows,defeat_ticks=removed,edge_removals=vanished_at_edge,
                gameplay_draws_after_actor=separate_draws,
                unsupported_player_shots=[shot['slot'] for shot in shots if not supported_gohei(shot) and (shot['hit_callback']!=0 or
                                          shot['tick_callback'] not in (0,0x518d0))],
                unsupported_damage_targets=[t['key'] for t in targets if t['flags']&0x1000],
                uncertain_target_motion={t['key']:t['unknown_from'] for t in targets if t['unknown_from'] is not None},
                scope="Conditional on supported native body motion, shot geometry and captured damage modifiers")


def _advance_gohei(shot, targets, game_speed, *, held=False):
    """Advance an active purification-rod shot and its release timer."""
    import math
    from copy import deepcopy
    from .native_enemy_motion import _add, _apply_motion, _calculate_velocity, _f32, _mul, _sub, _wrap_angle

    shot['release_ticks'] = 0 if held else shot['release_ticks'] + 1
    target = resolve_player_homing_target(shot, targets)
    if not held and (not (shot['target'] or target) or shot['release_ticks'] >= 12):
        return False

    motion = deepcopy(shot['motion'])
    if shot['recoil_ticks'] > 4:
        speed = shot['steering_speed']
        if target is None:
            speed = _mul(speed,.9)
        else:
            dx, dy = [_sub(x,y) for x,y in zip(target['position'][:2],motion['position'][:2])]
            distance_squared = _add(_mul(dx,dx),_mul(dy,dy))
            clamped = False
            if shot['tick'] > 30:
                if distance_squared <= 256.:
                    speed = _mul(speed,.5)
                elif distance_squared <= 14400.:
                    speed = _add(_mul(_sub(speed,3.),.9),3.)
                else:
                    speed = _add(speed,.15)
                    clamped = speed > 16.
                    speed = min(speed,16.)
            factor = .7 if clamped or speed >= 4. else .8 if speed >= 1. else .9
            angle = motion['angle']
            desired = _f32(math.atan2(dy,dx))
            delta = _sub(desired,angle)
            if delta > _f32(math.pi):
                delta = _sub(desired,_add(angle,2*math.pi))
            elif _sub(angle,desired) > _f32(math.pi):
                delta = _sub(desired,_sub(angle,2*math.pi))
            motion['angle'] = _wrap_angle(_add(angle,_mul(delta,factor)))
        shot['steering_speed'] = speed
        motion['speed'] = speed

    shot.update(target=target['owner'] if target else 0, target_key=target['key'] if target else 0)
    shot['tick'] += 1
    shot['recoil_ticks'] += 1
    _calculate_velocity(motion,game_speed)
    _apply_motion(motion)
    shot['motion'] = motion
    return True


def _gohei_contact(shot,target):
    """Check the current round damage region of the active purification rod."""
    from .native_enemy_motion import _add, _mul, _sub

    if target['flags'] & (0x21 | 0x1000) or target['dimensions'][0] <= 0.:
        return False
    size = _add(shot['circle_radius'],_mul(target['dimensions'][0],.5))
    x,y = [_sub(a,b) for a,b in zip(shot['motion']['position'][:2],target['position'][:2])]
    return _add(_mul(x,x),_mul(y,y)) <= _mul(size,size)



def _ordinary_sprite_sizes(shots):
    sizes = {}
    for shot in shots:
        render = shot.get("render")
        if render is None or shot["tick_callback"] not in (0, 0x518d0):
            continue
        if (render["flags"] >> 21 & 511) != 16 or render["draw_flags"] != 0x24240:
            raise UnsupportedNativeFire("Unsupported shot sprite anchor or coordinate mode")
        if (render["parent"] or render["angle"] != shot["motion"]["angle"]
                or render["center"] != shot["motion"]["position"]
                or any(render["translation"] + render["extra"] + render["pivot"])
                or render["scale"] != [1.] * 4):
            raise UnsupportedNativeFire("The native sprite does not use the centered unit-scale contract")
        sizes[shot["tick_callback"]] = render["sprite_size"]
    return sizes


def forecast_continuous_fire(initial, targets, viewport, path, actions, *, require_sprites=False):
    """Advance the captured shooting state along one four-frame input tape."""
    phase = initial["fire"]
    options = {int(key): value for key, value in phase["options"].items()}
    tables = {int(mode): shooters for mode, shooters in phase["tables"].items()}
    if phase["option_closing"] or phase["player_mode"] != 1:
        raise UnsupportedNativeFire("The shadow predictor needs an ordinary focused/unfocused firing table")
    path = np.asarray(path, dtype=float)
    horizon = len(path) - 1
    if len(actions) * 4 != horizon:
        raise ValueError("The movement tape must cover every predicted frame")
    follow_rate = phase["option_follow_rate"]
    clock_and_options = []
    initial_shots = [shot.copy() for shot in initial["shots"]]
    sizes = _ordinary_sprite_sizes(initial_shots) if viewport else {}
    if viewport and viewport["scale"] != 1.:
        raise UnsupportedNativeFire("The viewport proof needs a unit-scale original game")

    def attach_sprite(shot):
        size = sizes.get(shot["tick_callback"])
        if size is not None:
            shot.update(ordinary_sprite=size, viewport=viewport)
        elif require_sprites and shot["tick_callback"] in (0, 0x518d0):
            raise UnsupportedNativeFire("The ordinary shooter has no measured sprite for offscreen removal")

    for shot in initial_shots:
        attach_sprite(shot)

    def schedule(tick, targets, active):
        nonlocal phase, options
        planned = [*path[tick], 0.]
        focused = actions[(tick - 1) // 4] >= 9
        next_options = advance_player_options(options, planned, focused=focused, rate=follow_rate)
        if next_options is None:
            raise UnsupportedNativeFire("An option callback needs native motion support")
        options = next_options
        due, phase = advance_firing_clock(phase, shoot=True)
        clock_and_options.append((phase["clock"], phase["previous_clock"], focused,
                                  {index: option["position"] for index, option in options.items()}))
        if due is None:
            return [], focused
        special_active = any(shot["tick_callback"] == 0x51cd0 for shot in active)
        born = []
        for shooter in tables[int(focused)]:
            if due % shooter["period"] != shooter["phase"] or (shooter["kind"] == 2 and special_active):
                continue
            shot = initial_player_shot(shooter, planned, options, targets)
            if shot is None:
                raise UnsupportedNativeFire(f"Unsupported firing entry at tick {tick}")
            attach_sprite(shot)
            shot['shot_flags'] = 1 | (2 if focused else 0)
            born.append(shot)
            special_active |= shooter["kind"] == 2
        return born, focused

    track_drops = (all("drop_counts" in target for target in targets)
                   and "drop_gauges" in phase)
    result = predict_hits(initial_shots, targets, horizon,
                          frame_cap=phase["max_damage"], damage_zones=initial["damage_zones"],
                          allow_gohei=True, shot_schedule=schedule,
                          rng_gauges=phase["drop_gauges"] if track_drops else None,
                          power_level=phase["power_level"])
    return result, clock_and_options


def capture_firing_scene(reader, snapshot, targets=None, *, horizon=96, power_levels=(3, 4)):
    """Capture the shot, native target and drop inputs once for all candidate courses."""
    from .native_shot_rules import capture_enemy_states

    memory = reader.mem
    base = memory.base
    states = capture_enemy_states(reader, snapshot) if targets is None else targets
    if not states or any(state['global_game_speed'] != 1. for state in states):
        raise UnsupportedNativeFire('The damage forecast needs a populated original unit-speed scene')
    fire = read_player_firing_state(memory, base, power=snapshot.power, capture_tables=True)
    if not fire or fire['option_closing'] or fire['player_mode'] != 1 or not fire['shot_input']:
        raise UnsupportedNativeFire('The player is outside the supported continuous shooting mode')
    if not fire.get('tables') or fire.get('power_level') not in power_levels:
        raise UnsupportedNativeFire(
            f"Unsupported original shooter table at power {snapshot.power}, level {fire.get('power_level')}")
    size, left, top, *offsets = struct.unpack('<f8i', memory.read(base + 0xf7a6c, 36))
    viewport = dict(scale=size, origin=[left, top], offsets=offsets)
    shots = read_player_shot_motions(memory, base)
    for shot in shots:
        shot['render'] = read_shot_render_state(memory, base, shot['animation'])
    zones = read_player_damage_zones(memory, base)
    projected = conditional_targets(reader, states, horizon, include_screen=True, include_drops=True,
                                    include_damage_modifiers=True)
    return dict(initial=dict(fire=fire, shots=shots, damage_zones=zones),
                states=states, targets=projected, viewport=viewport)
