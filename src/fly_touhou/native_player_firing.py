"""Original TH14 player fire schedule and loaded shot definitions."""

import struct


def _shooter_records(memory, table):
    shooters = []
    for index in range(256):
        address = table + index * 0x58
        raw = memory.read(address, 0x58)
        period, phase, damage = struct.unpack_from('<bbh', raw)
        if period < 0:
            return shooters
        if period == 0:
            raise ValueError('An active native shooter has a zero firing period')
        x, y, width, height, angle, speed = struct.unpack_from('<6f', raw, 4)
        option, kind = struct.unpack_from('<bb', raw, 0x20)
        init_callback, tick_callback, _, hit_callback = struct.unpack_from('<4I', raw, 0x28)
        shooters.append(dict(shooter=index, address=address, period=period, phase=phase,
                             damage=damage, offset=[x, y], hitbox=[width, height],
                             angle=angle, speed=speed, option=option, kind=kind,
                             init_callback=init_callback, tick_callback=tick_callback,
                             hit_callback=hit_callback))
    raise ValueError('Original shooter table has no end marker')


def read_player_firing_state(memory, image_base, *, power, capture_tables=False):
    """Capture the loaded TH14 shooter tables, phase clock and option poses."""
    def u32(address):
        return struct.unpack('<I', memory.read(address, 4))[0]

    player = u32(image_base + 0xdb67c)
    if not player:
        return None
    definition = u32(player + 0x688)
    previous, tick, elapsed = struct.unpack('<iif', memory.read(player + 0x18338, 12))
    focus = bool(u32(player + 0x184b0))
    result = dict(previous_clock=previous, clock=tick, elapsed_clock=elapsed,
                  focused=focus, shot_input=bool(u32(image_base + 0xd6a90) & 1),
                  player_mode=u32(player + 0x684), special_active=u32(player + 0x1834c),
                  option_follow_rate=u32(player + 0x182bc), option_closing=bool(u32(player + 0x182d4) & 2),
                  drop_gauges=list(struct.unpack('<2i', memory.read(player + 0x184b4, 8))),
                  position=list(struct.unpack('<3f', memory.read(player + 0x5e0,12))))

    # 4509aa prefers an option's fixed-point position when the shooter names it.
    options = {}
    for index in range(1, 9):
        address = player + 0xd664 + index * 0xe4
        x, y, unfocused_x, unfocused_y, focused_x, focused_y = struct.unpack('<6i', memory.read(address,24))
        angle = struct.unpack('<f', memory.read(address + 0x4c,4))[0]
        options[index] = dict(position=[x / 128., y / 128., 0.], angle=angle,
                              unfocused_offset=[unfocused_x, unfocused_y], focused_offset=[focused_x, focused_y],
                              active=bool(u32(address - 0x5c)), snap=u32(address + 0x78),
                              callback=u32(address + 0x80))
    result['options'] = options

    if capture_tables and definition:
        step = u32(image_base + 0xf5860)
        levels = u32(definition + 0x20)
        power_level = power // step
        tables = {}
        for mode, table_index in ((False, power_level), (True, power_level + levels + 1)):
            pointer = u32(definition + 0xe0 + table_index * 4)
            table = _shooter_records(memory, pointer)
            for shooter in table:
                for name in ('init_callback', 'tick_callback', 'hit_callback'):
                    if shooter[name]:
                        shooter[name] -= image_base
            tables[int(mode)] = table
        result.update(power_level=power_level, power_step=step, tables=tables,
                      max_damage=u32(definition + 0x28))
    return result


def advance_firing_clock(state, *, shoot):
    """Advance the original unit-step 15-frame firing clock."""
    tick = state['clock']
    previous = state['previous_clock']
    if tick < 0:
        if not shoot:
            return None, state
        previous, tick = -1, 0
    due = tick if tick != previous else None
    following = (0 if shoot else -1) if tick == 14 else tick + 1
    return due, dict(previous_clock=tick, clock=following)


def advance_player_options(options, player_position, *, focused, rate):
    """Update original option positions with the integer follow calculation."""
    player = [int(round(value * 128.)) for value in player_position[:2]]
    projected = {}
    for index, option in options.items():
        if not option['active']:
            continue
        if option['callback']:
            return None
        start = [int(round(value * 128.)) for value in option['position'][:2]]
        offsets = option['focused_offset'] if focused else option['unfocused_offset']
        target = [p + d for p, d in zip(player, offsets)]
        difference = [(b-a)*rate for a,b in zip(start,target)]
        delta = [(abs(value) // 100) * (1 if value >= 0 else -1) for value in difference]
        if option['snap'] or (rate>=30 and not any(delta)):
            result = target
        elif rate>=30:
            result = [a+d for a,d in zip(start,delta)]
        else:
            result = start
        projected[index] = dict(option, position=[result[0]/128.,result[1]/128.,0.], snap=0)
    return projected


def initial_player_shot(shooter, position, options, targets, *, game_speed=1.):
    """Initialize a covered player shot from the loaded original shooter record.

    The actual first movement tick is separate. 450c5e subtracts the initial
    velocity before the first ordinary shot update.
    """
    import math

    from .native_enemy_damage import nearest_player_damage_target
    from .native_enemy_motion import _add, _calculate_velocity, _f32, _sub, _wrap_angle

    pair = (shooter['init_callback'], shooter['tick_callback'], shooter['hit_callback'])
    if pair not in ((0,0,0), (0x518b0,0x518d0,0), (0x51af0,0x51cd0,0x52050)) or shooter['angle']>=999.:
        return None
    if shooter['option']:
        option = options.get(shooter['option'])
        if option is None:
            return None
        position = option['position']

    motion = dict(mode=0, position=list(position), velocity=[0.,0.,0.],
                  speed=shooter['speed'], angle=_wrap_angle(shooter['angle']))
    _calculate_velocity(motion,game_speed)
    motion['position'][:2] = [_add(p,_sub(o,v)) for p,o,v in
                              zip(position[:2],shooter['offset'],motion['velocity'][:2])]
    result = dict(motion=motion, tick=0, target=0, target_key=0, definition=shooter['address'],
                  damage=shooter['damage'], hitbox=shooter['hitbox'],
                  tick_callback=shooter['tick_callback'], hit_callback=shooter['hit_callback'])
    if shooter['kind']==2:
        target = nearest_player_damage_target(motion['position'],targets,160.)
        if target is not None:
            dx,dy = [_sub(b,a) for a,b in zip(motion['position'][:2],target['position'][:2])]
            motion['angle'] = _wrap_angle(_f32(math.atan2(dy,dx)))
            result.update(target=target['owner'],target_key=target['key'])
        result.update(circle_radius=shooter['hitbox'][0],recoil_ticks=10,
                      release_ticks=0,steering_speed=motion['speed'])
    return result



def advance_shot_drop_gauges(gauges, impacts, power_level):
    """Advance the gameplay RNG consumers before each supported damage callback.

    453a90 consumes the existing gauges before the shot changes them.
    At max power a 500-point gauge payout consumes one draw and removes 50.
    Every 100 points in the second gauge consume two draws.
    """
    drops, green = gauges
    draws = 0
    for flags, purification_rod, pulse in impacts:
        if power_level >= 4:
            payouts = max(0, (drops - 450) // 50)
            drops -= 50 * payouts
            draws += payouts
        else:
            payout = 200 + 60 * power_level
            stays = max(0, (drops - 500) // payout + 1) if drops >= 500 else 0
            drops -= payout * stays

        green_payouts = max(0, green // 100)
        green -= 100 * green_payouts
        draws += 2 * green_payouts
        if purification_rod:
            green += 200 if pulse else 0
        elif flags & 2:
            green += 5
        else:
            drops += 5
    return (drops, green), draws


def scatter_drop_draws(counts):
    """4215b0 draws one shared starting angle and two numbers per scattered item."""
    return 1 + 2 * sum(max(0, count) for count in counts[:12])
