"""Opt-in model for the observed TH14 distance-reactive orb program.

Preserves its original EX command sequence, including repeated aimed
acceleration. The reader profile enables it only for the matched EX program.
"""
import math
import struct
import numpy as np

from .native_wall_reaction import marker_transition

F = np.float32
WAIT = 0x80000000
OPS = (2, 0x8000, 0, 0, 0, 0x200, 16, WAIT, 4, 0x10000,
       0, 0, 0, 0, 0x200, 16, WAIT, 0x200000)


def parse_program(program_hex):
    data = bytes.fromhex(program_hex)
    if len(data) != 18*44:
        raise ValueError('A complete native18-record program is required')
    records = [struct.unpack_from('<4f4i3I', data, i*44) for i in range(18)]
    if tuple(r[8] for r in records) != OPS or any(r[9] for r in records):
        raise ValueError('This is not the observed distance-reactive program')
    if (any(records[i][4] != 32 for i in (5, 14))
            or records[9][4] != 8 or any(records[i][0:2] != (0., 0.) for i in (6, 15))
            or records[6][6] != 0 or records[15][6] != 1
            or records[8][1] != 999. or records[17][1] != 999.):
        raise ValueError('Unsupported reactive-program physical parameters')
    return records


def _angle(y, x, fallback):
    return F(math.atan2(float(y), float(x))) if x != 0 or y != 0 else F(fallback)


def _polar(speed, angle):
    return np.asarray([math.cos(float(angle))*float(speed),
                       math.sin(float(angle))*float(speed)], dtype=np.float32)


def simulate(point, program, near_radius, far_radius, player_path, *, player_order='after'):
    raw = point['native']
    if raw['native_state'] != 1 or raw['native_flags'] & 0x200:
        raise ValueError('Candidate supports mature, moving orbs only')
    active = int(raw['native_ex_active_flags'])
    if active not in (0, 4, 16, WAIT, 0x200000):
        raise ValueError('Unsupported active EX state')
    if raw.get('native_motion_time_rate', 1.) != 1.:
        raise ValueError('Candidate requires unit native time')
    if player_order not in ('before', 'after'):
        raise ValueError('Unknown player update order')
    path = np.asarray(player_path, dtype=np.float32)
    position = np.asarray(raw['native_position'], dtype=np.float32).copy()
    velocity = np.asarray(raw['native_velocity'], dtype=np.float32).copy()
    speed, angle = F(raw['native_speed']), F(raw['native_angle'])
    marker, timer = int(point['marker']), int(point['reaction_timer'])
    index = int(raw.get('native_ex_index', 18))
    tick = duration = 0
    wait = 0
    target_speed = turn_angle = F(0.)
    mode = 0
    delta = np.zeros(2, np.float32)
    speed_delta = F(0.)
    if active == 16:
        tick, duration = raw['native_ex16_tick'], raw['native_ex16_duration']
        target_speed, turn_angle = F(raw['native_ex16_turn_speed']), F(raw['native_ex16_turn_angle'])
        mode = raw['native_ex16_turn_mode']
        if raw['native_ex16_repeats'] != 1 or mode not in (0, 1):
            raise ValueError('Unsupported native turn repetition/mode')
    elif active == WAIT:
        wait = raw['native_ex_wait_remaining']
    elif active in (4, 0x200000):
        tick, duration = raw['native_ex_acceleration_tick'], raw['native_ex_acceleration_duration']
        delta = np.asarray(raw['native_ex_acceleration'], dtype=np.float32)
        speed_delta = F(raw['native_ex_speed_delta'])
    output = []
    for step in range(1, len(path)):
        player = path[step-1 if player_order == 'before' else step]
        reaction = marker_transition(marker, timer, position, player, near_radius, far_radius)
        marker, timer = reaction['marker'], reaction['timer']
        if reaction['restart_ex_index'] is not None:
            index, active = reaction['restart_ex_index'], 0
        # 416a85 restarts418140 in the SAME native tick when any EX ends.
        for _ in range(40):
            if active == 0:
                if index >= 18 or program[index][8] == 0:
                    break
                record = program[index]
                opcode = record[8]
                index += 1
                if opcode == 0x200:  # Same physical kind32, different color.
                    continue
                if opcode == 0x10000:
                    index = record[4]
                    continue
                if opcode == 16:
                    active, tick, duration = 16, 0, record[4]
                    target_speed, turn_angle, mode = F(record[1]), F(record[0]), record[6]
                elif opcode == WAIT:
                    active, wait = WAIT, record[4]
                elif opcode in (4, 0x200000):
                    active, tick, duration = opcode, 0, record[4]
                    aim = _angle(F(player[1]-position[1]), F(player[0]-position[0]), math.pi/2)
                    aim = F(aim+F(record[2]))
                    # 418d38..418e2b uses a polar scalar-speed difference,
                    # NOT (target_vector-current_vector)/duration.
                    speed_delta = F(record[0]) if opcode == 4 else F(F(F(record[0])-speed)/F(duration))
                    delta = _polar(speed_delta, aim)
                else:
                    raise ValueError('Unmodeled program command during motion')
            if active == 16:
                if tick >= duration:
                    if mode == 1:
                        angle = F(_angle(F(player[1]-position[1]), F(player[0]-position[0]), math.pi/2)+turn_angle)
                    else:
                        angle = F(angle+turn_angle)
                    speed = target_speed
                    velocity = _polar(speed, angle)
                    active = 0
                    continue
                current_speed = F(speed-F(F(F(tick)*speed)/F(duration)))
                velocity = _polar(current_speed, angle)
                tick += 1
                break
            if active == WAIT:
                if wait <= 0:
                    active = 0
                    continue
                wait -= 1
                break
            if active in (4, 0x200000):
                if tick >= duration:
                    active = 0
                    continue
                speed = F(speed+speed_delta)
                velocity = np.asarray(velocity+delta, dtype=np.float32)
                angle = _angle(velocity[1], velocity[0], angle)
                if active == 4:
                    # 4196d9..419701: aimed vector additions recompute scalar
                    # speed from the resulting velocity before a later turn.
                    speed = F(math.sqrt(float(F(F(velocity[0]*velocity[0])+F(velocity[1]*velocity[1])))))
                tick += 1
                break
        else:
            raise ValueError('Reactive EX commands did not reach a timed state')
        position = np.asarray(position+velocity, dtype=np.float32)
        output.append(dict(position=position.copy(), velocity=velocity.copy(), speed=float(speed),
            angle=float(angle), marker=marker, timer=timer, active_flags=active, index=index,
            phase_tick=tick, wait_remaining=wait))
    return output
