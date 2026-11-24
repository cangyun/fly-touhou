"""Predict upcoming ordinary line lasers from the paused emitter's live ECL.

The emitter runs once per observation. Each feasible player path supplies its
own position at a recorded aim read, even if firing occurs on a later tick.
The emission remains conditional on the emitter surviving until that tick.
"""
import math
import struct
from functools import lru_cache

import numpy as np

from .native_enemy_ecl import ECLIncomplete, apply_tick_callback_effects, f32, forecast_ecl
from .native_enemy_motion import ALREADY_UPDATED_FLAG, _step_motion
from .native_shot_rules import ReadCache, RuleWorld


LASER_STOPS = {f'unimplemented_ecl_opcode_{opcode}' for opcode in (700, 701, 702)}


class _LineWorld(RuleWorld):
    def __init__(self, memory, state, instructions, *, strict=False):
        super().__init__(memory, state, player_path=(), rng=None, events=[],
                         shared_instructions=instructions)
        self.aims = {}
        self.radial_offsets = {}
        self.strict = strict
        self.point_bounds = []

    def global_value(self, variable, kind):
        if variable in (-10000, -9999, -9998, -9991, -9990, -9989, -9987):
            raise ECLIncomplete('path_or_random_dependent_emitter_control')
        return super().global_value(variable, kind)

    def _aimed_line_pair(self, thread, first):
        """Execute a local aim followed by two native polar offsets and line births."""
        if len(first) != 20 or first[12] or struct.unpack_from('<H', first, 8)[0] != 1:
            return False
        if struct.unpack_from('<f', first, 16)[0] != -9989.:
            return False
        # Both offset expressions consume their temporary stack value at 626.
        emission = ((44, 1, 0, 20), (44, 0, 0, 20), ((51, 53), 0, 0, 16),
                    (626, 2, 8, 28), (702, 0, 0, 20))
        pattern = ((45, 1, 0, 20), (604, 2, 0, 28)) + emission * 2
        instructions = []
        address = thread.ip + len(first)
        instruction_time = struct.unpack_from('<i', first)[0]
        for opcode, mask, cleanup, size in pattern:
            instruction = self.instruction(address)
            due, actual, _, flags = struct.unpack_from('<ihHH', instruction)
            opcodes = opcode if isinstance(opcode, tuple) else (opcode,)
            if (due != instruction_time or actual not in opcodes
                    or flags != mask or instruction[10] != first[10]
                    or instruction[12] != cleanup or len(instruction) != size):
                return False
            instructions.append(instruction)
            address += size
        assign, heading = instructions[:2]
        local, = struct.unpack_from('<f', assign, 16)
        index, source, spread = struct.unpack_from('<iff', heading, 16)
        if local < 0 or local != source or not math.isfinite(spread):
            return False
        offsets = []
        for start in (2, 7):
            read, literal, arithmetic, polar, birth = instructions[start:start+5]
            polar_index, stack_argument, distance = struct.unpack_from('<iff', polar, 16)
            rotation, = struct.unpack_from('<f', literal, 16)
            source, = struct.unpack_from('<f', read, 16)
            birth_index, = struct.unpack_from('<i', birth, 16)
            if (source != local or polar_index != index or birth_index != index or stack_argument != -1.
                    or not math.isfinite(rotation) or not math.isfinite(distance)):
                return False
            if struct.unpack_from('<h', arithmetic, 4)[0] == 53:
                rotation = -rotation
            offsets.append((rotation, distance))
        # Other instructions that read this local still see an unknown angle.
        self.target(thread, assign, 0, None, 'f')
        shooter = self.shooter(index)
        shooter.angle, shooter.spread = 0., spread
        shooter.offset = [0., 0.]
        aim_origin = tuple(self.state['final']['position'][:2])
        self.aims[index] = self.frame, aim_origin
        for rotation, distance in offsets:
            self.radial_offsets[index] = (self.frame, aim_origin, rotation, distance)
            self._line_birth(index)
        thread.ip = address - len(first)
        return True

    def _line_birth(self, index):
        shooter = self.shooter(index)
        initial, maximum, limit, width = shooter.laser
        first = shooter.ex[0]
        has_program = any(shooter.ex) if self.strict else bool(first and first['command'])
        if (has_program or shooter.radius != 0 or width <= 3 or shooter.speed <= 0
                or initial < 0 or maximum < initial or limit < 0):
            if self.strict:
                raise ECLIncomplete('unsupported_line_birth_geometry')
            return
        position = shooter.absolute if shooter.absolute is not None else self.state['final']['position'][:2]
        origin = [f32(p + offset) for p, offset in zip(position, shooter.offset)]
        if not all(math.isfinite(v) for v in (*origin, initial, maximum, limit, width, shooter.speed, shooter.angle)):
            if self.strict:
                raise ECLIncomplete('nonfinite_line_birth_geometry')
            return
        aim_tick, aim_origin = self.aims.get(index, (0, origin))
        event = dict(tick=self.frame, owner=self.owner, origin=origin, aim_tick=aim_tick,
                     aim_origin=aim_origin, angle=shooter.angle, speed=shooter.speed, width=width,
                     initial_length=initial, max_length=maximum, travel_limit=limit)
        if index in self.radial_offsets:
            event['radial_offset'] = self.radial_offsets[index]
        self.future_events.append(event)

    def rule_instruction(self, thread, ins, op, order):
        if op == 44 and self._aimed_line_pair(thread, ins):
            return True
        if op == 604:
            index = self.arguments(thread, ins, 'i')[0]
            direct_aim = (struct.unpack_from('<H', ins, 8)[0] & 2
                          and struct.unpack_from('<f', ins, 20)[0] == -9989.)
            if direct_aim:
                shooter = self.shooter(index)
                self.aims[index] = self.frame, tuple(self.state['final']['position'][:2])
                shooter.angle = 0.
                shooter.spread = self.argument(thread, ins, 2, 'f')
                return True
            self.aims.pop(index, None)
        elif op == 600:
            index = self.arguments(thread, ins, 'i')[0]
            self.aims.pop(index, None)
            self.radial_offsets.pop(index, None)
        elif op in (603, 626):
            self.radial_offsets.pop(self.argument(thread, ins, 0), None)
        elif op == 614:
            dest, source = self.arguments(thread, ins, 'ii')
            if source in self.aims:
                self.aims[dest] = self.aims[source]
            else:
                self.aims.pop(dest, None)
            if source in self.radial_offsets:
                self.radial_offsets[dest] = self.radial_offsets[source]
            else:
                self.radial_offsets.pop(dest, None)
        elif op == 702:
            self._line_birth(self.arguments(thread, ins, 'i')[0])
            return True
        elif op in (613, 615, 616):
            raise ECLIncomplete('future_global_projectile_mutation')
        return super().rule_instruction(thread, ins, op, order)

    def emit(self, shooter, kind, thread):
        if not self.strict:
            return
        if any(self.shooters[index] is shooter for index in self.radial_offsets):
            raise ECLIncomplete('unsupported_radial_offset_projectile_birth')
        commands = [p for p in shooter.ex if p is not None]
        supported = (kind == 'bullet' and shooter.style == 4 and shooter.aim in (1, 8)
            and shooter.layers == 1 and 0 < shooter.count <= 8192 and len(commands) == 1
            and commands[0]['command'] == 2 and commands[0]['condition'] == 0
            and commands[0]['ints'][0] == 1)
        if not supported:
            raise ECLIncomplete(f'unsupported_concurrent_{kind}_birth')
        position = self.state['final']['position'][:2] if shooter.absolute is None else shooter.absolute
        center = [f32(v + shift) for v, shift in zip(position, shooter.offset)]
        radius, = struct.unpack('<f', self.memory.read(self.base + 0xd1f94 + shooter.style * 0x114, 4))
        speed = abs(shooter.speed) + (abs(shooter.speed2) if shooter.aim == 8 else 0.)
        if radius <= 0 or not all(math.isfinite(v) for v in
                (*center, radius, speed, shooter.radius, shooter.angle, shooter.spread)):
            raise ECLIncomplete('unknown_point_birth_extent')
        event = dict(tick=self.frame, origin=center, bullet_radius=radius,
                     muzzle_radius=abs(shooter.radius), max_speed=speed)
        if (shooter.ex[0] is commands[0] and self.state.get('ecl_time_step') == 1.
                and struct.unpack('<f', self.memory.read(self.base + 0xd8f58, 4))[0] == 1.):
            from .native_bullet_animation import initial_ex2_half_steps

            event['unit_ex2_rewind'] = True
            event['minimum_half_steps'] = initial_ex2_half_steps(self.memory, shooter.style, shooter.color)
        signed_minimum = min(shooter.speed, shooter.speed + shooter.speed2) if shooter.aim == 8 else shooter.speed
        if signed_minimum >= 0:
            shooter_index = next(index for index, entry in self.shooters.items() if entry is shooter)
            aim_tick, aim_origin = self.aims.get(shooter_index, (0, center))
            spread = abs(shooter.spread) * ((shooter.count - 1) / 2 if shooter.aim == 1 else 1.)
            if spread < math.pi:
                event.update(angle=shooter.angle, aim_tick=aim_tick, aim_origin=aim_origin, spread=spread)
        self.point_bounds.append(event)


def forecast_line_emitter(memory, state, horizon, instructions):
    """Advance body motion, contact flags and births on one script timeline."""
    world = None

    def create(reader, observed):
        nonlocal world
        world = _LineWorld(reader, observed, instructions, strict=True)
        return world

    projection = forecast_ecl(memory, state, horizon, world_factory=create)
    if world is None:
        return projection, (), (), 1
    boundaries = [projection.get('unknown_from_tick'), projection.get('contact_unknown_from_tick')]
    boundaries.extend(row['tick'] for row in world.rule_unknown)
    first_unknown = min((tick for tick in boundaries if tick is not None), default=None)
    return projection, world.future_events, world.point_bounds, first_unknown


def point_birth_clearances(events, paths, start, end, player_extent):
    """Enclose simultaneous EX2 shots, including their four-velocity rewind."""
    answer = np.full(len(paths), np.inf)
    for event in events:
        first = max(start + 1, event['tick'])
        if first > end:
            continue
        ticks = np.arange(first, end + 1)
        age = ticks - event['tick']
        if event.get('unit_ex2_rewind'):
            # EX2 first rewinds four velocities. At most age+1 full
            # movement steps and the release tick's extra half-step have
            # elapsed. Round the resulting age-2.5 bound outward.
            half_steps = np.minimum(age + 1, event.get('minimum_half_steps', 0))
            travel = np.maximum(0., age - 2. - .5 * half_steps) * event['max_speed']
        else:
            travel = (age + 1) * event['max_speed']
        rewind = 4 * event['max_speed']
        delta = paths[:, ticks] - event['origin']
        distance = np.linalg.norm(delta, axis=-1)
        if 'spread' in event:
            angle = line_angles(event, paths).astype(float)
            axis_x, axis_y = np.cos(angle)[:, None], np.sin(angle)[:, None]
            along = delta[..., 0] * axis_x + delta[..., 1] * axis_y
            across = abs(delta[..., 1] * axis_x - delta[..., 0] * axis_y)
            cos_spread, sin_spread = math.cos(event['spread']), math.sin(event['spread'])
            boundary = distance * cos_spread
            # Project onto the nearest ray in the sector. Forward travel and
            # the initial backward rewind have different radial limits.
            forward = np.where(along >= boundary, distance, along * cos_spread + across * sin_spread)
            backward = np.where(-along >= boundary, distance, -along * cos_spread + across * sin_spread)
            along_forward = np.clip(forward, 0., travel)
            along_backward = np.clip(backward, 0., rewind)
            forward_sq = distance**2 + along_forward * (along_forward - 2 * forward)
            backward_sq = distance**2 + along_backward * (along_backward - 2 * backward)
            distance = np.sqrt(np.maximum(np.minimum(forward_sq, backward_sq), 0.))
        else:
            distance -= np.maximum(travel, rewind)
        radius = event['muzzle_radius'] + event['bullet_radius'] + player_extent + .1
        gap = distance - radius
        answer = np.minimum(answer, gap.min(axis=1))
    return answer


def forecast_line_births(memory, snapshot, horizon=96):
    resources = snapshot.extra.get('resources', {})
    cache = ReadCache(memory)
    instructions = {}
    events = []
    seen = set()
    for enemy in (*resources.get('enemies', ()), *resources.get('hidden_enemy_contacts', ())):
        state = enemy.get('native_motion')
        if not state or state['owner_address'] in seen or state['effective_game_speed'] != 1.:
            continue
        seen.add(state['owner_address'])
        observed = state.get('ecl_forecast') or {}
        if (observed.get('unknown_reason') not in LASER_STOPS and not any(
                LASER_STOPS.intersection(t.get('unmodeled', ())) for t in observed.get('threads', ()))):
            continue
        try:
            world = _LineWorld(cache, state, instructions)
        except ECLIncomplete:
            continue
        for tick in range(1, horizon + 1):
            if world.state.get('ecl_time_step') is None or world.motion_unknown:
                break
            skip = bool(world.state['flags'] & ALREADY_UPDATED_FLAG)
            try:
                _step_motion(world.state)
                if not skip:
                    world.tick(tick)
                    apply_tick_callback_effects(world)
            except (NotImplementedError, ECLIncomplete):
                break
        events.extend(world.future_events)
    return events


def _player_angles(paths, tick, origin):
    delta = paths[:, tick] - np.asarray(origin, np.float32)
    angles = np.arctan2(delta[:, 1].astype(float), delta[:, 0].astype(float))
    return np.where((delta == 0.).all(axis=1), math.pi / 2, angles).astype(np.float32)


def line_angles(event, paths):
    if not event['aim_tick']:
        return np.full(len(paths), event['angle'], np.float32)
    return _player_angles(paths, event['aim_tick'], event['aim_origin'])


def line_origins(event, paths):
    origin = np.broadcast_to(np.asarray(event['origin'], np.float32), (len(paths), 2))
    if 'radial_offset' not in event:
        return origin
    tick, aim_origin, rotation, distance = event['radial_offset']
    heading = (_player_angles(paths, tick, aim_origin) + np.float32(rotation)).astype(float)
    offset = (np.stack((np.cos(heading), np.sin(heading)), axis=1) * distance).astype(np.float32)
    return origin + offset


@lru_cache(maxsize=192)
def _line_growth(initial, maximum, limit, speed, ticks):
    length, maximum, limit, speed = map(np.float32, (initial, maximum, limit, speed))
    lengths = np.zeros(ticks+1, np.float32)
    translations = np.zeros(ticks+1, int)
    travel = np.float32(0.)
    lengths[0] = length
    for age in range(1, ticks+1):
        translations[age] = translations[age-1]
        if maximum > length:
            length = min(length + speed, maximum)
        else:
            travel += speed
            translations[age] += 1
            if limit > 0 and travel + length > limit:
                length = maximum = limit - travel
                if length <= 0:
                    break
        lengths[age] = length
    return lengths, translations


def line_clearances(events, paths, start, end, half_extents):
    """Worst rotated native rectangle clearance for each complete candidate path."""
    paths = np.asarray(paths, np.float32)
    answer = np.full(len(paths), np.inf)
    half = np.asarray(half_extents, np.float32)
    for event in events:
        birth = event['tick']
        if birth > end:
            continue
        times = np.arange(max(start + 1, birth), end + 1)
        if not len(times):
            continue
        age = times - birth + 1
        speed = np.float32(event['speed'])
        lengths, translation_counts = _line_growth(event['initial_length'], event['max_length'],
            event['travel_limit'], speed, max(120, end))
        length = lengths[age]
        live = length > 16
        if not live.any():
            continue
        times, length, translations = times[live], length[live], translation_counts[age[live]]
        angle = line_angles(event, paths).astype(float)
        direction = np.stack((np.cos(angle), np.sin(angle)), axis=1)
        # The original adds a float32 velocity to a float32 position on
        # every tick. One multiplication differs near a hitbox boundary.
        positions = np.empty((len(paths), int(translations.max()) + 1, 2), np.float32)
        positions[:, 0] = line_origins(event, paths)
        positions[:, 1:] = (speed * direction).astype(np.float32)[:, None, :]
        np.add.accumulate(positions, axis=1, out=positions)
        origin = positions[:, translations]
        anchor = origin + ((length / np.float32(10))[None, :, None] * direction[:, None, :]).astype(np.float32)
        diff = paths[:, times] - anchor
        c, s = direction.astype(np.float32).T
        along = diff[..., 0] * c[:, None] + diff[..., 1] * s[:, None]
        across = diff[..., 1] * c[:, None] - diff[..., 0] * s[:, None]
        width = np.float32(event['width'])
        half_width = (width * .5 if width < 32 else width * .5 - 8.) * .5
        dx = np.maximum((along - half[0]).astype(float) - (length*np.float32(4))/np.float32(5),
                        -(along + half[0]).astype(float))
        dy = np.maximum((across - half[1]).astype(float) - half_width,
                        -half_width - (across + half[1]).astype(float))
        margin = np.hypot(np.maximum(dx, 0), np.maximum(dy, 0)) + np.minimum(np.maximum(dx, dy), 0)
        answer = np.minimum(answer, margin.min(axis=1))
    return answer


def straight_line_warnings(events, snapshot, source, half_extents):
    """Conditional straight-move cues for the existing 4/12/24 sensory slots."""
    from .native_wall_geometry import player_paths
    from .thrl_movement_teacher import action_velocities
    velocity = action_velocities().copy()
    if source[18] < 0:
        velocity[9:] = velocity[:9]
    origin = [snapshot.player.x, snapshot.player.y]
    paths = player_paths(np.broadcast_to(origin, velocity.shape), velocity)
    margins = np.stack([line_clearances(events, paths, 0, h, half_extents) for h in (4, 12, 24)])
    return np.tanh((1. - np.clip(margins, -64., 64.)) / 8.).astype(np.float32)
