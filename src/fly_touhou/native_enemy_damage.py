"""Original TH14 1.00b incoming-damage bound, not an empirical DPS estimate.

4516dd..4516eb clamps the combined shot/Bomb result of451430 to
Player(+688)->shot_definition(+28).423a86..423aae can add an enemy callback
and a queued one-off amount AFTER that cap.424eb0/424f90/425070/425180 can
apply99999 damage to enemies with flag800. Those exceptional paths must not
be silently covered by the ordinary shot cap.
"""
import struct


REVISION = 'th14-native-shot-bomb-damage-bound-v1'


def read_damage_bound(memory, data, image_base, flags):
    def word(address, signed=False):
        return struct.unpack('<i' if signed else '<I', memory.read(address, 4))[0]
    answer = dict(revision=REVISION, per_frame=None, initial_extra=0,
                  source='native451430 cap;423a86 extra sources;424eb0 flag800 exception')
    player = word(image_base + 0xdb67c)
    if not player:
        return dict(answer, reason='player unavailable')
    definition = word(player + 0x688)
    if not definition:
        return dict(answer, reason='shot definition unavailable')
    cap = word(definition + 0x28, True)
    callback = word(data + 0x4114)
    extra = max(0, word(data + 0x3fd0, True))
    answer.update(shot_definition=definition, shot_bomb_cap=cap,
                  extra_damage_callback=callback, initial_extra=extra,
                  exceptional_area_damage=bool(flags & 0x800))
    if callback:
        return dict(answer, reason='enemy extra-damage callback is not bounded')
    if flags & 0x800:
        return dict(answer, reason='native99999 area-damage path is eligible')
    if cap < 0:
        return dict(answer, reason='negative native damage cap')
    # Subsequent rank/resistance/spell factors in423ac2..423b9b only reduce
    # damage;421480 health conversion also cannot exceed the raw damage.
    return dict(answer, per_frame=cap, reason=None)



def read_player_damage_zones(memory, image_base, *, retained_slots=()):
    """Read the active damage regions used by the original enemy hit routine.

    450130 and 450270 allocate these slots. 451430 reads their positions,
    active flags, timers and damage values. Projectile movement and damage
    callbacks require their own forecast.
    """
    player, = struct.unpack('<I', memory.read(image_base + 0xdb67c, 4))
    if not player:
        return []

    slot_size = 0xa4
    slot_count = 256
    data = memory.read(player + 0xde18, slot_count * slot_size)
    zones = []
    retained = frozenset(retained_slots)
    for slot in range(slot_count):
        offset = slot * slot_size
        flags, = struct.unpack_from('<I', data, offset + 0x10)
        active = bool(flags & 1)
        if not active and slot not in retained:
            continue

        position = list(struct.unpack_from('<3f', data, offset + 0x2c))
        previous_tick, tick = struct.unpack_from('<2i', data, offset + 0x70)
        damage, used, limit, interval = struct.unpack_from('<4i', data, offset + 0x84)
        last_contact, = struct.unpack_from('<I', data, offset + 0x94)
        callback, = struct.unpack_from('<I', data, offset + 0xa0)
        radius = struct.unpack_from('<f', data, offset + 0x14)[0] if flags & 2 else None
        zones.append(dict(slot=slot, active=active, flags=flags, position=position, circle_radius=radius,
                          previous_tick=previous_tick, tick=tick, damage=damage,
                          damage_accumulated=used, damage_limit=limit,
                          hit_interval=interval, last_contact_key=last_contact,
                          has_damage_callback=bool(callback)))
    return zones


def read_player_shot_motions(memory, image_base):
    """Read active TH14 player shots and their links to damage regions.

    ParaKit documents the shot-array layout. The original allocation routine
    at 450d96..450dc1 links each ordinary shot with its damage-region slot.
    """
    from .native_enemy_motion import _motion

    player, = struct.unpack('<I', memory.read(image_base + 0xdb67c, 4))
    if not player:
        return []

    shot_size = 0xd0
    slot_count = 256
    damage_size = 0xa4
    damage_start = player + 0xde18
    data = memory.read(player + 0x6c8, slot_count * shot_size)
    definitions = {}
    shots = []
    for slot in range(slot_count):
        offset = slot * shot_size
        if struct.unpack_from('<i', data, offset + 0x98)[0] != 1:
            continue

        definition, damage_region = struct.unpack_from('<2I', data, offset + 0xbc)
        if definition and definition not in definitions:
            source = memory.read(definition, 0x38)
            tick_callback, _, hit_callback = struct.unpack_from('<3I', source, 0x2c)
            definitions[definition] = (tick_callback, hit_callback)
        tick_callback, hit_callback = definitions.get(definition, (None, None))
        damage_offset = damage_region - damage_start
        linked_slot = (damage_offset // damage_size
                       if 0 <= damage_offset < slot_count * damage_size
                       and damage_offset % damage_size == 0 else None)
        animation, = struct.unpack_from('<I', data, offset + 0x14)
        tick, = struct.unpack_from('<i', data, offset + 0x1c)
        damage, = struct.unpack_from('<i', data, offset + 0xac)
        target, target_key = struct.unpack_from('<2I', data, offset + 0x9c)
        has_target_callback = tick_callback in (image_base + 0x518d0, image_base + 0x51cd0)
        flags, = struct.unpack_from('<I', data, offset + 0x10)
        shots.append(dict(slot=slot, damage_slot=linked_slot, tick=tick, damage=damage, shot_flags=flags,
                          definition=definition, animation=animation,
                          target=target if has_target_callback else None,
                          target_key=target_key if has_target_callback else None,
                          hitbox=list(struct.unpack_from('<2f', data, offset + 0xb0)),
                          motion=_motion(data, offset + 0x54),
                          tick_callback=tick_callback - image_base if tick_callback else tick_callback,
                          hit_callback=hit_callback - image_base if hit_callback else hit_callback))
        if tick_callback == image_base + 0x51cd0:
            shots[-1].update(release_ticks=struct.unpack_from('<i', data, offset + 0x30)[0],
                             recoil_ticks=struct.unpack_from('<i', data, offset + 0x44)[0],
                             steering_speed=struct.unpack_from('<f', data, offset + 0xb8)[0])
    return shots


def advance_known_player_shot(shot, *, homing_target, game_speed=1.):
    """Advance ordinary motion or the 4518d0 homing callback for one tick.

    Callers provide the already-resolved target position, or None when the
    engine has no valid target this tick. Target selection, hits, and culling
    are separate from this motion update.
    """
    import math
    from copy import deepcopy

    from .native_enemy_motion import _add, _apply_motion, _calculate_velocity, _f32, _mul, _sub, _wrap_angle

    callback = shot['tick_callback']
    if callback not in (0, 0x518d0) or shot['motion']['mode'] != 0:
        return None

    motion = deepcopy(shot['motion'])
    if callback:
        speed = motion['speed']
        if homing_target is None:
            motion['speed'] = min(16., _add(speed, .1))
        elif shot['tick'] >= 60:
            motion['speed'] = _add(speed, .2)
        else:
            x, y = motion['position'][:2]
            target_x, target_y = homing_target[:2]
            target_angle = _f32(math.atan2(_sub(target_y, y), _sub(target_x, x)))
            angle = motion['angle']
            delta = _sub(target_angle, angle)
            if delta > _f32(math.pi):
                delta = _sub(target_angle, _add(angle, 2 * math.pi))
            elif _sub(angle, target_angle) > _f32(math.pi):
                delta = _sub(target_angle, _sub(angle, 2 * math.pi))
            if abs(delta) >= _f32(math.pi / 4):
                motion['speed'] = max(4., _sub(speed, .2))
            elif abs(delta) < _f32(math.pi / 12):
                motion['speed'] = min(16., _add(speed, .2))
            motion['angle'] = _wrap_angle(_add(angle, _mul(delta, .08)))

    _calculate_velocity(motion, game_speed)
    _apply_motion(motion)
    return motion


def resolve_player_homing_target(shot, targets):
    """Return the original homing callback's target on this tick, if any."""
    callback = shot['tick_callback']
    if callback not in (0x518d0, 0x51cd0):
        return None
    if callback == 0x51cd0 and shot['tick'] < 10 and not shot['target']:
        return None
    immune_flags = 0xc000021
    if shot['target']:
        for target in targets:
            if target['key'] == shot['target_key'] and not target['flags'] & immune_flags:
                return target
        # A vanished or immune old target is cleared. Acquisition resumes on
        # the next tick, so another nearby enemy cannot steal this update.
        return None

    radius = 320. if callback == 0x51cd0 else 256.
    return nearest_player_damage_target(shot['motion']['position'], targets, radius)


def nearest_player_damage_target(position, targets, radius):
    """First nearest eligible original target within the specified radius."""
    from .native_enemy_motion import _add, _mul, _sub

    immune_flags = 0xc000021
    limit = radius ** 2
    nearest = None
    x, y = position[:2]
    for target in targets:
        if target['flags'] & immune_flags:
            continue
        dx, dy = _sub(x, target['position'][0]), _sub(y, target['position'][1])
        distance = _add(_mul(dx, dx), _mul(dy, dy))
        if distance < limit:
            limit = distance
            nearest = target
    return nearest


def ordinary_shot_culled(shot, sprite_size, viewport):
    """The unit-scale original viewport test for a centered, rotation-linked sprite.

    Before the tenth update the engine keeps new shots off-screen. After
    that, 45115f checks the four corners of the previous animation pose,
    before that pose is updated to the current shot position.
    """
    import math

    from .native_enemy_motion import _add, _f32, _mul, _sub

    if shot['tick'] < 10:
        return False
    motion = shot['motion']
    left, top = viewport['origin']
    offset_x, offset_y = viewport['offsets'][4:6]
    x = _add(motion['position'][0], offset_x)
    y = _add(motion['position'][1], offset_y)
    angle = motion['angle']
    cosine, sine = _f32(math.cos(angle)), _f32(math.sin(angle))
    hx, hy = (_mul(.5, dim) for dim in sprite_size)
    for u in (-hx, hx):
        for v in (-hy, hy):
            px = _add(x, _sub(_mul(u, cosine), _mul(v, sine)))
            py = _add(y, _add(_mul(u, sine), _mul(v, cosine)))
            if left < px < left + 384. and top < py < top + 448.:
                return False
    return True


def enemy_exits_view(target):
    """Original 42369e post-movement visibility removal for a measured sprite size."""
    from .native_enemy_motion import _add, _mul, _sub

    flags = target['flags']
    width, height = target['screen_size']
    x, y = target['position'][:2]
    hx, hy = _mul(.5, width), _mul(.5, height)
    x_outside = _add(x, hx) < -192. or _sub(x, hx) > 192.
    y_outside = _add(y, hy) < 0. or _sub(y, hy) > 448.
    seen = target.get('seen_on_screen', bool(flags & 0x10000)) or bool(flags & 0x10000)
    if not (x_outside or y_outside):
        target['seen_on_screen'] = True
        return False
    target['seen_on_screen'] = seen
    if not seen:
        return False
    return not flags & (4 if x_outside else 8)


def player_shot_overlaps_round_enemy(shot, target):
    """Check the original ordinary rectangle shot against a round damage target.

    None denotes a different native hit shape or an unsupported damage callback.
    """
    import math

    from .native_enemy_motion import _add, _f32, _mul, _sub

    if shot['hit_callback'] != 0 or target['flags'] & 0x1000:
        return None
    width = target['dimensions'][0]
    if target['flags'] & 0x21 or width <= 0.:
        return False
    radius = _mul(width, .5)
    motion = shot['motion']
    dx, dy = [_sub(t, s) for t, s in zip(target['position'][:2], motion['position'][:2])]
    cosine = _f32(math.cos(-motion['angle']))
    sine = _f32(math.sin(-motion['angle']))
    local_x = _sub(_mul(dx, cosine), _mul(dy, sine))
    local_y = _add(_mul(dy, cosine), _mul(dx, sine))
    half_x, half_y = [_mul(size, .5) for size in shot['hitbox']]
    if abs(local_x) <= _add(half_x, radius) and abs(local_y) <= half_y:
        return True
    if abs(local_x) <= half_x and abs(local_y) <= _add(half_y, radius):
        return True
    radius_sq = _mul(radius, radius)
    for x in (_sub(local_x, half_x), _add(local_x, half_x)):
        for y in (_sub(local_y, half_y), _add(local_y, half_y)):
            if _add(_mul(x, x), _mul(y, y)) < radius_sq:
                return True
    return False
