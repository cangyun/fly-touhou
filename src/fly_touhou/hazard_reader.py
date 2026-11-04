"""Opt-in raw hazard metadata for the original Japanese TH14 1.00b.

This adds a separate Snapshot.extra schema.  It does not redefine Bullet.active,
the resource sensor, an observation encoder, or native collision eligibility.
All offsets below were checked against the exact original executable; a trial,
translated executable, or unverified identity is deliberately unsupported.
"""
from __future__ import annotations

from dataclasses import replace
import math
import struct

from .native import NativeError
from .resources import ResourceNativeReader

HAZARD_SCHEMA = 'original-th14-hazard-metadata-v1'
COLLISION_GEOMETRY_SCHEMA = 'original-th14-collision-geometry-v1'
FULL_GAME_SHA256 = 'f08e885e28d2e0fc8c579e88034dd7de8f2a648d8f08ce781cf08eadc9ffe353'

# These are object offsets, independent of a relocated executable image base.
# Keep direct reads tied to the legacy field layout used for overlap checks.
_REQUIRED_LAYOUT = {
    'bullets': {
        'zBullet_flags': 0x20, 'zBullet_iframes': 0x24,
        'zBullet_pos': 0xbc0, 'zBullet_velocity': 0xbcc,
        'zBullet_speed': 0xbd8, 'zBullet_angle': 0xbdc,
        'zBullet_hitbox_radius': 0xbe0, 'zBullet_scale': 0x13bc,
        'zBullet_state': 0xc0e, 'zBullet_timer': 0x13c4,
    },
    'laser_base': {
        'zLaserBaseClass_state': 0x10, 'zLaserBaseClass_type': 0x14,
        'zLaserBaseClass_timer': 0x1c, 'zLaserBaseClass_offset': 0x54,
        'zLaserBaseClass_angle': 0x6c, 'zLaserBaseClass_length': 0x70,
        'zLaserBaseClass_width': 0x74, 'zLaserBaseClass_speed': 0x78,
        'zLaserBaseClass_iframes': 0x5b4,
    },
}


def _finite(values, label):
    if not all(math.isfinite(value) for value in values):
        raise NativeError(f'Non-finite raw {label} metadata')


def _same(actual, expected, label):
    # Both sides originate in the same stored float32/integer fields.  A loose
    # epsilon would hide a partial movement update within a frame.
    if actual != expected:
        raise NativeError(f'Raw hazard overlap changed during extraction: {label}')


def _entity_ids(entities, limit, label):
    if len(entities) > limit:
        raise NativeError(f'Too many {label} objects for raw hazard extraction')
    seen = set()
    for entity in entities:
        address = entity.id
        if (isinstance(address, bool) or not isinstance(address, int)
                or not 0x10000 <= address < 0x100000000 or address in seen):
            raise NativeError(f'Invalid or repeated {label} object address')
        seen.add(address)


class VersionedHazardReader(ResourceNativeReader):
    """Read inherited resources plus same-frame raw bullet/laser fields.

    Reading at the owned controller's input checkpoint is recommended.  A frame
    counter and overlapping field checks detect torn reads; they do not turn
    an unsuspended arbitrary process read into an atomic collision oracle.
    """

    def __init__(self, memory, profile, *, inspect_unverified=False):
        super().__init__(memory, profile, inspect_unverified=inspect_unverified)
        if (profile.game_id != 14 or not self.verified_identity
                or self.digest != FULL_GAME_SHA256):
            raise NativeError('Raw hazard metadata requires the pinned original TH14 1.00b SHA256')
        for group, fields in _REQUIRED_LAYOUT.items():
            for key, expected in fields.items():
                if self.o.get(group, {}).get(key) != expected:
                    raise NativeError(f'Unsupported raw hazard layout: {group}.{key}')

    def _bullet_metadata(self, bullet):
        address, memory = bullet.id, self.mem
        flags, counter = struct.unpack('<II', memory.read(address + 0x20, 8))
        geometry = struct.unpack('<9f', memory.read(address + 0xbc0, 36))
        state = struct.unpack('<H', memory.read(address + 0xc0e, 2))[0]
        clock = memory.read(address + 0x13bc, 12)
        scale = struct.unpack_from('<f', clock)[0]
        timer = struct.unpack_from('<i', clock, 8)[0]
        animation_gate = struct.unpack('<i', memory.read(address + 0x4d4, 4))[0]
        x, y, _, vx, vy, _, speed, angle, hitbox = geometry
        _finite((x, y, vx, vy, speed, angle, hitbox, scale), 'bullet')
        _same((x, y, vx, vy, hitbox),
              (bullet.x, bullet.y, bullet.vx, bullet.vy, bullet.radius), 'bullet geometry')
        _same(timer & 0xffffffff, bullet.age, 'bullet timer')
        _same(counter, bullet.invulnerable_frames, 'bullet cancellation counter')
        _same(state == 1, bullet.active, 'legacy bullet active')
        result = {
            'id': address, 'native_state': state, 'native_flags': flags,
            'native_timer': timer, 'native_cancel_protection_counter': counter,
            'native_animation_gate': animation_gate,
            'native_position': [x, y], 'native_velocity': [vx, vy],
            'native_speed': speed, 'native_angle': angle,
            'native_hitbox_value': hitbox, 'native_scale': scale,
        }
        if state in (1,2):
            # 41687c changes state2 to state1 before 416891 resumes 418140.
            # The current EX record may change speed/angle on that release.
            # Preserve its bytes without assigning unverified command meaning
            # or converting arbitrary integer/sentinel slots to floats.
            active_flags = struct.unpack('<I', memory.read(address + 0xc04, 4))[0]
            result['native_ex_active_flags'] = active_flags
        if state == 2 or (state == 1 and active_flags):
            index = struct.unpack('<i', memory.read(address + 0xc20, 4))[0]
            record = memory.read(address + 0xc28 + index * 0x2c, 0x2c) if 0 <= index < 18 else None
            result.update(native_ex_index=index,
                          native_ex_current_record_hex=record.hex() if record is not None else None)
            if active_flags & 0x400000:
                from .native_bullet_scale import read_scale_metadata,REVISION as scale_revision
                result.update(read_scale_metadata(memory,address))
                result['native_ex_scale_forecast_enabled']=(self.profile.hooks or {}).get('native_ex_scale_forecast')==scale_revision
            if active_flags & 0x1000:
                from .native_bullet_wrap import read_wrap_metadata
                result.update(read_wrap_metadata(memory,address))
            if active_flags & 1:
                tick, elapsed = struct.unpack('<if', memory.read(address + 0xf44, 8))
                _finite((elapsed,), 'bullet EX speed-boost timer')
                result.update(native_ex_speed_boost_tick=tick, native_ex_speed_boost_time=elapsed)
            if active_flags & (4|0x200000):
                source=0x200000 if active_flags&0x200000 else 4
                timer_offset,vector_offset,scalar_offset=((0x1214,0x1238,0x1224) if source==0x200000
                    else (0xf8c,0xfb0,0xf9c))
                tick, elapsed = struct.unpack('<if', memory.read(address + timer_offset, 8))
                ax, ay, _, duration = struct.unpack('<3fi', memory.read(address + vector_offset, 16))
                speed_delta = struct.unpack('<f', memory.read(address + scalar_offset, 4))[0]
                _finite((elapsed,ax,ay,speed_delta), 'bullet EX linear acceleration')
                result.update(native_ex_acceleration_tick=tick,native_ex_acceleration_time=elapsed,
                    native_ex_acceleration=[ax,ay],native_ex_acceleration_duration=duration,
                    native_ex_speed_delta=speed_delta,native_ex_acceleration_source=source)
            if active_flags & 8:
                # 419970 updates polar speed/angle from this separate timer.
                # Record the native fields; prediction is not yet certified.
                tick, elapsed, rate_pointer = struct.unpack('<ifI', memory.read(address + 0xfd4, 12))
                rate=(struct.unpack('<f',memory.read(rate_pointer,4))[0] if rate_pointer else 1.)
                motion_rate=(rate if rate_pointer==memory.base+0xd8f58 else
                    struct.unpack('<f',memory.read(memory.base+0xd8f58,4))[0])
                speed_delta, angle_delta = struct.unpack('<ff', memory.read(address + 0xfe4, 8))
                duration = struct.unpack('<i', memory.read(address + 0x1004, 4))[0]
                _finite((elapsed,speed_delta,angle_delta,rate,motion_rate), 'bullet EX polar motion')
                result.update(native_ex8_tick=tick,native_ex8_time=elapsed,
                    native_ex8_speed_delta=speed_delta,native_ex8_angle_delta=angle_delta,
                    native_ex8_duration=duration,native_ex8_time_rate=rate,native_motion_time_rate=motion_rate)
                if ((self.profile.hooks or {}).get('native_ex8_polar4_forecast')
                        == 'th14-ex8-polar4-candidate-v1'):
                    # Optional measured four-frame motion prefix. The default
                    # sensor continues to treat EX8 as an untrusted hint.
                    result['native_ex8_polar4_forecast_enabled']=True
            if active_flags & 16:
                from .native_bullet_turn import read_turn_metadata
                result.update(read_turn_metadata(memory,address))
            if active_flags & 0x80000000:
                remaining,clock,rate_pointer = struct.unpack('<ifI',memory.read(address+0x10ac,12))
                rate=(struct.unpack('<f',memory.read(rate_pointer,4))[0] if rate_pointer else 1.)
                _finite((clock,rate), 'bullet EX wait countdown')
                result.update(native_ex_wait_remaining=remaining,native_ex_wait_time=clock,native_ex_wait_rate=rate)
                if active_flags==0x80000000 and state==1 and 0<=remaining<=24:
                    from .native_bullet_wait_cancel import read_wait_cancel_metadata
                    result.update(read_wait_cancel_metadata(memory,address,index,record))
            if (active_flags & 0x04000000 and
                    (self.profile.hooks or {}).get('native_position_hold_forecast')=='th14-ex-position-hold-v1'):
                # 4169ed..416a74 controls a different EX timer: position is
                # held while it counts down. Preserve the timer and its speed
                # source, without treating the following EX command as known.
                remaining,clock,rate_pointer=struct.unpack('<ifI',memory.read(address+0x12ec,12))
                rate=(struct.unpack('<f',memory.read(rate_pointer,4))[0] if rate_pointer else 1.)
                _finite((clock,rate), 'bullet EX position-hold countdown')
                result.update(native_ex_hold_remaining=remaining,native_ex_hold_time=clock,
                              native_ex_hold_rate=rate,native_ex_hold_forecast_enabled=True)
            countdown=result.get('native_ex_hold_remaining') if active_flags==0x04000000 else result.get('native_ex_wait_remaining')
            if (active_flags in (0x04000000,0x80000000) and countdown is not None and countdown<=24
                    and state==1 and flags&0x10 and vx==vy==speed==0.):
                from .native_bullet_pending import read_pending_metadata
                result.update(read_pending_metadata(memory,address,index))
        return result

    def _laser_metadata(self, laser):
        address, memory = laser.id, self.mem
        header = memory.read(address + 0x10, 16)
        state, kind = struct.unpack_from('<II', header)
        timer = struct.unpack_from('<i', header, 12)[0]
        geometry = struct.unpack('<11f', memory.read(address + 0x54, 44))
        iframes = struct.unpack('<I', memory.read(address + 0x5b4, 4))[0]
        x, y, _, vx, vy, _, angle, length, width, speed, distance = geometry
        _finite((x, y, angle, length, width, speed), 'laser')
        _same(kind, laser.kind, 'laser type')
        _same((x, y, angle, length, width, speed),
              (laser.x, laser.y, laser.angle, laser.length, laser.width, laser.speed),
              'laser geometry')
        _same(timer & 0xffffffff, laser.age, 'laser timer')
        _same(width > 0 and length > 0 and iframes == 0, laser.active, 'legacy laser active')
        velocity, lifecycle = None, None
        if kind == 0:
            raw = memory.read(address + 0x5d0, 12)
            maximum = struct.unpack_from('<f', raw)[0]
            distance_limit = struct.unpack_from('<f', raw, 8)[0]
            _finite((vx, vy, distance, maximum, distance_limit), 'line laser lifecycle')
            velocity = [vx, vy]
            lifecycle = {'maximum_length': maximum, 'distance_limit': distance_limit,
                         'travelled_distance': distance}
        return {
            'id': address, 'native_state': state, 'native_type': kind,
            'native_timer': timer, 'native_iframes': iframes,
            'native_position': [x, y], 'native_angle': angle,
            'native_length': length, 'native_width': width, 'native_speed': speed,
            # Meanings of these derived-class fields are verified only for 0.
            'native_velocity': velocity, 'line_lifecycle': lifecycle,
        }

    def _extra_metadata(self, snapshot, bullets, lasers):
        """Opt-in extensions are extracted inside the same outer frame guard."""
        return {}

    def read(self, retries=6):
        if isinstance(retries, bool) or not isinstance(retries, int) or not 1 <= retries <= 32:
            raise ValueError('Raw hazard read retries must be an integer in [1,32]')
        last_error = None
        for _ in range(retries):
            try:
                # This outer guard must enclose the ENTIRE inherited resource
                # read, including its early return when no player is initialized.
                before = self.global_frame()
                snapshot = super().read(retries=1)
                _same(snapshot.frame, before, 'inherited snapshot frame')
                if not snapshot.consistent:
                    raise NativeError('Inherited resource snapshot is inconsistent')
                observed = snapshot.player is not None
                if observed:
                    _entity_ids(snapshot.bullets, 8192, 'bullet')
                    _entity_ids(snapshot.lasers, 2048, 'laser')
                    bullets = [self._bullet_metadata(bullet) for bullet in snapshot.bullets]
                    lasers = [self._laser_metadata(laser) for laser in snapshot.lasers]
                else:
                    bullets, lasers = [], []
                extensions = self._extra_metadata(snapshot, bullets, lasers)
                after = self.global_frame()
                _same(after, before, 'outer global frame')
                metadata = {
                    'schema': HAZARD_SCHEMA, 'frame': before, 'observed': observed,
                    'exe_sha256': self.digest,
                    'consistency': 'global-frame-guard-and-shared-field-checks',
                    'bullets': bullets, 'lasers': lasers,
                }
                if not observed:
                    metadata['unobserved_reason'] = 'no-initialized-player'
                return replace(snapshot, extra={**snapshot.extra, 'hazard_metadata': metadata,
                                                **extensions})
            except (NativeError, ValueError, struct.error) as error:
                last_error = error
        raise NativeError(f'Cannot read coherent raw hazards; no partial metadata returned: {last_error}')


class GeometryHazardReader(VersionedHazardReader):
    """Explicit V5 teacher geometry, preserving every V4 sensor value.

    The original helper at 416d70 gates scaling with flags0x40 and selects
    circles with flags0x10.  Rectangles require the second dimension at be4.
    Player bounds and the circular helper's definition radius are independent
    native fields; neither can be reconstructed from the legacy radius alone.
    This extension is deliberately ignored by the existing observation encoder.
    """

    def __init__(self, memory, profile, *, inspect_unverified=False):
        super().__init__(memory,profile,inspect_unverified=inspect_unverified)
        expected = {'zPlayer_pos':0x5e0,'zPlayer_hit_rad':0x648,
                    'zPlayer_state':0x684,'zPlayer_flags':0x182d4}
        if (any(self.o.get('player',{}).get(k)!=v for k,v in expected.items())
                or self.o.get('game_specific',{}).get('zPlayer_scale')!=0x18308
                or not (profile.hooks or {}).get('player_scale_verified')):
            raise NativeError('Unsupported native player collision geometry layout')

    def _extra_metadata(self, snapshot, bullets, lasers):
        result = {'schema': COLLISION_GEOMETRY_SCHEMA, 'frame': snapshot.frame,
                  'exe_sha256': self.digest, 'observed': snapshot.player is not None,
                  'player': None, 'bullets': []}
        if snapshot.player is None:
            result['unobserved_reason'] = 'no-initialized-player'
            return {'native_collision_geometry': result}
        address = self.pointer('player', 'player_pointer')
        memory = self.mem
        # Native rectangle bounds are [lower.x,lower.y,lower.z,upper.x,
        # upper.y,upper.z], populated by 44e227..44e315 using dynamic scale.
        bounds = struct.unpack('<6f', memory.read(address + 0x630, 24))
        base_radius = memory.floating(address + 0x648)
        scale = memory.floating(address + 0x18308)
        flags = memory.integer(address + 0x182d4)
        definition = memory.integer(address + 0x688)
        definition_radius = memory.floating(definition + 4)
        _finite((*bounds, base_radius, scale, definition_radius), 'player collision geometry')
        _same(base_radius, snapshot.extra.get('base_player_radius'), 'player base radius')
        _same(scale, snapshot.extra.get('player_scale'), 'player scale')
        _same(flags, snapshot.extra.get('resources', {}).get('player_flags'), 'player flags')
        _same(base_radius * scale, snapshot.player.radius, 'legacy scaled player radius')
        result['player'] = {'native_bounds_min': list(bounds[:2]),
            'native_bounds_max': list(bounds[3:5]), 'native_scale': scale,
            'native_base_radius': base_radius, 'native_definition_radius': definition_radius,
            'native_flags': flags}
        for row in bullets:
            # Circle helper does not consume be4.  Do not expose an unverified
            # unused slot (which may contain a non-finite value) as a dimension.
            height = None
            if not row['native_flags'] & 0x10:
                width, height = memory.pair(row['id'] + 0xbe0)
                _finite((width, height), 'bullet rectangle dimensions')
                _same(width, row['native_hitbox_value'], 'bullet rectangle width')
            result['bullets'].append({'id': row['id'],
                'native_flags': row['native_flags'], 'native_scale': row['native_scale'],
                'native_hitbox_value': row['native_hitbox_value'],
                'native_height_value': height})
        extra={'native_collision_geometry':result}
        if (self.profile.hooks or {}).get('native_wall_reactive_forecast')=='th14-distance-reactive-orbs-v1':
            from .native_wall_reader import read_reactive_orbs
            extra['native_reactive_orbs']=read_reactive_orbs(memory,snapshot,bullets)
        if (self.profile.hooks or {}).get('native_focus_gate')=='th14-native-focus-v1':
            from .native_focus_gate import read_native_focus_gate
            extra['native_focus_gate']=read_native_focus_gate(memory)
        if (self.profile.hooks or {}).get('native_curve_laser_geometry')=='th14-native-curve-v1':
            from .native_curve_laser_reader import read_native_curve_lasers
            extra['native_curve_lasers']=read_native_curve_lasers(memory,snapshot,player_address=address,offsets=self.o)
            from .native_line_laser_reader import read_native_line_lasers
            extra['native_line_lasers']=read_native_line_lasers(memory,snapshot,offsets=self.o,player_address=address)
        return extra
