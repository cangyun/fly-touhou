"""Versioned measured hazard inputs; the complete v2 vector is an exact prefix."""
from dataclasses import asdict, dataclass
import hashlib
import json
import math

import numpy as np

from .hazard_reader import HAZARD_SCHEMA, FULL_GAME_SHA256
from .planner import PlannerConfig, point_segment_distance
from .resource_observation import ResourceObservationConfig, ResourceObservationEncoder
from .resources import RESOURCE_SCHEMA

OBSERVATION_SCHEMA = 'native-resource-hazard-observation-v3'


@dataclass(frozen=True)
class HazardObservationConfig(ResourceObservationConfig):
    inactive_bullets: int = 24
    hazard_lasers: int = 8

    def __post_init__(self):
        super().__post_init__()
        for n in (self.inactive_bullets, self.hazard_lasers):
            if isinstance(n, bool) or not isinstance(n, int) or not 0 <= n <= 64:
                raise ValueError('Hazard slot counts must be integers in [0,64]')


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'Non-finite or missing measured {label}')
    return float(value)


def _integer(value, label, lo, hi):
    if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
        raise ValueError(f'Invalid measured integer {label}')
    return value


def _pair(value, label):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f'Missing measured pair {label}')
    return tuple(_number(v, label) for v in value)


class HazardObservationEncoder:
    GLOBAL_NAMES = ('hazards_observed', 'measured_bullet_count', 'legacy_inactive_count',
        'measured_laser_count', 'line_laser_count', 'complete_id_coverage')
    ACTIVE_NAMES = ('present', 'native_state', 'flag_mask_2', 'flags_byte0', 'flags_byte1',
        'flags_byte2', 'flags_byte3', 'signed_timer', 'animation_gate', 'scale',
        'cancel_protection', 'stored_speed')
    INACTIVE_NAMES = ('present', 'dx', 'dy', 'stored_vx', 'stored_vy', 'half_stored_vx',
        'half_stored_vy', 'radius', 'distance', 'legacy_active', 'legacy_intangible',
        'kind', 'native_state', 'flag_mask_2', 'flags_byte0', 'flags_byte1', 'flags_byte2',
        'flags_byte3', 'signed_timer', 'animation_gate', 'scale', 'cancel_protection',
        'stored_speed', 'sin_angle', 'cos_angle')
    LASER_NAMES = ('present', 'dx', 'dy', 'sin_angle', 'cos_angle', 'length', 'width',
        'stored_speed', 'legacy_active', 'kind', 'native_state', 'signed_timer', 'iframes',
        'line_fields_known', 'stored_vx', 'stored_vy', 'maximum_length', 'distance_limit',
        'travelled_distance', 'maximum_minus_current_length')

    def __init__(self, config=None, planner_config=None):
        self.config = config or HazardObservationConfig()
        if not isinstance(self.config, HazardObservationConfig):
            raise TypeError('Use HazardObservationConfig for the explicit v3 sensor')
        base_fields = ResourceObservationConfig.__dataclass_fields__
        self.prefix = ResourceObservationEncoder(ResourceObservationConfig(**{
            k: getattr(self.config, k) for k in base_fields}), planner_config)
        self.planner = self.prefix.planner

    @property
    def size(self):
        return (self.prefix.size + len(self.GLOBAL_NAMES) + self.config.nearest_bullets * len(self.ACTIVE_NAMES)
            + self.config.inactive_bullets * len(self.INACTIVE_NAMES) + self.config.hazard_lasers * len(self.LASER_NAMES))

    @property
    def feature_names(self):
        return (self.prefix.feature_names + self.GLOBAL_NAMES
            + tuple(f'active_raw_{i}_{n}' for i in range(self.config.nearest_bullets) for n in self.ACTIVE_NAMES)
            + tuple(f'inactive_raw_{i}_{n}' for i in range(self.config.inactive_bullets) for n in self.INACTIVE_NAMES)
            + tuple(f'laser_raw_{i}_{n}' for i in range(self.config.hazard_lasers) for n in self.LASER_NAMES))

    def specification(self):
        return {'schema': OBSERVATION_SCHEMA, 'observation': asdict(self.config),
            'planner': asdict(self.planner.config), 'size': self.size,
            'prefix': self.prefix.specification(), 'resource_schema': RESOURCE_SCHEMA,
            'hazard_metadata_schema': HAZARD_SCHEMA, 'engineered_prediction_features': False,
            'input_kind': 'measured-native-state-resources-and-raw-hazards',
            'feature_names_sha256': hashlib.sha256(json.dumps(self.feature_names, separators=(',', ':')).encode()).hexdigest(),
            'sorting': 'active slots match v2 distance then id; inactive distance then id; all lasers current segment distance then id',
            'normalization': {'position': [384, 448], 'velocity': 16, 'radius': 16, 'distance': 600,
                'state': 5, 'bullet_kind': 64, 'flags_byte': 255, 'signed_timer': 600,
                'animation_gate': 60, 'scale': 4, 'cancel_counter': 120,
                'laser_length': 448, 'laser_width': 32, 'laser_distance': 1024,
                'laser_kind': 3, 'final_clip': [-1, 1]},
            'half_velocity_semantics': '0.5 times measured stored velocity is a candidate, not an activation or collision oracle',
            'non_line_laser_fields': 'known mask 0 with zero padding; live metadata may never be omitted'}

    @classmethod
    def from_specification(cls, specification):
        result = cls(HazardObservationConfig(**specification['observation']), PlannerConfig(**specification['planner']))
        if result.specification() != specification:
            raise ValueError('Recorded hazard observation schema differs from the installed encoder')
        return result

    def _metadata(self, snapshot):
        metadata = snapshot.extra.get('hazard_metadata')
        if not isinstance(metadata, dict) or metadata.get('schema') != HAZARD_SCHEMA:
            raise ValueError('Missing measured hazard metadata or wrong hazard schema')
        if metadata.get('frame') != snapshot.frame or metadata.get('exe_sha256') != FULL_GAME_SHA256:
            raise ValueError('Hazard metadata frame/executable identity differs')
        if metadata.get('observed') is not (snapshot.player is not None):
            raise ValueError('Hazard observed flag differs from initialized player state')
        maps = []
        for name, entities in (('bullets', snapshot.bullets), ('lasers', snapshot.lasers)):
            rows = metadata.get(name)
            if not isinstance(rows, list) or any(not isinstance(row, dict) or 'id' not in row for row in rows):
                raise ValueError(f'Missing measured hazard {name}')
            ids = [row['id'] for row in rows]
            if any(isinstance(v, bool) or not isinstance(v, int) for v in ids) or len(set(ids)) != len(ids):
                raise ValueError('Invalid or duplicate hazard object IDs')
            expected = {e.id for e in entities} if snapshot.player else set()
            if set(ids) != expected:
                raise ValueError('Measured hazard IDs do not cover the same snapshot')
            maps.append(dict(zip(ids, rows)))
        return maps

    @staticmethod
    def _bullet(bullet, row):
        state = _integer(row.get('native_state'), 'bullet state', 0, 65535)
        flags = _integer(row.get('native_flags'), 'bullet flags', 0, 2**32-1)
        timer = _integer(row.get('native_timer'), 'bullet timer', -2**31, 2**31-1)
        cancel = _integer(row.get('native_cancel_protection_counter'), 'cancel counter', 0, 2**32-1)
        gate = _integer(row.get('native_animation_gate'), 'animation gate', -2**31, 2**31-1)
        scale = _number(row.get('native_scale'), 'bullet scale')
        speed = _number(row.get('native_speed'), 'bullet stored speed')
        angle = _number(row.get('native_angle'), 'bullet stored angle')
        if (_pair(row.get('native_position'), 'bullet position') != (bullet.x, bullet.y)
                or _pair(row.get('native_velocity'), 'bullet velocity') != (bullet.vx, bullet.vy)
                or _number(row.get('native_hitbox_value'), 'bullet radius') != bullet.radius
                or (state == 1) != bullet.active or timer & 0xffffffff != bullet.age
                or cancel != bullet.invulnerable_frames):
            raise ValueError('Measured bullet overlap differs from the same-frame snapshot')
        byte_features = [float(bool(flags & 2)), *[((flags >> shift) & 255) / 255 for shift in (0, 8, 16, 24)]]
        return [1, state/5, *byte_features, timer/600, gate/60, scale/4, cancel/120, speed/16], angle

    @staticmethod
    def _laser(laser, row, player):
        state = _integer(row.get('native_state'), 'laser state', 0, 2**32-1)
        kind = _integer(row.get('native_type'), 'laser type', 0, 2**32-1)
        timer = _integer(row.get('native_timer'), 'laser timer', -2**31, 2**31-1)
        iframes = _integer(row.get('native_iframes'), 'laser iframes', 0, 2**32-1)
        angle, length, width, speed = [_number(row.get('native_'+name), 'laser '+name)
                                      for name in ('angle', 'length', 'width', 'speed')]
        if (_pair(row.get('native_position'), 'laser position') != (laser.x, laser.y)
                or (kind, angle, length, width, speed) != (laser.kind, laser.angle, laser.length, laser.width, laser.speed)
                or (timer & 0xffffffff) != laser.age
                or (width > 0 and length > 0 and iframes == 0) != laser.active):
            raise ValueError('Measured laser overlap differs from the same-frame snapshot')
        if kind == 0:
            vx, vy = _pair(row.get('native_velocity'), 'line laser velocity')
            lifecycle = row.get('line_lifecycle')
            if not isinstance(lifecycle, dict):
                raise ValueError('Missing measured line laser lifecycle')
            maximum, limit, travelled = [_number(lifecycle.get(name), 'line '+name)
                for name in ('maximum_length', 'distance_limit', 'travelled_distance')]
            line = [1, vx/16, vy/16, maximum/448, limit/1024, travelled/1024, (maximum-length)/448]
        else:
            if row.get('native_velocity', 'missing') is not None or row.get('line_lifecycle', 'missing') is not None:
                raise ValueError('Unverified non-line laser fields must be explicit None')
            line = [0]*7
        return [1, (laser.x-player.x)/384, (laser.y-player.y)/448, math.sin(angle), math.cos(angle),
            length/448, width/32, speed/16, float(laser.active), kind/3, state/5, timer/600, iframes/120, *line]

    @staticmethod
    def _laser_distance(laser, player):
        origin = np.asarray((player.x, player.y))
        if laser.nodes:
            nodes = [np.asarray(v) for v in laser.nodes]
            segments = zip(nodes[:-1], nodes[1:]) if len(nodes) > 1 else [(nodes[0], nodes[0])]
            return min(point_segment_distance(origin, a, b) for a, b in segments)
        start = np.asarray((laser.x, laser.y))
        end = start + laser.length * np.asarray((math.cos(laser.angle), math.sin(laser.angle)))
        return point_segment_distance(origin, start, end)

    def encode(self, snapshot):
        snapshot.validate()
        bullets, lasers = self._metadata(snapshot)
        if snapshot.player is None:
            return np.zeros(self.size, dtype=np.float32)
        player = snapshot.player
        raw = {b.id: self._bullet(b, bullets[b.id]) for b in snapshot.bullets}
        laser_features = {l.id: self._laser(l, lasers[l.id], player) for l in snapshot.lasers}
        active = sorted((b for b in snapshot.bullets if b.active and not b.intangible),
            key=lambda b: ((b.x-player.x)**2+(b.y-player.y)**2, b.id))
        inactive = sorted((b for b in snapshot.bullets if not b.active),
            key=lambda b: ((b.x-player.x)**2+(b.y-player.y)**2, b.id))
        ordered_lasers = sorted(snapshot.lasers, key=lambda l: (self._laser_distance(l, player), l.id))
        extra = [1, len(bullets)/2048, len(inactive)/2048, len(lasers)/128,
                 sum(l.kind == 0 for l in snapshot.lasers)/128, 1]
        for i in range(self.config.nearest_bullets):
            extra.extend(raw[active[i].id][0] if i < len(active) else [0]*len(self.ACTIVE_NAMES))
        for i in range(self.config.inactive_bullets):
            if i >= len(inactive):
                extra.extend([0]*len(self.INACTIVE_NAMES))
                continue
            b = inactive[i]
            fields, angle = raw[b.id]
            dx, dy = b.x-player.x, b.y-player.y
            extra.extend([1, dx/384, dy/448, b.vx/16, b.vy/16, b.vx/32, b.vy/32,
                b.radius/16, math.hypot(dx, dy)/600, float(b.active), float(b.intangible), b.kind/64,
                *fields[1:7], *fields[7:], math.sin(angle), math.cos(angle)])
        for i in range(self.config.hazard_lasers):
            extra.extend(laser_features[ordered_lasers[i].id] if i < len(ordered_lasers) else [0]*len(self.LASER_NAMES))
        extra = np.asarray(extra, dtype=np.float32)
        if extra.shape != (self.size-self.prefix.size,) or not np.isfinite(extra).all():
            raise ValueError('Hazard feature shape/finite-value invariant failed')
        return np.concatenate((self.prefix.encode(snapshot), np.clip(extra, -1, 1)))
