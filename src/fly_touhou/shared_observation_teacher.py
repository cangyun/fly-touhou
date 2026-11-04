"""An avoidance teacher whose only argument is the student's encoded vector.

This deliberately does not accept a native Snapshot. Finite slots, clipping and
float32 quantization are shared with the student. Forecasts extrapolate measured
current motion; neither future native states nor future attack scripts are read.
"""
from __future__ import annotations

from dataclasses import asdict
import math

import numpy as np

from .dynamic_planner import _phase_envelope_distance
from .native_geometry import swept_rectangle_clearances
from .planner import CollisionPlanner, PlannerConfig, segment_distance
from .state import Player, Snapshot, decode_action


SHARED_TEACHER_SCHEMA = 'encoded-current-geometry-avoidance-teacher-v1'


def reimu_a_motion_config():
    # Measured original Reimu-A motion. Legacy encoder planner metadata remains
    # 4.0 / visible-field bounds solely to preserve its normalization contract.
    return PlannerConfig(horizon_frames=18., normal_speed=4.5, focused_speed=2.,
                         left=-184., right=184., top=32., bottom=432.)


def unbounded(value, scale=16.):
    """Invert V4's monotonic geometry transform, including signed offsets."""
    value = float(value)
    if abs(value) >= 1:
        raise ValueError('Monotonic geometry field cannot equal a clipped endpoint')
    return scale * value / (1. - abs(value))


class SharedObservationTeacher:
    def __init__(self, feature_names, planner_config=None):
        self.feature_names = tuple(feature_names)
        self.index = {name: i for i, name in enumerate(self.feature_names)}
        if len(self.index) != len(self.feature_names):
            raise ValueError('Duplicate student feature names')
        self.config = planner_config or reimu_a_motion_config()
        self.paths = CollisionPlanner(self.config)
        self.active_slots = sum(name.endswith('_present') and name.startswith('geometry_active_')
                                for name in self.feature_names)
        self.inactive_slots = sum(name.endswith('_present') and name.startswith('geometry_inactive_')
                                  for name in self.feature_names)
        self.laser_slots = sum(name.endswith('_present') and name.startswith('laser_raw_')
                               for name in self.feature_names)
        self.enemy_slots = sum(name.endswith('_present') and name.startswith('enemy_')
                               for name in self.feature_names)
        for name in ('player_present', 'player_x', 'player_y', 'player_circle_component',
                     'player_rectangle_half_width', 'player_rectangle_half_height'):
            if name not in self.index:
                raise ValueError('Measured student geometry protocol required: '+name)

    def reset(self):
        # This teacher is memoryless. GRU and teacher receive the same present
        # observation; the teacher does not retain unencoded native object IDs.
        self.paths.reset()

    def specification(self):
        return {'schema': SHARED_TEACHER_SCHEMA, 'input_size': len(self.feature_names),
            'input_only': 'student float32 observation vector', 'planner': asdict(self.config),
            'snapshot_or_native_process_access': False, 'future_native_trajectory_access': False,
            'motion_calibration': 'native-motion-calibration-v1 report sha256 10ead065bc9082daf96d731f6f2acd0cc7f66a916cac38dd2e9e727db22135cd',
            'motion_and_legacy_observation_normalization_are_distinct': True,
            'history': 'memoryless; no extra history unavailable to the student',
            'action_frames': 4, 'actions': list(range(18)), 'bomb': False, 'shot_held': True,
            'geometry': 'V4 monotonic circle and rectangle dimensions; present native gates',
            'state2': 'conservative half/full velocity envelope; future activation unknown',
            'laser': 'encoded current segment plus raw straight-laser growth/translation envelope',
            'nonline_laser_limit': 'only encoded nearest segment and raw straight approximation; no hidden polyline',
            'selection': 'threat: maximal-clearance band 0.5; quiet: >=5 clearance, lower-interior preference',
            'coverage': {'active': self.active_slots, 'inactive': self.inactive_slots,
                         'raw_lasers': self.laser_slots, 'enemies': self.enemy_slots},
            'teacher_is_native_collision_oracle': False}

    def decode(self, observation):
        vector = np.asarray(observation, dtype=np.float32)
        if vector.shape != (len(self.feature_names),) or not np.isfinite(vector).all():
            raise ValueError('Finite current student observation required')
        def f(name, default=0.):
            return float(vector[self.index[name]]) if name in self.index else default
        position = np.asarray((f('player_x')*192., (f('player_y')+1.)*224.))
        player_component = unbounded(f('player_circle_component'))
        player_half = np.asarray((unbounded(f('player_rectangle_half_width')),
                                  unbounded(f('player_rectangle_half_height'))))
        offset = np.asarray((unbounded(f('player_bounds_center_dx')),
                             unbounded(f('player_bounds_center_dy'))))
        circles, rectangles, growth, lasers, envelopes, contacts = [], [], [], [], [], []
        for active, count in ((True, self.active_slots), (False, self.inactive_slots)):
            for i in range(count):
                geo = f'geometry_{"active" if active else "inactive"}_{i}_'
                if not f(geo+'present'):
                    continue
                raw = f'active_raw_{i}_' if active else f'inactive_raw_{i}_'
                base = f'bullet_{i}_' if active else raw
                state = int(round(f(raw+'native_state')*5))
                if not f(raw+'flag_mask_2') or state not in (1, 2):
                    continue
                origin = position+np.asarray((f(base+'dx')*384, f(base+'dy')*448))
                velocity = np.asarray((f(base+('vx' if active else 'stored_vx'))*16,
                                       f(base+('vy' if active else 'stored_vy'))*16))
                circle = bool(f(geo+'native_circle'))
                radius = math.hypot(unbounded(f(geo+'effective_circle_radius')), player_component)
                half = player_half+np.asarray((unbounded(f(geo+'effective_rectangle_half_width')),
                                               unbounded(f(geo+'effective_rectangle_half_height'))))
                if state == 2:
                    speed = float(np.linalg.norm(velocity))
                    direction = velocity/speed if speed else np.asarray((1., 0.))
                    growth.append((origin, velocity*.5, direction, speed*.5,
                                   radius if circle else float(np.linalg.norm(half)),
                                   np.zeros(2) if circle else offset))
                elif circle:
                    circles.append((origin, velocity, radius))
                else:
                    rectangles.append((origin, velocity, half, offset))
        for i in range(self.laser_slots):
            raw = f'laser_raw_{i}_'
            if not f(raw+'present'):
                continue
            origin = position+np.asarray((f(raw+'dx')*384, f(raw+'dy')*448))
            direction = np.asarray((f(raw+'cos_angle'), f(raw+'sin_angle')))
            norm = float(np.linalg.norm(direction))
            direction = direction/norm if norm else np.asarray((1., 0.))
            length, half_width = f(raw+'length')*448, f(raw+'width')*16
            kind = int(round(f(raw+'kind')*3))
            radius = f('player_radius')*16+half_width
            if kind == 0:
                envelopes.append((origin, direction, length, f(raw+'stored_speed')*16, radius))
            else:
                lasers.append((origin, origin+length*direction, radius))
        # The v1 base laser block can carry an encoded nearest curved segment
        # not identical to raw straight geometry. It is also shared information.
        for i in range(self.laser_slots):
            base = f'laser_{i}_'
            if f(base+'present'):
                a = position+np.asarray((f(base+'ax')*384, f(base+'ay')*448))
                b = position+np.asarray((f(base+'bx')*384, f(base+'by')*448))
                lasers.append((a, b, f('player_radius')*16+f(base+'width')*16))
        for i in range(self.enemy_slots):
            base = f'enemy_{i}_'
            if f(base+'present') and f(base+'contactable'):
                origin = position+np.asarray((f(base+'dx')*384, f(base+'dy')*448))
                velocity = np.asarray((f(base+'vx')*16, f(base+'vy')*16))
                # Preserve the existing conservative convention: the measured
                # radius_x/y are treated as half extents, not halved again.
                half = np.asarray((f(base+'hitbox_x')*128, f(base+'hitbox_y')*128))
                half = np.where(half > 0, half, 24.)
                half += f('player_radius')*16
                contacts.append((origin, velocity, half, np.zeros(2)))
        return {'present': bool(f('player_present')), 'position': position,
            'invulnerable_frames': max(0., f('invulnerable_frames')*120),
            'focused': bool(f('focused')), 'circles': circles, 'rectangles': rectangles+contacts,
            'growth': growth, 'lasers': lasers, 'envelopes': envelopes,
            'counts': {'circles': len(circles), 'rectangles': len(rectangles),
                       'growth': len(growth), 'lasers': len(lasers),
                       'laser_envelopes': len(envelopes), 'contacts': len(contacts)}}

    @staticmethod
    def _clearance(decoded, path):
        clearance = math.inf
        circles, boxes = decoded['circles'], decoded['rectangles']
        if circles:
            origins = np.asarray([r[0] for r in circles])
            velocities = np.asarray([r[1] for r in circles])
            radii = np.asarray([r[2] for r in circles])
        if boxes:
            box_origins = np.asarray([r[0] for r in boxes])
            box_velocities = np.asarray([r[1] for r in boxes])
            box_half = np.asarray([r[2] for r in boxes])
            box_offset = np.asarray([r[3] for r in boxes])
        for first, last, intercept, velocity in path:
            first = max(first, decoded['invulnerable_frames'])
            if first > last:
                continue
            if circles:
                delta, motion = intercept-origins, velocity-velocities
                squared = np.einsum('ij,ij->i', motion, motion)
                times = np.divide(-np.einsum('ij,ij->i', delta, motion), squared,
                    out=np.full(len(circles), first), where=squared > 1e-12)
                times = np.clip(times, first, last)
                clearance = min(clearance, float((np.linalg.norm(delta+motion*times[:, None], axis=1)-radii).min()))
            if boxes:
                values = swept_rectangle_clearances(intercept+box_offset-box_origins,
                    velocity-box_velocities, box_half, first, last)
                clearance = min(clearance, float(values.min()))
            for origin, half_motion, direction, half_speed, radius, offset in decoded['growth']:
                clearance = min(clearance, _phase_envelope_distance(intercept+offset,
                    velocity-half_motion, origin, direction, half_speed, half_speed, first, last)-radius)
            for a, b, radius in decoded['lasers']:
                clearance = min(clearance, segment_distance(intercept+velocity*first,
                    intercept+velocity*last, a, b)-radius)
            for origin, direction, length, speed, radius in decoded['envelopes']:
                clearance = min(clearance, _phase_envelope_distance(intercept, velocity,
                    origin, direction, length, speed, first, last)-radius)
        return clearance

    def query(self, observation):
        decoded = self.decode(observation)
        if not decoded['present']:
            return {'action': 9, 'target18': np.eye(18)[9].tolist(), 'clearances18': [None]*18,
                    'mode': 'uninitialized', 'counts': decoded['counts']}
        x, y = decoded['position']
        snapshot = Snapshot(0, 0, Player(float(x), float(y)), consistent=True)
        paths = [self.paths._path(snapshot, a) for a in range(18)]
        c = np.asarray([self._clearance(decoded, path) for path, _, _ in paths])
        best, worst = float(c.max()), float(c.min())
        if math.isinf(best):
            allowed = c >= 5.
        elif best > 0:
            allowed = c >= min(5., best)
        else:
            allowed = c >= best-.5
        spread = 0. if worst == best else best-worst
        threat = worst < 12. and spread > 2.
        candidates = allowed & (c >= best-.5) if threat and math.isfinite(best) else allowed
        utilities = []
        for a in range(18):
            dx, dy, focused = decode_action(a)
            speed = self.config.focused_speed if focused else self.config.normal_speed
            delta = np.asarray((dx, dy))*speed*4/max(1., math.hypot(dx, dy))
            next_position = np.clip(decoded['position']+delta,
                (self.config.left, self.config.top), (self.config.right, self.config.bottom))
            outside = math.hypot(max(0., abs(next_position[0])-56.),
                                 max(0., 352.-next_position[1], next_position[1]-416.))
            utilities.append(-outside/18.-.02*float(np.linalg.norm(delta))-.1*(not focused))
        utilities = np.asarray(utilities)
        action = int(np.flatnonzero(candidates)[np.argmax(utilities[candidates])])
        mass = np.exp((utilities[candidates]-utilities[candidates].max())/.03)
        target = np.zeros(18)
        target[candidates] = mass/mass.sum()
        return {'action': action, 'target18': target.tolist(),
            'clearances18': [None if math.isinf(v) else float(v) for v in c],
            'admissible18': np.flatnonzero(allowed).tolist(),
            'mode': 'maximum_margin' if threat else 'quiet_lower_interior',
            'counts': decoded['counts'], 'teacher_snapshot_access': False,
            'future_native_trajectory_access': False}
