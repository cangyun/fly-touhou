"""Measured collision dimensions appended to the exact legacy V3 sensor.

No future motion, action scoring or controller output is encoded. Dimensions
interpret the verified original collision gates, including rectangle height.
"""
from dataclasses import asdict
import hashlib
import json

import numpy as np

from .hazard_observation import HazardObservationEncoder
from .hazard_reader import COLLISION_GEOMETRY_SCHEMA
from .native_geometry import collision_geometry

GEOMETRY_OBSERVATION_SCHEMA = 'native-collision-geometry-observation-v4'


def _bounded(value, scale=16.):
    return value/(scale+abs(value))


class GeometryObservationEncoder:
    GLOBAL_NAMES = ('geometry_observed', 'player_circle_component', 'player_rectangle_half_width',
        'player_rectangle_half_height', 'player_bounds_center_dx', 'player_bounds_center_dy',
        'large_player_flag', 'native_player_scale', 'player_definition_radius')
    BULLET_NAMES = ('present', 'native_circle', 'native_rectangle', 'collision_scale_enabled',
        'effective_circle_radius', 'effective_rectangle_half_width',
        'effective_rectangle_half_height', 'native_scale_monotonic')

    def __init__(self, config=None, planner_config=None, *, prefix=None):
        if prefix is not None and (config is not None or planner_config is not None):
            raise ValueError('Provide the exact V3 prefix or its construction fields, not both')
        self.prefix = prefix if prefix is not None else HazardObservationEncoder(config, planner_config)
        if type(self.prefix) is not HazardObservationEncoder:
            raise TypeError('Geometry V4 requires the explicit unchanged V3 prefix')
        self.config, self.planner = self.prefix.config, self.prefix.planner

    @property
    def prefix_size(self):
        return self.prefix.size

    @property
    def size(self):
        return self.prefix_size+len(self.GLOBAL_NAMES)+len(self.BULLET_NAMES)*(
            self.config.nearest_bullets+self.config.inactive_bullets)

    @property
    def feature_names(self):
        return (self.prefix.feature_names+self.GLOBAL_NAMES
            +tuple(f'geometry_active_{i}_{n}' for i in range(self.config.nearest_bullets) for n in self.BULLET_NAMES)
            +tuple(f'geometry_inactive_{i}_{n}' for i in range(self.config.inactive_bullets) for n in self.BULLET_NAMES))

    def specification(self):
        return {'schema':GEOMETRY_OBSERVATION_SCHEMA, 'prefix':self.prefix.specification(),
            'observation':asdict(self.config), 'size':self.size, 'prefix_size':self.prefix_size,
            'native_collision_geometry_schema':COLLISION_GEOMETRY_SCHEMA,
            'feature_names_sha256':hashlib.sha256(json.dumps(self.feature_names,separators=(',',':')).encode()).hexdigest(),
            'input_kind':'current measured native dimensions; exact V3 prefix with appended geometry',
            'engineered_prediction_features':False, 'teacher_action_calls':0,
            'geometry_normalization':'v/(16+abs(v)); native scales use v/(1+abs(v)); no hard scale4 saturation',
            'sorting':'same active/inactive distance-then-id slots as V3; no new forecast ordering',
            'future_scale_or_scripts_observed':False,
            'coverage_limit':'same finite nearest-object slots as V3; no claim that all hazards fit the input'}

    @classmethod
    def from_specification(cls, specification):
        result=cls(prefix=HazardObservationEncoder.from_specification(specification['prefix']))
        if result.specification()!=specification:
            raise ValueError('Geometry observation schema differs from the installed measured encoder')
        return result

    def encode(self, snapshot):
        base=self.prefix.encode(snapshot)
        bullets,_=self.prefix._metadata(snapshot)
        shapes,player=collision_geometry(snapshot,bullets)
        if snapshot.player is None:
            return np.concatenate((base,np.zeros(self.size-self.prefix_size,dtype=np.float32)))
        native=snapshot.extra['native_collision_geometry']['player']
        extra=[1.,_bounded(player['circle_component']),
            *[_bounded(float(v)) for v in player['bounds_half_extents']],
            *[_bounded(float(v)) for v in player['bounds_offset']],
            float(player['large_player_flag']),_bounded(native['native_scale'],1.),
            _bounded(native['native_definition_radius'])]
        active=sorted((b for b in snapshot.bullets if b.active and not b.intangible),
            key=lambda b:((b.x-snapshot.player.x)**2+(b.y-snapshot.player.y)**2,b.id))
        inactive=sorted((b for b in snapshot.bullets if not b.active),
            key=lambda b:((b.x-snapshot.player.x)**2+(b.y-snapshot.player.y)**2,b.id))
        geometry={row['id']:row for row in snapshot.extra['native_collision_geometry']['bullets']}
        for values,count in ((active,self.config.nearest_bullets),(inactive,self.config.inactive_bullets)):
            for index in range(count):
                if index>=len(values):
                    extra.extend([0.]*len(self.BULLET_NAMES))
                    continue
                row=geometry[values[index].id]
                shape=shapes[values[index].id]
                circle=shape['shape']=='circle'
                width=height=0.
                if not circle:
                    width,height=row['native_hitbox_value'],row['native_height_value']
                    if shape['scale_applies']:
                        width=float(np.float32(np.float32(width)*np.float32(row['native_scale'])))
                        height=float(np.float32(np.float32(height)*np.float32(row['native_scale'])))
                    width,height=width*.5,height*.5
                extra.extend([1.,float(circle),float(not circle),float(shape['scale_applies']),
                    _bounded(shape['bullet_radius']) if circle else 0.,
                    _bounded(width),_bounded(height),_bounded(row['native_scale'],1.)])
        result=np.concatenate((base,np.asarray(extra,dtype=np.float32)))
        if result.shape!=(self.size,) or not np.isfinite(result).all() or np.any(np.abs(result)>1.):
            raise ValueError('Invalid measured V4 geometry input')
        return result
