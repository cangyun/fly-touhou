"""Optional egocentric CNN features derived solely from the existing raw1911.

This does not read additional native objects, predict paths or score actions.
The original vector also enters a residual branch: clipping, the object budget
and distant information stay exactly those of the original list observation.
"""
from __future__ import annotations

import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class SpatialNativeFeatures(BaseFeaturesExtractor):
    CHANNELS = ('active_circle', 'active_rectangle', 'velocity_x', 'velocity_y',
                'inactive_shape', 'contactable_enemy', 'active_laser_segment', 'outside_playfield')

    def __init__(self, observation_space, *, feature_names, bounds, features_dim=128,
                 grid_size=48, half_span=96.):
        if (observation_space.shape not in ((1911,), (1912,)) or features_dim != 128
                or observation_space.shape != (len(feature_names),)):
            raise ValueError('Require raw1911, optional task clock, and128 recurrent features')
        if grid_size != 48 or half_span != 96.:
            raise ValueError('Use the declared48x48 crop of192x192 game units')
        super().__init__(observation_space, features_dim)
        self.feature_names = tuple(feature_names)
        index = {name: i for i, name in enumerate(feature_names)}
        if len(index) != len(feature_names) or (len(index) == 1912 and feature_names[-1] != 'task_progress'):
            raise ValueError('Require unique native feature names and optional trailing task_progress')
        self.grid_size, self.half_span = grid_size, float(half_span)
        self.cell = 2*self.half_span/grid_size
        self.bounds = tuple(float(v) for v in bounds)

        def indices(label, prefix, count, fields):
            values = [[index[f'{prefix}_{i}_{field}'] for field in fields] for i in range(count)]
            self.register_buffer(label, torch.tensor(values, dtype=torch.long))

        indices('active_index', 'bullet', 24, ('present', 'dx', 'dy', 'vx', 'vy'))
        indices('inactive_index', 'inactive_raw', 24, ('present', 'dx', 'dy'))
        geometry = ('present', 'native_circle', 'native_rectangle', 'effective_circle_radius',
                    'effective_rectangle_half_width', 'effective_rectangle_half_height')
        indices('active_geometry_index', 'geometry_active', 24, geometry)
        indices('inactive_geometry_index', 'geometry_inactive', 24, geometry)
        indices('enemy_index', 'enemy', 4,
                ('present', 'dx', 'dy', 'contactable', 'hitbox_x', 'hitbox_y'))
        # laser_segments() already filters inactive lasers in this legacy list.
        # The raw-laser list includes inactive objects, so its slot numbers must
        # never be used to look up activity for this active-segment list.
        indices('laser_index', 'laser', 8, ('present', 'ax', 'ay', 'bx', 'by', 'width'))
        self.register_buffer('player_index', torch.tensor([index['player_x'], index['player_y']]))
        centers = (torch.arange(grid_size, dtype=torch.float32)+.5)*self.cell-self.half_span
        yy, xx = torch.meshgrid(centers, centers, indexing='ij')
        # meshgrid returns overlapping broadcast views. Dense storage makes a
        # CPU state_dict round-trip safe as well as the CUDA path.
        self.register_buffer('xx', xx[None, None].clone())
        self.register_buffer('yy', yy[None, None].clone())
        self.raw = nn.Sequential(nn.Linear(len(feature_names), 64), nn.Tanh())
        self.cnn = nn.Sequential(
            nn.Conv2d(len(self.CHANNELS), 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(32*6*6, 64), nn.Tanh())

    @staticmethod
    def _native_dimension(value):
        # Inverse of the existing v/(16+abs(v)), not an additional measurement.
        return 16*value/(1-value).clamp_min(1e-6)

    def _bullet_masks(self, observation, position_index, geometry_index):
        positions = observation[:, position_index]
        geometry = observation[:, geometry_index]
        x = positions[:, :, 1, None, None]*384
        y = positions[:, :, 2, None, None]*448
        dx, dy = self.xx-x, self.yy-y
        valid = (positions[:, :, 0]*geometry[:, :, 0])[:, :, None, None]
        radius = self._native_dimension(geometry[:, :, 3])[:, :, None, None]
        width = self._native_dimension(geometry[:, :, 4])[:, :, None, None]
        height = self._native_dimension(geometry[:, :, 5])[:, :, None, None]
        circle = (.5+(radius-torch.sqrt(dx.square()+dy.square()+1e-12))/self.cell).clamp(0, 1)
        rectangle = (.5+torch.minimum(width-dx.abs(), height-dy.abs())/self.cell).clamp(0, 1)
        circle = circle*valid*geometry[:, :, 1, None, None]
        rectangle = rectangle*valid*geometry[:, :, 2, None, None]
        return circle, rectangle

    def rasterize(self, observation):
        circles, rectangles = self._bullet_masks(observation, self.active_index, self.active_geometry_index)
        active = torch.maximum(circles, rectangles)
        coverage = active.sum(1).clamp_min(1e-6)
        active_fields = observation[:, self.active_index]
        vx = (active*active_fields[:, :, 3, None, None]).sum(1)/coverage
        vy = (active*active_fields[:, :, 4, None, None]).sum(1)/coverage
        inactive = self._bullet_masks(observation, self.inactive_index, self.inactive_geometry_index)
        inactive_occupancy = torch.maximum(*inactive).amax(1)

        enemy = observation[:, self.enemy_index]
        ex = self.xx-enemy[:, :, 1, None, None]*384
        ey = self.yy-enemy[:, :, 2, None, None]*448
        rx = (enemy[:, :, 4, None, None]*128).clamp_min(.01)
        ry = (enemy[:, :, 5, None, None]*128).clamp_min(.01)
        # A display of current measured enemy extents, not a collision oracle.
        enemy_shape = (.5+torch.minimum(rx-ex.abs(), ry-ey.abs())/self.cell).clamp(0, 1)
        enemy_shape = enemy_shape*(enemy[:, :, 0]*enemy[:, :, 3])[:, :, None, None]

        laser = observation[:, self.laser_index]
        ax, ay = laser[:, :, 1, None, None]*384, laser[:, :, 2, None, None]*448
        bx, by = laser[:, :, 3, None, None]*384, laser[:, :, 4, None, None]*448
        ux, uy = bx-ax, by-ay
        parameter = (((self.xx-ax)*ux+(self.yy-ay)*uy)/(ux.square()+uy.square()).clamp_min(1e-6)).clamp(0, 1)
        distance = torch.sqrt((self.xx-ax-parameter*ux).square()+(self.yy-ay-parameter*uy).square()+1e-12)
        laser_shape = (.5+(laser[:, :, 5, None, None]*16-distance)/self.cell).clamp(0, 1)
        laser_shape = laser_shape*laser[:, :, 0, None, None]

        player = observation[:, self.player_index]
        absolute_x = self.xx[:, 0]+player[:, 0, None, None]*192
        absolute_y = self.yy[:, 0]+(player[:, 1, None, None]+1)*224
        left, right, top, bottom = self.bounds
        outside = ((absolute_x < left) | (absolute_x > right)
                   | (absolute_y < top) | (absolute_y > bottom)).to(observation.dtype)
        return torch.stack((circles.amax(1), rectangles.amax(1), vx, vy,
                            inactive_occupancy, enemy_shape.amax(1), laser_shape.amax(1), outside), dim=1)

    def forward(self, observations):
        return torch.cat((self.raw(observations), self.cnn(self.rasterize(observations))), dim=1)
