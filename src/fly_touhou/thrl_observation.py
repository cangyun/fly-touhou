# SPDX-License-Identifier: AGPL-3.0-or-later
# Derived from thrl schema1 observations.
# Copyright (C) 2026 T. Liu (touhourl@proton.me)
# TH14/NumPy adaptation Copyright (C) 2026 fly-touhou contributors.
#
# This program is free software: you may redistribute and/or modify it under
# the GNU Affero General Public License as published by the Free Software
# Foundation, version 3 or (at your option) any later version. This program is
# distributed WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See COPYING and the
# upstream full license at third_party/rl-reference-20261020/thrl/LICENSE;
# https://www.gnu.org/licenses/agpl-3.0.html
"""thrl's 273 scalars + 16-channel full-field map, explicitly adapted to TH14.

Only current, complete native Snapshots are accepted. No torch, attention,
teacher, memory reads, forecast, old24-slot observation or native actions.
The first port retains upstream's fixed map footprints, NOT GeometryV4 sizes.
"""
from __future__ import annotations

import math

import numpy as np

from .state import Snapshot

SOURCE_COMMIT = '41c4f77931a541849a31411bf766f93bbad5e597'
SCHEMA = 'thrl-schema1-th14-fullfield-v1'
THRL_FEATURE_DIM = 273
THRL_MAP_SHAPE = (16, 92, 96)
RESOURCE_SCHEMA = 'original-th14-resources-v1'
EVENT_SCHEMA = 'th14-thrl-events-v1'
MAP_CHANNELS = tuple(f'{kind}_{field}' for kind in ('bullet', 'enemy', 'projectile')
                     for field in ('occupancy', 'velocity_x', 'velocity_y', 'closest_dist', 'entity_type')) + ('player_dist',)
PLAYER_FIELDS = ('x_norm', 'y_norm', 'vx_norm', 'vy_norm', 'wall_left', 'wall_right',
                 'wall_top', 'wall_bottom', 'power_norm', 'lives_norm', 'bombs_norm',
                 'invincible', 'character_norm', 'stage_norm', 'cfg_lives_norm',
                 'cfg_bombs_norm', 'rank_norm')
BOSS_FIELDS = ('present', 'hp_norm', 'dx_norm', 'dy_norm', 'vx_norm', 'vy_norm', 'dist_norm')
STATE_FIELDS = ('stage_norm', 'graze_norm', 'score_norm', 'miss_count_norm', 'point_items_norm', 'rank_norm')
BULLET_FIELDS = ('dx', 'dy', 'vx', 'vy', 'type_id', 'speed', 'distance')
PROJECTILE_FIELDS = ('dx', 'dy', 'vx', 'vy', 'type_id', 'sub_type', 'distance')


def _clip(value, lo=0., hi=1.):
    return float(np.clip(float(value), lo, hi))


class THRLObservationEncoder:
    """Stateless sensor; one full snapshot in, independent NumPy arrays out.

    Feature Box(-1,1,(273,)); map Box(-1,1,(16,92,96)); both float32.
    Missing player/resources raises, rather than inventing an empty playfield.
    """
    feature_shape = (THRL_FEATURE_DIM,)
    map_shape = THRL_MAP_SHAPE
    feature_names = (tuple('player_'+f for f in PLAYER_FIELDS)
        + tuple(f'boss{index}_'+f for index in (1, 2) for f in BOSS_FIELDS)
        + tuple('state_'+f for f in STATE_FIELDS)
        + tuple(f'projectile_{i}_{f}' for i in range(16) for f in PROJECTILE_FIELDS)
        + tuple(f'bullet_{i}_{f}' for i in range(16) for f in BULLET_FIELDS)
        + tuple(f'drop_{i}_{f}' for i in range(4) for f in ('dx', 'dy', 'type_id')))

    def __init__(self, *, require_events=True):
        if type(require_events) is not bool:
            raise ValueError('require_events must explicitly be a boolean')
        self.require_events = require_events
        self.width, self.height = 96, 92
        self.span_x, self.span_y = 384., 448.
        self.cell_x, self.cell_y = self.span_x/self.width, self.span_y/self.height
        self.diagonal = math.hypot(self.span_x, self.span_y)
        self._xx, self._yy = np.meshgrid((np.arange(self.width)+.5)*self.cell_x,
                                        (np.arange(self.height)+.5)*self.cell_y)

    def specification(self):
        return {'schema': SCHEMA if self.require_events else SCHEMA+'-synthetic-no-events', 'source': {'repository': 'https://github.com/touhourl/thrl',
            'commit': SOURCE_COMMIT, 'license': 'AGPL-3.0-or-later', 'license_text': 'COPYING',
            'upstream_license': 'third_party/rl-reference-20261020/thrl/LICENSE',
            'files': ['src/observation/schema1/'+p+'.rs' for p in ('player', 'boss', 'state', 'map', 'builder')]
                     + ['src/games/th05c/observation.rs']},
            'features': {'shape': [273], 'dtype': 'float32', 'bounds': [-1, 1],
                'layout': {'player': [0, 17], 'boss': [17, 31], 'state': [31, 37],
                           'projectile_top16': [37, 149], 'bullet_top16': [149, 261], 'drop_top4': [261, 273]},
                'names': list(self.feature_names), 'sorting': 'Euclidean distance then native id/sample index for deterministic ties'},
            'maps': {'shape': [16, 92, 96], 'axis_order': 'channels,y,x', 'channels': list(MAP_CHANNELS),
                'dtype': 'float32', 'bounds': [-1, 1], 'full_field': True, 'top_k_before_map': False,
                'projection': 'native x+192, native y; full field x[-192,192],y[0,448]',
                'player_movement_bounds': [-184, 184, 32, 432],
                'cell_size_native_units': [self.cell_x, self.cell_y],
                'footprints': {'normal_bullet_square_width': 8., 'contactable_nonboss_enemy_square_width': 24.,
                               'laser_sample_square_width': 12.},
                'edge_rule': 'upstream clamp-to-border; outside weight=max(0.02,1/(1+excess_units/0.2))',
                'velocity_divisor_per_native_frame': 64., 'occupancy': 'min(weighted_count/4,1)'},
            'mapping': {'player_position_walls': 'TH14 legal movement bounds; not TH05 bounds',
                'power': 'measured TH14 power/400, replacing TH05 power/128',
                'difficulty': 'measured difficulty/3 analogue for both TH05 rank slots; not dynamic rank',
                'character': 'measured TH14 character/2 analogue; no assertion of cross-game character identity',
                'stage': 'measured stage/6', 'boss_roles': 'first two living boss-flag enemies by id; TH05 main/midboss identities unavailable',
                'enemy_velocity': 'same-frame resources.enemies measured vx/vy',
                'bullet_type': 'TH14 kind modulo256 /255 analogue of TH05 sprite/patnum',
                'items': 'full measured resources.items; nearest4; TH14 kind/12 analogue of TH05 item_type/6',
                'special_projectiles': 'TH14 active lasers only; TH05 firewave/cheeto/custom do not map to normal TH14 bullets',
                'laser_geometry': 'current polyline, or current origin+angle+length; no future trajectory',
                'laser_samples': 'map: every16 native units plus endpoint; list: uniform ceil(length/32)+1, max8 per laser, then global nearest16',
                'laser_velocity': 'constant0, matching upstream sampled laser points; not inferred from speed',
                'laser_subtype': 'TH14 kind/3 analogue, not TH05 lifecycle flag/7',
                'misses_and_points': 'required snapshot.extra.thrl_events observed:true schema:'+EVENT_SCHEMA
                    +'; run misses from read-only native loss-commit hooks; native point_items at VA0x4f5848 / point-collection inc VA0x439625, sidecar verified'},
            'constant_zero_unavailable': ['player_vx_norm', 'player_vy_norm', 'player_cfg_lives_norm',
                'player_cfg_bombs_norm', 'enemy_map_type_id'] + ([] if self.require_events else ['state_miss_count_norm', 'state_point_items_norm']),
            'events_policy': 'required_observed_native_counts' if self.require_events else 'synthetic_only_always_zero_even_if_events_present',
            'not_encoded': ['frame', 'stage_frame', 'task_progress', 'native RNG', 'future scripts/trajectories',
                'GeometryV4 variable radii/rectangles', 'inactive bullets', 'teacher action risks'],
            'current_only': True, 'include_action_risk': False, 'teacher_action_calls': 0,
            'missing_input': 'ValueError on missing player or measured resource block; never silently zero a loading state'}

    def _event_counts(self, snapshot):
        if not self.require_events:
            return 0, 0  # Different, explicitly synthetic-only specification.
        events = snapshot.extra.get('thrl_events', {})
        if events.get('schema') != EVENT_SCHEMA or events.get('observed') is not True:
            raise ValueError('THRL production observation requires observed native event counts')
        counts = []
        for name in ('misses', 'bombs_used', 'point_items'):
            value = events.get(name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
                raise ValueError('Invalid or missing observed native event count: '+name)
            counts.append(int(value))
        return counts[0], counts[2]

    @staticmethod
    def _finite(*values):
        if not all(math.isfinite(float(v)) for v in values):
            raise ValueError('Nonfinite THRL observation input')

    def _map(self, entities, px, py):
        """Port of SpatialMap::from_entities_with_mode, absolute default only.

        entities columns: shifted x,y, per-native-frame vx,vy,type,square width.
        Vectorized splats preserve all source entities; no top-K or count cap.
        """
        result = np.zeros((5, self.height, self.width), dtype=np.float64)
        result[3] = 1.
        if not entities:
            return result
        a = np.asarray(entities, dtype=np.float64)
        if a.ndim != 2 or a.shape[1] not in (6, 8) or not np.isfinite(a).all():
            raise ValueError('Invalid measured map entity')
        x, y, vx, vy, kind, size = a[:, :6].T
        height = a[:, 6] if a.shape[1] == 8 else size
        circles = a[:, 7].astype(bool) if a.shape[1] == 8 else np.zeros(len(a), bool)
        excess = np.maximum(np.maximum(-x, x-self.span_x), np.maximum(-y, y-self.span_y)).clip(0)
        weight = np.maximum(.02, 1./(1.+excess/.2))
        half = size*.5
        half_y = height*.5
        gx0 = np.floor((x-half)/self.cell_x).clip(0, self.width-1).astype(np.int64)
        gx1 = (np.ceil((x+half)/self.cell_x)-1).clip(0, self.width-1).astype(np.int64)
        gy0 = np.floor((y-half_y)/self.cell_y).clip(0, self.height-1).astype(np.int64)
        gy1 = (np.ceil((y+half_y)/self.cell_y)-1).clip(0, self.height-1).astype(np.int64)
        point = size <= 0
        gx0[point] = gx1[point] = np.floor(x[point]/self.cell_x).clip(0, self.width-1).astype(np.int64)
        point = height <= 0
        gy0[point] = gy1[point] = np.floor(y[point]/self.cell_y).clip(0, self.height-1).astype(np.int64)
        cells = self.width*self.height
        count = np.zeros(cells,np.float64)
        sums = np.zeros((3,cells),np.float64)
        distance = np.where(circles, np.maximum(np.hypot(x-px, y-py)-half, 0),
            np.hypot(np.maximum(np.abs(x-px)-half, 0), np.maximum(np.abs(y-py)-half_y, 0)))/self.diagonal
        closest = result[3].reshape(-1)
        # Group by footprint extent and cap temporary raster work. One huge
        # stage6 bullet must not allocate N*92*96 temporary cells for all the
        # ordinary bullets in that same frame.
        groups=np.stack((np.ceil(np.log2(gx1-gx0+1)),np.ceil(np.log2(gy1-gy0+1))),axis=1).astype(int)
        for group in np.unique(groups,axis=0):
            members=np.flatnonzero((groups==group).all(axis=1))
            width,height=int((gx1-gx0)[members].max())+1,int((gy1-gy0)[members].max())+1
            batch=max(1,262144//(width*height))
            for start in range(0,len(members),batch):
                selected=members[start:start+batch]
                gx=gx0[selected,None,None]+np.arange(width)[None,None,:]
                gy=gy0[selected,None,None]+np.arange(height)[None,:,None]
                valid=(gx<=gx1[selected,None,None])&(gy<=gy1[selected,None,None])
                if circles[selected].any():
                    dx=np.maximum(np.abs((gx+.5)*self.cell_x-x[selected,None,None])-self.cell_x*.5,0)
                    dy=np.maximum(np.abs((gy+.5)*self.cell_y-y[selected,None,None])-self.cell_y*.5,0)
                    valid &= (~circles[selected,None,None] | (dx*dx+dy*dy<=half[selected,None,None]**2)
                              | (excess[selected,None,None]>0))
                indices=np.broadcast_to(gy*self.width+gx,valid.shape)[valid]
                source=np.broadcast_to(selected[:,None,None],valid.shape)[valid]
                count += np.bincount(indices,weights=weight[source],minlength=cells)
                for channel,values in enumerate((vx,vy,kind)):
                    sums[channel] += np.bincount(indices,weights=(weight*values)[source],minlength=cells)
                np.minimum.at(closest,indices,np.minimum(distance[source],1.))
        occupied = count > 0
        result[0] = np.minimum(count/4., 1.).reshape(self.height, self.width)
        for index,(channel,divisor) in enumerate(((1,64.),(2,64.),(4,1.))):
            total = sums[index]
            np.divide(total, count*divisor, out=total, where=occupied)
            result[channel] = np.clip(total, -1 if channel in (1, 2) else 0, 1).reshape(self.height, self.width)
        return result

    @staticmethod
    def _polyline(laser):
        if laser.nodes:
            points = np.asarray(laser.nodes, dtype=np.float64)
        else:
            points = np.asarray(((laser.x, laser.y),
                (laser.x+math.cos(laser.angle)*laser.length, laser.y+math.sin(laser.angle)*laser.length)), dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 1 or not np.isfinite(points).all():
            raise ValueError('Invalid current laser polyline')
        keep = np.r_[True, np.any(np.diff(points, axis=0) != 0, axis=1)]
        points = points[keep]
        if len(points) < 2:
            return points, np.asarray([0.])
        arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        return points, arc

    def _laser_entities(self, lasers, *, measured_width=False, observed_motion=None):
        mapped, tokens = [], []
        for laser in sorted(lasers, key=lambda laser: laser.id):
            self._finite(laser.x, laser.y, laser.angle, laser.length, laser.width, laser.speed)
            if not laser.active and observed_motion is None:
                continue
            points, arc = self._polyline(laser)
            length = float(arc[-1])
            if (observed_motion is not None and len(laser.nodes)==1 and laser.width>0):
                # Native curves can be observed before a second node exists.
                # Retain their finite current footprint in V5, with unknown
                # motion. Formerly accepted multi-node/line inputs are exact.
                x,y=points[0];subtype=_clip(laser.kind/3.)
                width=laser.width if measured_width else 12.
                mapped.append((float(x)+192.,float(y),0.,0.,subtype*.25,width,width,1.))
                tokens.append((float(x)+192.,float(y),0.,0.,0.,subtype,laser.id,0))
                continue
            if length <= 0 or laser.width <= 0:
                continue
            map_spacing=min(16.,max(.25,laser.width*.5)) if measured_width else 16.
            if math.ceil(length/map_spacing)+1 > 65536:
                raise ValueError('Laser sampling exceeds explicit safety bound; no silent truncation')
            # Upstream: every16 + clamped endpoint, not <=8 point list.
            map_distances = np.minimum(np.arange(math.ceil(length/map_spacing)+1)*map_spacing, length)
            token_distances = np.linspace(0, length, min(math.ceil(length/32)+1, 8))
            motion = None if observed_motion is None else observed_motion.get(laser.id)
            for distances, destination, footprint in ((map_distances, mapped, 12.), (token_distances, tokens, 0.)):
                xs = np.interp(distances, arc, points[:, 0])+192.
                ys = np.interp(distances, arc, points[:, 1])
                for index, (x, y) in enumerate(zip(xs, ys)):
                    subtype = _clip(laser.kind/3.)
                    vx=vy=0.
                    if motion is not None and motion.motion_observed and not laser.nodes:
                        fraction=float(distances[index]/length)
                        vx,vy=(np.asarray(motion.origin_velocity)*(1-fraction)
                               +np.asarray(motion.endpoint_velocity)*fraction)
                    if footprint:
                        width = laser.width if measured_width else footprint
                        destination.append((x, y, vx, vy, subtype*.25, width, width, 1.)
                                           if measured_width else (x, y, vx, vy, subtype*.25, footprint))
                    else:
                        destination.append((x, y, vx, vy, 0., subtype, laser.id, index))
        return mapped, tokens

    def encode(self, snapshot: Snapshot, *, _geometry=None,_hp_independent_contacts=False,_laser_motion=None):
        snapshot.validate()
        if snapshot.player is None:
            raise ValueError('THRL observation needs a measured player; loading is not an empty field')
        resources = snapshot.extra.get('resources', {})
        if resources.get('observed') is not True or resources.get('schema') != RESOURCE_SCHEMA:
            raise ValueError('THRL observation requires the full measured resource block')
        if not isinstance(resources.get('items'), list) or not isinstance(resources.get('enemies'), list):
            raise ValueError('Measured resources must contain full items/enemies lists')
        misses, point_items = self._event_counts(snapshot)
        rows = {int(row['id']): row for row in resources['enemies']}
        if len(rows) != len(resources['enemies']) or set(rows) != {e.id for e in snapshot.enemies}:
            raise ValueError('Resource enemy identities do not match this full snapshot')
        p = snapshot.player
        px, py = p.x+192., p.y
        for e in snapshot.enemies:
            row = rows[e.id]
            self._finite(e.x, e.y, e.hp, row['vx'], row['vy'])
            if (row['x'], row['y'], row['hp']) != (e.x, e.y, e.hp):
                raise ValueError('Resource enemy measurements do not match the same snapshot')
        player = [_clip((p.x+184)/368), _clip((p.y-32)/400), 0., 0.,
            _clip((p.x+184)/368), _clip((184-p.x)/368), _clip((p.y-32)/400), _clip((432-p.y)/400),
            _clip(snapshot.power/400), _clip(snapshot.lives/8), _clip(snapshot.bombs/8),
            float(p.invulnerable_frames > 0), _clip(snapshot.character/2), _clip(snapshot.stage/6),
            0., 0., _clip(snapshot.difficulty/3)]
        relevant=lambda e:not e.intangible and (e.hp>0 or
            (_hp_independent_contacts and rows[e.id].get('contactable') is True))
        bosses = sorted((e for e in snapshot.enemies if e.boss and relevant(e)), key=lambda e: e.id)
        boss_features = []
        for index in range(2):
            if index >= len(bosses):
                boss_features.extend((0., 0., 0., 0., 0., 0., 1.))
                continue
            e = bosses[index]
            dx, dy = e.x-p.x, e.y-p.y
            boss_features.extend((1., _clip(e.hp/30000), _clip(dx/500, -1), _clip(dy/500, -1),
                _clip(rows[e.id]['vx']/12, -1), _clip(rows[e.id]['vy']/12, -1), _clip(math.hypot(dx, dy)/500)))
        state = [_clip(snapshot.stage/6), _clip(snapshot.graze/1000), _clip(snapshot.score/100000000),
                 _clip(misses/10), _clip(point_items/100), _clip(snapshot.difficulty/3)]
        bullets = sorted((b for b in snapshot.bullets if b.active and not b.intangible),
                         key=lambda b: ((b.x-p.x)**2+(b.y-p.y)**2, b.id))
        if _geometry is not None:
            bullets = sorted((b for b in snapshot.bullets if _geometry[b.id]['visible']),
                key=lambda b: ((b.x-p.x)**2+(b.y-p.y)**2,b.id))
        bullet_map = [(b.x+192., b.y, b.vx, b.vy, abs(b.kind)%256/255., 8.) for b in bullets]
        if _geometry is not None:
            bullet_map = []
            for b in bullets:
                g = _geometry[b.id]
                if g['collidable']:
                    bullet_map.append((b.x+192., b.y, b.vx, b.vy, abs(b.kind)%256/255.,
                                       g['width'], g['height'], float(g['circle'])))
        # Upstream's enemy map excludes its separately represented main/midboss.
        enemy_map = [(e.x+192., e.y, rows[e.id]['vx'], rows[e.id]['vy'], 0.,
                      24. if rows[e.id]['contactable'] else 0.) for e in snapshot.enemies
                     if not e.boss and relevant(e)]
        if _geometry is not None:
            enemy_map = [(e.x+192., e.y, rows[e.id]['vx'], rows[e.id]['vy'], 0.,
                max(0.,e.radius_x), max(0.,e.radius_y), 0.) for e in snapshot.enemies
                if not e.boss and relevant(e) and rows[e.id]['contactable']]
        projectile_map, projectiles = self._laser_entities(snapshot.lasers, measured_width=_geometry is not None,
            observed_motion=_laser_motion)
        projectiles.sort(key=lambda e: ((e[0]-px)**2+(e[1]-py)**2, e[6], e[7]))
        projectile_features = []
        bullet_features = []
        for index in range(16):
            if index < len(projectiles):
                x, y, vx, vy, kind, subtype, _, _ = projectiles[index]
                projectile_features.extend((_clip((x-px)/self.span_x, -1), _clip((y-py)/self.span_y, -1),
                    _clip(vx/12, -1), _clip(vy/12, -1), kind, subtype, _clip(math.hypot(x-px, y-py)/self.diagonal)))
            else:
                projectile_features.extend((0., 0., 0., 0., 0., 0., 1.))
            if index < len(bullets):
                b = bullets[index]
                dx, dy = b.x-p.x, b.y-p.y
                bullet_features.extend((_clip(dx/self.span_x, -1), _clip(dy/self.span_y, -1),
                    _clip(b.vx/12, -1), _clip(b.vy/12, -1), abs(b.kind)%256/255.,
                    _clip(math.hypot(b.vx, b.vy)/12), _clip(math.hypot(dx, dy)/self.diagonal)))
            else:
                bullet_features.extend((0., 0., 0., 0., 0., 0., 1.))
        items = list(resources['items'])
        for item in items:
            self._finite(item['x'], item['y'], item['kind'])
        items.sort(key=lambda it: ((it['x']-p.x)**2+(it['y']-p.y)**2, int(it['id'])))
        drops = []
        for index in range(4):
            if index < len(items):
                item = items[index]
                drops.extend((_clip((item['x']-p.x)/self.span_x, -1),
                              _clip((item['y']-p.y)/self.span_y, -1), _clip(item['kind']/12)))
            else:
                drops.extend((0., 0., 0.))
        features = np.asarray(player+boss_features+state+projectile_features+bullet_features+drops, dtype=np.float32)
        player_distance = np.clip(np.hypot(self._xx-px, self._yy-py)/self.diagonal, 0, 1)[None]
        maps = np.concatenate((self._map(bullet_map, px, py), self._map(enemy_map, px, py),
                               self._map(projectile_map, px, py), player_distance)).astype(np.float32)
        if (features.shape != self.feature_shape or maps.shape != self.map_shape
                or not np.isfinite(features).all() or not np.isfinite(maps).all()
                or np.abs(features).max() > 1 or np.abs(maps).max() > 1):
            raise ValueError('THRL observation schema/finite/range invariant failed')
        return {'features': features, 'maps': maps}
