"""Candidate route context; existing short-range native sensing is unchanged.

48/96-frame extrapolations are explicitly conditional on current objects.
They are NOT collision labels or predictions of not-yet-spawned bullets.
All values enter VP ports; there is no direct geometry-to-action bypass.
"""
import re
import numpy as np
from .compact_story_sensation import CompactStoryEncoder

SCHEMA = 'connectome-nbnm-route281-v1'
DIRECTION_WIDTH = 8
CONTEXT_WIDTH = 8
EXTENSION_WIDTH = DIRECTION_WIDTH + CONTEXT_WIDTH
FEATURE_WIDTH = 129 + 18 * DIRECTION_WIDTH + CONTEXT_WIDTH
DIRECTION_NAMES = ('clearance24_wide', 'linear_clearance48', 'linear_clearance96',
    'linear48_supported', 'linear96_supported', 'turn_exit_fraction',
    'best_turn_margin', 'opposed_exit_fraction')
CONTEXT_NAMES = ('attack_present', 'attack_card', 'attack_number', 'attack_age',
    'goal_attack', 'goal_stage', 'goal_stage_number', 'inside_goal_attack')


def candidate_extension(features):
    if features.shape[-1] != FEATURE_WIDTH:
        raise ValueError('NBNM route281 observation required')
    import torch
    shape = features.shape[:-1]
    direction = features[..., 129:273].reshape(*shape, 18, DIRECTION_WIDTH)
    context = features[..., 273:].unsqueeze(-2).expand(*shape, 18, CONTEXT_WIDTH)
    return torch.cat((direction, context), -1)


def current_attack(snapshot):
    roles = snapshot.extra.get('thrl_events', {}).get('ecl_roles', {})
    rows = roles.get('main_bosses', []) + roles.get('midbosses', [])
    return next((r['ecl_sub_name'] for r in rows if r.get('ecl_sub_name')), None)


def route_context(encoding):
    from .thrl_movement_teacher import MovementSafetyTeacher
    from .compact_spiking_sensation import point_and_laser_clearances
    f = encoding['source_features25']
    o = encoding['source_objects29']
    forecasts = encoding.get('source_native_forecasts') or {}
    margins = np.asarray(encoding['details']['margins'])
    v = MovementSafetyTeacher().velocities.copy()
    if f[18] < 0: v[9:] = v[:9]
    origin = np.array([f[0]*368-184, f[1]*400+32])
    starts = np.broadcast_to(origin, (18, 2)).copy()
    x = np.zeros((18, DIRECTION_WIDTH), np.float32)
    wide=np.nan_to_num(np.asarray(encoding['details'].get('unclipped_margins',margins))[2],
        nan=0.,posinf=1e6,neginf=-1e6)
    x[:, 0] = wide / (32 + np.abs(wide))
    # Extended motion is only a linear hint. Mark non-linear/native-script
    # objects and omitted curve nodes as unsupported, locally where possible.
    uncertain = np.zeros(len(o), bool)
    for key in ('enemies', 'bullets', 'reactive_bullets'):
        for row in forecasts.get(key, ()):
            uncertain[row['object_index']] = True
    if len(o):
        uncertain |= (o[:, 20] > .5) | (((o[:, 3] > .5) | (o[:, 4] > .5)) & (o[:, 26] < .5))
    curves_unknown = bool((forecasts.get('curve_geometry') or {}).get('curves'))
    hidden_unknown = bool(forecasts.get('hidden_enemy_unreadable_geometry'))
    for j, horizon in enumerate((48, 96)):
        linear = point_and_laser_clearances(f, o, starts, v, duration=horizon)
        x[:, 1+j] = np.clip(linear, -256, 256) / (32 + np.abs(np.clip(linear, -256, 256)))
        inside = ((starts + v*horizon >= [-184, 32]) & (starts + v*horizon <= [184, 432])).all(-1)
        supported = inside & ~np.full(18, curves_unknown or hidden_unknown)
        if uncertain.any():
            rel = o[uncertain, 6:8] * [384, 448]
            speed = np.linalg.norm(o[uncertain, 8:10]*12, axis=1)
            radius = np.linalg.norm(o[uncertain, 10:12]*128, axis=1)
            # Candidate-dependent reach, not one unknown object invalidating
            # every possible route across the entire field.
            for a in range(18):
                delta = rel - v[a]*horizon*.5
                nearby = np.linalg.norm(delta, axis=1) <= speed*horizon + radius + np.linalg.norm(v[a])*horizon*.5 + 24
                supported[a] &= not nearby.any()
        x[:, 3+j] = supported
    ends = np.clip(starts + v*4, [-184, 32], [184, 432])
    # All 18 continuations, including reversal; the old input inspected just
    # three nearby velocities. Keep native short-horizon models here.
    from .native_wall_geometry import player_paths
    paths = None
    if forecasts.get('reactive_bullets'):
        first = player_paths(starts, v)
        paths = player_paths(np.repeat(ends, 18, axis=0), np.tile(v, (18, 1)),
            prefix=np.repeat(first, 18, axis=0), turn_tick=4)
    future = point_and_laser_clearances(f, o, np.repeat(ends, 18, axis=0),
        np.tile(v, (18, 1)), duration=12, initial_time=4,
        anticipate_arming=True, growth_motion_envelope=True,
        native_forecasts=forecasts, unknown_laser_player_motion=True, player_path=paths).reshape(18,18)
    turn = np.minimum(margins[0, :, None], future)
    safe = turn > 1
    from .nbnm_route_support import physical_exit_support
    x[:, 5], x[:, 7] = physical_exit_support(ends, v, safe)
    best = np.clip(turn.max(-1), -256, 256)
    x[:, 6] = best / (32 + np.abs(best))
    return x


class NBNMEncoder(CompactStoryEncoder):
    schema = SCHEMA
    feature_shape = (FEATURE_WIDTH,)
    feature_names = (*CompactStoryEncoder.feature_names,
        *(f'action{a}_{name}' for a in range(18) for name in DIRECTION_NAMES), *CONTEXT_NAMES)

    def __init__(self, *, risk_mode='route', task=None):
        super().__init__()
        if risk_mode not in ('baseline', 'route'): raise ValueError('baseline or route input required')
        self.risk_mode = risk_mode
        self.task = dict(task or {'mode':'full_story'})
        self._attack = None
        self._attack_start = 0
        self._nbnm_cache = None

    def reset(self):
        super().reset()
        self._attack = None
        self._attack_start = 0
        self._nbnm_cache = None

    def encode(self, snapshot):
        key=(self.base.native_episode,snapshot.frame)
        if self._nbnm_cache==key:return {'features':self.last_result['features'].copy()}
        original = super().encode(snapshot)['features']
        extra = np.zeros((18, DIRECTION_WIDTH), np.float32)
        context = np.zeros(CONTEXT_WIDTH, np.float32)
        teacher_route = route_context(self.last_result)
        if self.risk_mode == 'route':
            extra = teacher_route
        # Task and phase identity are shared by both input ablations. A card
        # completion is terminal in practice but not in a full-game episode.
        attack = (snapshot.stage, current_attack(snapshot))
        if attack != self._attack:
            self._attack, self._attack_start = attack, snapshot.frame
        name = attack[1] or ''
        number = re.search(r'(\d+)', name)
        age = max(0, snapshot.frame-self._attack_start)
        from .nbnm_task import attack_key
        context[:] = (bool(name), 'Card' in name,
            min(int(number.group(1)),32)/32 if number else 0, age/(age+600),
            self.task['mode']=='attack', self.task['mode']=='stage',
            self.task.get('stage',0)/6,
            self.task['mode']=='attack' and snapshot.stage==self.task.get('stage')
                and attack_key(name)==attack_key(self.task.get('attack')))
        features = np.concatenate((original, extra.reshape(-1), context)).astype(np.float32)
        self.last_result = dict(self.last_result, features=features, route_context=teacher_route,
            route_context_scope='conditional current-scene forecast, not native future outcomes')
        self._nbnm_cache=key
        return {'features': features.copy()}

    def specification(self):
        return dict(super().specification(), schema=SCHEMA, risk_mode=self.risk_mode,
            features=dict(shape=[FEATURE_WIDTH],names=self.feature_names),
            feature_width=FEATURE_WIDTH, extension_per_candidate=EXTENSION_WIDTH,
            direction_names=DIRECTION_NAMES, context_names=CONTEXT_NAMES,
            physical_exit_revision='clipped-distinct-endpoints-v2',
            long_horizons_are_labels=False, future_spawn_information=False)
