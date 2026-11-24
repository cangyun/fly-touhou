"""Candidate route context; existing short-range native sensing is unchanged.

48/96-frame corridors are conditional on observed objects and, when enabled,
ordinary line births from observed emitters. All sensory values enter VP ports.
"""
import re
import numpy as np
from .compact_story_sensation import CompactStoryEncoder
from .native_bullet_release import share_release_sweeps
from .native_ex_conditional import share_waiting_aim
from .native_projection_cache import share_native_projections

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


def candidate_extension(features, *, route_prior=False, future_rules=False, corridor=False):
    from .nbnm_route_atlas import WIDTH
    from .nbnm_future_sensation import WIDTH as FUTURE_WIDTH
    from .nbnm_corridor import WIDTH as CORRIDOR_WIDTH
    expected=FEATURE_WIDTH+18*(WIDTH*route_prior+FUTURE_WIDTH*future_rules+CORRIDOR_WIDTH*corridor)
    if features.shape[-1] not in (FEATURE_WIDTH,expected):
        raise ValueError(f'Expected 281 legacy columns or {expected} configured NBNM columns, got {features.shape[-1]}')
    import torch
    shape = features.shape[:-1]
    direction = features[..., 129:273].reshape(*shape, 18, DIRECTION_WIDTH)
    context = features[..., 273:281].unsqueeze(-2).expand(*shape, 18, CONTEXT_WIDTH)
    result = torch.cat((direction, context), -1)
    if route_prior:
        # Old rehearsal histories have no prior. Unknown is represented by
        # zero confidence, not by an invented future trajectory.
        prior = (features[...,281:281+18*WIDTH].reshape(*shape,18,WIDTH) if features.shape[-1]>281
                 else features.new_zeros((*shape,18,WIDTH)))
        result = torch.cat((result, prior), -1)
    if future_rules:
        start=281+18*WIDTH*route_prior
        future=(features[...,start:start+18*FUTURE_WIDTH].reshape(*shape,18,FUTURE_WIDTH)
            if features.shape[-1]>start else features.new_zeros((*shape,18,FUTURE_WIDTH)))
        result=torch.cat((result,future),-1)
    if corridor:
        extra=(features[...,281:].reshape(*shape,18,CORRIDOR_WIDTH) if features.shape[-1]>281
            else features.new_zeros((*shape,18,CORRIDOR_WIDTH)))
        result=torch.cat((result,extra),-1)
    return result


def critic_scope(task, stage, name):
    from .nbnm_task import attack_key
    end_stage = task.get('end_stage', task.get('stage', 6))
    end = attack_key(task.get('end_attack', task.get('attack')))
    number = ((end[2] if end[1] else -end[2])/32) if end else 0.
    kind = {'full_story':0, 'stage':1, 'attack':2, 'chain':3}[task['mode']]/3
    inside = bool(end and stage == end_stage and attack_key(name) == end)
    return (kind, end_stage/6, number, inside)


def current_attack(snapshot):
    roles = snapshot.extra.get('thrl_events', {}).get('ecl_roles', {})
    for kind in ('main_bosses','midbosses'):
        for row in roles.get(kind,[]):
            name=row.get('ecl_sub_name')
            if name:
                # Seija's main BossCard3 temporarily executes MBossCard3_at.
                # The native MainBoss role stays the same; it is not a new
                # midboss/card or a successful task boundary.
                return name[1:] if kind=='main_bosses' and name.startswith('MBoss') else name
    return None


def route_context(encoding, *, capture_turns=False):
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
    wide=np.asarray(encoding['details'].get('unclipped_margins',margins))[2]
    if np.isnan(wide).any() or np.isneginf(wide).any():
        raise ValueError('Non-finite NBNM 24-frame route margin; refusing a silent zero replacement')
    wide=np.where(np.isposinf(wide),1e6,wide)
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
    for j, horizon in enumerate(() if forecasts.get('rule_program_horizon')==96 else (48, 96)):
        # Long features are explicitly conditional linear hints, not the
        # native4/12/24 risk-label contract. Lasers use the same distinct path.
        linear = point_and_laser_clearances(f, o, starts, v, duration=horizon,linear_hint=True)
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
        if forecasts.get('rule_program_revision') and forecasts.get('bullets'):
            # A24-frame execution is not a48/96-frame speed bound. Pending
            # accelerations, rebounds and child emission can invalidate the
            # old "far away at current speed" shortcut. Keep the numbers as
            # linear hints, but do not certify them for EX-controlled scenes.
            supported[:]=False
        x[:, 3+j] = supported
    if forecasts.get('rule_program_horizon')==96:
        from .nbnm_rule_geometry import long_hints
        x[:,1:5]=long_hints(encoding,starts,v)
    projection = forecasts.get('emitter_projection_horizon')
    boundary = forecasts.get('pending_emitter_unknown_from_tick')
    if projection is not None:
        for column, horizon in enumerate((48, 96), 3):
            if horizon > projection or (boundary is not None and boundary <= horizon):
                x[:, column] = 0.
    ends = np.clip(starts + v*4, [-184, 32], [184, 432])
    # All 18 continuations, including reversal; the old input inspected just
    # three nearby velocities. Keep native short-horizon models here.
    from .native_wall_geometry import player_paths
    paths = None
    upcoming = any(event['tick'] <= 16 for key in ('pending_line_births', 'pending_point_birth_bounds')
                   for event in forecasts.get(key, ()))
    if (forecasts.get('reactive_bullets') or upcoming or
            any(r.get('candidate_dependent') for r in forecasts.get('bullets',()))):
        first = player_paths(starts, v)
        paths = player_paths(np.repeat(ends, 18, axis=0), np.tile(v, (18, 1)),
            prefix=np.repeat(first, 18, axis=0), turn_tick=4)
    future = point_and_laser_clearances(f, o, np.repeat(ends, 18, axis=0),
        np.tile(v, (18, 1)), duration=12, initial_time=4,
        anticipate_arming=True, growth_motion_envelope=True,
        native_forecasts=forecasts, unknown_laser_player_motion=True, player_path=paths).reshape(18,18)
    turn = np.minimum(margins[0, :, None], future)
    if capture_turns:
        encoding['original_turn_clearances'] = turn
    safe = turn > 1
    from .nbnm_route_support import physical_exit_support
    x[:, 5], x[:, 7] = physical_exit_support(ends, v, safe)
    best = np.clip(turn.max(-1), -256, 256)
    x[:, 6] = best / (32 + np.abs(best))
    faults=forecasts.get('nonfinite_clearance_fallbacks',())
    if faults:
        details=encoding['details']
        details.setdefault('uncertainty',{})['nonfinite_clearance_fallbacks']=faults
        for index,horizon in enumerate((4,12,24)):
            if any(row['initial_time']+row['duration']<=horizon for row in faults):
                details['trusted'][index]=False
                if 'risk_training_trusted' in details:details['risk_training_trusted'][index]=False
    if not np.isfinite(x).all():raise ValueError('Non-finite NBNM route context')
    return x


def pending_straight_context(context, forecasts, source, position):
    """Limit legacy straight-line hints by known births on that same straight move."""
    lines = forecasts.get('pending_line_births')
    volleys = forecasts.get('pending_stationary_volleys')
    rings = forecasts.get('pending_rotating_rings')
    if not lines and volleys is None and rings is None:
        return context
    from .native_wall_geometry import player_paths
    from .thrl_movement_teacher import action_velocities
    from .native_pending_lines import line_clearances
    from .native_pending_volleys import volley_clearances
    from .native_pending_rings import ring_clearances
    velocity = np.trunc(action_velocities() * 128) / 128
    if source[18] < 0:
        velocity[9:] = velocity[:9]
    paths = player_paths(np.broadcast_to(position, velocity.shape), velocity, ticks=96)
    result = context.copy()
    gap = np.full(len(velocity), np.inf)
    begin = 0
    for col, end in enumerate((24, 48, 96)):
        if lines:
            gap = np.minimum(gap, line_clearances(lines, paths, begin, end, forecasts['pending_line_half_extents']))
        if volleys is not None:
            gap = np.minimum(gap, volley_clearances(volleys, paths, begin, end, source[13] * 64.))
        if rings is not None:
            gap = np.minimum(gap, ring_clearances(rings, paths, begin, end))
        known = np.isfinite(gap)
        value = gap[known] if col == 0 else np.clip(gap[known], -256., 256.)
        result[known, col] = np.minimum(result[known, col], value / (32. + np.abs(value)))
        if col:
            # These values now depend on future firings; the original linear
            # extrapolation alone cannot certify this direction.
            result[result[:, col] < context[:, col], col + 2] = 0.
        begin = end
    return result


class NBNMEncoder(CompactStoryEncoder):
    schema = SCHEMA
    native_rule_geometry = None
    frozen_exit_scoring = False
    frozen_exit_limit = 'auto'
    feature_shape = (FEATURE_WIDTH,)
    feature_names = (*CompactStoryEncoder.feature_names,
        *(f'action{a}_{name}' for a in range(18) for name in DIRECTION_NAMES), *CONTEXT_NAMES)

    def __init__(self, *, risk_mode='route', task=None, task_context='legacy', route_prior=None, future_rules=None,native_rule_geometry=None, corridor_risk=None):
        super().__init__()
        from .nbnm_geometry_revision import validate,schema
        self.native_rule_geometry=validate(native_rule_geometry)
        if self.native_rule_geometry:self.schema=schema(self.native_rule_geometry)
        if risk_mode not in ('baseline', 'route'): raise ValueError('baseline or route input required')
        self.risk_mode = risk_mode
        self.task = dict(task or {'mode':'full_story'})
        self.task_context = task_context
        self._attack = None
        self._attack_start = 0
        self._nbnm_cache = None
        self.route_atlas = None
        self.corridor_carry=None
        self.lure_carry=None
        self.exit48_carry=None
        from .nbnm_corridor import validate as validate_corridor
        self.corridor_risk=validate_corridor(corridor_risk)
        if self.corridor_risk:
            from .nbnm_corridor import NAMES as CORRIDOR_NAMES, SCHEMA as CORRIDOR_SCHEMA
            if (route_prior or {}).get('enabled') or (future_rules or {}).get('enabled'):
                raise ValueError('Use a single versioned extended observation')
            self.schema=CORRIDOR_SCHEMA
            self.feature_names=(*self.feature_names,*(f'action{a}_corridor_{name}' for a in range(18) for name in CORRIDOR_NAMES))
            self.feature_shape=(len(self.feature_names),)
        if route_prior and route_prior.get('enabled'):
            from .nbnm_route_atlas import RouteAtlas, NAMES
            self.route_atlas = RouteAtlas(route_prior['manifest'],
                phase_landmarks=route_prior.get('phase_landmarks',False))
            self.schema = 'connectome-nbnm-route389-atlas-v1'
            self.feature_names = (*self.feature_names,
                *(f'action{a}_atlas_{name}' for a in range(18) for name in NAMES))
            self.feature_shape = (len(self.feature_names),)
        self.future_rules=None
        if future_rules and future_rules.get('enabled'):
            if self.route_atlas is not None:raise ValueError('Compare rule forecasts without adding an atlas simultaneously')
            from .nbnm_future_sensation import FutureRuleSensor,NAMES,SCHEMA
            self.future_rules=FutureRuleSensor(future_rules)
            self.schema=SCHEMA
            self.feature_names=(*self.feature_names,*(f'action{a}_rule_{name}' for a in range(18) for name in NAMES))
            self.feature_shape=(len(self.feature_names),)

    def reset(self):
        super().reset()
        self.corridor_carry=None
        self.lure_carry=None
        self.exit48_carry=None
        if getattr(self,'future_rules',None) is not None:self.future_rules.reset()
        self._attack = None
        self._attack_start = 0
        self._nbnm_cache = None

    def _apply_pending_warnings(self, features):
        forecasts=self.last_result.get('source_native_forecasts') or {}
        events=forecasts.get('pending_line_births')
        if events and 'emitter_projection_horizon' not in forecasts:
            from .native_pending_lines import straight_line_warnings
            hints=straight_line_warnings(events,self._corridor_snapshot,self.last_result['source_features25'],
                forecasts['pending_line_half_extents'])
            # Only the sensory hints change. Factual short-horizon supervision
            # still comes from native contact and the student's executed route.
            features[18:72]=np.maximum(features[18:72],hints.ravel())
        volleys=forecasts.get('pending_stationary_volleys')
        rings=forecasts.get('pending_rotating_rings')
        if volleys is not None or rings is not None:
            from .native_pending_volleys import volley_clearances
            from .native_wall_geometry import player_paths
            from .thrl_movement_teacher import action_velocities
            velocity=action_velocities().copy()
            source=self.last_result['source_features25']
            if source[18]<0:
                velocity[9:]=velocity[:9]
            player=self._corridor_snapshot.player
            paths=player_paths(np.broadcast_to([player.x,player.y],velocity.shape),velocity)
            margins=np.full((3,len(velocity)),np.inf)
            if volleys is not None:
                margins=np.stack([volley_clearances(volleys,paths,0,h,source[13]*64.) for h in (4,12,24)])
            if rings is not None:
                from .native_pending_rings import ring_clearances
                margins=np.minimum(margins,np.stack([ring_clearances(rings,paths,0,h) for h in (4,12,24)]))
            hints=np.where(np.isposinf(margins),-1.,np.tanh((1.-np.clip(margins,-64.,64.))/8.)).astype(np.float32)
            features[18:72]=np.maximum(features[18:72],hints.ravel())

    def current_corridor(self):
        from .nbnm_corridor import plan
        stage=self.last_result['stage']
        phase=self._attack[1] if self._attack else ''
        forecasts=self.last_result.get('source_native_forecasts')
        if forecasts is not None:
            projected = forecasts.get('emitter_projection_horizon')
            if not projected:
                forecasts.pop('pending_line_births',None)
                forecasts.pop('pending_line_half_extents',None)
            forecasts.pop('pending_stationary_volleys',None)
            forecasts.pop('pending_rotating_rings',None)
            if self.corridor_risk.get('anticipate_rotating_rings') and self.last_result['details']['normal']:
                from .native_pending_rings import forecast_rotating_rings
                rings=forecast_rotating_rings(self.corridor_controller.reader.mem,self._corridor_snapshot,phase,
                    120 if stage in self.corridor_risk.get('onward_stages',()) else 96,
                    self.last_result['source_features25'][13]*64.,arc=self.corridor_risk.get('rotating_ring_arc',False))
                if rings is not None:
                    forecasts['pending_rotating_rings']=rings
            if self.corridor_risk.get('anticipate_stationary_volleys') and self.last_result['details']['normal']:
                from .native_pending_volleys import forecast_stationary_volleys
                upcoming=forecast_stationary_volleys(self.corridor_controller.reader.mem,self._corridor_snapshot,phase,
                    120 if stage in self.corridor_risk.get('onward_stages',()) else 96)
                if upcoming is not None:
                    forecasts['pending_stationary_volleys']=upcoming
            if (not projected and self.corridor_risk.get('anticipate_line_births') and
                    self.last_result['details']['normal']):
                from .native_pending_lines import forecast_line_births
                snapshot=self._corridor_snapshot
                extents=snapshot.extra.get('native_line_lasers',{}).get('player_half_extents')
                if extents is not None:
                    upcoming=forecast_line_births(self.corridor_controller.reader.mem,snapshot,
                        horizon=120 if stage in self.corridor_risk.get('onward_stages',()) else 96)
                    if upcoming:
                        forecasts.update(pending_line_births=upcoming,pending_line_half_extents=extents)
        refine=self.corridor_risk.get('refinement_phases','all')=='all'
        retry_spell=True
        if not refine:
            from .nbnm_task import attack_key
            key=attack_key(phase)
            refine=key is not None and key[0] in ('Boss','BossA','BossB') and key[1]
            if refine and 'refinement_first_boss_spell' in self.corridor_risk:
                refine=key[2]>=self.corridor_risk['refinement_first_boss_spell']
            if 'reactive_retry_boss_cards' in self.corridor_risk:
                retry_spell=refine and key[2] in self.corridor_risk['reactive_retry_boss_cards']
        wide=refine and stage in self.corridor_risk.get('wide_beam_stages',())
        retry=refine and retry_spell and stage in self.corridor_risk.get('reactive_retry_stages',())
        late_turns=self.corridor_risk.get('refine_uncertain_turns',False) or (
            retry and self.corridor_risk.get('refine_reactive_turns',False))
        from .nbnm_task import attack_key
        wall = stage == 6 and attack_key(phase) == ('Boss', True, 5)
        issun = wall and self.corridor_risk.get('issun_lure', False)
        if issun and self.corridor_risk.get('issun_lure_scope')=='constrained':issun='constrained'
        teach_lure = bool(wall and self.corridor_risk.get('issun_guidance_weight', 0.))
        # Use the spell's clock. A revalidated course can guide the intervening decisions.
        query = teach_lure and (self.last_result['frame'] - self._attack_start) % 16 < 4
        row=plan(self.last_result,beam=2 if wide else self.corridor_risk.get('beam',1),carry=self.corridor_carry,
            corner_reserve=refine and stage in self.corridor_risk.get('corner_reserve_stages',()),
            onward=refine and stage in self.corridor_risk.get('onward_stages',()),
            reactive_retry=retry,refine_uncertain_turns=late_turns,
            refine_early_turns=late_turns and self.corridor_risk.get('refine_early_turns',False),issun_lure=issun,
            issun_lure_tail_turns=bool(issun and self.corridor_risk.get('issun_lure_tail_turns',False)),
            teach_lure=teach_lure, lure_carry=self.lure_carry, query_lure=query)
        recheck_limit = self.corridor_risk.get('native_emission_recheck_limit', 0)
        if recheck_limit and not (row['features'][:,3] > 0).any():
            from .nbnm_newborn_course import add_reviewed_emission_courses, plan_birth_rechecked_course
            current = dict(self.last_result, corridor=row)
            report = plan_birth_rechecked_course(
                self.corridor_controller.reader, self._corridor_snapshot, current, limit=recheck_limit)
            actions = add_reviewed_emission_courses(current, row, report)
            row['native_emission_recheck'] = dict(actions=actions, courses=len(report['candidates']),
                covered_horizons=report.get('covered_horizons', ()), seconds=report.get('seconds', 0.),
                reason=report.get('reason'))
        local_turn_limit = self.corridor_risk.get('local_turn_limit', 0)
        if local_turn_limit:
            from .nbnm_corridor import refine_local_turns
            refine_local_turns(self.last_result, row, local_turn_limit)
        if self.frozen_exit_scoring:
            from .nbnm_exit48 import extend_enemy_window, score
            snapshot = self._corridor_snapshot
            memory = self.corridor_controller.reader.mem
            def extend_bodies(forecasts):
                return extend_enemy_window(forecasts, snapshot, memory)
            row['exit48'] = score(self.last_result, row, self.exit48_carry, attack=phase or '',
                                  issun=wall, extend_bodies=extend_bodies, finalists=self.frozen_exit_limit)
        self.last_result['corridor']=row
        return row['features'].ravel()

    def finish_corridor_migration(self):
        """Add the first new corridor at the committed original-game landing."""
        features=self.last_result['features'].copy()
        if self.base.last_result is not None:
            features[18:72]=self.base.last_result['features'][18:72]
        features[FEATURE_WIDTH:]=self.current_corridor()
        self._apply_pending_warnings(features)
        if self.corridor_risk.get('align_pending_clearances'):
            # Resume may switch forecasts at the same decision. Always rebuild
            # straight hints from the pre-birth values for this exact position.
            base = self.last_result.get('route_context_before_births', self.last_result['route_context'])
            self.last_result['route_context_before_births'] = base
            self.last_result['route_context'] = self._align_pending_context(base, features)
        elif 'route_context_before_births' in self.last_result:
            base = self.last_result.pop('route_context_before_births')
            self.last_result['route_context'] = base
            if self.risk_mode == 'route':
                features[129:273] = base.reshape(-1)
        features[72:75]=self.last_result['details']['trusted']
        self.last_result['route_context'] = self._apply_exit_forecast(features, self.last_result['route_context'])
        if not np.isfinite(features).all():
            raise ValueError('Non-finite corridor at resumed native landing')
        self.last_result['features']=features
        return {'features':features.copy()}

    def _align_pending_context(self, original, features):
        player = self._corridor_snapshot.player
        forecasts = self.last_result.get('source_native_forecasts') or {}
        if player is None:
            return original
        corrected = pending_straight_context(original, forecasts, self.last_result['source_features25'], [player.x, player.y])
        if self.risk_mode == 'route':
            features[129:273] = corrected.reshape(-1)
        return corrected

    def _apply_exit_forecast(self, features, context):
        effect = (self.last_result.get('corridor') or {}).get('exit48') or {}
        margins = effect.get('risk24')
        turns = effect.get('turn_cues')
        if margins is None and turns is None:
            return context
        context = context.copy()
        if margins is not None:
            if 'risk_prefix' in effect:
                features[18:54] = np.tanh((1. - effect['risk_prefix'].ravel()) / 8.)
            features[54:72] = np.tanh((1. - margins) / 8.)
            features[74] = 1.
            context[:, 0] = margins / (32. + np.abs(margins))
        if turns is not None:
            context[:, 5:8] = turns
        if self.risk_mode == 'route':
            features[129:273] = context.ravel()
        return context

    @share_native_projections()
    @share_release_sweeps()
    @share_waiting_aim()
    def encode(self, snapshot):
        key=(self.base.native_episode,snapshot.frame)
        if self._nbnm_cache==key:return {'features':self.last_result['features'].copy()}
        if self.native_rule_geometry:
            actual=snapshot.extra.get('hazard_metadata',{}).get('native_rule_geometry')
            if actual!=self.native_rule_geometry:
                raise ValueError('NBNM encoder and native reader geometry revisions must match')
        original = super().encode(snapshot)['features']
        extra = np.zeros((18, DIRECTION_WIDTH), np.float32)
        context = np.zeros(CONTEXT_WIDTH, np.float32)
        turn_options = {'capture_turns': True} if self.frozen_exit_scoring else {}
        teacher_route = route_context(self.last_result, **turn_options)
        if self.risk_mode == 'route':
            extra = teacher_route
        # Task and phase identity are shared by both input ablations. A card
        # completion is terminal in practice but not in a full-game episode.
        from .nbnm_task import attack_key
        name = current_attack(snapshot) or ''
        attack = (snapshot.stage, name)
        if attack != self._attack:
            self._attack, self._attack_start = attack, snapshot.frame
        number = re.search(r'(\d+)', name)
        age = max(0, snapshot.frame-self._attack_start)
        from .nbnm_task import attack_key
        context[:] = (bool(name), 'Card' in name,
            min(int(number.group(1)),32)/32 if number else 0, age/(age+600),
            self.task['mode']=='attack', self.task['mode']=='stage',
            self.task.get('stage',0)/6,
            self.task['mode']=='attack' and snapshot.stage==self.task.get('stage')
                and attack_key(name)==attack_key(self.task.get('attack')))
        if self.task_context == 'critic-only-v1':
            context[4:] = critic_scope(self.task, snapshot.stage, name)
        features = np.concatenate((original, extra.reshape(-1), context)).astype(np.float32)
        prior = None
        future=None
        if self.route_atlas is not None:
            from .nbnm_route_atlas import phase_progress
            phase, clock = phase_progress(snapshot)
            port_input, prior = self.route_atlas.query(phase, clock, features,
                focused_available=self.last_result['source_features25'][18]>=0)
            features = np.concatenate((features, port_input.reshape(-1))).astype(np.float32)
        if self.future_rules is not None:
            port_input,future=self.future_rules.query(snapshot,source_features=self.last_result['source_features25'])
            features=np.concatenate((features,port_input.reshape(-1))).astype(np.float32)
        if self.corridor_risk:
            self._corridor_snapshot=snapshot
            if getattr(self,'replaying_before_corridor_migration',False):
                extra=np.zeros(self.feature_shape[0]-FEATURE_WIDTH,np.float32)
            else:
                extra=self.current_corridor()
            features=np.concatenate((features,extra))
            self._apply_pending_warnings(features)
            forecasts = self.last_result.get('source_native_forecasts') or {}
            if self.corridor_risk.get('align_pending_clearances') or 'emitter_projection_horizon' in forecasts:
                self.last_result['route_context_before_births'] = teacher_route
                teacher_route = self._align_pending_context(teacher_route, features)
        # Queries for turning routes may discover a numerical fault after
        # the base 111 features were encoded. Expose its corrected trust bits.
        features[72:75]=self.last_result['details']['trusted']
        teacher_route = self._apply_exit_forecast(features, teacher_route)
        if not np.isfinite(features).all():
            bad=np.flatnonzero(~np.isfinite(features))
            raise ValueError('Non-finite NBNM sensory features: '+str([self.feature_names[i] for i in bad]))
        self.last_result = dict(self.last_result, features=features, route_context=teacher_route,
            route_context_scope='conditional current-scene forecast',
            route_prior=prior,future_rules=future)
        self._nbnm_cache=key
        return {'features': features.copy()}

    def specification(self):
        names=self.feature_names
        if self.task_context=='critic-only-v1':
            names=(*names[:277],'critic_task_kind','critic_end_stage','critic_end_attack','critic_inside_end_attack',*names[281:])
        return dict(super().specification(), schema=self.schema, risk_mode=self.risk_mode,
            native_rule_geometry=self.native_rule_geometry,
            features=dict(shape=list(self.feature_shape),names=names),
            feature_width=self.feature_shape[0], extension_per_candidate=EXTENSION_WIDTH+(6 if self.route_atlas else 0)+(8 if self.future_rules else 0)+(7 if self.corridor_risk else 0),
            corridor_risk=self.corridor_risk,
            direction_names=DIRECTION_NAMES, context_names=CONTEXT_NAMES,
            physical_exit_revision='clipped-distinct-endpoints-v2',
            task_context=self.task_context,
            task_scope_reaches_action_or_risk=self.task_context!='critic-only-v1',
            long_horizons_are_labels=bool(self.corridor_risk),
            corridor_long_label_scope=('conditional turning-route quality and witnessed exits' if self.corridor_risk else None),
            future_spawn_information=bool(self.future_rules or any((self.corridor_risk or {}).get(key)
                for key in ('anticipate_line_births','anticipate_stationary_volleys','anticipate_rotating_rings'))),
            future_rules=(self.future_rules.config if self.future_rules else None),
            route_atlas=(self.route_atlas.manifest if self.route_atlas else None))
