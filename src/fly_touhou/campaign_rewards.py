"""Explicit training-only feedback from measured original-game transitions.

No action selection, input change, extra game tick, or native resource write.
The actor still sees the original versioned raw observation and uses only its
complete-brain descending spike readout. Forecasts are bounded approximations,
not claims that future native collision gates or attack scripts are known.
"""
from __future__ import annotations

import math
import operator
from dataclasses import replace

from .background import BackgroundTouhouEnv
from .native_hazard_planner import NativeHazardPlanner

LEGACY_REWARD = 'native-legacy'
AVOIDANCE_REWARD = 'native-life-avoidance-v1'
EMERGENCY_BOMB_REWARD = 'native-life-emergency-bomb-v1'
GEOMETRY_EMERGENCY_REWARD = 'native-life-emergency-geometry-v2'


class NativeLifeAvoidanceReward:
    def __init__(self, *, gamma, planner_config):
        if not math.isfinite(gamma) or not 0 < gamma <= 1:
            raise ValueError('Reward shaping requires the actual finite positive PPO discount')
        self.gamma = float(gamma)
        self.planner = NativeHazardPlanner(planner_config)
        self._cached_snapshot = self._cached_potential = None

    def specification(self):
        return {'schema': AVOIDANCE_REWARD, 'gamma': self.gamma,
                'actual_life_decrement_coefficient': -5.,
                'deathbomb_entry_penalty': 0., 'legacy_hit_penalty_replaced': True,
                'survival_damage_power_and_stage_rewards': 'unchanged legacy measured components',
                'potential_scale': .25, 'forecast_horizon_native_frames': 8.,
                'clearance_decay_pixels': 12., 'sidewall_band_pixels': 32.,
                'sidewall_potential_weight': .1,
                'potential': 'negative bounded stationary-player short-horizon hazard plus small sidewall term',
                'field_bounds': [self.planner.config.left, self.planner.config.right],
                'shaping': 'scale * (gamma * Phi(next_native_state) - Phi(current_native_state))',
                'terminal_potential': 0., 'time_limit_potential': 'retain actual final observation',
                'prediction_source': self.planner.specification(),
                'changes_actions_or_native_resources': False,
                'risk_used_as_actor_input': False, 'teacher_action_calls': 0}

    def potential(self, snapshot):
        if snapshot is self._cached_snapshot:
            return self._cached_potential
        player = snapshot.player
        if player is None or player.state != 1 or player.invulnerable_frames > 0:
            result = {'value': 0., 'hazard': 0., 'sidewall': 0., 'clearance_pixels': None,
                      'currently_vulnerable': False}
        else:
            # Current native metadata gates are respected by the existing v4
            # predictor. Unknown future acceleration/activation remains unknown.
            clearance = self.planner.path_clearance(snapshot,
                [(0., 8., (player.x, player.y), (0., 0.))])
            if math.isnan(clearance) or clearance == -math.inf:
                raise FloatingPointError('Nonfinite native hazard clearance')
            hazard = 0. if math.isinf(clearance) else math.exp(-max(0., clearance) / 12.)
            room = min(player.x - self.planner.config.left, self.planner.config.right - player.x)
            sidewall = max(0., min(1., (32. - room) / 32.))
            result = {'value': -hazard - .1 * sidewall, 'hazard': hazard,
                      'sidewall': sidewall, 'clearance_pixels': None if math.isinf(clearance) else clearance,
                      'currently_vulnerable': True}
        self._cached_snapshot, self._cached_potential = snapshot, result
        return result

    def evaluate(self, before, after, elapsed, *, terminated, action=None):
        _, _, legacy = BackgroundTouhouEnv.reward(before, after, elapsed)
        current = self.potential(before)
        following = ({'value': 0., 'terminal': True} if terminated else self.potential(after))
        shaping = .25 * (self.gamma * following['value'] - current['value'])
        components = {name: value for name, value in legacy.items() if name != 'hit'}
        components['actual_life_decrement'] = -5. * max(0, before.lives - after.lives)
        components['avoidance_potential_difference'] = shaping
        details = {'schema': AVOIDANCE_REWARD, 'before': current, 'after': following,
                   'replaced_legacy_hit_penalty': legacy['hit'],
                   'life_decrements': max(0, before.lives - after.lives),
                   'native_deathbomb_entry': bool(before.player and after.player
                       and before.player.state != 4 and after.player.state == 4)}
        reward = float(sum(components.values()))
        if not math.isfinite(reward):
            raise FloatingPointError('Nonfinite native training reward')
        return reward, components, details


def bomb_transition_evidence(before, after, action):
    """Conservative attribution from adjacent measured snapshots, not a Bomb hook.

    Pickups may hide consumption. Same-step life/stage changes are ambiguous;
    never count them as a successful rescue or charge them as an observed use.
    """
    if isinstance(action, bool):
        raise ValueError('An actual recorded 36-way native action is required')
    try:
        action = operator.index(action)
    except TypeError as error:
        raise ValueError('An actual recorded 36-way native action is required') from error
    if not 0 <= action < 36:
        raise ValueError('An actual recorded 36-way native action is required')
    drop = max(0, before.bombs - after.bombs)
    life_drop = max(0, before.lives - after.lives)
    p, q = before.player, after.player
    same_world = (before.stage == after.stage and before.screen == after.screen == 7
                  and before.pause == after.pause == 2)
    supported = bool(action >= 18 and before.bombs > 0 and drop and not life_drop
                     and same_world and p and q and q.invulnerable_frames > 0)
    window = bool(p and p.state == 4 and p.deathbomb_timer > 0)
    recovery = bool(supported and window and q.state == 1 and after.lives == before.lives)
    return {'bomb_requested': action >= 18, 'inventory_decrement': drop,
            'life_decrements': life_drop, 'supported_spend_units': drop if supported else 0,
            'unattributed_inventory_decrement': 0 if supported else drop,
            'deathbomb_window_before': window, 'deathbomb_recovery_observed': recovery,
            'invulnerable_before': bool(p and p.invulnerable_frames > 0)}


class NativeEmergencyBombReward(NativeLifeAvoidanceReward):
    """Progress/life feedback with a cost for spending scarce Bomb inventory.

    No positive rescue bonus: deliberately touching bullets cannot farm it.
    A lower cost applies in a measured deathbomb window or when all current
    constant-movement forecasts collide. Neither condition selects an action.
    """
    def __init__(self, *, gamma, planner_config):
        super().__init__(gamma=gamma, planner_config=planner_config)
        self.emergency_planner = NativeHazardPlanner(replace(planner_config, horizon_frames=8.))

    def specification(self):
        result = super().specification()
        result.update(schema=EMERGENCY_BOMB_REWARD, survival_coefficient=0.,
            survival_damage_power_and_stage_rewards='elapsed survival reward removed; measured damage, Power and stage rewards retained',
            ordinary_bomb_spend_cost=2., emergency_bomb_spend_cost=.05,
            positive_deathbomb_bonus=0.,
            consumption_evidence='requested Bomb, positive net inventory decrement, no same-step life loss, same stage/running screen and subsequent invulnerability',
            consumption_limit='net lower bound only; pickups may hide use; ambiguous life/stage changes remain unattributed',
            emergency_definition='measured state4 with positive timer, or vulnerable state1 with all18 constant movement forecasts colliding within8 native frames',
            emergency_prediction_source=self.emergency_planner.specification(),
            emergency_forecast_is_native_oracle=False,
            rescue_evidence='supported spend plus state4 positive timer to state1 with unchanged lives and subsequent invulnerability')
        return result

    def bomb_feedback(self, before, after, action):
        evidence = bomb_transition_evidence(before, after, action)
        urgent = evidence['deathbomb_window_before']
        best_clearance = None
        if (evidence['supported_spend_units'] and not urgent and before.player
                and before.player.state == 1 and before.player.invulnerable_frames == 0):
            best = max(self.emergency_planner.action_metrics(before, a).clearance for a in range(18))
            if math.isnan(best) or best == -math.inf:
                raise FloatingPointError('Nonfinite emergency movement clearance')
            best_clearance = None if math.isinf(best) else best
            urgent = best <= 0.
        evidence.update(emergency_cost_eligible=urgent,
            best_forecast_movement_clearance_pixels=best_clearance,
            supported_spend_is_emergency=bool(evidence['supported_spend_units'] and urgent))
        cost = (.05 if urgent else 2.) * evidence['supported_spend_units']
        return -cost, evidence

    def evaluate(self, before, after, elapsed, *, terminated, action=None):
        _, components, details = super().evaluate(before, after, elapsed, terminated=terminated)
        spent_reward, evidence = self.bomb_feedback(before, after, action)
        details.update(schema=EMERGENCY_BOMB_REWARD, removed_elapsed_survival_reward=components['survival'],
                       bomb=evidence)
        components['survival'] = 0.
        components['bomb_inventory_spend'] = spent_reward
        reward = float(sum(components.values()))
        if not math.isfinite(reward):
            raise FloatingPointError('Nonfinite emergency-Bomb training reward')
        return reward, components, details


class NativeGeometryEmergencyReward(NativeEmergencyBombReward):
    """Same life/Bomb objective, with explicitly verified current V5 dimensions."""
    def __init__(self, *, gamma, planner_config):
        # This explicit profile targets the pinned original Reimu-A run. Do
        # not inherit the legacy V3 sensor's 4.0 speed / visual-field bounds.
        # Native calibration report SHA256 below records normal finite-life
        # original-game measurements; none of these values changes the input
        # prefix or controls the student's actions.
        planner_config=replace(planner_config,normal_speed=4.5,focused_speed=2.,
            left=-184.,right=184.,top=32.,bottom=432.)
        super().__init__(gamma=gamma,planner_config=planner_config)
        from .native_hazard_planner import GeometryNativeHazardPlanner
        self.planner=GeometryNativeHazardPlanner(planner_config)
        self.emergency_planner=GeometryNativeHazardPlanner(replace(planner_config,horizon_frames=8.))

    def specification(self):
        result=super().specification()
        result.update(schema=GEOMETRY_EMERGENCY_REWARD,
            geometry_required='original-th14-collision-geometry-v1',
            current_geometry='gated native circle/rectangle dimensions and corresponding measured player shapes',
            movement_reference={'schema':'original-th14-motion-calibration-v1',
                'report_sha256':'10ead065bc9082daf96d731f6f2acd0cc7f66a916cac38dd2e9e727db22135cd',
                'character':0,'subshot':0,
                'planner_overrides':{'normal_speed':4.5,'focused_speed':2.,
                    'left':-184.,'right':184.,'top':32.,'bottom':432.}},
            movement_limit='constant current Reimu-A speed model; unobserved later script speed changes remain unknown',
            future_geometry_known=False)
        return result

    def evaluate(self,before,after,elapsed,*,terminated,action=None):
        reward,components,details=super().evaluate(before,after,elapsed,terminated=terminated,action=action)
        details['schema']=GEOMETRY_EMERGENCY_REWARD
        return reward,components,details


def attach_training_reward(env, profile, *, gamma):
    """Configure a lazy environment; this never launches or attaches a game."""
    if profile == LEGACY_REWARD:
        return env
    if profile not in (AVOIDANCE_REWARD, EMERGENCY_BOMB_REWARD, GEOMETRY_EMERGENCY_REWARD):
        raise ValueError('Unknown explicit original-game training reward profile')
    if env.encoder.specification().get('schema') not in ('native-resource-hazard-observation-v3',
                                                       'native-collision-geometry-observation-v4'):
        raise ValueError('Avoidance reward requires complete measured v3 native hazard metadata')
    if (profile==GEOMETRY_EMERGENCY_REWARD
            and env.encoder.specification().get('schema')!='native-collision-geometry-observation-v4'):
        raise ValueError('Verified geometry reward requires the explicit V4 geometry sensor')
    if env.curriculum_first_hit:
        raise ValueError('Actual-life-loss feedback requires a normal finite-life native campaign')
    reward_type = {AVOIDANCE_REWARD:NativeLifeAvoidanceReward,
        EMERGENCY_BOMB_REWARD:NativeEmergencyBombReward,GEOMETRY_EMERGENCY_REWARD:NativeGeometryEmergencyReward}[profile]
    env.training_reward = reward_type(gamma=gamma, planner_config=env.encoder.planner.config)
    return env
