"""Resource-aware completion reward with measured protection and bounded HP credit.

This is a new experiment, not a reinterpretation of saved v2 trajectories.
The native event reader and all original game rules remain unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
import numpy as np

from .th14_completion_protocol import CompletionRewardModel
from .thrl_rewards import THRLRewardModel, THRLRewardResult, corner_penalty, update_bomb_tracking

RECIPE = 'th14-completion-v3'
REWARD_SCHEMA = 'th14-completion-reward-v3'


@dataclass(frozen=True)
class ResourceCompletionResult(THRLRewardResult):
    diagnostics: dict

    def info(self):
        return {**super().info(), 'reward_schema': REWARD_SCHEMA,
                'reward_scales': (1., 1., 1.), 'reward_diagnostics': self.diagnostics}


class ResourceCompletionReward(CompletionRewardModel):
    """Deaths cost 2; ordinary Bombs .5; verified pre-action deathbombs .1.

Safe progress is credited only across observations that start/end vulnerable
and have no loss/Bomb event. Protection/respawn/dialogue earns no time credit.
Boss HP credit is at most .2 per observed native phase, independent of raw HP.
No distance-to-danger or predicted teacher safety score enters the reward.
"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.previous_snapshot = None
        self.phase_credit = {}

    def specification(self):
        return {**super().specification(), 'schema': REWARD_SCHEMA,
            'survival': '+.001 per four vulnerable native frames; capped at four per decision',
            'death': -2., 'Bomb_used': -.5, 'confirmed_deathbomb': -.1,
            'deathbomb_rule': 'pre-action player state4 and remaining>0, actual native Bomb counter delta, and no committed Miss',
            'uncertain_Bomb_classification': 'ordinary cost; never inferred from a survived trajectory',
            'unused_Bombs_on_death': 0., 'boss_damage_per_HP': None,
            'boss_damage': '.2 * positive same-phase HP decrease / measured native hp_max',
            'boss_damage_phase_cap': .2, 'boss_damage_attribution': 'includes shot and Bomb; not separable',
            'midboss_defeat': 0., 'boss_disappearance_bonus': 0.,
            'boss_phase': 'stage + native identity + ECL subroutine + hp_max; HP refill resets comparison baseline, never cumulative credit',
            'boss_phase_cap_scope': 'same observed key across missing/intangible observations; bounded per-stage ledger; may undercredit native phase reuse',
            'multi_boss_role_coverage': 'same-frame complete native role list may classify more than two boss-flag objects; auxiliary objects never receive primary HP credit',
            'invalid_HP': 'nonpositive max or HP outside [0,max] gives zero credit and diagnostic',
            'corner_maximum': -.001, 'terminal': '-2 gameover in addition to miss; +5 true clear; 0 external cut',
            'time_credit_during_protection': 0., 'reward_parameters_are_unvalidated_hypotheses': True}

    def reset(self, snapshot):
        info = THRLRewardModel.reset(self, snapshot)
        self.rebase_snapshot(snapshot)
        return {**info, 'reward_schema': REWARD_SCHEMA}

    def rebase_snapshot(self, snapshot):
        """Prefix replay establishes state; its history never generates reward."""
        self.previous_snapshot = snapshot
        self.phase_credit = {}
        self.phase_credit_stage = snapshot.stage

    @classmethod
    def _classified_extra_bosses(cls, snapshot):
        """Verify all native boss identities before exempting a legacy slot cap.

        v3 HP credit already uses canonical ECL primary roles, not the inherited
        two HP slots. TH14 can additionally expose many boss-flag auxiliary
        heads. Their measured identities/HP/flags must all be accounted for;
        an unknown third object is still rejected, never silently dropped.
        """
        roles = snapshot.extra.get('thrl_events', {}).get('ecl_roles', {})
        if (roles.get('schema') != 'th14-thrl-ecl-roles-v1' or roles.get('observed') is not True
                or roles.get('midboss_observed') is not True or roles.get('errors') != []
                or any(roles.get(k) != snapshot.frame for k in
                       ('snapshot_frame', 'native_frame_before', 'native_frame_after'))):
            return False
        groups = [roles.get(k) for k in ('main_bosses', 'midbosses', 'auxiliary_bosses')]
        complete = roles.get('enemies')
        if not all(isinstance(g, list) for g in groups) or not isinstance(complete, list):
            return False
        main, middle, auxiliary = groups
        primary = main + middle
        if (len(primary) != 1 or not auxiliary
                or roles.get('main_thread_role') != ('main_boss' if main else 'midboss')):
            return False
        if any(not isinstance(row, dict) for row in complete + primary + auxiliary):
            return False
        identities = [row.get('id') for row in complete]
        if any(type(i) is not int or not 0 < i < 2**32 for i in identities) or len(set(identities)) != len(identities):
            return False
        complete_by_id = dict(zip(identities, complete))
        expected = {e.id: e for e in snapshot.enemies if e.boss}
        classified = primary + auxiliary
        classified_ids = [row.get('id') for row in classified]
        if (any(type(i) is not int for i in classified_ids)
                or len(set(classified_ids)) != len(classified_ids)
                or set(classified_ids) != set(expected)
                or {row['id'] for row in complete if row.get('boss') is True} != set(expected)):
            return False
        for row in classified:
            enemy = expected[row['id']]
            if (row.get('boss') is not True or type(row.get('hp')) is not int or row['hp'] != enemy.hp
                    or type(row.get('intangible')) is not bool or row['intangible'] != enemy.intangible
                    or type(row.get('subboss_id')) is not int):
                return False
            raw = complete_by_id[row['id']]
            if any(raw.get(k) != row.get(k) for k in ('hp', 'hp_max', 'boss', 'intangible', 'subboss_id', 'ecl_sub_name')):
                return False
        if primary[0]['subboss_id'] != 0 or any(row['subboss_id'] == 0 for row in auxiliary):
            return False
        _, error = cls._bosses(snapshot)
        return error is None

    def _frame(self, snapshot, *, end=0):
        current = super()._frame(snapshot, end=end)
        legacy_limit = 'more_than_two_unclassified_boss_objects'
        if legacy_limit in current.unsupported and self._classified_extra_bosses(snapshot):
            current = replace(current, unsupported=tuple(x for x in current.unsupported if x != legacy_limit))
        return current

    @staticmethod
    def _bosses(snapshot):
        roles = snapshot.extra.get('thrl_events', {}).get('ecl_roles', {})
        if roles.get('observed') is not True:
            return {}, 'roles-unavailable'
        if roles.get('schema') != 'th14-thrl-ecl-roles-v1':
            return {}, 'roles-schema-mismatch'
        if any(roles.get(k) != snapshot.frame for k in ('native_frame_before', 'native_frame_after')):
            return {}, 'roles-frame-mismatch'
        mains, mids = roles.get('main_bosses', []), roles.get('midbosses', [])
        if not isinstance(mains, list) or not isinstance(mids, list):
            return {}, 'invalid-boss-rows'
        rows = mains + mids
        output = {}
        identities = set()
        for row in rows:
            if not isinstance(row, dict):return {}, 'invalid-boss-row'
            hp, maximum = row.get('hp'), row.get('hp_max')
            name = row.get('ecl_sub_name')
            identity = row.get('id')
            if (type(hp) is not int or type(maximum) is not int or not 0 < maximum <= 2**31-1
                    or not 0 <= hp <= maximum or not isinstance(name, str) or not name):
                return {}, 'invalid-hp-or-phase'
            if type(identity) is not int or not 0 < identity < 2**32 or identity in identities:
                return {}, 'invalid-or-duplicate-boss-identity'
            identities.add(identity)
            if row.get('intangible'):
                continue
            key = (snapshot.stage, identity, name, maximum)
            output[key] = hp
        return output, None

    def _damage(self, before, after):
        old, error_before = self._bosses(before)
        new, error_after = self._bosses(after)
        damage = fraction = 0.
        phase_changes = 0
        # Missing, intangible or refilled observations must not replenish the
        # same native-key reward budget. Only a new stage clears the ledger.
        if self.phase_credit_stage != after.stage:
            self.phase_credit = {}
            self.phase_credit_stage = after.stage
        for key, hp in new.items():
            if key not in self.phase_credit:
                # A corrupt/reused pointer stream cannot grow memory forever.
                # On saturation, skip new keys instead of evicting old budgets.
                if len(self.phase_credit) >= 4096:
                    error_after = 'phase-credit-ledger-full'
                    continue
                self.phase_credit[key] = 0.
            if key not in old:
                phase_changes += 1
                continue
            previous = old[key]
            if hp > previous:
                phase_changes += 1
                continue
            delta = max(0, previous - hp)
            portion = min(delta / key[-1], max(0., 1. - self.phase_credit[key]))
            self.phase_credit[key] += portion
            damage += delta
            fraction += portion
        return .2 * fraction, dict(boss_raw_hp_damage=damage,
            boss_hp_fraction=fraction, boss_phase_changes=phase_changes,
            boss_hp_diagnostic=error_after or error_before)

    def transition(self, snapshot, *, end=0):
        if end not in (0,1,2):raise ValueError('Unknown native terminal kind')
        before_snapshot, before = self.previous_snapshot, self.previous
        if before_snapshot is None or before is None:
            raise ValueError('Version3 reward requires a measured reset or prefix rebase')
        current = self._frame(snapshot, end=end)
        self._require(current.unsupported)
        for name in ('misses', 'bombs_used', 'point_items'):
            old_count, new_count = getattr(before, name), getattr(current, name)
            if old_count is None or new_count is None:
                raise ValueError('Version3 reward requires observed cumulative '+name)
            if new_count < old_count:
                raise ValueError('Native cumulative '+name+' decreased inside one run')
        # Prefix-bank snapshots still serialize the inherited source tracker.
        current = update_bomb_tracking(before, current)
        misses = max(0, current.misses - before.misses)
        bombs = max(0, current.bombs_used - before.bombs_used)
        p, q = before_snapshot.player, snapshot.player
        confirmed_deathbomb = bool(bombs and not misses and p is not None and p.state == 4 and p.deathbomb_timer > 0)
        deathbombs = int(confirmed_deathbomb)  # One pre-action window proves at most one event.
        ordinary_bombs = bombs - deathbombs
        vulnerable = bool(p is not None and q is not None and p.state == q.state == 1
            and p.invulnerable_frames <= 0 and q.invulnerable_frames <= 0
            and not before_snapshot.dialogue and not snapshot.dialogue and not misses and not bombs)
        frames = min(4, max(0, snapshot.frame - before_snapshot.frame)) if vulnerable else 0
        damage, damage_info = self._damage(before_snapshot, snapshot)
        phi = lambda value: .2 * min(1., max(0., value / 400.))
        components = dict(vulnerable_progress=.001 * frames / 4.,
            corner=-.002 * corner_penalty(current.x, current.y) if vulnerable else 0.,
            death=-2. * misses, native_gameover=-2. if end == 1 else 0.,
            native_clear=5. if end == 2 else 0., boss_damage=damage,
            bomb_used=-.1 * deathbombs - .5 * ordinary_bombs,
            power_potential=self.gamma * (0. if end else phi(current.power)) - phi(before.power))
        vector = np.asarray([sum(components[k] for k in ('vulnerable_progress', 'corner', 'death',
            'native_gameover', 'native_clear')), damage,
            components['bomb_used'] + components['power_potential']], dtype=np.float32)
        if not np.isfinite(vector).all() or not all(math.isfinite(v) for v in components.values()):
            raise FloatingPointError('Nonfinite completion-v3 reward')
        diagnostics = dict(**damage_info, actual_bombs_used=bombs,
            confirmed_deathbombs=deathbombs,
            ordinary_or_unconfirmed_bombs=ordinary_bombs,
            native_misses=misses, vulnerable_frames=frames,
            protected_decision=int(p is not None and p.invulnerable_frames > 0),
            bomb_inventory_before=before_snapshot.bombs,
            invulnerability_frames_before=None if p is None else p.invulnerable_frames,
            deathbomb_frames_before=None if p is None else p.deathbomb_timer)
        self.previous, self.previous_snapshot = current, snapshot
        self.steps += 1
        return ResourceCompletionResult(vector.copy(), vector, components, current.unsupported, current, diagnostics)
