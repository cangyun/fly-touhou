"""Completion v4: remove the incentive to prolong an attackable Boss phase.

Position is guidance, not a renewable score. All v3 native event costs, three
value channels, HP budgets and resource potential remain unchanged.
"""
from dataclasses import dataclass
from .th14_completion_v3 import ResourceCompletionReward, ResourceCompletionResult

RECIPE = 'th14-completion-v4'
REWARD_SCHEMA = 'th14-completion-reward-v4'


@dataclass(frozen=True)
class CombatCompletionResult(ResourceCompletionResult):
    def info(self):
        return {**super().info(), 'reward_schema': REWARD_SCHEMA}


class CombatCompletionReward(ResourceCompletionReward):
    def specification(self):
        return {**super().specification(), 'schema': REWARD_SCHEMA,
            'survival': '+.001 per four vulnerable native frames outside an observed attackable primary Boss phase; zero during that phase',
            'attackable_boss_time_credit': 0.,
            'attackable_boss_rule': 'validated native main/midboss role, positive HP, not intangible, in either endpoint',
            'position_reward': 0., 'boss_alignment_reward': 0.,
            'v3_unchanged': 'HP credit, Miss/Bomb/clear costs, corners, power potential, three value channels'}

    def reset(self, snapshot):
        return {**super().reset(snapshot), 'reward_schema': REWARD_SCHEMA}

    def transition(self, snapshot, *, end=0):
        # Use the existing canonical native roles, not a visible target chosen
        # by the policy or an inferred absence of damage. Waiting without firing
        # therefore cannot turn an attackable phase into a survival section.
        previous = self.previous_snapshot
        old, old_error = self._bosses(previous) if previous is not None else ({}, 'no-reset')
        new, new_error = self._bosses(snapshot)
        attackable = any(hp > 0 for hp in old.values()) or any(hp > 0 for hp in new.values())
        base = super().transition(snapshot, end=end)
        removed = base.components['vulnerable_progress'] if attackable else 0.
        components = dict(base.components)
        components['vulnerable_progress'] -= removed
        vector = base.reward_vector.copy()
        vector[0] -= removed
        diagnostics = dict(base.diagnostics, attackable_boss_phase=int(attackable),
            boss_time_credit_removed=removed,
            boss_time_gate_diagnostic=new_error or old_error)
        return CombatCompletionResult(vector.copy(), vector, components,
            base.unsupported, base.frame, diagnostics)
