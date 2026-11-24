"""Small positional shaping on top of the unchanged native v4 event rewards.

The discounted potential is evaluated on every transition, including the true
terminal, so leaving and returning cannot mint extra shaping reward. Only the
upper-residence cost has safety/resource/pattern exceptions. It is a policy
preference, not a claim that a geometric forecast certifies native safety.
"""
import numpy as np
import torch

REVISION = 'lower-field-potential-and-conditional-residence-v1'
DEFAULTS = dict(target_y=256., potential_scale=.05, upper_cost_per_four_frames=.0005)


class LowerFieldReward:
    def __init__(self, *, gamma=.999, **settings):
        self.gamma = float(gamma)
        self.settings = dict(DEFAULTS, **settings)
        if set(self.settings) != set(DEFAULTS):
            raise ValueError('Unknown lower-field reward option')
        if not all(np.isfinite(v) for v in self.settings.values()):
            raise ValueError('Finite lower-field reward options required')
        if not (32 < self.settings['target_y'] < 432 and 0 < self.gamma <= 1
                and self.settings['potential_scale'] >= 0 and self.settings['upper_cost_per_four_frames'] >= 0):
            raise ValueError('Invalid lower-field reward scale or boundary')
        self.previous_potential = None

    def specification(self):
        return dict(revision=REVISION, **self.settings, gamma=self.gamma, value_channel=0,
            potential='Phi(y)=-scale*clip((target_y-y)/(target_y-32),0,1); delta=gamma*Phi(next)-Phi(before); true-terminal Phi=0',
            upper_cost='-cost*height_fraction*min(vulnerable_frames,4)/4 when trusted geometry supports a return without an exception',
            no_teacher_action_or_probability_used=True, no_execution_mask=True,
            native_Miss_Bomb_HP_clear_terms_unchanged=True)

    def height(self, y):
        target = self.settings['target_y']
        return float(np.clip((target - y) / (target - 32.), 0., 1.))

    def potential(self, snapshot):
        if snapshot.player is None:
            return 0.
        return -self.settings['potential_scale'] * self.height(snapshot.player.y)

    def reset(self, snapshot):
        self.previous_potential = self.potential(snapshot)

    def residence_eligible(self, encoding):
        if encoding is None:
            return False
        d = encoding['details']
        if not d['normal'] or not np.asarray(d['trusted'], bool).all():
            return False
        forecasts = encoding.get('source_native_forecasts') or {}
        if forecasts.get('reactive_bullets'):
            return False
        margins = np.asarray(d['margins'], np.float32)
        # Preserve late-game turning routes even if a one-leg return exists.
        if encoding.get('stage', 0) >= 5 and np.any((margins[0] > 1) & (margins[2] <= 1)):
            return False
        from .compact_lower_field import position_targets
        f = torch.as_tensor(np.asarray(encoding['features'], np.float32))[None, None]
        probability = torch.zeros(1, 1, 19)
        probability[..., :18] = 1 / 18
        eligible, _ = position_targets(f, torch.as_tensor(margins)[None, None],
            torch.ones(1, 1, 3, dtype=torch.bool), torch.ones(1, 1, dtype=torch.bool),
            probability, torch.ones(1, 1), exceptions=torch.zeros(1, 1, dtype=torch.bool),
            resource_support_required=False)
        return bool(eligible.item())

    def transition(self, before, after, encoding, diagnostics, *, terminal=False):
        if self.previous_potential is None:
            raise RuntimeError('Lower-field reward must be reset with the native opening')
        following = 0. if terminal else self.potential(after)
        potential = self.gamma * following - self.previous_potential
        self.previous_potential = following
        frames = min(4, max(0, diagnostics.get('vulnerable_frames', 0)))
        height = self.height(before.player.y) if before.player is not None else 0.
        eligible = bool(frames and height > 0 and encoding is not None
            and encoding.get('frame') == before.frame and self.residence_eligible(encoding))
        residence = -self.settings['upper_cost_per_four_frames'] * height * frames / 4 if eligible else 0.
        return dict(lower_field_potential=float(potential), avoidable_upper_residence=float(residence)), dict(
            lower_field_residence_eligible=int(eligible), lower_field_upper_frames=frames if eligible else 0,
            lower_field_reward_revision=REVISION)


def apply_reward(env, info, components, diagnostics):
    """Keep Gym reward, three-head returns, episode totals and logs identical."""
    amount = sum(components.values())
    for key in ('reward_vector', 'reward_raw_vector'):
        value = np.array(info[key], dtype=np.float32, copy=True)
        value[0] += amount
        info[key] = value
    env._episode_returns[0] += amount
    env._episode_raw_returns[0] += amount
    info['reward_components_raw'].update(components)
    info['reward_diagnostics'].update(diagnostics)
    info['reward_shaping_revision'] = REVISION
    totals = env._episode_reward_diagnostics
    if totals is not None:
        for key, value in components.items():
            sums = totals['reward_component_sums']
            sums[key] = sums.get(key, 0.) + value
        for key in ('lower_field_residence_eligible', 'lower_field_upper_frames'):
            totals[key] = totals.get(key, 0) + diagnostics[key]
        totals['lower_field_reward_revision'] = REVISION
    if 'episode_summary' in info:
        info['episode_summary'].update(reward_vector=env._episode_returns.tolist(),
            raw_reward_vector=env._episode_raw_returns.tolist(), reward_shaping_revision=REVISION,
            reward_diagnostics=env._reward_diagnostics_copy())
    return float(np.sum(info['reward_vector'], dtype=np.float32))
