"""Shared original-game diagnostic contract and sealed episode collection."""
from pathlib import Path
from types import SimpleNamespace
import gzip
import json
import math

import numpy as np
import torch

from .background import action_mask, BackgroundTouhouEnv
from .dense_geometry_ppo import forecast_regret_batch
from .diagnostic_gru import replay_history
from .experiment import file_sha256
from .first_hit_policy_gradient import first_hit_reward, finite_lesson_returns
from .geometry_observation import GeometryObservationEncoder
from .native_teacher_feedback import make_teacher
from .resource_teacher import ResourceTeacher
from .whole_brain_ppo import campaign_environment_for_policy, _native_observation, _campaign_runtime_audit


def write_json(path, value, *, exclusive=False):
    if exclusive:
        with Path(path).open('x', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
    else:
        temporary = Path(path).with_suffix('.tmp')
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        temporary.replace(path)


def load_protocol(path):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if value['schema'] != 'native-matched-list-diagnostic-v1':
        raise ValueError('Explicit immutable diagnostic protocol required')
    if file_sha256(value['game_exe']) != value['game_exe_sha256']:
        raise ValueError('Original executable identity changed')
    for item in value['protected_files']:
        if file_sha256(item['path']) != item['sha256']:
            raise ValueError(f'Protected shared source changed: {item["path"]}')
    encoder = GeometryObservationEncoder.from_specification(value['encoder'])
    if encoder.size != 1911 or encoder.config.include_action_risk:
        raise ValueError('The matched list diagnostic uses the existing raw1911 sensor')
    return value, encoder


class BeforeStateGeometry:
    """The former feedback's exact clearances, without asking it for an action."""
    def __init__(self):
        self.planner = make_teacher().planner
        self.contacts = ResourceTeacher(self.planner)

    def reset(self):
        self.planner.reset()

    def query(self, snapshot):
        self.planner._prepare(snapshot)
        p = snapshot.player
        gameplay = (p is not None and snapshot.screen == 7 and snapshot.pause == 2
            and snapshot.extra.get('native_gameover') is not True
            and 'native_dialogue_object' in snapshot.extra
            and snapshot.extra['native_dialogue_object'] in (0, None))
        if not gameplay or p.state != 1 or p.invulnerable_frames > 0:
            return {'reason': 'non_gameplay_or_not_vulnerable', 'forecast_is_native_oracle': False}
        enemies = self.contacts._enemies(snapshot, snapshot.extra.get('resources', {}))
        clearances = []
        for action in range(18):
            bullet = self.planner.action_metrics(snapshot, action).clearance
            path = self.planner._path(snapshot, action)[0]
            contact = self.contacts._contact_clearance(snapshot, path, enemies)
            value = min(bullet, contact)
            clearances.append(None if value == math.inf else float(value))
        return {'clearances': clearances, 'forecast_is_native_oracle': False,
                'source': 'same-before-state-geometric-estimate;not-actor-input'}


def make_environment(protocol, encoder, output, *, title, fine_start=False):
    env = campaign_environment_for_policy(SimpleNamespace(encoder=encoder), protocol['game_exe'],
        protocol['profile'], difficulty=3, frame_skip=4, max_episode_frames=1024,
        private_appdata=Path(output)/'game_userdata', dialogue_skip=True, parallel_startup=True,
        responsive_wait=True, hide_window=True, window_title=title)
    env.curriculum_first_hit = True
    if fine_start:
        from .fixed_start_controller import enable_fine_start
        enable_fine_start(env)
    return env


def checked_snapshot(snapshot, protocol):
    snapshot.validate()
    if (snapshot.extra.get('desktop_input') is not False or snapshot.difficulty != 3
        or snapshot.character != 0 or snapshot.subshot != 0
        or any(snapshot.extra.get(k) != 0 for k in ('continues', 'run_mode', 'replay_playback'))):
        raise ValueError('Diagnostic must remain original Lunatic/ReimuA/live/noContinue/noDesktop')
    if snapshot.extra.get('exe_sha256', protocol['game_exe_sha256']) != protocol['game_exe_sha256']:
        raise ValueError('Original executable differs from protocol')


def collect_episode(env, encoder, protocol, model, sampler, stream, *, role, identity,
                    policy_kind='gru', logger=None, reference_geometry=False):
    value, info = env.reset()
    snapshot = info['snapshot']
    checked_snapshot(snapshot, protocol)
    observation = _native_observation(value, encoder, snapshot, info)
    if logger is not None:
        logger.record_native_context(snapshot, info)
        logger.event('diagnostic_episode_started', {'episode_id': identity, 'role': role,
            'policy_kind': policy_kind, 'pid': env.pid, 'actual_game_rng': snapshot.extra.get('rng'),
            'game_rng_seed_applied': False, 'runtime': _campaign_runtime_audit(env)})
    geometry = BeforeStateGeometry()
    oracle = None
    if reference_geometry:
        from .avoidance_teacher import AvoidanceOnlyFeedback
        oracle = AvoidanceOnlyFeedback()
    if model is not None:
        with torch.no_grad():
            state = model.prime(torch.as_tensor(observation, device=model.device)[None],
                                protocol['warmup_repeated_observation_passes'])
    frames, vulnerable, observations, states, logps, actions, rows = 0, 0, [], [], [], [], []
    episode_return = 0.
    while True:
        before = snapshot
        detail = geometry.query(before)
        if oracle is not None:
            reference = oracle.query(before)[3]
            if reference.get('clearances') != detail.get('clearances'):
                raise ValueError('Geometric estimate changed from the original frozen method')
        if model is not None:
            states.append(state.detach().cpu()[0].clone())
            with torch.no_grad():
                logits, state = model(torch.as_tensor(observation, device=model.device)[None], state)
                lp = logits[0].double().log_softmax(-1).cpu()
        else:
            lp = torch.full((18,), -math.log(18), dtype=torch.float64)
        if policy_kind == 'static':
            action = 0
        else:
            action = int(torch.multinomial(lp.exp(), 1, generator=sampler))
        mask = action_mask(action)
        if not mask & 1 or mask & 2 or not 0 <= action < 18:
            raise ValueError('Diagnostic must hold shot and never request Bomb')
        value, native_reward, terminated, truncated, info = env.step(action)
        snapshot = info['snapshot']
        checked_snapshot(snapshot, protocol)
        elapsed = info['elapsed_frames']
        if elapsed != 4 or snapshot.frame-before.frame != elapsed:
            raise ValueError('Original action interval differs from frozen four ticks')
        if info['hit'] != BackgroundTouhouEnv.reward(before, snapshot, elapsed)[1]:
            raise ValueError('First-hit event differs from original measured collision gate')
        if env.bomb_actions:
            raise ValueError('Unexpected original Bomb request')
        row = {'record_type': 'transition', 'role': role, 'episode_id': identity,
            'episode_step': len(actions), 'snapshot': before.to_dict(), 'next_snapshot': snapshot.to_dict(),
            'action': action, 'requested_native_input_mask': mask, 'action_source': policy_kind,
            'elapsed_frames': elapsed, 'hit': bool(info['hit']), 'terminated': bool(terminated),
            'truncated': bool(truncated), 'teacher_action_calls': 0, 'teacher_label_queries': 0,
            'safety_takeovers': 0, 'movement_details': detail,
            'conditional_logp18': None if policy_kind == 'static' else lp.tolist(),
            'selection_rule': 'fixed-action0' if policy_kind == 'static' else 'categorical-sampling',
            'selected_conditional_log_probability': 0. if policy_kind == 'static' else float(lp[action]),
            'environment_reward_not_optimized': float(native_reward),
            'geometry_reference_parity_checked': oracle is not None}
        reward, components = first_hit_reward(row)
        row.update(reward=reward, reward_components=components)
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
        observations.append(observation.copy()); logps.append(lp); actions.append(action); rows.append(row)
        episode_return += reward
        frames += elapsed
        if before.player.state == 1 and before.player.invulnerable_frames == 0:
            vulnerable += elapsed
        observation = _native_observation(value, encoder, snapshot, info)
        if terminated or truncated:
            break
    result = {'episode_id': identity, 'role': role, 'policy_kind': policy_kind,
        'frames': frames, 'vulnerable_frames': vulnerable, 'decisions': len(actions),
        'hit': bool(info['hit']), 'horizon_completed_without_hit': bool(not info['hit'] and frames == 1024),
        'return': episode_return, 'terminated': bool(terminated), 'truncated': bool(truncated),
        'stage_history': env.stage_history, 'teacher_actions': 0, 'safety_takeovers': 0,
        'bomb_actions': env.bomb_actions, 'final_acceptance': False}
    if logger is not None:
        logger.event('diagnostic_episode_finished', result)
    behavior = {'observations': torch.from_numpy(np.asarray(observations, dtype=np.float32)),
        'old_logp18': torch.stack(logps), 'actions': torch.tensor(actions),
        'episode_ids': [identity]*len(actions), 'rows': rows}
    if states:
        behavior['states_before'] = torch.stack(states)
    return result, behavior


def prepare_behavior(episodes, critic):
    rows = [r for _, data in episodes for r in data['rows']]
    result = {name: torch.cat([data[name] for _, data in episodes])
        for name in ('observations', 'old_logp18', 'actions', 'states_before')}
    result['episode_ids'] = [identity for _, data in episodes for identity in data['episode_ids']]
    result['old_selected_logp'] = result['old_logp18'].gather(1, result['actions'][:, None]).squeeze(1)
    rewards = np.asarray([first_hit_reward(row)[0] for row in rows])
    ended = np.asarray([row['terminated'] or row['truncated'] for row in rows])
    returns = torch.from_numpy(finite_lesson_returns(rewards, ended))
    with torch.no_grad():
        values = critic(result['observations'].to(next(critic.parameters()).device)).double().cpu()
    advantages = returns-values
    advantages = ((advantages-advantages.mean())/advantages.std(unbiased=False).clamp_min(1e-8)).clamp(-5, 5)
    costs, weights, active = forecast_regret_batch(rows)
    result.update(rewards=torch.from_numpy(rewards), finite_returns=returns,
        baseline_values=values, advantages=advantages, forecast_costs=torch.from_numpy(costs),
        danger_weights=torch.from_numpy(weights), discriminative=torch.from_numpy(active))
    return result


def wilson(successes, count):
    z = 1.959963984540054
    p = successes/count
    denominator = 1+z*z/count
    center = (p+z*z/(2*count))/denominator
    half = z*math.sqrt(p*(1-p)/count+z*z/(4*count*count))/denominator
    return [max(0., center-half), min(1., center+half)]


def summarize_episodes(episodes):
    groups = {}
    for kind in sorted({e['policy_kind'] for e in episodes}):
        subset = [e for e in episodes if e['policy_kind'] == kind]
        successes = sum(e['horizon_completed_without_hit'] for e in subset)
        groups[kind] = {'episodes': len(subset), 'successes': successes,
            'no_hit_rate': successes/len(subset), 'no_hit_rate_Wilson95': wilson(successes, len(subset)),
            'mean_capped_first_hit_frames': float(np.mean([e['frames'] for e in subset])),
            'median_first_hit_frames': float(np.median([e['frames'] for e in subset])),
            'frames': [e['frames'] for e in subset]}
    return groups
