"""Versioned finite-task reward bookkeeping, standard library only.

The old binary completion contract remains distinct. The survival01 contract
adds0.1 actual elapsed native-frame fractions; it does not change termination.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path

COMPLETION = 'completion'
SURVIVAL01 = 'completion_survival01'
FINITE_MODES = (COMPLETION, SURVIVAL01)
SURVIVAL_CONFIG_SCHEMA = 'engineering-sb3-gru-native-survival01-v1'
SURVIVAL_PLAN_SCHEMA = 'engineering-GRU-PPO-fixed-evaluation-plan-survival01-v1'
SURVIVAL_REPORT_SCHEMA = 'engineering-GRU-PPO-heldout-seed-survival01-v1'
SURVIVAL_DEVELOPMENT_SCHEMA = 'engineering-gru-development-survival01-v1'
DESIGN_SHA = 'c39d93192ca91faeaa243cb1414b7c922d3a61814d4e3317e92f509c02c2c176'
PARENT_CORE_SHA = '503f8823ba5b996c6d7c5bc7e57831ce4a8bf09450a709a9e7cd0d80d49824bb'
POLICY_TRANSFER_NAMES = ('NativeFeatures', 'GRUStateAdapter', 'NativeGRUPolicy',
    'optimizer_steps', 'capture_rng_state', 'restore_rng_state', 'load_bc_actor')


def specification(mode):
    if mode not in FINITE_MODES:
        raise ValueError('Unknown finite avoidance reward mode')
    return {'mode': mode, 'version': 'completion-v1' if mode == COMPLETION else 'completion-survival01-v1',
        'formula': 'terminal_success+1_hit-1_other0' if mode == COMPLETION else 'r_completion+0.1*elapsed_native_frames/1024',
        'config_reward': 'terminal_success+1_hit-1_other0' if mode == COMPLETION else 'terminal_completion_plus_normalized_survival01_v1',
        'survival_coefficient': 0. if mode == COMPLETION else .1, 'horizon_frames': 1024,
        'hit_priority': True, 'external_interruption_is_success': False}


def completion_component(*, hit, terminal, frames, horizon=1024):
    """Same binary semantics as the frozen parent completion_reward function."""
    if hit:
        if not terminal:
            raise ValueError('A first-hit lesson must terminate on the hit')
        return -1.
    if terminal:
        if frames != horizon:
            raise ValueError('Non-hit native exit is not the declared task completion')
        return 1.
    if frames >= horizon:
        raise ValueError('The finite task failed to terminate at its horizon')
    return 0.


def step_reward(mode, *, hit, terminal, frames, elapsed_frames, horizon=1024):
    if mode not in FINITE_MODES:
        raise ValueError('Unknown finite avoidance reward mode')
    base = completion_component(hit=hit, terminal=terminal, frames=frames, horizon=horizon)
    if mode == COMPLETION:
        return base
    if (horizon != 1024 or type(elapsed_frames) is not int or type(frames) is not int
            or not 0 < elapsed_frames <= frames <= horizon):
        raise ValueError('Survival01 uses positive actual native frames inside the fixed1024-frame task')
    return base+.1*elapsed_frames/1024


def episode_return(mode, *, hit, frames, horizon=1024):
    base = completion_component(hit=hit, terminal=True, frames=frames, horizon=horizon)
    if mode == COMPLETION:
        return base
    if mode != SURVIVAL01 or horizon != 1024 or type(frames) is not int or not 0 < frames <= horizon:
        raise ValueError('Unknown or invalid finite episode return')
    return base+.1*frames/1024


def config_mode(config):
    """Only an explicitly versioned config may select the new reward."""
    if config.get('schema') == 'engineering-sb3-gru-native-v3':
        if (config.get('reward') != specification(COMPLETION)['config_reward']
                or config.get('reward_mode', COMPLETION) != COMPLETION):
            raise ValueError('Old binary config cannot carry the survival reward')
        return COMPLETION
    if config.get('schema') == SURVIVAL_CONFIG_SCHEMA:
        if (config.get('reward_mode') != SURVIVAL01
                or config.get('reward') != specification(SURVIVAL01)['config_reward']
                or config.get('reward_specification') != specification(SURVIVAL01)):
            raise ValueError('New reward config is missing its explicit formula/version')
        return SURVIVAL01
    raise ValueError('Unknown finite training config version')


def plan_mode(plan):
    if plan.get('schema') == 'engineering-GRU-PPO-fixed-evaluation-plan-v1':
        if plan.get('reward_mode', COMPLETION) != COMPLETION:
            raise ValueError('Old binary plan cannot select the survival reward')
        return COMPLETION
    if (plan.get('schema') == SURVIVAL_PLAN_SCHEMA and plan.get('reward_mode') == SURVIVAL01
            and plan.get('reward_specification') == specification(SURVIVAL01)):
        return SURVIVAL01
    raise ValueError('Unknown finite evaluation reward version')


def validate_episode_reward(row, mode=COMPLETION):
    if row.get('reward_mode') != mode:
        raise ValueError('Episode reward mode differs from the declared task')
    if mode == SURVIVAL01 and row.get('reward_specification') != specification(mode):
        raise ValueError('A survival episode must retain the exact reward specification')
    expected = episode_return(mode, hit=row['hit'], frames=row['frames'])
    if not math.isclose(row['return'], expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError('Episode return differs from its finite reward definition')


def validate_reward_trace(rows, *, mode=COMPLETION, episode=None):
    """Audit real compact rewards; an open prefix stays an open prefix."""
    if mode not in FINITE_MODES:
        raise ValueError('Unknown compact reward mode')
    frames, total, terminal = 0, 0., False
    for index, row in enumerate(rows, 1):
        if terminal or row['step'] != index:
            raise ValueError('Compact task continued after terminal or lost a decision')
        elapsed = row['next_frame']-row['frame']
        if elapsed != 4:
            raise ValueError('Finite compact actions must use4 actual native frames')
        frames += elapsed
        if mode == SURVIVAL01:
            if row.get('reward_mode') != mode or row.get('reward_version') != specification(mode)['version']:
                raise ValueError('Dense reward cannot masquerade as an old binary compact row')
        elif row.get('reward_mode', COMPLETION) != COMPLETION:
            raise ValueError('Old binary audit received another reward mode')
        expected = step_reward(mode, hit=row['hit'], terminal=row['task_terminal'],
                               frames=frames, elapsed_frames=elapsed)
        if not math.isclose(row['reward'], expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError('Compact reward differs from actual native-frame accounting')
        terminal = bool(row['task_terminal'])
        total += row['reward']
    if episode is not None:
        validate_episode_reward(episode, mode)
        if not terminal or episode['decisions'] != len(rows) or episode['frames'] != frames:
            raise ValueError('Compact reward and closed episode counters differ')
        if not math.isclose(episode['return'], total, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError('Compact reward sum differs from the native episode return')
    return {'decisions': len(rows), 'native_frames': frames, 'reward_sum': total,
            'terminal': terminal, 'outcome': 'known' if terminal else 'unknown_open_prefix'}


def _record(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def _policy_ast(path):
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    nodes = {node.name: node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
    return {name: ast.dump(nodes[name], include_attributes=False) for name in POLICY_TRANSFER_NAMES}


def reward_source_derivation(design_path, current_core):
    """Explicit BC source derivation, never a PPO checkpoint SHA exemption."""
    design_record = _record(design_path)
    if design_record['sha256'] != DESIGN_SHA:
        raise ValueError('The selected survival01 source design changed')
    design = json.loads(Path(design_path).read_text(encoding='utf-8'))
    parent_plan_record = _record(design['parent_plan']['path'])
    if parent_plan_record != design['parent_plan']:
        raise ValueError('The frozen reward-parent plan changed')
    parent_plan = json.loads(Path(parent_plan_record['path']).read_text(encoding='utf-8'))
    parent = next(record for record in parent_plan['runtime_sources'] if Path(record['path']).name == 'standard_gru_ppo.py')
    if parent['sha256'] != PARENT_CORE_SHA or _record(parent['path']) != parent:
        raise ValueError('The inherited BC/native policy source changed')
    old_ast, new_ast = _policy_ast(parent['path']), _policy_ast(current_core)
    if old_ast != new_ast:
        raise ValueError('Reward-only derivation changed network, BC transfer or RNG/optimizer state helpers')
    return {'schema': 'reward-only-BC-policy-source-derivation-v1', 'design': design_record,
            'parent_core': parent, 'current_core': _record(current_core),
            'policy_transfer_functions_AST_identical': list(POLICY_TRANSFER_NAMES),
            'policy_AST_sha256': hashlib.sha256(json.dumps(old_ast, sort_keys=True).encode()).hexdigest(),
            'old_PPO_checkpoint_source_SHA_exemption': False}
