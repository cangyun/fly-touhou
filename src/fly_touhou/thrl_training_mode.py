"""Explicit episode protocol; no native process, Torch or curriculum scheduler."""
from __future__ import annotations

FULL_STORY_MODE = 'full-story-v1'
LEGACY_START_KEYS = ('start_bank', 'start_bank_sha256', 'segment_end', 'full_story_fraction')
SEED_PARTITION = 'uint16 seed modulo10; train0..5, development6..7, test8..9'


def is_full_story(config):
    mode = config.get('training_mode')
    if mode not in (None, FULL_STORY_MODE):
        raise ValueError('Unknown explicit training_mode: '+repr(mode))
    return mode == FULL_STORY_MODE


def validate_training_mode(config):
    """Legacy mode is opt-out by absence; a declared full run cannot fall back."""
    if not is_full_story(config):
        return False
    forbidden = [key for key in config
                 if key.startswith('curriculum_') or key in LEGACY_START_KEYS]
    if forbidden:
        raise ValueError('Full-story mode forbids curriculum/segment configuration: '+', '.join(sorted(forbidden)))
    required = {'first_hit_terminal':False, 'native_frame_horizon':None,
                'continues_allowed':False, 'policy_action_count':19,
                'reward_schema':'th14-completion-reward-v3'}
    for key, expected in required.items():
        if key not in config or config[key] != expected or (
                isinstance(expected, bool) and type(config[key]) is not bool):
            raise ValueError('Full-story mode requires explicit '+key+'='+repr(expected))
    for key in ('finite_native_lives', 'bombs_enabled'):
        if key in config and config[key] is not True:
            raise ValueError('Full-story mode preserves '+key)
    for key in ('wall_time_limit_seconds', 'max_episode_frames'):
        if config.get(key) is not None:
            raise ValueError('Full-story episodes cannot have '+key)
    burn_in = config.get('full_story_burn_in', 64)
    if type(burn_in) is not int or not 1 <= burn_in <= 128:
        raise ValueError('full_story_burn_in must be 1..128 policy decisions')
    if config.get('full_story_history_storage','compressed-v1') not in ('compressed-v1','raw-v1'):
        raise ValueError('Unknown full-story lossless recurrent history storage')
    return True


def episode_protocol(config):
    if not validate_training_mode(config):
        return {'training_mode':config.get('training_mode'), 'legacy_episode_protocol':True}
    result = dict(training_mode=FULL_STORY_MODE, policy_count=1,
        reset_start='fresh original Story opening', episode_end='native Game Over or six-stage clear',
        first_hit_terminal=False, native_frame_horizon=None, continues_allowed=False,
        current_segment_fraction=0., review_fraction=0., full_story_fraction=1.,
        reverse_window=False, curriculum_gate=False, target_boundary_truncation=False,
        train_development_test_same_episode_horizon=True,
        native_seed_partition=SEED_PARTITION,
        budget_end='collector/learner budget only; unfinished game is not a terminal or failure',
        budget_tail_bootstrap='current measured final observation under frozen behavior policy',
        recurrent_refresh='bounded current-parameter observation history after each Adam update',
        full_story_burn_in=config.get('full_story_burn_in',64),
        full_story_history_storage=config.get('full_story_history_storage','compressed-v1'))
    if config.get('development_objective') == 'avoidance-first-v1':
        result.update(train_development_test_same_episode_horizon=False,
            development_role='separate no-Bomb first-observed-collision diagnostic',
            development_native_frame_cap=config.get('avoidance_development_max_frames',4096),
            full_game_test_protocol_changed=False,
            training_Bomb_rule_changed=False)
    return result


def seed_split(seed):
    if type(seed) is not int or not 0 <= seed <= 65535:
        raise ValueError('Native seed must be uint16')
    return 'train' if seed % 10 < 6 else 'development' if seed % 10 < 8 else 'test'


def draw_native_seed(rng, split):
    if split not in ('train','development','test'):
        raise ValueError('Unknown native seed partition')
    while True:
        value = int(rng.integers(0,65536))
        if seed_split(value) == split:
            return value


def full_story_environment_options(config, count, role):
    if not validate_training_mode(config):
        return None
    if role not in ('train','teacher','development','test'):
        raise ValueError('Explicit full-story environment role required')
    if type(count) is not int or count < 1:
        raise ValueError('Positive full-story worker count required')
    from .th14_completion_protocol import reward_options
    split = 'train' if role in ('train','teacher') else role
    offset = {'train':0,'development':200000,'test':400000}[split]
    return [dict(training_mode=FULL_STORY_MODE, full_story_split=split,
                 full_story_seed=config['seed']+1009*i+offset, **reward_options(config))
            for i in range(count)]
