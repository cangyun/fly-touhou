"""Original30-world task training with the existing labels-only V10 reference."""
from copy import deepcopy
from pathlib import Path


def make_task_reference_vector(config, output, *, vector_class=None):
    from .resource_teacher_v10 import REVISION
    from .thrl_inputs import SEPARATED_LASER_INPUT
    from .thrl_training_mode import validate_training_mode
    from .thrl_vector import THRLNativeVector
    import train_thrl_native as native_entry
    cfg = deepcopy(config)
    cfg['teacher_revision'] = REVISION
    cfg.setdefault('movement_guidance', {})['online_teacher_coefficient'] = .1
    if (not validate_training_mode(cfg) or cfg.get('input_schema') != SEPARATED_LASER_INPUT
            or cfg.get('headless') is not True or cfg.get('autonomous_bomb') is not True):
        raise ValueError('Natural full-story V5/source19 configuration required')
    kwargs = native_entry.make_env_kwargs(cfg, Path(output), 30, 'train')
    if len(kwargs) != 30 or any(k.get('online_teacher_revision') != REVISION
            or not k['headless'] or not k['hide_window'] or k['frame_skip'] != 4
            or k['policy_action_count'] != 19 or k['wall_time_limit_seconds'] is not None
            or k.get('full_story_split') != 'train' for k in kwargs):
        raise ValueError('Thirty existing labels-only natural reference workers required')
    # thrl_vector's existing _native_factory selects OnlineResourceReferenceEnv,
    # whose step accepts the student's19 actions and never substitutes a label.
    return (vector_class or THRLNativeVector)(kwargs, Path(output)/'workers')
