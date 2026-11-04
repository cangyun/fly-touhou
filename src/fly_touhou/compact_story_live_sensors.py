"""Enable the measured EX hold prefix on an existing native environment."""
from pathlib import Path

HOLD_REVISION = 'th14-ex-position-hold-v1'
POLAR4_REVISION = 'th14-ex8-polar4-candidate-v1'


def enable_position_hold(env, profile_path, *, polar4=False):
    """Keep game, encoder history and policy state; change only reader hooks.

    The existing reader captures the countdown before certifying its stationary
    prefix. No cached observation is relabeled: new native reads use the hook.
    """
    controller = env.controller
    reader = controller.reader
    profiles = {id(p): p for p in (env.profile, controller.profile, reader.profile)}
    before = [dict(p.hooks or {}) for p in profiles.values()]
    hooks = dict(native_position_hold_forecast=HOLD_REVISION)
    if polar4:
        hooks['native_ex8_polar4_forecast'] = POLAR4_REVISION
    for profile in profiles.values():
        profile.hooks = dict(profile.hooks or {}, **hooks)
    old_path = str(env.profile_path)
    env.profile_path = Path(profile_path)
    if hasattr(controller, 'profile_path'):
        controller.profile_path = Path(profile_path)
    return dict(before_hooks=before, revision=HOLD_REVISION, enabled_hooks=hooks,
                previous_profile_path=old_path, profile_path=str(env.profile_path),
                distinct_profile_objects=len(profiles), game_reset=False,
                cached_observation_rewritten=False, policy_parameters_changed=False)
