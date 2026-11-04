"""Retired desktop backend; retained only to fail old callers explicitly."""
from .background import BackgroundTouhouEnv
from .native import NativeError


class NativeTouhouEnv:
    def __init__(self, *args, **kwargs):
        raise NativeError('The desktop-input backend is disabled. Use BackgroundTouhouEnv '
                          'with an executable and its exact pinned profile.')

    @staticmethod
    def reward(before, after, elapsed):
        """Compatibility for the original pure reward contract."""
        reward, hit, _ = BackgroundTouhouEnv.reward(before, after, elapsed)
        return reward, hit
