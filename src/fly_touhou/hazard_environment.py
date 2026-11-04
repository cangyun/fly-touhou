"""Opt-in same-frame hazard metadata on the responsive original-game backend.

MetadataResourceCampaignEnv keeps the frozen v2 observation vector. Raw fields
remain in each trajectory for an explicitly versioned future encoder. The v3
environment imports its encoder only when constructed; neither class starts a
process before reset.
"""
from .hazard_reader import VersionedHazardReader
from .responsive_campaign import ResponsiveParallelResourceCampaignEnv
from .responsive_controller import (ResponsiveParallelResourceGameController,
    ResponsiveParallelDialogueResourceGameController)


class _HazardReaderMixin:
    def start(self):
        super().start()
        if not isinstance(self.reader, VersionedHazardReader):
            self.reader = VersionedHazardReader(self._memory, self.profile)
        return self.read()


class HazardResourceGameController(_HazardReaderMixin, ResponsiveParallelResourceGameController):
    pass


class HazardDialogueResourceGameController(_HazardReaderMixin, ResponsiveParallelDialogueResourceGameController):
    pass


class MetadataResourceCampaignEnv(ResponsiveParallelResourceCampaignEnv):
    """Preserve v2 inputs while recording additional native hazard fields."""

    def _create_controller(self):
        kind = HazardDialogueResourceGameController if self.dialogue_skip else HazardResourceGameController
        return kind(self.executable, self.profile_path, timeout=self.timeout,
            private_appdata=self.private_appdata, worker_name=self.window_title or self.private_appdata.name)


class HazardResourceCampaignEnv(MetadataResourceCampaignEnv):
    """Explicit v3 sensor contract; old checkpoints cannot be used silently."""

    def __init__(self, *args, hazard_observation_config=None, **kwargs):
        if 'resource_observation_config' in kwargs:
            raise ValueError('Use explicit hazard_observation_config for the v3 environment')
        super().__init__(*args, **kwargs)
        import gymnasium as gym
        import numpy as np
        from .hazard_observation import HazardObservationEncoder
        self.encoder = HazardObservationEncoder(hazard_observation_config, self.encoder.planner.config)
        self.observation_space = gym.spaces.Box(-1., 1., shape=(self.encoder.size,), dtype=np.float32)
