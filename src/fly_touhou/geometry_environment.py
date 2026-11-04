"""Opt-in V5 geometry sidecar on the unchanged V3 student observation."""
from .hazard_environment import (HazardResourceCampaignEnv,HazardResourceGameController,
                                  HazardDialogueResourceGameController)
from .hazard_reader import GeometryHazardReader


class _GeometryReaderMixin:
    def start(self):
        super().start()
        if not isinstance(self.reader,GeometryHazardReader):
            self.reader=GeometryHazardReader(self._memory,self.profile)
        return self.read()


class GeometryHazardResourceGameController(_GeometryReaderMixin,HazardResourceGameController):
    pass


class GeometryHazardDialogueResourceGameController(_GeometryReaderMixin,HazardDialogueResourceGameController):
    pass


class GeometryHazardResourceCampaignEnv(HazardResourceCampaignEnv):
    """Collect extra native geometry without changing any encoded feature."""

    def _create_controller(self):
        kind=GeometryHazardDialogueResourceGameController if self.dialogue_skip else GeometryHazardResourceGameController
        return kind(self.executable,self.profile_path,timeout=self.timeout,
            private_appdata=self.private_appdata,worker_name=self.window_title or self.private_appdata.name)


class GeometryObservationCampaignEnv(GeometryHazardResourceCampaignEnv):
    """Explicit new student sensor; V5 teacher's V3 environment stays unchanged."""

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        import gymnasium as gym
        import numpy as np
        from .geometry_observation import GeometryObservationEncoder
        self.encoder=GeometryObservationEncoder(prefix=self.encoder)
        self.observation_space=gym.spaces.Box(-1.,1.,shape=(self.encoder.size,),dtype=np.float32)
