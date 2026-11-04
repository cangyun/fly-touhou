"""Finite-life original campaigns with observed items and attack resources."""
import gymnasium as gym
import numpy as np

from .campaign import CampaignTouhouEnv
from .frida_controller import OriginalGameController
from .resources import ResourceNativeReader
from .resource_observation import ResourceObservationConfig,ResourceObservationEncoder
from .observation import ObservationConfig


class ResourceGameController(OriginalGameController):
    def start(self):
        super().start()
        self.reader=ResourceNativeReader(self._memory,self.profile)
        return self.read()


class ResourceCampaignEnv(CampaignTouhouEnv):
    def __init__(self,*args,resource_observation_config=None,**kwargs):
        if 'observation_config' in kwargs:
            raise ValueError('Use versioned resource_observation_config for the resource environment')
        super().__init__(*args,observation_config=ObservationConfig(include_action_risk=False),**kwargs)
        self.encoder=ResourceObservationEncoder(resource_observation_config or ResourceObservationConfig(),
                                               self.encoder.planner.config)
        self.observation_space=gym.spaces.Box(-1.,1.,shape=(self.encoder.size,),dtype=np.float32)

    def _create_controller(self):
        return ResourceGameController(self.executable,self.profile_path,timeout=self.timeout)
