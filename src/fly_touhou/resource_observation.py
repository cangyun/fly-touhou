"""Versioned current-state resource inputs, without teacher decisions or risks."""
from __future__ import annotations

from dataclasses import asdict,dataclass
import math

import numpy as np

from .observation import ObservationConfig,ObservationEncoder
from .planner import PlannerConfig
from .resources import POWER_KINDS,SURVIVAL_KINDS,RESOURCE_SCHEMA

OBSERVATION_SCHEMA = 'native-resource-observation-v2'


@dataclass(frozen=True)
class ResourceObservationConfig:
    nearest_bullets:int=24
    nearest_lasers:int=8
    include_action_risk:bool=False
    nearest_items:int=16
    nearest_enemies:int=4

    def __post_init__(self):
        if self.include_action_risk:
            raise ValueError('Resource model uses current state without teacher action-risk inputs')
        if any(isinstance(n,bool) or not isinstance(n,int) or not 0<=n<=64 for n in (
            self.nearest_bullets,self.nearest_lasers,self.nearest_items,self.nearest_enemies)):
            raise ValueError('Resource observation slot counts must be integers in [0,64]')


class ResourceObservationEncoder:
    GLOBAL_NAMES=('resources_observed','life_pieces','bomb_pieces','poc_dy',
        'power_item_count','survival_item_count','item_count','autocollecting_count',
        'player_can_shoot','player_state','power_deficit')
    ITEM_NAMES=('present','dx','dy','vx','vy','kind','state','age')
    ENEMY_NAMES=('present','dx','dy','vx','vy','hp_log','boss','damageable',
                 'contactable','hitbox_x','hitbox_y')

    def __init__(self,config=None,planner_config=None):
        self.config=config or ResourceObservationConfig()
        self.base=ObservationEncoder(ObservationConfig(
            nearest_bullets=self.config.nearest_bullets,nearest_lasers=self.config.nearest_lasers,
            include_action_risk=False),planner_config)
        self.planner=self.base.planner

    @property
    def size(self):
        return (self.base.size+len(self.GLOBAL_NAMES)+len(self.ITEM_NAMES)*self.config.nearest_items
                +len(self.ENEMY_NAMES)*self.config.nearest_enemies)

    @property
    def feature_names(self):
        return self.base.feature_names+self.GLOBAL_NAMES+tuple(
            f'item_{i}_{name}' for i in range(self.config.nearest_items) for name in self.ITEM_NAMES)+tuple(
            f'enemy_{i}_{name}' for i in range(self.config.nearest_enemies) for name in self.ENEMY_NAMES)

    def specification(self):
        return {'schema':OBSERVATION_SCHEMA,'observation':asdict(self.config),
                'planner':asdict(self.planner.config),'size':self.size,
                'input_kind':'measured-native-state-with-resources',
                'engineered_prediction_features':False,'resource_schema':RESOURCE_SCHEMA}

    @classmethod
    def from_specification(cls,specification):
        result=cls(ResourceObservationConfig(**specification['observation']),
                   PlannerConfig(**specification['planner']))
        if result.specification()!=specification:
            raise ValueError('Recorded resource input schema differs from the current encoder')
        return result

    def encode(self,snapshot):
        base=self.base.encode(snapshot)
        if snapshot.player is None:
            return np.zeros(self.size,dtype=np.float32)
        resources=snapshot.extra.get('resources',{})
        if resources.get('observed') is not True or resources.get('schema')!=RESOURCE_SCHEMA:
            raise ValueError('Resource model requires measured items; missing items cannot mean an empty field')
        player=snapshot.player
        items=list(resources['items'])
        priority=lambda item: (0 if item['kind'] in SURVIVAL_KINDS else
            1 if item['kind'] in POWER_KINDS and snapshot.power<resources['power_max'] else 2,
            (item['x']-player.x)**2+(item['y']-player.y)**2,item['id'])
        items.sort(key=priority)
        enemies=sorted(resources['enemies'],key=lambda enemy:(not enemy['boss'],not enemy['damageable'],
            (enemy['x']-player.x)**2+(enemy['y']-player.y)**2,enemy['id']))
        values=[1,resources['life_pieces']/3,resources['bomb_pieces']/8,
            (resources['poc_line_y']-player.y)/448,
            sum(item['kind'] in POWER_KINDS for item in items)/64,
            sum(item['kind'] in SURVIVAL_KINDS for item in items)/16,len(items)/256,
            sum(item.get('autocollecting',False) for item in items)/256,
            float(resources['player_can_shoot']),player.state/8,
            max(0,resources['power_max']-snapshot.power)/resources['power_max']]
        for index in range(self.config.nearest_items):
            if index>=len(items):
                values.extend([0.]*len(self.ITEM_NAMES))
                continue
            item=items[index]
            values.extend((1,(item['x']-player.x)/384,(item['y']-player.y)/448,
                item['vx']/16,item['vy']/16,item['kind']/12,item['state']/4,item['age']/600))
        for index in range(self.config.nearest_enemies):
            if index>=len(enemies):
                values.extend([0.]*len(self.ENEMY_NAMES))
                continue
            enemy=enemies[index]
            values.extend((1,(enemy['x']-player.x)/384,(enemy['y']-player.y)/448,
                enemy['vx']/16,enemy['vy']/16,math.log1p(max(0,enemy['hp']))/math.log(100001),
                float(enemy['boss']),float(enemy['damageable']),float(enemy['contactable']),
                enemy['radius_x']/128,enemy['radius_y']/128))
        result=np.concatenate((base,np.asarray(values,dtype=np.float32)))
        if result.shape!=(self.size,) or not np.isfinite(result).all():
            raise ValueError('Malformed measured resource input')
        return np.clip(result,-1,1)
