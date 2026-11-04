"""Versioned completion objective. Legacy THRL port remains byte-identical."""
from __future__ import annotations
import math
import numpy as np
from .thrl_rewards import THRLRewardModel,THRLRewardResult,REWARD_SCHEMA as LEGACY_REWARD

RECIPE='th14-completion-v2'
REWARD_SCHEMA='th14-completion-reward-v2'


def reward_options(config):
    return dict(reward_schema=config.get('reward_schema',LEGACY_REWARD),
        reward_gamma=config.get('parameters',{}).get('gamma',config.get('ppo_baseline',{}).get('gamma',.995)))


class CompletionRewardResult(THRLRewardResult):
    def info(self):
        return {**super().info(),'reward_schema':REWARD_SCHEMA,'reward_scales':(1.,1.,1.)}


class CompletionRewardModel(THRLRewardModel):
    """Reuse verified events; small dense aid, explicit death and Bomb costs.

    Firepower aid uses gamma*Phi(next)-Phi(before), Phi=.2*clip(P/400).
    True terminal potential is zero; curriculum truncations retain it.
    """
    def __init__(self,*args,gamma=.999,**kwargs):
        if isinstance(gamma,bool) or not math.isfinite(gamma) or not 0<gamma<=1:
            raise ValueError('Reward shaping discount must match PPO in (0,1]')
        super().__init__(*args,**kwargs);self.gamma=float(gamma)

    def specification(self):
        return {**super().specification(),'schema':REWARD_SCHEMA,'source_event_adapter':LEGACY_REWARD,
            'scales':(1.,1.,1.),'survival':'+.001 per observation transition; unchanged action interval',
            'death':-1.,'Bomb_used':-.1,'unused_Bombs_on_death':0.,'graze':0.,'point_items':0.,
            'boss_damage_per_HP':.0001,'midboss_defeat':.2,'corner_maximum':-.001,
            'terminal':'-2 native gameover; +5 native six-stage clear; 0 external cut',
            'power':'gamma*Phi(next)-Phi(before); Phi=.2*clip(nativeP/400,0,1); zero at true terminal',
            'shaping_gamma':self.gamma,'bomb_tracking':'verified native bombs_used delta; unused-Bomb cost disabled',
            'not_identical_to_upstream_THRL':True}

    def reset(self,snapshot):
        return {**super().reset(snapshot),'reward_schema':REWARD_SCHEMA}

    def transition(self,snapshot,*,end=0):
        before=self.previous
        source=super().transition(snapshot,end=end)
        raw=source.components
        phi=lambda power:.2*min(1.,max(0.,power/400.))
        components={
            'survival_per_observation':raw.get('survival_per_observation',0.)*.01,
            'corner':raw.get('corner',0.)*.002,
            'death':raw.get('death',0.)*.02,
            'native_gameover':raw.get('native_gameover',0.)*.02,
            'native_clear':raw.get('native_clear',0.)*.05,
            'boss_damage':raw.get('boss_damage',0.)*.0002,
            'midboss_defeat':raw.get('midboss_defeat',0.)*.004,
            'bomb_used':(-.1*max(0,source.frame.bombs_used-before.bombs_used)
                if source.frame.bombs_used is not None and before.bombs_used is not None else 0.),
            'power_potential':self.gamma*(0. if end else phi(source.frame.power))-phi(before.power)}
        vector=np.asarray([sum(components[k] for k in ('survival_per_observation','corner','death','native_gameover','native_clear')),
            components['boss_damage']+components['midboss_defeat'],components['bomb_used']+components['power_potential']],dtype=np.float32)
        return CompletionRewardResult(vector.copy(),vector,components,source.unsupported,source.frame)


def make_reward_model(bounds,*,reward_schema=LEGACY_REWARD,reward_gamma=.995,require_complete=True):
    if reward_schema==LEGACY_REWARD:return THRLRewardModel(bounds,require_complete=require_complete)
    if reward_schema==REWARD_SCHEMA:return CompletionRewardModel(bounds,gamma=reward_gamma,require_complete=require_complete)
    if reward_schema=='th14-completion-reward-v3':
        from .th14_completion_v3 import ResourceCompletionReward
        return ResourceCompletionReward(bounds,gamma=reward_gamma,require_complete=require_complete)
    raise ValueError('Unknown versioned reward schema: '+str(reward_schema))


def baseline_parameters(config):
    if config.get('training_recipe')=='th14-completion-v3':
        return dict(learning_rate=1e-4,gamma=.999,gae_lambda=.98,entropy_coeff_start=.01)
    return (dict(learning_rate=1e-4,gamma=.999,gae_lambda=.98,entropy_coeff_start=.02)
        if config.get('training_recipe')==RECIPE else
        dict(learning_rate=1e-4,gamma=.995,gae_lambda=.98,entropy_coeff_start=.05))


def exploration_parameters(config,start):
    if config.get('training_recipe')=='th14-completion-v3':
        decisions=config['entropy_decay_decisions']
        if type(decisions) is not int or decisions<=0:raise ValueError('Positive decision-based exploration budget required')
        return dict(entropy_coeff_end=start*.1,entropy_anneal_updates=math.ceil(decisions/config['rollout_total_decisions']))
    if config.get('training_recipe')!=RECIPE:return dict(entropy_coeff_end=start*.2)
    decisions=config['entropy_decay_decisions']
    if type(decisions) is not int or decisions<=0:raise ValueError('Positive decision-based exploration budget required')
    return dict(entropy_coeff_end=start*.1,entropy_anneal_updates=math.ceil(decisions/config['rollout_total_decisions']))


def learning_overrides(config):
    """Pinned recipe options, independent of the panel's worker count."""
    if config.get('training_recipe')=='th14-completion-v3':
        from .thrl_learning_revision import auxiliary_parameters
        return dict(seqs_per_batch=16,gae_chunk_size=0,target_kl=.03,**auxiliary_parameters(config))
    return {}


def search_space(config):
    if config.get('training_recipe')=='th14-completion-v3':
        return dict(learning_rate=(3e-5,1e-4),gamma=[.999],gae_lambda=[.98],entropy_coeff_start=[.005,.01])
    if config.get('training_recipe')==RECIPE:
        return dict(learning_rate=(3e-5,3e-4),gamma=[.995,.999],gae_lambda=[.95,.98],entropy_coeff_start=[.01,.02])
    return dict(learning_rate=(3e-5,3e-4),gamma=[.99,.995,.999],gae_lambda=[.95,.98],entropy_coeff_start=[.02,.05])
