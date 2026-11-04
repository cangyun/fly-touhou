"""Independent diagnostic only: no Bomb, stop on observed first collision.

Never selected by the training factory or connected to an optimizer. Uses the
same original-game opening, seed partition, observations, shot and movement.
The returned diagnostic endpoint is not a native Game Over or full-game score.
No Torch import: native worker processes stay model-free.
"""
from __future__ import annotations

from .thrl_full_story_env import THRLFullStoryEnv

PROBE_SCHEMA='th14-no-bomb-first-observed-collision-v1'
MAX_DIAGNOSTIC_FRAMES=32768


def collision_observed(player,misses):
    return bool((player is not None and player.state==4) or misses>0)


class AvoidanceProbeEnv(THRLFullStoryEnv):
    diagnostic_only=True

    def __init__(self,*args,probe_max_frames=4096,**kwargs):
        if type(probe_max_frames) is not int or not 256<=probe_max_frames<=MAX_DIAGNOSTIC_FRAMES:
            raise ValueError('Explicit diagnostic frame cap must be 256..32768')
        self.probe_max_frames=probe_max_frames
        super().__init__(*args,**kwargs)

    def reset(self,*,seed=None,options=None):
        if not options or set(options)!={'evaluation_seed'}:
            raise ValueError('Each probe needs an explicit matched evaluation seed')
        self._probe_decisions=0;self._probe_unprotected_decisions=0
        self._probe_first_vulnerable_frame=None
        observation,info=super().reset(seed=int(options['evaluation_seed']),options=None)
        if self.snapshot.player.state==1 and self.snapshot.player.invulnerable_frames<=0:
            self._probe_first_vulnerable_frame=0
        info.update(diagnostic_only=True,diagnostic_schema=PROBE_SCHEMA)
        return observation,info

    def step(self,action):
        if not 0<=int(action)<18:raise ValueError('Probe forbids Bomb for every policy/baseline')
        before=self.snapshot
        if before.player.state==1 and before.player.invulnerable_frames<=0:
            self._probe_unprotected_decisions+=1
        observation,reward,terminated,truncated,info=super().step(action)
        self._probe_decisions+=1
        misses=info.get('misses')
        if type(misses) is not int or misses<0 or info.get('bombs_used')!=0:
            raise ValueError('Probe requires measured Miss counter and zero actual Bombs')
        player=self.snapshot.player
        if (player is not None and player.state==1 and player.invulnerable_frames<=0
                and self._probe_first_vulnerable_frame is None):
            self._probe_first_vulnerable_frame=self.episode_frames
        hit=collision_observed(player,misses)
        capped=self.episode_frames>=self.probe_max_frames and not hit
        if hit or capped or terminated or truncated:
            if not hit and not capped:
                raise ValueError('Unexpected native terminal before collision or diagnostic cap')
            info.pop('episode_summary',None)
            info['probe_episode']={
                'schema':PROBE_SCHEMA,'diagnostic_only':True,'first_collision_observed':hit,
                'right_censored_at_cap':capped,'native_frames':self.episode_frames,
                'stage':self.snapshot.stage,'stage_frame':self.snapshot.stage_frame,
                'native_seed':self.native_seed,'seed_applied':info.get('game_rng_seed_applied'),
                'native_misses':misses,'actual_bombs_used':info['bombs_used'],
                'policy_decisions':self._probe_decisions,
                'unprotected_preaction_checkpoints':self._probe_unprotected_decisions,
                'first_vulnerable_observed_frame':self._probe_first_vulnerable_frame,
                'player_state':player.state if player else None,
                'deathbomb_remaining':player.deathbomb_timer if player else None,
                'first_collision_time_is_observed_endpoint_not_exact_native_event':True,
                'formal_full_game_result':False,'policy_sampling':'conditional on movement actions0..17'}
            from .thrl_probe_metrics import endpoint_fields
            info['probe_episode'].update(endpoint_fields(self.episode_frames,self.probe_max_frames))
            if info['probe_episode']['seed_applied'] is not True:
                raise ValueError('Matched native seed was not actually applied')
            self._needs_reset=True
            # External diagnostic stopping only. No PPO reward/terminal is used.
            return observation,reward,True,False,info
        return observation,reward,False,False,info


def probe_factory(kwargs):
    return AvoidanceProbeEnv(**kwargs)
