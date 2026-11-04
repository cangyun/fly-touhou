"""Optional verified native starting-point bank; ordinary evaluation stays Story.

Prefix simulation is paid reset cost, never counted as learner decisions or
reward. No fake clear/death reward at a declared curriculum boundary. Starting
from a later stage still executes its recorded original-game prefix; this does
not claim an instant TH14 ECL practice jump or a full-memory savestate.
"""
import json
import hashlib
from dataclasses import replace
from pathlib import Path
import time

import numpy as np

from .native_replay import START_SCHEMA,replay_entry,validate_tape
from .thrl_native_env import THRLNativeEnv


def load_start_bank(path,expected_sha256=None):
    raw=Path(path).read_bytes()
    if expected_sha256 is not None and hashlib.sha256(raw).hexdigest()!=expected_sha256:
        raise ValueError('Native start bank changed after study preparation')
    payload=json.loads(raw.decode('utf-8'))
    entries=payload.get('entries',[payload] if payload.get('schema')==START_SCHEMA else [])
    if not entries or len({row.get('id') for row in entries})!=len(entries):
        raise ValueError('A nonempty bank of distinct native starting points is required')
    for row in entries:
        if (row.get('schema')!=START_SCHEMA or row.get('verified_independent_resets',0)<2
                or row.get('resources_injected') is not False or row.get('native_deaths') is not True
                or type(row.get('native_seed')) is not int or not 0<=row['native_seed']<=65535
                or type(row.get('base_stage_frame')) is not int or row['base_stage_frame']<1
                or row.get('landing',{}).get('lives',-1)<0):
            raise ValueError('Native starting point lacks repeated exact verification or finite native lives')
        tracking=row.get('reward_tracking',{})
        if (tracking.get('schema')!='thrl-prefix-reward-state-v1'
                or any(type(tracking.get(k)) is not int or tracking[k]<0 for k in
                       ('cfg_bombs','rem_bombs_internal','misses','bombs_used','point_items'))
                or tracking['rem_bombs_internal']>tracking['cfg_bombs']):
            raise ValueError('Native prefix lacks the original reward tracker state')
        validate_tape(row['raw_input_segments'])
        if 'base_visual_rng_alignment' in row:
            from .native_replay import VISUAL_ALIGNMENT_SCHEMA,semantic_fingerprint
            alignment=row['base_visual_rng_alignment']
            if (alignment.get('schema')!=VISUAL_ALIGNMENT_SCHEMA or
                    semantic_fingerprint(alignment['reference_state'])!=row['base_state_sha256']):
                raise ValueError('Invalid declared visual RNG base alignment')
        for key in ('base_state_sha256','landing_state_sha256','game_exe_sha256'):
            value=row.get(key)
            if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('Starting point lacks a pinned state/executable digest')
    return entries


def rebase_prefix_reward(model,snapshot,tracking):
    """Restore captured source bookkeeping, rebind boss IDs to THIS process.

    Native inventory is never written. In particular, remaining inventory at
    a later start is not the native initial Bomb configuration. No source PPO
    or reward formula is modified, and no reward is earned for the prefix.
    """
    events=snapshot.extra.get('thrl_events',{})
    if any(events.get(k)!=tracking[k] for k in ('misses','bombs_used','point_items')):
        raise ValueError('Prefix reward/event counter state mismatch')
    model.cfg_bombs=tracking['cfg_bombs']
    model.previous=None
    frame=model._frame(snapshot)
    model._require(frame.unsupported)
    model.previous=replace(frame,rem_bombs_internal=tracking['rem_bombs_internal'])
    model.steps=0
    if hasattr(model,'rebase_snapshot'):model.rebase_snapshot(snapshot)
    from .thrl_rewards import REWARD_SCHEMA
    return {'reward_schema':model.specification()['schema'],'reward_mapping_complete':not frame.unsupported,
        'reward_unsupported':list(frame.unsupported),'initial_cfg_bombs':model.cfg_bombs}


class THRLSegmentEnv(THRLNativeEnv):
    def __init__(self,*args,start_bank,start_bank_sha256=None,segment_end=None,full_story_fraction=0.,
                 curriculum_seed=11,**kwargs):
        self.start_bank=load_start_bank(start_bank,start_bank_sha256)
        self.segment_end=segment_end
        if segment_end is not None and (set(segment_end)!={'stage','frame'}
                or type(segment_end['stage']) is not int or not 1<=segment_end['stage']<=6
                or type(segment_end['frame']) is not int or segment_end['frame']<1):
            raise ValueError('An explicit native stage/frame curriculum end is required')
        self._selection_rng=np.random.default_rng(curriculum_seed)
        if not isinstance(full_story_fraction,(int,float)) or not 0<=full_story_fraction<=1:
            raise ValueError('full_story_fraction must be in [0,1]')
        self.full_story_fraction=float(full_story_fraction)
        self._full_story_episode=False
        self.prefix_receipt=None
        super().__init__(*args,**kwargs)
        self._userdata_root=getattr(self,'private_appdata',None)

    def reset(self,*,seed=None,options=None):
        if options is not None and (not isinstance(options,dict) or set(options)-{'start_index','force_prefix'}):
            raise ValueError('Unknown curriculum reset options')
        if seed is not None:self._selection_rng=np.random.default_rng(seed)
        if self._userdata_root is not None:
            # Match bank construction's fresh native save/config contract on
            # every episode; native score/config writes cannot shift the next
            # reset's menu/defaults. The root remains private to this worker.
            self.private_appdata=self._userdata_root/f'episode_{self.episode_id+1:06d}'
        self._full_story_episode=(not (options or {}).get('force_prefix',False)
            and self.full_story_fraction>0 and self._selection_rng.random()<self.full_story_fraction)
        if self._full_story_episode:
            if self.controller is not None:self.controller.close();self.controller=None
            self.native_seed=None;self.prefix_receipt=None
            return super().reset(seed=seed)
        index=int((options or {}).get('start_index',self._selection_rng.integers(len(self.start_bank))))
        if not 0<=index<len(self.start_bank):raise ValueError('Unknown curriculum starting point')
        entry=self.start_bank[index]
        self.native_seed=entry['native_seed']
        # Fresh engine ownership makes the recorded app seed/counters and all
        # unobserved native initialization reproducible; retry is not a savestate.
        if self.controller is not None:self.controller.close();self.controller=None
        self.prefix_receipt=None
        started=time.perf_counter()
        observation,info=super().reset(seed=seed)
        try:
            if self.controller.reader.digest!=entry['game_exe_sha256']:
                raise ValueError('Native start bank belongs to a different executable')
            self.snapshot=self._check_snapshot(replay_entry(self.controller,entry))
            self._verify_native_run(self.snapshot)
            if self._native_end(self.snapshot):raise ValueError('Native prefix landed at a terminal state')
            observation=self._encode(self.snapshot)
            self.episode_frames=0;self._native_input_ticks=0;self._nonplay_ticks=0
            self.policy_steps=0;self.stage_history=[self.snapshot.stage]
            self._episode_start_frame=self.snapshot.frame;self._episode_start_endpoint=self._endpoint()
            self._episode_returns[:]=0;self._episode_raw_returns[:]=0
            mapping=rebase_prefix_reward(self.reward_model,self.snapshot,entry['reward_tracking'])
            self._mapping_unsupported=set(mapping['reward_unsupported'])
            self._episode_started_wall=time.monotonic()
            self.prefix_receipt={'schema':'th14-native-prefix-reset-receipt-v1','entry_id':entry['id'],
                'native_seed':entry['native_seed'],'landing_state_sha256':entry['landing_state_sha256'],
                'reset_seconds':time.perf_counter()-started,
                'prefix_input_ticks':sum(row[1] for row in entry['raw_input_segments']),
                'new_policy_decisions':0,'resources_injected':False,
                'initial_cfg_bombs':self.reward_model.cfg_bombs,
                'rem_bombs_internal':self.reward_model.previous.rem_bombs_internal,
                'native_bombs_remaining':self.snapshot.bombs,
                'visual_rng_alignment':self.snapshot.extra.get('prefix_visual_rng_alignment'),
                'instant_stage_jump':False,'formal_full_game':False}
            info=self._info();info.update(mapping)
            info.update(reward_vector=np.zeros(3,np.float32),reward_raw_vector=np.zeros(3,np.float32),
                bootstrap_allowed=True,native_terminal_kind='ongoing',prefix_reset=self.prefix_receipt)
            return observation,info
        except BaseException:
            self.close();raise

    def _info(self,elapsed=0):
        info=super()._info(elapsed)
        info.update(episode_mode='native-finite-life-story' if self._full_story_episode else 'verified-native-story-prefix',formal_full_game=False,
                    prefix_reset=self.prefix_receipt)
        return info

    def step(self,action):
        observation,reward,terminated,truncated,info=super().step(action)
        end=None if self._full_story_episode else self.segment_end
        reached=(end is not None and (self.snapshot.stage>end['stage'] or (
            self.snapshot.stage==end['stage'] and self.snapshot.stage_frame>=end['frame'])))
        if reached and not (terminated or truncated):
            truncated=True;self._needs_reset=True
            info.update(truncated=True,**{'TimeLimit.truncated':True},bootstrap_allowed=True,
                external_truncation='declared-curriculum-boundary',terminal_observation=observation,
                lunatic_six_stage_no_continue=False)
            info['episode_summary']={'outcome':'curriculum-cut','native_frames':self.episode_frames,
                'total_frames':self.episode_frames,'input_ticks':self._native_input_ticks,
                'policy_decisions':self.policy_steps,'stage':self.snapshot.stage,
                'stage_frame':self.snapshot.stage_frame,'end_reason':'declared-curriculum-boundary',
                'prefix_reset':self.prefix_receipt,'formal_full_game':False,
                'reward_vector':self._episode_returns.tolist(),'raw_reward_vector':self._episode_raw_returns.tolist()}
        if 'episode_summary' in info:
            info['episode_summary'].update(prefix_reset=self.prefix_receipt,formal_full_game=False)
        return observation,reward,terminated,truncated,info
