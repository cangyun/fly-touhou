"""Progressive native TH14 resets and live semantic lesson boundaries."""
from __future__ import annotations
from pathlib import Path
import time
import uuid
import numpy as np

from .thrl_native_env import THRLNativeEnv
from .thrl_segment_env import rebase_prefix_reward
from .thrl_course_store import CourseStore,PhaseTracker,lesson_order,lesson_id,trajectory_entry
from .native_replay import replay_entry,state_fingerprint,semantic_state,VISUAL_ALIGNMENT_SCHEMA
from .thrl_curriculum import (configuration,native_seed,SCHEMA,schema_for,chain_horizon,
                              CHAIN_LESSON_BOUNDARY,CHAIN_NATIVE_GAME)
from .thrl_recurrent import pack_observations


class THRLCourseEnv(THRLNativeEnv):
    def __init__(self,*args,curriculum_manifest,curriculum_manifest_sha256=None,
                 curriculum_overlay=None,curriculum_split='train',curriculum_options=None,
                 curriculum_seed=11,harvest=False,curriculum_default_mode='current',
                 curriculum_inherited_pools=None,curriculum_recurrent=True,
                 curriculum_chain_horizon=CHAIN_LESSON_BOUNDARY,**kwargs):
        self.chain_horizon=chain_horizon(curriculum_chain_horizon)
        if curriculum_split!='train' and self.chain_horizon!=CHAIN_LESSON_BOUNDARY:
            raise ValueError('Native-game chain horizon applies only to training rollouts')
        self.course=CourseStore(curriculum_manifest,curriculum_manifest_sha256,curriculum_overlay,
                               curriculum_inherited_pools)
        self.course_options=configuration(curriculum_options)
        self.curriculum_recurrent=bool(curriculum_recurrent)
        if curriculum_split not in ('train','development','test'):
            raise ValueError('Explicit native curriculum seed partition required')
        self.course_split=curriculum_split
        self._default_mode=curriculum_default_mode
        self._course_rng=np.random.default_rng(curriculum_seed)
        self.assignment=None;self.prefix_receipt=None;self._phase=PhaseTracker()
        self._target_boundary=None
        self._harvest_enabled=bool(harvest and curriculum_split=='train')
        self._harvest=None
        self._course_episode_number=0
        super().__init__(*args,**kwargs)
        self.course_loading_bridge_frames=4
        self._course_userdata_root=self.private_appdata

    def _prepare_missing_lesson(self,assignment):
        """Lazy course data preparation, outside all PPO transitions/rewards.

        One OS-locked builder per branch. This is explicitly teacher-assisted
        reset preparation; no teacher action enters a student's rollout.
        Failed preparation remains a failure, never a zero success score.
        """
        if self.course_split=='test' or self.course.overlay is None:
            raise ValueError('Independent course evaluation lacks verified data for '+assignment['lesson'])
        from .thrl_control import controller_lock
        from .thrl_course_native import capture_trajectory,verify_trajectory
        from .atomic_json_io import atomic_write_json
        self.course.overlay.mkdir(parents=True,exist_ok=True)
        begun=time.monotonic()
        def usable():
            return self.course.has_starts(assignment,self.course_split,self.course_options['min_tail'])
        while True:
            self.course.refresh()
            if usable():return
            try:
                with controller_lock(self.course.overlay):
                    self.course.refresh()
                    if usable():return
                    for attempt in range(2):
                        seed=native_seed(self._course_rng,self.course_split)
                        folder=self.course.overlay/f'preparation_{seed}_{uuid.uuid4().hex[:10]}'
                        observer=getattr(self,'_course_observe_controller',None)
                        trace=capture_trajectory(self.executable,self.profile_path,folder,seed,
                            max_stage=int(assignment['lesson'][1]),max_decisions=30000,
                            controller_observer=observer)
                        trace=verify_trajectory(trace,self.executable,self.profile_path,folder,
                                                controller_observer=observer)
                        atomic_write_json(self.course.overlay/f'verified_{trace["id"]}.json',trace)
                        self.course.refresh()
                        if usable():return
                    raise ValueError('Two bounded preparations did not reach '+assignment['lesson'])
            except RuntimeError as error:
                if 'lock' not in str(error).lower() and '控制' not in str(error):raise
                if time.monotonic()-begun>900:raise TimeoutError('Native course preparation lock wait expired') from error
                time.sleep(.2)

    def reset(self,*,seed=None,options=None):
        if seed is not None:self._course_rng=np.random.default_rng(seed)
        options=options or {}
        if set(options)-{'assignment','split','evaluation_seed'}:raise ValueError('Unknown progressive reset option')
        if 'evaluation_seed' in options:
            value=options['evaluation_seed']
            if self.course_split not in ('development','test') or type(value) is not int or not 0<=value<2**32:
                raise ValueError('Per-episode evaluation seed requires a held-out worker and uint32')
            self._course_rng=np.random.default_rng(value)
        if options.get('split',self.course_split)!=self.course_split:
            raise ValueError('A worker may not cross native seed partitions')
        # Independent verification is asynchronous in the learner's bounded
        # maintenance worker. A slow replay must not stall an entire PPO batch.
        self.course.refresh()
        self.assignment=dict(options.get('assignment') or dict(schema=schema_for(self.course_options),mode=self._default_mode,
            lesson='s1_opening',lesson_index=0,frontier=0,depth=self.course_options['initial_depth']))
        if self.assignment.get('schema')!=schema_for(self.course_options) or self.assignment['mode'] not in ('current','review','chain'):
            raise ValueError('Versioned explicit lesson assignment required')
        lesson_order(self.assignment['lesson'])
        self._course_episode_number+=1
        self.private_appdata=self._course_userdata_root/f'episode_{self._course_episode_number:06d}'
        if self.controller is not None:self.controller.close();self.controller=None
        self.prefix_receipt=None;self._harvest=None;self._target_boundary=None
        begun=time.perf_counter();burn=[]
        if self.assignment['mode']=='chain':
            self.native_seed=native_seed(self._course_rng,self.course_split)
            observation,info=super().reset()
            initialization_ticks=0
            while self.snapshot.stage==1 and self.snapshot.stage_frame<64:
                self.snapshot=self._check_snapshot(self.controller.advance(1,1))
                initialization_ticks+=1
            if self.snapshot.stage!=1 or self.snapshot.stage_frame!=64:
                raise ValueError('Fresh chain missed the canonical frame64 initialization')
            self._verify_native_run(self.snapshot)
            observation=self._encode(self.snapshot)
            mapping=self.reward_model.reset(self.snapshot)
            self._reset_episode_reward_diagnostics()
            self._episode_start_frame=self.snapshot.frame
            self._episode_start_endpoint=self._endpoint()
            self._episode_started_wall=time.monotonic()
            info.update(self._info(),**mapping,canonical_initialization_input_ticks=initialization_ticks)
            self._phase=PhaseTracker()
            self._phase.observe(self.snapshot)
            # The entire preceding game is played by this current policy.
            # Only a designated worker prepares extra verified self-play
            # trajectories, bounding cost independently of the worker count.
            if self._harvest_enabled and self.course.overlay is not None:
                self._start_harvest()
        else:
            if not self.course.has_starts(self.assignment,self.course_split,self.course_options['min_tail']):
                self._prepare_missing_lesson(self.assignment)
            trace,index=self.course.choose(self.assignment,self._course_rng,self.course_split,self.course_options['min_tail'])
            entry=trajectory_entry(trace,index);self.native_seed=trace['native_seed']
            observation,info=super().reset()
            rows=(trace['records'][max(0,index-self.course_options['burn_in']):index]
                  if self.curriculum_recurrent else [])
            def warm(snapshot,row):burn.append(self.encoder.encode(snapshot))
            self.snapshot=self._check_snapshot(replay_entry(self.controller,entry,checkpoints=rows,on_checkpoint=warm))
            if self._native_end(self.snapshot):raise ValueError('Course reset landed at a true terminal')
            self._verify_native_run(self.snapshot)
            landing=trace['records'][index]
            self._phase=PhaseTracker(landing['phase'],state=landing.get('phase_state'))
            # Every preceding stage is present in, and was actually executed
            # by, this verified prefix. Never invent a missing stage history.
            self.stage_history=list(dict.fromkeys(r['stage'] for r in trace['records'][:index+1]))
            observation=self._encode(self.snapshot)
            mapping=rebase_prefix_reward(self.reward_model,self.snapshot,entry['reward_tracking'])
            self._reset_episode_reward_diagnostics()
            self._episode_returns[:]=0;self._episode_raw_returns[:]=0
            self.episode_frames=0;self._native_input_ticks=0;self._nonplay_ticks=0;self.policy_steps=0
            self._episode_start_endpoint=self._endpoint();self._episode_start_frame=self.snapshot.frame
            self._episode_started_wall=time.monotonic()
            self.prefix_receipt=dict(schema='th14-progressive-prefix-reset-v1',trajectory=trace['id'],
                input_ticks=entry['raw_input_segments'] and sum(t for _,t in entry['raw_input_segments']),
                selected_checkpoint=index,selected_state_sha256=entry['landing_state_sha256'],
                native_seed=self.native_seed,verified_independent_resets=trace['verified_independent_resets'],
                reset_seconds=time.perf_counter()-begun,burn_in_observations=len(burn),
                new_policy_decisions=0,resources_injected=False,instant_stage_jump=False)
            if self.snapshot.extra.get('prefix_replay_cost'):
                self.prefix_receipt['replay_cost']=self.snapshot.extra['prefix_replay_cost']
            info=self._info();info.update(mapping,reward_vector=np.zeros(3,np.float32),
                reward_raw_vector=np.zeros(3,np.float32),bootstrap_allowed=True,native_terminal_kind='ongoing')
        events=self.snapshot.extra['thrl_events']
        self._segment_start_events={k:events[k] for k in ('misses','bombs_used','point_items')}
        info.update(curriculum_assignment=self.assignment,prefix_reset=self.prefix_receipt,
                    policy_recurrent=self.curriculum_recurrent,
                    recurrent_burn_in_observations=len(burn),_policy_burn_in=pack_observations(burn),
                    course_store=self.course.summary(),course_native_seed=self.native_seed,formal_full_game=False)
        return observation,info

    def _info(self,elapsed=0):
        info=super()._info(elapsed)
        info.update(curriculum_assignment=self.assignment,prefix_reset=self.prefix_receipt,
                    episode_mode=('native-game-training-chain' if self._full_native_chain()
                                  else 'progressive-native-lesson'),formal_full_game=False,
                    curriculum_chain_horizon=self.chain_horizon)
        return info

    def _full_native_chain(self):
        return (self.chain_horizon==CHAIN_NATIVE_GAME and self.assignment is not None
                and self.assignment['mode']=='chain')

    def _target_receipt(self,info):
        return dict(schema='th14-training-chain-target-boundary-v1',
            outcome='clear' if info.get('native_terminal_kind')=='clear' else 'target-reached',
            segment_reached=True,curriculum_mode='chain',
            lesson=self.assignment['lesson'],stage=self.snapshot.stage,stage_frame=self.snapshot.stage_frame,
            native_frame=self.snapshot.frame,native_frames=self.episode_frames,policy_decisions=self.policy_steps,
            segment_misses=info['misses']-self._segment_start_events['misses'],
            segment_bombs_used=info['bombs_used']-self._segment_start_events['bombs_used'],
            continues=info['continues'],lives=self.snapshot.lives,bombs_remaining=self.snapshot.bombs,
            power=self.snapshot.power,formal_full_game=False)

    def step(self,action):
        decision_lesson=lesson_id(self._phase.stage,self._phase.phase)
        observation,reward,terminated,truncated,info=super().step(action)
        phase=self._phase.observe(self.snapshot)
        reached=(not (terminated and info['native_terminal_kind']=='gameover') and
                 (lesson_order(phase)>lesson_order(self.assignment['lesson'])
                  or info.get('lunatic_six_stage_no_continue') is True))
        full_chain=self._full_native_chain()
        if full_chain and reached and self._target_boundary is None:
            self._target_boundary=self._target_receipt(info)
            # A one-time observation event, never a terminal or bonus reward.
            info['curriculum_target_boundary_event']=dict(self._target_boundary)
        if self._harvest is not None and not terminated:self._record_harvest()
        if reached and not full_chain and not (terminated or truncated):
            truncated=True;self._needs_reset=True
            info.update(truncated=True,**{'TimeLimit.truncated':True},bootstrap_allowed=True,
                external_truncation='progressive-curriculum-boundary',terminal_observation=observation,
                lunatic_six_stage_no_continue=False)
        if terminated or truncated:
            if 'episode_summary' not in info:
                info['episode_summary']=dict(outcome='curriculum-cut',native_frames=self.episode_frames,
                    total_frames=self.episode_frames,input_ticks=self._native_input_ticks,
                    policy_decisions=self.policy_steps,stage=self.snapshot.stage,
                    stage_frame=self.snapshot.stage_frame,stage_history=list(self.stage_history),
                    reward_vector=self._episode_returns.tolist(),raw_reward_vector=self._episode_raw_returns.tolist())
            summary=info['episode_summary']
            if self._episode_reward_diagnostics is not None:
                summary['reward_diagnostics']=self._reward_diagnostics_copy()
            summary.update(segment_reached=(self._target_boundary is not None if full_chain else reached),lesson=self.assignment['lesson'],
                curriculum_mode=self.assignment['mode'],curriculum_assignment=self.assignment,
                segment_misses=info['misses']-self._segment_start_events['misses'],
                segment_bombs_used=info['bombs_used']-self._segment_start_events['bombs_used'],
                continues=info['continues'],lives=self.snapshot.lives,bombs_remaining=self.snapshot.bombs,
                power=self.snapshot.power,formal_full_game=False,
                full_story_prefix_played_by_current_policy=self.assignment['mode']=='chain')
            if full_chain:
                summary.update(curriculum_chain_horizon=self.chain_horizon,
                    full_story_episode=True,episode_statistics_scope='entire-policy-played-native-run',
                    curriculum_target_boundary=(dict(self._target_boundary) if self._target_boundary is not None else None))
            if self._harvest is not None:self._finish_harvest(info)
        info['curriculum_phase']=phase
        info['curriculum_decision_lesson']=decision_lesson
        return observation,reward,terminated,truncated,info

    def _start_harvest(self):
        from .thrl_course_store import TRACE_SCHEMA
        # Canonical Story reset currently returns frame64. Do not pretend a
        # differently initialized native run belongs to that replay protocol.
        if self.snapshot.stage!=1 or self.snapshot.stage_frame!=64:return
        self.controller.begin_input_tape()
        self._harvest=dict(schema=TRACE_SCHEMA,id=f'selfplay-{self.native_seed}-{uuid.uuid4().hex[:10]}',
            native_seed=self.native_seed,split=self.course_split,game_exe_sha256=self.controller.reader.digest,
            base_stage_frame=64,base_state_sha256=state_fingerprint(self.snapshot),
            base_visual_rng_alignment=dict(schema=VISUAL_ALIGNMENT_SCHEMA,reference_state=semantic_state(self.snapshot)),
            records=[],resources_injected=False,native_deaths=True,verified_independent_resets=0,
            source='current policy fresh native chain; excluded from PPO until re-sampled',native_six_stage_clear=False)
        self._record_harvest()

    def _record_harvest(self):
        s=self.snapshot;events=s.extra['thrl_events'];roles=events['ecl_roles']
        self._harvest['records'].append(dict(ticks=sum(t for _,t in self.controller._input_tape),
            state_sha256=state_fingerprint(s),frame=s.frame,stage=s.stage,stage_frame=s.stage_frame,
            phase=self._phase.observe(s),lives=s.lives,bombs=s.bombs,power=s.power,
            phase_state=self._phase.state_dict(),
            playable=bool(s.player.state==1 and not s.dialogue),
            attacks=[r['ecl_sub_name'] for r in roles.get('midbosses',[])+roles.get('main_bosses',[])],
            reward_tracking=dict(schema='thrl-prefix-reward-state-v1',cfg_bombs=self.reward_model.cfg_bombs,
                rem_bombs_internal=self.reward_model.previous.rem_bombs_internal,
                **{k:events[k] for k in ('misses','bombs_used','point_items')})))

    def _finish_harvest(self,info):
        from .thrl_course_store import slice_tape
        from .atomic_json_io import atomic_write_json
        trace=self._harvest;self._harvest=None
        tape=self.controller.end_input_tape()
        if len(trace['records'])<16 or trace['records'][-1]['ticks']<=0:return
        trace['raw_input_segments']=slice_tape(tape,0,trace['records'][-1]['ticks'])
        trace['capture_end']=info['episode_summary']
        trace['native_six_stage_clear']=info.get('native_terminal_kind')=='clear'
        folder=self.course.overlay/f'harvest_{trace["id"]}';folder.mkdir(parents=True,exist_ok=False)
        atomic_write_json(folder/'captured.json',trace)
        info['curriculum_harvest_candidate']=str(folder/'captured.json')
