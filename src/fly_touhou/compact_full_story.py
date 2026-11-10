"""Map-free SNN full-story task: fresh opening to native Game Over or clear."""
from pathlib import Path
import time
import numpy as np
import torch
from .thrl_full_story_env import THRLFullStoryEnv
from .thrl_teacher import THRLTeacherEnv,make_teacher,teacher_schema
from .thrl_durable_records import write_json,append_jsonl


def apply_sensor_patches(model,config,*,native_motion=False,native_focus=False,
                         unknown_laser_motion=False,native_curve=False):
    """Apply requested native sensor semantics to a loaded story checkpoint.

    Checkpoints retain their enabled flags. A patched environment can also
    enable them for an older checkpoint, including in standalone evaluation.
    These flags affect the sensory interface, not the connectome or actions.
    """
    import json
    before={name:bool(getattr(model,name,False)) for name in (
        'native_motion_forecast','native_focus_gate','unknown_laser_player_motion',
        'native_curve_laser_geometry','native_curve_motion_forecast','native_enemy_damage_bound')}
    focus=before['native_focus_gate'] or native_focus or bool(config.get('compact_native_focus_gate'))
    curve_motion=before['native_curve_motion_forecast'] or bool(config.get('compact_native_curve_motion_forecast'))
    damage_bound=before['native_enemy_damage_bound'] or bool(config.get('compact_native_enemy_damage_bound'))
    curve=before['native_curve_laser_geometry'] or native_curve or curve_motion or bool(config.get('compact_native_curve_laser_geometry'))
    unknown=(before['unknown_laser_player_motion'] or unknown_laser_motion or curve
             or bool(config.get('compact_unknown_laser_player_motion')))
    motion=(before['native_motion_forecast'] or native_motion or focus or curve or damage_bound
            or bool(config.get('compact_native_motion_forecast')))
    if focus or curve:
        profile=config.get('profile')
        if not profile:raise ValueError('Native sensor patches require their game profile')
        hooks=json.loads(Path(profile).read_text(encoding='utf-8-sig')).get('hooks',{})
        if focus and hooks.get('native_focus_gate')!='th14-native-focus-v1':
            raise ValueError('The selected game profile does not provide native focus-gate data')
        if curve and hooks.get('native_curve_laser_geometry')!='th14-native-curve-v1':
            raise ValueError('The selected game profile does not provide native curve-laser data')
    after=dict(native_motion_forecast=motion,native_focus_gate=focus,
               unknown_laser_player_motion=unknown,native_curve_laser_geometry=curve,native_curve_motion_forecast=curve_motion,
               native_enemy_damage_bound=damage_bound)
    for name,value in after.items():setattr(model,name,value)
    hold_revision=native_position_hold_revision(config) if motion else None
    hidden_revision=native_hidden_enemy_revision(config)
    polar_revision=native_ex8_polar4_revision(config) if motion else None
    if hidden_revision and not motion:raise ValueError('Hidden enemy contacts require native motion forecasts')
    return dict(before=before,enabled=after,changed=[name for name in after if after[name]!=before[name]],
                native_position_hold_forecast=hold_revision,native_hidden_enemy_contacts=hidden_revision,
                native_ex8_polar4_forecast=polar_revision,native_wall_reactive_forecast=native_wall_revision(config),
                native_ex_scale_forecast=native_scale_revision(config))


def native_position_hold_revision(config):
    """Report the profile-selected bullet hold forecast used by game workers."""
    import json
    profile=config.get('profile')
    if not profile:return None
    hooks=json.loads(Path(profile).read_text(encoding='utf-8-sig')).get('hooks') or {}
    return hooks.get('native_position_hold_forecast')


def native_hidden_enemy_revision(config):
    import json
    profile=config.get('profile')
    if not profile:return None
    hooks=json.loads(Path(profile).read_text(encoding='utf-8-sig')).get('hooks') or {}
    return hooks.get('native_hidden_enemy_contacts')


def native_ex8_polar4_revision(config):
    """Report the separately opted-in, short EX8 candidate used by workers."""
    import json
    profile=config.get('profile')
    if not profile:return None
    hooks=json.loads(Path(profile).read_text(encoding='utf-8-sig')).get('hooks') or {}
    return hooks.get('native_ex8_polar4_forecast')


def native_wall_revision(config):
    import json
    profile=config.get('profile')
    if not profile:return None
    return (json.loads(Path(profile).read_text(encoding='utf-8-sig')).get('hooks') or {}).get('native_wall_reactive_forecast')


def native_scale_revision(config):
    import json
    profile=config.get('profile')
    if not profile:return None
    return (json.loads(Path(profile).read_text(encoding='utf-8-sig')).get('hooks') or {}).get('native_ex_scale_forecast')


class CompactFullStoryEnv(THRLFullStoryEnv):
    _attach=THRLTeacherEnv._attach
    def __init__(self,*args,teacher_revision=None,forecast_native_arming=False,primary_attack_target=False,forecast_growth_motion=False,full_native_teacher_support=False,protected_exit_forecast=False,protected_exit_from_stage=1,protected_exit_preserve_long_shield=False,stage6_prefix_capture=None,frozen_prefix_replay=None,native_motion_forecast=False,native_focus_gate=False,unknown_laser_player_motion=False,native_curve_laser_geometry=False,native_curve_motion_forecast=False,native_enemy_damage_bound=False,**kwargs):
        self.frozen_stage6_metrics=kwargs.pop('frozen_stage6_metrics',None)
        lower_reward=kwargs.pop('lower_field_reward',None)
        if lower_reward is not None and kwargs.get('reward_schema')!='th14-completion-reward-v4':
            raise ValueError('Lower-field shaping is calibrated to completion reward v4')
        super().__init__(*args,**kwargs)
        self.lower_field_reward=None
        if lower_reward is not None:
            from .compact_lower_field_reward import LowerFieldReward
            self.lower_field_reward=LowerFieldReward(gamma=self.reward_model.gamma,**lower_reward)
        self.stage6_prefix_capture=stage6_prefix_capture
        self.frozen_prefix_replay=frozen_prefix_replay
        self.native_enemy_damage_bound=bool(native_enemy_damage_bound)
        if primary_attack_target:
            if not hasattr(self.encoder,'primary_attack_target'):raise ValueError('Primary attack target requires the compact story sensor')
            self.encoder.primary_attack_target=True
        if forecast_native_arming:
            encoder=getattr(self.encoder,'base',self.encoder)
            if not hasattr(encoder,'anticipate_arming'):raise ValueError('Native arming forecast requires the compact sensor')
            encoder.anticipate_arming=True
        if forecast_growth_motion:
            if not forecast_native_arming:raise ValueError('Growth forecast requires native arming forecast')
            encoder=getattr(self.encoder,'base',self.encoder)
            encoder.growth_motion_envelope=True
        if protected_exit_forecast:
            encoder=getattr(self.encoder,'base',self.encoder)
            if not hasattr(encoder,'protected_exit_forecast'):raise ValueError('Protected exit forecast requires the compact sensor')
            encoder.protected_exit_forecast=True
            encoder.protected_exit_from_stage=int(protected_exit_from_stage)
            encoder.protected_exit_preserve_long_shield=bool(protected_exit_preserve_long_shield)
        if native_motion_forecast:
            encoder=getattr(self.encoder,'base',self.encoder)
            encoder.native_motion_forecast=True
        if native_focus_gate:
            encoder=getattr(self.encoder,'base',self.encoder)
            encoder.native_motion_forecast=True;encoder.native_focus_gate=True
        if unknown_laser_player_motion:
            encoder=getattr(self.encoder,'base',self.encoder)
            encoder.unknown_laser_player_motion=True
        if native_curve_laser_geometry:
            encoder=getattr(self.encoder,'base',self.encoder)
            encoder.native_motion_forecast=True;encoder.native_curve_laser_geometry=True
        if native_curve_motion_forecast:
            encoder=getattr(self.encoder,'base',self.encoder)
            encoder.native_curve_motion_forecast=True
        self._teacher=make_teacher(teacher_revision) if teacher_revision else None
        self._teacher_schema=teacher_schema(teacher_revision) if teacher_revision else None
        self._teacher_label=None;self._teacher_queries=0;self._teacher_actions=0
        self.full_native_teacher_support=bool(full_native_teacher_support)
        self.protected_exit_forecast=bool(protected_exit_forecast)
    def _create_controller(self):
        controller=super()._create_controller()
        if getattr(self,'native_enemy_damage_bound',False):
            controller.profile.hooks=dict(controller.profile.hooks or {},native_enemy_damage_bound=True)
        return controller
    def _label_current(self,info):
        restore=getattr(self,'full_native_teacher_support',False)
        bind=getattr(self._teacher,'bind_native_geometry',None)
        if bind is not None:
            bind(self.snapshot,getattr(self.encoder,'last_result',None) if restore else None)
        result=THRLTeacherEnv._label_current(self,info)
        protected=getattr(self,'protected_exit_forecast',False)
        if not (restore or protected):return result
        encoding=getattr(self.encoder,'last_result',None)
        if encoding is None:return result
        support_mix=getattr(self._teacher,'support_mix',.1)
        if restore:
            from .compact_teacher_support import restore_full_native_support,correct_avoidable_bomb_reference
            self._teacher_label,detail=restore_full_native_support(self._teacher_label,encoding,support_mix=support_mix)
            self._teacher_label['full_native_support_attempt']=detail
            self._teacher_label,resource_detail=correct_avoidable_bomb_reference(self._teacher_label,encoding)
            self._teacher_label['resource_reference_correction']=resource_detail
            from .compact_route_guidance import improve_route_reference
            self._teacher_label,route_detail=improve_route_reference(self._teacher_label,encoding,support_mix=support_mix)
            self._teacher_label['native_route_guidance']=route_detail
        if protected:
            from .compact_protected_exit_candidate import protected_exit_reference
            self._teacher_label,detail=protected_exit_reference(self._teacher_label,encoding,support_mix=support_mix)
            self._teacher_label['protected_exit_support_attempt']=detail
        return self._attach(result)
    def _info(self,elapsed=0):
        info=super()._info(elapsed)
        r=getattr(getattr(self,'encoder',None),'last_result',None)
        if r is not None and self.snapshot is not None and (r['frame'],r['stage'])==(self.snapshot.frame,self.snapshot.stage):
            d=r['details']
            info['compact_observation_frame']=self.snapshot.frame
            info['compact_targets']=dict(margins=d['margins'].tolist(),trusted=d['trusted'].tolist(),normal=d['normal'],
                frame=r['frame'],native_episode=int(self.episode_id),uncertainty=d.get('uncertainty'),
                lower_field_exception=bool((r.get('source_native_forecasts') or {}).get('reactive_bullets')))
            if 'partial_point_danger' in d:
                info['compact_targets'].update(partial_point_danger=d['partial_point_danger'].tolist(),
                    partial_point_danger_revision=d['partial_point_danger_revision'])
            if getattr(self,'protected_exit_forecast',False):
                info['compact_targets'].update(risk_training_eligible=bool(d['risk_training_eligible']),
                    risk_training_trusted=d['risk_training_trusted'].tolist(),
                    protected_exit_revision=d['protected_exit_revision'])
        return info
    def reset(self,*,seed=None,options=None):
        if options is not None:
            if set(options)!={'evaluation_seed'}:raise ValueError('Only a fresh seeded original opening is allowed')
            seed=int(options['evaluation_seed'])
        if self._teacher is not None:self._teacher.reset();self._teacher_label=None
        obs,info=super().reset(seed=seed,options=None)
        if getattr(self,'frozen_prefix_replay',None):
            from .compact_frozen_prefix_replay import restore_environment
            obs,info=restore_environment(self,self.frozen_prefix_replay)
        if self._teacher is not None:info=self._label_current(info)
        if getattr(self,'lower_field_reward',None) is not None:
            self.lower_field_reward.reset(self.snapshot)
            info['reward_shaping']=self.lower_field_reward.specification()
        if getattr(self,'stage6_prefix_capture',None):
            from .compact_frozen_prefix import start
            start(self)
        return obs,info
    def step(self,action):
        before=self._teacher_label
        previous_snapshot=getattr(self,'snapshot',None)
        previous_encoding=getattr(self.encoder,'last_result',None)
        from .compact_frozen_prefix import before_step,after_step
        capture_before=before_step(self)
        obs,reward,done,truncated,info=super().step(action)
        if getattr(self,'lower_field_reward',None) is not None:
            from .compact_lower_field_reward import apply_reward
            components,diagnostics=self.lower_field_reward.transition(previous_snapshot,self.snapshot,
                previous_encoding,info['reward_diagnostics'],terminal=done)
            reward=apply_reward(self,info,components,diagnostics)
        from .compact_story_contact_evidence import contact_transition
        evidence=contact_transition(previous_snapshot,getattr(self,'snapshot',None),
            before_encoding=previous_encoding,after_encoding=getattr(self.encoder,'last_result',None),
            teacher_label=before)
        if evidence is not None:info['compact_contact_evidence']=evidence
        if self._teacher is not None:
            self._teacher_label=None;info=self._attach(info)
            info.update(reference_label_frame=before['frame'],reference_action=before['action'])
            if not done:info=self._label_current(info)
        after_step(self,capture_before,info,done)
        if done and getattr(self,'_frozen_prefix_replay_receipt',None):
            receipt=self._frozen_prefix_replay_receipt
            info['episode_summary'].update(frozen_prefix_replay=receipt,
                full_story_prefix_played_by_current_policy=False,
                episode_statistics_scope=receipt['episode_statistics_scope'])
        return obs,reward,done,truncated,info


def factory(options):
    options=dict(options);revision=options.pop('online_teacher_revision',None)
    if options.get('nbnm'):
        from .nbnm_env import NBNMEnv
        return NBNMEnv(**options,teacher_revision=revision)
    return CompactFullStoryEnv(**options,teacher_revision=revision)


def story_kwargs(config,output,workers,role,*,teacher=False):
    import copy,train_thrl_native
    from .compact_story_sensation import SCHEMA
    cfg=copy.deepcopy(config);cfg['input_schema']=SCHEMA
    if teacher:
        from .resource_teacher_v10 import REVISION
        cfg['teacher_revision']=REVISION;cfg.setdefault('movement_guidance',{})['online_teacher_coefficient']=.1
    rows=train_thrl_native.make_env_kwargs(cfg,Path(output),workers,role)
    if config.get('nbnm'):
        from .nbnm_sensation import SCHEMA as NBNM_SCHEMA
        for i,r in enumerate(rows):
            r.update(input_schema=NBNM_SCHEMA,nbnm=dict(config['nbnm'],seed=cfg['seed']+1009*i))
            tasks=config['nbnm'].get('world_tasks')
            if tasks:
                if len(tasks)!=workers:raise ValueError('One NBNM task per native worker required')
                r['nbnm']['task']=dict(tasks[i])
    if cfg.get('compact_lower_field_reward') is not None:
        for r in rows:r['lower_field_reward']=dict(cfg['compact_lower_field_reward'])
    if cfg.get('compact_capture_stage6_prefix'):
        for i,r in enumerate(rows):r['stage6_prefix_capture']=str(Path(output)/'stage6-prefix'/f'world-{i}.json')
    if cfg.get('compact_capture_stage6_prefix') or cfg.get('compact_frozen_prefix_replay'):
        for i,r in enumerate(rows):r['frozen_stage6_metrics']=str(Path(output)/f'stage6-motion-world-{i}.json')
    if cfg.get('compact_frozen_prefix_replay'):
        if teacher or cfg.get('compact_capture_stage6_prefix'):
            raise ValueError('Frozen prefix continuation is a separate teacher-free evaluation')
        for r in rows:r['frozen_prefix_replay']=cfg['compact_frozen_prefix_replay']['native']
    if cfg.get('compact_forecast_native_arming'):
        for r in rows:r['forecast_native_arming']=True
    if cfg.get('compact_primary_attack_target'):
        for r in rows:r['primary_attack_target']=True
    if cfg.get('compact_forecast_growth_motion'):
        for r in rows:r['forecast_growth_motion']=True
    if cfg.get('compact_protected_exit_forecast'):
        for r in rows:
            r.update(protected_exit_forecast=True,
                protected_exit_from_stage=int(cfg.get('compact_protected_exit_from_stage',1)),
                protected_exit_preserve_long_shield=bool(cfg.get('compact_protected_exit_preserve_long_shield',False)))
    if cfg.get('compact_native_motion_forecast'):
        for r in rows:r['native_motion_forecast']=True
    if cfg.get('compact_native_focus_gate'):
        for r in rows:r['native_focus_gate']=True
    if cfg.get('compact_unknown_laser_player_motion'):
        for r in rows:r['unknown_laser_player_motion']=True
    if cfg.get('compact_native_curve_laser_geometry'):
        for r in rows:r['native_curve_laser_geometry']=True
    if cfg.get('compact_native_curve_motion_forecast'):
        for r in rows:r['native_curve_motion_forecast']=True
    if cfg.get('compact_native_enemy_damage_bound'):
        for r in rows:r['native_enemy_damage_bound']=True
    if teacher and cfg.get('compact_full_native_teacher_support'):
        for r in rows:r['full_native_teacher_support']=True
    if not teacher:
        for r in rows:r.pop('online_teacher_revision',None)
    return rows


@torch.no_grad()
def evaluate(model,config,output,*,episodes=4,seed=2027432926,worker_cpus=None,guard=lambda:None,progress=lambda x:None,record_diagnostics=True,
             action_selection=None,policy_seed=2027432927,evaluation_split='development',opening_seeds=None,
             movement_sampling_temperature=1.,bomb_sampling_temperature=1.):
    from . import thrl_vector
    from .compact_spiking_run import kwargs,infer
    from .fly_dn_cost_native_policy import make_dn_cost_evaluation_vector
    from .compact_story_action_selection import StoryActionSelector
    if bool(config.get('compact_protected_exit_forecast',False))!=bool(getattr(model,'protected_exit_forecast',False)):
        raise ValueError('Frozen model and environment protected exit semantics differ')
    config=dict(config,compact_native_motion_forecast=bool(getattr(model,'native_motion_forecast',False)),
        compact_native_focus_gate=bool(getattr(model,'native_focus_gate',False)),
        compact_unknown_laser_player_motion=bool(getattr(model,'unknown_laser_player_motion',False)),
        compact_native_curve_laser_geometry=bool(getattr(model,'native_curve_laser_geometry',False)),
        compact_native_curve_motion_forecast=bool(getattr(model,'native_curve_motion_forecast',False)),
        compact_native_enemy_damage_bound=bool(getattr(model,'native_enemy_damage_bound',False)))
    if evaluation_split not in ('development','test'):raise ValueError('Frozen development or held-out test split required')
    action_selection=action_selection or getattr(model,'full_story_action_selection','joint_mode')
    replay_spec=config.get('compact_frozen_prefix_replay')
    if replay_spec and (episodes!=1 or action_selection=='sample'):
        raise ValueError('Recorded brain continuation uses one deterministic frozen branch')
    selector=StoryActionSelector(action_selection,episodes,seed=policy_seed,
        movement_temperature=movement_sampling_temperature,bomb_temperature=bomb_sampling_temperature)
    if action_selection!='joint_mode' and not getattr(model,'candidate_conditioned',False):
        raise ValueError('Alternate decoding requires action-consistent candidate-state commits')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    maker=story_kwargs if getattr(model,'input_schema',None)=='connectome-compact129-full-story-v1' else kwargs
    rows=maker(config,output,episodes,evaluation_split);old=thrl_vector.worker_factory;vector=None
    plans=([int(np.random.SeedSequence([seed,i]).generate_state(2)[0]) for i in range(episodes)]
        if opening_seeds is None else [int(value) for value in opening_seeds])
    if len(plans)!=episodes:raise ValueError('One explicit original opening seed per frozen game is required')
    write_json(output/'episode_plan.json',dict(seed=seed,seeds=plans,full_six_stage=True,episode_stop='native Game Over or six-stage clear',
        finite_native_lives=True,continues=False,frame_cap=None,teacher_actions=0,optimizer_updates=0,
        detailed_contact_diagnostics=record_diagnostics,action_selection=action_selection,policy_seed=policy_seed,
        evaluation_split=evaluation_split,frozen_prefix_replay=replay_spec,
        movement_sampling_temperature=movement_sampling_temperature,bomb_sampling_temperature=bomb_sampling_temperature,
        live_policy_all_six_stages=not bool(replay_spec)))
    states={};pending={};infos={};observations={};results=[];decisions=0;begun=time.monotonic();last=0
    captured_brains=set()
    from .compact_story_diagnostics import StoryDiagnostics
    diagnostics=StoryDiagnostics(output/'diagnostics',episodes) if record_diagnostics else None
    from .compact_combat_position import PositionMonitor
    position_monitor=PositionMonitor()
    try:
        thrl_vector.worker_factory=factory
        Vector=make_dn_cost_evaluation_vector(worker_cpus,output)
        vector=Vector(rows,output/'workers',**({'timeout':600} if replay_spec else {}))
        vector.submit_at(list(range(episodes)),'reset',options=[dict(evaluation_seed=s) for s in plans])
        while vector._inflight:
            guard();ready=[]
            for msg in vector.receive_ready(max_wait_ms=2,min_batch=min(4,episodes),check_stop=guard):
                i=msg['index'];info=msg['info'];features=msg['observation']['features']
                if msg['operation']!='reset':
                    previous=infos[i]
                    position_monitor.add(observations[i],
                        frames=max(0,int(info.get('episode_frames',0))-int(previous.get('episode_frames',0))),
                        contacts=int((info.get('misses',0)>previous.get('misses',0) and observations[i][14]>.5)
                            or (features[17]>.5 and observations[i][17]<.5)),
                        misses=max(0,int(info.get('misses',0))-int(previous.get('misses',0))),
                        bombs=max(0,int(info.get('bombs_used',0))-int(previous.get('bombs_used',0))),world=i)
                if diagnostics is not None and msg['operation']!='reset':
                    diagnostics.after_action(i,infos[i],info,observations[i],features,terminated=bool(msg.get('terminated')))
                infos[i]=info;observations[i]=features
                if msg.get('truncated'):raise ValueError('Full-game score cannot include a frame/time cap')
                if msg['operation']=='reset':
                    if info.get('game_rng_seed_applied') is not True:raise ValueError('Original opening seed was not applied')
                    states[i]=model.initial_state(1)
                    if replay_spec:
                        from .compact_frozen_prefix_replay import restore_brain
                        states[i]=restore_brain(model,replay_spec['brain'],features,plans[i],
                            receipt_path=output/'frozen-brain-restore.json')
                        vector.timeout=90.
                else:
                    decisions+=1
                    if info['policy_action']!=pending[i]:raise ValueError('Native executed a different action')
                if msg.get('terminated'):
                    r=dict(info['episode_summary'],evaluation_game=i+1,evaluation_seed=plans[i])
                    if r['outcome'] not in ('gameover','clear'):raise ValueError('Only native complete games count')
                    results.append(r);append_jsonl(output/'episodes.jsonl',r)
                else:ready.append(i)
            if ready:
                f=np.stack([observations[i] for i in ready])
                if config.get('compact_capture_stage6_prefix'):
                    for j,i in enumerate(ready):
                        stage=infos[i].get('stage')
                        if stage in (5,6) and (i,stage) not in captured_brains:
                            destination=output/f'stage{stage}-prefix';destination.mkdir(parents=True,exist_ok=True)
                            torch.save(dict(schema='frozen-SNN-stage-entry-brain-v2',
                                state=[v.detach().cpu().clone() for v in states[i].tensors()],
                                features=np.array(f[j],copy=True),opening_seed=plans[i],
                                context={k:infos[i].get(k) for k in ('stage','stage_frame','episode_frames','lives','bombs','misses','bombs_used')},
                                frozen_decoding=getattr(model,'frozen_decoding',None),optimizer_updates=0),
                                destination/f'brain-world-{i}.pt')
                            captured_brains.add((i,stage))
                if getattr(model,'candidate_conditioned',False):
                    out,following,_=model.infer_candidates(f,[states[i] for i in ready],return_all_DN=diagnostics is not None,
                        action_selector=lambda logits:selector(logits,ready))
                else:out,following,_=infer(model,f,[states[i] for i in ready])
                actions=out.get('actions',out['logits'].argmax(-1)).cpu().numpy()
                if diagnostics is not None:diagnostics.before_actions(ready,f,infos,out,actions)
                for j,i in enumerate(ready):states[i]=following[j];pending[i]=int(actions[j])
                vector.submit_at(ready,'step',actions.tolist())
            if time.monotonic()-last>5:
                d=dict(phase='frozen_full_story',completed=len(results),target=episodes,student_decisions=decisions,
                    elapsed_seconds=time.monotonic()-begun,action_selection=action_selection,
                    movement_sampling_temperature=movement_sampling_temperature,bomb_sampling_temperature=bomb_sampling_temperature,
                    active=[dict(world=i,**{k:infos[i].get(k) for k in
                    ('stage','stage_frame','episode_frames','lives','bombs','misses','bombs_used')}) for i in vector._inflight if i in infos])
                write_json(output/'progress.json',d);progress(d);last=time.monotonic()
                write_json(output/'position-metrics.json',position_monitor.summary())
        from .compact_native_forecast import native_repair_revisions
        report=dict(status='completed',episodes=episodes,clears=sum(r['outcome']=='clear' for r in results),
            frozen_decoding=getattr(model,'frozen_decoding',None),
            native_repair_revisions=native_repair_revisions(reactive_orbs=native_wall_revision(config),
                bullet_scale=native_scale_revision(config)),
            stage_counts={str(s):sum(max(r['stage_history'])>=s for r in results) for s in range(1,7)},
            episode_results=sorted(results,key=lambda r:r['evaluation_game']),full_game=not bool(replay_spec),
            full_native_run=True,frozen_prefix_replay=replay_spec,teacher_actions=0,optimizer_updates=0,
            elapsed_seconds=time.monotonic()-begun,student_decisions=decisions,action_selection=action_selection,policy_seed=policy_seed,
            evaluation_split=evaluation_split,
            movement_sampling_temperature=movement_sampling_temperature,bomb_sampling_temperature=bomb_sampling_temperature,
            native_position_hold_forecast=(
                native_position_hold_revision(config) if getattr(model,'native_motion_forecast',False) else None),
            native_ex8_polar4_forecast=(native_ex8_polar4_revision(config)
                if getattr(model,'native_motion_forecast',False) else None),
            native_hidden_enemy_contacts=native_hidden_enemy_revision(config),
            native_wall_reactive_forecast=native_wall_revision(config))
        if diagnostics is not None:report['dodge_diagnostics']=diagnostics.finish()
        report['position_metrics']=position_monitor.summary()
        write_json(output/'position-metrics.json',report['position_metrics'])
        write_json(output/'report.json',report);return report
    finally:
        thrl_vector.worker_factory=old
        if vector is not None:
            vector.close();write_json(output/'cleanup.json',dict(errors=vector.close_errors))
            if vector.close_errors:raise RuntimeError('Full-story compact worker cleanup unresolved')
