"""CUDA THRL port: full native-life TH14, baseline first, then bounded Optuna.

No desktop input, browser game, teacher, action override, or PPO update rejection.
The first trial is the unmodified-size no-attention model and fixed source PPO.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from fly_touhou.atomic_json_io import atomic_write_json as write_json
from fly_touhou.thrl_study_io import StatusAPI, read_shared_json, utc_now
from fly_touhou.thrl_policy_actions import policy_logits, validated_action_temperature
from fly_touhou.thrl_training_mode import (
    is_full_story, validate_training_mode, episode_protocol, full_story_environment_options, SEED_PARTITION)


class StopRequested(Exception):
    pass


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def development_summary(rows):
    """Selection uses native progress, never the tunable shaped reward itself."""
    if not rows or any(row['outcome'] not in ('clear', 'gameover') for row in rows):
        raise ValueError('Development selection requires complete native episodes')
    count = len(rows)
    clears = sum(row['outcome'] == 'clear' for row in rows)
    mean_frames = sum(row['native_frames'] for row in rows)/count
    mean_stage = sum(max(row['stage_history']) for row in rows)/count
    # With <=20 evaluation games, one additional clear wins over stage/time;
    # one additional reached stage in one game wins over every time tie-break.
    score = 1000.*clears/count + 10.*mean_stage + .25*mean_frames/(mean_frames+3600.)
    return {'episodes':count, 'clears':clears, 'clear_rate':clears/count,
        'mean_native_frames':mean_frames, 'mean_max_stage':mean_stage,
        'mean_misses':sum(row['misses'] for row in rows)/count,
        'mean_bombs_used':sum(row['bombs_used'] for row in rows)/count,
        'score':score, 'episode_results':rows, 'formal_test':False,
        'original_game_rng_controlled':False, 'selection_action':'categorical sampling, temperature1',
        'score_definition':'1000*clear_rate +10*mean_max_stage + .25*mean_frames/(mean_frames+3600)',
        'teacher_actions':0, 'safety_takeovers':0, 'continues':0}


def make_env_kwargs(config, output, count, role):
    from fly_touhou.th14_completion_protocol import reward_options
    full_story_options = full_story_environment_options(config,count,role)
    extra = ({'start_bank':config['start_bank'],'segment_end':config.get('segment_end'),
              'start_bank_sha256':config.get('start_bank_sha256'),
              'full_story_fraction':config.get('full_story_fraction',0.)}
             if role=='train' and config.get('start_bank') else {})
    from fly_touhou.thrl_course_learning import environment_options
    courses=None if full_story_options else environment_options(config,output,count,role)
    online=({'online_teacher_revision':config['teacher_revision']} if role=='train' and config.get('movement_guidance',{}).get('online_teacher_coefficient',0)>0 else {})
    return [dict(executable=config['game_exe'], profile_path=config['profile'],
        private_appdata=str(output/'native'/role/f'worker{i}'/'appdata'),
        difficulty=3, frame_skip=4, wall_time_limit_seconds=None,
        require_complete_reward_mapping=True, hide_window=True,
        headless=config.get('headless',True),
        input_schema=config.get('input_schema','semantic-separated-v1'),
        **({'laser_motion_audit_path':str(output/'laser_motion'/f'worker{i}.jsonl')}
           if role=='train' and config.get('laser_motion_telemetry') else {}),
        **({'point_motion_audit_path':str(output/'point_motion'/role/f'worker{i}.jsonl')}
           if config.get('point_motion_telemetry') else {}),
        window_title=f'THRL {role} {i}', timeout=25,
        policy_action_count=config['policy_action_count'],**online,
        **(full_story_options[i] if full_story_options else courses[i] if courses else
           {**reward_options(config),**extra,'curriculum_seed':config['seed']+1009*i} if extra else reward_options(config))) for i in range(count)]


def native_status(infos):
    return [{key: info.get(key) for key in ('pid','stage','stage_frame','lives','bombs','power',
                'misses','bombs_used','native_input_ticks','episode_frames','policy_steps')}
            | {'worker':i} | ({'laser_sensor':info['laser_sensor']} if 'laser_sensor' in info else {})
            for i, info in enumerate(infos)]


def save_checkpoint(torch, path, model, optimizer, *, decisions, updates, adam, config,
                    runtime_state=None):
    payload = {'schema':'th14-thrl-training-checkpoint-v1', 'created_utc':utc_now(),
        'model_config':model.model_config(), 'model':model.state_dict(),
        'optimizer':optimizer.state_dict(), 'decisions':decisions,
        'optimizer_parameter_names':list(dict(model.named_parameters())),
        'rollout_updates':updates, 'adam_minibatch_steps':adam, 'config':config,
        'action_temperature':validated_action_temperature(config.get('parameters',{}).get('action_temperature',1.)),
        'branch_decisions':decisions,'branch_rollout_updates':updates,
        'branch_adam_minibatch_steps':adam,
        'lifetime_decisions':decisions+config.get('inherited_decisions',0),
        'lifetime_rollout_updates':updates+config.get('inherited_rollout_updates',0),
        'lifetime_adam_minibatch_steps':adam+config.get('inherited_adam_minibatch_steps',0),
        'candidate_decisions':decisions+config.get('candidate_decision_offset',0),
        'candidate_optimized_decisions':updates*config.get('rollout_total_decisions',0)
                                        +config.get('candidate_decision_offset',0),
        'candidate_paid_decisions_lower_bound':decisions+config.get('candidate_paid_decision_offset',0),
        'unoptimized_sampled_decisions':decisions-updates*config.get('rollout_total_decisions',0),
        'inherited_unoptimized_sampling_lower_bound':config.get('inherited_unoptimized_sampling_lower_bound',0),
        'additional_inflight_native_cost_unknown':config.get('unknown_inflight_decisions',False),
        'optimizer_transaction_complete':(runtime_state or {}).get('optimizer_transaction_complete',True),
        'checkpoint_phase':(runtime_state or {}).get('phase'),
        'curriculum_state':(runtime_state or {}).get('curriculum_state'),
        'torch_rng':torch.get_rng_state(),
        'cuda_rng':torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else [],
        'numpy_rng':__import__('numpy').random.get_state(), 'python_rng':random.getstate(),
        'native_resume':('new native runs, current-parameter bounded observation burn-in; no full-world snapshot'
                         if config.get('curriculum_manifest') else 'new native runs, zero recurrent state; no full-world snapshot')}
    temporary = path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def allocate_rollout(torch, steps, envs, hidden_size, input_schema='semantic-separated-v1',*,movement_guidance=False,online_teacher=False):
    # Source FP32 maps, no lossy quantization or full-rollout CUDA allocation.
    from fly_touhou.thrl_inputs import observation_shapes
    result={key:torch.empty(steps,envs,*shape) for key,shape in observation_shapes(input_schema).items()}
    result.update(dict(
        actions=torch.empty(steps,envs,dtype=torch.long),
        old_log_probs=torch.empty(steps,envs), values=torch.empty(steps,envs,3),
        rewards=torch.empty(steps,envs,3), hidden_states=torch.empty(steps,envs,hidden_size),
        terminated=torch.empty(steps,envs,dtype=torch.bool),
        truncated=torch.empty(steps,envs,dtype=torch.bool),
        episode_starts=torch.empty(steps,envs,dtype=torch.bool),
        truncation_values=torch.zeros(steps,envs,3)))
    if movement_guidance:result['movement_outcomes']=torch.full((steps,envs),-1,dtype=torch.int8)
    if online_teacher:
        result['teacher_targets']=torch.empty(steps,envs,19)
        from fly_touhou.thrl_inputs import SEPARATED_LASER_INPUT
        if input_schema==SEPARATED_LASER_INPUT:result['teacher_weights']=torch.empty(steps,envs)
    return result


def evaluate(torch, model, config, output, *, episodes, seed, check_stop, progress=None,native_split='development'):
    """Fresh complete native games. Evaluation RNG never advances training RNG."""
    full_story = validate_training_mode(config)
    action_temperature=validated_action_temperature(config.get('parameters',{}).get('action_temperature',1.))
    selection=config.get('full_story_evaluation_action_selection','sampled')
    if selection not in ('sampled','greedy'):
        raise ValueError('Full-game action selection must be sampled or greedy')
    if full_story and config.get('development_objective')=='avoidance-first-v1' and Path(output).name.startswith('development_'):
        if action_temperature != 1.:
            raise ValueError('Legacy no-Bomb development has no non-T1 temperature contract; use an explicit full-story evaluation')
        if selection!='sampled':raise ValueError('No-Bomb development has its own action-selection protocol')
        from fly_touhou.thrl_avoidance_development import evaluate as avoidance_evaluate
        count=min(episodes,config.get('avoidance_development_workers',1))
        kwargs=make_env_kwargs(config,output,count,'development')
        for row in kwargs:row['probe_max_frames']=config.get('avoidance_development_max_frames',4096)
        return avoidance_evaluate(torch,model,kwargs,output,episodes=episodes,
            seed=config.get('avoidance_development_seed',seed),check_stop=check_stop,progress=progress)
    if full_story and config.get('full_story_evaluation_backend')=='full-story-ready-parallel-v1':
        from fly_touhou.thrl_full_story_evaluation import evaluate as parallel_evaluate
        workers=min(episodes,config.get('full_story_evaluation_workers',6))
        return parallel_evaluate(torch,model,make_env_kwargs(config,output,workers,native_split),output,
            episodes=episodes,seed=seed,check_stop=check_stop,progress=progress,
            native_split=native_split,episode_protocol=episode_protocol(config),summary=development_summary,
            trace_config=config.get('full_story_trace'),action_selection=selection,
            action_temperature=action_temperature)
    if selection!='sampled':raise ValueError('Greedy full-game evaluation requires the qualified parallel backend')
    import numpy as np
    from torch.distributions import Categorical
    from fly_touhou.thrl_vector import THRLNativeVector
    from fly_touhou.thrl_inputs import forward_observation
    output.mkdir(parents=True, exist_ok=False)
    vector = None
    rows = []
    movement_tracker=None
    if config.get('movement_guidance_revision'):
        from fly_touhou.thrl_movement_learning import MovementEvidenceTracker,transition_outcome
        movement_tracker=MovementEvidenceTracker()
    model.eval()
    # Full-story environments seed the original game in a disjoint native split.
    # Legacy unconstrained runs retain their explicit uncontrolled-RNG label.
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        try:
            check_stop()
            vector = THRLNativeVector(make_env_kwargs(config,output,1,native_split), output/'workers')
            course_options=None
            if not full_story and config.get('curriculum_manifest'):
                from fly_touhou.thrl_curriculum import schema_for
                course_options=[dict(split=native_split,assignment=dict(schema=schema_for(config.get('curriculum_options',{})),mode='chain',
                    lesson='s6_boss',lesson_index=23,frontier=23,depth=None))]
            observation, infos = vector.reset(options=course_options) if course_options else vector.reset()
            hidden = torch.zeros(1,1,model.gru_hidden_size,device='cuda')
            last_write = 0.
            while len(rows) < episodes:
                check_stop()
                with torch.no_grad():
                    logits, _, hidden = forward_observation(torch,model,observation,hidden)
                    action = Categorical(logits=policy_logits(logits,
                        action_temperature=action_temperature)).sample().cpu().numpy()
                if movement_tracker is not None:
                    before_features=observation['features'].copy();before_infos=infos
                observation, _, terminated, truncated, infos = vector.step(action)
                if movement_tracker is not None:
                    outcome=transition_outcome(before_features[0],before_infos[0],int(action[0]),
                        observation['features'][0],infos[0])
                    movement_tracker.record(0,before_features[0],before_infos[0],int(action[0]),
                        observation['features'][0],infos[0],outcome)
                if np.any(truncated):
                    raise ValueError('Unplanned external cut cannot be counted as a complete development game')
                now = time.monotonic()
                if terminated[0]:
                    row = dict(infos[0]['episode_summary'], evaluation_game=len(rows)+1,
                        action_temperature=action_temperature, action_selection='sampled')
                    rows.append(row)
                    with (output/'episodes.jsonl').open('a',encoding='utf-8',buffering=1) as stream:
                        stream.write(json.dumps(row,ensure_ascii=False)+'\n')
                    if len(rows) < episodes:
                        if course_options:vector.reset_at([0],options=course_options)
                        else:vector.reset_at([0])
                        hidden.zero_()
                        observation = vector.observation
                if now-last_write >= 5 or terminated[0]:
                    data = {'status':'development','completed':len(rows),'target':episodes,
                            'current_environments':native_status(infos)}
                    write_json(output/'progress.json',data)
                    if progress:
                        progress(data)
                    last_write = now
            result = development_summary(rows)
            if course_options or full_story:
                result.update(original_game_rng_controlled=True,native_seed_split=native_split,
                    native_seed_partition=SEED_PARTITION)
            if full_story:
                result.update(episode_protocol=episode_protocol(config),training_mode=config['training_mode'])
            write_json(output/'report.json',result)
            return result
        except BaseException:
            write_json(output/'failure.json',{'traceback':traceback.format_exc(),
                'complete_episodes':len(rows), 'unknown_prefix_not_counted':True})
            raise
        finally:
            if vector is not None:
                vector.close()
                write_json(output/'native_cleanup.json',{'close_errors':vector.close_errors,
                    'ownership':read_shared_json(vector.ownership_path)})
                if vector.close_errors:
                    raise RuntimeError('Development native cleanup has unresolved errors')


def validate_evaluation_schedule(config):
    """Reject an overlapping evaluation plan before launching any native world."""
    if config.get('evaluation_after_training_shutdown') and config['development_decisions']!=[config['trial_decisions']]:
        raise ValueError('Thirty-world continuation requires final-only development after training shutdown')


def train_trial(torch, config, study_output, output, parameters, *, trial, study_started,
                on_progress, attention='none',model_factory=None):
    full_story = validate_training_mode(config)
    validate_evaluation_schedule(config)
    import numpy as np
    from torch.distributions import Categorical
    from fly_touhou.thrl_model import MOActorCritic
    from fly_touhou.thrl_ppo import PPOConfig, RolloutBatch, make_optimizer, update
    from fly_touhou.thrl_inputs import make_encoder, observation_shapes, forward_observation, SEPARATED_INPUT
    from fly_touhou.th14_completion_protocol import make_reward_model,reward_options
    from fly_touhou.thrl_vector import THRLNativeVector
    import optuna

    output.mkdir(parents=True,exist_ok=False)
    started = time.monotonic()
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    overrides = dict(config.get('model_overrides',{}))
    schema=config.get('input_schema',SEPARATED_INPUT)
    feature_dim=observation_shapes(schema)['features'][0]
    overrides.setdefault('feature_dim',feature_dim)
    overrides.setdefault('input_schema',schema)
    if attention != 'none':
        overrides.setdefault('object_embed_dim',256)
        overrides.setdefault('object_attention_heads',8)
        overrides.setdefault('object_attention_layers',4)
        overrides.setdefault('spatial_attention_layers',2)
    if overrides['feature_dim'] != feature_dim or overrides['input_schema']!=schema:
        raise ValueError('Collector and model input schemas must match')
    explicit_model=config.get('policy_model_config')
    if explicit_model is not None:
        if model_factory is None:raise ValueError('Versioned policy configuration requires its explicit factory')
        model=model_factory(**explicit_model).cuda()
        if model.input_schema!=schema:raise ValueError('Explicit policy and collector schemas differ')
    else:
        model = (model_factory or MOActorCritic)(attention=attention,action_dim=config['policy_action_count'],
                             **overrides).cuda()
    from fly_touhou.th14_completion_protocol import learning_overrides
    ppo = PPOConfig(**{**parameters,**learning_overrides(config)})
    config={**config,'parameters':asdict(ppo)}
    optimizer = make_optimizer(model,ppo)
    initialization = None
    if config.get('warm_start_checkpoint'):
        from fly_touhou.thrl_branching import warm_start
        initialization = warm_start(torch,model,optimizer,config['warm_start_checkpoint'],
                                    config['warm_start_sha256'],
                                    exact_optimizer=config.get('candidate_resume_applied',False),model_factory=model_factory)
        if config.get('reset_Adam_after_BC'):
            if config.get('candidate_resume_applied') or not config.get('teacher_pretraining_result'):
                raise ValueError('Fresh PPO Adam is permitted only for an explicit BC transfer, never a PPO recovery')
            optimizer=make_optimizer(model,ppo)
            initialization.update(optimizer_moments_preserved=False,BC_optimizer_discarded=True,
                PPO_optimizer_initialization='fresh Adam at the declared PPO learning rate')
        if config.get('candidate_resume_applied'):
            if initialization['checkpoint_candidate_optimized_decisions'] != config['candidate_decision_offset']:
                raise ValueError('Recovery prefix differs from actual checkpoint optimized decisions')
            if initialization['source_rollout_total_decisions'] != config['rollout_total_decisions']:
                raise ValueError('Recovery cannot change the total rollout size')
            prior=initialization['source_parameters']
            if prior is None:
                raise ValueError('Recovery requires explicit source PPO parameters')
            prior=asdict(PPOConfig(**prior))
            current=asdict(ppo)
            prior.pop('update_index',None)
            current.pop('update_index',None)
            from fly_touhou.thrl_learning_revision import verify_ppo_resume
            parameter_migration=verify_ppo_resume(prior,current,config)
            if parameter_migration:
                initialization['learning_parameter_migration']=parameter_migration
                initialization['learning_migration']=config['learning_migration']
            if config['candidate_paid_decision_offset'] < initialization['checkpoint_candidate_paid_decisions_lower_bound']:
                raise ValueError('Recovery paid lower bound omits known source sampling')
        write_json(output/'initialization_audit.json',initialization)
    inherited_updates = 0 if initialization is None else initialization['inherited_rollout_updates']
    inherited_decisions = 0 if initialization is None else initialization['inherited_decisions']
    inherited_adam = 0 if initialization is None else initialization['inherited_adam_minibatch_steps']
    candidate_offset=config.get('candidate_decision_offset',0)
    full = {**config,'parameters':asdict(ppo),'model_config':model.model_config(),
        'parameter_counts':model.parameter_counts(),'attention':attention,
        'attention_specification':model.attention_specification(),
        'observation':make_encoder(schema).specification(),
        'reward':make_reward_model((-184,184,32,432),**reward_options(config)).specification(),
        'device':'cuda:0','gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,
        'initial_decisions':0,'initial_adam_steps':0,
        'BC_transfer':bool(config.get('teacher_pretraining_result')),
        'initialization':initialization,
        'inherited_decisions':inherited_decisions,'inherited_adam_minibatch_steps':inherited_adam,
        'inherited_rollout_updates':inherited_updates,
        'source_runtime':str(ROOT),'trial_number':None if trial is None else trial.number,
        'native_frame_horizon':None,'first_hit_terminal':False,'bombs_enabled':True,
        'continues_allowed':False,'teacher_actions':0,'safety_takeovers':0}
    if full_story:
        full['episode_protocol'] = episode_protocol(config)
    write_json(output/'config.json',full)
    history=None
    if config.get('candidate_resume_applied'):
        from fly_touhou.thrl_branching import inherit_candidate_history
        history=inherit_candidate_history(full,output)
        full.update(candidate_initial_checkpoint=history['candidate_initial']['copy'],
                    candidate_initial_sha256=history['candidate_initial']['sha256'])
    else:
        full['candidate_initial_checkpoint']=str((output/'initial.pt').resolve())
    decisions = updates = adam = games = clears = 0
    metrics, latest_development, recent = {}, None, None
    best_score = (history['best']['score'] if history and history['best'] else float('-inf'))
    sealed_baseline=None
    if config.get('sealed_avoidance_baseline'):
        if history:raise ValueError('A new sealed baseline cannot replace recovery history')
        from fly_touhou.thrl_entity_ppo import sealed_avoidance_baseline
        sealed_baseline=sealed_avoidance_baseline(torch,model,config['sealed_avoidance_baseline'])
        latest_development=sealed_baseline;best_score=sealed_baseline['score']
    completed_development=[] if history is None else history['completed_development'].copy()
    if completed_development:
        latest_development={**completed_development[-1]['result'],'inherited':True,
                            'candidate_decisions':completed_development[-1]['decisions']}
    optimizer_transaction_complete=True
    phase = 'initializing'
    infos = []
    vector = None
    course = None
    if not full_story and config.get('curriculum_manifest'):
        if config.get('collector_mode','ready')!='ready':
            raise ValueError('Progressive curriculum requires the qualified ready-driven collector')
        from fly_touhou.thrl_course_learning import CourseRuntime
        from fly_touhou.thrl_learning_revision import upgraded_course_state
        course=CourseRuntime(config,output,config['workers'],
            state=upgraded_course_state((initialization or {}).get('curriculum_state'),config)
                if config.get('candidate_resume_applied') else None)
    policy_history = None if course is None else course.history
    if full_story:
        from fly_touhou.thrl_recurrent import PolicyHistory
        from fly_touhou.thrl_inputs import policy_is_recurrent
        policy_history = PolicyHistory(config['workers'],config.get('full_story_burn_in',64),
                                       enabled=policy_is_recurrent(config),
                                       storage=config.get('full_story_history_storage','compressed-v1'))
    last_write = 0.
    total_collect = total_learn = total_evaluation = 0.
    total_recurrent_refresh=0.
    latest_rollout_timing = {}
    reward_recent = np.zeros(3)

    def check_stop():
        comparison_stop=config.get('comparison_stop_path')
        operator_stop=config.get('operator_stop_path')
        if ((study_output/'STOP').exists() or (output/'STOP').exists()
                or comparison_stop and Path(comparison_stop).exists()
                or operator_stop and Path(operator_stop).exists()):
            raise StopRequested('operator STOP; no other trial will start')
        if config.get('study_wall_cap_seconds') is not None and time.monotonic()-study_started >= config['study_wall_cap_seconds']:
            raise StopRequested('predeclared study wall budget reached')
        if config.get('trial_wall_cap_seconds') is not None and time.monotonic()-started >= config['trial_wall_cap_seconds']:
            raise StopRequested('predeclared trial wall budget reached')

    def report(force=False, evaluation=None):
        nonlocal last_write
        now = time.monotonic()
        if not force and now-last_write < 5:
            return
        elapsed = now-started
        value = {'schema':'th14-thrl-training-progress-v1',
            'status':phase if phase in ('failed','finished','pruned','operator_or_budget_stop') else 'training',
            'stage':phase,
            'updated_utc':utc_now(),'actual_decisions':decisions,'adam_minibatch_steps':adam,
            'rollout_updates':updates,'target_decisions':config['trial_decisions'],
            'candidate_decisions':decisions+candidate_offset,
            'candidate_target_decisions':config.get('candidate_trial_decisions',config['trial_decisions']),
            'candidate_decision_offset':candidate_offset,
            'candidate_optimized_decisions':candidate_offset+updates*config['rollout_total_decisions'],
            'candidate_paid_decisions_lower_bound':config.get('candidate_paid_decision_offset',0)+decisions,
            'unoptimized_sampled_decisions':decisions-updates*config['rollout_total_decisions'],
            'inherited_unoptimized_sampling_lower_bound':config.get('inherited_unoptimized_sampling_lower_bound',0),
            'additional_inflight_native_cost_unknown':config.get('unknown_inflight_decisions',False),
            'cost_is_lower_bound':True,
            'completed_development_nodes':[row['decisions'] for row in completed_development],
            'inherited_development_nodes':[] if history is None else [row['decisions'] for row in history['completed_development']],
            'candidate_resume_applied':config.get('candidate_resume_applied',False),
            'inherited_decisions':inherited_decisions,'lifetime_decisions':decisions+inherited_decisions,
            'inherited_adam_minibatch_steps':inherited_adam,
            'lifetime_adam_minibatch_steps':adam+inherited_adam,
            'workers':config['workers'],'parameter_counts':model.parameter_counts(),
            'episodes':games,'clear_episodes':clears,'wall_seconds':elapsed,
            'decisions_per_second':decisions/max(elapsed,1e-6),'metrics':metrics,
            'latest_rollout_timing':latest_rollout_timing,
            'frame_control':{'native_fps_cap_removed':bool(config.get('headless',True)),
                             'action_repeat_input_ticks':4,'episode_frame_limit':None},
            'current_environments':native_status(infos),'recent_episode_summary':recent,
            'reward_components_recent':reward_recent.tolist(),
            'latest_development':latest_development,'evaluation_progress':evaluation,
            'attention':attention,'collector_mode':config.get('collector_mode','ready'),'device':'cuda:0','gpu':torch.cuda.get_device_name(0),
            'action_temperature':ppo.action_temperature,'training_action_selection':'categorical_sample',
            'input_schema':schema,'observation_shapes':{key:list(value) for key,value in observation_shapes(schema).items()},
            'cuda_allocated_mb':torch.cuda.memory_allocated()/2**20,
            'cuda_reserved_mb':torch.cuda.memory_reserved()/2**20,
            'timing':{'collect_seconds':total_collect,'learn_seconds':total_learn,
                      'evaluation_seconds':total_evaluation,'recurrent_refresh_seconds':total_recurrent_refresh},
            'custom_update_rejections':0,'KL_early_stopping':bool(metrics.get('KL_early_stopping',False)),
            'KL_early_stopping_enabled':ppo.target_kl is not None,
            'teacher_actions':0,'safety_takeovers':0,'formal_test':False}
        if course is not None:value['curriculum']=course.summary()
        if full_story:
            value.update(training_mode=config['training_mode'],episode_protocol=episode_protocol(config),
                         recurrent_history=policy_history.summary())
        write_json(output/'progress.json',value)
        on_progress(value)
        last_write = now
        return value

    def checkpoint(name):
        save_checkpoint(torch,output/name,model,optimizer,decisions=decisions,
                        updates=updates,adam=adam,config=full,
                        runtime_state={'phase':phase,'optimizer_transaction_complete':optimizer_transaction_complete,
                                       'curriculum_state':None if course is None else course.scheduler.state_dict()})

    def develop(candidate_decisions):
        nonlocal phase,latest_development,total_evaluation,best_score
        phase='development'
        report(True)
        evaluation_started=time.monotonic()
        destination=output/f'development_{candidate_decisions:07d}'
        latest_development=evaluate(torch,model,config,destination,
            episodes=config['development_episodes'],seed=config['seed']+100000+candidate_decisions,
            check_stop=check_stop,progress=lambda row:report(True,evaluation=row))
        total_evaluation+=time.monotonic()-evaluation_started
        completed_development.append({'decisions':candidate_decisions,'score':latest_development['score'],
                                      'inherited':False,'report':str(destination/'report.json')})
        if latest_development['score']>best_score:
            best_score=latest_development['score']
            checkpoint('best_development.pt')
            write_json(output/'best_development.json',{'decisions':candidate_decisions,
                'segment_decisions':decisions,'score':best_score,'checkpoint':'best_development.pt',
                'checkpoint_sha256':digest(output/'best_development.pt'),
                'report':str(destination/'report.json'),'inherited':False})
        report(True)
        if trial is not None:
            grid=config.get('candidate_development_decisions',config['development_decisions'])
            node=grid.index(candidate_decisions)+1
            trial.report(latest_development['score'],node)
            if (trial.number>0 and not config.get('protected_architecture_baseline',False)
                    and trial.should_prune()):
                trial.set_user_attr('decisions',candidate_decisions)
                trial.set_user_attr('adam_minibatch_steps',adam)
                checkpoint('pruned.pt')
                phase='pruned'
                report(True)
                write_json(output/'training_result.json',{'status':'pruned','decisions':decisions,
                    'candidate_decisions':candidate_decisions,'adam_minibatch_steps':adam,
                    'latest_development':latest_development,'whole_trial_early_stop':True,
                    'PPO_epoch_rejections':0})
                raise optuna.TrialPruned('MedianPruner after complete independent development games')

    try:
        checkpoint('initial.pt')
        if sealed_baseline is not None:
            import shutil
            destination=output/'initial_avoidance_development'
            destination.mkdir(parents=True,exist_ok=False)
            write_json(destination/'report.json',sealed_baseline)
            shutil.copyfile(output/'initial.pt',output/'best_development.pt')
            write_json(output/'best_development.json',dict(decisions=0,segment_decisions=0,
                score=best_score,checkpoint='best_development.pt',
                checkpoint_sha256=digest(output/'best_development.pt'),
                report=str(destination/'report.json'),inherited=True,reused_existing_games=True))
        if not history:
            full['candidate_initial_sha256']=digest(output/'initial.pt')
        write_json(output/'config.json',full)
        if trial is not None and history:
            grid=config['candidate_development_decisions']
            for row in history['completed_development']:
                trial.report(row['score'],grid.index(row['decisions'])+1)
        report(True)
        check_stop()
        if history and history['pending_development_at_start'] is not None:
            develop(history['pending_development_at_start'])
            check_stop()
        pending_gate=(course.pending_gate_request(initialization,candidate_offset)
                      if course is not None and config.get('candidate_resume_applied') else None)
        if pending_gate is not None:
            phase='curriculum_gate';report(True)
            evaluation_started=time.monotonic()
            course.gate(torch,model,candidate_offset,check_stop,
                on_progress=lambda row:report(True,evaluation=row),retry_request=pending_gate)
            total_evaluation+=time.monotonic()-evaluation_started
            phase='initializing';checkpoint('latest.pt');report(True)
        vector = THRLNativeVector(make_env_kwargs(config,output,config['workers'],'train'),output/'workers',
                                 **({'timeout':900} if course is not None else {}))
        observation, infos = (vector.reset(options=[course.reset_options(i) for i in range(config['workers'])])
                              if course is not None else vector.reset())
        count = config['workers']
        steps = config['rollout_total_decisions']//count
        data = allocate_rollout(torch,steps,count,model.gru_hidden_size,schema,
            movement_guidance=ppo.movement_teacher_coefficient>0 or ppo.movement_collision_coefficient>0 or ppo.movement_dense_coefficient>0 or ppo.online_teacher_coefficient>0,
            online_teacher=ppo.online_teacher_coefficient>0)
        movement_tracker=None
        if 'movement_outcomes' in data:
            from fly_touhou.thrl_movement_learning import MovementEvidenceTracker
            movement_tracker=MovementEvidenceTracker()
        hidden = torch.zeros(1,count,model.gru_hidden_size,device='cuda')
        if course is not None:
            course.initial_history(vector);hidden=course.history.refresh(torch,model)
        starts = torch.ones(count,dtype=torch.bool)
        while decisions < config['trial_decisions']:
            check_stop()
            phase = 'collecting'
            model.eval()
            rollout_started = time.monotonic()
            if config.get('collector_mode','ready') == 'ready':
                from fly_touhou.thrl_ready_collection import collect_ready
                def ready_step(index, info, terminated, truncated):
                    nonlocal decisions,games,clears,recent,reward_recent
                    decisions += 1
                    if course is not None:course.acknowledge(index,info)
                    reward_recent = .95*reward_recent + .05*np.asarray(info['reward_vector'])
                    if terminated or truncated:
                        recent=info['episode_summary'];games+=1
                        clears+=int(recent['outcome']=='clear')
                        with (output/'training_episodes.jsonl').open('a',encoding='utf-8',buffering=1) as stream:
                            stream.write(json.dumps(dict(recent,at_decisions=decisions,worker=int(index)),ensure_ascii=False)+'\n')
                    report()
                observation,hidden,starts,infos=collect_ready(torch,model,vector,data,hidden,starts,infos,
                    logit_clamp=ppo.logit_clamp,action_temperature=ppo.action_temperature,
                    check_stop=check_stop,on_step=ready_step,
                    shared_observations=config.get('ready_shared_observations',False),
                    **(dict(movement_tracker=movement_tracker) if movement_tracker is not None else {}),
                    **(dict(reset_options=course.reset_options) if course is not None else {}),
                    **(dict(history=policy_history) if policy_history is not None else {}))
            else:
                for t in range(steps):
                    check_stop()
                    for key in observation:
                        data[key][t].copy_(torch.from_numpy(observation[key]))
                    if 'teacher_targets' in data:
                        from fly_touhou.thrl_online_reference import current_targets
                        for i in range(count):data['teacher_targets'][t,i].copy_(torch.from_numpy(current_targets(infos[i],input_schema=schema)))
                        if 'teacher_weights' in data:
                            from fly_touhou.thrl_online_reference import current_target_weight
                            for i in range(count):data['teacher_weights'][t,i]=current_target_weight(infos[i],input_schema=schema)
                    data['hidden_states'][t].copy_(hidden[0].detach().cpu())
                    data['episode_starts'][t].copy_(starts)
                    if policy_history is not None:
                        for i in range(count):
                            policy_history.append(i,{key:value[i] for key,value in observation.items()})
                    with torch.no_grad():
                        logits,values,next_hidden = forward_observation(torch,model,
                            {key:data[key][t] for key in observation},hidden)
                        distribution = Categorical(logits=policy_logits(logits,ppo.logit_clamp,
                            action_temperature=ppo.action_temperature))
                        actions = distribution.sample()
                        data['actions'][t].copy_(actions.cpu())
                        data['old_log_probs'][t].copy_(distribution.log_prob(actions).cpu())
                        data['values'][t].copy_(values.cpu())
                    before_infos=infos
                    observation,_,terminated,truncated,infos = vector.step(actions.cpu().numpy())
                    if 'movement_outcomes' in data:
                        from fly_touhou.thrl_movement_learning import transition_outcome
                        for i in range(count):
                            outcome=transition_outcome(data['features'][t,i],before_infos[i],
                                int(data['actions'][t,i]),observation['features'][i],infos[i])
                            data['movement_outcomes'][t,i]=outcome
                            movement_tracker.record(i,data['features'][t,i],before_infos[i],int(data['actions'][t,i]),
                                observation['features'][i],infos[i],outcome)
                    # A successful action acknowledgement remains paid if a later
                    # episode reset/encoding operation fails in the same loop.
                    decisions += count
                    data['rewards'][t].copy_(torch.as_tensor(np.stack([i['reward_vector'] for i in infos])))
                    reward_recent = .95*reward_recent+.05*data['rewards'][t].mean(0).numpy()
                    data['terminated'][t].copy_(torch.from_numpy(terminated))
                    data['truncated'][t].copy_(torch.from_numpy(truncated))
                    data['truncation_values'][t].zero_()
                    if np.any(truncated):
                        with torch.no_grad():
                            _,final_values,_ = forward_observation(torch,model,observation,next_hidden)
                            data['truncation_values'][t].copy_(final_values.cpu())
                    done = terminated | truncated
                    for i in np.flatnonzero(done):
                        if policy_history is not None:policy_history.reset(int(i))
                        recent = infos[i]['episode_summary']
                        games += 1
                        clears += int(recent['outcome']=='clear')
                        with (output/'training_episodes.jsonl').open('a',encoding='utf-8',buffering=1) as stream:
                            stream.write(json.dumps(dict(recent,at_decisions=decisions,worker=int(i)),
                                                    ensure_ascii=False)+'\n')
                    if done.any():
                        vector.reset_at(np.flatnonzero(done).tolist())
                        observation = vector.observation
                    hidden = next_hidden*(~torch.as_tensor(done,device='cuda'))[None,:,None]
                    starts = torch.from_numpy(done)
                    report()
            collect_elapsed = time.monotonic()-rollout_started
            total_collect += collect_elapsed
            with torch.no_grad():
                _,bootstrap,_ = forward_observation(torch,model,observation,hidden)
            phase = 'updating'
            report(True)
            learning_started = time.monotonic()
            optimizer_transaction_complete=False
            metrics = update(model,optimizer,RolloutBatch(**data,bootstrap_values=bootstrap.cpu(),
                             action_temperature=ppo.action_temperature,
                             recorded_reset_hidden=course is not None),
                             replace(ppo,update_index=inherited_updates+updates))
            torch.cuda.synchronize()
            learn_elapsed = time.monotonic()-learning_started
            total_learn += learn_elapsed
            latest_rollout_timing = {'decisions':config['rollout_total_decisions'],
                'collect_seconds':collect_elapsed,'learn_seconds':learn_elapsed,
                'sampling_decisions_per_second':config['rollout_total_decisions']/max(collect_elapsed,1e-6),
                'sampling_and_update_decisions_per_second':config['rollout_total_decisions']/max(collect_elapsed+learn_elapsed,1e-6),
                'scope':'one group; complete rollout; excludes development, checkpoint writes and initial startup'}
            if config.get('collector_mode','ready')=='ready':
                latest_rollout_timing['ready_collection_transport']=getattr(vector,'last_collection_transport',None)
            adam += metrics['optimizer_updates']
            updates += 1
            optimizer_transaction_complete=True
            # Native games stay parked. Replay only past observations under
            # CURRENT weights; this is never an old PPO optimization buffer.
            if policy_history is None:hidden.zero_()
            else:
                if course is not None:course.maintenance.poll()
                refresh_started=time.monotonic()
                hidden=policy_history.refresh(torch,model)
                refresh_elapsed=time.monotonic()-refresh_started
                total_recurrent_refresh+=refresh_elapsed
                latest_rollout_timing['recurrent_refresh_seconds']=refresh_elapsed
                latest_rollout_timing['sampling_and_update_decisions_per_second']=config['rollout_total_decisions']/max(collect_elapsed+learn_elapsed+refresh_elapsed,1e-6)
                latest_rollout_timing['scope']='one group; collection + Adam + current-parameter recurrent refresh; excludes curriculum gates, development, checkpoints and initial startup'
            if course is not None:
                phase='curriculum_gate'
                gate_started=time.monotonic()
                course.gate(torch,model,decisions+candidate_offset,check_stop,
                            on_progress=lambda row:report(True,evaluation=row))
                total_evaluation+=time.monotonic()-gate_started
                phase='updating'
            if updates % config['checkpoint_every_updates'] == 0:
                checkpoint('latest.pt')
            report(True)
            with (output/'updates.jsonl').open('a',encoding='utf-8',buffering=1) as stream:
                stream.write(json.dumps({'decisions':decisions,'updates':updates,'adam':adam,
                    'metrics':metrics,'collect_seconds':total_collect,'learn_seconds':total_learn,
                    'wall_seconds':time.monotonic()-started,'latest_rollout_timing':latest_rollout_timing})+'\n')
            if decisions in config['development_decisions']:
                if config.get('evaluation_after_training_shutdown'):
                    if decisions!=config['trial_decisions']:
                        raise ValueError('Shutdown-before-evaluation requires final-only development')
                    vector.close()
                    write_json(output/'native_cleanup.json',{'close_errors':vector.close_errors,
                        'ownership':read_shared_json(vector.ownership_path),
                        'reason':'training budget complete; release30 worlds before independent evaluation'})
                    if vector.close_errors:raise RuntimeError('Training worlds must close before development')
                    vector=None
                    infos=[]
                develop(decisions+candidate_offset)
        checkpoint('final.pt')
        phase = 'finished'
        report(True)
        result = {'status':'finished','decisions':decisions,'rollout_updates':updates,
            'candidate_decisions':decisions+candidate_offset,
            'candidate_optimized_decisions':candidate_offset+updates*config['rollout_total_decisions'],
            'candidate_paid_decisions_lower_bound':config.get('candidate_paid_decision_offset',0)+decisions,
            'additional_inflight_native_cost_unknown':config.get('unknown_inflight_decisions',False),
            'inherited_development_nodes':[] if history is None else [row['decisions'] for row in history['completed_development']],
            'adam_minibatch_steps':adam,'episodes':games,'clears':clears,
            'wall_seconds':time.monotonic()-started,'latest_development':latest_development,
            'best_development_score':best_score,'best_checkpoint':str(output/'best_development.pt'),
            'final_sha256':digest(output/'final.pt'),'formal_acceptance':False,
            'timing':{'collect_seconds':total_collect,'learn_seconds':total_learn,
                      'evaluation_seconds':total_evaluation}}
        if full_story:
            result.update(training_mode=config['training_mode'],episode_protocol=episode_protocol(config),
                native_prefixes_not_counted_as_complete=True,
                budget_tail_bootstrap_applied=True)
        write_json(output/'training_result.json',result)
        return result
    except optuna.TrialPruned:
        raise
    except StopRequested as error:
        checkpoint('interrupted.pt')
        phase = 'operator_or_budget_stop'
        report(True)
        write_json(output/'training_result.json',{'status':phase,'reason':str(error),
            'decisions':decisions,'adam_minibatch_steps':adam,
            'native_prefixes_not_counted_as_complete':True})
        raise
    except BaseException:
        checkpoint('interrupted.pt')
        phase = 'failed'
        report(True)
        write_json(output/'failure.json',{'traceback':traceback.format_exc(),'decisions':decisions,
            'adam_minibatch_steps':adam,'native_prefixes_not_counted_as_complete':True})
        raise
    finally:
        maintenance_error=None
        if course is not None:
            try:course.maintenance.close()
            except BaseException as error:maintenance_error=error
        if vector is not None:
            vector.close()
            write_json(output/'native_cleanup.json',{'close_errors':vector.close_errors,
                'ownership':read_shared_json(vector.ownership_path)})
            if vector.close_errors:
                raise RuntimeError('Training native cleanup has unresolved errors')
        if maintenance_error is not None:raise RuntimeError('Native course maintenance did not close') from maintenance_error


def validate_config(config):
    validate_training_mode(config)
    required = ('game_exe','profile','seed','workers','rollout_total_decisions','trial_decisions',
                'development_decisions','development_episodes','optimization_trials',
                'study_wall_cap_seconds','trial_wall_cap_seconds','checkpoint_every_updates')
    if any(key not in config for key in required):
        raise ValueError('Incomplete explicit THRL study configuration')
    if config['workers'] != 2 or config['rollout_total_decisions'] != 2048:
        raise ValueError('This selected backend uses two workers and2048 total decisions/rollout')
    if config['trial_decisions'] % config['rollout_total_decisions']:
        raise ValueError('No partial final PPO rollout is silently discarded')
    if (sorted(config['development_decisions']) != config['development_decisions']
            or config['development_decisions'][-1] != config['trial_decisions']
            or any(n % 2048 for n in config['development_decisions'])):
        raise ValueError('Development nodes must match complete rollout boundaries including the final budget')
    if not 1 <= config['optimization_trials'] <= 8 or not 1 <= config['development_episodes'] <= 20:
        raise ValueError('Finite Optuna/development quotas required')
    if not 0 <= config.get('validation_episodes_per_policy',20) <= 20:
        raise ValueError('Extended validation quota must be between0 and20 per policy')
    if config['seed'] != 11 or config.get('native_frame_horizon') is not None:
        raise ValueError('This study uses seed11 and no native frame horizon')
    if config['policy_action_count'] != 19:
        raise ValueError('The first THRL baseline uses source19 actions, single stationary Bomb')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args = parser.parse_args()
    config = read_shared_json(args.config)
    validate_config(config)
    import torch
    import optuna
    from fly_touhou.thrl_ppo import PPOConfig
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; no CPU training fallback')
    if digest(config['game_exe']) != config['game_exe_sha256']:
        raise ValueError('Pinned original executable identity changed')
    if digest(config['profile']) != config['profile_sha256']:
        raise ValueError('Frozen native profile identity changed')
    manifest_path = ROOT/'source_manifest.json'
    if digest(manifest_path) != config['source_manifest_sha256']:
        raise ValueError('Selected frozen source manifest changed')
    for item in read_shared_json(manifest_path)['files']:
        path = (ROOT/item['path']).resolve()
        if not path.is_relative_to(ROOT) or digest(path) != item['sha256']:
            raise ValueError('Frozen runtime file differs: '+item['path'])
    torch.set_num_threads(2)
    output = args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'study_config.json',config)
    started = time.monotonic()
    state = {'schema':'th14-thrl-optuna-study-v1','status':'starting','updated_utc':utc_now(),
        'current_run':None,'planned_trials':config['optimization_trials'],
        'attention_comparison_after_baseline':True,'completed_trials':0,'wall_seconds':0,
        'device':'cuda:0','gpu':torch.cuda.get_device_name(0),'trials':[],
        'study_name':config['study_name'],'seed':config['seed'],
        'native_frame_horizon':None,'formal_acceptance':False}
    write_json(output/'study_status.json',state)
    api = StatusAPI(output,config.get('api_port',18714))
    state['dashboard_api'] = api.url
    study = None

    def check_study_stop():
        if (output/'STOP').exists():
            raise StopRequested('operator STOP; no later trial/native game will start')
        if config.get('study_wall_cap_seconds') is not None and time.monotonic()-started >= config['study_wall_cap_seconds']:
            raise StopRequested('predeclared study wall budget reached')

    def refresh(progress=None):
        state['updated_utc'] = utc_now()
        state['wall_seconds'] = time.monotonic()-started
        if progress is not None:
            state['current_progress'] = {key:progress.get(key) for key in (
                'stage','actual_decisions','adam_minibatch_steps','episodes','latest_development')}
        if study is not None:
            state['trials'] = [{'number':t.number,'variant':'none','state':t.state.name,
                'score':t.value,'parameters':t.params,**t.user_attrs} for t in study.trials]
            state['completed_trials'] = sum(t.state.name=='COMPLETE' for t in study.trials)
        write_json(output/'study_status.json',state)

    try:
        study = optuna.create_study(study_name=config['study_name'],direction='maximize',
            storage='sqlite:///'+str(output/'optuna.sqlite3'),load_if_exists=False,
            sampler=optuna.samplers.TPESampler(seed=config['seed'],n_startup_trials=2),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=1,n_warmup_steps=2,
                                               interval_steps=1,n_min_trials=1))
        from fly_touhou.th14_completion_protocol import baseline_parameters,exploration_parameters,learning_overrides,search_space
        baseline = baseline_parameters(config)
        study.enqueue_trial(baseline,user_attrs={'role':'fixed-source-baseline'})
        refresh()

        def objective(trial):
            check_study_stop()
            parameters = asdict(PPOConfig())
            space=search_space(config)
            parameters.update(learning_rate=trial.suggest_float('learning_rate',*space['learning_rate'],log=True),
                gamma=trial.suggest_categorical('gamma',space['gamma']),
                gae_lambda=trial.suggest_categorical('gae_lambda',space['gae_lambda']),
                entropy_coeff_start=trial.suggest_categorical('entropy_coeff_start',space['entropy_coeff_start']))
            parameters.update(learning_overrides(config))
            parameters.update(exploration_parameters(config,parameters['entropy_coeff_start']))
            run = output/f'trial_{trial.number:03d}_none'
            trial.set_user_attr('run_output',str(run))
            state.update(status='training',current_run=str(run))
            refresh()
            result = train_trial(torch,config,output,run,parameters,trial=trial,
                                 study_started=started,on_progress=refresh)
            trial.set_user_attr('decisions',result['decisions'])
            trial.set_user_attr('adam_minibatch_steps',result['adam_minibatch_steps'])
            trial.set_user_attr('best_checkpoint',result['best_checkpoint'])
            trial.set_user_attr('best_stage',max(max(r['stage_history'])
                for r in result['latest_development']['episode_results']))
            return result['latest_development']['score']

        study.optimize(objective,n_trials=config['optimization_trials'],n_jobs=1,
                       callbacks=[lambda *_:refresh()],gc_after_trial=True)
        best = study.best_trial
        selected_run = Path(best.user_attrs['run_output'])
        selected_label = f'optuna_trial_{best.number}_none'
        selected_score = best.value
        if config.get('attention_comparison',True):
            check_study_stop()
            # A separate paired architecture experiment, never mixed into first baseline.
            parameters = asdict(PPOConfig())
            parameters.update(best.params)
            parameters.update(learning_overrides(config))
            parameters.update(exploration_parameters(config,parameters['entropy_coeff_start']))
            run = output/'attention_pair'
            state.update(status='attention_comparison',current_run=str(run))
            state['attention_parent_trial'] = best.number
            refresh()
            result = train_trial(torch,config,output,run,parameters,trial=None,
                study_started=started,on_progress=refresh,attention='spatial')
            state['attention_result'] = result
            if result['latest_development']['score'] > selected_score:
                selected_run,selected_label = run,'attention_pair'
                selected_score = result['latest_development']['score']
            refresh()
        selected_checkpoint = selected_run/'best_development.pt'
        selected_node = read_shared_json(selected_run/'best_development.json')
        state['selected_checkpoint'] = str(selected_checkpoint)
        state['selection_before_validation'] = {'label':selected_label,
            'trial_and_architecture_selection_final_node_score':selected_score,
            'checkpoint_development_decisions':selected_node['decisions'],
            'checkpoint_development_score':selected_node['score'],
            'checkpoint_sha256':digest(selected_checkpoint)}
        refresh()
        validation = {}
        if config.get('validation_episodes_per_policy',20):
            from fly_touhou.thrl_model import MOActorCritic
            # Selection is frozen before these independent extended comparisons.
            for kind,checkpoint in (('selected',selected_checkpoint),('same_initial',selected_run/'initial.pt')):
                check_study_stop()
                payload = torch.load(checkpoint,map_location='cpu',weights_only=False)
                model = MOActorCritic(**payload['model_config']).cuda()
                model.load_state_dict(payload['model'],strict=True)
                del payload
                run = output/f'validation20_{kind}'
                state.update(status='validation20',current_run=str(run))
                refresh()
                original_progress = read_shared_json(selected_run/'progress.json')

                def validation_progress(row):
                    progress = {**original_progress,'stage':'validation20','status':'validation20',
                                'evaluation_progress':row,'updated_utc':utc_now()}
                    write_json(run/'progress.json',progress)
                    refresh(progress)

                validation[kind] = evaluate(torch,model,config,run,
                    episodes=config.get('validation_episodes_per_policy',20),
                    seed=config['seed']+900000,check_stop=check_study_stop,progress=validation_progress,
                    **({'native_split':'test'} if config.get('curriculum_manifest') or is_full_story(config) else {}))
                state['validation20'] = validation
                refresh()
                del model
        state.update(status='finished',current_run=None)
        refresh()
        write_json(output/'study_result.json',{'status':'finished','best_trial':best.number,
            'best_value':best.value,'best_parameters':best.params,'trials':state['trials'],
            'attention_result':state.get('attention_result'),
            'selected_checkpoint':str(selected_checkpoint),'selection_before_validation':state['selection_before_validation'],
            'extended_validation':validation,'formal_acceptance':False,
            'next_step':'Review independent20-game comparisons before additional seeds or fly training.'})
    except StopRequested as error:
        state.update(status='operator_or_budget_stop',reason=str(error))
        refresh()
    except BaseException:
        state.update(status='failed')
        refresh()
        write_json(output/'failure.json',{'traceback':traceback.format_exc(),'current_run':state['current_run']})
        raise
    finally:
        api.close()


if __name__ == '__main__':
    main()
