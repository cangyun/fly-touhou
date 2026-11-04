"""Continue the verified complete-connectome SNN toward original-game clear.

Student worlds persist across PPO cycles and frozen development.
All episodes begin at the opening and end only at native Game Over/clear.
"""
from pathlib import Path
import argparse,hashlib,json,os,sys,time,traceback,gc
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]


def register_tensorboard_run(output):
    """Point the existing live exporter at this continuation, preserving history."""
    from fly_touhou.thrl_study_io import read_shared_json
    from fly_touhou.thrl_durable_records import write_json
    config_path=ROOT/'artifacts/engineering/tensorboard-live-v3/configuration.json'
    if not config_path.exists():return dict(status='monitor_not_configured')
    try:
        config=read_shared_json(config_path)
        sources=config.setdefault('fly_algorithm_sources',{})
        previous_label=config.get('compact_story_active_label','native_batch16_long')
        previous=sources.pop(previous_label,None)
        if previous and Path(previous)!=Path(output):
            config.setdefault('historical_fly_algorithm_sources',{})[Path(previous).name]=previous
        label=Path(output).name.replace('-','_')
        sources[label]=str(output)
        config['compact_story_active_label']=label
        write_json(config_path,config)
        return dict(status='registered',source=str(output),label=label,previous_source=previous)
    except (OSError,ValueError,TypeError) as problem:
        return dict(status='registration_failed',error=repr(problem))


def native_teacher_options(args, restored):
    """Resume the teacher replay unless an explicit CLI value overrides it."""
    saved=restored or {}
    manifest=args.native_rule_teacher_manifest
    if manifest is None:manifest=saved.get('native_rule_teacher_manifest')
    coefficient=(args.native_rule_teacher_coefficient if args.native_rule_teacher_coefficient is not None
                 else saved.get('native_rule_teacher_coefficient',0.))
    every=(args.native_rule_teacher_every if args.native_rule_teacher_every is not None
           else saved.get('native_rule_teacher_every',8))
    weights=(args.native_rule_teacher_stage_weights if args.native_rule_teacher_stage_weights is not None
             else saved.get('native_rule_teacher_stage_weights'))
    windows=getattr(args,'native_rule_teacher_windows',None)
    if windows is None:windows=saved.get('native_rule_teacher_windows',1)
    if not isinstance(windows,int) or isinstance(windows,bool) or not 1<=windows<=7:
        raise ValueError('One to seven native teacher windows per update required')
    if not 0<=coefficient<=1 or every<1:
        raise ValueError('Native rule-teacher coefficient must be in [0,1] and its interval positive')
    if coefficient and not manifest:
        raise ValueError('Current-sensor teacher supervision requires its manifest')
    if args.native_rule_teacher_stage_weights is not None and not coefficient:
        raise ValueError('Stage weights require enabled current-sensor teacher supervision')
    return dict(manifest=Path(manifest) if manifest else None,coefficient=coefficient,
                every=every,stage_weights=weights if coefficient else None,windows=windows)


def native_teacher_checkpoint_fields(options):
    return dict(native_rule_teacher_manifest=str(options['manifest']) if options['manifest'] else None,
        native_rule_teacher_coefficient=options['coefficient'],native_rule_teacher_every=options['every'],
        native_rule_teacher_stage_weights=options['stage_weights'],native_rule_teacher_windows=options['windows'])


def native_teacher_stage_weight_map(manifest,weights):
    """Map weights to the actual training stages in ascending stage order."""
    if weights is None:return None
    document=json.loads(Path(manifest).read_text(encoding='utf-8-sig'))
    stages=sorted({int(row['stage']) for row in document['windows'] if row['split']=='train'})
    if len(weights)!=len(stages):
        raise ValueError(f'Teacher weights must cover training stages {stages} in ascending order')
    return dict(zip(stages,weights))


def check_other_native_games(psutil_module, game_name, owned_controllers):
    """Allow only original games descending from explicitly identified jobs."""
    allowed=set()
    for pid_text,created_text in owned_controllers:
        try:
            process=psutil_module.Process(int(pid_text))
            if abs(process.create_time()-float(created_text))>.01:
                continue
            allowed.update(child.pid for child in process.children(recursive=True))
        except psutil_module.NoSuchProcess:
            continue
    for process in psutil_module.process_iter(['pid','name']):
        if (process.info['name'] or '').lower()==game_name.lower() and process.info['pid'] not in allowed:
            raise RuntimeError('Another original game is active outside the allowed owned evaluators')
    return allowed


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('checkpoint','connectome','environment-config','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--resume-training',type=Path)
    p.add_argument('--allow-owned-game-controller',nargs=2,action='append',default=[],
        metavar=('PID','CREATED'),help='Allow original games descending from this exact owned evaluator')
    p.add_argument('--fused-edge-grad',action='store_true')
    p.add_argument('--reuse-kernel-cache-from',type=Path,
        help='Reuse an earlier job CUDA build cache when the native kernels are unchanged')
    p.add_argument('--deterministic-spmm',action='store_true')
    p.add_argument('--unpad-candidates',action='store_true')
    p.add_argument('--batch-invariant-sensory',action='store_true')
    p.add_argument('--compress-tapes',action='store_true')
    p.add_argument('--secondary-tape-directory',type=Path,
        help='Store even-cycle archives under this directory/run-name;all path receipts stay in the main output')
    p.add_argument('--training-checkpoint-directory',type=Path,
        help='Store resumable model+Adam checkpoints here;path receipts stay in the main output')
    p.add_argument('--parameter-checkpoint-every',type=int,default=1,
        help='Keep numbered parameters at this cycle interval;always keep development/final nodes and rolling latest')
    p.add_argument('--critical-teacher-replay',action='store_true')
    p.add_argument('--balanced-critical-teacher-replay',action='store_true',
        help='Pair actual stocked non-Bomb teacher states with rare deathbomb history; same guidance coefficients')
    p.add_argument('--critical-event-port-amplitude',type=float,default=0.)
    p.add_argument('--critical-replay-source',type=Path,action='append',default=[])
    p.add_argument('--late-stage-replay-source',type=Path,action='append',default=[],
        help='Student training rollout for compatible stage3/4+ supervised rehearsal;never a PPO source')
    p.add_argument('--late-stage-replay-bank',type=Path,action='append',default=[],
        help='Saved matching-interface student-controlled stage3/4+ supervised windows')
    p.add_argument('--late-stage-replay-coefficient',type=float,
        help='Optional late-stage teacher and selected-contact risk rehearsal;inherit on resume')
    p.add_argument('--rule-teacher-archive-manifest',type=Path,
        help='Partial V5 ResourceV10 teacher windows;supervised movement only,never PPO')
    p.add_argument('--rule-teacher-archive-coefficient',type=float,default=0.,
        help='Auxiliary weight for archived rule-teacher movement preferences;default0')
    p.add_argument('--rule-teacher-archive-every',type=int,default=8,
        help='Apply one archived teacher window every N PPO updates')
    p.add_argument('--native-rule-teacher-manifest',type=Path,
        help='Current-sensor ResourceV10 full-game windows;supervision only')
    p.add_argument('--native-rule-teacher-coefficient',type=float,
        help='Auxiliary teacher weight;inherit on resume, otherwise default0;set0 to disable')
    p.add_argument('--native-rule-teacher-every',type=int,
        help='Apply teacher supervision every N PPO updates;inherit on resume, otherwise default8')
    p.add_argument('--native-rule-teacher-windows',type=int,
        help='Pack1..7 independent teacher histories per supervised update;inherit on resume, otherwise default1')
    p.add_argument('--native-rule-teacher-stage-weights',type=float,nargs='+',metavar='WEIGHT',
        help='Sampling weights in ascending order of training stages present in the teacher manifest')
    p.add_argument('--bomb-dagger-warmup-steps',type=int,default=0,
        help='Before PPO, collect this many student deployment decisions per persistent world for Bomb teacher correction')
    p.add_argument('--bomb-dagger-anchor-cache',type=Path,
        help='Original TRAIN DN cache from the same brain, supplying rare real deathbomb positives')
    p.add_argument('--bomb-dagger-readout-updates',type=int,default=128)
    p.add_argument('--policy-lr-scale',type=float)
    p.add_argument('--context-policy-lr-multiplier',type=float,
        help='Relative LR for the context sensory ports;omit to inherit optimizer state or use1')
    p.add_argument('--policy-group-lr-multipliers',type=json.loads,
        help='JSON mapping of policy group names to relative rates;keeps Adam history and the shared KL limit')
    p.add_argument('--minimum-policy-lr-scale',type=float,
        help='Minimum adaptive policy LR multiplier;default.05,or inherit from resumed optimizer')
    p.add_argument('--boss-attack-guidance-coefficient',type=float,
        help='Optional teacher closing-direction loss only when all three hazard horizons are safe;inherit on resume')
    p.add_argument('--selective-attack-guidance-coefficient',type=float,
        help='Optional teacher-supported approach for other trusted4/12-frame Boss scenes;inherit on resume')
    p.add_argument('--conditional-wall-guidance-coefficient',type=float,
        help='Weak movement-only edge recovery preference where the native teacher abstains;inherit on resume')
    p.add_argument('--guided-wall-guidance-coefficient',type=float,
        help='Optional teacher-supported edge recovery loss;default0 or inherit on resume')
    p.add_argument('--adaptive-policy-lr',action=argparse.BooleanOptionalAction,default=False)
    p.add_argument('--kl-early-stop',action=argparse.BooleanOptionalAction,default=False,
        help='Optional KL stopping;disabled by default so selected rows finish their full epoch')
    p.add_argument('--target-kl',type=float,default=.03,
        help='Used only when KL stopping or KL-adaptive learning rates are explicitly enabled')
    p.add_argument('--world-aligned-updates',action='store_true')
    p.add_argument('--checkpoint-candidate-chunks',action='store_true')
    p.add_argument('--phase-valid-control',action='store_true')
    p.add_argument('--risk-protection-transition',action='store_true')
    p.add_argument('--forecast-native-arming',action='store_true')
    p.add_argument('--forecast-growth-motion',action='store_true',
        help='Use the native-grounded state2 half/full motion envelope in existing sensory hints and labels')
    p.add_argument('--native-motion-forecast',action='store_true',
        help='Use native composed/clamped enemy movement, contact schedules and bullet release programs')
    p.add_argument('--task-progress-ports',action='store_true',
        help='Add action-conditioned task currents on existingVP escape/wall populations;inherit when resuming')
    p.add_argument('--task-progress-amplitude',type=float,default=.025)
    p.add_argument('--native-focus-gate',action='store_true',
        help='Read original Shift availability;predict fast movement when native focus is unavailable')
    p.add_argument('--unknown-laser-player-motion',action='store_true',
        help='Use the proposed player path with current unknown-motion laser geometry as an unverified hint')
    p.add_argument('--native-curve-laser-geometry',action='store_true',
        help='Read native curve-node collision rectangles;future motion remains explicitly unknown')
    p.add_argument('--protected-exit-forecast',action='store_true',
        help='Opt in to conditional post-protection sensory hints, exit reference and exposed-horizon calibration;inherit on resume')
    p.add_argument('--primary-attack-target',action='store_true')
    p.add_argument('--risk-policy-gradient',action=argparse.BooleanOptionalAction,default=None,
        help='PPO gradient through risk scores;--no-risk-policy-gradient leaves risk calibration trainable. Inherit when omitted.')
    p.add_argument('--native-contact-risk-calibration',action=argparse.BooleanOptionalAction,default=None,
        help='Use measured selected-action TRAIN contacts for25%% of the existing4f risk BCE;inherit checkpoint when omitted')
    p.add_argument('--soft-risk-composition',action='store_true')
    p.add_argument('--risk-input-soft-clip',type=float)
    p.add_argument('--risk-base-input-soft-clip',type=float)
    p.add_argument('--neural-risk-scale',type=float,
        help='Finite neural movement-risk preference;preserve Bomb probability. Inherit checkpoint,else1.')
    p.add_argument('--idle-prior-scale',type=float,
        help='Scale fast-idle preference[0,1],preserve Bomb probability;inherit checkpoint by default')
    p.add_argument('--movement-sampling-temperature',type=float,
        help='Training movement exploration only;preserve Bomb probability and frozen decoder. Inherit on resume,else1.')
    p.add_argument('--bomb-sampling-temperature',type=float,
        help='Training Bomb-family exploration only;frozen decoder unchanged. Inherit on resume,else1.')
    p.add_argument('--trainable-risk-baseline',action='store_true')
    p.add_argument('--cycles',type=int,default=2);p.add_argument('--workers',type=int,default=16)
    p.add_argument('--games-per-world',type=int,default=0,
        help='Finish after every persistent world completes this many natural games;cycles stays a hard cap. No stage resets.')
    p.add_argument('--cycle-offset',type=int,default=0,
        help='Completed predecessor cycles;continue file and monitoring cycle numbers from here')
    p.add_argument('--steps',type=int,default=512);p.add_argument('--micro-scenes',type=int,default=8)
    p.add_argument('--ready-driven',action='store_true',
        help='First-ready scheduling with a global workers*steps decision budget;retain original worlds across updates')
    p.add_argument('--ready-batch-size',type=int,default=7)
    p.add_argument('--ready-wait-ms',type=float,default=1.)
    p.add_argument('--ready-per-world-quota',action='store_true',
        help='Ready-first dispatch with an equal long per-world quota before each policy update')
    p.add_argument('--ppo-sample-decisions',type=int,default=0,
        help='Select this many real rows as intact16-step sequences from each rollout;0 uses all')
    p.add_argument('--ppo-stage-weights',type=float,nargs=6,default=(.125,.125,.30,.30,.075,.075))
    p.add_argument('--keep-rollout-tapes',type=int,default=0,
        help='Keep latest N full tapes after checkpointing;archive all live supervision separately;0 keeps all')
    p.add_argument('--max-adam-updates',type=int,default=0,
        help='Per-segment compute quota,after shuffling whole recurrent sequences;0 keeps legacy unlimited epochs')
    p.add_argument('--student-decision-budget',type=int,default=0,
        help='Fixed ready-driven job budget,independent of hot collection-length changes')
    p.add_argument('--evaluation-every-decisions',type=int,default=0)
    p.add_argument('--first-evaluation-decisions',type=int,default=0,
        help='Remaining decisions until the next frozen node when resuming a partial experiment')
    p.add_argument('--parameter-checkpoint-every-decisions',type=int,default=0)
    p.add_argument('--lr-growth-reference-decisions',type=int,default=0,
        help='Keep low-KL learning-rate growth proportional to collected decisions when shortening segments')
    p.add_argument('--epochs',type=int,default=1);p.add_argument('--evaluation-episodes',type=int,default=4)
    p.add_argument('--evaluation-every',type=int,default=1)
    p.add_argument('--evaluation-selection',choices=('joint_mode','family_mode','sample'),default='joint_mode',
        help='Frozen development decoder only;PPO collection continues sampling its original joint19 policy')
    p.add_argument('--reference-development-report',type=Path)
    p.add_argument('--initial-decision-budget',type=int,default=0,
        help='Count warmup and PPO decisions;further cycles require frozen benefit against --budget-benefit-reference')
    p.add_argument('--budget-benefit-reference',type=Path,
        help='External frozen development evidence for a budget extension;never labelled as initial model performance')
    p.add_argument('--stop-after-flat-evaluations',type=int,default=0,
        help='Stop this finite job after this many frozen nodes fail to improve its best native evidence;0 disables')
    p.add_argument('--stop-at-development-clears',type=int,default=0,
        help='Finish this learner at a qualifying frozen development node for independent validation;0 disables')
    p.add_argument('--wall-hours',type=float,default=8.)
    p.add_argument('--seed',type=int,default=202610329);p.add_argument('--execute',action='store_true');a=p.parse_args()
    student_budget=a.student_decision_budget or (a.cycles*a.steps+a.bomb_dagger_warmup_steps)*a.workers
    if not a.execute:print(dict(status='prepared',cycles=a.cycles,
        target_games_per_world=a.games_per_world,cycles_are_hard_cap=True,
        PPO_student_decisions=a.cycles*a.workers*a.steps,
        teacher_only_student_decisions=a.bomb_dagger_warmup_steps*a.workers,
        student_decisions=student_budget,max_adam_updates=a.max_adam_updates));return
    ready_schedule=None
    if a.student_decision_budget or a.evaluation_every_decisions:
        if not a.ready_driven:raise ValueError('Decision-based live scheduling requires ready-driven collection')
        from fly_touhou.compact_story_schedule import ReadySchedule
        ready_schedule=ReadySchedule(workers=a.workers,segment_decisions=a.workers*a.steps,
            max_updates=a.max_adam_updates,decision_budget=student_budget,
            evaluation_every=a.evaluation_every_decisions or a.evaluation_every*a.workers*a.steps,
            first_evaluation=a.first_evaluation_decisions,
            checkpoint_every=a.parameter_checkpoint_every_decisions or a.parameter_checkpoint_every*a.workers*a.steps)
    if a.games_per_world<0:raise ValueError('games-per-world must be nonnegative')
    if a.ppo_sample_decisions<0 or a.ppo_sample_decisions%256:
        raise ValueError('PPO sample budget must be nonnegative whole256 rows')
    if a.ready_per_world_quota and not a.ready_driven:raise ValueError('Per-world ready quotas require ready collection')
    if a.keep_rollout_tapes<0:raise ValueError('Nonnegative tape retention required')
    if any(not 0<=v<float('inf') for v in a.ppo_stage_weights) or sum(a.ppo_stage_weights)<=0:
        raise ValueError('Stage sampling weights must be finite,nonnegative,and have positive total')
    if a.lr_growth_reference_decisions<0:raise ValueError('Nonnegative LR decision reference required')
    if a.workers+a.evaluation_episodes>30 or a.steps%16 or 16%a.micro_scenes or a.evaluation_every<1:raise ValueError('Owned<=30 worlds,16-frame BPTT and divisibility required')
    if a.ready_driven and (a.steps*a.workers%256 or not 1<=a.ready_batch_size<=7 or not 0<=a.ready_wait_ms<=2):
        raise ValueError('Ready collection requires full256-row global batches and bounded coalescing')
    if a.world_aligned_updates and not a.ready_driven:
        from fly_touhou.compact_story_sequence_groups import world_aligned_groups
        import numpy as np
        if a.micro_scenes!=16:raise ValueError('World-aligned learning uses16 real sequences per microbatch')
        world_aligned_groups(a.steps,a.workers,16,np.random.default_rng(0))
    if a.risk_input_soft_clip is not None and not 1<=a.risk_input_soft_clip<=1000:raise ValueError('Risk soft limit[1,1000] required')
    if a.risk_base_input_soft_clip is not None and not 1<=a.risk_base_input_soft_clip<=1000:raise ValueError('Risk baseline soft limit[1,1000] required')
    if a.stop_after_flat_evaluations<0:raise ValueError('Nonnegative native gate patience required')
    if not 0<=a.rule_teacher_archive_coefficient<=1 or a.rule_teacher_archive_every<1:
        raise ValueError('Teacher archive coefficient must be in [0,1] and its interval positive')
    if a.rule_teacher_archive_coefficient and a.rule_teacher_archive_manifest is None:
        raise ValueError('Teacher archive supervision requires a converted manifest')
    if a.native_rule_teacher_coefficient is not None and not 0<=a.native_rule_teacher_coefficient<=1:
        raise ValueError('Native rule-teacher coefficient must be in [0,1]')
    if a.native_rule_teacher_every is not None and a.native_rule_teacher_every<1:
        raise ValueError('Native rule-teacher interval must be positive')
    if a.balanced_critical_teacher_replay and not a.critical_teacher_replay:raise ValueError('Balanced replay requires --critical-teacher-replay')
    if a.parameter_checkpoint_every<1:raise ValueError('Positive parameter checkpoint interval required')
    if a.bomb_dagger_warmup_steps<0 or a.bomb_dagger_warmup_steps%16:
        raise ValueError('Nonnegative whole16-step teacher warmup required')
    if a.bomb_dagger_warmup_steps and (not a.bomb_dagger_anchor_cache or a.evaluation_selection=='sample'):
        raise ValueError('Teacher warmup requires an original training anchor cache and deterministic deployment decoder')
    if not 1<=a.bomb_dagger_readout_updates<=512:raise ValueError('Readout calibration budget1..512 required')
    if a.initial_decision_budget<0 or bool(a.initial_decision_budget)!=bool(a.budget_benefit_reference):
        raise ValueError('Positive initial decision budget and external native reference must be paired')
    first_evaluation=(ready_schedule.next_evaluation if ready_schedule else
        (a.bomb_dagger_warmup_steps+min(a.evaluation_every,a.cycles)*a.steps)*a.workers)
    if a.initial_decision_budget and min(first_evaluation,student_budget)>a.initial_decision_budget:
        raise ValueError('Initial allowance must cover teacher warmup and its first scheduled frozen evaluation')
    import psutil
    from fly_touhou.thrl_study_io import read_shared_json as read
    from fly_touhou.thrl_durable_records import write_json,append_jsonl
    from fly_touhou.owned_worker_affinity import worker_cpu_pool
    cfg=read(a.environment_config);cfg['seed']=a.seed
    check_other_native_games(psutil,Path(cfg['game_exe']).name,a.allow_owned_game_controller)
    current=psutil.Process();available_cpus=current.cpu_affinity()
    cpus=worker_cpu_pool(available_cpus);current.cpu_affinity(available_cpus[-2:])
    current.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    from fly_touhou.runtime_cache_paths import configure_generated_caches,generated_kernel_cache_base
    configure_generated_caches()
    kernel_cache_base=generated_kernel_cache_base(a.reuse_kernel_cache_from or a.output)
    os.environ.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',FLY_TOUHOU_DESKTOP_NAME='fly-touhou-job-story-PPO-'+str(current.pid))
    if a.fused_edge_grad:
        os.environ['FLY_TOUHOU_FUSED_EDGE_GRAD']='1'
        os.environ['FLY_TOUHOU_EDGE_GRAD_CACHE']=str(kernel_cache_base/'kernel-cache')
    if a.deterministic_spmm:
        os.environ['FLY_TOUHOU_DETERMINISTIC_SPMM']='1'
        os.environ['FLY_TOUHOU_SPMM_CACHE']=str(kernel_cache_base/'spmm-kernel-cache')
    a.output.mkdir(parents=True,exist_ok=False);begun=time.monotonic();updates=decisions=0;collector=None;native_temp=None;native_closed=False;action_origin=update_origin=0
    trained_rows=0
    secondary=None
    training_directory=a.output
    if a.training_checkpoint_directory:
        training_directory=(a.training_checkpoint_directory/a.output.name).resolve()
        if training_directory==a.output.resolve():raise ValueError('Training checkpoint directory must differ from main output')
        training_directory.mkdir(parents=True,exist_ok=False)
    if a.secondary_tape_directory:
        secondary=(a.secondary_tape_directory/a.output.name).resolve()
        if secondary==a.output.resolve():raise ValueError('Secondary tape directory must differ from the main output')
        secondary.mkdir(parents=True,exist_ok=False)
    launch=dict(schema='candidate-SNN-full-story-PPO-v1',PID=current.pid,create_time=current.create_time(),
        started_UTC=datetime.now(timezone.utc).isoformat(),argv=sys.argv,source_checkpoint=str(a.checkpoint),
        observation_features=129,neural_input_per_candidate=47,full_story=True,CUDA_started=False,
        worlds=a.workers,development_worlds=a.evaluation_episodes,teacher_actions=0,
        CPU_affinity_before_narrowing=available_cpus,learner_cpu_affinity=current.cpu_affinity(),worker_cpu_pool=cpus,
        resume_training=str(a.resume_training) if a.resume_training else None,fused_edge_gradient=a.fused_edge_grad,
        unpad_candidates=a.unpad_candidates,batch_invariant_sensory=a.batch_invariant_sensory,
        evaluation_every_cycles=a.evaluation_every,
        development_action_selection=a.evaluation_selection)
    launch['collection']=dict(mode=('ready-per-world-long-batch-v1' if a.ready_per_world_quota else 'ready-global-blocks-v1') if a.ready_driven else 'synchronous-vector',
        segment_decisions=a.workers*a.steps,per_world_step_quota=a.ready_per_world_quota or not a.ready_driven,
        ready_batch_size=a.ready_batch_size if a.ready_driven else None,
        ready_wait_ms=a.ready_wait_ms if a.ready_driven else None,
        lr_growth_reference_decisions=a.lr_growth_reference_decisions or a.workers*a.steps,
        persistent_native_worlds=True,policy_version_barrier=True)
    launch['collection'].update(max_adam_updates=a.max_adam_updates,
        fixed_student_decision_budget=student_budget,
        evaluation_every_decisions=ready_schedule.evaluation_every if ready_schedule else None,
        first_evaluation_decisions=ready_schedule.next_evaluation if ready_schedule else None,
        runtime_collection_control='collection-control.json' if ready_schedule else None)
    launch['PPO_sample_selection']=dict(rows=a.ppo_sample_decisions or None,
        stage_weights=dict(enumerate(a.ppo_stage_weights,1)),intact_sequences=True,
        GAE_uses_full_collected_histories=True,old_teacher_data_is_not_PPO=True)
    launch['lossless_compressed_tapes']=a.compress_tapes
    launch['critical_teacher_replay']=a.critical_teacher_replay
    launch['balanced_critical_teacher_replay']=a.balanced_critical_teacher_replay
    launch['Bomb_DAgger_warmup']=dict(steps_per_world=a.bomb_dagger_warmup_steps,
        new_student_decisions=a.bomb_dagger_warmup_steps*a.workers,PPO_samples=0,
        readout_updates=a.bomb_dagger_readout_updates,student_decoder=a.evaluation_selection,
        anchor_cache=str(a.bomb_dagger_anchor_cache) if a.bomb_dagger_anchor_cache else None)
    launch['world_aligned_updates']=a.world_aligned_updates
    launch['checkpoint_candidate_chunks']=a.checkpoint_candidate_chunks
    launch['policy_lr_scale']=a.policy_lr_scale;launch['adaptive_policy_lr']=a.adaptive_policy_lr
    launch['secondary_tape_directory']=str(secondary) if secondary else None
    launch['training_checkpoint_directory']=str(training_directory)
    launch['parameter_checkpoint_every']=a.parameter_checkpoint_every
    launch['completed_predecessor_cycles']=a.cycle_offset
    launch['reuse_kernel_cache_from']=str(a.reuse_kernel_cache_from) if a.reuse_kernel_cache_from else None
    launch['conditional_budget']=dict(initial_decisions=a.initial_decision_budget or None,
        native_benefit_reference=str(a.budget_benefit_reference) if a.budget_benefit_reference else None,
        maximum_decisions=student_budget)
    write_json(a.output/'launch.json',launch);write_json(a.output/'environment-config.json',cfg)
    write_json(a.output/'tensorboard-registration.json',register_tensorboard_run(a.output))
    sources=[Path(__file__),*(ROOT/'src/fly_touhou'/n for n in ('native.py','compact_story_model.py','compact_story_ppo.py',
        'compact_story_ready.py','compact_story_ready_data.py','compact_story_schedule.py','compact_story_coverage.py',
        'compact_story_shared_checkpoint.py',
        'compact_story_batch_selection.py',
        'compact_story_tape_retention.py',
        'compact_story_sensation.py','compact_full_story.py','compact_story_diagnostics.py','compact_story_storage.py',
        'compact_story_critical_replay.py','compact_story_bomb_dagger.py','compact_story_bomb_calibration.py',
        'compact_teacher_support.py','compact_protection_forecast.py','compact_protected_exit_candidate.py','compact_native_contact_targets.py',
        'compact_selective_attack_guidance.py','compact_late_stage_replay.py','compact_rule_teacher_replay.py','compact_native_rule_replay.py',
        'compact_story_learning_rate.py','compact_story_runtime_control.py','runtime_cache_paths.py','compact_story_behavior.py','compact_story_sequence_groups.py',
        'compact_story_action_selection.py','compact_story_contact_evidence.py','compact_spiking_sensation.py','fly_gru_aligned_ppo.py',
        'compact_native_budget.py','compact_story_risk_composition.py','compact_growth_forecast.py','dynamic_planner.py','compact_story_idle_prior.py',
        'compact_native_forecast.py','native_enemy_motion.py','native_enemy_damage.py','native_enemy_ecl.py','native_enemy_contact.py','native_bullet_release.py','native_bullet_wrap.py','native_bullet_pending.py','resources.py','hazard_reader.py',
        'thrl_geometry_observation.py','thrl_movement_teacher.py','compact_story_task_drive.py','native_focus_gate.py',
        'compact_boss_attack_guidance.py','compact_guided_wall_candidate.py',
        'native_curve_laser_reader.py','native_curve_laser_contact.py','native_curve_laser_paths.py',
        'native_curve_laser_motion.py','native_curve_motion_paths.py',
        'fly_gru_aligned_guidance.py','th14_completion_v3.py','th14_completion_protocol.py','continuous_sparse.py',
        'fused_edge_gradient.py','csrc/fused_edge_gradient.cu','deterministic_spmm.py','csrc/deterministic_csr_spmm.cu',
        'candidate_spiking_model.py','compact_spiking_model.py','spiking_engineered_control.py','dn_residual_risk.py'))]
    pinned={n:n.read_bytes() for n in sources}
    write_json(a.output/'source-pins.json',{str(n):hashlib.sha256(data).hexdigest() for n,data in pinned.items()})
    for n,data in pinned.items():
        snapshot=a.output/'source-snapshot'/n.relative_to(ROOT)
        snapshot.parent.mkdir(parents=True,exist_ok=True);snapshot.write_bytes(data)
    def guard():
        if (a.output/'STOP').exists():raise RuntimeError('Operator STOP')
        if time.monotonic()-begun>a.wall_hours*3600:raise TimeoutError('Finite full-story training wall budget')
    completed_cycle=a.cycle_offset
    def progress(phase,**data):
        observed=action_origin+collector.actions if collector is not None else decisions
        write_json(a.output/'progress.json',dict(phase=phase,joint_updates=updates,PPO_updates=updates,
            new_student_actions=max(decisions,observed),student_actions_this_job=max(decisions,observed)-action_origin,
            PPO_updates_this_job=updates-update_origin,completed_cycle=completed_cycle,
            PPO_trained_rows_this_job=trained_rows,
            collection_segment_decisions=ready_schedule.segment_decisions if ready_schedule else a.workers*a.steps,
            max_adam_updates_per_segment=ready_schedule.max_updates if ready_schedule else a.max_adam_updates,
            next_frozen_evaluation_decisions=ready_schedule.next_evaluation if ready_schedule else None,
            elapsed_seconds=time.monotonic()-begun,**data))
    try:
        import torch
        from fly_touhou.compact_story_model import from_avoidance_checkpoint,from_story_checkpoint,SCHEMA
        from fly_touhou.compact_story_ppo import Collector,optimizer_for,restore_optimizer,update
        from fly_touhou.compact_full_story import evaluate,apply_sensor_patches,native_wall_revision,native_scale_revision
        from fly_touhou.compact_native_forecast import native_repair_revisions
        from fly_touhou.compact_story_storage import save_rollout
        from fly_touhou.thrl_job_paths import configure_native_temp
        torch.set_num_threads(2);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        native_temp=configure_native_temp(a.output);progress('loading_warm_started_full_story_SNN')
        restored=torch.load(a.resume_training,map_location='cpu',weights_only=False) if a.resume_training else None
        current_revisions=native_repair_revisions(reactive_orbs=native_wall_revision(cfg),bullet_scale=native_scale_revision(cfg))
        launch['native_repair_revisions']=current_revisions
        native_teacher=native_teacher_options(a,restored)
        attack_coefficient=(a.boss_attack_guidance_coefficient if a.boss_attack_guidance_coefficient is not None
            else (restored or {}).get('boss_attack_guidance_coefficient',0.))
        if not 0<=attack_coefficient<=1:raise ValueError('Boss attack guidance coefficient must be in[0,1]')
        wall_coefficient=(a.conditional_wall_guidance_coefficient if a.conditional_wall_guidance_coefficient is not None
            else cfg.get('compact_conditional_wall_guidance_coefficient',
                (restored or {}).get('conditional_wall_guidance_coefficient',0.)))
        if not 0<=wall_coefficient<=1:raise ValueError('Conditional wall guidance coefficient must be in[0,1]')
        guided_wall_coefficient=(a.guided_wall_guidance_coefficient if a.guided_wall_guidance_coefficient is not None
            else cfg.get('compact_guided_wall_guidance_coefficient',
                (restored or {}).get('guided_wall_guidance_coefficient',0.)))
        if not 0<=guided_wall_coefficient<=1:raise ValueError('Guided wall guidance coefficient must be in[0,1]')
        selective_attack_coefficient=(a.selective_attack_guidance_coefficient if a.selective_attack_guidance_coefficient is not None
            else cfg.get('compact_selective_attack_guidance_coefficient',
                (restored or {}).get('selective_attack_guidance_coefficient',0.)))
        if not 0<=selective_attack_coefficient<=1:raise ValueError('Selective attack guidance coefficient must be in[0,1]')
        late_coefficient=(a.late_stage_replay_coefficient if a.late_stage_replay_coefficient is not None
            else (restored or {}).get('late_stage_replay_coefficient',0.))
        if not 0<=late_coefficient<=1:raise ValueError('Late-stage replay coefficient must be in[0,1]')
        # Old geometry labels expire; observed stocked deathbomb facts do not.
        late_saved=(restored or {}) if (restored or {}).get('native_repair_revisions')==current_revisions else {}
        replay_archive={};student_replay_archive=(restored or {}).get('student_replay_archive')
        if student_replay_archive:
            from fly_touhou.compact_story_tape_retention import load_replays
            replay_archive=load_replays(student_replay_archive,current_revisions)
            if replay_archive.get('geometry_revision_migration'):
                write_json(a.output/'student-replay-migration.json',replay_archive['geometry_revision_migration'])
        late_sources=[] if replay_archive else [Path(path) for path in late_saved.get('late_stage_replay_sources',[])]
        late_sources.extend(a.late_stage_replay_source)
        late_bank_paths=[Path(path) for path in late_saved.get('late_stage_replay_banks',[])]
        late_bank_paths.extend(a.late_stage_replay_bank)
        late_bank_paths=list(dict.fromkeys(path.resolve() for path in late_bank_paths))
        from fly_touhou.compact_story_behavior import sampling_temperature
        movement_temperature=sampling_temperature(a.movement_sampling_temperature if a.movement_sampling_temperature is not None
            else (restored or {}).get('movement_sampling_temperature',1.))
        launch['movement_sampling_temperature']=movement_temperature
        bomb_temperature=sampling_temperature(a.bomb_sampling_temperature if a.bomb_sampling_temperature is not None
            else (restored or {}).get('bomb_sampling_temperature',1.))
        launch['bomb_sampling_temperature']=bomb_temperature
        point=restored['model_payload'] if restored else torch.load(a.checkpoint,map_location='cpu',weights_only=False)
        inherited_costs=point.get('costs',{})
        model=(from_story_checkpoint if point['schema']==SCHEMA else from_avoidance_checkpoint)(a.connectome,point);del point
        if a.unpad_candidates:model.pad_candidate_batch=False
        if a.batch_invariant_sensory:model.core.input_encoder.batch_invariant=True
        if a.deterministic_spmm:model.core.sparse.deterministic_spmm=True
        if a.phase_valid_control:model.phase_valid_control=True
        if a.soft_risk_composition:model.soft_risk_composition=True
        if a.risk_protection_transition:
            if not (model.phase_valid_control and model.soft_risk_composition):
                raise ValueError('Protection transition requires finite-risk phase-aware control')
            model.protection_aware_risk=True
        if a.forecast_native_arming:model.anticipate_point_arming=True
        if a.forecast_growth_motion:model.forecast_growth_motion=True
        if a.task_progress_ports:model.enable_task_progress_ports(amplitude=a.task_progress_amplitude)
        launch['sensor_patches']=apply_sensor_patches(model,cfg,
            native_motion=a.native_motion_forecast,native_focus=a.native_focus_gate,
            unknown_laser_motion=a.unknown_laser_player_motion,native_curve=a.native_curve_laser_geometry)
        cfg['compact_native_position_hold_forecast']=launch['sensor_patches']['native_position_hold_forecast']
        cfg['compact_native_hidden_enemy_contacts']=launch['sensor_patches']['native_hidden_enemy_contacts']
        cfg['compact_native_ex8_polar4_forecast']=launch['sensor_patches']['native_ex8_polar4_forecast']
        if a.protected_exit_forecast:model.protected_exit_forecast=True
        if model.forecast_growth_motion and not model.anticipate_point_arming:
            raise ValueError('Growth forecast requires the native arming forecast')
        if a.primary_attack_target:model.primary_attack_target=True
        if a.risk_policy_gradient is not None:model.risk_policy_gradient=a.risk_policy_gradient
        if a.native_contact_risk_calibration is not None:model.native_contact_risk_calibration=a.native_contact_risk_calibration
        if a.neural_risk_scale is not None:
            from fly_touhou.compact_story_risk_composition import checked_risk_scale
            model.neural_risk_scale=checked_risk_scale(a.neural_risk_scale)
        if a.idle_prior_scale is not None:
            from fly_touhou.compact_story_idle_prior import checked_idle_scale
            model.idle_prior_scale=checked_idle_scale(a.idle_prior_scale)
        model.full_story_action_selection=a.evaluation_selection
        cfg['compact_forecast_native_arming']=model.anticipate_point_arming
        cfg['compact_forecast_growth_motion']=model.forecast_growth_motion
        cfg['compact_native_motion_forecast']=model.native_motion_forecast
        cfg['compact_native_focus_gate']=model.native_focus_gate
        cfg['compact_unknown_laser_player_motion']=model.unknown_laser_player_motion
        cfg['compact_native_curve_laser_geometry']=model.native_curve_laser_geometry
        cfg['compact_native_curve_motion_forecast']=getattr(model,'native_curve_motion_forecast',False)
        cfg['compact_native_enemy_damage_bound']=getattr(model,'native_enemy_damage_bound',False)
        cfg['compact_protected_exit_forecast']=model.protected_exit_forecast
        cfg['compact_primary_attack_target']=model.primary_attack_target
        write_json(a.output/'environment-config.json',cfg)
        if a.risk_input_soft_clip is not None:model.risk.input_soft_clip=a.risk_input_soft_clip
        if a.risk_base_input_soft_clip is not None:model.risk.base_input_soft_clip=a.risk_base_input_soft_clip
        if a.trainable_risk_baseline:model.risk.enable_trainable_baseline()
        optimizer=optimizer_for(model)
        launch['policy_optimizer_precision']=getattr(optimizer,'precision_revision',None)
        critical_replay=None
        if a.critical_teacher_replay:
            from fly_touhou.compact_story_critical_replay import CriticalTeacherReplay
            critical_replay=CriticalTeacherReplay(balanced=a.balanced_critical_teacher_replay,
                protected_exit_forecast=model.protected_exit_forecast,native_motion_forecast=model.native_motion_forecast,
                native_focus_gate=model.native_focus_gate,
                unknown_laser_player_motion=model.unknown_laser_player_motion,
                native_curve_laser_geometry=model.native_curve_laser_geometry,
                native_repair_revisions=current_revisions)
            for source in ([] if replay_archive.get('critical_replay') is not None else a.critical_replay_source):
                old_tape=torch.load(source,map_location='cpu',weights_only=False)
                critical_replay.add(old_tape,source=str(source.resolve()));del old_tape
            if replay_archive.get('critical_replay') is not None:critical_replay=replay_archive['critical_replay']
            write_json(a.output/'critical-teacher-replay.json',critical_replay.summary())
        late_replay=None
        if late_coefficient:
            from fly_touhou.compact_late_stage_replay import LateStageReplay,SENSOR_FLAGS
            from fly_touhou.thrl_dodge_diagnostics import _shared_binary
            late_replay=LateStageReplay(sensor_flags={key:getattr(model,key,False) for key in SENSOR_FLAGS},
                native_repair_revisions=current_revisions)
            if replay_archive.get('late_replay') is not None:
                late_replay.merge_payload(replay_archive['late_replay'],source=str(student_replay_archive))
            for bank_path in late_bank_paths:
                with _shared_binary(bank_path) as stream:
                    late_replay.merge_payload(torch.load(stream,map_location='cpu',weights_only=False),source=str(bank_path))
            for source in late_sources:
                launch_file=source.resolve().parent/'launch.json'
                settings=read(launch_file) if launch_file.exists() else {}
                old_tape=torch.load(source,map_location='cpu',weights_only=False)
                late_replay.add(old_tape,source=str(source.resolve()),source_settings=settings)
                del old_tape
            write_json(a.output/'late-stage-replay.json',late_replay.summary())
        del replay_archive
        rule_archive_replay=None
        if a.rule_teacher_archive_coefficient:
            from fly_touhou.compact_rule_teacher_replay import RuleTeacherArchiveReplay
            rule_archive_replay=RuleTeacherArchiveReplay(a.rule_teacher_archive_manifest)
            write_json(a.output/'rule-teacher-archive-replay.json',rule_archive_replay.summary())
        native_rule_replay=None
        if native_teacher['coefficient']:
            from fly_touhou.compact_native_rule_replay import NativeRuleTeacherReplay
            weights=native_teacher_stage_weight_map(native_teacher['manifest'],native_teacher['stage_weights'])
            native_rule_replay=NativeRuleTeacherReplay(native_teacher['manifest'],stage_weights=weights)
            write_json(a.output/'native-rule-teacher-replay.json',native_rule_replay.summary())
        if restored:
            launch['optimizer_restore']=restore_optimizer(optimizer,restored['optimizer']);torch.set_rng_state(restored['torch_RNG'])
            torch.cuda.set_rng_state_all(restored['CUDA_RNG'])
            updates=int(inherited_costs.get('PPO_updates',0));decisions=int(inherited_costs.get('new_student_actions',0))
            del restored
        from fly_touhou.compact_story_learning_rate import PolicyLearningRate
        learning_rate=PolicyLearningRate(optimizer,scale=a.policy_lr_scale,adaptive=a.adaptive_policy_lr,
            min_scale=a.minimum_policy_lr_scale,kl_early_stop=a.kl_early_stop,
            target_kl=a.target_kl if a.kl_early_stop or a.adaptive_policy_lr else None)
        multipliers={} if a.policy_group_lr_multipliers is None else a.policy_group_lr_multipliers
        if not isinstance(multipliers,dict):raise ValueError('Named policy-group multiplier mapping required')
        multipliers=dict(multipliers)
        if a.context_policy_lr_multiplier is not None:
            if ('new_context_ports' in multipliers
                    and multipliers['new_context_ports']!=a.context_policy_lr_multiplier):
                raise ValueError('Conflicting context-port multiplier settings')
            multipliers['new_context_ports']=a.context_policy_lr_multiplier
        learning_rate.set_group_multipliers(multipliers)
        launch['policy_lr_scale']=learning_rate.scale
        launch['minimum_policy_lr_scale']=learning_rate.min_scale
        launch['kl_early_stop']=learning_rate.kl_early_stop
        launch['target_kl']=learning_rate.target
        launch['context_policy_lr_multiplier']=next(g['policy_group_multiplier'] for g in optimizer.param_groups
            if g['name']=='new_context_ports')
        launch['policy_group_lr_multipliers']={g['name']:g['policy_group_multiplier'] for g in optimizer.param_groups
            if g['name']!='three_critics'}
        launch['boss_attack_guidance_coefficient']=attack_coefficient
        launch['conditional_wall_guidance_coefficient']=wall_coefficient
        launch['guided_wall_guidance_coefficient']=guided_wall_coefficient
        launch['selective_attack_guidance_coefficient']=selective_attack_coefficient
        launch['late_stage_replay_coefficient']=late_coefficient
        launch['late_stage_replay_banks']=[str(path) for path in late_bank_paths]
        launch['rule_teacher_archive_manifest']=str(a.rule_teacher_archive_manifest) if a.rule_teacher_archive_manifest else None
        launch['rule_teacher_archive_coefficient']=a.rule_teacher_archive_coefficient
        launch['rule_teacher_archive_every']=a.rule_teacher_archive_every
        launch.update(native_teacher_checkpoint_fields(native_teacher))
        event_ports=None
        if a.critical_event_port_amplitude:
            from fly_touhou.compact_story_model import prime_critical_event_ports
            event_ports=prime_critical_event_ports(model,amplitude=a.critical_event_port_amplitude)
        launch['critical_event_ports']=event_ports
        launch['phase_valid_control']=model.phase_valid_control
        launch['protection_aware_risk']=model.protection_aware_risk
        launch['protection_expiry_inclusive']=model.protection_expiry_inclusive
        launch['anticipate_point_arming']=model.anticipate_point_arming
        launch['forecast_growth_motion']=model.forecast_growth_motion
        launch['native_motion_forecast']=model.native_motion_forecast
        launch['native_position_hold_forecast']=cfg['compact_native_position_hold_forecast']
        launch['native_hidden_enemy_contacts']=cfg['compact_native_hidden_enemy_contacts']
        launch['native_ex8_polar4_forecast']=cfg['compact_native_ex8_polar4_forecast']
        launch['native_focus_gate']=model.native_focus_gate
        launch['unknown_laser_player_motion']=model.unknown_laser_player_motion
        launch['native_curve_laser_geometry']=model.native_curve_laser_geometry
        launch['native_curve_motion_forecast']=getattr(model,'native_curve_motion_forecast',False)
        launch['native_enemy_damage_bound']=getattr(model,'native_enemy_damage_bound',False)
        launch['task_progress_ports']=model.task_progress_ports
        launch['task_progress_drive']=(model.core.input_encoder.task_progress.specification()
            if model.task_progress_ports else None)
        launch['protected_exit_forecast']=model.protected_exit_forecast
        launch['primary_attack_target']=model.primary_attack_target
        launch['risk_policy_gradient']=model.risk_policy_gradient
        launch['native_contact_risk_calibration']=model.native_contact_risk_calibration
        launch['neural_risk_scale']=model.neural_risk_scale
        launch['idle_prior_scale']=model.idle_prior_scale
        launch['risk_readout_kind']=getattr(model.risk,'readout_kind','legacy')
        launch['soft_risk_composition']=model.soft_risk_composition
        launch['risk_input_soft_clip']=model.risk.input_soft_clip
        launch['risk_base_input_soft_clip']=getattr(model.risk,'base_input_soft_clip',None)
        launch['trainable_risk_baseline']=model.risk.baseline_trainable
        launch['deterministic_spmm']=model.core.sparse.deterministic_spmm
        action_origin=decisions;update_origin=updates
        launch['inherited_student_actions']=action_origin;launch['inherited_PPO_updates']=update_origin
        launch.update(CUDA_started=True,CUDA_device=torch.cuda.get_device_name(),
            global_memory_cap=None,learner_memory_cap=None)
        write_json(a.output/'launch.json',launch)
        from fly_touhou.compact_native_forecast import native_repair_revisions
        write_json(a.output/'algorithm.json',dict(whole_graph=True,neurons=138639,signed_directed_edges=15091983,
            native_repair_revisions=current_revisions,
            input_VP=8053,output_DN=1299,neural_ticks=16,one_shared_policy=True,input_features=129,neural_candidate_features=47,
            maps=False,PPO='original THRL three-value clipped objective;gamma.999,lambda.995,epsilon.2,value.5,entropy1e-4',
            BPTT=16,effective_minibatch_rows=256,micro_scenes=a.micro_scenes,epochs=a.epochs,
            policy_KL_control=('Before each optimizer step;keep applied updates' if learning_rate.kl_early_stop
                else 'Diagnostic only;complete selected-data epochs without KL early stopping'),
            native_game_round=dict(target_games_per_world=a.games_per_world,maximum_cycles=a.cycles,
                game_states_persist_between_PPO_chunks=True,checkpoints_do_not_restore_native_games=True),
            partial_risk='Known-point conditional danger retained under scene uncertainty;existing BCE budget shared;no safe-label or margin invention',
            teacher_resource_correction='native-positive-route-resource-reference-v1',
            guidance='original safe,future,hit,ResourceV10 rule,deathbomb;teacher never executes',
            critical_teacher_replay=critical_replay.summary() if critical_replay is not None else None,
            critical_event_ports=event_ports,
            phase_valid_control=model.phase_valid_control,
            protection_aware_risk=model.protection_aware_risk,
            anticipate_point_arming=model.anticipate_point_arming,
            forecast_growth_motion=model.forecast_growth_motion,
            native_motion_forecast=model.native_motion_forecast,
            native_focus_gate=model.native_focus_gate,
            unknown_laser_player_motion=model.unknown_laser_player_motion,
            native_curve_laser_geometry=model.native_curve_laser_geometry,
            minimum_policy_lr_scale=learning_rate.min_scale,
            boss_attack_guidance_coefficient=attack_coefficient,
            conditional_wall_guidance_coefficient=wall_coefficient,
            guided_wall_guidance_coefficient=guided_wall_coefficient,
            selective_attack_guidance_coefficient=selective_attack_coefficient,
            late_stage_replay_coefficient=late_coefficient,
            rule_teacher_archive_replay=rule_archive_replay.summary() if rule_archive_replay is not None else None,
            rule_teacher_archive_coefficient=a.rule_teacher_archive_coefficient,
            rule_teacher_archive_every=a.rule_teacher_archive_every,
            native_rule_teacher_replay=native_rule_replay.summary() if native_rule_replay is not None else None,
            native_rule_teacher_coefficient=native_teacher['coefficient'],
            native_rule_teacher_every=native_teacher['every'],
            native_rule_teacher_windows=native_teacher['windows'],
            rule_guidance_contact_veto='Skip teacher imitation on an observed unprotected movement contact;retain hit and native risk learning',
            task_progress_drive=launch['task_progress_drive'],
            protected_exit_forecast=model.protected_exit_forecast,
            primary_attack_target=model.primary_attack_target,
            risk_policy_gradient=model.risk_policy_gradient,
            native_contact_risk_calibration=model.native_contact_risk_calibration,
            risk_4f_native_BCE_share=.25 if model.native_contact_risk_calibration else 0.,
            native_contact_target_source='Actual selected TRAIN movement outcome and measured native frames;no unexecuted-action or development/test labels' if model.native_contact_risk_calibration else None,
            neural_risk_scale=model.neural_risk_scale,
            idle_prior_scale=model.idle_prior_scale,
            soft_risk_composition=model.soft_risk_composition,
            risk_input_soft_clip=model.risk.input_soft_clip,trainable_risk_baseline=model.risk.baseline_trainable,
            risk_base_input_soft_clip=getattr(model.risk,'base_input_soft_clip',None),
            runtime_control='run-control.json revisions apply only before fresh collection;readout soft limit, policy LR and context-port LR;no native/brain resets',
            deterministic_spmm=model.core.sparse.deterministic_spmm,
            risk='existing physical4/12/24 Huber+BCE calibration;no ranking objective',
            acting_telemetry='all18 neural risk forecasts,route scores,19logits,FP32 selected DN,pre-action trust and observed contacts;pooled DN at contact/Bomb/deathbomb. Telemetry excluded from learner inputs/targets.',
            Bomb='source18 policy logit,DN-only learned head plus trainable neural-risk initialization prior;inventory/state legality only',
            values='three heads from mean DN over all candidates,independent of sampled action',
            training='students play from opening through native Game Over/clear;worlds persist across optimizer cycles',
            movement_sampling_temperature=movement_temperature,
            bomb_sampling_temperature=bomb_temperature,
            exploration='Movement conditional temperature only;Bomb probability unchanged. PPO log-probs,KL and entropy use the actual behavior distribution. Frozen decoder unchanged.',
            evaluation='frozen full finite-life games,not first-stage diagnostic;independent formal20 follows a qualifying candidate',
            budget=dict(cycles=a.cycles,student_decisions=student_budget,
                PPO_student_decisions=student_budget-a.bomb_dagger_warmup_steps*a.workers,
                teacher_only_student_decisions=a.bomb_dagger_warmup_steps*a.workers,wall_hours=a.wall_hours),
            full_game_evaluation_every_cycles=None if ready_schedule else a.evaluation_every,
            full_game_evaluation_every_decisions=ready_schedule.evaluation_every if ready_schedule else None,
            PPO_batches_rejected=0))
        last_learning_row=None
        def save(tag,*,training_state=True):
            payload=model.parameter_payload(dict(inherited_costs,PPO_updates=updates,new_student_actions=decisions))
            path=a.output/(tag+'-parameters.pt');temp=path.with_suffix('.pt.tmp')
            with temp.open('wb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
            os.replace(temp,path)
            if training_state:
                training=training_directory/(tag+'-training.pt');temp=training.with_suffix('.pt.tmp')
                with temp.open('wb') as f:
                    torch.save(dict(model_payload=payload,optimizer=optimizer.state_dict(),torch_RNG=torch.get_rng_state(),CUDA_RNG=torch.cuda.get_rng_state_all(),
                        last_learning_row=last_learning_row,
                        movement_sampling_temperature=movement_temperature,bomb_sampling_temperature=bomb_temperature,
                        completed_cycle=completed_cycle,boss_attack_guidance_coefficient=attack_coefficient,
                        conditional_wall_guidance_coefficient=wall_coefficient,
                        guided_wall_guidance_coefficient=guided_wall_coefficient,
                        selective_attack_guidance_coefficient=selective_attack_coefficient,
                        **native_teacher_checkpoint_fields(native_teacher),
                        native_repair_revisions=current_revisions,
                        late_stage_replay_coefficient=late_coefficient,student_replay_archive=student_replay_archive,
                        late_stage_replay_sources=late_replay.sources if late_replay is not None else [],
                        late_stage_replay_banks=[str(path) for path in late_bank_paths] if late_replay is not None else []),f)
                    f.flush();os.fsync(f.fileno())
                os.replace(temp,training)
                write_json(a.output/(tag+'-training-path.json'),dict(path=str(training.resolve())))
            return path
        initial_checkpoint=save('initial');progress('starting_full_story_students')
        nodes=[];control_revision=0;flat_nodes=0;budget_stop=None
        best_report=read(a.reference_development_report) if a.reference_development_report else None
        best_checkpoint=str(initial_checkpoint) if best_report is not None else None
        from fly_touhou.compact_native_budget import compare,metrics as native_metrics
        allowance=None
        if a.initial_decision_budget:
            from fly_touhou.compact_native_budget import NativeBudgetAllowance
            allowance=NativeBudgetAllowance(a.initial_decision_budget,read(a.budget_benefit_reference))
            write_json(a.output/'external-budget-reference.json',dict(path=str(a.budget_benefit_reference),
                reference=native_metrics(allowance.reference),initial_model_score_assigned=False,
                initial_decision_budget=a.initial_decision_budget))
        if best_report is not None:
            native_metrics(best_report)
            # A standalone frozen baseline may label the initial checkpoint
            # only when it really used these weights and deployment settings.
            reference_launch=read(a.reference_development_report.resolve().parent.parent/'launch.json')
            digest=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
            same_flags=(reference_launch.get('protection_aware_risk',False)==model.protection_aware_risk
                and reference_launch.get('protection_expiry_inclusive',False)==model.protection_expiry_inclusive
                and reference_launch.get('anticipate_point_arming',False)==model.anticipate_point_arming
                and reference_launch.get('forecast_growth_motion',False)==model.forecast_growth_motion
                and reference_launch.get('protected_exit_forecast',False)==model.protected_exit_forecast
                and reference_launch.get('primary_attack_target',False)==model.primary_attack_target
                and reference_launch.get('risk_base_input_soft_clip')==getattr(model.risk,'base_input_soft_clip',None)
                and reference_launch.get('neural_risk_scale',1.)==model.neural_risk_scale
                and reference_launch.get('idle_prior_scale',1.)==model.idle_prior_scale
                and best_report.get('action_selection','joint_mode')==a.evaluation_selection)
            reference_point=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
            same_weights=all(torch.equal(v.detach().cpu(),reference_point['state_dict'][k]) for k,v in model.state_dict().items())
            del reference_point
            if digest!=reference_launch.get('checkpoint_SHA256') or not same_flags or not same_weights:
                raise ValueError('Frozen reference does not bind to the initial weights and deployment settings')
            write_json(a.output/'native-reference-binding.json',dict(report=str(a.reference_development_report),
                initial_parameters=str(initial_checkpoint),checkpoint_SHA256=digest,neural_weights_equal=True,
                deployment_settings_equal=True,risk_policy_gradient_changes_forward=False))
        collector_type=Collector;collection_options={}
        if a.ready_driven:
            from fly_touhou.compact_story_ready import ReadyCollector
            collector_type=ReadyCollector
            collection_options=dict(ready_batch_size=a.ready_batch_size,ready_wait_ms=a.ready_wait_ms,
                per_world_quota=a.ready_per_world_quota)
        collector=collector_type(model,cfg,a.output/'native-training',workers=a.workers,worker_cpus=cpus,
            movement_sampling_temperature=movement_temperature,bomb_sampling_temperature=bomb_temperature,
            **collection_options)
        if a.bomb_dagger_warmup_steps:
            from fly_touhou.compact_story_bomb_dagger import verify_anchor_model,refit,reset_live_bomb_moments
            anchor=torch.load(a.bomb_dagger_anchor_cache,map_location='cpu',weights_only=False)
            anchor_source=torch.load(anchor['metadata']['checkpoint'],map_location='cpu',weights_only=False)
            verify_anchor_model(model,anchor,anchor_source);del anchor_source
            warmup=collector.rollout(model,steps=a.bomb_dagger_warmup_steps,guard=guard,
                supervision_selection=a.evaluation_selection,
                progress=lambda info:progress('student_deployment_teacher_collection',acquisition=info))
            decisions+=a.workers*a.bomb_dagger_warmup_steps
            warmup_path=a.output/'Bomb-DAgger-warmup.pt'
            save_rollout(warmup,warmup_path,compressed=a.compress_tapes)
            result,fresh=refit(model,warmup,anchor,updates=a.bomb_dagger_readout_updates,guard=guard,
                progress=lambda info:progress('Bomb_DAgger_teacher_correction',learning=info))
            torch.save(fresh,a.output/'Bomb-DAgger-fresh-DN.pt')
            write_json(a.output/'Bomb-DAgger-report.json',dict(result,native_worlds_reset=False,
                carried_brain_state_reset=False,source_rollout=str(warmup_path)))
            inherited_costs['Bomb_DAgger_readout_updates_paid']=int(inherited_costs.get('Bomb_DAgger_readout_updates_paid',0))+a.bomb_dagger_readout_updates
            if result['applied']:reset_live_bomb_moments(optimizer)
            if critical_replay is not None:
                critical_replay.add(warmup,source=str(warmup_path.resolve()))
                write_json(a.output/'critical-teacher-replay.json',critical_replay.summary())
            save('Bomb-DAgger-warmup')
            del warmup,anchor,fresh;gc.collect();torch.cuda.empty_cache()
        from fly_touhou.compact_story_runtime_control import apply_control
        for local_cycle in range(1,a.cycles+1):
            cycle=a.cycle_offset+local_cycle
            guard()
            if ready_schedule:
                collection_control=ready_schedule.read_control(a.output/'collection-control.json')
                if collection_control is not None:
                    append_jsonl(a.output/'collection-control-applied.jsonl',dict(collection_control,cycle=cycle))
                segment_decisions=ready_schedule.next_size(decisions-action_origin)
                if not segment_decisions:break
            else:segment_decisions=a.workers*a.steps
            if allowance is not None and not allowance.can_collect(decisions-action_origin,segment_decisions):
                budget_stop=dict(reason='initial_budget_exhausted_without_external_native_benefit',cycle=cycle,
                    student_actions_this_job=decisions-action_origin,initial_decision_budget=allowance.initial_decisions,
                    unused_maximum_decisions=student_budget-(decisions-action_origin),best_parameters=best_checkpoint)
                write_json(a.output/'native-budget-stop.json',budget_stop);break
            control_revision,control=apply_control(a.output/'run-control.json',model,learning_rate,seen_revision=control_revision)
            if control is not None:
                append_jsonl(a.output/'runtime-control-applied.jsonl',dict(control,cycle=cycle,PPO_updates=updates))
                save('latest')
            rollout=collector.rollout(model,steps=a.steps,guard=guard,
                **(dict(total_decisions=segment_decisions) if a.ready_driven else {}),
                progress=lambda info:progress('student_full_story_collection',cycle=cycle,acquisition=info))
            decisions+=rollout['metadata']['student_actions']
            tape=(secondary if secondary is not None and cycle%2==0 else a.output)/f'rollout-{cycle:03}.pt'
            storage=save_rollout(rollout,tape,compressed=a.compress_tapes)
            write_json(a.output/f'rollout-{cycle:03}-path.json',dict(path=str(tape.resolve())))
            write_json(a.output/f'rollout-{cycle:03}-storage.json',storage)
            write_json(a.output/f'rollout-{cycle:03}.json',rollout['metadata'])
            write_json(a.output/'stage-coverage.json',collector.coverage.summary(games_per_world=a.games_per_world))
            if critical_replay is not None:
                critical_replay.add(rollout,source=str(tape.resolve()))
                write_json(a.output/'critical-teacher-replay.json',critical_replay.summary())
            if late_replay is not None:
                late_replay.add(rollout,source=str(tape.resolve()),source_settings=launch)
                write_json(a.output/'late-stage-replay.json',late_replay.summary())
            if a.keep_rollout_tapes:
                from fly_touhou.compact_story_tape_retention import save_replays
                student_replay_archive=str((a.output/f'student-supervision-{cycle:03}.pt').resolve())
                save_replays(student_replay_archive,critical_replay,late_replay,current_revisions)
            base_updates=updates
            def learning(info):
                nonlocal updates,trained_rows,last_learning_row
                trained_rows+=info['trained_student_rows']
                updates=base_updates+info['update'];row=dict(info,step=updates,cycle=cycle,new_student_actions=decisions,
                    student_actions_this_job=decisions-action_origin,PPO_updates_this_job=updates-update_origin)
                last_learning_row=dict(row)
                append_jsonl(a.output/'learning.jsonl',row);progress('full_story_PPO_update',cycle=cycle,learning=row)
                if updates%4==0:save('latest')
            numerical_before=optimizer.sampled_update_snapshot()
            learning_started=time.monotonic()
            learning_result=update(model,optimizer,rollout,epochs=a.epochs,micro_scenes=a.micro_scenes,seed=a.seed+cycle,
                guard=guard,progress=learning,critical_replay=critical_replay,
                world_aligned=a.world_aligned_updates,checkpoint_chunks=a.checkpoint_candidate_chunks,
                boss_attack_coefficient=attack_coefficient,conditional_wall_coefficient=wall_coefficient,
                guided_wall_coefficient=guided_wall_coefficient,selective_attack_coefficient=selective_attack_coefficient,
                late_replay=late_replay,late_replay_coefficient=late_coefficient,
                rule_archive_replay=rule_archive_replay,rule_archive_coefficient=a.rule_teacher_archive_coefficient,
                rule_archive_every=a.rule_teacher_archive_every,
                native_rule_replay=native_rule_replay,native_rule_coefficient=native_teacher['coefficient'],
                native_rule_every=native_teacher['every'],native_rule_windows=native_teacher['windows'],
                max_updates=ready_schedule.max_updates if ready_schedule else a.max_adam_updates,
                sample_decisions=a.ppo_sample_decisions,stage_weights=dict(enumerate(a.ppo_stage_weights,1)),
                target_kl=learning_rate.target if learning_rate.kl_early_stop else None)
            if learning_result['selection'] is not None:
                write_json(a.output/f'PPO-selection-{cycle:03}.json',learning_result['selection'])
            append_jsonl(a.output/'numerical-updates.jsonl',dict(cycle=cycle,PPO_updates=updates,
                precision=optimizer.precision_revision,
                scope='Fixed samples, at most4096 per policy parameter tensor; not full-graph counts. Diagnostic only.',
                parameters=optimizer.sampled_update_changes(numerical_before)))
            del numerical_before
            write_json(a.output/f'update-{cycle:03}.json',dict(
                optimizer_updates=learning_result['optimizer_updates'],early_stop=learning_result['early_stop'],
                compute_quota_stop=learning_result['compute_quota_stop'],
                PPO_trained_rows=learning_result['trained_student_rows'],
                PPO_unique_trained_rows=learning_result['unique_trained_student_rows'],
                PPO_unoptimized_rows=learning_result['unoptimized_student_rows'],
                PPO_selected_rows=learning_result['selected_student_rows'],
                PPO_unselected_rows=rollout['metadata']['student_actions']-learning_result['selected_student_rows'],
                PPO_selected_but_not_updated_rows=learning_result['selected_student_rows']-learning_result['unique_trained_student_rows'],
                PPO_trained_rows_by_stage=learning_result['trained_rows_by_stage'],
                learner_seconds=time.monotonic()-learning_started,collection_seconds=rollout['metadata'].get('elapsed_seconds'),
                segment_decisions=rollout['metadata']['student_actions'],
                world_decisions=rollout['metadata'].get('world_decisions'),
                native_worlds_parked_during_update=True,native_resets_at_update=0))
            schedule=learning_rate.after_rollout(learning_result['logs'],early_stop=learning_result['early_stop'],
                growth_fraction=min(1.,segment_decisions/(a.lr_growth_reference_decisions or a.workers*a.steps)))
            if native_rule_replay is not None:
                write_json(a.output/'native-rule-teacher-replay.json',native_rule_replay.summary())
            completed_cycle=cycle
            append_jsonl(a.output/'learning-rate.jsonl',dict(schedule,cycle=cycle,PPO_updates=updates))
            # The last minibatch checkpoint was saved before completed_cycle
            # advanced. Save the actual boundary (including Adam and the
            # adaptive LR) so a resumed run does not relabel a finished cycle.
            save('latest')
            game_target_met=collector.coverage.reached(a.games_per_world)
            if a.keep_rollout_tapes:
                from fly_touhou.compact_story_tape_retention import prune_tapes
                retention=prune_tapes(a.output,a.keep_rollout_tapes)
                append_jsonl(a.output/'tape-retention.jsonl',dict(retention,cycle=cycle,
                    supervision_archive=student_replay_archive,checkpoint_cycle=completed_cycle))
            paid=decisions-action_origin
            evaluation_node=((ready_schedule.evaluation_due(paid) if ready_schedule else local_cycle%a.evaluation_every==0)
                or local_cycle==a.cycles or game_target_met)
            parameter_node=ready_schedule.checkpoint_due(paid) if ready_schedule else local_cycle%a.parameter_checkpoint_every==0
            checkpoint=(save(f'cycle{cycle:03}',training_state=evaluation_node)
                if evaluation_node or parameter_node else None)
            if checkpoint is not None and ready_schedule:ready_schedule.checkpointed(paid)
            del rollout;gc.collect();torch.cuda.empty_cache();model.eval()
            if (a.output/'STOP_AFTER_CYCLE').exists():raise RuntimeError('Operator STOP_AFTER_CYCLE')
            if not evaluation_node:
                progress('student_cycle_completed',cycle=cycle)
                continue
            # Training worlds stay parked with their genuine carried brain/game
            # states. These separate frozen games share no training actions.
            report=evaluate(model,cfg,a.output/f'development-cycle{cycle:03}',episodes=a.evaluation_episodes,worker_cpus=cpus,
                guard=guard,progress=lambda info:progress('frozen_full_story_development',cycle=cycle,evaluation=info),
                action_selection=a.evaluation_selection)
            nodes.append(dict(cycle=cycle,checkpoint=str(checkpoint),report=report))
            if ready_schedule:ready_schedule.evaluated(paid)
            write_json(a.output/'evaluations.json',dict(nodes=nodes))
            if allowance is not None:
                append_jsonl(a.output/'native-budget-extension.jsonl',dict(allowance.observe(report),cycle=cycle,
                    student_actions_this_job=decisions-action_origin,training_worlds_reset=False))
            decision=(compare(report,best_report) if best_report is not None else
                dict(improved=True,reason='first_frozen_native_reference',candidate=native_metrics(report)))
            if decision['improved']:best_report=report;best_checkpoint=str(checkpoint);flat_nodes=0
            else:flat_nodes+=1
            append_jsonl(a.output/'native-budget-gate.jsonl',dict(decision,cycle=cycle,consecutive_flat_nodes=flat_nodes,
                best_parameters=best_checkpoint,optimizer_updates=updates))
            if game_target_met:break
            if ready_schedule and paid>=student_budget:break
            if a.stop_at_development_clears and report['clears']>=a.stop_at_development_clears:break
            if a.stop_after_flat_evaluations and flat_nodes>=a.stop_after_flat_evaluations:
                budget_stop=dict(reason='no_frozen_native_benefit',cycle=cycle,consecutive_flat_nodes=flat_nodes,
                    unused_planned_decisions=student_budget-paid,best_parameters=best_checkpoint)
                write_json(a.output/'native-budget-stop.json',budget_stop);break
        coverage=collector.coverage.summary(games_per_world=a.games_per_world)
        write_json(a.output/'stage-coverage.json',coverage)
        save('final');collector.close();collector=None;native_closed=True
        best=dict(checkpoint=best_checkpoint,report=best_report)
        write_json(a.output/'result.json',dict(status='completed_finite_full_story_PPO',PPO_updates=updates,
            new_student_actions=decisions,best_parameters=best['checkpoint'],best_clears=best['report']['clears'],
            student_actions_this_job=decisions-action_origin,PPO_updates_this_job=updates-update_origin,
            PPO_trained_rows_this_job=trained_rows,
            stage_coverage=coverage,requested_native_game_round_complete=coverage['game_target_met'],
            evaluated_episodes=a.evaluation_episodes,best_action_selection=a.evaluation_selection,native_budget_stop=budget_stop,
            goal_complete_claimed=False,full_lunatic_stable_clear_not_yet_established=True))
        progress('completed_finite_full_story_PPO')
    except BaseException as error:
        intentional=isinstance(error,RuntimeError) and str(error) in ('Operator STOP','Operator STOP_AFTER_CYCLE')
        stopped_checkpoint=checkpoint_save_error=None
        if 'save' in locals():
            try:stopped_checkpoint=str(save('operator-stop' if intentional else 'interrupted'))
            except BaseException as save_error:checkpoint_save_error=repr(save_error)
        write_json(a.output/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),PPO_updates=updates,
            new_student_actions=decisions,operator_requested_stop=intentional,
            operator_stop_checkpoint=stopped_checkpoint if intentional else None,
            interrupted_checkpoint=stopped_checkpoint if not intentional else None,
            checkpoint_save_error=checkpoint_save_error,no_automatic_retry=True));raise
    finally:
        if collector is not None:collector.close()
        if native_closed and native_temp is not None:
            from fly_touhou.thrl_job_paths import cleanup_native_temp
            write_json(a.output/'native-temp-cleanup.json',cleanup_native_temp(native_temp))


if __name__=='__main__':main()
