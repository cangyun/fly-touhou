"""Finite risk-path7 cost-only stage; default NoTorch plan, no automatic execution.

Adopt a real joint26 checkpoint. Collect student actions in30 original worlds,
replay actual own-state T4 blocks, update only input2+gain1+risk4, then frozen20.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from train_fly_dn_cost_online import (SCHEMA as JOINT_SCHEMA, D95_SHA, ROLLOUT_ROWS,
    ContinuousCostConfig, joint_algorithm, restore_joint_source, guidance_protocol, refresh_protocol, sha)
from train_fly_task_supervision import require_completed_predecessor
from fly_touhou.thrl_study_io import read_shared_json

SCHEMA = 'fly-risk-path7-cost-only-original30-v1'
HANDOFF_SCHEMA = 'fly-goal-authorized-risk-path7-phase-handoff-v1'


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('environment-config','training-source','predecessor-output','output','d95-checkpoint'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--training-source-sha256',required=True)
    parser.add_argument('--phase-handoff-receipt',type=Path)
    parser.add_argument('--connectome',type=Path,default=ROOT/'data/connectome')
    parser.add_argument('--decisions',type=int,default=30720)
    parser.add_argument('--joint-decisions',type=int,default=0)
    parser.add_argument('--enable-joint-training-cuda-graph',action='store_true')
    parser.add_argument('--epochs',type=int,choices=(1,2,3),default=3)
    parser.add_argument('--max-seconds',type=int,required=True)
    parser.add_argument('--seed',type=int,default=613711)
    parser.add_argument('--action-seed',type=int,default=971137)
    parser.add_argument('--geometry-label-budget',type=int,default=2048)
    parser.add_argument('--state-refresh-steps',type=int,choices=(0,16),default=16)
    parser.add_argument('--execute-cuda',action='store_true')
    args=parser.parse_args(argv)
    if (args.decisions<ROLLOUT_ROWS or args.decisions%ROLLOUT_ROWS or args.max_seconds<1
            or args.joint_decisions<0 or args.joint_decisions%ROLLOUT_ROWS
            or not 1<=args.geometry_label_budget<=2048
            or any(not 0<=seed<2**31 for seed in (args.seed,args.action_seed))
            or len(args.training_source_sha256)!=64
            or any(c not in '0123456789abcdef' for c in args.training_source_sha256)):
        parser.error('Explicit finite7680-row budget,31bit seeds and lowercase source SHA required')
    return args


def plan(a):
    return dict(schema=SCHEMA,status='prepared_not_executed',Torch_imported='torch' in sys.modules,
        source_kind='joint26',source_checkpoint=str(a.training_source),source_sha256=a.training_source_sha256,
        actual_native_worlds=30,decisions=a.decisions,rollouts=a.decisions//ROLLOUT_ROWS,
        joint_decisions=a.joint_decisions,joint_rollouts=a.joint_decisions//ROLLOUT_ROWS,
        total_new_student_decisions=a.decisions+a.joint_decisions,
        epochs=a.epochs,risk_path_Adam_max=a.decisions//ROLLOUT_ROWS*30*a.epochs,
        joint_Adam_max=a.joint_decisions//ROLLOUT_ROWS*90,
        Adam_max=a.decisions//ROLLOUT_ROWS*30*a.epochs+a.joint_decisions//ROLLOUT_ROWS*90,
        trainable_tensors=7,optimizer_tensors_retained=26,frozen_actor_critic_tensors=19,
        optimizer_reset=False,learning_rate=3e-5,global_grad_clip=.5,
        loss=asdict(ContinuousCostConfig()),PPO_updates=0,value_updates=0,guidance_updates=0,
        zero_PPO_value_guidance_scope='risk-path7 phase only',
        neural_batch=16,real_sampling_groups=[16,14],brain_BPTT=4,minibatch_true_rows=256,
        cached_DN_substituted_for_replay=False,actual_rollout_block_states_required=True,
        activation_checkpoint=True,new_training_CUDA_graph=False,
        optional_existing_joint_training_graph=a.enable_joint_training_cuda_graph,input_drive='poisson',
        fixed_DN_statistics_runtime_updates=False,source_reward_schema='th14-completion-reward-v3',
        teacher_actions=0,safety_takeovers=0,worlds_reset_on_Miss_stage_or_update=False,
        final_frozen_development_games=20,development_native_workers=20,
        development_seed=2027431601,development_action_selection='greedy T1',
        native_frame_cap=None,Miss_cap=None,continues=False,artificial_memory_caps=False,
        predecessor_must_be_exited_and_cleaned=True,handoff_schema=HANDOFF_SCHEMA,
        source_may_be_outside_predecessor_directory=True,maximum_seconds=a.max_seconds,
        GPU_or_native_started=False,current_job_modified=False)


def _exited(launch,process_factory=None):
    if process_factory is None:
        import psutil
        def process_factory(pid):
            try:return psutil.Process(pid)
            except psutil.NoSuchProcess:return None
    process=process_factory(int(launch['PID']))
    if process is not None:
        if abs(float(process.create_time())-float(launch['create_time']))>=.01:
            raise RuntimeError('Predecessor PID reused;explicit ownership cannot be verified')
        if process.is_running():raise RuntimeError('Predecessor is still alive;no overlapping execution')


def require_phase_predecessor(folder,source_path,source_sha,*,handoff_receipt=None,process_factory=None):
    """Normal completion or an honest owner/goal-authorized OperatorSTOP receipt.

    The source checkpoint is independently bound and need not be in folder.
    No claim that the user directly requested this particular phase is required.
    """
    folder=Path(folder)
    if handoff_receipt is None:
        if (folder/'training-cleanup.json').is_file():
            return require_completed_predecessor(folder,process_factory=process_factory)
        # A separately-owned frozen20 has native_cleanup but no training-cleanup.
        launch=read_shared_json(folder/'launch.json');_exited(launch,process_factory)
        result=read_shared_json(folder/'result.json')
        cleanup=read_shared_json(folder/'development/native_cleanup.json')
        if result.get('status')!='completed' or cleanup.get('close_errors')!=[]:
            raise RuntimeError('Completed frozen evaluation and actual native cleanup required')
        recovery_receipt=None
        if launch.get('recovery_receipt'):
            receipt_path=Path(launch['recovery_receipt'])
            recovery_receipt=dict(path=str(receipt_path),sha256=sha(receipt_path),record=read_shared_json(receipt_path))
        return dict(kind='completed_frozen_evaluation',actual_PID=launch['PID'],
            actual_create_time=launch['create_time'],verified_exited=True,
            result_sha256=sha(folder/'result.json'),cleanup_sha256=sha(folder/'development/native_cleanup.json'),
            launch_record=launch,recovery_receipt=recovery_receipt)
    receipt=read_shared_json(handoff_receipt);launch=read_shared_json(folder/'launch.json')
    _exited(launch,process_factory)
    failure=read_shared_json(folder/'failure.json');cleanup=read_shared_json(folder/'training-cleanup.json')
    if (receipt.get('schema')!=HANDOFF_SCHEMA or receipt.get('authorization_basis')!='existing_user_goal'
            or receipt.get('owner_authorized_phase_switch') is not True
            or receipt.get('transition')!='joint26_to_risk_path7'
            or Path(receipt.get('predecessor_output','')).resolve()!=folder.resolve()
            or receipt.get('predecessor_PID')!=launch['PID']
            or abs(float(receipt.get('predecessor_create_time',-1))-float(launch['create_time']))>=.01
            or receipt.get('failure_sha256')!=sha(folder/'failure.json')
            or receipt.get('cleanup_sha256')!=sha(folder/'training-cleanup.json')
            or receipt.get('owned_native_processes_remaining')!=[] or cleanup.get('close_errors')!=[]
            or 'OperatorSTOP' not in str(failure.get('error',''))
            or Path(receipt.get('source_training_checkpoint','')).resolve()!=Path(source_path).resolve()
            or receipt.get('source_training_sha256')!=source_sha or sha(source_path)!=source_sha):
        raise ValueError('Goal-authorized handoff must match actual owner/failure/cleanup/source identities')
    extra=None
    if receipt.get('completed_evaluation_output') is not None:
        extra=require_phase_predecessor(receipt['completed_evaluation_output'],source_path,source_sha,
            process_factory=process_factory)
    return dict(kind='goal_authorized_owner_phase_handoff',verified_exited=True,actual_PID=launch['PID'],
        actual_create_time=launch['create_time'],receipt_sha256=sha(handoff_receipt),
        source_training_checkpoint=str(source_path),source_training_sha256=source_sha,
        completed_evaluation=extra,direct_user_phase_switch_claimed=False)


def risk_path_algorithm(source_joint,a):
    if not isinstance(source_joint,dict) or source_joint.get('schema')!=JOINT_SCHEMA:
        raise ValueError('Explicit current joint26 source algorithm required;not head-only warm or a renamed phase')
    return dict(schema=SCHEMA,phase='risk_path_only_continuous_cost_pretraining',
        parent_joint_algorithm=copy.deepcopy(source_joint),epochs=a.epochs,
        cost=asdict(ContinuousCostConfig()),trainable_parameter_tensors=7,
        frozen_actor_critic_tensors=19,optimizer_parameter_tensors=26,learning_rate=3e-5,
        max_grad_norm=.5,brain_BPTT=4,neural_width=16,worlds=30,rollout_steps=256,
        PPO_updates=0,value_updates=0,guidance_updates=0,
        checkpoint_replay=True,training_graph=False,source_reward_structure_unchanged=True,
        source_Adam_states_retained=True,actor_critic_Adam_updates=False,
        fixed_feature_statistics_runtime_updates=False,teacher_actions=0,
        input_drive_protocol=copy.deepcopy(source_joint['input_drive_protocol'])
            if 'input_drive_protocol' in source_joint else None)


def source_owner_provenance(source_path):
    """Preserve a stopped source owner's receipt chain even when eval is predecessor.

    This is provenance, not a fabricated direct-user instruction or a rule that
    the source must reside in the completed evaluation's output directory.
    """
    folder=Path(source_path).parent;records={}
    for name in ('launch.json','failure.json','training-cleanup.json','result.json'):
        path=folder/name
        if path.is_file():records[name]=dict(path=str(path),sha256=sha(path),record=read_shared_json(path))
    return dict(folder=str(folder),records=records,source_directory_assumed_to_equal_predecessor=False)


def restore_risk_path_source(base,source,*,configure=None):
    """Restore ALL26 actual moments before declaring the restricted learning scope."""
    if source['recovery']['algorithm'].get('schema')!=JOINT_SCHEMA:
        raise ValueError('This entry begins explicitly from joint26 only')
    model,optimizer,counters,record=restore_joint_source(base,source)
    if configure is None:
        from fly_touhou.fly_risk_path_pretraining import configure_risk_path_learning
        configure=configure_risk_path_learning
    selection=configure(model,optimizer)
    return model,optimizer,counters,dict(restoration=record,learning_scope=selection)


def frozen19_snapshot(model,optimizer):
    """Exact value and Adam hashes;no moments are treated as gradient evidence."""
    parameters=[model.brain.action_readout.weight]+list(model.critic.parameters())
    if len(parameters)!=19:raise ValueError('Original actor1+three-critic18 tensors required')
    names={id(p):n for n,p in model.named_parameters()}
    def digest(tensor):
        return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    return {names[id(p)]:dict(parameter_sha256=digest(p),
        Adam={key:digest(value) for key,value in optimizer.state[p].items()}) for p in parameters}


def begin_joint_phase(model,optimizer,risk_algorithm,*,collector=None):
    """Same26 storage/moments and live worlds, original joint algorithm verbatim.

    Current brain capture signature omits requires_grad. Still compare the
    actual signature around this authorized flag change; recapture only if
    that flag transition invalidates a previously valid capture.
    """
    from fly_touhou.fly_gru_risk26_update import risk26_parameters
    from fly_touhou.brain_cuda_capture import _signature
    if risk_algorithm.get('schema')!=SCHEMA or risk_algorithm.get('parent_joint_algorithm',{}).get('schema')!=JOINT_SCHEMA:
        raise ValueError('Explicit risk-path7 -> retained original joint26 transition required')
    parameters=[p for g in optimizer.param_groups for p in g['params']]
    tokens=[(id(p),id(optimizer.state[p]),tuple((k,id(v),v._version) for k,v in optimizer.state[p].items())) for p in parameters]
    capture=model.brain._captured_inference;before_signature=_signature(model.brain)
    if capture is not None and capture.signature!=before_signature:
        raise RuntimeError('Sampling capture already had unexpected source/storage changes before phase switch')
    optimizer.zero_grad(set_to_none=True)
    for module in (model.brain,model.critic,model.risk_head):module.requires_grad_(True)
    risk26_parameters(model)
    if tokens!=[(id(p),id(optimizer.state[p]),tuple((k,id(v),v._version) for k,v in optimizer.state[p].items())) for p in parameters]:
        raise RuntimeError('Joint phase switch changed inherited Adam objects')
    recreated=False;setup_lanes=0
    if capture is not None and capture.signature!=_signature(model.brain):
        if collector is None:raise ValueError('Collector required to rebuild changed inference capture')
        model.brain.clear_cuda_capture();collector.sampling_capture=None
        collector.enable_sampling_capture();recreated=True;setup_lanes=4*collector.neural_batch_size
    if hasattr(model,'_risk26_initial_logp_verified'):delattr(model,'_risk26_initial_logp_verified')
    return copy.deepcopy(risk_algorithm['parent_joint_algorithm']),dict(
        schema='fly-risk-path7-to-original-joint26-transition-v1',all26_enabled=True,
        original26_Adam_objects_and_steps_preserved=True,optimizer_reset=False,
        native_worlds_restarted=0,collector_states_reset=False,neural_data_forwards=0,
        sampling_capture_recreated=recreated,sampling_capture_setup_neural_lane_forwards=setup_lanes,
        original_joint_algorithm_restored_verbatim=True)


def write_rollout_trace(rollout,output,version,*,phase='risk_path'):
    """Compact real observations/labels;no invented state or cached-DN training."""
    import numpy as np
    from fly_touhou.thrl_durable_records import write_json
    names=('encoded_visual','features','actions','action_probs','old_log_probs','legal_actions',
        'raw_logits','composed_logits','predicted_costs','DN_features','values','rewards','reward_vectors',
        'reward_raw_vectors','terminated','truncated','episode_starts','game_ids','native_episodes',
        'brain_rng_keys','noise_counters','d95_risk_costs','d95_risk_queried_mask','d95_risk_valid_mask')
    arrays={n:getattr(rollout,n).detach().cpu().numpy() for n in names}
    path=Path(output)/'traces'/f'rollout-{version:03}.npz';temporary=path.with_suffix('.npz.tmp')
    with temporary.open('wb') as stream:
        np.savez(stream,**arrays);stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)
    write_json(path.with_suffix('.json'),dict(metadata=rollout.metadata,step_info=rollout.step_info,
        completed_episodes=list(rollout.completed_episodes),reward_components=rollout.reward_components,
        trace_sha256=sha(path),phase=phase,PPO_updates_applied_before_trace=0,
        planned_learning_objective=('cost-only' if phase=='risk_path' else 'parent three-value PPO/guidance/cost'),
        actual_CPU_block_states_available_for_replay=True,full_block_states_archived=False,
        raw_maps_or_objects_saved=False,teacher_costs_are_not_native_counterfactual_truth=True))


def setup_population_diagnostics(model,output):
    """Prepare the old runner's passive layout once;diagnostics never reject learning."""
    from fly_touhou.fly_population_path_trace import build_population_layout
    from fly_touhou.thrl_durable_records import write_json
    try:
        layout=build_population_layout(model.brain)
        report=dict(status='ready',max_scenes_per_rollout=64,
            engineering_middle_is_not_a_functional_region=True,
            annotation_provenance=layout.provenance,extra_brain_forwards=0)
    except (ValueError,OSError,RuntimeError) as error:
        layout=None
        report=dict(status='annotation_layout_unavailable',error=repr(error),
            PPO_rows_retained=True,extra_brain_forwards=0)
    try:write_json(Path(output)/'population-trace-setup.json',report)
    except (ValueError,OSError) as error:report['report_write_error']=repr(error)
    return layout,report


def record_after_collect_diagnostics(model,rollout,output,version,layout,*,phase,costs):
    """Use actual CPU snapshots aftercollect/beforeAdam;zero neural/teacher calls."""
    from fly_touhou.fly_representation_trace import record_rollout_representations
    from fly_touhou.fly_population_path_trace import record_population_path_trace
    from fly_touhou.thrl_durable_records import write_json
    directory=Path(output)/'neural-diagnostics'/f'rollout-{version:06}'
    unavailable=lambda error:dict(status='diagnostic_unavailable',error=repr(error),
        extra_brain_forwards=0,PPO_rows_retained=True)
    try:representation=record_rollout_representations(model,rollout,directory/'representations',max_scenes=512)
    except (ValueError,OSError,RuntimeError) as error:representation=unavailable(error)
    population=dict(status='annotation_layout_unavailable',extra_brain_forwards=0)
    if layout is not None:
        try:population=record_population_path_trace(model,rollout,directory/'population-path',max_scenes=64,layout=layout)
        except (ValueError,OSError,RuntimeError) as error:population=unavailable(error)
    report=dict(representations=representation,population_path=population,phase=phase,
        timing='after_collect_before_any_update',diagnostic_metrics_reject_updates=False,
        extra_neural_lane_forwards=0,teacher_queries=0)
    try:write_json(directory/'report.json',report)
    except (ValueError,OSError) as error:report['report_write_error']=repr(error)
    for key in ('representation_record_calls','population_record_calls'):
        costs[key]=costs.get(key,0)+1
    return report


def execute(a):
    if a.output.exists():raise FileExistsError('New finite-stage output required;no overwrite or retry')
    predecessor=require_phase_predecessor(a.predecessor_output,a.training_source,a.training_source_sha256,
        handoff_receipt=a.phase_handoff_receipt)
    if sha(a.training_source)!=a.training_source_sha256 or sha(a.d95_checkpoint)!=D95_SHA:
        raise ValueError('Actual typed26 source and frozenD95 SHA required')
    source_owner=source_owner_provenance(a.training_source)
    if 'launch.json' in source_owner['records']:
        _exited(source_owner['records']['launch.json']['record'])
    environment=read_shared_json(a.environment_config)
    from fly_touhou.thrl_training_mode import validate_training_mode
    if (not validate_training_mode(environment) or environment.get('reward_schema')!='th14-completion-reward-v3'
            or environment.get('training_mode')!='full-story-v1' or environment.get('headless') is not True
            or environment.get('parameters',{}).get('gamma',environment.get('ppo_baseline',{}).get('gamma',.995))!=.999
            or environment.get('development_objective')=='avoidance-first-v1'):
        raise ValueError('Keep original private natural full-story reward-v3/source19 environment')
    if 'torch' in sys.modules or os.environ.get('CUDA_VISIBLE_DEVICES') not in (None,'','0'):
        raise RuntimeError('Fresh explicit GPU0 process required')
    import psutil
    current=psutil.Process();native_name=Path(environment['game_exe']).name.lower()
    if [p.info for p in psutil.process_iter(['name']) if (p.info['name'] or '').lower()==native_name]:
        raise RuntimeError('Existing original games;no overlapping native/GPU phase')
    from fly_touhou.owned_worker_affinity import worker_cpu_pool,spread_owned_workers
    allowed=current.cpu_affinity();worker_cpus=worker_cpu_pool(allowed)
    current.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);current.cpu_affinity(allowed[-2:])
    os.environ.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',FLY_TOUHOU_DESKTOP_NAME='fly-touhou-job-risk-path7-'+str(os.getpid()))
    from fly_touhou.thrl_job_paths import configure_native_temp
    from fly_touhou.thrl_durable_records import write_json
    a.output.mkdir(parents=True,exist_ok=False);(a.output/'traces').mkdir();configure_native_temp(a.output)
    files=[Path(__file__),ROOT/'scripts/train_fly_dn_cost_online.py']+[ROOT/'src/fly_touhou'/n for n in (
        'fly_risk_path_pretraining.py','fly_dn_cost_model.py','fly_dn_cost_checkpoint.py',
        'fly_dn_cost_collector.py','fly_dn_cost_labels.py','fly_dn_cost_native_policy.py',
        'fly_gru_risk26_update.py','fly_dn_continuous_risk_policy.py','fly_neural_state_refresh.py',
        'fly_representation_trace.py','fly_population_path_trace.py')]
    launch=dict(plan(a),PID=os.getpid(),create_time=current.create_time(),predecessor_verified=predecessor,
        worker_cpu_pool=worker_cpus,source_sha256s={str(p):sha(p) for p in files},
        environment_config=str(a.environment_config),environment_config_sha256=sha(a.environment_config))
    write_json(a.output/'launch.json',launch)
    started=time.monotonic();heartbeat=[started];torch=model=optimizer=collector=vector=None
    parent=algorithm=inherited=None;paid={'Adam':0,'risk_path':0,'joint':0};active_phase={'name':'risk_path'}
    latest={'phase':'initializing'};bridge=None
    costs=dict(completed_rollouts=0,student_actions_executed=0,risk_path_Adam_steps=0,PPO_updates=0,
        risk_path_completed_rollouts=0,joint_completed_rollouts=0,
        risk_path_student_actions_executed=0,joint_student_actions_executed=0,joint_Adam_steps=0,
        actor_optimizer_updates=0,critic_optimizer_updates=0,backward_passes=0,trained_real_rows=0,loss_real_rows=0,
        training_graph_backend_phase_totals={},backend_and_training_counts_overlap=True)
    training_closed=evaluation_started=evaluation_completed=False;close_errors=[]
    latest_checkpoint=None;last_saved=-1;transaction_complete=True

    def progress(row):
        latest.clear();latest.update(row)
        write_json(a.output/'progress.json',dict(row,schema=SCHEMA,costs=copy.deepcopy(costs),
            elapsed_seconds=time.monotonic()-started,actual_new_Adam_steps=paid['Adam'],PID=os.getpid()))

    def guard():
        reason=('OperatorSTOP' if (a.output/'STOP').exists() else
            'Finite wall budget reached' if time.monotonic()-started>a.max_seconds else
            'Output disk below4GiB' if shutil.disk_usage(a.output).free<4*1024**3 else None)
        if reason:raise RuntimeError(reason)
        if time.monotonic()-heartbeat[0]>30:
            heartbeat[0]=time.monotonic();progress(dict(latest))

    def save(tag,partial=False):
        nonlocal latest_checkpoint,last_saved
        from fly_touhou.fly_dn_cost_checkpoint import training_checkpoint_payload
        counters=dict(copy.deepcopy(costs),Adam_steps=paid['Adam'],partial_rollout_update=partial,
            source_counters=copy.deepcopy(inherited),
            lifetime_completed_rollouts=int(inherited.get('lifetime_completed_rollouts',inherited.get('completed_rollouts',0)))+costs['joint_completed_rollouts'],
            lifetime_completed_rollouts_scope='inherited and new PPO only;risk-path rollouts separately recorded',
            lifetime_Adam_steps=int(inherited.get('lifetime_Adam_steps',inherited.get('Adam_steps',0)))+paid['Adam'],
            lifetime_Adam_steps_scope='recorded optimizer calls across phases;not equal per-parameter steps')
        payload=training_checkpoint_payload(model,optimizer,counters=counters,algorithm=algorithm,parent_provenance=parent)
        payload.update(input_drive_protocol=copy.deepcopy(parent['input_drive_protocol']),
            parent_joint_algorithm=copy.deepcopy(parent['parent_joint_algorithm']))
        for path,value in ((a.output/(tag+'-training.pt'),payload),
                (a.output/(tag+'-parameters.pt'),model.parameter_checkpoint_payload())):
            temporary=path.with_suffix('.pt.tmp')
            with temporary.open('wb') as stream:
                torch.save(value,stream);stream.flush();os.fsync(stream.fileno())
            os.replace(temporary,path)
        latest_checkpoint=str(a.output/(tag+'-training.pt'));last_saved=paid['Adam']

    def add_learning(record):
        for key in ('backward_passes','trained_real_rows','loss_real_rows','actor_replay_neural_lane_forwards',
                'checkpoint_recompute_neural_lane_forwards','risk_head_lane_forwards',
                'post_epoch_diagnostic_neural_lane_forwards','post_epoch_diagnostic_risk_head_lane_forwards',
                'skipped_no_valid_cost_minibatches','skipped_no_valid_cost_rows'):
            costs[key]=costs.get(key,0)+record.get(key,0)
        costs['risk_path_Adam_steps']=paid['risk_path'];costs['joint_Adam_steps']=paid['joint']
        costs['PPO_updates']=paid['joint'];costs['actor_optimizer_updates']=paid['joint'];costs['critic_optimizer_updates']=paid['joint']
        for phase,values in record.get('training_graph',{}).get('backend_cost_delta',{}).items():
            target=costs['training_graph_backend_phase_totals'].setdefault(phase,{})
            for key,value in values.items():
                if type(value) in (int,float):target[key]=target.get(key,0)+value

    def close_training():
        nonlocal training_closed
        if not training_closed:
            training_closed=True
            if vector is not None:
                try:vector.close()
                except BaseException as error:close_errors.append(repr(error))
                close_errors.extend(str(x) for x in vector.close_errors)
        errors=list(close_errors)
        if evaluation_started:
            cleanup=a.output/'development/native_cleanup.json'
            values=read_shared_json(cleanup).get('close_errors') if cleanup.is_file() else None
            if isinstance(values,list):errors.extend(str(x) for x in values)
            else:errors.append('Development cleanup unavailable/unverified')
        write_json(a.output/'training-cleanup.json',dict(close_errors=errors,worlds=30,
            training_closed=training_closed,development_started=evaluation_started,development_completed=evaluation_completed))
        return errors

    try:
        import torch as torch_module
        torch=torch_module;torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
        torch.cuda.init();torch.manual_seed(a.seed)
        from fly_touhou.fly_dn_cost_checkpoint import read_training_source
        from fly_touhou.fly_dn_cost_model import PARAMETER_SCHEMA
        from fly_touhou.fly_gru_aligned_model import FlyGRUAlignedActorCritic
        from fly_touhou.fly_gru_aligned_ppo import aligned_ppo_config
        from fly_touhou.fly_gru_aligned_update import GUIDANCE_CONFIG
        from fly_touhou.fly_input_drive import input_drive_protocol
        from fly_touhou.fly_risk_path_pretraining import update_risk_path_pool
        from fly_touhou.fly_dn_cost_collector import DNCostOnlinePool
        from fly_touhou.fly_d95_reference_refinement import D95RiskPenalty
        from fly_touhou.fly_task_reference_pool import make_task_reference_vector
        from fly_touhou.fly_visual_online_collector_continuation import MovementDiagnosticsVector
        from fly_touhou.thrl_vector import THRLNativeVector
        from train_fly_visual_long_continuous import staggered_reset_vector
        from fly_touhou.whole_brain import CACHED_PROPAGATION
        source=read_training_source(a.training_source,expected_sha256=a.training_source_sha256,source_kind='dn_cost26')
        if source.get('input_drive_protocol')!=input_drive_protocol('poisson'):
            raise ValueError('Explicit unchanged source Poisson required')
        parameters=source['recovery']['model_payload'];source_algorithm=source['recovery']['algorithm']
        expected=joint_algorithm(a,asdict(aligned_ppo_config()),asdict(GUIDANCE_CONFIG),parameters['head_configuration'])
        if source_algorithm!=expected:raise ValueError('Actual joint26/source PPO/guidance/physics protocol differs')
        algorithm=risk_path_algorithm(source_algorithm,a)
        algorithm['input_drive_protocol']=copy.deepcopy(source['input_drive_protocol'])
        base_path=a.output/'source-base-parameters.pt';torch.save(parameters['base_payload'],base_path)
        base=FlyGRUAlignedActorCritic.from_actor_critic_checkpoint(base_path,a.connectome,
            expected_sha256=sha(base_path),device='cpu').to('cuda:0')
        model,optimizer,inherited,transition=restore_risk_path_source(base,source)
        if model.brain.propagation_backend!=CACHED_PROPAGATION:raise ValueError('Unchanged qualified cached-LIF required')
        parent=dict(source=str(a.training_source),source_sha256=a.training_source_sha256,
            source_algorithm=copy.deepcopy(source_algorithm),source_counters=copy.deepcopy(inherited),
            parent_joint_algorithm=copy.deepcopy(source_algorithm),
            source_parent_provenance=copy.deepcopy(source['recovery'].get('parent_provenance')),
            source_owner_records=source_owner,
            predecessor=predecessor,Adam_transition=transition,source_Adam_states_preserved=26,
            input_drive_protocol=copy.deepcopy(source['input_drive_protocol']),
            new_native_games=True,exact_mid_episode_native_resume=False,head_statistics_refitted=False)
        frozen=frozen19_snapshot(model,optimizer)
        write_json(a.output/'initialization.json',dict(parent=parent,frozen19=frozen))
        def paid_hook(*unused):
            paid['Adam']+=1;paid[active_phase['name']]+=1
        optimizer.register_step_post_hook(paid_hook)
        save('initial');del source,parameters,base;gc.collect()
        penalty=D95RiskPenalty.from_checkpoint(a.d95_checkpoint,device=model.device)
        vector=MovementDiagnosticsVector(make_task_reference_vector(environment,a.output/'training',
            vector_class=staggered_reset_vector(THRLNativeVector)))
        write_json(a.output/'worker-affinity-before-reset.json',spread_owned_workers(os.getpid(),current.create_time(),worker_cpus))
        collector=DNCostOnlinePool(model,vector,reference_refiner=penalty,geometry_label_budget=a.geometry_label_budget,
            brain_seed=a.seed,action_seed=a.action_seed)
        write_json(a.output/'sampling-capture.json',collector.enable_sampling_capture())
        costs['sampling_capture_setup_neural_lane_forwards']=4*collector.neural_batch_size
        collector.start();launch.update(status='risk_path_collecting',GPU_or_native_started=True,native_worlds_started=30)
        write_json(a.output/'launch.json',launch)
        population_layout,population_setup=setup_population_diagnostics(model,a.output)
        for version in range(a.decisions//ROLLOUT_ROWS):
            guard();progress(dict(phase='collecting',rollout=version))
            rollout=collector.collect(steps=256,policy_version=version,check_stop=guard)
            costs['student_actions_executed']+=int(rollout.actions.numel())
            costs['risk_path_student_actions_executed']+=int(rollout.actions.numel())
            for key in ('data_neural_lane_forwards','real_neural_lane_forwards','padding_lane_forwards',
                    'bootstrap_neural_lane_forwards','D95_risk_label_query_rows','D95_additional_normal_cost_queries'):
                costs[key]=costs.get(key,0)+rollout.metadata.get(key,0)
            write_rollout_trace(rollout,a.output,version)
            record_after_collect_diagnostics(model,rollout,a.output,version,population_layout,phase='risk_path',costs=costs)
            progress(dict(phase='risk_path_learning',rollout=version));previous=paid['Adam'];transaction_complete=False
            try:
                learning=update_risk_path_pool(model,optimizer,rollout,epochs=a.epochs,
                    minibatch_seed=a.seed+version,cost_config=ContinuousCostConfig(),
                    max_grad_norm=aligned_ppo_config().max_grad_norm,check_stop=guard)
            except BaseException as error:
                record=getattr(error,'risk_path_update_diagnostics',{})
                add_learning(record);transaction_complete=record.get('optimizer_transaction_complete') is True
                raise
            transaction_complete=True;add_learning(learning)
            if learning['optimizer_updates']!=paid['Adam']-previous:raise RuntimeError('Actual restricted Adam ledger differs')
            if frozen19_snapshot(model,optimizer)!=frozen:raise RuntimeError('Frozen19 weights/Adam changed')
            costs['completed_rollouts']=version+1
            costs['risk_path_completed_rollouts']=version+1
            refresh=None
            if a.state_refresh_steps:
                from fly_touhou.fly_neural_state_refresh import refresh_collector_state
                try:refresh=refresh_collector_state(model,collector,rollout,history_steps=a.state_refresh_steps,check_stop=guard)
                except BaseException as error:
                    detail=getattr(error,'neural_state_refresh_diagnostics',{})
                    for key in ('neural_lane_forwards','real_neural_lane_forwards','padding_neural_lane_forwards'):
                        costs['state_refresh_'+key]=costs.get('state_refresh_'+key,0)+detail.get(key,0)
                    raise
                for key in ('neural_lane_forwards','real_neural_lane_forwards','padding_neural_lane_forwards'):
                    costs['state_refresh_'+key]=costs.get('state_refresh_'+key,0)+refresh[key]
            save('last-completed')
            row=dict(phase='risk_path_learning_complete',learning=learning,state_refresh=refresh,
                completed_native_episodes=list(rollout.completed_episodes),frozen19_exact=True,worlds_restarted=False)
            write_json(a.output/'traces'/f'update-{version+1:06}.json',dict(row,costs=copy.deepcopy(costs)))
            progress(row);del rollout,learning;gc.collect()
        save('risk-path7-completed')
        write_json(a.output/'frozen19-preservation.json',dict(all19_weights_and_Adam_exact=True,initial=frozen,
            final=frozen19_snapshot(model,optimizer),risk_path_Adam=paid['Adam'],PPO_updates=0))
        if a.joint_decisions:
            risk_checkpoint=a.output/'risk-path7-completed-training.pt'
            original_joint,transition=begin_joint_phase(model,optimizer,algorithm,collector=collector)
            parent['risk_path7_completed']=dict(checkpoint=str(risk_checkpoint),sha256=sha(risk_checkpoint),
                risk_path_Adam=paid['risk_path'],student_decisions=costs['risk_path_student_actions_executed'],
                frozen19_exact=True,algorithm=copy.deepcopy(algorithm))
            parent['risk_path7_to_joint_transition']=transition
            algorithm=original_joint;active_phase['name']='joint'
            costs['phase_transition_sampling_setup_neural_lane_forwards']=transition['sampling_capture_setup_neural_lane_forwards']
            write_json(a.output/'risk-path7-to-joint.json',dict(transition=transition,parent=parent))
            save('joint-initial')
            from fly_touhou.fly_gru_risk26_update import update_risk26_pool,risk26_tensor_block
            from fly_touhou.fly_gru_risk26_graph import Risk26GraphBridge
            bridge=Risk26GraphBridge(risk26_tensor_block,enabled=a.enable_joint_training_cuda_graph)
            for j in range(a.joint_decisions//ROLLOUT_ROWS):
                version=a.decisions//ROLLOUT_ROWS+j;guard();progress(dict(phase='joint_collecting',rollout=version))
                rollout=collector.collect(steps=256,policy_version=version,check_stop=guard)
                rows=int(rollout.actions.numel());costs['student_actions_executed']+=rows;costs['joint_student_actions_executed']+=rows
                for key in ('data_neural_lane_forwards','real_neural_lane_forwards','padding_lane_forwards',
                        'bootstrap_neural_lane_forwards','D95_risk_label_query_rows','D95_additional_normal_cost_queries'):
                    costs[key]=costs.get(key,0)+rollout.metadata.get(key,0)
                write_rollout_trace(rollout,a.output,version,phase='joint')
                record_after_collect_diagnostics(model,rollout,a.output,version,population_layout,phase='joint',costs=costs)
                progress(dict(phase='joint_learning',rollout=version));previous=paid['Adam'];transaction_complete=False
                try:
                    learning=update_risk26_pool(model,optimizer,rollout,cfg=aligned_ppo_config(),
                        cost_config=ContinuousCostConfig(),graph_bridge=bridge,minibatch_seed=a.seed+version,check_stop=guard)
                except BaseException as error:
                    detail=getattr(error,'ppo_update_diagnostics',{})
                    add_learning(detail);transaction_complete=detail.get('optimizer_transaction_complete') is True
                    raise
                transaction_complete=True;add_learning(learning)
                if learning['optimizer_updates']!=paid['Adam']-previous:raise RuntimeError('Joint actual Adam ledger differs')
                costs['joint_completed_rollouts']=j+1;costs['completed_rollouts']=version+1
                refresh=None
                if a.state_refresh_steps:
                    from fly_touhou.fly_neural_state_refresh import refresh_collector_state
                    try:refresh=refresh_collector_state(model,collector,rollout,history_steps=a.state_refresh_steps,check_stop=guard)
                    except BaseException as error:
                        detail=getattr(error,'neural_state_refresh_diagnostics',{})
                        for key in ('neural_lane_forwards','real_neural_lane_forwards','padding_neural_lane_forwards'):
                            costs['state_refresh_'+key]=costs.get('state_refresh_'+key,0)+detail.get(key,0)
                        raise
                    for key in ('neural_lane_forwards','real_neural_lane_forwards','padding_neural_lane_forwards'):
                        costs['state_refresh_'+key]=costs.get('state_refresh_'+key,0)+refresh[key]
                save('last-completed')
                row=dict(phase='joint_learning_complete',learning=learning,state_refresh=refresh,
                    completed_native_episodes=list(rollout.completed_episodes),worlds_restarted=False)
                write_json(a.output/'traces'/f'update-{version+1:06}.json',dict(row,costs=copy.deepcopy(costs)))
                progress(row);del rollout,learning;gc.collect()
            bridge.close();bridge=None
        save('final')
        if close_training():raise RuntimeError('Collection cleanup incomplete')
        collector=vector=None;gc.collect();model.eval().requires_grad_(False)
        from fly_touhou.fly_dn_cost_native_policy import evaluate_dn_cost_native_pool
        evaluation_started=True;progress(dict(phase='frozen_development'))
        parameter_path=a.output/'final-parameters.pt';parameter_sha=sha(parameter_path)
        versions=tuple(p._version for p in model.parameters())
        evaluation=evaluate_dn_cost_native_pool(model,environment,a.output/'development',
            source_schema=PARAMETER_SCHEMA,checkpoint_sha256=parameter_sha,episodes=20,native_workers=20,
            seed=2027431601,brain_seed=a.seed,check_stop=guard,worker_cpus=worker_cpus,
            progress=lambda r:progress(dict(phase='frozen_development',evaluation=r)))
        evaluation_completed=True
        if versions!=tuple(p._version for p in model.parameters()):raise RuntimeError('Frozen20 changed parameters')
        if close_training():raise RuntimeError('Frozen20 cleanup incomplete')
        write_json(a.output/'result.json',dict(schema=SCHEMA,status='completed',costs=costs,source=parent,
            final_training_checkpoint=str(a.output/'final-training.pt'),final_parameter_checkpoint=str(parameter_path),
            final_parameter_sha256=parameter_sha,development=evaluation,frozen_actor_critic19_exact_during_risk_path=True,
            frozen19_scope='risk-path7 phase only;joint phase explicitly trains all26',
            optional_joint_phase_executed=bool(a.joint_decisions),
            elapsed_seconds=time.monotonic()-started,formal_stability_test=False))
    except BaseException as error:
        checkpoint_error=None
        if model is not None and algorithm is not None and parent is not None and transaction_complete and paid['Adam']>last_saved:
            try:save('stopped-after-update',partial=True)
            except BaseException as second:checkpoint_error=repr(second)
        write_json(a.output/'failure.json',dict(schema=SCHEMA,error=repr(error),costs=costs,
            actual_new_Adam_steps=paid['Adam'],latest_recoverable_training_checkpoint=latest_checkpoint,
            optimizer_transaction_complete=transaction_complete,checkpoint_error=checkpoint_error,
            update_diagnostics=getattr(error,'risk_path_update_diagnostics',getattr(error,'ppo_update_diagnostics',None)),no_automatic_retry=True,
            acknowledged_native_actions=costs['student_actions_executed'] if collector is None else collector.acknowledged_actions))
        raise
    finally:
        try:
            if bridge is not None:bridge.close()
        finally:
            try:close_training()
            finally:
                if model is not None:model.brain.clear_cuda_capture();model.brain.clear_cached_propagation()


def main(argv=None):
    args=parse_args(argv)
    if not args.execute_cuda:print(json.dumps(plan(args),ensure_ascii=False));return
    execute(args)


if __name__=='__main__':main()
