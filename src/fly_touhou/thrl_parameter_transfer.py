"""Explicit evaluation-policy parameter transfer into a fresh PPO experiment.

No BC, numerical resume, optimizer/RNG inheritance, world or optimization step.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

TRANSFER_SCHEMA = 'B1-evaluation-policy-fresh-Adam-parameter-transfer-v1'
SOURCE_SCHEMA = 'B1-risk-calibrated-frozen-candidate-v1'
INPUT_SCHEMA = 'semantic-separated-observed-laser-segments-v5'
EXPERIMENT_SCHEMA = 'B1-evaluation-parameter-transfer-finite-training-v1'


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def configuration_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def validate_target_parameters(source,target):
    before=json.loads(json.dumps(source,allow_nan=False))
    after=json.loads(json.dumps(target,allow_nan=False))
    before.setdefault('action_temperature',1.)
    if set(before)!=set(after) or any(before[key]!=after[key] for key in before if key!='action_temperature'):
        raise ValueError('This experiment may change action temperature only;reward gamma and PPO weights remain fixed')
    temperature=after['action_temperature']
    if isinstance(temperature,bool) or not isinstance(temperature,(int,float)) or not 0<temperature<100:
        raise ValueError('Explicit positive action temperature required')
    return dict(before=before['action_temperature'],after=temperature,
                experiment_not_optimality_claim=True,source_rollout_or_logp_reused=False)


def validate_finite_config(config):
    from .thrl_training_mode import validate_training_mode
    from .thrl_branching import validate_parallel_config
    forbidden=('baseline_resume','parent_PPO_policy','teacher_pretraining_result','full_teacher_actor_transfer',
               'warm_start_checkpoint','warm_start_sha256','development_objective','initial_avoidance_development',
               'candidate_initial_checkpoint','candidate_initial_sha256','candidate_decision_offset',
               'candidate_paid_decision_offset','inherited_unoptimized_sampling_lower_bound')
    if any(key in config for key in forbidden) or any(key.startswith('curriculum_') for key in config):
        raise ValueError('Use a clean parameter-transfer experiment,without BC/resume/offset/avoidance history fields')
    if (config.get('schema')!=EXPERIMENT_SCHEMA or config.get('branch')!='B1' or config.get('workers')!=30
            or config.get('training_mode')!='full-story-v1' or config.get('input_schema')!=INPUT_SCHEMA
            or config.get('policy_action_count')!=19 or config.get('headless') is not True
            or config.get('autonomous_bomb') is not True
            or config.get('rollout_total_decisions')!=7680
            or type(config.get('trial_decisions')) is not int or config['trial_decisions']<=0
            or config.get('development_decisions')!=[config['trial_decisions']]
            or config.get('evaluation_after_training_shutdown') is not True
            or config.get('development_episodes')!=20
            or config.get('full_story_evaluation_backend')!='full-story-ready-parallel-v1'
            or config.get('full_story_evaluation_workers')!=20
            or config.get('full_story_evaluation_action_selection')!='greedy'
            or config.get('training_recipe')!='th14-completion-v3'
            or config.get('reward_schema')!='th14-completion-reward-v3'
            or config.get('teacher_pretraining',False) or config.get('reset_Adam_after_BC',False)):
        raise ValueError('Require one fresh B1/source19 natural30-world run and final20-game greedy full-story development')
    validate_training_mode(config)
    capacity=validate_parallel_config(config)
    validate_model_configuration(config['policy_model_config'])
    origin=config.get('parameter_transfer') or {}
    if not origin.get('source_checkpoint') or not origin.get('source_sha256'):
        raise ValueError('An explicit evaluation checkpoint parameter source is required')
    return capacity


def restricted_load(torch,path):
    # The pinned evaluation file stores a TorchVersion string subclass.
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        return torch.load(path,map_location='cpu',weights_only=True)


def validate_model_configuration(configuration):
    wrapper=configuration.get('pair_risk_wrapper',{})
    base=wrapper.get('base_model_config',{})
    if (configuration.get('input_schema')!=INPUT_SCHEMA or configuration.get('temporal_core')!='gru'
            or base.get('input_schema')!=INPUT_SCHEMA or base.get('feature_dim')!=25
            or base.get('map_channels')!=16 or base.get('action_dim')!=19
            or base.get('gru_hidden_size')!=256 or base.get('num_objectives')!=3
            or wrapper.get('frozen_readout') is not True):
        raise ValueError('Parameter transfer requires the exact V5 B1/source19 recurrent risk-policy architecture')


def read_evaluation_source(torch,path,expected_sha256):
    path=Path(path)
    if file_sha(path)!=expected_sha256:
        raise ValueError('Evaluation parameter source SHA256 differs')
    source=restricted_load(torch,path)
    if (source.get('schema')!=SOURCE_SCHEMA or source.get('evaluation_only') is not True
            or source.get('optimizer_state_valid_for_resume') is not False
            or any(key in source for key in ('optimizer','torch_rng','cuda_rng','numpy_rng','python_rng'))):
        raise ValueError('Expected an evaluation-only policy without resumable optimizer/RNG state')
    validate_model_configuration(source['model_config'])
    if not source.get('model'):
        raise ValueError('Source contains no model state')
    for name,value in source['model'].items():
        if not torch.is_tensor(value) or not bool(torch.isfinite(value).all()):
            raise ValueError('Invalid evaluation source tensor: '+name)
    return source


def _validate_exact_state(torch,model,configuration,state):
    validate_model_configuration(configuration)
    if model.model_config()!=configuration or model.input_schema!=INPUT_SCHEMA:
        raise ValueError('Parameter transfer cannot change the declared model/input configuration')
    current=model.state_dict()
    if set(current)!=set(state):
        raise ValueError('Parameter transfer requires all tensors and buffers, without additions or omissions')
    for name,value in state.items():
        if (not torch.is_tensor(value) or value.shape!=current[name].shape
                or value.dtype!=current[name].dtype or not bool(torch.isfinite(value).all())):
            raise ValueError('Parameter transfer shape/dtype/finiteness differs: '+name)
    return current


def source_historical_metadata(source):
    """Retain checkpoint-local counters without inventing full paid history."""
    return dict(source_checkpoint_local_counters={key:copy.deepcopy(source.get(key)) for key in (
        'model_decisions_inherited','risk_fit_Adam_steps','PPO_optimizer_updates')},
        scope='as stored in the selected source checkpoint;not complete source candidate or lifetime paid totals',
        risk_fit_counter_semantics='selected model fit steps;not all paid calibration steps',
        complete_source_candidate_or_lifetime_costs_reconstructed=False,
        source_parent=copy.deepcopy(source.get('parent')),source_parent_sha256=source.get('parent_sha256'),
        inherited_as_new_run_progress=False)


def build_parameter_transfer(torch,model,source_checkpoint,expected_sha256,target_parameters,*,seed):
    """Build a distinct, optimizer-free artifact. This performs no update."""
    from .thrl_study_io import utc_now
    source=read_evaluation_source(torch,source_checkpoint,expected_sha256)
    temperature_change=validate_target_parameters(source.get('config',{}).get('parameters',{}),target_parameters)
    _validate_exact_state(torch,model,source['model_config'],source['model'])
    temperature=target_parameters.get('action_temperature',1.)
    if (isinstance(temperature,bool) or not isinstance(temperature,(int,float))
            or not 0<temperature<100 or type(seed) is not int or not 0<=seed<2**32):
        raise ValueError('An explicit positive target temperature and fresh uint32 sampling seed are required')
    model.load_state_dict(source['model'],strict=True)
    if not all(torch.equal(model.state_dict()[name].detach().cpu(),value.cpu()) for name,value in source['model'].items()):
        raise ValueError('Source model tensors were not copied exactly')
    history=source_historical_metadata(source)
    return dict(schema=TRANSFER_SCHEMA,created_utc=utc_now(),
        source_checkpoint=str(Path(source_checkpoint).resolve()),source_sha256=expected_sha256,
        source_checkpoint_schema=SOURCE_SCHEMA,model_config=copy.deepcopy(source['model_config']),
        model={key:value.detach().cpu().clone() for key,value in model.state_dict().items()},
        model_configuration_sha256=configuration_sha(source['model_config']),
        target_parameters=copy.deepcopy(target_parameters),fresh_sampling_seed=seed,
        source_action_temperature=source.get('config',{}).get('parameters',{}).get('action_temperature',1.),
        target_action_temperature=float(temperature),source_historical_work=history,
        temperature_change=temperature_change,
        optimizer_inherited=False,sampling_RNG_inherited=False,numerical_PPO_resume=False,BC_transfer=False,
        candidate_decisions=0,candidate_optimized_decisions=0,candidate_paid_decisions_lower_bound=0,
        native_world_state_inherited=False,recurrent_state='zero in new natural full-story worlds',
        initialization_semantics='all evaluation model parameters copied exactly;new empty target Adam and fresh sampling stream')


def apply_parameter_transfer(torch,model,optimizer,payload,*,checkpoint,checkpoint_sha256,exact_optimizer=False):
    """Used only by the explicit transfer-schema branch of warm_start."""
    if exact_optimizer:
        raise ValueError('Parameter-only transfer is never exact numerical PPO recovery')
    if (payload.get('schema')!=TRANSFER_SCHEMA or payload.get('optimizer_inherited') is not False
            or payload.get('sampling_RNG_inherited') is not False or payload.get('numerical_PPO_resume') is not False
            or payload.get('BC_transfer') is not False
            or any(key in payload for key in ('optimizer','torch_rng','cuda_rng','numpy_rng','python_rng'))
            or any(payload.get(key)!=0 for key in ('candidate_decisions','candidate_optimized_decisions','candidate_paid_decisions_lower_bound'))):
        raise ValueError('Explicit parameter transfer cannot contain optimizer/RNG/resume/BC progress')
    if payload.get('model_configuration_sha256')!=configuration_sha(payload['model_config']):
        raise ValueError('Transfer model configuration digest differs')
    source=read_evaluation_source(torch,payload['source_checkpoint'],payload['source_sha256'])
    validate_target_parameters(source.get('config',{}).get('parameters',{}),payload['target_parameters'])
    expected_history=source_historical_metadata(source)
    if (payload.get('source_action_temperature')!=source.get('config',{}).get('parameters',{}).get('action_temperature',1.)
            or payload.get('target_action_temperature')!=payload['target_parameters'].get('action_temperature',1.)
            or payload.get('source_historical_work')!=expected_history):
        raise ValueError('Transfer temperature/history metadata differs from its actual source')
    _validate_exact_state(torch,model,payload['model_config'],payload['model'])
    if (source['model_config']!=payload['model_config'] or set(source['model'])!=set(payload['model'])
            or any(not torch.equal(source['model'][name].cpu(),value.cpu()) for name,value in payload['model'].items())):
        raise ValueError('Transfer artifact altered an evaluation parameter or buffer')
    if not isinstance(optimizer,torch.optim.Adam):
        raise ValueError('The target experiment requires a newly created Adam')
    if (not getattr(model,'frozen_readout',False) or not hasattr(model,'base')
            or any(not value.requires_grad for value in model.base.parameters())
            or any(value.requires_grad for value in model.risk.parameters())
            or (getattr(model,'laser_risk',None) is not None
                and any(value.requires_grad for value in model.laser_risk.parameters()))):
        raise ValueError('Train the full shared B1 base while retaining its frozen point/laser risk readouts')
    target_state=optimizer.state_dict()
    groups=target_state['param_groups']
    if target_state['state'] or len(groups)!=1 or len(groups[0]['params'])!=len(list(model.parameters())):
        raise ValueError('Target Adam must be fresh, empty, and cover every model parameter')
    if groups[0]['lr']!=payload['target_parameters']['learning_rate']:
        raise ValueError('Target fresh Adam learning rate differs from the explicit transfer contract')
    if (groups[0]['eps']!=1e-8 or tuple(groups[0]['betas'])!=(.9,.999)
            or groups[0]['weight_decay']!=0 or groups[0]['amsgrad']):
        raise ValueError('Fresh Adam must retain the existing native PPO optimizer defaults')
    if ([id(value) for value in optimizer.param_groups[0]['params']]
            !=[id(value) for value in model.parameters()]):
        raise ValueError('Fresh Adam parameter order differs from the target policy')
    model.load_state_dict(payload['model'],strict=True)
    exact=all(torch.equal(model.state_dict()[name].detach().cpu(),value.cpu()) for name,value in source['model'].items())
    if not exact or optimizer.state_dict()['state']:
        raise ValueError('Fresh parameter transfer did not retain exact parameters and empty Adam')
    # No model construction, RNG reset, source moment loading, or update here.
    return dict(schema='B1-evaluation-fresh-Adam-transfer-audit-v1',
        checkpoint=str(Path(checkpoint).resolve()),sha256=checkpoint_sha256,
        evaluation_source_checkpoint=payload['source_checkpoint'],evaluation_source_sha256=payload['source_sha256'],
        common_weights_exact=True,common_parameter_tensors=len(list(model.parameters())),
        all_state_tensors_exact=True,all_state_tensor_count=len(source['model']),
        trainable_parameter_count=sum(value.numel() for value in model.parameters() if value.requires_grad),
        common_Adam_states=0,optimizer_moments_preserved=False,sampling_RNG_restored=False,
        target_Adam_state_entries=0,PPO_optimizer_initialization='fresh empty Adam;no inherited moments',
        inherited_decisions=0,inherited_rollout_updates=0,inherited_adam_minibatch_steps=0,
        branch_decisions_restart_at_zero=True,exact_optimizer_recovery=False,BC_transfer=False,
        numerical_PPO_resume=False,source_action_temperature=payload['source_action_temperature'],
        target_action_temperature=payload['target_action_temperature'],fresh_sampling_seed=payload['fresh_sampling_seed'],
        source_historical_work=payload['source_historical_work'],
        world_state_restored=False,recurrent_state='zero in fresh natural full-story worlds')
