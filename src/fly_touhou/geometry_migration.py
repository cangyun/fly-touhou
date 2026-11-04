"""Explicit PPO learner extension with old tensors and Adam moments preserved."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import torch

from .geometry_observation import GeometryObservationEncoder
from .hazard_observation import HazardObservationEncoder
from .whole_brain_learning import CHECKPOINT_SCHEMA
from .whole_brain_ppo import PPO_SCHEMA, GEOMETRY_ACTOR_SUFFIX

TRANSITION_SCHEMA='explicit-native-geometry-observation-transition-v1'
PROOF_SCHEMA='full-brain-native-geometry-extension-proof-v1'
PROOF_BINDING_SCHEMA='verified-native-geometry-extension-binding-v1'
REQUIRED_PROOF_CHECKS=('frozen_parent_forward_state_bitwise_equal','critic_zero_extension_bitwise_equal',
    'original_Adam_state_preserved','new_geometry_gradient_nonzero',
    'captured_objective_and_gradients_equal','continuing_Adam_and_parameters_equal',
    'full_source_graph_identity','native_geometry_inputs_validated')
PROOF_SOURCE_FILES=('src/fly_touhou/whole_brain.py','src/fly_touhou/whole_brain_learning.py',
    'src/fly_touhou/whole_brain_ppo.py','src/fly_touhou/ppo_cuda_capture.py',
    'src/fly_touhou/brain_cuda_capture.py','src/fly_touhou/geometry_observation.py',
    'src/fly_touhou/geometry_migration.py','src/fly_touhou/native_geometry.py',
    'src/fly_touhou/hazard_reader.py','src/fly_touhou/geometry_environment.py',
    'src/fly_touhou/campaign_rewards.py','src/fly_touhou/geometry_action_feedback.py',
    'src/fly_touhou/resource_teacher.py','src/fly_touhou/native_teacher_feedback.py',
    'src/fly_touhou/resource_teacher_v2.py','src/fly_touhou/resource_teacher_v5.py',
    'src/fly_touhou/resource_teacher_v6.py','src/fly_touhou/native_hazard_planner.py',
    'src/fly_touhou/dynamic_planner.py','src/fly_touhou/planner.py',
    'src/fly_touhou/state.py','src/fly_touhou/hazard_observation.py',
    'src/fly_touhou/serial_gradient_accumulation.py','src/fly_touhou/critical_full_rollout.py')


def _transition(checkpoint):
    return checkpoint.get('geometry_observation_transition',checkpoint.get('training_config',{}).get('geometry_observation_transition'))


def _check_proof(checkpoint,proof):
    transition=_transition(checkpoint)
    if (not isinstance(transition,dict) or transition.get('schema')!=TRANSITION_SCHEMA
            or proof.get('schema')!=PROOF_SCHEMA or proof.get('status')!='passed'
            or proof.get('lineage_id')!=transition.get('lineage_id')
            or proof.get('model_config')!=checkpoint['model_config']
            or proof.get('encoder')!=checkpoint['encoder']
            or proof.get('graph_fingerprint')!=checkpoint['graph_fingerprint']
            or any(proof.get('checks',{}).get(k) is not True for k in REQUIRED_PROOF_CHECKS)):
        raise ValueError('Complete CUDA geometry extension proof does not match this learner lineage')
    feedback=checkpoint.get('training_config',{}).get('action_feedback')
    expert=checkpoint.get('training_config',{}).get('teacher_feedback')
    if feedback is not None:
        required_auxiliary=['training_action_targets_verified','auxiliary_gradient_through_full_brain_nonzero',
            'auxiliary_loss_and_continuing_Adam_equal']
        if feedback.get('schema')=='native-geometry-action-set-v2':
            required_auxiliary.append('Bomb_gate_entropy_not_rewarded')
        transition=checkpoint.get('action_feedback_transition',checkpoint.get('training_config',{}).get('action_feedback_transition',{}))
        if (proof.get('action_feedback')!=feedback
                or transition.get('schema')!='explicit-native-action-feedback-transition-v1'
                or (transition!=proof.get('inherited_action_feedback_transition') if expert is not None
                    else transition.get('parent_checkpoint_sha256')!=proof.get('learning_checkpoint_sha256'))
                or any(proof.get('checks',{}).get(k) is not True for k in required_auxiliary)):
            raise ValueError('Complete action-set objective proof and exact learner ancestry are required')
    if expert is not None:
        config=checkpoint['training_config'];transition=config.get('teacher_feedback_transition',{})
        if (feedback is None or expert!=proof.get('teacher_feedback') or config.get('teacher_reference')!=proof.get('teacher_reference')
                or transition.get('schema')!='explicit-native-v6-teacher-feedback-transition-v1'
                or transition.get('parent_checkpoint_sha256')!=proof.get('learning_checkpoint_sha256')
                or any(proof.get('checks',{}).get(k) is not True for k in (
                    'teacher_targets_verified_from_measured_states','teacher_gradient_through_full_brain_nonzero',
                    'teacher_combined_loss_and_continuing_Adam_equal'))):
            raise ValueError('Complete V6 teacher objective proof and exact learner ancestry required')
    critical=checkpoint.get('training_config',{}).get('critical_full_rollout_first_step')
    if critical is not None:
        from .critical_full_rollout import specification,CRITICAL_TRANSITION_SCHEMA
        transition=checkpoint['training_config'].get('critical_first_step_transition',{})
        if (expert is None or critical!=specification() or critical!=proof.get('critical_full_rollout_first_step')
                or transition.get('schema')!=CRITICAL_TRANSITION_SCHEMA
                or transition.get('parent_checkpoint_sha256')!=proof.get('learning_checkpoint_sha256')
                or any(proof.get('checks',{}).get(k) is not True for k in (
                    'critical_and_all_controls_share_first_optimizer_step',
                    'critical_first_step_ordinary_captured_continuing_Adam_equal'))):
            raise ValueError('Complete critical first-step proof and exact learner ancestry required')
    root=Path(__file__).resolve().parents[2]
    sources=proof.get('source_files',{})
    if set(sources)!=set(PROOF_SOURCE_FILES):
        raise ValueError('Incomplete geometry proof source identity')
    for relative,digest in sources.items():
        path=(root/relative).resolve()
        if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError('Geometry proof code differs: '+relative)


def validate_geometry_deployment(checkpoint):
    if not checkpoint.get('model_config',{}).get('geometry_prefix_dim'):
        return
    binding=checkpoint.get('geometry_validation_proof',checkpoint.get('training_config',{}).get('geometry_validation_proof'))
    if not isinstance(binding,dict) or binding.get('schema')!=PROOF_BINDING_SCHEMA:
        raise ValueError('Unproved geometry migration cannot be deployed; bind the complete CUDA proof first')
    path=Path(binding['path'])
    content=path.read_bytes()
    if hashlib.sha256(content).hexdigest()!=binding.get('sha256'):
        raise ValueError('Sealed geometry proof file changed')
    _check_proof(checkpoint,json.loads(content))


def bind_geometry_checkpoint(checkpoint_path,proof_path):
    checkpoint_path,proof_path=Path(checkpoint_path),Path(proof_path)
    content=checkpoint_path.read_bytes();proof_content=proof_path.read_bytes()
    proof=json.loads(proof_content)
    if proof.get('migrated_checkpoint_sha256')!=hashlib.sha256(content).hexdigest():
        raise ValueError('Proof was not made against this exact initial geometry checkpoint')
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    _check_proof(checkpoint,proof)
    binding={'schema':PROOF_BINDING_SCHEMA,'path':str(proof_path.resolve()),
             'sha256':hashlib.sha256(proof_content).hexdigest(),'lineage_id':proof['lineage_id']}
    checkpoint['geometry_validation_proof']=binding
    checkpoint['training_config']['geometry_validation_proof']=binding
    validate_geometry_deployment(checkpoint)
    return checkpoint


def verify_previous_geometry(checkpoint,previous_source):
    """Run the pinned old validator only after checking its sealed identity."""
    source=Path(previous_source).resolve(strict=True)
    binding=checkpoint.get('geometry_validation_proof',checkpoint.get('training_config',{}).get('geometry_validation_proof'))
    if not isinstance(binding,dict) or binding.get('schema')!=PROOF_BINDING_SCHEMA:
        raise ValueError('An already qualified geometry parent is required')
    content=Path(binding['path']).read_bytes()
    if hashlib.sha256(content).hexdigest()!=binding.get('sha256'):
        raise ValueError('Previous proof changed')
    proof=json.loads(content)
    # Do not execute a substituted old validator based only on a rewritten
    # source manifest. Its bytes must match the proof pinned in the parent.
    relative='src/fly_touhou/geometry_migration.py'
    path=source/relative
    if hashlib.sha256(path.read_bytes()).hexdigest()!=proof.get('source_files',{}).get(relative):
        raise ValueError('Previous validator is not the one sealed in the parent proof')
    manifest_content=(source/'snapshot_manifest.json').read_bytes()
    manifest=json.loads(manifest_content)
    if manifest.get('schema')!='fly-touhou-frozen-source-v1':raise ValueError('Missing old source manifest')
    for row in manifest['files']:
        file=(source/row['path']).resolve()
        if not file.is_relative_to(source) or hashlib.sha256(file.read_bytes()).hexdigest()!=row['sha256']:
            raise ValueError('Previous frozen source changed: '+row['path'])
    name='fly_touhou._verified_previous_geometry_'+binding['sha256'][:16]
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    module.validate_geometry_deployment(checkpoint)
    return {'proof':proof,'binding':binding,'source':str(source),
            'manifest_sha256':hashlib.sha256(manifest_content).hexdigest()}


def tensor_digest(value):
    value=value.detach().cpu().contiguous()
    h=hashlib.sha256(str((str(value.dtype),tuple(value.shape))).encode())
    h.update(value.numpy().tobytes())
    return h.hexdigest()


def migrate_geometry_checkpoint(parent, *, parent_sha256, parent_path):
    if (parent.get('schema')!=CHECKPOINT_SCHEMA or parent.get('fixed_weights_stored') is not False
            or parent.get('neurons')!=138639 or parent.get('edge_records')!=15091983
            or parent.get('model_config',{}).get('require_full_graph') is not True
            or parent['model_config'].get('n_actions')!=36):
        raise ValueError('Geometry extension requires an intact complete-brain PPO parent')
    if (not isinstance(parent_sha256,str) or len(parent_sha256)!=64
            or any(c not in '0123456789abcdef' for c in parent_sha256)):
        raise ValueError('An exact immutable parent SHA256 is required')
    old_encoder=HazardObservationEncoder.from_specification(parent['encoder'])
    if (parent['model_config']['obs_dim']!=old_encoder.size
            or parent['model_config'].get('geometry_prefix_dim',0)
            or 'geometry_observation_transition' in parent.get('training_config',{})):
        raise ValueError('Geometry migration is a one-time explicit V3-to-V4 extension')
    old_rl=parent.get('rl_state',{})
    if old_rl.get('schema')!=PPO_SCHEMA:
        raise ValueError('Full continuing PPO critic/Adam/RNG state is required')
    for field in ('completed_timesteps','native_frames','updates','episodes_started',
                  'action_sampler_state','shuffle_state','critic_hidden_sizes','parallel_sampler_state'):
        if field not in old_rl:
            raise ValueError('Missing continuing learner state: '+field)
    if (parent['training_config'].get('model_config')!=parent['model_config']
            or parent['training_config'].get('encoder')!=parent['encoder']):
        raise ValueError('Parent actor identity differs between checkpoint and PPO configuration')
    old_actor=parent['trainable_parameters'];old_critic=old_rl['critic_state_dict']
    if 'geometry_input_encoder.weight' in old_actor or 'geometry_input_encoder.weight' in old_critic:
        raise ValueError('Refusing to replace an existing geometry projection')
    if old_actor['input_encoder.weight'].shape[1]!=old_encoder.size or old_critic['network.0.weight'].shape[1]!=old_encoder.size:
        raise ValueError('Original actor and critic prefix dimensions differ')
    for value in (*old_actor.values(),*old_critic.values()):
        if value.dtype!=torch.float32 or not bool(torch.isfinite(value).all()):
            raise ValueError('Nonfinite or non-FP32 original learned parameter')
    encoder=GeometryObservationEncoder(prefix=old_encoder)
    extra=encoder.size-old_encoder.size
    result=deepcopy(parent)
    result['encoder']=encoder.specification()
    result['model_config'].update(obs_dim=encoder.size,geometry_prefix_dim=old_encoder.size)
    actor=result['trainable_parameters'];critic=result['rl_state']['critic_state_dict']
    actor['geometry_input_encoder.weight']=torch.zeros(old_actor['input_encoder.weight'].shape[0],extra)
    critic['geometry_input_encoder.weight']=torch.zeros(old_critic['network.0.weight'].shape[0],extra)
    old_names=[('actor',n) for n in old_actor]+[('critic',n) for n in old_critic]
    new_names=[('actor',n) for n in actor]+[('critic',n) for n in critic]
    old_optimizer=old_rl['optimizer_state_dict']
    if len(old_optimizer.get('param_groups',[]))!=1:
        raise ValueError('The original one-group continuing Adam contract is required')
    old_ids=old_optimizer['param_groups'][0]['params']
    if (len(old_ids)!=len(old_names) or len(set(old_ids))!=len(old_ids)
            or set(old_optimizer['state'])!=set(old_ids)):
        raise ValueError('Cannot silently invent or discard original Adam parameter histories')
    old_id_by_name=dict(zip(old_names,old_ids))
    new_optimizer={'param_groups':deepcopy(old_optimizer['param_groups']),'state':{}}
    new_optimizer['param_groups'][0]['params']=list(range(len(new_names)))
    mapping={};unchanged={}
    for new_id,key in enumerate(new_names):
        role,name=key
        parameter=(actor if role=='actor' else critic)[name]
        if key in old_id_by_name:
            old_id=old_id_by_name[key]
            state=deepcopy(old_optimizer['state'][old_id])
            for moment in ('exp_avg','exp_avg_sq'):
                if (moment not in state or state[moment].shape!=parameter.shape
                        or not bool(torch.isfinite(state[moment]).all())):
                    raise ValueError('Incomplete original Adam moments: '+role+'.'+name)
            if ('step' not in state or not torch.is_tensor(state['step']) or state['step'].numel()!=1
                    or not bool(torch.isfinite(state['step'])) or float(state['step'])<0):
                raise ValueError('Invalid original Adam step: '+role+'.'+name)
            mapping[role+'.'+name]={'old_id':old_id,'new_id':new_id}
            original=(old_actor if role=='actor' else old_critic)[name]
            assert torch.equal(original,parameter)
            unchanged[role+'.'+name]=tensor_digest(parameter)
        else:
            prototype=next(iter(old_optimizer['state'].values()))['step']
            state={'step':torch.zeros_like(prototype),'exp_avg':torch.zeros_like(parameter),
                   'exp_avg_sq':torch.zeros_like(parameter)}
            if old_optimizer['param_groups'][0].get('amsgrad',False):
                state['max_exp_avg_sq']=torch.zeros_like(parameter)
        new_optimizer['state'][new_id]=state
    result['rl_state']['optimizer_state_dict']=new_optimizer
    lineage={'parent_checkpoint_sha256':parent_sha256,'parent_model':parent['model_config'],
             'model':result['model_config'],'encoder':result['encoder']}
    lineage_id=hashlib.sha256(json.dumps(lineage,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    transition={'schema':TRANSITION_SCHEMA,'lineage_id':lineage_id,
        'parent_checkpoint_path':str(parent_path),'parent_checkpoint_sha256':parent_sha256,
        'from_schema':parent['encoder']['schema'],'to_schema':result['encoder']['schema'],
        'prefix_size':old_encoder.size,'new_size':encoder.size,
        'new_parameters':['actor.geometry_input_encoder.weight','critic.geometry_input_encoder.weight'],
        'initialization':'exact zero additional sensory/critic projections; original matrix dimensions retained',
        'original_parameters_and_Adam_moments_preserved':True,
        'existing_Adam_steps_preserved':True,'new_parameter_Adam_steps':0,
        'native_counters_and_all_sampler_states_preserved':True,
        'learning_performed_in_migration':False,
        'required_validation':'full-size CUDA old/new zero-extension state/action equality and continuing optimizer/captured-loss proof before deployment'}
    result['geometry_observation_transition']=transition
    cfg=result['training_config'];cfg.update(model_config=deepcopy(result['model_config']),
        encoder=deepcopy(result['encoder']),geometry_observation_transition=transition)
    cfg['actor_parameters']+=GEOMETRY_ACTOR_SUFFIX
    result['metrics']={**result.get('metrics',{}),'parameter_migration_only':True,
                       'learning_performed_in_migration':False}
    report={'schema':TRANSITION_SCHEMA,'lineage_id':lineage_id,'status':'migrated-unproved',
        'transition':transition,'original_parameter_sha256':unchanged,'Adam_parameter_id_mapping':mapping,
        'new_projection_shapes':{'actor':list(actor['geometry_input_encoder.weight'].shape),
                                 'critic':list(critic['geometry_input_encoder.weight'].shape)},
        'graph_fingerprint_unchanged':result['graph_fingerprint']==parent['graph_fingerprint'],
        'full_graph_runtime_revalidated_at_deployment':False}
    return result,report
