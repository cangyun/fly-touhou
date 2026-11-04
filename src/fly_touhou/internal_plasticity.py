"""Explicit identity initialization for a new internal-gain PPO experiment."""
from copy import deepcopy
import math

import torch

from .whole_brain import EXPECTED_EDGE_RECORDS
from .connectome import EXPECTED_NEURONS
from .whole_brain_learning import CHECKPOINT_SCHEMA


def enable_bounded_recurrent_gain(parent: dict, *, limit: float, parent_sha256: str) -> dict:
    """Add identity gains without treating migration as training or dropping RL.

    The caller validates the full graph/executable/ports with the ordinary
    deployment loader before invoking this function. PPO always starts a new
    optimizer after distillation; existing PPO learning must not be discarded.
    """
    if (isinstance(limit, bool) or not math.isfinite(limit) or not 0 < limit < 1):
        raise ValueError("Internal gain limit must be finite in (0,1)")
    if (parent.get("schema") != CHECKPOINT_SCHEMA or parent.get("fixed_weights_stored") is not False
            or parent.get("neurons") != EXPECTED_NEURONS
            or parent.get("edge_records") != EXPECTED_EDGE_RECORDS):
        raise ValueError("Internal plasticity requires a complete pinned-brain parent")
    if "rl_state" in parent:
        raise ValueError("Use a distilled parent; migration may not discard learned PPO critic/Adam/RNGs")
    if parent["model_config"].get("recurrent_gain_limit", 0) != 0:
        raise ValueError("Parent already has internal plasticity; do not reset learned gains")
    parameters = parent.get("trainable_parameters", {})
    if set(parameters) != {"input_encoder.weight", "input_encoder.bias", "action_readout.weight"}:
        raise ValueError("Unexpected parent actor parameters")
    if (len(parent_sha256) != 64 or any(c not in "0123456789abcdef" for c in parent_sha256)):
        raise ValueError("Parent SHA256 must be explicit")
    result = deepcopy(parent)
    result["model_config"]["recurrent_gain_limit"] = float(limit)
    result["trainable_parameters"]["recurrent_gain_logits"] = torch.zeros(EXPECTED_NEURONS, dtype=torch.float32)
    # Adam groups have changed; the old file remains the complete optimizer
    # record. This migration is only an initializer for a NEW native PPO run.
    result.pop("optimizer_state", None)
    result["metrics"] = {"migration_only": True, "new_training_steps": 0}
    result["learning_scope"] = "identity initialization for bounded internal-gain PPO; no learning during migration"
    result["internal_plasticity_migration"] = {
        "schema": "bounded-postsynaptic-gain-initializer-v1", "parent_sha256": parent_sha256,
        "limit": float(limit), "new_parameters": EXPECTED_NEURONS,
        "existing_actor_tensors_copied_exactly": True, "initial_effective_weights_unchanged": True,
        "brain_topology_changed": False, "base_signed_weights_changed": False,
        "independent_edge_plasticity": False, "biological_learning_rule_claimed": False,
        "old_offline_optimizer_reused": False,
        "intended_use": "new PPO experiment from distillation; retain fixed-brain control",
    }
    return result


def retime_distilled_checkpoint(parent: dict, *, substeps: int, parent_sha256: str) -> dict:
    """New explicit neural-clock experiment; keep physiology/learned tensors.

    This changes the amount of neural processing before a game action. It is
    not identity deployment, and not a silent continuation of an existing PPO.
    """
    if isinstance(substeps, bool) or not isinstance(substeps, int) or substeps < 1:
        raise ValueError('substeps must be a positive integer')
    if (parent.get('schema') != CHECKPOINT_SCHEMA or parent.get('fixed_weights_stored') is not False
            or parent.get('neurons') != EXPECTED_NEURONS or parent.get('edge_records') != EXPECTED_EDGE_RECORDS):
        raise ValueError('Neural retiming requires a complete pinned-brain parent')
    if 'rl_state' in parent:
        raise ValueError('Do not retime a learned PPO checkpoint with this distilled initializer')
    old = parent['model_config']['substeps']
    if old == substeps or substeps % old:
        raise ValueError('Explicit retiming requires a larger integer multiple of the old neural window')
    result = deepcopy(parent)
    result['model_config']['substeps'] = substeps
    training = result['training_config']
    duration = substeps*result['model_config']['dt_ms']
    training.update(substeps=substeps, neural_ms_per_decision=duration,
        actual_warmup_ms=math.ceil(training['warmup_ms']/duration)*duration)
    result['metrics'] = {'migration_only':True,'new_training_steps':0}
    result['neural_clock_migration'] = {'schema':'explicit-distilled-neural-clock-v1',
        'parent_sha256':parent_sha256,'old_substeps':old,'new_substeps':substeps,
        'old_neural_ms_per_decision':old*result['model_config']['dt_ms'],
        'new_neural_ms_per_decision':duration,'dt_ms':result['model_config']['dt_ms'],
        'requested_warmup_ms':training['warmup_ms'],'actual_new_warmup_ms':training['actual_warmup_ms'],
        'old_actor_tensors_and_offline_Adam_copied_exactly':True,
        'source_physiology_topology_and_signed_base_weights_changed':False,
        'deployment_behavior_expected_to_change':True,'new_independent_evaluation_required':True}
    result['learning_scope'] = 'explicit neural-clock initializer; no new learning during migration'
    return result


def assert_learning_state_equal(first, second):
    """Verify actor/critic/Adam/RNG/counters without serialization assumptions."""
    if isinstance(first, torch.Tensor):
        if (not isinstance(second, torch.Tensor) or first.dtype != second.dtype
                or first.shape != second.shape or not torch.equal(first, second)):
            raise ValueError('PPO learning tensor changed during clock transition')
    elif isinstance(first, dict):
        if not isinstance(second, dict) or first.keys() != second.keys():
            raise ValueError('PPO learning state keys changed')
        for key in first:
            assert_learning_state_equal(first[key], second[key])
    elif isinstance(first, (list,tuple)):
        if type(first) is not type(second) or len(first) != len(second):
            raise ValueError('PPO learning sequence changed')
        for a,b in zip(first,second):
            assert_learning_state_equal(a,b)
    elif type(first) is not type(second) or first != second:
        raise ValueError('PPO learning scalar changed')


def transition_ppo_neural_clock(parent: dict, *, substeps: int, parent_sha256: str) -> dict:
    """Explicit changed-clock curriculum preserving all existing PPO learning.

    Only future neural integration per game decision changes. No previous
    rollout is relabeled as if collected under the new clock. A new native
    campaign is required, just as for the ordinary linked PPO continuation.
    """
    from .whole_brain_ppo import _validate_ppo_checkpoint
    if (parent.get('schema') != CHECKPOINT_SCHEMA or parent.get('neurons') != EXPECTED_NEURONS
            or parent.get('edge_records') != EXPECTED_EDGE_RECORDS):
        raise ValueError('Clock transition requires a complete pinned PPO brain')
    if isinstance(substeps,bool) or not isinstance(substeps,int) or substeps < 1:
        raise ValueError('substeps must be a positive integer')
    old=parent['model_config']['substeps']
    if substeps <= old or substeps % old:
        raise ValueError('PPO clock transition requires a larger integer multiple')
    if (not isinstance(parent_sha256,str) or len(parent_sha256)!=64
            or any(c not in '0123456789abcdef' for c in parent_sha256)):
        raise ValueError('Explicit parent SHA256 required')
    _validate_ppo_checkpoint(parent,parent['training_config'])
    result=deepcopy(parent)
    result['model_config']['substeps']=substeps
    transition={'schema':'explicit-ppo-neural-clock-transition-v1','parent_sha256':parent_sha256,
        'old_substeps':old,'new_substeps':substeps,'dt_ms':parent['model_config']['dt_ms'],
        'old_neural_ms_per_decision':old*parent['model_config']['dt_ms'],
        'new_neural_ms_per_decision':substeps*parent['model_config']['dt_ms'],
        'parent_updates':parent['rl_state']['updates'],
        'parent_native_decisions':parent['rl_state']['completed_timesteps'],
        'parent_native_frames':parent['rl_state']['native_frames'],
        'preserved':['actor_including_internal_gains','critic','Adam','action_sampler_rng','shuffle_rng','counters','offline_split_history'],
        'source_physiology_topology_and_signed_base_weights_changed':False,
        'deployment_behavior_expected_to_change':True,'native_campaign_restarted':True,
        'old_rollouts_reinterpreted_under_new_clock':False,
        'parent_model_config':deepcopy(parent['model_config'])}
    warmup=parent['training_config']['warmup_ms']
    transition.update(requested_warmup_ms=warmup,
        old_actual_warmup_ms=math.ceil(warmup/transition['old_neural_ms_per_decision'])*transition['old_neural_ms_per_decision'],
        new_actual_warmup_ms=math.ceil(warmup/transition['new_neural_ms_per_decision'])*transition['new_neural_ms_per_decision'])
    result['neural_clock_transition']=transition
    result['training_config']['model_config']=deepcopy(result['model_config'])
    result['training_config']['neural_clock_transition']=deepcopy(transition)
    result['metrics']={**result.get('metrics',{}),'configuration_transition_only':True,
        'inherited_metrics_neural_ms_per_decision':transition['old_neural_ms_per_decision']}
    result['learning_scope']='explicit neural-clock curriculum transition; actor/critic/Adam/RNG/counters retained; no training in migration'
    assert_learning_state_equal(parent['trainable_parameters'],result['trainable_parameters'])
    assert_learning_state_equal(parent['rl_state'],result['rl_state'])
    return result


def transition_numerical_runtime(parent: dict, *, parent_sha256: str, backend='coo-bmm-deterministic') -> dict:
    """Change the sparse reduction algorithm without discarding any learning."""
    from .whole_brain import DETERMINISTIC_PROPAGATION,CACHED_PROPAGATION,numerical_runtime_specification
    if (parent.get('schema')!=CHECKPOINT_SCHEMA or parent.get('neurons')!=EXPECTED_NEURONS
            or parent.get('edge_records')!=EXPECTED_EDGE_RECORDS or parent.get('fixed_weights_stored') is not False):
        raise ValueError('Numerical transition requires the complete pinned brain')
    if (not isinstance(parent_sha256,str) or len(parent_sha256)!=64
            or any(c not in '0123456789abcdef' for c in parent_sha256)):
        raise ValueError('Explicit parent SHA256 required')
    if backend not in (DETERMINISTIC_PROPAGATION,CACHED_PROPAGATION):
        raise ValueError('Transition destination must retain deterministic COO ALG2')
    old_backend=parent['model_config'].get('propagation_backend','coo-mm')
    if old_backend==backend:
        raise ValueError('Parent already uses an explicit numerical backend; do not reinitialize')
    if old_backend not in ('coo-mm',DETERMINISTIC_PROPAGATION,CACHED_PROPAGATION):
        raise ValueError('Unrecognized parent numerical backend')
    if 'rl_state' in parent:
        from .whole_brain_ppo import _validate_ppo_checkpoint
        _validate_ppo_checkpoint(parent,parent['training_config'])
    result=deepcopy(parent)
    result['model_config']['propagation_backend']=backend
    runtime=numerical_runtime_specification(backend)
    transition={'schema':'explicit-brain-numerical-runtime-transition-v1','parent_sha256':parent_sha256,
        'old_backend':old_backend,'new_runtime':runtime,
        'reason':('cache immutable source indices/workspaces while retaining complete COO ALG2 forward and adjoint'
                  if backend==CACHED_PROPAGATION else 'select deterministic COO ALG2 for complete forward and adjoint'),
        'actor_parameters_preserved_exactly':True,'neural_clock_and_physiology_changed':False,
        'source_topology_and_signed_weights_changed':False,
        'old_rollouts_reinterpreted_as_new_runtime_data':False,
        'roundoff_trajectory_changes_possible':True,'new_native_campaign_required':True,
        'old_model_config':deepcopy(parent['model_config'])}
    if 'rl_state' in parent:
        transition['preserved_ppo']=['actor','critic','Adam','action_sampler_rng','shuffle_rng','counters','offline_splits']
        transition['parent_updates']=parent['rl_state']['updates']
        transition['parent_native_decisions']=parent['rl_state']['completed_timesteps']
        result['training_config']['model_config']=deepcopy(result['model_config'])
    history=deepcopy(parent.get('numerical_runtime_history',parent.get('training_config',{}).get('numerical_runtime_history',[])))
    previous_transition=parent.get('numerical_runtime_transition',parent.get('training_config',{}).get('numerical_runtime_transition'))
    if previous_transition is not None and (not history or history[-1]!=previous_transition):
        history.append(deepcopy(previous_transition))
    if history:
        result['numerical_runtime_history']=history
        result['training_config']['numerical_runtime_history']=deepcopy(history)
    result['numerical_runtime_transition']=transition
    result['training_config']['numerical_runtime_transition']=deepcopy(transition)
    result['training_config']['numerical_runtime']=runtime
    result['metrics']={**result.get('metrics',{}),'configuration_transition_only':True}
    result['learning_scope']='explicit deterministic-runtime transition; no training or parameter reset during migration'
    assert_learning_state_equal(parent['trainable_parameters'],result['trainable_parameters'])
    for name in ('rl_state','optimizer_state','split_history','train_episode_ids','validation_episode_ids'):
        if name in parent:
            assert_learning_state_equal(parent[name],result[name])
    return result
