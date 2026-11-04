"""Explicit BC-to-full-policy transfer, never a numerical PPO resume.

The supervised actor optimizer and its RNG format are not a PPO checkpoint.
Restore the shared policy, unfreeze its encoders and critics, and create one
fresh, empty PPO Adam. Preserve ancestral PPO cost separately from BC work.
"""
from __future__ import annotations

import random

from .thrl_cached_actor_fit import TRAINABLE_PREFIXES


def validate_actor_payload(torch, warmup, result, parent):
    if (warmup.get('schema')!='B1-full-teacher-temporary-actor-warmup-v1'
            or result.get('schema')!='B1-full-natural-teacher-actor-warmup-result-v1'
            or result.get('status')!='completed' or result.get('CUDA') is not True):
        raise ValueError('Completed actual-CUDA full-teacher actor warmup required')
    for row in (warmup,result):
        if row.get('numerical_qualification_only') or row.get('deployment_forbidden'):
            raise ValueError('Numerical qualification policies must never seed native learning')
        if row.get('PPO_updates')!=0 or row.get('full_policy_optimization_still_required') is not True:
            raise ValueError('BC work must remain distinct from following full-policy PPO')
    if (warmup.get('PPO_optimizer_missing_intentionally') is not True
            or warmup.get('new_PPO_optimizer_required_after_BC') is not True
            or 'optimizer' in warmup):
        raise ValueError('Do not reinterpret the restricted BC Adam as full-model PPO Adam')
    if warmup['model_config']!=parent['model_config']:
        raise ValueError('BC transfer cannot change policy, input or neural risk architecture')
    if set(warmup['model'])!=set(parent['model']):
        raise ValueError('Missing policy parameters or buffers')
    changed=[]
    for name,tensor in warmup['model'].items():
        before=parent['model'][name]
        if tensor.shape!=before.shape or tensor.dtype!=before.dtype or not torch.isfinite(tensor).all():
            raise ValueError('Invalid actor transfer tensor: '+name)
        equal=torch.equal(tensor.cpu(),before.cpu())
        if not equal:
            if not name.startswith(TRAINABLE_PREFIXES):
                raise ValueError('Temporary BC altered a frozen encoder, critic or risk: '+name)
            changed.append(name)
    for key in ('selected_epoch','selected_BC_optimizer_steps','paid_BC_optimizer_steps'):
        if (type(warmup.get(key)) is not int or warmup[key]<0 or result.get(key)!=warmup[key]):
            raise ValueError('Selected BC model and paid work counters disagree: '+key)
    if (warmup['selected_BC_optimizer_steps']>warmup['paid_BC_optimizer_steps']
            or warmup['selected_epoch']>result['epochs_executed']
            or warmup['selected_heldout']!=result['selected_heldout']):
        raise ValueError('Selected BC provenance differs from the final restored model')
    if parent.get('schema')!='th14-thrl-training-checkpoint-v1' or parent.get('optimizer_transaction_complete') is not True:
        raise ValueError('Ancestral PPO model must be a complete committed transaction')
    return dict(changed_actor_parameter_tensors=changed,
        encoder_critics_and_neural_risk_equal_to_parent=True,
        BC_paid_optimizer_steps=warmup['paid_BC_optimizer_steps'],
        BC_selected_optimizer_steps=warmup['selected_BC_optimizer_steps'],
        BC_selected_epoch=warmup['selected_epoch'])


def build_transfer(torch, model, warmup, result, parent, ppo, *, seed):
    """A constructed checkpoint; caller pins files and saves atomically."""
    import numpy as np
    from .thrl_ppo import make_optimizer
    from .thrl_study_io import utc_now
    audit=validate_actor_payload(torch,warmup,result,parent)
    if model.model_config()!=warmup['model_config']:
        raise ValueError('Fresh full-policy model differs from the BC model')
    model.load_state_dict(warmup['model'],strict=True)
    # Module constructors normally do this already. Make the temporary nature
    # of the cached-encoder phase explicit, and retain fitted risk calibration.
    model.base.requires_grad_(True)
    model.risk.requires_grad_(False)
    if model.laser_risk is not None:model.laser_risk.requires_grad_(False)
    if not model.frozen_readout or any(not p.requires_grad for p in model.base.parameters()):
        raise ValueError('Following PPO must train every shared policy parameter')
    optimizer=make_optimizer(model,ppo)
    if optimizer.state_dict()['state']:
        raise ValueError('PPO Adam must start empty after restricted BC fitting')
    # This is a declared fresh sampling stream, not an inherited BC generator
    # dict passed into numpy.random.set_state, nor an old PPO stream resume.
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    inherited={key:int(parent.get(lifetime,parent.get(local,0))) for key,lifetime,local in (
        ('decisions','lifetime_decisions','decisions'),
        ('rollout_updates','lifetime_rollout_updates','rollout_updates'),
        ('adam_minibatch_steps','lifetime_adam_minibatch_steps','adam_minibatch_steps'))}
    payload=dict(schema='th14-thrl-training-checkpoint-v1',created_utc=utc_now(),
        model_config=model.model_config(),model={k:v.detach().cpu() for k,v in model.state_dict().items()},
        optimizer=optimizer.state_dict(),optimizer_parameter_names=list(dict(model.named_parameters())),
        decisions=0,rollout_updates=0,adam_minibatch_steps=0,candidate_decisions=0,candidate_optimized_decisions=0,
        candidate_paid_decisions_lower_bound=0,optimizer_transaction_complete=True,
        lifetime_decisions=inherited['decisions'],lifetime_rollout_updates=inherited['rollout_updates'],
        lifetime_adam_minibatch_steps=inherited['adam_minibatch_steps'],config=dict(warmup['config']),
        checkpoint_phase='actor-BC-transfer-fresh-full-policy-PPO-Adam',
        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        numpy_rng=np.random.get_state(),python_rng=random.getstate(),
        BC_to_PPO_transfer_audit={**audit,**inherited, 'full_base_trainable':True,
            'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'PPO_optimizer_state_entries':0,'fresh_sampling_seed':seed,
            'restricted_BC_optimizer_not_reused':True,'numerical_PPO_resume':False},
        teacher_manifest=warmup['teacher_manifest'],teacher_manifest_sha256=warmup['teacher_manifest_sha256'],
        parent_checkpoint=warmup['parent_checkpoint'],parent_sha256=warmup['parent_sha256'],
        native_resume='fresh full Story opening;one shared policy;no world snapshot')
    return payload
