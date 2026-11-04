"""Preserve calibrated full-brain interfaces and named Adam when entering PPO."""
from __future__ import annotations
import copy
import torch
from .continuous_guided_dagger import SCHEMA,build_guided_model,parameter_payload
from .guided_optimizer_io import groups_from_signature,parameter_signature


def restore_guided_PPO_source(source,connectome,*,device='cuda:0'):
    if source.get('schema')!=SCHEMA+'-training':raise ValueError('Complete calibrated guided training source required')
    model=build_guided_model(source['model_payload'],connectome,device=device)
    optimizer=torch.optim.Adam(groups_from_signature(model,source['optimizer'],source['optimizer_parameter_signature']))
    optimizer.load_state_dict(source['optimizer'])
    torch.set_rng_state(source['torch_RNG'])
    if model.device.type=='cuda':torch.cuda.set_rng_state_all(source['CUDA_RNG'])
    payload=source['model_payload']
    context=dict(base_source=payload['base_source'],base_sha256=payload['base_sha256'],
        calibration=copy.deepcopy(payload['calibration']),inherited=copy.deepcopy(payload['counters']),
        numpy_RNG=copy.deepcopy(source['numpy_RNG']))
    return model,optimizer,copy.deepcopy(payload['counters']),context


def guided_PPO_training_payload(model,optimizer,*,algorithm,counters,provenance,context):
    cumulative=copy.deepcopy(context['inherited'])
    cumulative['student_actions']=cumulative.get('student_actions',0)+counters['student_actions']
    cumulative['PPO_updates']=cumulative.get('PPO_updates',0)+counters['PPO_Adam']
    cumulative['PPO_rollouts']=cumulative.get('PPO_rollouts',0)+counters['completed_rollouts']
    cumulative['brain_updates_in_this_phase']=counters['PPO_Adam']
    payload=parameter_payload(model,context['base_source'],context['base_sha256'],context['calibration'],cumulative)
    return dict(schema=SCHEMA+'-training',model_payload=payload,optimizer=optimizer.state_dict(),
        optimizer_parameter_signature=parameter_signature(model,optimizer),costs=cumulative,phase_costs=copy.deepcopy(counters),
        algorithm=copy.deepcopy(algorithm),provenance=copy.deepcopy(provenance),numpy_RNG=copy.deepcopy(context['numpy_RNG']),
        torch_RNG=torch.get_rng_state(),CUDA_RNG=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
        neural_midgame_resume=False,calibrated_interfaces_preserved=True,teacher_actions_at_inference=0)
