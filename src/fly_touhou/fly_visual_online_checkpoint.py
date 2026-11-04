"""Typed actor+critic and real Adam recovery;native episodes restart explicitly."""
from __future__ import annotations

import copy
import torch

SCHEMA = 'fly-visual-online-actor-critic-Adam-recovery-v1'


def _cpu_tree(value):
    if torch.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:_cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [_cpu_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(_cpu_tree(v) for v in value)
    return copy.deepcopy(value)


def optimizer_parameter_names(model,optimizer):
    names = {id(p):name for name,p in model.named_parameters()}
    result = [[names[id(p)] for p in group['params']] for group in optimizer.param_groups]
    flat = [name for group in result for name in group]
    if len(flat)!=len(set(flat)):
        raise ValueError('Optimizer parameter identities must be unique')
    return result


def training_checkpoint_payload(model,optimizer,*,counters,algorithm,parent_provenance):
    if not isinstance(optimizer,torch.optim.Adam):
        raise ValueError('This recovery schema records the actual Adam optimizer')
    return dict(schema=SCHEMA,model_payload=model.parameter_checkpoint_payload(),
        optimizer_state=_cpu_tree(optimizer.state_dict()),optimizer_defaults=_cpu_tree(optimizer.defaults),
        optimizer_parameter_names=optimizer_parameter_names(model,optimizer),
        counters=copy.deepcopy(counters),algorithm=copy.deepcopy(algorithm),parent_provenance=copy.deepcopy(parent_provenance),
        critic_saved=True,optimizer_saved=True,torch_version=str(torch.__version__),
        recovery_requires_fresh_native_episodes=True,exact_mid_episode_native_resume=False,
        brain_teacher_action_RNG_or_environment_state_saved=False)


def restore_training_checkpoint(model,optimizer,payload,*,expected_algorithm):
    """Restore all model/critic parameters, Adam moments and counters.

    A future caller must start NEW native games and fresh neural/teacher/RNG
    state. This is not an exact continuation of an in-flight original process.
    """
    if (not isinstance(optimizer,torch.optim.Adam) or payload.get('schema')!=SCHEMA
            or payload.get('algorithm')!=expected_algorithm or payload.get('critic_saved') is not True
            or payload.get('optimizer_saved') is not True
            or payload.get('recovery_requires_fresh_native_episodes') is not True
            or payload.get('exact_mid_episode_native_resume') is not False
            or payload.get('optimizer_parameter_names')!=optimizer_parameter_names(model,optimizer)
            or payload.get('optimizer_defaults')!=optimizer.defaults):
        raise ValueError('Exact typed model/critic/Adam algorithm and parameter ordering required')
    saved = payload.get('optimizer_state',{})
    current = optimizer.state_dict()
    if set(saved)!={'state','param_groups'} or len(saved['param_groups'])!=len(current['param_groups']):
        raise ValueError('Complete Adam state and parameter groups required')
    allowed = set()
    for old,new,live in zip(saved['param_groups'],current['param_groups'],optimizer.param_groups):
        if old!=new:
            raise ValueError('Recovery cannot silently alter Adam groups or learning rates')
        allowed.update(old['params'])
        for index,parameter in zip(old['params'],live['params']):
            state = saved['state'].get(index)
            if state is None:continue
            expected = {'step','exp_avg','exp_avg_sq'}|({'max_exp_avg_sq'} if old.get('amsgrad') else set())
            if set(state)!=expected:
                raise ValueError('Adam moment/step fields differ')
            for name,value in state.items():
                if (not torch.is_tensor(value) or not bool(torch.isfinite(value).all())
                        or (name=='step' and (value.numel()!=1 or float(value)<0))
                        or (name!='step' and (value.shape!=parameter.shape or value.dtype!=parameter.dtype))):
                    raise ValueError('Adam moment shape/dtype/finiteness differs:'+name)
    if set(saved['state'])-allowed:
        raise ValueError('Unknown optimizer parameter state')
    # The typed model loader separately validates the full graph/ports/frontend,
    # all brain parameters AND the scalar critic before making the copy.
    model.load_parameter_payload(payload['model_payload'])
    optimizer.load_state_dict(_cpu_tree(saved))
    return copy.deepcopy(payload['counters'])
