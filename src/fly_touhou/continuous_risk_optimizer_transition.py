"""Explicit optimizer coordinates for an optional risk-only stage.

No forward,parameter/flag/RNG write,optimizer construction or training occurs.
Dedicated mode drops complete loss-path Adam states,including their step.
Protected actor/three-critic state and all non-LR group settings stay exact.
"""
from __future__ import annotations
import hashlib
import math

SCHEMA='continuous-risk-stage-optimizer-transition-v1'
LOSS_PATH_GROUPS=('input','brain','risk','sensory_object','sensory_action_query','sensory_CNN_numeric')
OPTIONAL_LOSS_PATH_GROUPS=('sensory_pair_geometry',)
PROTECTED_GROUPS=('actor','critic')


def _json(value):
    import torch
    if isinstance(value,(tuple,list)):return [_json(v) for v in value]
    if isinstance(value,dict):return {k:_json(v) for k,v in value.items()}
    if torch.is_tensor(value):
        cpu=value.detach().cpu().contiguous()
        return dict(dtype=str(cpu.dtype),shape=list(cpu.shape),
            sha256=hashlib.sha256(cpu.numpy().tobytes()).hexdigest())
    if value is None or type(value) in (bool,int,float,str):return value
    raise TypeError('Unsupported optimizer setting/state type:'+type(value).__name__)


def _protected(model,optimizer,groups):
    return {name:dict(settings=_json({k:v for k,v in groups[name].items() if k!='params'}),
        parameter_order=[id(p) for p in groups[name]['params']],
        weights=_json(getattr(model,name).state_dict()),
        Adam=[_json(optimizer.state[p]) if p in optimizer.state else None for p in groups[name]['params']])
        for name in PROTECTED_GROUPS}


def _model_signature(model):
    return [(id(v),v.data_ptr(),tuple(v.shape),v.dtype,v.device,v._version,
        v.requires_grad) for v in tuple(model.parameters())+tuple(model.buffers())]


def transition_risk_stage_optimizer(model,optimizer,*,mode='dedicated-risk-v1'):
    """Reuse the same Adam;only named loss-path states/LRs may change.

    All structural validation precedes the first mutation. The receipt is phase
    provenance,not evidence that this LR/reset choice improves native behavior.
    """
    import torch
    if mode not in ('preserve','dedicated-risk-v1') or not isinstance(optimizer,torch.optim.Adam):
        raise ValueError('Named original Adam and explicit preserve|dedicated-risk-v1 mode required')
    names=[g.get('name') for g in optimizer.param_groups]
    required=set(LOSS_PATH_GROUPS+PROTECTED_GROUPS)
    if (len(names)!=len(set(names)) or not required.issubset(names)
            or set(names)-required-set(OPTIONAL_LOSS_PATH_GROUPS)):
        raise ValueError('Exact named Full/FullPair optimizer group coverage required')
    groups={g['name']:g for g in optimizer.param_groups}
    all_params=[p for g in optimizer.param_groups for p in g['params']]
    if len(all_params)!=len({id(p) for p in all_params}):raise ValueError('Duplicate optimizer parameter')
    model_ids={id(p) for p in model.parameters()}
    if any(id(p) not in model_ids for p in all_params):raise ValueError('Optimizer parameter outside this model')
    for name in PROTECTED_GROUPS:
        if [id(p) for p in groups[name]['params']]!=[id(p) for p in getattr(model,name).parameters()]:
            raise ValueError('Protected group differs from the retained '+name)
    records={}
    for name,g in groups.items():
        if not g['params'] or type(g['lr']) not in (int,float) or not math.isfinite(g['lr']) or g['lr']<=0:
            raise ValueError('Nonempty group and positive finite scalar source LR required')
        settings=_json({k:v for k,v in g.items() if k not in ('params','lr')})
        target=(1e-3 if name=='risk' else 3e-4) if mode=='dedicated-risk-v1' and name not in PROTECTED_GROUPS else g['lr']
        records[name]=dict(source_lr=g['lr'],target_lr=target,parameter_tensors=len(g['params']),
            source_state_count=sum(p in optimizer.state for p in g['params']),
            source_initialized_state_count=sum(bool(optimizer.state.get(p,{})) for p in g['params']),
            source_settings_except_lr=settings)
    protected_before=_protected(model,optimizer,groups);signature=_model_signature(model)
    optimizer_id=id(optimizer)
    if mode=='dedicated-risk-v1':
        for name,g in groups.items():
            if name in PROTECTED_GROUPS:continue
            for p in g['params']:optimizer.state.pop(p,None)
            g['lr']=records[name]['target_lr']
    for name,g in groups.items():
        row=records[name]
        row.update(target_state_count=sum(p in optimizer.state for p in g['params']),
            state_cleared=mode=='dedicated-risk-v1' and name not in PROTECTED_GROUPS,
            target_settings_except_lr=_json({k:v for k,v in g.items() if k not in ('params','lr')}))
        row['settings_except_lr_preserved']=row['source_settings_except_lr']==row['target_settings_except_lr']
    protected_exact=_protected(model,optimizer,groups)==protected_before
    untouched=_model_signature(model)==signature
    if not protected_exact or not untouched:raise RuntimeError('Risk transition touched a protected model/Adam field')
    return dict(schema=SCHEMA,mode=mode,optimizer_object_reused=id(optimizer)==optimizer_id,groups=records,
        actor_critic_groups_and_Adam_exact=protected_exact,parameters_and_flags_untouched=untouched,
        RNG_used=False,new_optimizer_created=False,reset_includes_step=mode=='dedicated-risk-v1',
        source_algorithm_modified=False,performance_improvement_claimed=False)


def validate_risk_optimizer_transition(receipt,*,expected_mode=None):
    """JSON-only contract for the prepared reviewer/monitor;no Torch import."""
    def require(condition,message):
        if not condition:raise ValueError('Risk optimizer transition:'+message)
    require(isinstance(receipt,dict) and receipt.get('schema')==SCHEMA,'schema')
    mode=receipt.get('mode');require(mode in ('preserve','dedicated-risk-v1'),'mode')
    if expected_mode is not None:require(mode==expected_mode,'declared/actual mode mismatch')
    for key in ('optimizer_object_reused','actor_critic_groups_and_Adam_exact','parameters_and_flags_untouched'):
        require(receipt.get(key) is True,key)
    for key in ('RNG_used','new_optimizer_created','source_algorithm_modified','performance_improvement_claimed'):
        require(receipt.get(key) is False,key)
    require(receipt.get('reset_includes_step') is (mode=='dedicated-risk-v1'),'reset step protocol')
    groups=receipt.get('groups') or {};required=set(LOSS_PATH_GROUPS+PROTECTED_GROUPS)
    require(required.issubset(groups) and not(set(groups)-required-set(OPTIONAL_LOSS_PATH_GROUPS)),'group coverage')
    for name,row in groups.items():
        source=row.get('source_lr');target=row.get('target_lr')
        require(type(source) in (int,float) and math.isfinite(source) and source>0,'source LR '+name)
        clear=mode=='dedicated-risk-v1' and name not in PROTECTED_GROUPS
        require(target==((1e-3 if name=='risk' else 3e-4) if clear else source),'target LR '+name)
        n=row.get('parameter_tensors');start=row.get('source_state_count');end=row.get('target_state_count')
        require(type(n) is int and n>0 and type(start) is int and 0<=start<=n,'state counts '+name)
        require(type(end) is int and end==(0 if clear else start),'target states '+name)
        require(row.get('state_cleared') is clear,'reset groups '+name)
        require(row.get('settings_except_lr_preserved') is True
            and row.get('source_settings_except_lr')==row.get('target_settings_except_lr'),'other settings '+name)
    return receipt
