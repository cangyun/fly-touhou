"""Named Adam resume and one-query learning-rate change without resetting moments."""
from __future__ import annotations


def parameter_signature(model,optimizer):
    names={id(p):name for name,p in model.named_parameters()}
    return [[names[id(p)] for p in group['params']] for group in optimizer.param_groups]


def groups_from_signature(model,state,signature):
    named=dict(model.named_parameters())
    if len(state['param_groups'])!=len(signature):raise ValueError('Saved Adam group signature differs')
    names=[name for group in signature for name in group]
    if len(set(names))!=len(names) or set(names)!=set(named):
        raise ValueError('Named Adam signature must cover the exact model once')
    groups=[]
    for saved,names in zip(state['param_groups'],signature):
        if len(saved['params'])!=len(names):raise ValueError('Saved Adam parameter count differs')
        groups.append(dict({k:v for k,v in saved.items() if k!='params'},params=[named[name] for name in names]))
    return groups


def separate_attention_query(optimizer,query,*,lr):
    if not lr>0:raise ValueError('Positive query learning rate required')
    containing=[g for g in optimizer.param_groups if any(p is query for p in g['params'])]
    if len(containing)!=1:raise ValueError('Attention query must occur in exactly one Adam group')
    group=containing[0];old_name=group['name'];old_lr=group['lr']
    state=optimizer.state.get(query)
    if old_name=='sensory_geometry_pool_query':
        if len(group['params'])!=1:raise ValueError('Dedicated query group contains other parameters')
        group['lr']=lr
    else:
        options={k:v for k,v in group.items() if k not in ('params','name','lr')}
        group['params']=[p for p in group['params'] if p is not query]
        optimizer.add_param_group(dict(options,name='sensory_geometry_pool_query',params=[query],lr=lr))
    if optimizer.state.get(query) is not state:raise RuntimeError('Query Adam moments changed during regrouping')
    return dict(old_group=old_name,old_lr=old_lr,new_group='sensory_geometry_pool_query',new_lr=lr,
        query_parameters=query.numel(),query_reinitialized=False,Adam_moments_preserved=True,
        inherited_query_Adam_step=float(state['step']) if state else 0.)
