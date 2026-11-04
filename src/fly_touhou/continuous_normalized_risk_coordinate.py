"""Opt-in fixed risk-input coordinates; no model/core/job is constructed here.

The Sequential subclass normalizes both layers(DN) signed supervision and the
same layers(DN) used by softplus cost. Forward equivalence is approximate in
FP32; diagonal Adam and global gradient clipping are not coordinate invariant.
"""
from __future__ import annotations
import copy
import hashlib

import numpy as np
import torch
from torch import nn

PROTOCOL = 'continuous-fixed-normalized-risk-coordinate-v1'
TRAINING_SCHEMA = 'continuous-full-connectome-threevalue-normalized-risk-training-v1'
PARAMETER_SCHEMA = 'continuous-full-connectome-DN-actor-threevalue-normalized-risk-parameters-v1'
BASE_PARAMETER_SCHEMA = 'continuous-full-connectome-DN-actor-threevalue-cost-parameters-v1'
FIRST_KEYS = ('layers.0.weight','layers.0.bias')
BUFFER_KEYS = ('layers.DN_mean','layers.DN_scale')


def tensor_sha(value):
    a=value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(a.tobytes()).hexdigest()


class NormalizedRiskLayers(nn.Sequential):
    """Preserves[0]/[2] and all four parameter names; adds two fixed buffers."""
    def __init__(self,first,activation,last,mean,scale):
        super().__init__(first,activation,last)
        if (not isinstance(first,nn.Linear) or not isinstance(last,nn.Linear)
                or first.out_features!=64 or last.out_features!=18
                or mean.shape!=(first.in_features,) or scale.shape!=mean.shape
                or mean.dtype!=torch.float32 or scale.dtype!=torch.float32
                or not bool(torch.isfinite(mean).all() and torch.isfinite(scale).all() and (scale>0).all())):
            raise ValueError('Original risk64 topology and finite original FP32 train-only statistics required')
        self.register_buffer('DN_mean',mean.detach().to(first.weight.device).clone())
        self.register_buffer('DN_scale',scale.detach().to(first.weight.device).clone())

    def forward(self,DN):
        return super().forward((DN-self.DN_mean)/self.DN_scale)


@torch.no_grad()
def convert_risk_layers(head,mean,scale):
    """Explicit future injection; keeps every Parameter object,including last2."""
    old=head.layers
    if isinstance(old,NormalizedRiskLayers):raise ValueError('Do not apply coordinates twice')
    if len(old)!=3 or not isinstance(old[1],nn.SiLU):raise ValueError('Existing64SiLU cost head required')
    first=old[0]
    normalized=NormalizedRiskLayers(old[0],old[1],old[2],mean,scale)
    m,s=normalized.DN_mean,normalized.DN_scale
    W=first.weight.detach().clone();b=first.bias.detach().clone()
    first.weight.copy_(W*s)
    first.bias.copy_((b.double()+W.double()@m.double()).float())
    head.layers=normalized
    return head


def _update_hash(value,digest):
    if torch.is_tensor(value):
        a=value.detach().cpu().contiguous().numpy();digest.update(str((a.dtype,a.shape)).encode())
        raw=memoryview(a.reshape(-1).view(np.uint8))
        for start in range(0,len(raw),4*1024*1024):digest.update(raw[start:start+4*1024*1024])
    elif isinstance(value,dict):
        for key in sorted(value,key=str):digest.update(str(key).encode());_update_hash(value[key],digest)
    elif isinstance(value,(tuple,list)):
        for item in value:_update_hash(item,digest)
    else:digest.update(repr(value).encode())


def optimizer_mapping(payload):
    signature=payload['optimizer_parameter_signature'];groups=payload['optimizer']['param_groups']
    if len(signature)!=len(groups):raise ValueError('Dynamic optimizer group signature mismatch')
    mapping=[]
    for gi,(specs,group) in enumerate(zip(signature,groups)):
        if len(specs)!=len(group['params']):raise ValueError('Dynamic optimizer parameter signature mismatch')
        for position,(spec,identifier) in enumerate(zip(specs,group['params'])):
            if spec['name'].startswith('risk_head.'):
                state=payload['optimizer']['state'].get(identifier)
                mapping.append(dict(name=spec['name'],group_index=gi,group_name=group.get('name'),
                    position=position,optimizer_identifier=identifier,shape=spec['shape'],
                    coordinate_transformed=spec['name'].removeprefix('risk_head.') in FIRST_KEYS,
                    state_present=state is not None,step=float(state.get('step',0)) if state is not None else None))
    if {m['name'] for m in mapping}!={'risk_head.'+k for k in (*FIRST_KEYS,'layers.2.weight','layers.2.bias')}:
        raise ValueError('Exactly the original four risk parameters required')
    return mapping


def unaffected_identity(payload):
    """Hash every unchanged tensor and all non-first-layer Adam state/groups."""
    digest=hashlib.sha256()
    def network(value,path=()):
        if torch.is_tensor(value):
            if path in [('risk_head',k) for k in (*FIRST_KEYS,*BUFFER_KEYS)]:return
            digest.update('/'.join(path).encode());_update_hash(value,digest)
        elif isinstance(value,dict):
            for key in sorted(value):network(value[key],path+(key,))
        elif isinstance(value,(tuple,list)):
            for i,item in enumerate(value):network(item,path+(str(i),))
    network(payload['model_payload'])
    transformed={m['optimizer_identifier'] for m in optimizer_mapping(payload) if m['coordinate_transformed']}
    _update_hash({k:v for k,v in payload['optimizer']['state'].items() if k not in transformed},digest)
    _update_hash(payload['optimizer']['param_groups'],digest)
    _update_hash(payload['optimizer_parameter_signature'],digest)
    _update_hash(payload.get('RNG'),digest)
    return digest.hexdigest()


def tag_normalized_payload(payload,spec):
    """Explicit subtype prevents an old raw-input loader accepting Wnorm."""
    payload['schema']=TRAINING_SCHEMA
    model=payload['model_payload'];model['schema']=PARAMETER_SCHEMA
    model['configuration']=copy.deepcopy(model['configuration'])
    model['configuration']['schema']=PARAMETER_SCHEMA
    model['configuration']['risk']='fixed standardized DN->64SiLU->18softplus;same normalization for layers(DN) signed supervision'
    model['risk_input_normalization']=copy.deepcopy(spec)
    payload['algorithm']=copy.deepcopy(payload['algorithm'])
    payload['algorithm']['risk_input_coordinates']=copy.deepcopy(spec)
    return payload


def convert_training_payload(payload,normalized_head_state,spec):
    """No optimizer construction/step; only first-layer2 moments are invalidated."""
    if payload['schema']!='continuous-full-connectome-threevalue-training-v1':
        raise ValueError('Original raw-risk complete continuous training checkpoint required')
    before=unaffected_identity(payload);mapping=optimizer_mapping(payload)
    old=payload['model_payload']['risk_head']
    if set(normalized_head_state)!=set(old)|set(BUFFER_KEYS):raise ValueError('Only two fixed buffers may be added')
    for key in old:
        value=normalized_head_state[key]
        if value.shape!=old[key].shape or value.dtype!=old[key].dtype or not bool(torch.isfinite(value).all()):
            raise ValueError('Same finite four parameter shapes required')
        if key not in FIRST_KEYS and not torch.equal(value,old[key]):raise ValueError('Output layer must be byte-identical')
    payload['model_payload']['risk_head']={k:v.detach().cpu().clone() for k,v in normalized_head_state.items()}
    ids=[m['optimizer_identifier'] for m in mapping if m['coordinate_transformed']]
    present=sum(i in payload['optimizer']['state'] for i in ids)
    for identifier in ids:payload['optimizer']['state'].pop(identifier,None)
    tag_normalized_payload(payload,spec)
    after=unaffected_identity(payload)
    if before!=after:raise RuntimeError('A non-transformed network tensor,Adam state,group or RNG changed')
    return dict(unaffected_parameters_Adam_groups_and_RNG_bitwise_preserved=True,
        identity_before=before,identity_after=after,actual_mapping=mapping,
        coordinate_parameters_requiring_reset=2,states_present_before=present,states_removed=present,
        optimizer_steps_executed=0,numeric_optimizer_resume=False,
        diagonal_Adam_cross_moments_unavailable=True,global_clip_parameter_norm_changes=True,
        initial_dL_dDN_reduction_claim=False)


def load_normalized_parameter_payload(model,payload):
    """For a future explicit variant factory; never called by current runner.

    Construct the same base model/core first,call this instead of the old raw
    parameter loader,and use tag_normalized_payload when serializing training.
    This adapter deliberately recognizes the new schema before any translation.
    """
    if payload.get('schema')!=PARAMETER_SCHEMA:raise ValueError('Explicit normalized-risk parameter schema required')
    state=payload['risk_head'];spec=payload['risk_input_normalization']
    mean,scale=state[BUFFER_KEYS[0]],state[BUFFER_KEYS[1]]
    if tensor_sha(mean)!=spec['mean_sha256'] or tensor_sha(scale)!=spec['scale_sha256']:
        raise ValueError('Fixed normalization buffers differ from their recorded identity')
    old=model.risk_head.layers
    if not isinstance(old,NormalizedRiskLayers):
        model.risk_head.layers=NormalizedRiskLayers(old[0],old[1],old[2],mean,scale)
    translated=dict(payload);translated['schema']=BASE_PARAMETER_SCHEMA
    translated['configuration']=dict(payload['configuration'],schema=BASE_PARAMETER_SCHEMA)
    model.load_parameter_payload(translated)
    return model
