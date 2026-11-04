"""Explicit DN-only128x3 risk subtype; original64 loaders remain unchanged.

The complete brain, perception, actor, critics and policy composition are reused.
Import never constructs a graph or starts a learner/native world.
"""
from __future__ import annotations
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

PARAMETER_SCHEMA='continuous-full-connectome-DN-actor-threevalue-normalized-risk128x3-parameters-v1'
TRAINING_SCHEMA='continuous-full-connectome-threevalue-normalized-risk128x3-training-v1'
TOPOLOGY=dict(name='DN1299-128SiLU-128SiLU-128SiLU-18score-softplus-v1',
    input_dim=1299,hidden_widths=[128,128,128],output_dim=18,
    parameter_tensors=8,trainable_parameters=201746,
    layer_indices=dict(first='0',middle='1',last='2'),
    risk_input='following DN only;fixed train-only normalization')


def _parts(width=1299):
    return (nn.Linear(width,128),nn.Sequential(nn.SiLU(),nn.Linear(128,128),
        nn.SiLU(),nn.Linear(128,128),nn.SiLU()),nn.Linear(128,18))


class DeepSignedClearanceReadout(nn.Module):
    """Fresh private initialization; original margin/8 fitting coordinates."""
    def __init__(self,width=1299,*,seed=71931):
        super().__init__()
        if width!=1299:raise ValueError('Explicit DN1299 capacity contrast required')
        with torch.random.fork_rng(devices=[]),torch.device('cpu'):
            torch.random.default_generator.manual_seed(seed)
            self.layers=nn.Sequential(*_parts(width))
    def forward(self,x):return 8.*self.layers(x)


class DeepNormalizedRiskLayers(nn.Sequential):
    """Keep layers[0]/[2] semantics; middle[1] owns two additional Linear layers."""
    def __init__(self,first,middle,last,mean,scale):
        super().__init__(first,middle,last)
        if (not isinstance(first,nn.Linear) or first.in_features!=1299 or first.out_features!=128
                or not isinstance(last,nn.Linear) or last.in_features!=128 or last.out_features!=18
                or not isinstance(middle,nn.Sequential) or len(middle)!=5
                or not all(isinstance(middle[i],nn.SiLU) for i in (0,2,4))
                or not all(isinstance(middle[i],nn.Linear) and middle[i].in_features==128
                           and middle[i].out_features==128 for i in (1,3))
                or mean.shape!=(1299,) or scale.shape!=mean.shape
                or mean.dtype!=torch.float32 or scale.dtype!=torch.float32
                or not bool(torch.isfinite(mean).all() and torch.isfinite(scale).all() and (scale>0).all())):
            raise ValueError('Exact normalized DN1299->128x3->18 topology required')
        self.register_buffer('DN_mean',mean.detach().to(first.weight.device).clone())
        self.register_buffer('DN_scale',scale.detach().to(first.weight.device).clone())
    def forward(self,DN):return super().forward((DN-self.DN_mean)/self.DN_scale)


class DeepDNCostHead(nn.Module):
    def __init__(self,layers,*,beta=8.):
        super().__init__()
        if not isinstance(layers,DeepNormalizedRiskLayers) or not math.isfinite(beta) or not 0<=beta<=8:
            raise ValueError('Explicit deep normalized head and original beta required')
        self.layers=layers;self.beta=float(beta)
    def forward(self,DN):return F.softplus(self.layers(DN))


def install_deep_normalized_risk(model,optimizer,signed_net,mean,scale,spec):
    """Replace only risk module/group; preserve non-risk parameter identities/Adam."""
    from .continuous_fullhistory_head_calibration import fixed_body_identity
    if not isinstance(signed_net,DeepSignedClearanceReadout):raise TypeError('Explicit deep fitted head required')
    before=fixed_body_identity(model,optimizer,exclude_actor=False)
    old=list(model.risk_head.parameters());oldids={id(p) for p in old}
    groups=[g for g in optimizer.param_groups if any(id(p) in oldids for p in g['params'])]
    if len(groups)!=1 or {id(p) for p in groups[0]['params']}!=oldids or groups[0].get('name')!='risk':
        raise ValueError('One complete named risk group required;do not remap body Adam by position')
    previous_steps=[float(optimizer.state.get(p,{}).get('step',0)) for p in old]
    removed=sum(p in optimizer.state for p in old)
    first,middle,last=copy.deepcopy(tuple(signed_net.layers))
    layers=DeepNormalizedRiskLayers(first,middle,last,mean.to(model.device),scale.to(model.device))
    with torch.no_grad():layers[2].weight.mul_(-16.);layers[2].bias.mul_(-16.).add_(2.)
    head=DeepDNCostHead(layers,beta=model.risk_head.beta).to(model.device).requires_grad_(False)
    for p in old:p.grad=None;optimizer.state.pop(p,None)
    model.risk_head=head
    risk=list(head.parameters())
    for p in risk:p.grad=None
    groups[0]['params']=risk
    if len(risk)!=8 or sum(p.numel() for p in risk)!=201746:raise RuntimeError('Risk topology changed')
    after=fixed_body_identity(model,optimizer,exclude_actor=False)
    if before!=after:raise RuntimeError('Deep risk replacement changed body/actor/critic/Adam/group settings')
    return dict(old_risk_parameter_tensors=len(old),new_risk_parameter_tensors=len(risk),
        new_risk_parameters=201746,existing_old_risk_states_removed=removed,
        old_risk_steps=previous_steps,new_risk_states_present=sum(p in optimizer.state for p in risk),
        all_nonrisk_parameters_Adam_and_group_settings_bitwise_preserved=True,
        private_signed_Adam_transferred=False,input_normalization=copy.deepcopy(spec),
        training_output_coordinate='z=margin/8',runtime_score='2-16*z')


def tag_deep_payload(payload,spec):
    """Independent typed schema; old raw/64 loader must reject this topology."""
    model=payload['model_payload'];payload['schema']=TRAINING_SCHEMA;model['schema']=PARAMETER_SCHEMA
    model['configuration']=copy.deepcopy(model['configuration'])
    model['configuration'].update(schema=PARAMETER_SCHEMA,risk=TOPOLOGY['name'],risk_topology=copy.deepcopy(TOPOLOGY))
    model['risk_input_normalization']=copy.deepcopy(spec)
    payload['algorithm']=copy.deepcopy(payload['algorithm'])
    payload['algorithm'].update(risk_topology=copy.deepcopy(TOPOLOGY),risk_input_coordinates=copy.deepcopy(spec))
    return payload


def load_deep_parameter_payload(model,payload):
    """Explicit subtype recognition, then original strict names/shapes/body checks."""
    from .continuous_connectome_model import SCHEMA as BASE_SCHEMA
    from .continuous_normalized_risk_coordinate import tensor_sha
    if (payload.get('schema')!=PARAMETER_SCHEMA
            or payload['configuration'].get('risk_topology')!=TOPOLOGY
            or payload['configuration'].get('risk')!=TOPOLOGY['name']):
        raise ValueError('Explicit deep128x3 typed parameter payload required')
    state=payload['risk_head'];mean,scale=state['layers.DN_mean'],state['layers.DN_scale']
    spec=payload['risk_input_normalization']
    if tensor_sha(mean)!=spec['mean_sha256'] or tensor_sha(scale)!=spec['scale_sha256']:
        raise ValueError('Deep risk fixed input statistics differ from recorded identity')
    with torch.random.fork_rng(devices=[]),torch.device('cpu'):
        first,middle,last=_parts()
    model.risk_head=DeepDNCostHead(DeepNormalizedRiskLayers(first,middle,last,mean,scale),
        beta=payload['configuration']['risk_beta']).to(model.device)
    translated=dict(payload,schema=BASE_SCHEMA,
        configuration=dict(payload['configuration'],schema=BASE_SCHEMA))
    model.load_parameter_payload(translated)
    return model


def build_deep_frozen_model(payload,connectome):
    """Strict new native builder; no source state or raw observation bypass."""
    from .continuous_connectome import ContinuousConnectomeCore
    from .continuous_connectome_model import ContinuousConnectomeActorCritic
    from .fly_frozen_v5_perception import FrozenV5Perception
    from .fly_visual_distillation import FixedStandardizedVisual
    if payload.get('schema')!=PARAMETER_SCHEMA:raise ValueError('Deep family builder requires its explicit schema')
    config=payload['configuration'];cfg=config['core']
    if (cfg['K']!=8 or cfg['n_neurons']!=138639 or cfg['n_edges']!=15091983 or config['risk_beta']!=8.):
        raise ValueError('Same complete K8 graph and beta8 required')
    perception=FrozenV5Perception(payload['perception_configuration'])
    perception._source_provenance=copy.deepcopy(payload['perception_provenance'])
    adapter=FixedStandardizedVisual(perception,payload['fixed_mean'],payload['fixed_std'],payload['statistics_provenance'])
    core=ContinuousConnectomeCore.from_full_connectome(connectome,K=8,trainable_brain=True,device='cpu',
        seed=cfg['seed'],epsilon=cfg['epsilon'],edge_chunk_size=cfg['sparse']['edge_chunk_size'])
    model=ContinuousConnectomeActorCritic(core,adapter,seed=config['initialization_seed']).to('cuda:0')
    load_deep_parameter_payload(model,payload)
    return model.eval().requires_grad_(False)


def restore_deep_training_checkpoint(model,optimizer,payload,*,restore_rng=False):
    """Caller first installs deep parameters,then constructs named optimizer groups."""
    from .continuous_connectome_training import restore_training_checkpoint,SCHEMA as BASE_TRAINING_SCHEMA
    from .continuous_connectome_model import SCHEMA as BASE_PARAMETER_SCHEMA
    if payload.get('schema')!=TRAINING_SCHEMA or not isinstance(model.risk_head,DeepDNCostHead):
        raise ValueError('Explicit deep model and full deep training checkpoint required')
    if payload['model_payload']['configuration'].get('risk_topology')!=TOPOLOGY:
        raise ValueError('Deep training topology differs')
    p=payload['model_payload']
    translated=dict(payload,schema=BASE_TRAINING_SCHEMA,model_payload=dict(p,schema=BASE_PARAMETER_SCHEMA,
        configuration=dict(p['configuration'],schema=BASE_PARAMETER_SCHEMA)))
    return restore_training_checkpoint(model,optimizer,translated,restore_rng=restore_rng)
