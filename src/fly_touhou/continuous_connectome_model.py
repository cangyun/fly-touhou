"""New continuous-connectome actor/critic; definitions never start a job.

The exact frozen D95 perception/standardization supplies384 sensory values.
Every actor and cost output then traverses the signed graph. The state and DN
features are dimensionless ANN activations, not voltages, rates or spikes.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .continuous_connectome import ContinuousConnectomeCore
from .fly_frozen_v5_perception import FrozenV5Perception
from .fly_visual_distillation import FixedStandardizedVisual,load_visual_statistics,DEFAULT_STATISTICS
from .fly_gru_aligned_model import ThreeDNValueCritics
from .fly_dn_cost_native_policy import (compose_from_active_mask,
    normal_unprotected_from_features,source_policy_logits)

SCHEMA='continuous-full-connectome-DN-actor-threevalue-cost-parameters-v1'


class ContinuousDNCostHead(nn.Module):
    """Only dimensionless DN activity enters this head; no source observations."""
    def __init__(self,width,*,beta=8.):
        super().__init__()
        if not math.isfinite(beta) or not 0<=beta<=8:raise ValueError('Explicit beta in[0,8] required')
        self.beta=float(beta)
        self.layers=nn.Sequential(nn.Linear(width,64),nn.SiLU(),nn.Linear(64,18))
        nn.init.zeros_(self.layers[-1].weight);nn.init.zeros_(self.layers[-1].bias)

    def forward(self,DN):return F.softplus(self.layers(DN))


class ContinuousConnectomeActorCritic(nn.Module):
    """One persistent whole-graph RNN with actor, DN-only cost and three critics."""
    source_schema=SCHEMA

    def __init__(self,core,visual_adapter,*,beta=8.,seed=613711):
        super().__init__()
        if not isinstance(core,ContinuousConnectomeCore) or not isinstance(visual_adapter,FixedStandardizedVisual):
            raise TypeError('New continuous core and existing exact standardized visual frontend required')
        self.core=core
        self.sensory_adapter=visual_adapter.to(core.device)
        self.sensory_adapter.requires_grad_(False).eval()
        width=len(core.output_indices)
        # Constructor RNG is private; sampling streams/global RNG are untouched.
        with torch.random.fork_rng(devices=[]),torch.device('cpu'):
            torch.random.default_generator.manual_seed(seed)  # CPU only;never seed an already-live CUDA stream.
            self.actor=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,256),nn.SiLU(),nn.Linear(256,19))
            nn.init.normal_(self.actor[-1].weight,std=.01);nn.init.zeros_(self.actor[-1].bias)
            self.critic=ThreeDNValueCritics(width)
            self.risk_head=ContinuousDNCostHead(width,beta=beta)
        self.actor.to(core.device);self.critic.to(core.device);self.risk_head.to(core.device)
        self.initialization_seed=int(seed)

    @classmethod
    def from_d95_full_connectome(cls,directory,*,checkpoint,statistics=DEFAULT_STATISTICS,
            K=8,trainable_brain=True,device='cpu',seed=613711,**core_kwargs):
        perception=FrozenV5Perception.from_d95_checkpoint(checkpoint)
        mean,std,provenance=load_visual_statistics(statistics)
        core=ContinuousConnectomeCore.from_full_connectome(directory,K=K,
            trainable_brain=trainable_brain,device=device,seed=seed,**core_kwargs)
        return cls(core,FixedStandardizedVisual(perception,mean,std,provenance),seed=seed)

    @property
    def device(self):return self.core.device

    def train(self,mode=True):
        super().train(mode)
        self.sensory_adapter.requires_grad_(False).eval()
        return self

    def initial_state(self,batch,device=None):return self.core.initial_state(batch,device=device)

    def encode_visual(self,features,maps,*,objects):
        f,m,o=(torch.as_tensor(x,dtype=torch.float32,device=self.device) for x in (features,maps,objects))
        return self.sensory_adapter(f,m,o).detach()

    def DN_from_state(self,following):return following.activity.index_select(1,self.core.output_indices)

    def heads_from_DN(self,DN,*,shared_grad_scale=.1):
        if DN.ndim!=2 or DN.shape[1]!=len(self.core.output_indices):
            raise ValueError('Same following descending ANN activity required')
        if not math.isfinite(shared_grad_scale) or not 0<=shared_grad_scale<=1:
            raise ValueError('Explicit critic shared-gradient scale in[0,1] required')
        shared=DN.detach()+shared_grad_scale*(DN-DN.detach())
        return self.actor(DN),self.critic(shared)

    def forward_encoded(self,encoded,before,*,critic_shared_grad_scale=.1):
        DN,following=self.core(encoded,before)  # Exactly one complete graph advance.
        raw,values=self.heads_from_DN(DN,shared_grad_scale=critic_shared_grad_scale)
        return raw,values,following  # Bare actor. Composition is always explicit.

    def compose(self,raw,cost,features,*,check_finite=False,beta=None):
        return compose_from_active_mask(raw,cost,normal_unprotected_from_features(features),
            beta=self.risk_head.beta if beta is None else beta,check_finite=check_finite)

    def policy_logits(self,combined,features,*,check_finite=False):
        return source_policy_logits(combined,features,check_finite=check_finite)

    def learning_parameters(self):return [p for p in self.parameters() if p.requires_grad]

    def optimizer_groups(self,*,input_lr=3e-5,brain_lr=1e-3,actor_lr=1e-3,
            critic_lr=1e-3,risk_lr=3e-4):
        groups=[('input',list(self.core.input_encoder.parameters()),input_lr),
            ('brain',list(self.core.internal_parameters()),brain_lr),
            ('actor',list(self.actor.parameters()),actor_lr),
            ('critic',list(self.critic.parameters()),critic_lr),
            ('risk',list(self.risk_head.parameters()),risk_lr)]
        return [dict(name=name,params=[p for p in ps if p.requires_grad],lr=lr)
            for name,ps,lr in groups if any(p.requires_grad for p in ps)]

    def configuration(self):
        return dict(schema=SCHEMA,core=self.core.config(),perception=self.sensory_adapter.specification(),
            actor='DN ANN activity->LayerNorm->256SiLU->19;final normal.01',
            critic='three independent DN->512ReLU->512ReLU->1;shared gradient.1',
            risk='DN ANN activity->64SiLU->18softplus;final zero',risk_beta=self.risk_head.beta,
            action_temperature=1.,teacher_hidden_or_action_bypass=False,LIF=False,
            brain_state='activity only;no neural RNG',initialization_seed=self.initialization_seed)

    def parameter_checkpoint_payload(self):
        cpu=lambda d:{k:v.detach().cpu().clone() for k,v in d.items()}
        return dict(schema=SCHEMA,configuration=self.configuration(),
            graph_provenance=copy.deepcopy(self.core.graph_provenance),
            core_parameters=cpu(dict(self.core.named_parameters())),
            actor=cpu(self.actor.state_dict()),critic=cpu(self.critic.state_dict()),
            risk_head=cpu(self.risk_head.state_dict()),
            perception_configuration=copy.deepcopy(self.sensory_adapter.perception.perception_configuration),
            perception_provenance=copy.deepcopy(self.sensory_adapter.perception.source_provenance()),
            perception=cpu(self.sensory_adapter.perception.state_dict()),
            fixed_mean=self.sensory_adapter.fixed_mean.detach().cpu().clone(),
            fixed_std=self.sensory_adapter.fixed_std.detach().cpu().clone(),
            statistics_provenance=copy.deepcopy(self.sensory_adapter.statistics_provenance),
            contains_optimizer=False,contains_neural_state=False,contains_teacher_hidden=False)

    def load_parameter_payload(self,payload):
        if payload.get('schema')!=SCHEMA:raise ValueError('Independent continuous parameter schema required')
        beta=payload['configuration']['risk_beta']
        if not isinstance(beta,(int,float)) or not math.isfinite(beta) or not 0<=beta<=8:
            raise ValueError('Finite recorded risk beta in[0,8] required')
        saved=payload['configuration']['core'];current=self.core.config()
        for key in ('n_neurons','n_edges','encoded_dim','input_ports','output_ports','K',
                'graph_fingerprint','ports_identity_sha256','normalization','epsilon'):
            if saved[key]!=current[key]:raise ValueError('Continuous core identity differs:'+key)
        if (payload['graph_provenance']!=self.core.graph_provenance
                or payload['perception_configuration']!=self.sensory_adapter.perception.perception_configuration
                or payload['statistics_provenance']!=self.sensory_adapter.statistics_provenance
                or not torch.equal(payload['fixed_mean'],self.sensory_adapter.fixed_mean.detach().cpu())
                or not torch.equal(payload['fixed_std'],self.sensory_adapter.fixed_std.detach().cpu())):
            raise ValueError('Exact source graph, perception and fixed train-only normalization required')
        targets=dict(self.core.named_parameters());incoming=payload['core_parameters']
        if set(targets)!=set(incoming):raise ValueError('Continuous input/internal parameter names differ')
        for name,value in incoming.items():
            if value.shape!=targets[name].shape or value.dtype!=torch.float32 or not bool(torch.isfinite(value).all()):
                raise ValueError('Finite source continuous parameter shape required:'+name)
        for name,module in (('actor',self.actor),('critic',self.critic),('risk_head',self.risk_head),
                ('perception',self.sensory_adapter.perception)):
            expected=module.state_dict()
            if set(payload[name])!=set(expected):raise ValueError('Saved module keys differ:'+name)
            for key,value in payload[name].items():
                if value.shape!=expected[key].shape or value.dtype!=expected[key].dtype or not bool(torch.isfinite(value).all()):
                    raise ValueError('Saved finite module shape differs:'+name+'.'+key)
        with torch.no_grad():
            for name,value in incoming.items():targets[name].copy_(value.to(self.device))
            for name,module in (('actor',self.actor),('critic',self.critic),('risk_head',self.risk_head),
                    ('perception',self.sensory_adapter.perception)):
                module.load_state_dict(payload[name],strict=True)
        self.risk_head.beta=float(payload['configuration']['risk_beta'])
        self.sensory_adapter.requires_grad_(False).eval()
        return self

    @classmethod
    def from_parameter_payload(cls,payload,core):
        if payload.get('schema')!=SCHEMA:raise ValueError('New continuous parameter payload required')
        perception=FrozenV5Perception(payload['perception_configuration'])
        perception._source_provenance=copy.deepcopy(payload['perception_provenance'])
        adapter=FixedStandardizedVisual(perception,payload['fixed_mean'],payload['fixed_std'],payload['statistics_provenance'])
        return cls(core,adapter,seed=payload['configuration']['initialization_seed']).load_parameter_payload(payload)

