"""Independent whole-brain actor with three GRU-structured DN value critics.

No existing model/core/runner is changed or started. All actions still use the
original whole-brain raw-DN19 readout. Values use that SAME following DN state,
not present visual384 alone. Scalar-value/Adam state is never relabeled as V3.
"""
from __future__ import annotations

import copy
from dataclasses import fields
import math

import torch
from torch import nn

from .fly_visual_actor_critic import VisualFlyActorCritic
from .fly_visual_distillation import VisualFlyDistillationStudent, file_sha
from .fly_frozen_v5_perception import FrozenV5Perception
from .whole_brain import WholeBrainState

MODEL_SCHEMA='fly-visual-wholebrain-rawDN19-three-DN-critics-GRU-aligned-v1'
PARAMETER_SCHEMA='fly-visual-wholebrain-three-DN-critic-parameters-v1'
SHARED_GRAD_SCALE=.1


def critic_configuration(input_dim):
    return dict(input_dim=input_dim,heads=3,hidden_dims=[512,512],activation='relu',
        input='same following brain.readout_features(rate/reference150Hz)',DN_whitening=False,
        shared_gradient_scale=SHARED_GRAD_SCALE,final_weight_std=.001,final_bias=0.,
        actor_feedback=False,old_scalar_critic_inherited=False,D95_critic_inherited=False)


class ThreeDNValueCritics(nn.Module):
    """Three independent DN->512 ReLU->512 ReLU->1 value paths."""
    def __init__(self,input_dim):
        super().__init__()
        if type(input_dim) is not int or input_dim<1:raise ValueError('Existing descending-port dimension required')
        self.input_dim=input_dim
        self.trunks=nn.ModuleList([nn.Sequential(nn.Linear(input_dim,512),nn.ReLU(),
            nn.Linear(512,512),nn.ReLU()) for _ in range(3)])
        self.heads=nn.ModuleList([nn.Linear(512,1) for _ in range(3)])
        for head in self.heads:
            nn.init.normal_(head.weight,mean=0.,std=.001);nn.init.zeros_(head.bias)

    def forward(self,DN_features):
        if DN_features.ndim!=2 or DN_features.shape[1]!=self.input_dim:
            raise ValueError('Three critics require the same following DN[B,ports] activity')
        return torch.cat([head(trunk(DN_features)) for trunk,head in zip(self.trunks,self.heads)],-1)


def _student_from_payload(payload,directory,*,device,port_manifest):
    """Same pinned full-student reconstruction as the old explicit loader."""
    c=payload['brain_config'];f=payload['frozen_perception']
    if c.get('require_full_graph') is not True:raise ValueError('Production three-value loader requires the full graph')
    perception=FrozenV5Perception(f['configuration']);perception.load_state_dict(f['state'],strict=True)
    perception._source_provenance=copy.deepcopy(f['provenance'])
    options={k:c[k] for k in ('dt_ms','readout_tau_ms','max_input_rate_hz','rng_seed','encoder_init_scale',
        'surrogate_scale','surrogate_slope_per_mv','parameters','recurrent_gain_limit')}
    return VisualFlyDistillationStudent.from_full_connectome(directory,substeps=c['substeps'],device=device,
        port_manifest=port_manifest,perception=perception,fixed_mean=payload['fixed_mean'],fixed_std=payload['fixed_std'],
        statistics_provenance=payload['statistics_provenance'],initialize_input_gain=False,
        propagation_backend=c.get('propagation_backend','coo-mm'),**options)


class FlyGRUAlignedActorCritic(VisualFlyActorCritic):
    """Original actor/input/gain, new independent V3 heads on the following brain.

    Constructor may adopt an existing student. initialize_actor=False preserves
    its actual readout. Full typed loading preserves all three trained critics;
    it never runs a fresh actor initialization after loading trained weights.
    """
    def __init__(self,student,*,initialize_actor=False):
        super().__init__(student,initialize_actor=initialize_actor)
        # The old scalar critic is discarded, including all of its parameters.
        self.critic=ThreeDNValueCritics(len(self.brain.output_indices)).to(self.device)
        self.fresh_critic_initialization=True
        self.migration_provenance=None

    @classmethod
    def from_single_critic_model(cls,source):
        if type(source) is not VisualFlyActorCritic:
            raise ValueError('Explicit original single-critic actor model required for migration')
        result=cls(source.student,initialize_actor=False)
        result.migration_provenance=dict(student_parameter_storage_adopted=True,
            original_four_brain_parameters_preserved=True,single_visual_critic_discarded=True,
            three_DN_critics_newly_initialized=True,old_critic_or_optimizer_loaded=False,
            numerical_resume=False)
        return result

    @classmethod
    def from_single_critic_checkpoint(cls,checkpoint,directory,*,expected_sha256,device='cpu',port_manifest=None):
        source=VisualFlyActorCritic.from_actor_critic_checkpoint(checkpoint,directory,
            expected_sha256=expected_sha256,device=device,port_manifest=port_manifest)
        result=cls.from_single_critic_model(source)
        result.migration_provenance.update(source_checkpoint=str(checkpoint),source_sha256=expected_sha256)
        return result

    def value_from_following_state(self,following,shared_grad_scale=SHARED_GRAD_SCALE):
        """No extra brain forward. Only the shared-state gradient is scaled."""
        if (isinstance(shared_grad_scale,bool) or not isinstance(shared_grad_scale,(float,int))
                or not math.isfinite(shared_grad_scale) or not 0<=shared_grad_scale<=1):
            raise ValueError('Explicit shared-state gradient scale in[0,1] required')
        if not isinstance(following,WholeBrainState):raise ValueError('Own complete following brain state required')
        self.brain._check_state(following,following.membrane.shape[0])
        DN=self.brain.readout_features(following)
        # The existing GRU sequence critic uses exactly this gradient bridge.
        # Forward values are unchanged; critic parameter gradients are unscaled.
        shared=DN.detach()+shared_grad_scale*(DN-DN.detach())
        return self.critic(shared)

    def forward_encoded(self,encoded_visual,own_state):
        encoded=torch.as_tensor(encoded_visual,dtype=torch.float32,device=self.device).detach()
        if encoded.ndim!=2 or encoded.shape[1]!=384 or not bool(torch.isfinite(encoded).all()) or own_state is None:
            raise ValueError('Finite visual[B,384] and complete explicit own state required')
        logits,following=self.brain(encoded,own_state)
        return logits,self.value_from_following_state(following),following

    def value_encoded(self,encoded_visual):
        raise ValueError('V3 requires following DN activity; visual384-only bootstrap is forbidden')

    @torch.no_grad()
    def bootstrap_values(self,encoded_visual,own_state):
        """One PAID branch brain forward per row; return only V3, never commit it.

        Caller must count this additional neural work. No selected action,
        observation, RNG or live-state cursor is consumed by the branch.
        """
        if not isinstance(own_state,WholeBrainState):raise ValueError('Explicit own state required')
        branch={}
        for f in fields(own_state):
            x=getattr(own_state,f.name)
            branch[f.name]=tuple(v.detach().clone() for v in x) if f.name=='delay_queue' else x.detach().clone()
        _,values,_=self.forward_encoded(encoded_visual,WholeBrainState(**branch))
        return values

    # The inherited optimizer_groups uses the new self.critic:input2,gain1,
    # actor1,critic18 tensors. It does not migrate/create any Adam state.
    def parameter_checkpoint_payload(self):
        return dict(schema=PARAMETER_SCHEMA,model_schema=MODEL_SCHEMA,model_source_sha256=file_sha(__file__),
            critic_configuration=critic_configuration(len(self.brain.output_indices)),
            student_payload=self.student.parameter_checkpoint_payload(),
            critic_state={k:v.detach().cpu().clone() for k,v in self.critic.state_dict().items()},
            migration_provenance=copy.deepcopy(self.migration_provenance),
            critic_initialization='three fresh heads on explicit migration; typed load preserves saved heads',
            optimizer_RNG_teacher_or_neural_state_saved=False,numerical_resume=False)

    @staticmethod
    def _header(payload):
        if (not isinstance(payload,dict) or payload.get('schema')!=PARAMETER_SCHEMA
                or payload.get('model_schema')!=MODEL_SCHEMA or payload.get('model_source_sha256')!=file_sha(__file__)
                or payload.get('optimizer_RNG_teacher_or_neural_state_saved') is not False
                or payload.get('numerical_resume') is not False
                or any(k in payload for k in ('optimizer','teacher','teacher_hidden','neural_state','torch_rng','cuda_rng'))):
            raise ValueError('New three-DN-value parameter schema required; not a single-head/value/Adam resume')
        VisualFlyDistillationStudent._header(payload.get('student_payload'))
        c=payload['student_payload']['brain_config'];outputs=c.get('output_indices')
        if not isinstance(outputs,list) or not outputs or payload.get('critic_configuration')!=critic_configuration(len(outputs)):
            raise ValueError('Original descending-port-bound three-head configuration required')

    def load_parameter_payload(self,payload):
        self._header(payload)
        if payload['critic_configuration']!=critic_configuration(len(self.brain.output_indices)):
            raise ValueError('Current descending-port dimension differs')
        current=self.critic.state_dict();stored=payload.get('critic_state',{})
        if set(current)!=set(stored):raise ValueError('All three independent critic parameter names required')
        for name,value in stored.items():
            if (not torch.is_tensor(value) or value.layout!=torch.strided or value.dtype!=torch.float32
                    or value.shape!=current[name].shape or not bool(torch.isfinite(value).all())):
                raise ValueError('Invalid three-head parameter:'+name)
        self.student.load_parameter_payload(payload['student_payload'])
        self.critic.load_state_dict(stored,strict=True)
        self.migration_provenance=copy.deepcopy(payload.get('migration_provenance'))
        self.fresh_actor_initialization=False;self.fresh_critic_initialization=False

    @classmethod
    def from_actor_critic_checkpoint(cls,checkpoint,directory,*,expected_sha256,device='cpu',port_manifest=None):
        if file_sha(checkpoint)!=expected_sha256:raise ValueError('Explicit three-head checkpoint SHA mismatch')
        payload=torch.load(checkpoint,map_location='cpu',weights_only=True);cls._header(payload)
        student=_student_from_payload(payload['student_payload'],directory,device=device,port_manifest=port_manifest)
        result=cls(student,initialize_actor=False);result.load_parameter_payload(payload)
        return result
