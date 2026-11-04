"""Frozen visual384 -> whole-brain actor, separate training-only scalar critic.

Importing defines modules only. No graph, native process, CUDA, optimizer or
training is started. The critic never produces actions or enters neural state.
"""
from __future__ import annotations

import copy
from pathlib import Path

import torch
from torch import nn

from . import fly_visual_distillation as visual_student
from .fly_visual_distillation import VisualFlyDistillationStudent
from .fly_frozen_v5_perception import FrozenV5Perception

MODEL_SCHEMA='fly-visual384-wholebrain-actor-scalar-critic-v1'
PARAMETER_SCHEMA='fly-visual384-wholebrain-actor-scalar-critic-parameters-v1'
CRITIC_CONFIG=dict(input_dim=384,hidden_dim=128,output_dim=1,activation='tanh',
                   detached_visual_input=True,actor_feedback=False)
BRAIN_NAMES={'input_encoder.weight','input_encoder.bias','action_readout.weight','recurrent_gain_logits'}


class PrivilegedVisualCritic(nn.Module):
    """Independent visual baseline; no gradient into perception or actor."""
    def __init__(self):
        super().__init__()
        self.network=nn.Sequential(nn.Linear(384,128),nn.Tanh(),nn.Linear(128,1))
        nn.init.normal_(self.network[-1].weight,mean=0.,std=.001)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self,encoded_visual):
        if encoded_visual.ndim!=2 or encoded_visual.shape[1]!=384:
            raise ValueError('Critic requires current standardized visual[B,384]')
        return self.network(encoded_visual.detach()).squeeze(-1)


class VisualFlyActorCritic(nn.Module):
    """All actions use the original raw-DN bias-free19 head, without whitening.

Caller owns complete WholeBrainState and resets only on a new original game.
Use eval()+no_grad() for collection, not requires_grad_(False) on this model;
input, actor and gain remain the learner's four trainable brain parameters.
"""
    def __init__(self,student:VisualFlyDistillationStudent,*,initialize_actor):
        super().__init__()
        if not isinstance(student,VisualFlyDistillationStudent) or type(initialize_actor) is not bool:
            raise ValueError('Existing typed frozen-visual student and explicit fresh/load semantics required')
        b=student.brain
        if (b.obs_dim!=384 or b.n_actions!=19 or b.substeps!=100 or b.dt_ms!=.2 or b.recurrent_gain_limit!=.1
                or b.action_readout.bias is not None or b.geometry_input_encoder is not None
                or set(dict(b.named_parameters()))!=BRAIN_NAMES or b.recurrent_gain_logits.shape!=(b.n_neurons,)):
            raise ValueError('Keep the existing visual384/full-gain/raw-DN19 actor architecture')
        self.student=student
        self.critic=PrivilegedVisualCritic().to(student.device)
        for parameter in b.parameters():parameter.requires_grad_(True)
        student.sensory_adapter.perception.requires_grad_(False).eval()
        if initialize_actor:
            with torch.no_grad():nn.init.normal_(b.action_readout.weight,mean=0.,std=.01)
        self.fresh_actor_initialization=initialize_actor

    @property
    def device(self):return self.student.device

    @property
    def brain(self):return self.student.brain

    def initial_state(self,batch_size=1,*,rng_seed=None):
        return self.student.initial_state(batch_size,rng_seed=rng_seed)

    def train(self,mode=True):
        super().train(mode)
        self.student.sensory_adapter.perception.requires_grad_(False).eval()
        return self

    @torch.no_grad()
    def encode_visual(self,features,maps,*,objects):
        features,maps,objects=self.student._modalities(features,maps,objects)
        if features.ndim!=2 or features.shape[1]!=25:
            raise ValueError('Explicit batch of complete V5 observations required')
        encoded=self.student.sensory_adapter(features,maps,objects)
        if encoded.shape!=(len(features),384) or encoded.dtype!=torch.float32 or not bool(torch.isfinite(encoded).all()):
            raise ValueError('Frozen standardized perception must return finite FP32[B,384]')
        return encoded.detach()

    def forward_encoded(self,encoded_visual,own_state):
        encoded=torch.as_tensor(encoded_visual,dtype=torch.float32,device=self.device).detach()
        if encoded.ndim!=2 or encoded.shape[1]!=384 or not bool(torch.isfinite(encoded).all()) or own_state is None:
            raise ValueError('Finite visual[B,384] and explicit complete own state required')
        logits,following=self.student.brain(encoded,own_state)
        values=self.value_encoded(encoded)
        return logits,values,following

    def value_encoded(self,encoded_visual):
        """Bootstrap the current scalar value without advancing the brain."""
        encoded=torch.as_tensor(encoded_visual,dtype=torch.float32,device=self.device).detach()
        if encoded.ndim!=2 or encoded.shape[1]!=384 or not bool(torch.isfinite(encoded).all()):
            raise ValueError('Finite current visual[B,384] required')
        return self.critic(encoded)

    def forward_step(self,features,maps,own_state,*,objects):
        return self.forward_encoded(self.encode_visual(features,maps,objects=objects),own_state)

    def forward(self,observation,own_state):
        return self.forward_step(observation['features'],observation['maps'],own_state,objects=observation['objects'])

    def optimizer_groups(self,*,input_lr=3e-5,gain_lr=1e-3,actor_lr=1e-3,critic_lr=1e-3):
        b=self.brain
        if any(not p.requires_grad for p in b.parameters()) or any(p.requires_grad for p in self.student.sensory_adapter.perception.parameters()):
            raise ValueError('Actor/input/gain must be trainable and perception frozen')
        return [dict(name='input',params=[b.input_encoder.weight,b.input_encoder.bias],lr=input_lr),
                dict(name='gain',params=[b.recurrent_gain_logits],lr=gain_lr),
                dict(name='actor',params=[b.action_readout.weight],lr=actor_lr),
                dict(name='critic',params=list(self.critic.parameters()),lr=critic_lr)]

    def parameter_checkpoint_payload(self):
        return dict(schema=PARAMETER_SCHEMA,model_schema=MODEL_SCHEMA,critic_configuration=copy.deepcopy(CRITIC_CONFIG),
            model_source_sha256=visual_student.file_sha(__file__),student_payload=self.student.parameter_checkpoint_payload(),
            critic_state={k:v.detach().cpu().clone() for k,v in self.critic.state_dict().items()},
            actor_input='original brain.readout_features; no RMS/centering/whitening of DN',
            fresh_actor_initialization='normal std.01; checkpoint load preserves stored actor',
            optimizer_RNG_teacher_or_neural_state_saved=False,numerical_resume=False)

    @staticmethod
    def _header(payload):
        if (not isinstance(payload,dict) or payload.get('schema')!=PARAMETER_SCHEMA or payload.get('model_schema')!=MODEL_SCHEMA
                or payload.get('critic_configuration')!=CRITIC_CONFIG
                or payload.get('optimizer_RNG_teacher_or_neural_state_saved') is not False
                or payload.get('numerical_resume') is not False
                or any(k in payload for k in ('optimizer','teacher','teacher_hidden','neural_state','torch_rng','cuda_rng'))):
            raise ValueError('Independent actor/scalar-critic parameter-only schema required')
        VisualFlyDistillationStudent._header(payload.get('student_payload'))

    def load_parameter_payload(self,payload):
        self._header(payload)
        current=self.critic.state_dict();stored=payload.get('critic_state',{})
        if set(stored)!=set(current):raise ValueError('Critic state keys differ')
        for name,value in stored.items():
            if (not torch.is_tensor(value) or value.layout!=torch.strided or value.dtype!=torch.float32
                    or value.shape!=current[name].shape or not bool(torch.isfinite(value).all())):
                raise ValueError('Invalid scalar critic parameter:'+name)
        self.student.load_parameter_payload(payload['student_payload'])
        self.critic.load_state_dict(stored,strict=True)
        self.fresh_actor_initialization=False

    @classmethod
    def from_visual_student_checkpoint(cls,checkpoint,directory,*,expected_sha256,device='cpu',port_manifest=None,initialize_actor=False):
        student=VisualFlyDistillationStudent.from_visual_parameter_checkpoint(checkpoint,directory,
            expected_sha256=expected_sha256,device=device,port_manifest=port_manifest)
        return cls(student,initialize_actor=initialize_actor)

    @classmethod
    def from_actor_critic_checkpoint(cls,checkpoint,directory,*,expected_sha256,device='cpu',port_manifest=None):
        if visual_student.file_sha(checkpoint)!=expected_sha256:raise ValueError('Explicit actor-critic checkpoint SHA differs')
        payload=torch.load(checkpoint,map_location='cpu',weights_only=True);cls._header(payload)
        p=payload['student_payload'];c=p['brain_config'];f=p['frozen_perception']
        if c.get('require_full_graph') is not True:raise ValueError('Production loader requires the original full graph')
        perception=FrozenV5Perception(f['configuration']);perception.load_state_dict(f['state'],strict=True)
        perception._source_provenance=copy.deepcopy(f['provenance'])
        options={k:c[k] for k in ('dt_ms','readout_tau_ms','max_input_rate_hz','rng_seed','encoder_init_scale',
            'surrogate_scale','surrogate_slope_per_mv','parameters','recurrent_gain_limit')}
        student=VisualFlyDistillationStudent.from_full_connectome(directory,substeps=c['substeps'],device=device,
            port_manifest=port_manifest,perception=perception,fixed_mean=p['fixed_mean'],fixed_std=p['fixed_std'],
            statistics_provenance=p['statistics_provenance'],initialize_input_gain=False,
            propagation_backend=c.get('propagation_backend','coo-mm'),**options)
        model=cls(student,initialize_actor=False);model.load_parameter_payload(payload)
        return model
