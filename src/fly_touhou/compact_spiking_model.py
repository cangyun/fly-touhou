"""Full signed-connectome hard-spiking controller, compact111 input only."""
import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .spiking_engineered_control import SpikingConnectomeCore,SpikingState,HardSpike
from .compact_spiking_sensation import SCHEMA as INPUT_SCHEMA

SCHEMA='full-connectome-compact111-opponent-LIF-risk-controller-v1'


class OpponentPortDrive(nn.Module):
    """Engineered population code, all VP ports, no retinotopy claim.

    Each bounded feature has positive/negative populations. Quiet features do
    not create the old softplus(0) baseline drive at every port. A trainable
    dense residual and positive gains may modify the initial allocation.
    """
    def __init__(self,ports,*,seed,input_dim=111):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            rng=torch.Generator().manual_seed(seed)
            order=torch.randperm(ports,generator=rng)
            self.register_buffer('feature',((order//2)%input_dim).long())
            self.register_buffer('polarity',torch.where(order%2==0,1.,-1.))
            self.residual=nn.Linear(input_dim,ports,bias=True)
            nn.init.zeros_(self.residual.weight);nn.init.zeros_(self.residual.bias)
            self.gain_logits=nn.Parameter(torch.full((ports,),math.log(math.expm1(.8))))
    def forward(self,x):
        code=x[:,self.feature]*self.polarity
        return F.relu(F.softplus(self.gain_logits)*code+self.residual(x))


class CompactSpikingCore(SpikingConnectomeCore):
    sensory_width=111
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.input_encoder=OpponentPortDrive(len(self.input_indices),seed=self.initialization_seed+13,input_dim=self.sensory_width)
        self.geometry_encoder=nn.Identity();self.visual_norm=nn.Identity()
    @property
    def device(self):return self.input_encoder.residual.weight.device
    def message(self,spikes,values):return self.sparse(spikes,values)
    def forward(self,features,state):
        if features.shape!=(len(state.membrane),self.sensory_width):raise ValueError('Typed compact input and complete carried LIF state required')
        drive=torch.zeros_like(state.membrane).index_add(1,self.input_indices,self.input_encoder(features))
        values=self.effective_edge_values();alpha=self.alpha()
        syn_alpha=.05+.9*self.synaptic_logits.sigmoid();threshold=.05+F.softplus(self.threshold_logits)
        membrane,synaptic,spikes,rate=state.tensors()
        for _ in range(self.K):
            message=self.message(spikes,values)
            synaptic=(1-syn_alpha)*synaptic+syn_alpha*message
            candidate=(1-alpha)*membrane+alpha*(drive+self.neuron_bias+self.message_gain()*synaptic)
            spikes=HardSpike.apply((candidate-threshold)/threshold)*(spikes.detach()==0)
            membrane=candidate-spikes.detach()*threshold;rate=.75*rate+.25*spikes
        following=SpikingState(membrane,synaptic,spikes,rate)
        return self.read_DN(following),following
    def config(self):
        cfg=super().config();cfg.update(schema=SCHEMA,input_schema=INPUT_SCHEMA,inputs=self.sensory_width,
            sensory='opponent +/- populations plus trainable dense residual;no map/CNN/object Transformer',
            feature_to_port_allocation='seeded balanced engineered populations,not reconstructed biological retinotopy')
        return cfg


class CompactSpikingPolicy(nn.Module):
    def __init__(self,core,*,seed=202610307):
        super().__init__();self.core=core;self.seed=seed;width=2*len(core.output_indices)
        from .fly_gru_aligned_model import ThreeDNValueCritics
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            # Linear risk readout follows the useful diagnostic linear probe.
            self.risk=nn.Linear(width,18)
            nn.init.normal_(self.risk.weight,std=.001);nn.init.constant_(self.risk.bias,-.3)
            self.actor=nn.Linear(width,19)
            nn.init.zeros_(self.actor.weight);nn.init.zeros_(self.actor.bias)
            with torch.no_grad():self.actor.bias[0]=.3
            self.critic=ThreeDNValueCritics(width)
    @property
    def device(self):return self.core.device
    def initial_state(self,batch):return self.core.initial_state(batch)
    def heads(self,DN,features):
        score=self.risk(DN);danger=torch.sigmoid(8*score-1)
        raw=self.actor(DN)
        active=(features[...,14]>.5)&(features[...,13]<=0)&(features[...,5]<=0)
        # Risk has its own physical supervision. Policy guidance cannot alter
        # its calibrated output through the composition leg.
        penalty=F.pad(8*danger.detach()*active[...,None],(0,1))
        logits=(raw-penalty).clamp(-80,80)
        mask=torch.zeros_like(logits,dtype=torch.bool);mask[...,18]=True
        return dict(scores=score,danger_probabilities=danger,raw_logits=raw,logits=logits.masked_fill(mask,-torch.inf),
            predicted_margin=1-8*score)
    def parameter_payload(self,costs):
        return dict(schema=SCHEMA,input_schema=INPUT_SCHEMA,core=self.core.config(),seed=self.seed,costs=dict(costs),
            state_dict={k:v.detach().cpu().clone() for k,v in self.state_dict().items()},
            actual_binary_spikes=True,map_or_object_encoder_exists=False,raw_danger_or_teacher_bypass=False,
            risk_readout='following DN rate+membrane only;linear18;independent physical supervision')


def build_compact_model(connectome,*,seed=202610307,device='cuda:0'):
    from .connectome import load_connectome
    from .continuous_connectome import load_full_ports
    graph=load_connectome(connectome,max_neurons=None,verify=True)
    inputs,outputs,provenance=load_full_ports(graph,connectome)
    core=CompactSpikingCore(graph,inputs,outputs,K=8,trainable_brain=True,require_full_graph=True,
        port_provenance=provenance,edge_chunk_size=262144,seed=seed)
    return CompactSpikingPolicy(core,seed=seed).to(device)


def sequence(model,features,starts,state,*,checkpointed=True,actions=None):
    rows=[];DNs=[]
    for t in range(len(features)):
        state=state.reset(starts[t])
        def step(x,*states):
            DN,following=model.core(x,SpikingState(*states));return DN,*following.tensors()
        out=checkpoint(step,features[t],*state.tensors(),use_reentrant=False) if checkpointed else step(features[t],*state.tensors())
        DN,*s=out;state=SpikingState(*s);DNs.append(DN);rows.append(model.heads(DN,features[t]))
    return {**{key:torch.stack([r[key] for r in rows]) for key in rows[0]},'DN':torch.stack(DNs),'state':state}


def risk_supervision(result,task,valid):
    m=task['margins'][...,0,:].detach();score=result['scores']
    mask=valid&task['trusted'][...,0]&task['unprotected_mask']
    target=torch.tanh((1-m)/8)
    regression=F.smooth_l1_loss(score.tanh(),target,reduction='none',beta=.1)
    known=(m<=0)|(m>1)
    classification=F.binary_cross_entropy_with_logits(8*score-1,(m<=0).float(),reduction='none')
    pool=mask[...,None].expand_as(m);total=score.sum()*0;detail={}
    for name,w,q in (('danger',.25,pool&(m<=0)),('boundary',.5,pool&(m.abs()<8)),('all',.25,pool)):
        q=torch.where(q.any(),q,pool)
        total=total+w*(regression.masked_select(q).sum()/q.sum().clamp_min(1)
            +.2*classification.masked_select(q&known).sum()/(q&known).sum().clamp_min(1))
    with torch.no_grad():
        danger=pool&(m<=0);safe=pool&(m>1)
        detail=dict(danger_queries=int(danger.sum()),danger_recall=float(((result['predicted_margin']<=0)&danger).sum()/danger.sum().clamp_min(1)),
            safe_false_alarm=float(((result['predicted_margin']<=0)&safe).sum()/safe.sum().clamp_min(1)))
    return total,detail
