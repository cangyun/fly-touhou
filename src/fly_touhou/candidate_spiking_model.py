"""One full SNN scores all candidate actions with one shared neural readout.

This is counterfactual evaluation by the SAME weights and predecision state,
not18 experts/brains. Only the selected candidate's following state persists.
The neural input is the8 compact candidate cues; no action ID/directional
output bias, image, object list or raw numeric risk may choose an action.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .compact_spiking_model import CompactSpikingCore
from .compact_spiking_sensation import SCHEMA as INPUT_SCHEMA
from .spiking_engineered_control import SpikingState

SCHEMA='full-connectome-candidate8-shared-LIF-risk-v1'


def candidates(features):
    shape=features.shape[:-1]
    hints=features[...,18:72].reshape(*shape,3,18).transpose(-1,-2)
    known=features[...,72:75].unsqueeze(-2).expand(*shape,18,3)
    return torch.cat((hints,known,features[...,75:93,None],features[...,93:111,None]),-1)


class CandidateCore(CompactSpikingCore):
    sensory_width=8
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        with torch.no_grad():
            self.alpha_logits.fill_(0.)  # alpha=.5, shorter engineered integration lag
            self.synaptic_logits.fill_(math.log((.75-.05)/(.95-.75)))
            # Baseline .05 + neuron bias .20 remains below threshold .30.
            # Opponent changes near clearance0 now cross threshold rather
            # than disappearing inside the old +/- .125 sensory dead zone.
            self.input_encoder.residual.bias.fill_(.05)
    def config(self):
        cfg=super().config();cfg.update(schema=SCHEMA,candidate_conditioned=True,
            sensory='8 per-candidate cues to opponent VP populations,one shared brain',
            neural_ticks_per_candidate=self.K,observation_features=111)
        return cfg
    def message(self,spikes,values):
        if self.sparse.deterministic_spmm:return self.sparse(spikes,values)
        if self.training or torch.is_grad_enabled():return self.sparse(spikes,values)
        version=self.edge_gain_logits._version
        cached=getattr(self,'_inference_CSR',None)
        if cached is None or cached[0]!=version:
            op=self.sparse
            matrix=torch.sparse_csr_tensor(op.forward_crow,op.forward_columns,values.index_select(0,op.forward_permutation),
                size=(self.n_neurons,self.n_neurons),dtype=values.dtype,device=values.device)
            self._inference_CSR=(version,matrix)
        return torch.sparse.mm(self._inference_CSR[1],spikes.T.contiguous()).T.contiguous()


class CandidatePolicy(nn.Module):
    candidate_conditioned=True
    def __init__(self,core,*,seed=202610311):
        super().__init__();self.core=core;self.seed=seed
        from .fly_gru_aligned_model import ThreeDNValueCritics
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.risk=nn.Linear(2*len(core.output_indices),3)
            nn.init.normal_(self.risk.weight,std=.001);nn.init.constant_(self.risk.bias,-.3)
            self.actor=nn.Identity()  # logits are the shared neural risk readout below
            self.critic=ThreeDNValueCritics(2*len(core.output_indices))
    @property
    def device(self):return self.core.device
    def initial_state(self,batch):return self.core.initial_state(batch)
    def policy_from_scores(self,scores,features):
        probability=(8*scores-1).sigmoid()
        weights=getattr(self,'horizon_weights',(8.,1.,.3))
        logits=-weights[0]*probability[...,0]-weights[1]*probability[...,1]-weights[2]*probability[...,2]
        logits=logits.clone();logits[...,0]+=getattr(self,'stay_prior',.3)
        # Eligibility depends on predicted NEURAL immediate risk only.
        safe=probability[...,0]<.2;any_safe=safe.any(-1,keepdim=True)
        logits=logits.masked_fill(any_safe&~safe,-torch.inf)
        return dict(logits=F.pad(logits,(0,1),value=-torch.inf),predicted_margin=1-8*scores[...,0],
            scores=scores[...,0],horizon_scores=scores,danger_probabilities=probability[...,0])
    @torch.no_grad()
    def infer_candidates(self,features,states,*,committed_actions=None,return_all_DN=False):
        x=torch.as_tensor(features,device=self.device,dtype=torch.float32);n=len(x)
        cue=candidates(x).reshape(-1,8)
        before=SpikingState.pack(states)
        repeated=SpikingState(*(v.repeat_interleave(18,dim=0) for v in before.tensors()))
        outputs=[];following=[];DNs=[]
        for first in range(0,len(cue),128):
            count=min(128,len(cue)-first);part=cue[first:first+count]
            state=SpikingState(*(v[first:first+count] for v in repeated.tensors()))
            if count<128:
                part=torch.cat((part,part.new_zeros(128-count,8)))
                zero=self.initial_state(128-count)
                state=SpikingState(*(torch.cat((v,z)) for v,z in zip(state.tensors(),zero.tensors())))
            DN,after=self.core(part,state)
            outputs.append(self.risk(DN[:count]));DNs.append(DN[:count])
            following.append(SpikingState(*(v[:count] for v in after.tensors())))
        scores=torch.cat(outputs).reshape(n,18,3);out=self.policy_from_scores(scores,x)
        if return_all_DN:out['candidate_DN']=torch.cat(DNs).reshape(n,18,-1)
        action=out['logits'].argmax(-1) if committed_actions is None else committed_actions
        index=torch.arange(n,device=self.device)*18+action
        all_after=SpikingState.pack(following)
        chosen=SpikingState(*(v.index_select(0,index) for v in all_after.tensors()))
        DN=torch.cat(DNs).index_select(0,index)
        return out,[chosen.row(i) for i in range(n)],DN
    def parameter_payload(self,costs):
        return dict(schema=SCHEMA,input_schema=INPUT_SCHEMA,core=self.core.config(),seed=self.seed,costs=dict(costs),
            state_dict={k:v.detach().cpu().clone() for k,v in self.state_dict().items()},actual_binary_spikes=True,
            one_shared_brain=True,experts=1,candidates=18,neural_features_per_candidate=8,
            map_or_object_encoder_exists=False,raw_danger_or_teacher_bypass=False,
            horizon_weights=getattr(self,'horizon_weights',(8.,1.,.3)),stay_prior=getattr(self,'stay_prior',.3),
            readout_kind='dn-linear3' if isinstance(self.risk,nn.Linear) else 'DN-only-nonlinear-risk-v1',
            readout_width=None if isinstance(self.risk,nn.Linear) else self.risk.residual[0].out_features,
            brain_state_commit='selected candidate only;all candidates start from same actual preceding state')


def build_compact_model(connectome,*,seed=202610311,device='cuda:0'):
    from .connectome import load_connectome
    from .continuous_connectome import load_full_ports
    graph=load_connectome(connectome,max_neurons=None,verify=True);inputs,outputs,provenance=load_full_ports(graph,connectome)
    core=CandidateCore(graph,inputs,outputs,K=16,trainable_brain=True,require_full_graph=True,
        port_provenance=provenance,edge_chunk_size=262144,seed=seed)
    return CandidatePolicy(core,seed=seed).to(device)


def restore_parameters(model,payload):
    """Restore the complete trained readout type and deployed action scoring."""
    if payload.get('readout_kind')=='DN-only-nonlinear-risk-v1':
        from .dn_residual_risk import load_head
        state={k.removeprefix('risk.'):v for k,v in payload['state_dict'].items() if k.startswith('risk.')}
        model.risk=load_head(dict(schema='DN-only-nonlinear-risk-v1',state_dict=state,width=payload['readout_width']),device=model.device)
    model.load_state_dict(payload['state_dict'],strict=True)
    model.horizon_weights=tuple(payload.get('horizon_weights',(8.,1.,.3)))
    model.stay_prior=payload.get('stay_prior',.3)
    return model


def sequence(model,features,starts,state,*,checkpointed=True,actions=None):
    if actions is None:raise ValueError('Actual recorded source actions are required to replay neural history')
    rows=[];queries=[];q_scores=[];DNs=[];b=features.shape[1];index=torch.arange(b,device=features.device)
    for t in range(len(features)):
        state=state.reset(starts[t]);cue=candidates(features[t])
        if not checkpointed:
            out,following,DN=model.infer_candidates(features[t],[state.row(i) for i in range(b)],committed_actions=actions[t])
            state=SpikingState.pack(following);rows.append(out);DNs.append(DN);continue
        danger=cue[:,:,0]>.124;boundary=cue[:,:,0].abs()<.8
        def mix(mask):return torch.where(mask.sum(-1,keepdim=True)>0,mask.float()/mask.sum(-1,keepdim=True).clamp_min(1),torch.full_like(mask,.05555556,dtype=torch.float32))
        probability=.25*mix(danger)+.5*mix(boundary)+.25/18
        query=torch.multinomial(probability,1).squeeze(-1)
        encoded=torch.cat((cue[index,query],cue[index,actions[t]]))
        twice=SpikingState(*(torch.cat((v,v)) for v in state.tensors()))
        def step(x,*s):
            DN,after=model.core(x,SpikingState(*s));return DN,*after.tensors()
        out=checkpoint(step,encoded,*twice.tensors(),use_reentrant=False)
        DN,*following=out
        q_scores.append(model.risk(DN[:b]));queries.append(query)
        state=SpikingState(*(v[b:] for v in following));DNs.append(DN[b:])
    if not checkpointed:return {**{key:torch.stack([r[key] for r in rows]) for key in rows[0]},'state':state,'DN':torch.stack(DNs)}
    return dict(query_scores=torch.stack(q_scores),query_actions=torch.stack(queries),state=state,DN=torch.stack(DNs))


def risk_supervision(result,task,valid):
    q=result['query_actions'];target=task['margins'].gather(-1,q[...,None,None].expand(*q.shape,3,1))[...,0].detach()
    score=result['query_scores'];mask=valid[...,None]&task['trusted']&task['unprotected_mask'][...,None]
    reg=F.smooth_l1_loss(score.tanh(),torch.tanh((1-target)/8),reduction='none',beta=.1)
    known=(target<=0)|(target>1);bce=F.binary_cross_entropy_with_logits(8*score-1,(target<=0).float(),reduction='none')
    weights=score.new_tensor([1.,.3,.1]);w=mask*weights
    loss=(reg*w).sum()/w.sum().clamp_min(1)+.2*(bce*w*known).sum()/(w*known).sum().clamp_min(1)
    with torch.no_grad():
        danger=mask[...,0]&(target[...,0]<=0);safe=mask[...,0]&(target[...,0]>1);pred=(1-8*score[...,0])<=0
        detail=dict(danger_queries=int(danger.sum()),danger_recall=float((danger&pred).sum()/danger.sum().clamp_min(1)),
            safe_false_alarm=float((safe&pred).sum()/safe.sum().clamp_min(1)))
    return loss,detail
