"""B1 policy plus learned per-action/entity short-motion readout."""
import math
import torch
from torch import nn
from .thrl_model import MOActorCritic as BaseB1
from .thrl_pair_risk import PairRiskNetwork,action_clearances,QUERY_SCHEMA
from .thrl_action_risk_readout import adjust_movement_logits
from .thrl_inputs import PRIORITY_INPUT_SCHEMAS,SEPARATED_LASER_INPUT,object_dimension


class EntityRiskPolicy(nn.Module):
    def __init__(self,base_model_config,width=128,beta=8.,buffer=1.,risk_width=.5,scope='normal-unprotected-v1',query_schema=QUERY_SCHEMA,frozen_readout=False,
                 laser_width=None,laser_query_schema=None,laser_buffer=4.,laser_risk_width=1.,
                 aggregation='sum-directional-v1',laser_buffer_weight=.1):
        super().__init__()
        if (not math.isfinite(beta) or beta<0 or not math.isfinite(buffer) or buffer<0
                or not math.isfinite(risk_width) or risk_width<=0):
            raise ValueError('Finite soft risk settings required')
        self.base=BaseB1(**base_model_config);self.risk=PairRiskNetwork(width)
        if not math.isfinite(laser_buffer) or laser_buffer<0 or not math.isfinite(laser_risk_width) or laser_risk_width<=0:
            raise ValueError('Finite explicit laser uncertainty margin required')
        self.laser_buffer=float(laser_buffer);self.laser_risk_width=float(laser_risk_width)
        if aggregation not in ('sum-directional-v1','worst-known-margin-v2'):
            raise ValueError('Explicit supported neural risk aggregation required')
        if not math.isfinite(laser_buffer_weight) or not 0<=laser_buffer_weight<=1:
            raise ValueError('Finite bounded secondary laser comfort weight required')
        self.aggregation=aggregation;self.laser_buffer_weight=float(laser_buffer_weight)
        self.laser_risk=None
        if self.base.input_schema==SEPARATED_LASER_INPUT:
            from .thrl_laser_risk import LaserRiskNetwork,QUERY_SCHEMA as LASER_QUERY
            if laser_width is None or laser_query_schema!=LASER_QUERY:
                raise ValueError('V5 requires its explicitly fitted laser risk network and query schema')
            self.laser_risk=LaserRiskNetwork(laser_width)
        elif laser_width is not None or laser_query_schema is not None:
            raise ValueError('Laser risk must not reinterpret historical input schemas')
        if type(frozen_readout) is not bool:raise ValueError('Explicit frozen readout flag required')
        self.frozen_readout=frozen_readout
        self.risk.requires_grad_(not frozen_readout)
        if self.laser_risk is not None:self.laser_risk.requires_grad_(not frozen_readout)
        if scope!='normal-unprotected-v1':raise ValueError('The fitted physical readout has only normal unprotected labels')
        if query_schema!=QUERY_SCHEMA:raise ValueError('Pair readout query schema differs from the fitted feature contract')
        if not self.base.is_recurrent or self.base.input_schema not in PRIORITY_INPUT_SCHEMAS:
            raise ValueError('Prototype requires the existing separated B1 GRU')
        self.input_schema=self.base.input_schema;self.gru_hidden_size=self.base.gru_hidden_size
        self.is_recurrent=True;self.beta=float(beta);self.buffer=float(buffer);self.risk_width=float(risk_width)
        self.base_configuration=dict(base_model_config)

    @property
    def gru(self):return self.base.gru

    def _encode(self,features,maps,objects=None):return self.base._encode(features,maps,objects)

    def _risk_scores(self,features,objects):
        predicted=action_clearances(self.risk,features,objects[...,:21])
        risk=(self.buffer-predicted)/self.risk_width
        if self.laser_risk is not None:
            if self.aggregation=='sum-directional-v1':
                from .thrl_laser_risk import action_penalties
                laser=action_penalties(self.laser_risk,features,objects,buffer=self.laser_buffer,risk_width=self.laser_risk_width)
                # Historical operator retained exactly for old checkpoints.
                cost=torch.nn.functional.softplus(risk)+laser
            else:
                from .thrl_laser_risk import known_margin_costs
                laser,comfort=known_margin_costs(self.laser_risk,features,objects,
                    primary_buffer=self.buffer,primary_width=self.risk_width,
                    comfort_buffer=self.laser_buffer,comfort_width=self.laser_risk_width)
                # A repeated field of safe-but-close beams must not outweigh
                # one predicted actual collision. The primary cost compares
                # worst learned clearances in the same units across types;
                # the separate comfort term is bounded and count invariant.
                cost=torch.maximum(torch.nn.functional.softplus(risk),laser)+self.laser_buffer_weight*comfort
            cost=cost.clamp_min(1e-30)
            risk=cost+torch.log(-torch.expm1(-cost))
        return risk

    def precompute_frozen_risk(self,features,objects):
        if not self.frozen_readout or any(p.requires_grad for p in self.risk.parameters()) or (self.laser_risk is not None and any(p.requires_grad for p in self.laser_risk.parameters())):
            raise ValueError('Only the explicitly frozen entity readout can be cached')
        with torch.no_grad():
            return self._risk_scores(features,objects).detach()

    def _adjust(self,logits,features,objects,*,precomputed_risk=None,check_finite=True):
        if self.beta==0:return logits
        if precomputed_risk is None:
            risk=self._risk_scores(features,objects)
        else:
            if (not self.frozen_readout or any(p.requires_grad for p in self.risk.parameters())
                    or (self.laser_risk is not None and any(p.requires_grad for p in self.laser_risk.parameters()))
                    or precomputed_risk.shape!=(len(features),18)
                    or precomputed_risk.device!=logits.device or precomputed_risk.dtype!=logits.dtype
                    or precomputed_risk.requires_grad):
                raise ValueError('Frozen matched per-state risk cache required')
            risk=precomputed_risk
        adjusted=adjust_movement_logits(logits,risk,self.beta,check_finite=check_finite)
        # Label-domain validity only: no risk-based action mask or Bomb rule.
        # A frozen collision predictor fitted on vulnerable frames must not
        # invent costs during native protection, respawn or deathbomb states.
        active=(features[:,21]>.5)&(features[:,20]<=0)&(features[:,5]<=0)
        return torch.where(active[:,None],adjusted,logits)

    def forward_step(self,features,maps,hidden=None,*,objects=None,check_finite=True):
        logits,values,following=self.base.forward_step(features,maps,hidden,objects=objects)
        return self._adjust(logits,features,objects,check_finite=check_finite),values,following

    def forward_sequence(self,features_seq,maps_seq,initial_hidden,dones_seq,
                         critic_shared_grad_scale=.1,*,objects=None,episode_hidden=None,
                         precomputed_risk=None):
        logits,values=self.base.forward_sequence(features_seq,maps_seq,initial_hidden,dones_seq,
            critic_shared_grad_scale,objects=objects,episode_hidden=episode_hidden)
        return self._adjust(logits,features_seq.reshape(-1,25),objects.reshape(-1,96,object_dimension(self.input_schema)),
            precomputed_risk=precomputed_risk.reshape(-1,18) if precomputed_risk is not None else None),values

    def model_config(self):
        wrapper=dict(
            base_model_config=self.base_configuration,width=self.risk.width,beta=self.beta,
            buffer=self.buffer,risk_width=self.risk_width,scope='normal-unprotected-v1',query_schema=QUERY_SCHEMA)
        if self.frozen_readout:wrapper['frozen_readout']=True
        if self.laser_risk is not None:
            from .thrl_laser_risk import QUERY_SCHEMA as LASER_QUERY
            wrapper.update(laser_width=self.laser_risk.width,laser_query_schema=LASER_QUERY,
                laser_buffer=self.laser_buffer,laser_risk_width=self.laser_risk_width)
        if self.aggregation!='sum-directional-v1':
            wrapper.update(aggregation=self.aggregation,laser_buffer_weight=self.laser_buffer_weight)
        return dict(input_schema=self.input_schema,temporal_core='gru',pair_risk_wrapper=wrapper)

    def attention_specification(self):return self.base.attention_specification()

    def parameter_counts(self):
        counts=self.base.parameter_counts()
        extra=sum(p.numel() for p in self.risk.parameters())
        laser=sum(p.numel() for p in self.laser_risk.parameters()) if self.laser_risk is not None else 0
        result={**counts,'entity_risk_readout':extra,'total':counts['total']+extra+laser}
        if self.laser_risk is not None:result['laser_risk_readout']=laser
        return result


def model_from_config(**configuration):
    wrapper=configuration.get('pair_risk_wrapper')
    if wrapper is None:
        from .thrl_risk_policy import model_from_config as previous_factory
        return previous_factory(**configuration)
    if configuration.get('input_schema')!=wrapper['base_model_config']['input_schema'] or configuration.get('temporal_core')!='gru':
        raise ValueError('Pair policy input/recurrence metadata differ')
    return EntityRiskPolicy(**wrapper)
