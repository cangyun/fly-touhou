"""Evaluation prototype: one B1 encoder/GRU with a learned movement readout."""
import torch
from torch import nn
from .thrl_model import MOActorCritic as BaseB1
from .thrl_action_risk_readout import ActionRiskReadout,adjust_movement_logits


class LearnedRiskPolicy(nn.Module):
    def __init__(self,base_model_config,input_dim=640,hidden_dim=128,beta=4.):
        super().__init__();self.base=BaseB1(**base_model_config)
        if not self.base.is_recurrent:raise ValueError('This measured readout requires B1 GRU')
        self.risk=ActionRiskReadout(input_dim,hidden_dim);self.beta=float(beta)
        self.input_schema=self.base.input_schema;self.gru_hidden_size=self.base.gru_hidden_size
        self.is_recurrent=True
        self.configuration=dict(input_schema=self.input_schema,temporal_core='gru',risk_wrapper=dict(
            base_model_config=base_model_config,input_dim=input_dim,hidden_dim=hidden_dim,beta=self.beta))

    def model_config(self):
        return {**self.configuration,'risk_wrapper':{**self.configuration['risk_wrapper'],'beta':self.beta}}

    def forward_step(self,features,maps,hidden=None,*,objects=None):
        encoded=self.base._encode(features,maps,objects)
        if hidden is None:hidden=encoded.new_zeros(1,len(encoded),self.gru_hidden_size)
        state,following=self.base.gru(encoded[None],hidden.contiguous());state=state[0]
        logits=self.base._policy_logits(state,features)
        risk=self.risk(torch.cat((encoded,state),-1))
        return adjust_movement_logits(logits,risk,self.beta),self.base._critics(state),following


def model_from_config(**configuration):
    wrapper=configuration.get('risk_wrapper')
    if wrapper is None:return BaseB1(**configuration)
    if configuration.get('input_schema')!=wrapper['base_model_config']['input_schema'] or configuration.get('temporal_core')!='gru':
        raise ValueError('Wrapper observation and recurrence metadata differ')
    return LearnedRiskPolicy(**wrapper)
