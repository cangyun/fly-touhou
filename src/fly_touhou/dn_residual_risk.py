"""Small nonlinear risk readout; its only input is actual DN activity."""
import torch
from torch import nn


class DNResidualRisk(nn.Module):
    readout_kind='DN-only-nonlinear-risk-v1'
    def __init__(self,mean,std,weight,bias,*,width=128):
        super().__init__()
        self.width=width
        self.register_buffer('mean',mean.detach().clone())
        self.register_buffer('std',std.detach().clone().clamp_min(1e-4))
        self.register_buffer('base_weight',weight.detach().clone())
        self.register_buffer('base_bias',bias.detach().clone())
        self.residual=nn.Sequential(nn.Linear(len(mean),width),nn.SiLU(),nn.Linear(width,len(bias)))
        nn.init.zeros_(self.residual[-1].weight);nn.init.zeros_(self.residual[-1].bias)
        self.input_soft_clip=None
        self.base_input_soft_clip=None
    @property
    def baseline_trainable(self):return isinstance(self.base_weight,nn.Parameter)
    def enable_trainable_baseline(self):
        if self.baseline_trainable:return
        for name in ('base_weight','base_bias'):
            value=self._buffers.pop(name)
            self.register_parameter(name,nn.Parameter(value))
    def normalized(self,DN):
        z=(DN-self.mean)/self.std
        if self.input_soft_clip is not None:
            limit=self.input_soft_clip;absolute=z.abs();excess=(absolute-limit).clamp_min(0)
            bounded=z.sign()*(limit+excess/(1+excess/limit))
            # Exactly preserve calibrated ordinary values. Outliers approach
            # +/-2limit smoothly, with a positive derivative rather than a veto.
            z=torch.where(absolute>limit,bounded,z)
        return z
    def forward(self,DN):
        baseline=DN
        if self.base_input_soft_clip is not None:
            # Quiet calibration neurons can later move thousands of standard
            # deviations. Bound extrapolation in the linear branch as well as
            # the residual,without discarding ordinary values or their gradients.
            limit=self.base_input_soft_clip;z=(DN-self.mean)/self.std
            excess=(z.abs()-limit).clamp_min(0)
            bounded=z.sign()*(limit+excess/(1+excess/limit))
            baseline=torch.where(z.abs()>limit,self.mean+self.std*bounded,DN)
        return torch.nn.functional.linear(baseline,self.base_weight,self.base_bias)+self.residual(self.normalized(DN))


class DNLayerNormRisk(nn.Module):
    """Three horizon scores from DN only,with per-observation normalization."""
    readout_kind='DN-layernorm-risk-v1'
    def __init__(self,input_width,*,width=128):
        super().__init__();self.width=width
        self.input_soft_clip=None;self.base_input_soft_clip=None
        self.network=nn.Sequential(nn.LayerNorm(input_width),nn.Linear(input_width,width),nn.SiLU(),nn.Linear(width,3))
    @property
    def baseline_trainable(self):return False
    def enable_trainable_baseline(self):
        raise ValueError('LayerNorm risk readout has no inherited linear baseline;its complete MLP is trainable')
    def forward(self,DN):return self.network(DN)


def load_head(payload,*,device):
    if payload['schema']=='DN-layernorm-risk-v1':
        s=payload['state_dict'];head=DNLayerNormRisk(len(s['network.0.weight']),width=payload['width'])
        head.load_state_dict(s,strict=True);return head.to(device)
    if payload['schema']!='DN-only-nonlinear-risk-v1':raise ValueError('Typed DN-only readout required')
    s=payload['state_dict']
    head=DNResidualRisk(s['mean'],s['std'],s['base_weight'],s['base_bias'],width=payload['width'])
    head.load_state_dict(s,strict=True)
    head.input_soft_clip=payload.get('input_soft_clip')
    head.base_input_soft_clip=payload.get('base_input_soft_clip')
    if payload.get('baseline_trainable',False):head.enable_trainable_baseline()
    return head.to(device)
