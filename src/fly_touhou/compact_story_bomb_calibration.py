"""Bomb-only calibration of the existing DN readout, with both teacher classes.

The scalar loss is the Bomb-dependent part of the existing 19-action rule CE
and deathbomb CE. Movement logits, neural risk and the connectome stay fixed
during calibration. This is teacher replay, not fresh on-policy PPO data.
"""
import torch
from torch.nn import functional as F


def log_odds(head, gain, data):
    bomb=(head(data['DN'])[:,0]+gain*data['urgency']).clamp(-80.,80.)
    return bomb-data['movement_logmass']


def loss(logits, teacher, weights, deathbomb):
    rule=(F.binary_cross_entropy_with_logits(logits,teacher,reduction='none')*weights).sum()/weights.sum().clamp_min(1)
    critical=F.softplus(-logits)[deathbomb].sum()/deathbomb.sum().clamp_min(1)
    return .1*rule+.05*critical


@torch.no_grad()
def metrics(logits, teacher, weights, deathbomb):
    normal=(weights>0)&(teacher<=1e-6)&~deathbomb
    p=logits.sigmoid()
    if not bool(normal.any()) or not bool(deathbomb.any()):
        raise ValueError('Both ordinary stocked teacher rows and actual deathbomb rows are required')
    return dict(normal_rows=int(normal.sum()),deathbomb_rows=int(deathbomb.sum()),
        normal_mean_Bomb_probability=float(p[normal].mean()),
        normal_max_Bomb_probability=float(p[normal].max()),
        normal_family_Bomb_choices=int((p[normal]>.5).sum()),
        deathbomb_mean_probability=float(p[deathbomb].mean()),
        deathbomb_family_choices=int((p[deathbomb]>.5).sum()),
        balanced_BCE=float((F.softplus(logits[normal]).mean()+F.softplus(-logits[deathbomb]).mean())/2))


def reset_bomb_moments(saved):
    groups=[g for g in saved['param_groups'] if g.get('name')=='Bomb']
    if len(groups)!=1:raise ValueError('One named Bomb Adam group is required')
    remove=set(groups[0]['params'])
    return dict(saved,state={k:v for k,v in saved['state'].items() if k not in remove})
