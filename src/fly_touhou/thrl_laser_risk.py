"""Learned short-step segment risk for explicitly versioned V5 observations.

The MLP receives relative endpoint/projection features and measured dimensions,
not analytic clearances or a teacher action. Unknown motion gets a horizon0
query identical for all actions: current geometry is retained without inventing
an action-dependent stationary four-frame forecast. Modeled capsule training
targets are conservative geometry proxies, not verified native laser hitboxes.
"""
import torch
from torch import nn
from .thrl_tensor_constants import constant_like
from .thrl_pair_risk import _action_tensor

QUERY_SCHEMA='observed-endpoint-segment-projection-native-ticks-v1'
QUERY_DIM=40


def decode(value,scale):return value*scale/(1-value.abs()).clamp_min(1e-6)


def segment_query_features(starts,velocities,a,b,va,vb,half_width,width_rate,
                           player_half,player_offset,player_circle,known,state,iframes):
    """Broadcast paired geometry to four integer native ticks, including walls."""
    known=known.to(starts.dtype);horizon=known*4
    velocities=velocities*known[...,None]
    va=va*known[...,None];vb=vb*known[...,None];width_rate=width_rate*known
    lo=constant_like(starts,[-184.,32.]);hi=constant_like(starts,[184.,432.])
    projections=[];fractions=[]
    for tick in range(5):
        t=known*tick
        player=(starts+velocities*t[...,None]).clamp(lo,hi)+player_offset
        aa=a+va*t[...,None];bb=b+vb*t[...,None];line=bb-aa
        fraction=(((player-aa)*line).sum(-1)/line.square().sum(-1).clamp_min(1e-9)).clamp(0,1)
        projections.append(torch.asinh((aa+fraction[...,None]*line-player)/8)/4.852091)
        fractions.append(fraction)
    span=b-a;length=span.square().sum(-1,keepdim=True).sqrt().clamp_min(1e-9)
    conservative=torch.maximum(player_circle+player_offset.square().sum(-1).sqrt(),player_half.square().sum(-1).sqrt())+.05
    result=torch.cat([*projections,(a-starts)/448,(b-starts)/448,va/12,vb/12,
        half_width[...,None]/32,width_rate[...,None]/12,span/768,span/length,
        velocities/4.5,((starts+velocities*4).clamp(lo,hi)-starts)/18,
        player_half/32,player_offset/8,player_circle[...,None]/32,conservative[...,None]/32,
        known[...,None],state[...,None],iframes[...,None],
        fractions[0][...,None],fractions[-1][...,None],horizon[...,None]/4],-1)
    if result.shape[-1]!=QUERY_DIM:raise ValueError('Unexpected segment query contract')
    return result


class LaserRiskNetwork(nn.Module):
    def __init__(self,width=64):
        super().__init__();self.width=width
        if type(width) is not int or width<16:raise ValueError('Positive explicit laser MLP width required')
        self.register_buffer('endpoint_segment_query_revision1',torch.tensor(1,dtype=torch.int64))
        self.network=nn.Sequential(nn.Linear(QUERY_DIM,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),
                                   nn.Linear(width,width),nn.SiLU(),nn.Linear(width,1))
    def forward(self,query):return 8*self.network(query).squeeze(-1)


def observation_queries(features,objects):
    if features.ndim!=2 or features.shape[-1]!=25 or objects.shape!=(len(features),96,29):
        raise ValueError('Laser risk requires explicitly versioned globals25 and objects96x29')
    n=len(features);laser=(objects[...,0]>.5)&((objects[...,3]>.5)|(objects[...,4]>.5))&(objects[...,13]>.5)
    # Non-laser columns can legally equal1; mask before the invertible transform.
    o=torch.where(laser[...,None],objects,torch.zeros_like(objects))
    origin=torch.stack((features[:,0]*368-184,features[:,1]*400+32),-1)
    midpoint=origin[:,None]+decode(o[...,6:8],constant_like(o,[384.,448.]))
    half=decode(o[...,21:23],constant_like(o,[384.,448.]))
    velocity=decode(o[...,8:10],12.);dv=decode(o[...,23:25],12.)
    starts=origin[:,None,None,:].expand(n,18,96,2)
    def vectors(x):return x[:,None].expand(n,18,96,2)
    def scalars(x):return x[:,None].expand(n,18,96)
    queries=segment_query_features(starts,_action_tensor(features.device,features.dtype)[None,:,None].expand_as(starts),
        vectors(midpoint-half),vectors(midpoint+half),vectors(velocity-dv),vectors(velocity+dv),
        scalars(decode(o[...,10],128.)),scalars(decode(o[...,25],12.)),
        vectors((features[:,None,14:16]*64).expand(n,96,2)),
        vectors((features[:,None,16:18]*64).expand(n,96,2)),
        scalars((features[:,None,13]*64).expand(n,96)),
        scalars(o[...,26]>.5),scalars(o[...,27]),scalars(o[...,28]))
    return queries,laser


def action_clearances(network,features,objects):
    queries,valid=observation_queries(features,objects)
    predicted=network(queries)
    return torch.where(valid[:,None],predicted,torch.full_like(predicted,64.)).amin(-1)


def action_penalties(network,features,objects,*,buffer=1.,risk_width=.5):
    """Add independent learned segment costs; unknown constant costs cancel.

    A severe but action-independent UNKNOWN beam must not shadow a directional
    known beam by winning a global min. Removing each entity's common offset
    changes no normalized action probabilities for an additive soft penalty.
    """
    queries,valid=observation_queries(features,objects)
    predicted=network(queries)
    costs=torch.nn.functional.softplus((buffer-predicted)/risk_width)
    costs=costs-costs.amin(1,keepdim=True)
    return torch.where(valid[:,None],costs,torch.zeros_like(costs)).sum(-1)


def known_margin_costs(network,features,objects,*,primary_buffer=1.,primary_width=.5,
                       comfort_buffer=4.,comfort_width=1.):
    """Worst known collision cost and bounded comfort cost, not a segment sum.

    Unknown-motion queries are direction independent. They must not shadow a
    known directional hazard through a max/min. Their current geometry stays
    in the policy input; no stationary forecast or directional cost is added.
    Identical repeated segment tokens cannot amplify either returned cost.
    """
    queries,valid=observation_queries(features,objects)
    predicted=network(queries)
    known=valid&(objects[...,26]>.5)
    primary=torch.nn.functional.softplus((primary_buffer-predicted)/primary_width)
    comfort=torch.sigmoid((comfort_buffer-predicted)/comfort_width)
    primary=torch.where(known[:,None],primary,torch.zeros_like(primary)).amax(-1)
    comfort=torch.where(known[:,None],comfort,torch.zeros_like(comfort)).amax(-1)
    return primary,comfort
