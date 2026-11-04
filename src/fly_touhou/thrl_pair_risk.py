"""Learned action-conditioned entity risk, using existing B1 observations only.

No analytic collision clearance/teacher runs at inference. Coordinates and
native motion endpoints are geometric features; an MLP learns their relation
to the measured effective dimensions. All18 actions remain legal.
"""
import torch
from torch import nn
from functools import lru_cache
from .thrl_tensor_constants import constant_like

QUERY_DIM = 28
QUERY_SCHEMA = 'relative-pair-clamped-closest-v3'


def query_features(starts, velocities, positions, motions, circle, radii, extents, offsets):
    """Broadcastable current geometry and proposed4-frame motion, no risk labels."""
    relative=positions-starts;speed=motions-velocities
    lo=constant_like(starts,[-184.,32.]);hi=constant_like(starts,[184.,432.])
    points=[]
    for time in (0.,1.,2.,3.,4.):
        player=(starts+velocities*time).clamp(lo,hi)
        points.append(torch.asinh((positions+motions*time-player)/8)/4.852091)
    wall=torch.where(velocities>0,hi,lo)
    moving=velocities.abs()>1e-12
    times=torch.where(moving,(wall-starts)/torch.where(moving,velocities,torch.ones_like(velocities)),
                      torch.full_like(velocities,float('inf'))).clamp(0,4).sort(-1).values
    bounds=torch.cat((torch.zeros_like(times[...,:1]),times,torch.full_like(times[...,:1],4.)),-1)
    candidates=[];distances=[];clocks=[];speeds=[]
    for segment in range(3):
        first,last=bounds[...,segment],bounds[...,segment+1];duration=last-first
        midpoint=starts+velocities*((first+last)/2)[...,None]
        actual_velocity=velocities*((midpoint>lo)&(midpoint<hi))
        rel=positions+motions*first[...,None]-(starts+velocities*first[...,None]).clamp(lo,hi)
        rv=motions-actual_velocity;sq=rv.square().sum(-1)
        local=torch.minimum((-(rel*rv).sum(-1)/(sq+1e-6)).clamp_min(0),duration)
        point=rel+rv*local[...,None]
        candidates.append(point);distances.append(torch.where(duration>1e-10,point.square().sum(-1),torch.full_like(sq,1e16)))
        clocks.append(first+local);speeds.append(sq)
    selected=torch.stack(distances,-1).argmin(-1)
    closest=torch.stack(candidates,-2).gather(-2,selected[...,None,None].expand(*selected.shape,1,2)).squeeze(-2)
    tau=torch.stack(clocks,-1).gather(-1,selected[...,None]).squeeze(-1)
    squared=torch.stack(speeds,-1).gather(-1,selected[...,None]).squeeze(-1)
    # Closest-approach coordinates are not a clearance or safety decision.
    # Shapes/dimensions remain separate, and wall endpoints remain explicit.
    displacement=((starts+velocities*4).clamp(lo,hi)-starts)/18
    values=[*points,torch.asinh(closest/8)/4.852091,motions/12,velocities/4.5,
            relative/384,displacement,
            radii[...,None]/32,extents/32,offsets/8,circle[...,None].to(starts.dtype),
            tau[...,None]/4,squared[...,None]/144]
    result=torch.cat(values,-1)
    if result.shape[-1]!=QUERY_DIM:
        raise ValueError('Unexpected action/entity feature contract')
    return result


class PairRiskNetwork(nn.Module):
    def __init__(self,width=128):
        super().__init__();self.width=width
        # An old feature contract must fail a strict checkpoint load.
        self.register_buffer('query_schema_revision3',torch.tensor(3,dtype=torch.int64))
        self.network=nn.Sequential(nn.Linear(QUERY_DIM,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),
                                   nn.Linear(width,width),nn.SiLU(),nn.Linear(width,1))
    def forward(self,queries):
        return 8*self.network(queries).squeeze(-1)


@lru_cache(maxsize=16)
def _action_tensor(device,dtype):
    from .thrl_movement_teacher import action_velocities
    return torch.as_tensor(action_velocities(),device=device,dtype=dtype)


def observation_queries(features,objects):
    if features.ndim!=2 or features.shape[-1]!=25 or objects.shape!=(len(features),96,21):
        raise ValueError('Require the existing B1 globals25 and independent objects96x21')
    n=len(features)
    origin=torch.stack((features[:,0]*368-184,features[:,1]*400+32),-1)
    starts=origin[:,None,None,:].expand(n,18,96,2)
    velocities=_action_tensor(features.device,features.dtype)[None,:,None,:].expand_as(starts)
    positions=origin[:,None,:]+objects[...,6:8]*constant_like(objects,[384.,448.])
    positions=positions[:,None].expand_as(starts);motions=(objects[...,8:10]*12)[:,None].expand_as(starts)
    circle=(objects[...,1]>.5)&(objects[...,12]>.5)
    radius=(objects[...,10].square()*128**2+features[:,None,13].square()*64**2).sqrt()+.05
    extents=objects[...,10:12]*128+features[:,None,14:16]*64+.05
    offsets=features[:,None,16:18].expand(n,96,2)*64
    queries=query_features(starts,velocities,positions,motions,circle[:,None].expand(n,18,96),
        radius[:,None].expand(n,18,96),extents[:,None].expand_as(starts),offsets[:,None].expand_as(starts))
    valid=(objects[...,0]>.5)&((objects[...,1]>.5)|(objects[...,2]>.5))&(objects[...,13]>.5)
    return queries,valid


def action_clearances(network,features,objects):
    queries,valid=observation_queries(features,objects)
    predicted=network(queries)
    return torch.where(valid[:,None],predicted,torch.full_like(predicted,64.)).amin(-1)
