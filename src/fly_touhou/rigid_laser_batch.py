"""Batch the existing exact2D segment-distance teacher calculation only."""
import math
import numpy as np


def segment_distances(a,b,c,d):
    """Broadcast matching (...,2) segment pairs; same epsilon as planner.py."""
    a,b,c,d=np.broadcast_arrays(*[np.asarray(x,dtype=float) for x in (a,b,c,d)])
    ab,cd=b-a,d-c
    cross=lambda x,y:x[...,0]*y[...,1]-x[...,1]*y[...,0]
    denominator=cross(ab,cd);nonparallel=np.abs(denominator)>1e-12
    t=np.divide(cross(c-a,cd),denominator,out=np.zeros_like(denominator),where=nonparallel)
    u=np.divide(cross(c-a,ab),denominator,out=np.zeros_like(denominator),where=nonparallel)
    intersection=nonparallel&(t>=0)&(t<=1)&(u>=0)&(u<=1)
    def point_distance(p,first,last):
        delta=last-first;squared=np.einsum('...i,...i->...',delta,delta)
        fraction=np.divide(np.einsum('...i,...i->...',p-first,delta),squared,
            out=np.zeros_like(squared),where=squared>1e-12).clip(0,1)
        return np.linalg.norm(p-(first+fraction[...,None]*delta),axis=-1)
    answer=np.minimum.reduce([point_distance(a,c,d),point_distance(b,c,d),
        point_distance(c,a,b),point_distance(d,a,b)])
    return np.where(intersection,0.,answer)


def prepare_segments(rows):
    rows=tuple(rows)
    return (np.asarray([r[0] for r in rows],float).reshape(-1,2),
        np.asarray([r[1] for r in rows],float).reshape(-1,2),
        np.asarray([r[2] for r in rows],float).reshape(-1,2),
        np.asarray([r[3] for r in rows],float))


def path_segment_clearance(arrays,intercept,velocity,start,last,player_radius):
    a,b,translations,widths=arrays
    if not len(widths):return math.inf
    movement=np.asarray(velocity)-translations
    first=np.asarray(intercept)+movement*start;final=np.asarray(intercept)+movement*last
    return float((segment_distances(first,final,a,b)-player_radius-widths).min())
