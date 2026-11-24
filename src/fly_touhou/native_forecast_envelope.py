"""Unions of conditional forecasts; uncertainty is never a positive hit label."""
import numpy as np


def union_forecasts(rows):
    result=dict(rows[0]);points=np.asarray([p['bounds'] for p in rows]);extras=np.asarray([p['extra_radii'] for p in rows])
    low=(points-extras[:,:,None,None]).min(axis=(0,2));high=(points+extras[:,:,None,None]).max(axis=(0,2))
    middle=(low+high)*.5
    result.update(positions=middle.tolist(),bounds=np.stack((np.column_stack((low[:,0],middle[:,1])),
        np.column_stack((high[:,0],middle[:,1]))),axis=1).tolist(),direction=[1.,0.],
        extra_radii=((high[:,1]-low[:,1])*.5).tolist(),exact_center_trajectory=False,
        known_by_frame=np.all([p['known_by_frame'] for p in rows],axis=0).tolist(),
        collision_enabled_by_frame=np.any([p['collision_enabled_by_frame'] for p in rows],axis=0).tolist(),
        collision_scale_ratio_by_frame=np.max([p['collision_scale_ratio_by_frame'] for p in rows],axis=0).tolist())
    if any('collision_bounds' in p for p in rows):
        points=np.asarray([p.get('collision_bounds',p['bounds']) for p in rows])
        lo=(points-extras[:,:,None,None]).min(axis=(0,2));hi=(points+extras[:,:,None,None]).max(axis=(0,2));mid=(lo+hi)*.5
        result['collision_bounds']=np.stack((np.column_stack((lo[:,0],mid[:,1])),np.column_stack((hi[:,0],mid[:,1]))),1).tolist()
        result['extra_radii']=np.maximum(result['extra_radii'],(hi[:,1]-lo[:,1])*.5).tolist()
    result.pop('growth_collision',None)
    return result


def pad_birth(forecast,tick,origin,*,_clearance_arrays=False):
    """Enclose both native allocation orders and shift descendants as well."""
    bound=np.asarray(forecast['bounds']);extra=np.asarray(forecast['extra_radii'])
    low=np.minimum(bound[:-1].min(1)-extra[:-1,None],bound[1:].min(1)-extra[1:,None])
    high=np.maximum(bound[:-1].max(1)+extra[:-1,None],bound[1:].max(1)+extra[1:,None])
    # A wrap has separate pre-teleport collision coordinates. Keep those in
    # the birth-time envelope instead of dropping them when shifting ticks.
    if 'collision_bounds' in forecast:
        contact=np.asarray(forecast['collision_bounds'])
        low=np.minimum(low,np.minimum(contact[:-1].min(1)-extra[:-1,None],contact[1:].min(1)-extra[1:,None]))
        high=np.maximum(high,np.maximum(contact[:-1].max(1)+extra[:-1,None],contact[1:].max(1)+extra[1:,None]))
    if 'growth_collision' in forecast:
        g=forecast['growth_collision'];p=np.asarray(g['positions']);active=np.asarray(g['possible_by_frame'])
        low=np.minimum(low,np.where(active[1:,None],p[1:],np.inf));high=np.maximum(high,np.where(active[1:,None],p[1:],-np.inf))
    mid=(low+high)*.5;gap=(high[:,1]-low[:,1])*.5
    bounds=np.stack((np.column_stack((low[:,0],mid[:,1])),np.column_stack((high[:,0],mid[:,1]))),1)
    known=forecast['known_by_frame']
    collision=forecast['collision_enabled_by_frame']
    scale=forecast['collision_scale_ratio_by_frame']
    arrays=dict(positions=np.vstack((np.repeat([origin],tick,0),mid)),
        bounds=np.concatenate((np.tile([[origin,origin]],(tick,1,1)),bounds)),
        extra_radii=np.r_[np.zeros(tick),gap],
        known_by_frame=np.concatenate((np.ones(tick,bool),np.logical_and(known[:-1],known[1:]))),
        collision_enabled_by_frame=np.concatenate((np.zeros(tick,bool),np.logical_or(collision[:-1],collision[1:]))),
        collision_scale_ratio_by_frame=np.concatenate((np.ones(tick),np.maximum(scale[:-1],scale[1:]))))
    if not _clearance_arrays:
        arrays={key:value.tolist() for key,value in arrays.items()}
    born=dict(forecast,**arrays,direction=[1.,0.],exact_center_trajectory=False)
    born.pop('growth_collision',None);born.pop('collision_bounds',None)
    born['virtual_children']=[pad_birth(p,tick,origin,_clearance_arrays=_clearance_arrays)
                              for p in forecast.get('virtual_children',())]
    return born
