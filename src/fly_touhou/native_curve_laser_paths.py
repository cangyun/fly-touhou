"""Player-path hints against current native curve-laser collision rectangles."""
import numpy as np
from .native_curve_laser_contact import curve_laser_contact_batch

REVISION='native-curve-pending-head-activation-hint-v2'


def native_curve_path_clearances(geometry,starts,velocities,*,duration=4,move_player=False,
                                 max_relevant_margin=None,anticipate_head_activation=True,initial_time=0):
    """Minimum current-shape clearance along each proposed player path.

    Observed rectangles retain their positions. At future ticks, previously
    excluded head nodes are treated as potentially collidable: prepending new
    head nodes can move the old nodes past the native16px exclusion. The
    unknown-motion flag remains required; unseen future nodes, their motion,
    and the exact activation time are NOT predicted or certified safe.
    """
    starts=np.asarray(starts,float);velocities=np.asarray(velocities,float)
    ticks=np.unique(np.r_[np.arange(duration,dtype=float),duration]) if move_player or geometry.get('motion_forecast_enabled') else np.array([0.])
    points=np.clip(starts[None,:,:]+ticks[:,None,None]*(velocities[None,:,:] if move_player else 0),[-184,32],[184,432])
    flattened=points.reshape(-1,2);best=np.full(len(flattened),np.inf)
    for curve in geometry['curves']:
        if curve['width']<=0 or len(curve['nodes'])<2:continue
        if geometry.get('motion_forecast_enabled'):
            from .native_curve_motion_paths import motion_covered,predicted_curve_clearances
            if motion_covered(curve,initial_time+duration):
                value=predicted_curve_clearances(curve,points,geometry['player_half_extents'],ticks+initial_time,
                                                max_relevant_margin=max_relevant_margin)
                best=np.minimum(best,value.reshape(-1))
                continue
        # Bound peak temporary arrays when an original curve has many nodes.
        chunk=max(18,min(256,180000//len(curve['nodes'])))
        # Keep the t=0 current-geometry test exact. Only future path samples
        # use the conservative pending-head activation hypothesis.
        for begin,end,pending in ((0,len(starts),False),
                                  (len(starts),len(flattened),bool(anticipate_head_activation))):
            for first in range(begin,end,chunk):
                last=min(first+chunk,end)
                value=curve_laser_contact_batch(curve['nodes'],curve['width'],flattened[first:last],
                    geometry['player_half_extents'],max_relevant_margin=max_relevant_margin,
                    include_pending_head=pending)['signed_clearances']
                best[first:last]=np.minimum(best[first:last],value)
    return best.reshape(len(ticks),len(starts)).min(axis=0)
