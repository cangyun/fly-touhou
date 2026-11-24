"""Replan a short safe prefix and a conditional 48- or 96-frame corridor online.

Beyond24f, currently observed objects are extrapolated, including all lasers;
new spawns and unsupported native programs are not certified. A collecting
teacher executes the first4f command and replans from its actual position.
Only native-successful tasks enter the verified replay bank. Optional live
preferences remain conditional guidance, not verified future trajectories.
"""
import numpy as np

REVISION='nbnm-linked-prefix-and-corridor-v2'


def configured_beam(stage, settings, phase=None):
    """Allow a measured road-search improvement without changing other stages."""
    value=settings.get('stage_beams',{}).get(str(stage),settings.get('beam',8))
    value=settings.get('phase_beams',{}).get(phase,value)
    if type(value) is not int or not 1<=value<=64:
        raise ValueError('Live route beam must be an integer in 1..64')
    return value


def in_scope(snapshot,settings):
    """Enable extra search only in the stages/road intervals actually selected."""
    if snapshot.stage in settings.get('stages',range(1,7)):return True
    window=settings.get('road_stage_windows',{}).get(str(snapshot.stage))
    if window is None:return False
    from .nbnm_route_atlas import phase_progress
    phase,clock=phase_progress(snapshot)
    return bool(phase.endswith('/road') and clock is not None and window[0]<=clock<window[1])


def configured_horizon(snapshot, settings, phase=None):
    """Keep the established longer search on a measured stage or phase."""
    value=settings.get('stage_horizons',{}).get(str(snapshot.stage),settings.get('horizon',96))
    overrides=settings.get('phase_horizons',{})
    if overrides:
        if phase is None:
            from .nbnm_route_atlas import phase_progress
            phase,_=phase_progress(snapshot)
        value=overrides.get(phase,value)
    if value not in (48,96):
        raise ValueError('Rolling forecasts support 48 or 96 native frames')
    return value


def reserve(points):
    p=np.asarray(points)
    return np.minimum(184-np.abs(p[...,0]),48)/48+np.minimum(np.minimum(p[...,1]-32,432-p[...,1]),64)/64


class LiveRoutePlanner:
    def __init__(self, *, beam=8, interval=16, horizon=96):
        if horizon not in (48,96):raise ValueError('Use a rolling horizon of 48 or 96 native frames')
        self.beam=int(beam);self.interval=int(interval);self.horizon=horizon;self.previous=None

    def query(self, encoding, routes, demo=None):
        from .thrl_movement_teacher import action_velocities
        from .native_wall_geometry import player_paths
        from .compact_spiking_sensation import point_and_laser_clearances
        d=encoding['details'];f=np.asarray(encoding['source_features25'],float)
        result=dict(revision=REVISION,available=False,teacher_only=True,
            native_prefix_horizon=24,conditional_corridor_horizon=self.horizon,new_spawns_known=False,
            missing_path_is_collision_label=False,search_beam=self.beam)
        if routes is None or not np.asarray(d['trusted']).all():return result
        first=np.flatnonzero((np.asarray(d['margins'])[0]>1)&(np.asarray(routes['exits'])>0))
        if not len(first):return result
        v=action_velocities().copy()
        if f[18]<0:v[9:]=v[:9]
        origin=np.array([f[0]*368-184,f[1]*400+32])
        # A waypoint reached through action A is not evidence that moving
        # toward it through action B is feasible. Recompute from the current
        # native state and retain the first action of EVERY surviving branch.
        # The old nearby-waypoint cache discarded that link across decisions.
        second=np.asarray(routes.get('best_second',np.full(18,-1)))
        first=first[second[first]>=0]
        if not len(first):return result
        prefix=player_paths(np.broadcast_to(origin,(len(first),2)),v[first])
        paths=player_paths(prefix[:,4],v[second[first]],prefix=prefix,turn_tick=4)
        third=np.asarray(routes.get('best_third',np.full(18,-1)))[first]
        turning=np.flatnonzero(third>=0)
        if len(turning):
            paths[turning]=player_paths(paths[turning,12],v[third[turning]],
                prefix=paths[turning],turn_tick=12)
        positions=paths[:,24];costs=np.clip(np.asarray(routes['best_margin'])[first],0,16)/16
        costs+=.8*reserve(positions)
        origins=first.copy();evaluated=len(first)
        covered=24
        choices=np.sort(np.unique(v,axis=0,return_index=True)[1])
        for clock in range(24,self.horizon,24):
            order=np.argsort(-costs,kind='stable');kept=[];cells=set()
            for i in order:
                cell=tuple(np.rint(positions[i]/12).astype(int))
                if cell in cells:continue
                kept.append(i);cells.add(cell)
                if len(kept)>=self.beam:break
            ids=np.repeat(kept,len(choices));actions=np.tile(choices,len(kept))
            starts=positions[ids];speed=v[actions]
            # No short sidecar is passed beyond its supported horizon:
            # doing so would silently omit its objects from rigid geometry.
            gap=point_and_laser_clearances(f,encoding['source_objects29'],starts,speed,
                duration=24,initial_time=clock,anticipate_arming=True,
                native_forecasts=None,unknown_laser_player_motion=True,
                max_laser_margin=64.,linear_hint=True)
            ends=np.clip(starts+24*speed,[-184,32],[184,432]);evaluated+=len(gap)
            good=gap>1
            if not good.any():
                return dict(result,reason='no_complete_conditional_corridor',
                    covered_corridor_horizon=covered,candidate_segments=evaluated,
                    replanned=True,first_action_linked=True)
            score=costs[ids]+.25*np.clip(gap,0,16)/16+.5*reserve(ends)
            if demo and demo.get('available'):
                goal=np.asarray(demo['waypoints'][1])
                score+=.15*(np.linalg.norm(goal-origin)-np.linalg.norm(ends-goal,axis=1))/96
            positions=ends[good];costs=score[good];origins=origins[ids][good]
            covered=clock+24
        # Rank the complete conditional path through its OWN first move.
        # Beam-pruned alternatives remain legal, not labelled colliding.
        support=np.zeros(18,bool);preference=np.full(18,-64.,float)
        for action in np.unique(origins):
            support[action]=True;preference[action]=costs[origins==action].max()
        order=np.argsort(-costs,kind='stable')[:self.beam]
        goals=positions[order];goal_actions=origins[order]
        support &= (np.asarray(d['margins'])[0]>1)&(np.asarray(routes['exits'])>0)
        return dict(result,available=True,scores=preference.tolist(),supported=support.tolist(),
            waypoints=goals.tolist(),candidate_segments=evaluated,replanned=bool(evaluated),
            origin=origin.tolist(),conditional_only_after24=True,first_action_linked=True,
            waypoint_first_actions=goal_actions.tolist(),covered_corridor_horizon=covered)


def apply_live_advice(probabilities, advice, *, mix=.7):
    if not advice or not advice.get('available'):return probabilities
    valid=np.asarray(advice['supported'],bool);scores=np.asarray(advice['scores'])
    target=np.where(valid,np.exp(np.clip(5*(scores-scores[valid].max()),-60,0)),0)
    target/=target.sum()
    return (1-mix)*np.asarray(probabilities)+mix*target
