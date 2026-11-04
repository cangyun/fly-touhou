"""V9 preferences with immediate-safe support restored, never an action mask.

A bounded one-turn search supplies positive route preferences, not a proof
that every other adaptive continuation fails. Keep that prior, but reserve a
small movement-reference mass for every reliable next-four-frame alternative.
"""
import math
import numpy as np

from .resource_teacher_v9 import ResourceTeacherV9
from .thrl_movement_teacher import normal_unprotected

REVISION='resource-v10-immediate-safe-soft-route-support-v1'
UNKNOWN_REASONS=frozenset(('growth_motion_unknown','clipped_collision_geometry',
    'nearby_unrepresented_map_hazard','laser_motion_unknown','unrepresented_laser_map_motion_unknown'))


def soften_route_reference(probabilities,observation,movement,*,support_mix=.1):
    """Preserve Bomb mass and confident first-step safety, soften route zeros."""
    from .thrl_native_env import SOURCE19_TO_NATIVE36
    p=np.asarray(probabilities,dtype=np.float64)
    if (p.shape!=(36,) or not np.isfinite(p).all() or (p<0).any() or abs(float(p.sum())-1)>2e-6):
        raise ValueError('Normalized current native36 reference required')
    if isinstance(support_mix,bool) or not math.isfinite(support_mix) or not 0<support_mix<=1:
        raise ValueError('A convex support mixture in (0,1] is required')
    f,o,m=(np.asarray(observation[k]) for k in ('features','objects','maps'))
    if f.shape!=(25,) or o.shape!=(96,29) or m.shape!=(16,92,96) or not all(np.isfinite(x).all() for x in (f,o,m)):
        raise ValueError('Paired finite V5 observation required')
    detail=dict(revision=REVISION,support_mix=support_mix,teacher_is_native_oracle=False,
        action_execution_modified=False,Bomb_mass_preserved=True,guidance_usable=True)
    if not normal_unprotected(f):return p.copy(),dict(detail,reason='protected_or_non_normal')
    reason=movement._uncertainty(f,o,m,horizon=4)
    if reason:
        if reason not in UNKNOWN_REASONS:raise ValueError('Unknown immediate-geometry abstention')
        return p.copy(),dict(detail,reason=reason,guidance_usable=False)
    origin=movement._scene(f,o)[0]
    margins=movement.clearances(f,o,np.repeat(origin[None],18,axis=0),movement.velocities,duration=4)
    if np.isnan(margins).any():raise ValueError('NaN in first-step reference geometry')
    safe=margins>movement.config.margin
    if not safe.any():return p.copy(),dict(detail,reason='no_buffered_four_frame_action',safe_first_actions=0)
    native=np.asarray(SOURCE19_TO_NATIVE36[:18]);old=p[native]
    mass=float(old.sum());preferred=np.where(safe,old,0.)
    preferred=preferred/preferred.sum() if preferred.sum()>0 else safe.astype(float)/safe.sum()
    target=(1-support_mix)*preferred+support_mix*safe/safe.sum()
    result=p.copy();result[native]=mass*target
    return result,dict(detail,reason='soft_immediate_safe_support',safe_first_actions=int(safe.sum()),
        restored_zero_reference_actions=int((safe&(old==0)).sum()),
        restored_native_actions=native[safe&(old==0)].tolist(),
        removed_unbuffered_movement_mass=float(old[~safe].sum()),
        minimum_conditional_probability=support_mix/int(safe.sum()),
        supported_first_actions_are_not_guaranteed_safe_long_routes=True)


class ResourceTeacherV10(ResourceTeacherV9):
    def __init__(self,planner,config=None,*,support_mix=.1,**settings):
        if isinstance(support_mix,bool) or not math.isfinite(support_mix) or not 0<support_mix<=1:
            raise ValueError('A convex support mixture in (0,1] is required')
        self.support_mix=float(support_mix)
        super().__init__(planner,config,**settings)
        self.planner._batch_rigid_lasers=True

    def reset(self):
        super().reset();self._soft_snapshot=None;self._soft_probabilities=None;self._soft_detail=None
        self._native_geometry_snapshot=None;self._native_geometry=None

    def bind_native_geometry(self,snapshot,encoding):
        """Share the actual current compact sensor; do not rebuild clipped96 tokens."""
        self._native_geometry_snapshot=snapshot
        self._native_geometry=(encoding if encoding is not None
            and (encoding.get('frame'),encoding.get('stage'))==(snapshot.frame,snapshot.stage)
            and encoding.get('source_native_forecasts') is not None else None)
        self.planner.bind_native_geometry(snapshot,self._native_geometry)

    def _contact_clearance(self,snapshot,path,enemies):
        encoding=getattr(self,'_native_geometry',None)
        if snapshot is not getattr(self,'_native_geometry_snapshot',None) or encoding is None:
            return super()._contact_clearance(snapshot,path,enemies)
        from .compact_native_forecast import teacher_enemy_path_clearance
        gap,tail=teacher_enemy_path_clearance(encoding,list(path),
            invulnerable_frames=snapshot.player.invulnerable_frames)
        # The resource planner can look farther ahead than the24-frame native
        # capture. Keep its explicit heuristic tail, never call it verified.
        if tail:gap=min(gap,super()._contact_clearance(snapshot,tail,enemies))
        return gap

    def probabilities(self,snapshot):
        if snapshot is self._soft_snapshot:return self._soft_probabilities.copy()
        original=super().probabilities(snapshot)
        observation=self.movement_encoder.encode(snapshot)
        result,detail=soften_route_reference(original,observation,self.movement,support_mix=self.support_mix)
        self._soft_snapshot=snapshot;self._soft_probabilities=result;self._soft_detail=detail
        return result.copy()

    def _contact_clearances(self,snapshot,paths,enemies):
        encoding=getattr(self,'_native_geometry',None)
        if snapshot is not getattr(self,'_native_geometry_snapshot',None) or encoding is None:
            return super()._contact_clearances(snapshot,paths,enemies)
        from .compact_native_forecast import teacher_enemy_paths_clearances
        gaps,tails=teacher_enemy_paths_clearances(encoding,paths,
            invulnerable_frames=snapshot.player.invulnerable_frames)
        for i,tail in enumerate(tails):
            if tail:gaps[i]=min(gaps[i],super()._contact_clearance(snapshot,tail,enemies))
        return gaps.tolist()

    def config(self):
        result=super().config()
        result.update(teacher='ResourceTeacherV10',schema=REVISION,
            movement_support='90% inherited route/resource preference conditioned on immediate-safe set;10% uniform immediate-safe support by default',
            support_mix=self.support_mix,
            rigid_unknown_laser_computation='batched original exact2D segment distances;forecast and labels unchanged',
            one_turn_route_absence_is_not_a_negative_adaptive_route_label=True,
            online_reference_confidence='unknown laser motion or unreliable immediate geometry:weight0;all PPO samples retained',
            Bomb_distribution='inherited V9 marginal preserved;no execution takeover',
            teacher_is_clear_verified=False)
        result['compact_native_contact_prefix']='same-frame native positions, shapes and contact schedule when bound;conditional beyond unresolved ECL;legacy heuristic after24 frames'
        result['compact_native_line_prefix']='same-frame native trimmed line hitboxes and motion through24 frames;covered lines excluded from old capsule proxy;legacy heuristic after24 frames'
        return result

    def diagnostics(self):
        result=super().diagnostics()
        result.update(teacher='ResourceTeacherV10',schema=REVISION,soft_route_reference=self._soft_detail)
        return result
