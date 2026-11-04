"""V8 contact/resources plus observed endpoint-motion laser forecasts.

No unverified native state is treated as a laser collision gate. When endpoint
history is unknown, retain the explicitly conservative legacy envelope instead
of manufacturing observed stationary motion. No clear/avoidance claim follows.
"""
import math
import numpy as np
from .native_hazard_planner import GeometryNativeHazardPlanner
from .resource_teacher_v8 import ResourceTeacherV8,_CurrentObservation
from .thrl_inputs import SEPARATED_LASER_INPUT
from .thrl_laser_teacher_batch import prepare_segments, path_clearance as batch_line_clearance

REVISION='resource-v9-observed-laser-segment-v1'


class ObservedLaserGeometryPlanner(GeometryNativeHazardPlanner):
    def __init__(self,config):
        self.observation_provider=None
        super().__init__(config)

    def reset(self):
        super().reset();self._endpoint_snapshot=None;self._observed_lines=()
        self._observed_segment_arrays=prepare_segments(())
        self._native_line_snapshot=None;self._native_line_encoding=None
        self._native_line_covered=set()

    def bind_native_geometry(self,snapshot,encoding):
        self._native_line_snapshot=snapshot;self._native_line_encoding=encoding

    def _legacy_lines(self,snapshot,known,excluded=()):
        geometry=[];envelopes=[]
        for laser in snapshot.lasers:
            if laser.id in known or laser.id in excluded:continue
            if not laser.active and not self.prediction_config.include_inactive_lasers:continue
            direction=np.asarray((math.cos(laser.angle),math.sin(laser.angle)))
            if laser.kind==0 and not laser.nodes:
                envelopes.append((np.asarray((laser.x,laser.y)),direction,laser.length,laser.speed,laser.width/2))
            else:
                from .thrl_observation import THRLObservationEncoder
                points,_=THRLObservationEncoder._polyline(laser)
                pairs=[(points[0],points[0])] if len(laser.nodes)==1 else zip(points[:-1],points[1:])
                geometry.extend((a,b,np.zeros(2),laser.width/2) for a,b in pairs)
        return geometry,envelopes

    def _prepare(self,snapshot):
        super()._prepare(snapshot)
        if snapshot is self._endpoint_snapshot or snapshot.player is None:return
        if self.observation_provider is None:raise ValueError('Measured V5 reference provider required')
        observation,motion=self.observation_provider(snapshot)
        if motion is None or (motion.frame,motion.stage)!=(snapshot.frame,snapshot.stage):
            raise ValueError('Planner laser context does not belong to this current native frame')
        measured={row.identity:row for row in motion.lines if row.motion_observed}
        known={laser.id for laser in snapshot.lasers if laser.width>0 and laser.id in measured}
        # Reconstruct only UNKNOWN legacy candidates. Known laser endpoints
        # must not be counted again under the old phase-union/static model.
        geometry,envelopes=self._legacy_lines(snapshot,known)
        self._laser_geometry=geometry;self._kind0_envelopes=envelopes
        self._observed_lines=tuple(measured[i] for i in sorted(known))
        self._observed_segment_arrays=prepare_segments(self._observed_lines)
        encoding=self._native_line_encoding if snapshot is self._native_line_snapshot else None
        rows=((encoding.get('source_native_forecasts') or {}).get('line_geometry') or {}).get('lines',()) if encoding else ()
        covered={row['forecast']['id'] for row in rows}
        self._native_line_covered=covered
        if covered:
            self._uncovered_laser_geometry,self._uncovered_kind0_envelopes=self._legacy_lines(snapshot,known,covered)
            self._uncovered_segment_arrays=prepare_segments(tuple(row for row in self._observed_lines if row.identity not in covered))
        f=observation['features'];offset=np.asarray(f[16:18],float)*64
        radius=max(float(f[13])*64+np.linalg.norm(offset),np.linalg.norm(np.asarray(f[14:16],float)*64))+.05
        self._laser_player_offset=offset;self._laser_player_radius=radius
        self._hazard_counts.update(observed_endpoint_lines=len(known),native_program_lines=len(covered),
            unknown_motion_laser_candidates=len(snapshot.lasers)-len(known|covered),
            laser_future_motion_guaranteed=False,laser_collision_state_verified=False)
        self._endpoint_snapshot=snapshot

    def path_clearance(self,snapshot,path):
        path=list(path);self._prepare(snapshot)
        if snapshot is self._native_line_snapshot and self._native_line_encoding and self._native_line_covered:
            from .native_line_laser_paths import teacher_native_line_path_clearance
            native,tail=teacher_native_line_path_clearance(self._native_line_encoding,path,
                invulnerable_frames=snapshot.player.invulnerable_frames)
            prefix=[(first,min(last,24.),intercept,velocity) for first,last,intercept,velocity in path if first<=24.]
            # Remove covered native lines from BOTH old approximations for the
            # covered prefix. Taking min with the old full-width capsule would
            # otherwise erase the benefit of the actual trimmed hit geometry.
            legacy=(self._laser_geometry,self._kind0_envelopes)
            try:
                self._laser_geometry=self._uncovered_laser_geometry
                self._kind0_envelopes=self._uncovered_kind0_envelopes
                clearance=min(native,super().path_clearance(snapshot,prefix),
                    batch_line_clearance(self._uncovered_segment_arrays,prefix,
                        self._laser_player_offset,self._laser_player_radius,snapshot.player.invulnerable_frames))
            finally:self._laser_geometry,self._kind0_envelopes=legacy
            if tail:
                clearance=min(clearance,super().path_clearance(snapshot,tail),
                    batch_line_clearance(self._observed_segment_arrays,tail,
                        self._laser_player_offset,self._laser_player_radius,snapshot.player.invulnerable_frames))
            return clearance
        clearance=super().path_clearance(snapshot,path)
        if snapshot.player is None or not self._observed_lines:return clearance
        return min(clearance,batch_line_clearance(self._observed_segment_arrays,path,
            self._laser_player_offset,self._laser_player_radius,snapshot.player.invulnerable_frames))

    def specification(self):
        result=super().specification()
        result.update(schema='native-hazard-observed-laser-endpoints-v1',
            laser_motion='known: independently observed origin/end/width finite differences;unknown: explicit legacy conservative approximation',
            laser_future_motion_guaranteed=False,laser_collision_state_verified=False,
            laser_shape='conservative complete visible segment capsule, not verified native longitudinal collision trims')
        return result


class ResourceTeacherV9(ResourceTeacherV8):
    def __init__(self,planner,config=None,**settings):
        super().__init__(ObservedLaserGeometryPlanner(planner.config),config,**settings)
        self.movement_encoder=_CurrentObservation(SEPARATED_LASER_INPUT)
        self.planner.observation_provider=self._observation_with_motion

    def _observation_with_motion(self,snapshot):
        observation=self.movement_encoder.encode(snapshot)
        return observation,self.movement_encoder.laser_motion

    def _poc_route(self,snapshot,resources,enemies):
        result=super()._poc_route(snapshot,resources,enemies)
        if any(laser.width>0 for laser in snapshot.lasers):
            result.update(safe=False,why='long ascent is not certified from short observed laser endpoint motion')
        return result

    def config(self):
        result=super().config()
        from .thrl_movement_teacher import specification
        result.update(teacher='ResourceTeacherV9',schema=REVISION,
            student_observation_schema=SEPARATED_LASER_INPUT,movement=specification(SEPARATED_LASER_INPUT),
            teacher_is_clear_verified=False,laser_motion='measured endpoint finite differences; explicit unknown fallback',
            online_reference_confidence='unknown endpoint or curve motion:teacher-label weight0;all PPO samples retained',
            dynamic_prediction=self.planner.specification())
        return result

    def diagnostics(self):
        result=super().diagnostics()
        result.update(teacher='ResourceTeacherV9',schema=REVISION,student_observation_schema=SEPARATED_LASER_INPUT,
            forecast='measured endpoint/width extrapolation conditional on persistence; unknown laser motion is explicit')
        return result
