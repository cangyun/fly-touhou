"""Raw-state-aware conservative forecasts without reading future attack scripts."""
import math
import numpy as np

from .dynamic_planner import DynamicCollisionPlanner, _phase_envelope_distance
from .hazard_observation import HazardObservationEncoder


class NativeHazardPlanner(DynamicCollisionPlanner):
    def reset(self):
        super().reset()
        self._raw_prepared_snapshot = None
        self._growth_envelopes = []

    def specification(self):
        result = super().specification()
        result.update(schema='native-hazard-prediction-v4', raw_metadata_required=True,
            bullet_flag_policy='mask0x2 SET enables hit/graze processing; hold current gate fixed within the horizon',
            bullet_flag_evidence='original EXE 416d7e: test al,2; 416d80: je 416f29 returns without hit processing when clear',
            state2_motion='time-matched collinear envelope from half stored velocity to full velocity plus a half-velocity transition-step allowance',
            state2_activation='conservative from forecast start; native timer>=8 is currently collidable, but future animation transition timing is unknown',
            current_state_policy='ordinary native hit calls occur in state1 and state2; other current dispatch branches do not call the hit helper',
            laser_motion='inherited unknown-phase kind0 envelope from measured speed/angle; non-kind0 measured geometry held stationary',
            limitations_additional=['unknown future flag/state changes and attack-script acceleration are not predicted',
                'state2 transition allowance is conservative, not an exact reconstructed native trajectory',
                'raw state/flags are current measurements, not a complete native collision oracle'])
        return result

    def _prepare(self, snapshot):
        if snapshot is self._raw_prepared_snapshot:
            return
        # Validate every raw/shared field using the same measured sensor contract.
        validator = HazardObservationEncoder()
        bullet_metadata, laser_metadata = validator._metadata(snapshot)
        if snapshot.player is not None:
            for b in snapshot.bullets:
                validator._bullet(b, bullet_metadata[b.id])
            for laser in snapshot.lasers:
                validator._laser(laser, laser_metadata[laser.id], snapshot.player)
        super()._prepare(snapshot)
        if snapshot.player is None:
            self._growth_envelopes = []
            self._raw_prepared_snapshot = snapshot
            return
        rigid, growth, excluded, state_excluded = [], [], 0, 0
        for b in snapshot.bullets:
            row = bullet_metadata[b.id]
            if b.intangible or not row['native_flags'] & 2:
                excluded += 1
                continue
            if row['native_state'] not in (1, 2):
                state_excluded += 1
                continue
            if row['native_state'] == 2:
                velocity = np.asarray((b.vx, b.vy), dtype=float)
                speed = float(np.linalg.norm(velocity))
                direction = velocity/speed if speed else np.asarray((1., 0.))
                growth.append((np.asarray((b.x, b.y)), velocity*.5, direction,
                    speed*.5, b.radius))
            else:
                rigid.append(b)
        self._bullet_positions = np.asarray([(b.x,b.y) for b in rigid], dtype=float).reshape(-1,2)
        self._bullet_velocities = np.asarray([(b.vx,b.vy) for b in rigid], dtype=float).reshape(-1,2)
        self._bullet_radii = np.asarray([b.radius for b in rigid], dtype=float)
        self._growth_envelopes = growth
        self._hazard_counts.update(raw_flag_exclusions=excluded, state2_motion_envelopes=len(growth),
            current_noncolliding_state_exclusions=state_excluded,
            rigid_state1_candidates=len(rigid), raw_metadata_observed=snapshot.player is not None)
        self._raw_prepared_snapshot = snapshot

    def path_clearance(self, snapshot, path):
        self._prepare(snapshot)
        clearance = super().path_clearance(snapshot, path)
        if snapshot.player is None:
            return clearance
        for first, last, intercept, velocity in path:
            start = max(first, snapshot.player.invulnerable_frames)
            if start > last:
                continue
            for origin, half_velocity, direction, half_speed, radius in self._growth_envelopes:
                distance = _phase_envelope_distance(np.asarray(intercept), np.asarray(velocity)-half_velocity,
                    origin, direction, half_speed, half_speed, start, last)
                clearance = min(clearance, distance-radius-snapshot.player.radius)
        return clearance


class GeometryNativeHazardPlanner(NativeHazardPlanner):
    """Opt-in V5 native circles, rectangles and large-player collision size.

    V4 remains available unchanged for old runs. No observation field is
    reinterpreted: V5 requires the explicit reader extension instead.
    """

    def reset(self):
        super().reset()
        self._geometry_snapshot = None
        self._native_shapes = []
        self._circle_positions = self._circle_velocities = np.zeros((0,2),dtype=float)
        self._circle_radii = np.zeros(0,dtype=float)
        self._rectangle_shapes = []
        self._rectangle_positions = self._rectangle_velocities = np.zeros((0,2),dtype=float)
        self._rectangle_extents = self._rectangle_offsets = np.zeros((0,2),dtype=float)
        self._shape_growth_envelopes = []

    def specification(self):
        result = super().specification()
        result.update(schema='native-hazard-prediction-v5-geometry',
            collision_geometry_required='original-th14-collision-geometry-v1',
            collision_scale='multiply bullet dimension by measured scale only with native_flags&0x40',
            native_shapes='flags0x10: circle; otherwise axis-aligned rectangle with full widths be0/be4',
            circle_clearance='continuous centre distance minus hypot(native bullet radius,native player circle component), with outward float32 guard',
            player_circle='definition radius; when player_flags&0x10, native float32 definition_radius*scale*3.6',
            rectangle_clearance='continuous signed L-infinity separation of native bullet rectangle and measured player bounds; conservative outside Euclidean distance',
            state2_shape='circumscribed combined-shape circle over the inherited half/full velocity phase envelope',
            current_geometry_only=True,
            geometry_limitations=['future scale/shape changes, interpolation targets and attack-script updates remain unobserved',
                'state2 rectangle uses a conservative circumscribed envelope',
                'laser geometry and movement stay at the previously documented conservative approximation'])
        return result

    def _prepare(self, snapshot):
        if snapshot is self._geometry_snapshot:
            return
        from .native_geometry import collision_geometry
        super()._prepare(snapshot)
        rows, _ = HazardObservationEncoder()._metadata(snapshot)
        shapes, player = collision_geometry(snapshot,rows)
        self._native_shapes = []
        for bullet in snapshot.bullets:
            raw = rows[bullet.id]
            if bullet.intangible or not raw['native_flags'] & 2 or raw['native_state'] not in (1,2):
                continue
            self._native_shapes.append((bullet,raw['native_state'],shapes[bullet.id]))
        circles,rectangles,growth = [],[],[]
        for bullet,state,shape in self._native_shapes:
            origin = np.asarray((bullet.x,bullet.y))
            motion = np.asarray((bullet.vx,bullet.vy))
            rectangle = shape['shape']=='rectangle'
            if state==2:
                speed = float(np.linalg.norm(motion))
                direction = motion/speed if speed else np.asarray((1.,0.))
                radius = float(np.linalg.norm(shape['half_extents'])) if rectangle else shape['combined_radius']
                offset = shape['player_bounds_offset'] if rectangle else np.zeros(2)
                growth.append((origin,motion*.5,direction,speed*.5,radius,offset))
            elif rectangle:
                rectangles.append((origin,motion,shape['half_extents'],shape['player_bounds_offset']))
            else:
                circles.append((origin,motion,shape['combined_radius']))
        self._circle_positions = np.asarray([r[0] for r in circles],dtype=float).reshape(-1,2)
        self._circle_velocities = np.asarray([r[1] for r in circles],dtype=float).reshape(-1,2)
        self._circle_radii = np.asarray([r[2] for r in circles],dtype=float)
        self._rectangle_shapes,self._shape_growth_envelopes = rectangles,growth
        self._rectangle_positions = np.asarray([r[0] for r in rectangles],dtype=float).reshape(-1,2)
        self._rectangle_velocities = np.asarray([r[1] for r in rectangles],dtype=float).reshape(-1,2)
        self._rectangle_extents = np.asarray([r[2] for r in rectangles],dtype=float).reshape(-1,2)
        self._rectangle_offsets = np.asarray([r[3] for r in rectangles],dtype=float).reshape(-1,2)
        # The inherited evaluator still supplies its validated laser paths.
        # Disable its old scalar bullet interpretation for this V5 instance.
        self._bullet_positions = np.zeros((0,2),dtype=float)
        self._bullet_velocities = np.zeros((0,2),dtype=float)
        self._bullet_radii = np.zeros(0,dtype=float)
        self._growth_envelopes = []
        self._hazard_counts.update(native_geometry_schema='original-th14-collision-geometry-v1',
            native_circles=sum(s['shape']=='circle' for _,_,s in self._native_shapes),
            native_rectangles=sum(s['shape']=='rectangle' for _,_,s in self._native_shapes),
            collision_scaled_bullets=sum(s['scale_applies'] for _,_,s in self._native_shapes),
            large_player_circle_component=player['circle_component'] if player else None,
            large_player_collision_flag=player['large_player_flag'] if player else None)
        self._geometry_snapshot = snapshot

    def path_clearance(self, snapshot, path):
        from .native_geometry import swept_rectangle_clearances
        path = list(path)
        self._prepare(snapshot)
        clearance = super().path_clearance(snapshot,path)
        if snapshot.player is None:
            return clearance
        for first,last,intercept,velocity in path:
            start = max(first,snapshot.player.invulnerable_frames)
            if start>last:
                continue
            intercept,velocity = np.asarray(intercept),np.asarray(velocity)
            if len(self._circle_radii):
                position = intercept-self._circle_positions
                relative = velocity-self._circle_velocities
                squared = np.einsum('ij,ij->i',relative,relative)
                times = np.divide(-np.einsum('ij,ij->i',position,relative),squared,
                    out=np.full(len(squared),start,dtype=float),where=squared>1e-12)
                times = np.clip(times,start,last)
                distances = np.linalg.norm(position+relative*times[:,None],axis=1)
                clearance = min(clearance,float((distances-self._circle_radii).min()))
            if len(self._rectangle_extents):
                values=swept_rectangle_clearances(intercept+self._rectangle_offsets-self._rectangle_positions,
                    velocity-self._rectangle_velocities,self._rectangle_extents,start,last)
                clearance=min(clearance,float(values.min()))
            for origin,half_motion,direction,half_speed,radius,offset in self._shape_growth_envelopes:
                distance = _phase_envelope_distance(intercept+offset,velocity-half_motion,
                    origin,direction,half_speed,half_speed,start,last)
                clearance = min(clearance,distance-radius)
        return clearance
