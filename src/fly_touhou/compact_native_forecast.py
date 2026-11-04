"""Native enemy trajectories and contact schedules for the existing129 input.

These metadata travel beside the29-column geometry; they do not add sensory
ports, actions, learned parameters, or connections to the fly network.
"""
from __future__ import annotations
import math
import numpy as np

REVISION = 'compact-native-ecl-motion-contact-line-spawn-v3'


def native_repair_revisions(*,reactive_orbs=None,bullet_scale=None):
    """Describe the label/sensory meaning without inspecting checkpoints."""
    from .native_enemy_ecl import REVISION as ecl
    from .native_line_laser_paths import REVISION as line
    from .compact_partial_risk import REVISION as partial
    from .native_bullet_release import COLLISION_REVISION as bullet_collision
    result=dict(geometry=REVISION,enemy_ecl=ecl,line_geometry=line,partial_point_danger=partial,
        bullet_collision=bullet_collision,
        resource_reference='native-positive-route-resource-reference-v1')
    if reactive_orbs is not None:result['reactive_orbs']=reactive_orbs
    if bullet_scale is not None:result['bullet_scale']=bullet_scale
    return result


def _rows(objects, forecasts):
    for row in (forecasts or {}).get('enemies', ()):
        yield objects[row['object_index']], row


def _shape(token, contact_tick):
    hitbox = contact_tick.get('hitbox', {})
    width, height = hitbox.get('width'), hitbox.get('height')
    half = np.array([token[10] * 128 if width is None else max(0., width) / 2,
                     token[11] * 128 if height is None else max(0., height) / 2])
    return half, not bool(contact_tick.get('rectangle', token[12] < .5))


def enemy_forecast_uncertainty(features, objects, forecasts, horizon):
    """Report uncovered body motion/contact and observed upcoming creations."""
    f = np.asarray(features, float)
    origin = np.array([f[0] * 368 - 184, f[1] * 400 + 32])
    player_extent = max(abs(f[13] * 64), np.linalg.norm((abs(f[14:16]) + abs(f[16:18])) * 64))
    unknown = []
    for token, row in _rows(objects, forecasts):
        movement = row['motion']
        spawns = movement.get('pending_enemy_spawns')
        if spawns is None:
            # Older captured records already retain this in their ECL threads.
            spawns = [event for thread in row['contact'].get('ecl_threads', ())
                      for event in thread.get('spawn_instructions', ())]
        due = [event['tick'] for event in spawns if 1 <= event['tick'] <= horizon]
        if due:
            # This is uncertainty, not invented child positions or forced
            # player actions. Positive known hazards and native outcomes are
            # still available, but absence of hazards is not a safe label.
            unknown.append(dict(object_index=row['object_index'],
                reason='pending_enemy_spawn_geometry_unknown', first_tick=min(due)))
        positions = np.asarray(movement['positions'], float)[:, :2]
        tick_rows = row['contact']['forecast']['ticks'][:horizon]
        phase = movement.get('phase_unknown_from_frame')
        phase_can_change_contact = (movement.get('damage_bound_enabled',False)
                                    and phase is not None and phase <= horizon)
        if (not phase_can_change_contact and tick_rows and len(tick_rows) == horizon
                and all(t['contactable'] is False for t in tick_rows)):
            continue
        body_radius=float(np.linalg.norm(token[10:12]*128))
        for contact in tick_rows:
            half,circle=_shape(token,contact)
            if not circle:
                rectangle=contact.get('rectangle_geometry') or {}
                half=np.asarray(rectangle.get('all_rotations_aabb_half_extents',
                    [half[1]+math.hypot(2*half[1],half[0])]*2))
            body_radius=max(body_radius,float(np.linalg.norm(half)))
        radius = body_radius + max(16., player_extent)
        # Same reachable-window interpretation as the existing compact sensor.
        envelope=np.asarray(movement.get('reachable_radius_by_frame',np.zeros(len(positions))),float)
        distance = (np.linalg.norm(positions[:horizon + 1] - origin, axis=-1)
                    -envelope[:horizon+1]).min()
        if distance > 4.5 * horizon + radius:
            continue
        boundary = movement.get('unknown_from_frame')
        if len(positions) <= horizon or (boundary is not None and boundary <= horizon):
            unknown.append(dict(object_index=row['object_index'], reason=movement.get('unknown_reason') or 'unsupported_native_motion'))
        if len(tick_rows) < horizon or any(not tick['known'] or
                (tick.get('contactable') is not False and tick.get('rectangle') and not tick.get('geometry_forecast_complete'))
                for tick in tick_rows):
            unknown.append(dict(object_index=row['object_index'], reason='native_contact_schedule_unknown'))
    return unknown


def _enemy_tick_geometry(f,origin,records,tick):
    positions, circles, radii, extents = [], [], [], []
    for token, record in records:
        rows = record['contact']['forecast']['ticks']
        contact = rows[tick - 1] if tick <= len(rows) else {}
        if contact.get('contactable') is False: continue
        path = record['motion']['positions']
        position = (np.asarray(path[tick][:2], float) if tick < len(path) else
                    origin + token[6:8] * [384, 448] + token[8:10] * 12 * tick)
        half, circle = _shape(token, contact)
        # Unresolved random motion remains a conditional conservative envelope.
        envelope=record['motion'].get('reachable_radius_by_frame',())
        extra=float(envelope[tick]) if tick<len(envelope) else 0.
        positions.append(position)
        if circle:
            circles.append(True)
            radii.append(math.sqrt(half[0] ** 2 + (f[13] * 64) ** 2) + .05+extra)
        else:
            rectangle = contact.get('rectangle_geometry') or {}
            half = np.asarray(rectangle.get('all_rotations_aabb_half_extents',
                [half[1] + math.hypot(2 * half[1], half[0])] * 2))
            half=half+extra
            circles.append(False);radii.append(0.)
        extents.append(half + f[14:16] * 64 + .05)
    if not positions:return None
    return tuple(np.asarray(values) for values in (positions,circles,radii,extents))


def _enemy_tick_clearances(f,geometry,starts,velocities,duration):
    from .thrl_movement_teacher import swept_clearances
    if geometry is None:return np.full(len(starts),np.inf)
    positions,circles,radii,extents=geometry
    if duration>0:
        return swept_clearances(starts,velocities,positions,np.zeros((len(positions),2)),
            circles,radii,extents,f[16:18]*64,duration=duration)
    delta=starts[:,None]-positions[None]
    circle_value=np.linalg.norm(delta,axis=-1)-radii[None]
    rectangle_value=(abs(delta+f[16:18]*64)-extents[None]).max(-1)
    return np.where(circles[None],circle_value,rectangle_value).min(-1)


def enemy_clearances(features, objects, forecasts, starts, velocities, *, duration, initial_time=0.):
    """Test each upcoming native contact tick at its composed/clamped position.

    Player segments enclose both sides of the within-tick player update. A body
    disabled at that tick is omitted. Unsupported programs remain conditional
    hints and are separately excluded from risk supervision by the trust flags.
    """
    f = np.asarray(features, float)
    starts, velocities = np.asarray(starts, float), np.asarray(velocities, float)
    answer = np.full(len(starts), np.inf)
    origin = np.array([f[0] * 368 - 184, f[1] * 400 + 32])
    records = list(_rows(objects, forecasts))
    if not records: return answer
    end = initial_time + duration
    # Collision is a discrete native check, not a sweep of the enemy through
    # its old and new positions. Future ECL flags apply on that same tick.
    for tick in range(max(1, math.ceil(initial_time)), math.floor(end) + 1):
        geometry=_enemy_tick_geometry(f,origin,records,tick)
        if geometry is None:continue
        begins = max(initial_time, tick - 1.)
        player_starts = np.clip(starts + velocities * (begins - initial_time), [-184, 32], [184, 432])
        value=_enemy_tick_clearances(f,geometry,player_starts,velocities,tick-begins)
        answer = np.minimum(answer, value)
    return answer


def teacher_enemy_path_clearance(encoding,path,*,invulnerable_frames=0):
    """Use the student's captured native enemy geometry on a teacher path.

    Returns the covered prefix and the remaining path. Beyond the captured
    horizon this makes no claim about future ECL or enemy contact state.
    A currently intangible enemy is retained: its contact gate can open on
    a later tick. Motion coverage flags remain separate from this hint.
    """
    forecasts=encoding['source_native_forecasts']
    enemies=forecasts.get('enemies',())
    if not enemies:return float('inf'),[]
    horizon=min(24,min(len(row['contact']['forecast']['ticks']) for row in enemies))
    value=float('inf');tail=[]
    for first,last,intercept,velocity in path:
        first=max(float(first),float(invulnerable_frames))
        last=float(last)
        if first>last:continue
        end=min(last,horizon)
        if first<=end:
            v=np.asarray(velocity,float)
            start=np.asarray(intercept,float)+v*first
            gap=enemy_clearances(encoding['source_features25'],encoding['source_objects29'],forecasts,
                start[None],v[None],duration=end-first,initial_time=first)
            value=min(value,float(gap[0]))
        if last>horizon:tail.append((max(first,float(horizon)),last,intercept,velocity))
    return value,tail


def teacher_enemy_paths_clearances(encoding,paths,*,invulnerable_frames=0):
    """Evaluate independent teacher paths against shared per-tick bodies.

    Fractional wall segments and protection exits keep their original time
    boundaries. Only equal tick/duration operations are packed together.
    """
    forecasts=encoding['source_native_forecasts'];enemies=forecasts.get('enemies',())
    answer=np.full(len(paths),np.inf);tails=[[] for _ in paths]
    if not enemies:return answer,tails
    horizon=min(24,min(len(row['contact']['forecast']['ticks']) for row in enemies))
    groups={}
    for index,path in enumerate(paths):
        for first,last,intercept,velocity in path:
            first=max(float(first),float(invulnerable_frames));last=float(last)
            if first>last:continue
            end=min(last,horizon)
            if first<=end:
                v=np.asarray(velocity,float);start=np.asarray(intercept,float)+v*first
                for tick in range(max(1,math.ceil(first)),math.floor(end)+1):
                    begins=max(first,tick-1.)
                    position=np.clip(start+v*(begins-first),[-184,32],[184,432])
                    groups.setdefault(tick,{}).setdefault(tick-begins,[]).append((index,position,v))
            if last>horizon:tails[index].append((max(first,float(horizon)),last,intercept,velocity))
    f=np.asarray(encoding['source_features25'],float)
    origin=np.array([f[0]*368-184,f[1]*400+32])
    records=list(_rows(encoding['source_objects29'],forecasts))
    for tick,durations in groups.items():
        geometry=_enemy_tick_geometry(f,origin,records,tick)
        for duration,rows in durations.items():
            indices,starts,velocities=zip(*rows)
            value=_enemy_tick_clearances(f,geometry,np.asarray(starts),np.asarray(velocities),duration)
            np.minimum.at(answer,np.asarray(indices),value)
    return answer,tails
