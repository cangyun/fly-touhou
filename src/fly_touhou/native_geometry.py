"""Versioned original-TH14 collision dimensions for an opt-in teacher.

The existing Bullet.radius/Player.radius observation contract is unchanged.
These functions consume the separately measured collision-geometry extension.
"""
from __future__ import annotations

import math

import numpy as np

from .hazard_reader import COLLISION_GEOMETRY_SCHEMA, FULL_GAME_SHA256

# Original float32 at VA4c1ab8; selected by Player.flags bit0x10 in 44efa0.
LARGE_PLAYER_CIRCLE_MULTIPLIER = 3.5999999046325684


def _finite_number(value, name, *, nonnegative=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or (nonnegative and value < 0)):
        raise ValueError('Invalid measured collision geometry: '+name)
    return float(value)


def _pair(value, name):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError('Invalid measured collision geometry: '+name)
    return np.asarray([_finite_number(v,name) for v in value],dtype=float)


def collision_geometry(snapshot, bullet_metadata):
    """Validate same-frame overlap and return native constant-shape primitives.

    Circle helper uses r_b**2+r_p**2 rather than (r_b+r_p)**2. Rectangle
    helper compares native axis-aligned bullet/player bounds. Computed bounds
    carry a small outward float32 guard; future script changes are not known.
    """
    data = snapshot.extra.get('native_collision_geometry')
    if (not isinstance(data,dict) or data.get('schema') != COLLISION_GEOMETRY_SCHEMA
            or data.get('frame') != snapshot.frame or data.get('exe_sha256') != FULL_GAME_SHA256
            or data.get('observed') is not (snapshot.player is not None)):
        raise ValueError('Missing or mismatched measured native collision geometry')
    rows = data.get('bullets')
    if not isinstance(rows,list) or any(not isinstance(row,dict) for row in rows):
        raise ValueError('Missing measured collision geometry bullets')
    ids = [row.get('id') for row in rows]
    if (any(isinstance(i,bool) or not isinstance(i,int) for i in ids)
            or len(ids) != len(set(ids)) or set(ids) != {b.id for b in snapshot.bullets}):
        raise ValueError('Collision geometry IDs do not cover the same snapshot')
    if snapshot.player is None:
        if data.get('player') is not None or rows:
            raise ValueError('Unobserved collision geometry has live objects')
        return {}, None
    player = data.get('player')
    if not isinstance(player,dict):
        raise ValueError('Missing measured player collision geometry')
    flags = player.get('native_flags')
    if (isinstance(flags,bool) or not isinstance(flags,int) or not 0 <= flags < 2**32
            or flags != snapshot.extra.get('resources',{}).get('player_flags')):
        raise ValueError('Native player flags differ from same-frame resources')
    scale = _finite_number(player.get('native_scale'),'player scale',nonnegative=True)
    base = _finite_number(player.get('native_base_radius'),'player base radius',nonnegative=True)
    definition = _finite_number(player.get('native_definition_radius'),'player definition radius',nonnegative=True)
    if (scale != snapshot.extra.get('player_scale') or base != snapshot.extra.get('base_player_radius')
            or base*scale != snapshot.player.radius):
        raise ValueError('Native player scale/base differ from the legacy same-frame snapshot')
    lower = _pair(player.get('native_bounds_min'),'player lower bounds')
    upper = _pair(player.get('native_bounds_max'),'player upper bounds')
    if np.any(upper < lower):
        raise ValueError('Reversed native player collision bounds')
    half = (upper-lower)*.5
    offset = (upper+lower)*.5 - np.asarray((snapshot.player.x,snapshot.player.y))
    # Match native float32 arithmetic before adding outward rounding allowance.
    component = np.float32(definition)
    if flags & 0x10:
        component = np.float32(np.float32(scale)*np.float32(LARGE_PLAYER_CIRCLE_MULTIPLIER))
        component = np.float32(component*np.float32(definition))
    component = float(component)
    result = {}
    for row in rows:
        original = bullet_metadata[row['id']]
        if any(row.get(key) != original[key] for key in (
                'native_flags','native_scale','native_hitbox_value')):
            raise ValueError('Native bullet geometry differs from raw same-frame fields')
        value = _finite_number(row.get('native_hitbox_value'),'bullet dimension',nonnegative=True)
        bullet_scale = _finite_number(row.get('native_scale'),'bullet scale',nonnegative=True)
        # A changing render scale does NOT alter native collision without 0x40.
        effective = float(np.float32(np.float32(value)*np.float32(bullet_scale))) if row['native_flags'] & 0x40 else value
        if row['native_flags'] & 0x10:
            if row.get('native_height_value','missing') is not None:
                raise ValueError('Circle must not claim an unused height field')
            combined = math.hypot(effective,component)
            guard = max(1e-6,combined*2e-7)
            result[row['id']] = {'shape':'circle','bullet_radius':effective,
                'combined_radius':combined+guard,'player_circle_component':component,
                'scale_applies':bool(row['native_flags'] & 0x40)}
        else:
            height = _finite_number(row.get('native_height_value'),'bullet height',nonnegative=True)
            if row['native_flags'] & 0x40:
                height = float(np.float32(np.float32(height)*np.float32(bullet_scale)))
            extents = np.asarray((effective,height))*.5+half
            guard = np.maximum(1e-6,extents*2e-7)
            result[row['id']] = {'shape':'rectangle','half_extents':extents+guard,
                'player_bounds_offset':offset,'scale_applies':bool(row['native_flags'] & 0x40)}
    return result, {'circle_component':component,'bounds_half_extents':half,
                    'bounds_offset':offset,'large_player_flag':bool(flags & 0x10)}


def swept_rectangle_clearance(position, velocity, half_extents, start, end):
    """Minimum signed L-infinity separation over a continuous linear path.

    max(|x|-hx, |y|-hy) <= 0 exactly characterizes contact with the expanded
    rectangle. Its four affine branches have minima only at an endpoint or a
    pairwise branch intersection. The outside distance is a conservative lower
    bound on Euclidean clearance, in the same game-coordinate units.
    """
    position, velocity, half_extents = (np.asarray(v,dtype=float) for v in (position,velocity,half_extents))
    slopes = np.asarray((velocity[0],-velocity[0],velocity[1],-velocity[1]))
    intercepts = np.asarray((position[0]-half_extents[0],-position[0]-half_extents[0],
                             position[1]-half_extents[1],-position[1]-half_extents[1]))
    times = [start,end]
    for first in range(4):
        for last in range(first+1,4):
            denominator = slopes[first]-slopes[last]
            if abs(denominator)>1e-12:
                t=(intercepts[last]-intercepts[first])/denominator
                if start<t<end:times.append(float(t))
    return min(float(np.max(intercepts+slopes*t)) for t in times)


def swept_rectangle_clearances(positions, velocities, half_extents, start, end):
    """Same four-branch exact minimization, batched over independent boxes."""
    positions,velocities,half_extents=(np.asarray(v,dtype=float) for v in (positions,velocities,half_extents))
    if not len(positions):return np.empty(0,dtype=float)
    slopes=np.stack((velocities[:,0],-velocities[:,0],velocities[:,1],-velocities[:,1]),axis=1)
    intercepts=np.stack((positions[:,0]-half_extents[:,0],-positions[:,0]-half_extents[:,0],
        positions[:,1]-half_extents[:,1],-positions[:,1]-half_extents[:,1]),axis=1)
    times=[np.full(len(positions),start,dtype=float),np.full(len(positions),end,dtype=float)]
    for first in range(4):
        for last in range(first+1,4):
            denominator=slopes[:,first]-slopes[:,last]
            candidate=np.divide(intercepts[:,last]-intercepts[:,first],denominator,
                out=np.full(len(positions),start,dtype=float),where=np.abs(denominator)>1e-12)
            times.append(np.where((candidate>start)&(candidate<end),candidate,start))
    times=np.stack(times,axis=1)
    return (intercepts[:,None,:]+slopes[:,None,:]*times[:,:,None]).max(axis=2).min(axis=1)
