"""Current TH14 curve-laser collision geometry from 440d80 / 44f0b0.

Required node fields are native position, angle (+0x18) and speed/length
(+0x1c), stride 0x20. The current array count is LaserBase+0x5e0 and its
pointer is +0x14ac. There is no reconstruction from adjacent positions.

440d80 visits indices 0..n-2. It adds each raw speed/length to a float32
accumulator and skips collision while that sum is below 16. Each remaining
node produces a rotated rectangle, not a circle or an adjacent-node capsule.
Its native total width is visual/native LaserBase+0x74 divided by two, so
the actual collision half-width is LaserBase+0x74 / 4.

Player half extents must be the measured Player+0x648/+0x64c values. As the
native helper does, these expand its local long/short axes directly; they
are not a circle radius or a mathematically reprojected world-space AABB.

Only current geometric overlap is returned. No future node motion, engine
laser lifecycle, player state, invulnerability or deathbomb state is implied.
Python/NumPy double trig approximates native x87/CRT trig before the same
float32 stores; this is a geometric transcription, not bit-exact CPU emulation.
"""
from __future__ import annotations

from collections.abc import Mapping
import math

import numpy as np


SCHEMA = 'th14-native-curve-contact-v1'


def _node_array(nodes):
    """Accept Nx4 (x,y,angle,s) or named raw native-node records."""
    if len(nodes) and isinstance(nodes[0], Mapping):
        values = []
        for node in nodes:
            position = node.get('position', node.get('pos'))
            speed = node.get('speed', node.get('s'))
            if position is None or speed is None or node.get('angle') is None:
                raise ValueError('Native curve nodes require position, angle and speed/length')
            values.append([position[0], position[1], node['angle'], speed])
        result = np.asarray(values, dtype=np.float32)
    else:
        result = np.asarray(nodes, dtype=np.float32)
    if result.size == 0:
        result = np.empty((0, 4), dtype=np.float32)
    if result.ndim != 2 or result.shape[1] != 4 or not np.isfinite(result).all():
        raise ValueError('Native curve nodes must be finite Nx4 (x,y,angle,speed/length)')
    return result


def _curve_evaluation(nodes, laser_width, player_centers, player_half_extents,
                      max_relevant_margin=None, include_pending_head=False):
    raw = _node_array(nodes)
    player = np.asarray(player_centers, dtype=np.float32)
    half = np.asarray(player_half_extents, dtype=np.float32)
    width = np.float32(laser_width)
    if (player.ndim != 2 or player.shape[1] != 2 or half.shape != (2,) or
            not np.isfinite(player).all() or not np.isfinite(half).all() or
            not np.isfinite(width) or np.any(half < 0) or width < 0):
        raise ValueError('Finite player XY rows, nonnegative native half extents and width required')
    used = raw[:-1]
    count = len(used)
    lengths = used[:, 3]
    cumulative = np.cumsum(lengths, dtype=np.float32)
    eligible = cumulative >= np.float32(16.)
    if include_pending_head:
        # Sensory forecast only: observed head nodes can become collidable
        # when newly prepended nodes increase their accumulated head distance.
        eligible = eligible | (lengths > 0)
    angle64 = used[:, 2].astype(np.float64)
    half_length = (lengths * np.float32(.5)).astype(np.float32)
    # 443fe0 uses x87 sincos then stores the vector components as float32.
    directions = np.column_stack((np.cos(angle64), np.sin(angle64)))
    offset = (directions * half_length.astype(np.float64)[:, None]).astype(np.float32)
    anchor = (used[:, :2] + offset).astype(np.float32)
    if max_relevant_margin is not None and len(player) and count:
        # Discard rectangles too far from every sampled player position to
        # change a clipped sensory margin.  Eligibility is calculated BEFORE
        # this filter because native head exclusion depends on all prior nodes.
        margin = float(max_relevant_margin)
        if not np.isfinite(margin) or margin < 0:
            raise ValueError('A finite nonnegative clipping margin is required')
        half_width = float(width) / 4. + float(half[1])
        long_extent = np.maximum(float(half[0]), np.abs(lengths.astype(float) + float(half[0])))
        extent_x = np.abs(directions[:, 0]) * long_extent + np.abs(directions[:, 1]) * half_width
        extent_y = np.abs(directions[:, 1]) * long_extent + np.abs(directions[:, 0]) * half_width
        lower, upper = player.min(axis=0), player.max(axis=0)
        # Extra rounding room covers the float32 native coordinate stores.
        room = margin + 0.01
        keep = (eligible & (anchor[:, 0] + extent_x >= lower[0] - room)
                & (anchor[:, 0] - extent_x <= upper[0] + room)
                & (anchor[:, 1] + extent_y >= lower[1] - room)
                & (anchor[:, 1] - extent_y <= upper[1] + room))
        used, cumulative, eligible = used[keep], cumulative[keep], eligible[keep]
        lengths, angle64, anchor = lengths[keep], angle64[keep], anchor[keep]
    delta = (player[:, None, :] - anchor[None, :, :]).astype(np.float32)
    cos_negative = np.cos(-angle64).astype(np.float32)
    sin_negative = np.sin(-angle64).astype(np.float32)
    local_x = (delta[:, :, 0] * cos_negative - delta[:, :, 1] * sin_negative).astype(np.float32)
    local_y = (delta[:, :, 1] * cos_negative + delta[:, :, 0] * sin_negative).astype(np.float32)
    half_width = np.float32(np.float32(width * np.float32(.5)) * np.float32(.5))
    # Preserve native float32 rounding of +/- player extents, then compare the
    # resulting values to the exact native rectangle bounds [0,s],[-W/4,W/4].
    lower_x = (local_x - half[0]).astype(np.float32).astype(np.float64)
    upper_x = (local_x + half[0]).astype(np.float32).astype(np.float64)
    lower_y = (local_y - half[1]).astype(np.float32).astype(np.float64)
    upper_y = (local_y + half[1]).astype(np.float32).astype(np.float64)
    separation_x = np.maximum(lower_x - lengths.astype(np.float64), -upper_x)
    separation_y = np.maximum(lower_y - float(half_width), -float(half_width) - upper_y)
    margins = (np.hypot(np.maximum(separation_x, 0.), np.maximum(separation_y, 0.)) +
               np.minimum(np.maximum(separation_x, separation_y), 0.))
    margins[:, ~eligible] = math.inf
    overlaps = eligible & (separation_x <= 0.) & (separation_y <= 0.)
    return dict(raw=raw, used=used, cumulative=cumulative, eligible=eligible,
        anchor=anchor, local_x=local_x, local_y=local_y, half_width=half_width,
        margins=margins, overlaps=overlaps, player_count=len(player),
        potential_head_activation=bool(include_pending_head))


def _summary(values):
    raw, eligible = values['raw'], values['eligible']
    return dict(schema=SCHEMA, node_count=len(raw), candidate_node_count=len(eligible),
        tested_node_count=int(np.count_nonzero(eligible)),
        skipped_head_node_count=int(np.count_nonzero(~eligible)),
        ignored_last_node_index=len(raw)-1 if len(raw) else None,
        player_protection_applied=False, future_motion_predicted=False,
        potential_head_activation=values['potential_head_activation'],
        geometry='native per-node oriented rectangles; no adjacent-node inference')


def curve_laser_contact_batch(nodes, laser_width, player_centers, player_half_extents,
                              *, include_per_node=False, max_relevant_margin=None, include_pending_head=False):
    """Evaluate P candidate player positions with one shared native-node batch.

    ``player_centers`` is Px2; return NumPy arrays ``signed_clearances`` and
    ``geometric_collisions`` of length P. No candidate position is treated as
    an actually executed action or a forecast of the laser's future geometry.
    With ``include_per_node`` the additional arrays have shape (P,n-1).
    ``include_pending_head`` adds possible future activation of currently
    excluded head nodes. That option is not a current native collision test.
    """
    if include_per_node and max_relevant_margin is not None:
        raise ValueError('Per-node reports require the unfiltered native array')
    values = _curve_evaluation(nodes, laser_width, player_centers, player_half_extents,
                               max_relevant_margin=max_relevant_margin,include_pending_head=include_pending_head)
    margins, overlaps = values['margins'], values['overlaps']
    result = _summary(values)
    result.update(player_count=values['player_count'],
        signed_clearances=np.min(margins, axis=1) if margins.shape[1] else
                          np.full(values['player_count'], math.inf, dtype=np.float64),
        geometric_collisions=np.any(overlaps, axis=1))
    if include_per_node:
        result.update(per_node_clearances=margins, per_node_collisions=overlaps,
                      node_indices=np.arange(len(values['eligible']), dtype=np.int64),
                      node_tested=values['eligible'])
    return result


def curve_laser_contact(nodes, laser_width, player_center, player_half_extents):
    """Return per-node and overall signed clearance/current geometric overlap.

    Clearance is the signed Euclidean distance to the native local rectangle
    expanded by the measured player half extents. Zero is touching and counts
    as overlap, matching the helper's inclusive comparisons. Untested head
    nodes have +inf clearance. The last array node is never tested. Negative
    raw speed/length is retained, rather than replaced by a neighbor distance.

    ``geometric_collision`` is the native rectangle intersection test only;
    the caller must separately apply native lifecycle and player protection.
    """
    player = np.asarray(player_center, dtype=np.float32)
    if player.shape != (2,):
        raise ValueError('A finite single player XY position is required')
    values = _curve_evaluation(nodes, laser_width, player[None, :], player_half_extents)
    used, cumulative, eligible = values['used'], values['cumulative'], values['eligible']
    anchor, local_x, local_y = values['anchor'], values['local_x'][0], values['local_y'][0]
    margins, overlaps = values['margins'][0], values['overlaps'][0]
    lengths, count = used[:, 3], len(used)
    per_node = [dict(node_index=i, cumulative_length=float(cumulative[i]),
        tested=bool(eligible[i]), position=used[i, :2].astype(float).tolist(),
        angle=float(used[i, 2]), length=float(lengths[i]),
        anchor=anchor[i].astype(float).tolist(),
        local_player=[float(local_x[i]), float(local_y[i])],
        half_width=float(values['half_width']), signed_clearance=float(margins[i]),
        geometric_collision=bool(overlaps[i])) for i in range(count)]
    result = _summary(values)
    result.update(signed_clearance=float(np.min(margins)) if count else math.inf,
        geometric_collision=bool(np.any(overlaps)), per_node=per_node)
    return result
