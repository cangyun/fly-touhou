"""Continuous minimum clearance of a moving player against rigid circles and rectangles."""
import math

import numpy as np
from numba import njit


@njit(cache=True, inline='always')
def _wall_time(position, speed, low, high, duration):
    if abs(speed) <= 1.e-12:
        return duration
    wall = high if speed > 0. else low
    return min(duration, max(0., (wall - position) / speed))


@njit(cache=True, inline='always')
def _rect_at(intercepts, slopes, t):
    return max(intercepts[0] + slopes[0]*t, intercepts[1] + slopes[1]*t,
               intercepts[2] + slopes[2]*t, intercepts[3] + slopes[3]*t)


@njit(cache=True)
def minimum_clearances(starts, velocities, positions, motions, circles, radii, extents,
                       offset, initial_time, duration):
    result = np.full(len(starts), np.inf)
    for k in range(len(starts)):
        x, y = starts[k]
        vx, vy = velocities[k]
        tx = _wall_time(x, vx, -184., 184., duration)
        ty = _wall_time(y, vy, 32., 432., duration)
        boundaries = (0., min(tx, ty), max(tx, ty), duration)
        for part in range(3):
            first, last = boundaries[part], boundaries[part + 1]
            length = last - first
            if length <= 1.e-10:
                continue
            middle = (first + last)*.5
            px = min(184., max(-184., x + vx*first))
            py = min(432., max(32., y + vy*first))
            sx = vx if -184. < x + vx*middle < 184. else 0.
            sy = vy if 32. < y + vy*middle < 432. else 0.
            clock = initial_time + first
            for j in range(len(positions)):
                dx = px - (positions[j, 0] + motions[j, 0]*clock)
                dy = py - (positions[j, 1] + motions[j, 1]*clock)
                ux = sx - motions[j, 0]
                uy = sy - motions[j, 1]
                if circles[j]:
                    travel_x, travel_y = abs(ux)*length, abs(uy)*length
                    lower = max(abs(dx) - travel_x, abs(dy) - travel_y) - radii[j]
                    # Bound each component over the entire wall-clamped
                    # phase. Allow for FP64 rounding before skipping a root.
                    slack = 1.e-12 * (1. + abs(dx) + abs(dy) + travel_x + travel_y + abs(radii[j]))
                    if lower > result[k] + slack:
                        continue
                    speed2 = ux*ux + uy*uy
                    nearest = -(dx*ux + dy*uy)/speed2 if speed2 > 1.e-12 else 0.
                    t = min(length, max(0., nearest))
                    xx, yy = dx + ux*t, dy + uy*t
                    clearance = math.sqrt(xx*xx + yy*yy) - radii[j]
                else:
                    dx += offset[0]
                    dy += offset[1]
                    w, h = extents[j]
                    travel_x, travel_y = abs(ux)*length, abs(uy)*length
                    lower = max(abs(dx) - travel_x - w, abs(dy) - travel_y - h)
                    slack = 1.e-12 * (1. + abs(dx) + abs(dy) + travel_x + travel_y + abs(w) + abs(h))
                    if lower > result[k] + slack:
                        continue
                    intercepts = (dx-w, -dx-w, dy-h, -dy-h)
                    slopes = (ux, -ux, uy, -uy)
                    clearance = min(max(intercepts), _rect_at(intercepts, slopes, length))
                    # The maximum of four affine faces reaches its minimum at
                    # an endpoint or at an intersection of a pair of faces.
                    for a in range(4):
                        for b in range(a+1, 4):
                            denominator = slopes[a] - slopes[b]
                            if abs(denominator) > 1.e-12:
                                t = min(length, max(0., (intercepts[b]-intercepts[a])/denominator))
                                clearance = min(clearance, _rect_at(intercepts, slopes, t))
                result[k] = min(result[k], clearance)
    return result


@njit(cache=True)
def stationary_volley_clearances(volleys, paths, begin, end, player_radius):
    result=np.full(len(paths),np.inf)
    for i in range(len(volleys)):
        first,last,x,y,radius=volleys[i]
        start=max(begin+1,int(first))
        stop=min(end,int(last))
        if start>stop:
            continue
        contact_radius=math.sqrt(radius*radius+player_radius*player_radius)+.05
        for j in range(len(paths)):
            closest=np.inf
            for t in range(start,stop+1):
                dx=paths[j,t,0]-x
                dy=paths[j,t,1]-y
                closest=min(closest,dx*dx+dy*dy)
            result[j]=min(result[j],math.sqrt(closest)-contact_radius)
    return result


@njit(cache=True)
def moving_ring_clearances(discs,counts,paths,begin,end):
    result=np.full(len(paths),np.inf)
    for t in range(begin+1,end+1):
        for p in range(len(paths)):
            x,y=paths[p,t]
            nearest=result[p]
            for i in range(counts[t]):
                dx=x-discs[t,i,0]
                dy=y-discs[t,i,1]
                radius=discs[t,i,2]
                if max(abs(dx),abs(dy))-radius<nearest:
                    nearest=min(nearest,math.sqrt(dx*dx+dy*dy)-radius)
            result[p]=nearest
    return result


@njit(cache=True)
def moving_arc_clearances(arcs, counts, paths, begin, end):
    """Exact minimum over a fixed-radius arc and the native contact radius."""
    result = np.full(len(paths), np.inf)
    for t in range(begin + 1, end + 1):
        for p in range(len(paths)):
            x, y = paths[p, t]
            nearest = result[p]
            for i in range(counts[t]):
                center_x, center_y, ux, uy, radius, cosine, sine, contact, bound = arcs[t, i]
                dx, dy = x - center_x, y - center_y
                if max(abs(dx), abs(dy)) - bound >= nearest:
                    continue
                along = radius + dx * ux + dy * uy
                across = abs(dy * ux - dx * uy)
                radial_distance = math.sqrt(along * along + across * across)
                if along >= radial_distance * cosine:
                    distance = abs(radial_distance - radius)
                else:
                    a, b = along - radius * cosine, across - radius * sine
                    distance = math.sqrt(a * a + b * b)
                nearest = min(nearest, distance - contact)
            result[p] = nearest
    return result


@njit(cache=True)
def accumulate_reactive_clearance(answer, paths, centers, offset, radius, begin, end):
    """Swept relative segments after the candidate-specific orb simulation."""
    for k in range(len(paths)):
        nearest = answer[k]
        for t in range(math.floor(begin), math.ceil(end)):
            x = np.float64(paths[k,t,0]) - np.float64(centers[k,t,0]) + offset[0]
            y = np.float64(paths[k,t,1]) - np.float64(centers[k,t,1]) + offset[1]
            dx = np.float64(paths[k,t+1,0]) - np.float64(centers[k,t+1,0]) + offset[0] - x
            dy = np.float64(paths[k,t+1,1]) - np.float64(centers[k,t+1,1]) + offset[1] - y
            length2 = dx*dx + dy*dy
            fraction = -(x*dx+y*dy)/length2 if length2 > 1e-12 else 0.
            fraction = np.minimum(np.maximum(fraction, max(0.,begin-t)), min(1.,end-t))
            x, y = x + fraction*dx, y + fraction*dy
            nearest = np.minimum(nearest, math.sqrt(x*x+y*y)-radius)
        answer[k] = nearest
