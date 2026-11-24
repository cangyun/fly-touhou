"""Candidates for the minimum signed distance to TH14 curve-laser rectangles."""
import numpy as np
from numba import njit


@njit(cache=True)
def _minimum_candidates(points, anchors, lengths, eligible, cosine, sine, half, half_width):
    frames, paths = points.shape[:2]
    count = lengths.shape[1]
    # The scratch arrays cover one point. The upper bound usually leaves
    # only one or two rectangles for the final NumPy hypot.
    local_x = np.empty(count, np.float64)
    local_y = np.empty(count, np.float64)
    offsets = np.empty(frames * paths, np.intp)
    capacity = max(frames * paths, count, 1)
    kept_x = np.empty(capacity, np.float64)
    kept_y = np.empty(capacity, np.float64)
    size = 0
    for frame in range(frames):
        for path in range(paths):
            upper = np.inf
            start = size
            offsets[frame * paths + path] = start
            for column in range(count):
                if not eligible[frame, column]:
                    continue
                dx = points[frame, path, 0] - anchors[frame, column, 0]
                dy = points[frame, path, 1] - anchors[frame, column, 1]
                co = cosine[frame, column]
                si = sine[frame, column]
                x = dx * co - dy * si
                y = dy * co + dx * si
                sx = max(np.float64(x - half[0]) - lengths[frame, column], -np.float64(x + half[0]))
                sy = max(np.float64(y - half[1]) - half_width, -np.float64(half_width) - np.float64(y + half[1]))
                local_x[column] = sx
                local_y[column] = sy
                # Inside, the distance is max(sx, sy). Outside, the L1
                # distance is an upper bound on the Euclidean distance.
                value = max(sx, 0.) + max(sy, 0.) + min(max(sx, sy), 0.)
                upper = min(upper, value)
            upper = np.nextafter(upper, np.inf)
            if size + count > len(kept_x):
                capacity = max(size + count, 2 * len(kept_x))
                xs = np.empty(capacity, np.float64)
                ys = np.empty(capacity, np.float64)
                xs[:size] = kept_x[:size]
                ys[:size] = kept_y[:size]
                kept_x, kept_y = xs, ys
            for column in range(count):
                if not eligible[frame, column]:
                    continue
                sx, sy = local_x[column], local_y[column]
                if max(sx, sy) <= upper:
                    kept_x[size] = sx
                    kept_y[size] = sy
                    size += 1
            if size == start:
                kept_x[size] = np.inf
                kept_y[size] = 0.
                size += 1
    return offsets, kept_x[:size], kept_y[:size]


def minimum_clearances(points, anchors, lengths, eligible, cosine, sine, half, half_width):
    offsets, sx, sy = _minimum_candidates(points, anchors, lengths, eligible, cosine, sine, half, half_width)
    # Apply exactly the same double-precision NumPy operations to every
    # rectangle that can attain the minimum, including ties.
    distances = np.hypot(np.maximum(sx, 0), np.maximum(sy, 0)) + np.minimum(np.maximum(sx, sy), 0)
    return np.minimum.reduceat(distances, offsets).reshape(points.shape[:2])
