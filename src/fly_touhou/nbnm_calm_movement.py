"""Bounded cost for avoidable movement in measured, settled NBNM states."""
import numpy as np

_IDLE = [0, 9]


def waiting_actions(encoding, *, require_continuation=False):
    """Recognize a usable pause and leave room for its next evasive turn."""
    if not encoding:
        return ()
    f = encoding['features']
    details = encoding['details']
    if not details.get('normal') or not np.asarray(details['trusted'], bool).all():
        return ()
    if f[5] > 0 or f[13] > 0:
        return ()
    if encoding['stage'] == 6 and f[274] > .5 and round(float(f[275]) * 32) == 5:
        return ()  # The orb's approach and release are active choices.
    x, y = f[0] * 368 - 184, f[1] * 400 + 32
    if abs(x) > 160 or not 248 <= y <= 415:
        return ()
    if f[113] > .5 and not (f[117] > .5 and abs(f[114] * 368) <= 32 and f[115] * 400 <= -80):
        return ()
    if f[2] < .95 and f[118] > 0 and np.hypot(f[119] * 368, f[120] * 400) < 160:
        return ()
    if f[123] > .5 and np.hypot(f[124] * 368, f[125] * 400) < 180:
        return ()
    # Include birth/EX warnings composed into the observation and actual native
    # geometry. Two speed variants of a pause have the same displacement.
    margin = np.asarray(details['margins']).reshape(3, 18)
    hints = f[18:72].reshape(3, 18)
    if not (margin[:, _IDLE] >= 32).all() or not (hints[:, _IDLE] <= np.tanh((1 - 32) / 8)).all():
        return ()
    corridor = encoding.get('corridor')
    if corridor is not None:
        paths = corridor['features']
        found = (paths[:, 6] > 0) & (paths[:, 2:4] > 0).all(1)
        if found.any():
            idle = paths[_IDLE]
            best = paths[found, :2].max(0)
            supported = found[_IDLE] & (idle[:, :2] >= .5).all(1) & (idle[:, :2] + .15 >= best).all(1)
            if require_continuation:
                from .nbnm_corridor import commands4
                for i, action in enumerate(_IDLE):
                    if not supported[i] or idle[i, 3] < .75 or idle[i, 6] < .5:
                        supported[i] = False
                        continue
                    course = commands4(corridor, action)
                    supported[i] = course is not None and len(course) >= 24 and course[0] % 9 == 0
            return tuple(action for action, allowed in zip(_IDLE, supported) if allowed)
        if (paths[:, 6] > 0).any():
            return ()
    return tuple(_IDLE) if not require_continuation and margin.min() >= 24 else ()



def can_wait(encoding, *, require_continuation=False):
    return bool(waiting_actions(encoding, require_continuation=require_continuation))


class CalmMovement:
    """Measure actual displacement on two successive quiet observations."""
    def __init__(self, limit, frames, *, continuation_supervision=False):
        self.limit = float(limit)
        self.frames = frames
        self.continuation_supervision = continuation_supervision
        self.spent = 0.
        self.previous_quiet = None
        self.previous_displacement = None

    def observe(self, encoding, before, after, *, charge=True):
        wait_actions = waiting_actions(encoding, require_continuation=self.continuation_supervision)
        quiet = bool(wait_actions)
        consecutive = quiet and self.previous_quiet == (before.stage, before.frame - 4)
        self.previous_quiet = (before.stage, before.frame) if quiet else None
        previous = self.previous_displacement
        self.previous_displacement = None
        ticks = after.frame - before.frame
        if not consecutive or before.stage != after.stage or ticks != 4 or after.player is None:
            return dict(eligible=False, idle=False, reversal=False, distance=0., cost=0.)
        delta = np.asarray([after.player.x - before.player.x, after.player.y - before.player.y])
        distance = float(np.linalg.norm(delta))
        idle = distance <= 1.
        reversal = False
        if not idle and previous is not None:
            reversal = bool(delta @ previous < -.5 * distance * np.linalg.norm(previous))
        if not idle:
            self.previous_displacement = delta
        fraction = min(1., distance / (4.5 * ticks)) if not idle else 0.
        reversal_cost = .5 * reversal if not self.continuation_supervision else 0.
        rate = self.limit * (fraction + reversal_cost) * ticks / self.frames
        cost = min(max(0., self.limit - self.spent), rate) if charge else 0.
        self.spent += cost
        result = dict(eligible=True, idle=idle, reversal=reversal, distance=distance, cost=cost)
        if self.continuation_supervision:
            result['wait_actions'] = wait_actions
        return result
