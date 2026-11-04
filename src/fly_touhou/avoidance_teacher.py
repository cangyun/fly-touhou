"""Pure reactive movement lesson; no collection, score or chase objectives."""
from copy import deepcopy
import math

import numpy as np

from .movement_curriculum import specification, target_from_clearances
from .native_teacher_feedback import make_teacher
from .resource_teacher import ResourceTeacher
from .state import decode_action


def avoidance_specification():
    result = deepcopy(specification())
    result.update(schema='native-pure-avoidance-demonstration-curriculum-v2',
        teacher='measured geometry only; no resource mode, item target or shooting-lane target',
        target='vulnerable threat: maximum-clearance band0.5px; other vulnerable states: quiet lower-interior movement, focused stillness preferred',
        data_source='explicit native teacher demonstrations or student-visited native states, declared per run',
        teacher_actions_executed='training demonstrations explicitly permitted and counted; independent student evaluation must be zero',
        ordinary_anchor={'x_interval': [-56., 56.], 'y_interval': [352., 416.]},
        phase='basic imitation of pure avoidance before student-state correction and reward PPO')
    return result


class AvoidanceOnlyFeedback:
    def __init__(self):
        self.planner = make_teacher().planner
        self.contacts = ResourceTeacher(self.planner)

    def reset(self):
        self.planner.reset()

    @staticmethod
    def direction_target(snapshot, clearances, planner_config):
        c = np.asarray(clearances, dtype=np.float64)
        _, allowed, threat = target_from_clearances(np.full(36, 1/36), c)
        best = float(c.max())
        candidates = ((c >= best - .5) & allowed) if threat and best != math.inf else allowed
        p, config = snapshot.player, planner_config
        utilities = []
        for action in range(18):
            dx, dy, focused = decode_action(action)
            speed = config.focused_speed if focused else config.normal_speed
            norm = max(1., math.hypot(dx, dy))
            displacement = np.asarray((dx, dy)) * speed * 4. / norm
            position = np.clip(np.asarray((p.x, p.y)) + displacement,
                               (config.left, config.top), (config.right, config.bottom))
            outside = math.hypot(max(0., abs(position[0]) - 56.),
                                 max(0., 352. - position[1], position[1] - 416.))
            utilities.append(-outside / 18. - .02 * float(np.linalg.norm(displacement)) - .1 * (not focused))
        values = np.asarray(utilities)[candidates]
        mass = np.exp((values - values.max()) / .03)
        target = np.zeros(18, dtype=np.float64)
        target[candidates] = mass / mass.sum()
        return target, allowed, threat

    def query(self, snapshot):
        self.planner._prepare(snapshot)
        p = snapshot.player
        gameplay = (p is not None and snapshot.screen == 7 and snapshot.pause == 2
                    and snapshot.extra.get('native_gameover') is not True
                    and 'native_dialogue_object' in snapshot.extra
                    and snapshot.extra['native_dialogue_object'] in (0, None))
        if not gameplay or p.state != 1 or p.invulnerable_frames > 0:
            return np.full(18, 1/18), np.ones(18, dtype=bool), 'inactive', {
                'reason': 'non_gameplay_or_not_vulnerable', 'forecast_is_native_oracle': False}
        enemies = self.contacts._enemies(snapshot, snapshot.extra.get('resources', {}))
        clearances = []
        for action in range(18):
            bullet = self.planner.action_metrics(snapshot, action).clearance
            path = self.planner._path(snapshot, action)[0]
            contact = self.contacts._contact_clearance(snapshot, path, enemies)
            clearances.append(min(bullet, contact))
        target, allowed, threat = self.direction_target(snapshot, clearances, self.planner.config)
        kind = 'threat' if threat else 'control'
        return target, allowed, kind, {'reason': kind,
            'clearances': [None if v == math.inf else float(v) for v in clearances],
            'admissible': np.flatnonzero(allowed).tolist(), 'teacher_direction': int(target.argmax()),
            'teacher_mode': 'maximum_margin' if threat else 'quiet_lower_interior',
            'resource_or_shooting_lane_objective': False, 'forecast_is_native_oracle': False}
