"""Structured access to the existing original TH14 environment."""

import numpy as np

from ..contracts import Coverage, HORIZONS, HorizonResult, NativeFrame, RouteResult, ScriptObservation
from ...nbnm_corridor import LEGS
from .items import collection_state


class TH14Adapter:
    def __init__(self, environment):
        self.environment = environment
        self.info = {}

    def start(self, *, seed=None, options=None):
        _, self.info = self.environment.reset(seed=seed, options=options)
        return self.observe()

    def step(self, action):
        _, _, _, _, self.info = self.environment.step(action)
        return self.observe()

    def observe(self):
        env = self.environment
        raw = env.snapshot
        events = raw.extra.get("thrl_events", {})
        geometry = raw.extra.get("native_collision_geometry", {})
        return NativeFrame("th14", raw.frame, raw.stage, raw.stage_frame,
                           dict(player=raw.player, collision=geometry.get("player")),
                           env.controller.profile.bounds, raw.bullets, raw.lasers, raw.enemies,
                           raw.extra.get("native_attack"), raw.lives, raw.bombs, events.get("misses"),
                           events.get("bombs_used"), raw.extra.get("continues"), self.info.get("outcome"), events,
                           item_collection=collection_state(raw))

    def predict(self, frame, candidates, horizons=HORIZONS):
        raw = self._check_frame(frame)
        encoded = getattr(self.environment.encoder, "last_result", None)
        if encoded is None or (encoded["frame"], encoded["stage"]) != (frame.frame, raw.stage):
            encoded = {}
        data = encoded.get("details", {})
        corridor = encoded.get("corridor")
        result = []
        for tape in candidates:
            if not tape:
                raise ValueError("Supply nonempty movement tapes")
            windows = []
            rule_frames = 0
            requested = (np.repeat([step.action for step in tape], [min(step.frames, 96) // 4 for step in tape])
                         if corridor is not None and all(step.frames % 4 == 0 for step in tape) else ())
            for horizon in horizons:
                if type(horizon) is not int or horizon < 1:
                    raise ValueError("Prediction horizons are positive original frames")
                action = tape[0].action
                if (corridor is not None and horizon in LEGS and horizon <= len(requested) * 4):
                    index = LEGS.index(horizon)
                    commands = corridor["commands"][index, action, :index + 1]
                    margin = float(corridor["margins"][index, action])
                    if (margin > 0 and np.isfinite(margin) and (commands >= 0).all()
                            and np.array_equal(np.repeat(commands, np.diff((0, *LEGS[:index+1])) // 4),
                                               requested[:horizon // 4])):
                        windows.append(HorizonResult(horizon, Coverage.RULE, margin, contact_free=True))
                        rule_frames = max(rule_frames, horizon)
                        continue
                if (len(tape) == 1 and tape[0].frames >= horizon and horizon in HORIZONS[:3]
                        and "margins" in data and "trusted" in data):
                    index = HORIZONS.index(horizon)
                    trust = data["trusted"][index]
                    if bool(trust[action] if np.ndim(trust) else trust):
                        gap = float(data["margins"][index, action])
                        windows.append(HorizonResult(horizon, Coverage.RULE, gap, contact_free=gap > 0))
                        continue
                windows.append(HorizonResult(horizon, Coverage.UNKNOWN))
            result.append(RouteResult(tape, tuple(windows), rule_corridor_frames=rule_frames))
        return tuple(result)

    def script(self, frame):
        raw = self._check_frame(frame)
        from ...thrl_event_geometry import read_enemy_ecl_roles
        data = getattr(self.environment.encoder, "last_result", None) or {}
        emissions = data.get("source_native_forecasts") if data.get("frame") == frame.frame else None
        return ScriptObservation(frame.frame, frame.attack, raw.extra.get("native_attack_clock"),
                                 read_enemy_ecl_roles(self.environment.controller, raw), emissions)

    def _check_frame(self, frame):
        raw = self.environment.snapshot
        if frame.game != "th14" or raw.frame != frame.frame:
            raise ValueError("Use the current TH14 original input checkpoint")
        return raw
