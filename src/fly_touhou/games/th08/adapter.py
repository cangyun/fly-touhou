"""Structured access to the existing original TH08 Story world."""

from ..contracts import Coverage, HORIZONS, HorizonResult, NativeFrame, RouteResult, ScriptObservation
from .items import collection_state
from .native_forecast import query_live_native_routes
from .sensation import POLICY_MASKS


class TH08Adapter:
    def __init__(self, world):
        self.world = world

    def start(self):
        self.world.start()
        return self.observe()

    def step(self, action):
        self.world.step(action)
        return self.observe()

    def observe(self):
        world = self.world
        info, game = world.status(), world.game
        collection = None
        if world.outcome is None:
            motion = world.controller.read_motion(world.state)
            objects = world.controller.read_projectiles()
            player = motion
            bounds = motion.minimum, motion.maximum
            bullets, lasers = objects["bullets"], objects["lasers"]
            enemies = world.controller.read_enemies()
            collection = collection_state(world.state, game, motion, world.controller.read_items())
        else:
            player = bounds = bullets = lasers = enemies = None
        return NativeFrame("th08", world.state["input_poll"], info["stage"], info["stage_frame"],
                           player, bounds, bullets, lasers, enemies, info["native_spell_id"],
                           game.get("lives"), game.get("bombs"), info["misses"], info["Bombs"],
                           info["continues"], info["outcome"], dict(contact=info["contact"],
                           spell_result=game.get("native_spell_result_id"), normal_ending=game["normal_ending"]),
                           item_collection=collection)

    def predict(self, frame, candidates, horizons=HORIZONS):
        self._check_frame(frame)
        result = self.world.encoder.last_result
        if not candidates:
            return ()
        if result is None or self.world.encoder.native_forecast_controller is None:
            return tuple(RouteResult(tape, tuple(HorizonResult(h, Coverage.UNKNOWN) for h in horizons))
                         for tape in candidates)
        if any(not tape for tape in candidates) or any(type(h) is not int or h < 1 for h in horizons):
            raise ValueError("Supply nonempty movement tapes and positive native horizons")
        masks = [[(POLICY_MASKS[step.action], step.frames) for step in tape] for tape in candidates]
        traces, witness = query_live_native_routes(self.world.controller, masks,
                                                  max_ticks=max(sum(n for _, n in tape) for tape in masks))
        if witness["input_poll"] != frame.frame:
            raise RuntimeError("Original TH08 returned a different input checkpoint")
        return tuple(self._route(tape, samples, horizons)
                     for tape, samples in zip(candidates, traces, strict=True))

    @staticmethod
    def _route(tape, trace, horizons):
        prefix = []
        for entry in trace:
            if not entry["geometry_valid"] and not entry["contact"]:
                break
            prefix.append(entry)
            if entry["contact"]:
                break
        contact = next((i for i, row in enumerate(prefix, 1) if row["contact"]), None)
        windows = []
        for horizon in horizons:
            sample = prefix[:horizon]
            hit = contact if contact is not None and contact <= horizon else None
            complete = hit is not None or len(sample) == horizon
            gap = min((row["clearance"] for row in sample if row["clearance"] is not None), default=None)
            windows.append(HorizonResult(horizon, Coverage.ORIGINAL if complete else Coverage.UNKNOWN,
                                         gap, hit, hit is None if complete else None, len(sample)))
        length = len(prefix) if contact is None else contact - 1
        return RouteResult(tape, tuple(windows), length)

    def script(self, frame):
        self._check_frame(frame)
        source = self.world.controller.read_forecast_source()
        ecl = source["ecl"]
        projectiles = source["projectiles"]
        return ScriptObservation(frame.frame, frame.attack, ecl["timelines"], ecl["emitters"],
                                 {key: projectiles[key] for key in ("scripted_beams", "pending_births")
                                  if key in projectiles})

    def _check_frame(self, frame):
        if frame.game != "th08" or self.world.state["input_poll"] != frame.frame:
            raise ValueError("Use the current TH08 original input checkpoint")
