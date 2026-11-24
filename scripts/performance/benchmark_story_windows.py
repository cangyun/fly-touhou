"""Profile short windows of the archived, successful TH14 action tape on the original game.

Outside windows, use the existing tape advancement and ordinary dialogue bridges.
The archived controls isolate the native worker from policy inference.
"""
import argparse
from collections import defaultdict
from functools import wraps
import json
from pathlib import Path
import pickle
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]


class StepTimer:
    def __init__(self):
        self.active = False
        self.durations = defaultdict(float)

    def wrap(self, obj, name, label):
        original = getattr(obj, name)

        @wraps(original)
        def timed(*args, **kwargs):
            if not self.active:
                return original(*args, **kwargs)
            start = perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                self.durations[label] += perf_counter() - start

        setattr(obj, name, timed)
        return original

    def start(self):
        self.durations.clear()
        self.active = True
        self.started = perf_counter()

    def stop(self):
        elapsed = perf_counter() - self.started
        self.active = False
        d = self.durations
        return dict(total=elapsed, native=d["advance"] - d["read"], read=d["read"],
                    corridor=d["corridor"], route_context=d["route_context"],
                    base_encoder=d["encode"] - d["corridor"] - d["route_context"],
                    teacher=d["teacher"], other=elapsed - d["advance"] - d["encode"] - d["teacher"])


def choose_windows(stages, clock, features, warm, size):
    windows = []
    for stage in range(1, 7):
        group = np.flatnonzero(stages == stage)
        for part in (.12, .36, .61, .84):
            target = group[int(len(group) * part)]
            for i in range(int(target), int(group[-1]) - size):
                interval = slice(i - warm, i + size + 1)
                if stages[i-warm] != stage or (features[i:i+size, 14] <= .5).any():
                    continue
                if np.all(np.diff(clock[interval]) == 4):
                    windows.append((stage, i))
                    break
            else:
                raise ValueError(("No regular tape interval", stage, part))
    if any(b - a < warm + size for (_, a), (_, b) in zip(windows, windows[1:])):
        raise ValueError("Chosen windows overlap")
    return windows


def bridges(stages, phases):
    """Keep published input-tape semantics around the original dialogue transitions."""
    result = np.zeros(len(stages), dtype=bool)
    for s in range(1, 7):
        group = np.flatnonzero(stages == s)
        bosses = np.flatnonzero((stages == s) & (phases == f"s{s}/Boss1"))
        if len(bosses):
            cut = int(bosses[0])
            lo, hi = max(group[0], cut - 420), min(group[-1] + 1, cut + 190)
            result[lo:hi] = True
        if group[-1] + 1 < len(stages):
            result[max(group[0], group[-1] - 170):group[-1] + 19] = True
    return result


def assert_native(env, index, recorded):
    s = env.snapshot
    features, stages, clocks = recorded
    expected = int(stages[index]), int(clocks[index])
    if (s.stage, s.stage_frame) != expected:
        raise ValueError(("Original clock differs", index, (s.stage, s.stage_frame), expected))
    if s.player is not None and features[index, 14] > .5:
        x, y = features[index, :2].astype(float) * [368, 400] + [-184, 32]
        if np.hypot(s.player.x - x, s.player.y - y) > .05:
            raise ValueError(("Original position differs", index, s.player.x, s.player.y, x, y))
    events = s.extra["thrl_events"]
    if events["misses"] or events["bombs_used"] or (s.player is not None and s.player.state == 4):
        raise ValueError(("Archived actions did not preserve NBNM", index))


def skip_to(env, i, dest, actions, recorded, single, masks):
    _, stages, clocks = recorded
    while i < dest:
        last = i
        if not env.snapshot.dialogue:
            while last < min(dest, i + 120) and not single[last] and stages[last+1] == stages[last] and clocks[last+1] - clocks[last] == 4:
                last += 1
        if last == i:
            _, _, done, truncated, _ = env.step(int(actions[i]))
            if done or truncated:
                raise ValueError(("A covered tape interval ended early", i))
            i += 1
        else:
            before = env.snapshot
            after = env._check_snapshot(env.controller.advance_tape(masks[i:last]))
            env.snapshot = after
            env._native_input_ticks += (last - i) * env.frame_skip
            env.episode_frames += after.frame - before.frame
            env.policy_steps += last - i
            env._verify_native_run(after)
            env._record_stage(after)
            if env._native_end(after):
                raise ValueError(("An archived tape interval ended early", last, after.stage_frame))
            env.reward_model.reset(after)
            env._last_measurable_frame = after.frame
            i = last
        assert_native(env, i, recorded)
    return i


def run(demo, output, size, warm, *, teacher_enabled=False):
    from fly_touhou.background import action_mask
    from fly_touhou.compact_full_story import factory, story_kwargs
    from fly_touhou.nbnm_run import environment, runtime, task_settings
    from fly_touhou.thrl_durable_records import write_json
    from fly_touhou.thrl_job_paths import cleanup_native_temp
    from fly_touhou import nbnm_corridor, nbnm_sensation

    model_dir = demo / "model"
    spec = json.loads((model_dir / "archived-inference.json").read_text(encoding="utf-8-sig"))
    archived = model_dir / "original-decisions.npz"
    with np.load(archived, allow_pickle=False) as data:
        features, stages, clocks, phases, actions = (data[k] for k in ("features", "stages", "stage_frames", "phases", "actions"))
    recorded = features, stages, clocks
    windows = choose_windows(stages, clocks, features, warm, size)
    single = bridges(stages, phases)
    temp = runtime(output)
    env = None
    started = perf_counter()
    timers = StepTimer()
    captured = []
    record_fixture = None
    original_plan = nbnm_corridor.plan
    original_context = nbnm_sensation.route_context

    @wraps(original_plan)
    def plan(encoding, *args, **kwargs):
        if record_fixture:
            record_fixture.write_bytes(pickle.dumps((encoding, args, kwargs), protocol=5))
        return original_plan(encoding, *args, **kwargs)

    try:
        config = environment(spec, evaluation=True)
        config["nbnm"] = task_settings(spec, spec["tasks"]["full_story"])
        options = story_kwargs(config, output / "worker", 1, "development", teacher=teacher_enabled)[0]
        if teacher_enabled:
            from fly_touhou.resource_teacher_v10 import REVISION
            options["online_teacher_revision"] = REVISION
        env = factory(options)
        _, _ = env.reset(options={"native_seed": 60036})
        teacher = env._teacher
        timers.wrap(env, "_encode", "encode")
        timers.wrap(env, "_label_current", "teacher")
        timers.wrap(env.encoder, "current_corridor", "corridor")
        timers.wrap(env.controller, "read", "read")
        timers.wrap(env.controller, "advance", "advance")
        timed_encode = env._encode
        nbnm_corridor.plan = plan
        timers.wrap(nbnm_sensation, "route_context", "route_context")
        masks = [(action_mask(env.policy_to_native[int(a)] % 18), env.frame_skip) for a in actions]
        blank = env._last_measurable_observation

        def placeholder(snapshot):
            env._last_measurable_frame = snapshot.frame
            env._last_measurable_observation = blank
            return blank

        i = 0
        for stage, first in windows:
            env._teacher = None
            env._encode = placeholder
            env.encoder.last_result = None
            go = perf_counter()
            i = skip_to(env, i, first-warm, actions, recorded, single, masks)
            fast_forward = perf_counter() - go
            env._encode = timed_encode
            env.encoder.reset()
            env._encode(env.snapshot)
            age = float(features[i, 276])
            env.encoder._attack_start = env.snapshot.frame - round(600 * age / (1-age))
            env.encoder._nbnm_cache = None
            env._encode(env.snapshot)
            env._teacher = teacher
            if teacher is not None:
                teacher.reset()
                env._label_current(env._info())
            samples = []
            while i < first + size:
                if i == first - 1:
                    record_fixture = output / "scenes" / f"s{stage}-{int(clocks[first])}.pkl"
                if i >= first:
                    timers.start()
                obs, _, done, truncated, info = env.step(int(actions[i]))
                elapsed = timers.stop() if i >= first else None
                record_fixture = None
                if done or truncated:
                    raise ValueError(("Measured tape window ended early", i))
                assert_native(env, i+1, recorded)
                if elapsed is not None:
                    observed = obs["features"]
                    expected = features[i+1]
                    samples.append(dict(index=i, stage_frame=int(clocks[i+1]),phase=str(phases[i+1]),
                                        features281_max_abs=float(np.max(np.abs(observed[:281]-expected[:281]))),
                                        corridor_max_abs=float(np.max(np.abs(observed[281:]-expected[281:]))),
                                        teacher_label=({k: info["teacher_label"].get(k) for k in
                            ("action", "probabilities", "guidance_weight", "guidance_weight_reason")}
                            if teacher_enabled else None),
                        **elapsed))
                i += 1
            captured.append(dict(stage=stage, clock=int(clocks[first]), phase=str(phases[first]),
                                 fast_forward_seconds=fast_forward, samples=samples))
            write_json(output/"windows.json", dict(seed=60036, windows=captured,
                       samples=sum(len(x["samples"]) for x in captured), elapsed_seconds=perf_counter()-started,
                       scope="In-process native worker only. Excludes GPU and parent/worker pipes."))
            print(stage, int(clocks[first]), str(phases[first]),
                  round(np.mean([a["total"] for a in samples])*1000, 1), "ms",
                  "obs", max(a["features281_max_abs"] for a in samples),
                  "corridor", max(a["corridor_max_abs"] for a in samples), flush=True)
    finally:
        nbnm_corridor.plan = original_plan
        nbnm_sensation.route_context = original_context
        if env is not None:
            env.close()
        cleanup_native_temp(temp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--warm", type=int, default=12)
    parser.add_argument("--with-teacher", action="store_true")
    args = parser.parse_args()
    (args.output/"scenes").mkdir(exist_ok=True, parents=True)
    try:
        run(args.demo, args.output, args.window, args.warm, teacher_enabled=args.with_teacher)
    except BaseException:
        import traceback
        (args.output/"failure.txt").write_text(traceback.format_exc(),encoding="utf-8")
        raise
