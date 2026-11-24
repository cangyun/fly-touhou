"""Retail TH08 branch traces and their per-tick collision geometry."""

from collections import Counter
import math
import time

import numpy as np

from ...nbnm_corridor import HORIZONS
from ...th08_native_ecl import TH08RngState


FORECAST_REVISION = "th08-original-branches-v1"


def checked_clearance(player_centers, hitboxes):
    """Signed clearance at positions in one fixed native branch frame."""
    centers = np.asarray(player_centers, dtype=np.float64)
    clearance = np.full(centers.shape[:-1], np.inf)
    for box in hitboxes:
        if box["kind"] == "bullet" and box["result"] == 2:
            # The original canceled this bullet before its lethal test.
            continue
        size = np.asarray(box["size"], dtype=np.float64) * .5
        center = np.asarray(box["position"], dtype=np.float64)
        if box["kind"] == "beam":
            origin = np.asarray(box["origin"], dtype=np.float64)
            dx, dy = np.moveaxis(centers - origin, -1, 0)
            angle = box["angle"] if box["angle"] is not None else math.nan
            cosine, sine = math.cos(angle), math.sin(angle)
            along = dx * cosine + dy * sine
            across = dy * cosine - dx * sine
            delta = np.stack((along, across), axis=-1) - (center - origin)
            gap = (np.abs(delta) - size - np.asarray(box["player_half"], dtype=np.float64)).max(axis=-1)
        else:
            player = np.asarray(box["player"], dtype=np.float64)
            lower = center - size - (np.asarray(box["player_max"], dtype=np.float64) - player)
            upper = center + size - (np.asarray(box["player_min"], dtype=np.float64) - player)
            gap = np.maximum(lower - centers, centers - upper).max(axis=-1)
        np.minimum(clearance, gap, out=clearance)
    return clearance


def sampled_safe_region(hitboxes, bounds, spacing=4.):
    """Sample the safe area for one frame of a fixed candidate branch."""
    lower, upper = bounds
    x = np.arange(lower[0], upper[0] + .001, spacing)
    y = np.arange(lower[1], upper[1] + .001, spacing)
    coordinates = np.stack(np.meshgrid(x, y), axis=-1)
    clearance = checked_clearance(coordinates, hitboxes)
    if np.isnan(clearance).any():
        return {"known": False}
    safe = clearance > 0.
    return {"known": True, "origin": list(lower), "spacing": spacing, "shape": list(safe.shape),
            "safe_cells": int(safe.sum()), "packed_rows": np.packbits(safe, axis=1, bitorder="little").tolist()}


def compare_legacy_geometry(scene, forecast):
    """Use the measured path for both calculations to isolate the hazard forecast."""
    frames = forecast["frames"]
    path = np.asarray([forecast["entry"]["position"], *(row["position"] for row in frames)], dtype=np.float32)
    legacy_margins = scene.path_margins(path[None])[0]
    contact = next((row["frame"] for row in frames if row["contact"]), None)
    return {
        "legacy_minimum_clearance": {
            str(horizon): float(legacy_margins[:horizon].min().clip(-256., 256.))
            for horizon in HORIZONS if len(legacy_margins) >= horizon
        },
        "legacy_clearance_at_contact": (float(legacy_margins[contact - 1].clip(-256., 256.))
                                         if contact is not None else None),
    }


def pool_projection(projectiles):
    fields = {
        "bullets": ("slot", "position", "velocity", "hitbox_size", "state", "illusion", "ex_flags", "flags"),
        "lasers": ("slot", "origin", "angle", "start_offset", "end_offset", "width", "state", "elapsed", "flags"),
    }
    return {kind: {name: projectiles[kind][name].tolist() for name in names} for kind, names in fields.items()}


def checkpoint_projection(source):
    ecl = source["ecl"]
    rng = ecl["rng"]
    state = source["state"]
    motion = source["motion"]
    return {
        "stage": state["stage"], "stage_frame": state["stage_timer"],
        "position": list(state["player_position"]), "time_scale": motion.time_scale,
        "movement_bounds": [list(motion.minimum), list(motion.maximum)],
        "rng": [rng.seed, rng.saved_seed, rng.draws],
        "program_bytes": len(ecl["program"].data), "subroutines": len(ecl["program"].subroutines),
        "timelines": [dict(timer=list(row["timer"]), pc=row["instruction"]["offset"]) for row in ecl["timelines"]],
        "emitters": [
            {"slot": slot, "subroutine": actor["main"]["subroutine"], "timer": list(actor["main"]["timer"]),
             "pc": actor["main"]["instruction"]["offset"] if actor["main"]["instruction"] else None,
             "stack_depth": len(actor["stack"]), "child_contexts": sum(x is not None for x in actor["children"]),
             "collision_body": bool(body), "flags": int(flags), "life": int(life), "position": position.tolist()}
            for (slot, actor), body, flags, life, position in zip(
                ecl["emitters"].items(), source["enemies"]["main_body_contact"], source["enemies"]["flags"],
                source["enemies"]["life"], source["enemies"]["position"], strict=True)
        ],
        "pools": pool_projection(source["projectiles"]),
    }


def _horizon_results(frames):
    results = {}
    for horizon in HORIZONS:
        sample = frames[:horizon]
        gaps = [x["clearance"] for x in sample if x["clearance"] is not None]
        hit = any(x["contact"] for x in sample)
        known = hit or len(sample) == horizon
        results[str(horizon)] = {
            "known": known,
            "verified_frames": len(sample),
            "without_contact": not hit if known else None,
            "minimum_clearance": min(gaps, default=None),
        }
    return results


def query_live_native_routes(controller, trajectories, **measurements):
    """Measure all supplied action tapes and leave the original player at the root."""
    started = time.perf_counter()
    try:
        root, traces = controller.trace_branch_routes(trajectories, **measurements)
    except BaseException:
        controller.close()
        raise
    stats = {
        "revision": FORECAST_REVISION, "input_poll": root["sequence"], "candidates": len(traces),
        "simulated_frames": sum(map(len, traces)), "elapsed_ms": (time.perf_counter() - started) * 1000.,
    }
    for key in ("spell_damage", "spell_health", "spell_completions", "power_gain"):
        if key in root:
            stats[key] = root[key]
    return traces, stats


def capture_native_branch(controller, masks):
    """Execute a supplied tape at the paused checkpoint and retain native witnesses."""
    root = checkpoint_projection(controller.read_forecast_source())
    frames = []
    previous = {kind: set(pool["slot"]) for kind, pool in root["pools"].items()}
    births = Counter()
    tested = Counter()
    rng_draws = 0
    matching_rng_draws = 0
    stop = "tape-complete"
    for frame, mask in enumerate(masks, start=1):
        state = controller.advance(mask, 1, capture_geometry=True)
        hitboxes = controller.tested_geometry()
        projectiles = pool_projection(controller.read_projectiles())
        new_slots = {}
        for kind, pool in projectiles.items():
            occupied = set(pool["slot"])
            new_slots[kind] = sorted(occupied - previous[kind])
            births[kind] += len(new_slots[kind])
            previous[kind] = occupied
        tested.update(box["kind"] for box in hitboxes)
        # During the original tick the hurtbox can still use a latched player pose.
        gaps = [float(checked_clearance(box["player"], (box,))) for box in hitboxes]
        geometry_valid = not any(math.isnan(value) for value in gaps)
        gap = min(gaps, default=math.inf) if geometry_valid else math.nan
        same_stage = state["mode"] == 2 and state["stage"] == root["stage"]
        rng, draws = controller.forecast_rng()
        rng_draws += len(draws)
        matching_rng_draws += sum(TH08RngState(before, 0, 0).after_draws(1).seed == after
                                  for before, after in draws)
        frames.append({
            "frame": frame, "mask": int(mask), "stage_frame": state["stage_timer"],
            "position": list(state["player_position"]), "contact": state["native_contact"],
            "rng": [rng.seed, rng.saved_seed, rng.draws],
            "clearance": gap if math.isfinite(gap) else None,
            "geometry_valid": geometry_valid,
            "new_slots": new_slots, "pools": projectiles, "tested_hitboxes": hitboxes,
        })
        if state["native_contact"] or not same_stage or state["player_state"] != 0:
            stop = "contact" if state["native_contact"] else "left-playable-stage"
            break
    summary = {
        "frames": len(frames), "stop": stop, "horizons": _horizon_results(frames),
        "new_occupied_slots": dict(births), "collision_tests": dict(tested),
        "rng_draws": [matching_rng_draws, rng_draws],
        "invalid_geometry_frames": sum(not row["geometry_valid"] for row in frames),
        "first_contact_frame": next((row["frame"] for row in frames if row["contact"]), None),
    }
    samples = sorted({horizon for horizon in HORIZONS if horizon <= len(frames)} | {len(frames)})
    safe_regions = {
        str(horizon): sampled_safe_region(frames[horizon - 1]["tested_hitboxes"], root["movement_bounds"])
        for horizon in samples if horizon > 0
    }
    return {"entry": root, "summary": summary, "frames": frames, "safe_regions": safe_regions}
