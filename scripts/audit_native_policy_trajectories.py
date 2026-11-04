"""Read-only CPU diagnostics of sealed original-game policy trajectories.

This script imports no game controller or torch. It never controls a game,
replays a brain, changes a checkpoint, or treats PPO training as held-out
evaluation. A live event log is captured as a verified complete-line prefix;
only already-sealed rollout files referenced by that prefix are admitted.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np


SCHEMA = "original-game-policy-trajectory-audit-v1"
FULL_N, FULL_E = 138639, 15091983
FORMAL_EXE_SHA = "f08e885e28d2e0fc8c579e88034dd7de8f2a648d8f08ce781cf08eadc9ffe353"
DIRECTIONS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1),
              (-1, -1), (1, -1), (-1, 1), (1, 1))
DEFAULT_EVALS = ("wholebrain-eval-resource-epoch12-v1",
                 "wholebrain-eval-learned-dagger-v1",
                 "wholebrain-eval-ppo-update16-v1")


def encoded(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def stable_bytes(path):
    path = Path(path)
    before = path.stat()
    data = path.read_bytes()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"Input changed while reading: {path}")
    return data


def file_evidence(path, expected_sha=None):
    path = Path(path).resolve()
    data = stable_bytes(path)
    digest = sha(data)
    if expected_sha is not None and digest != expected_sha:
        raise ValueError(f"Sealed input hash mismatch: {path}")
    return {"path": str(path), "sha256": digest, "bytes": len(data)}


def verify_events(data):
    if data and not data.endswith(b"\n"):
        raise ValueError("Event snapshot includes a partial line")
    previous, run_id, events = "0" * 64, None, []
    for i, line in enumerate(data.splitlines()):
        event = json.loads(line)
        recorded = event.pop("event_sha256")
        if (event["schema"] != "fly-touhou-experiment-event-v1"
                or event["sequence"] != i or event["previous_sha256"] != previous
                or sha(encoded(event)) != recorded
                or (run_id is not None and event["run_id"] != run_id)):
            raise ValueError(f"Event hash chain failed at event {i}")
        run_id = event["run_id"]
        event["event_sha256"] = recorded
        events.append(event)
        previous = recorded
    return events


def verified_manifest(run):
    data = stable_bytes(run / "manifest.json")
    manifest = json.loads(data)
    if sha(encoded(manifest["config"])) != manifest["config_sha256"]:
        raise ValueError("Manifest configuration hash mismatch")
    archives = []
    for name, item in manifest["files"].items():
        if item.get("archive_path"):
            archive = run / item["archive_path"]
            archives.append({"name": name, **file_evidence(archive, item["sha256"])})
    return manifest, {"manifest": file_evidence(run / "manifest.json"),
                      "archived_sources_verified": archives}


def graph_evidence(events):
    proofs = [x["payload"] for x in events if x["event"] == "whole_brain_verified"]
    if not proofs:
        raise ValueError("No recorded whole-brain verification")
    p = proofs[-1]
    if (p.get("neurons") != FULL_N or p.get("directed_edges") != FULL_E
            or p.get("neuron_coverage") != 1.0 or p.get("connection_record_coverage") != 1.0
            or p.get("recurrent_weights_frozen") is not True):
        raise ValueError("Recorded actor is not the required complete frozen graph")
    return p


def rows(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def validate_transitions(transitions, substeps):
    previous_counter, previous_key = None, None
    for t in transitions:
        if t["action_source"] != "whole-brain-descending-spikes" or t.get("teacher_action_calls",0) or t.get("safety_takeovers",0):
            raise ValueError("Trajectory contains another action source or a takeover")
        for s in (t["snapshot"],t["next_snapshot"]):
            extra=s.get("extra",{})
            if (s.get("source") != "native-process" or s.get("consistent") is not True
                    or extra.get("exe_sha256") != FORMAL_EXE_SHA
                    or extra.get("backend_kind") != "original-th14-frida-ipc"
                    or s.get("difficulty") != 3):
                raise ValueError("Snapshot does not establish the pinned formal original Lunatic source")
        if "noise_counter_before" in t:
            before,after=t["noise_counter_before"],t["noise_counter_after"]
            if after-before != substeps or previous_counter is not None and before != previous_counter:
                raise ValueError("Explicit neural noise counters do not continue through the episode")
            if previous_key is not None and t["neural_rng_key"] != previous_key:
                raise ValueError("Neural noise key changed inside a complete episode")
            previous_key=t["neural_rng_key"]
        else:
            after=t["neural_noise_counter"]
            if previous_counter is not None and after-previous_counter != substeps:
                raise ValueError("Evaluation neural noise counter reset or skipped inside the episode")
        previous_counter=after


def decode(action):
    if not isinstance(action, int) or not 0 <= action < 36:
        raise ValueError(f"Unexpected campaign action {action}")
    base = action % 18
    dx, dy = DIRECTIONS[base % 9]
    return dx, dy, base < 9, action >= 18


def entropy(actions):
    if not actions:
        return None
    counts = np.asarray(list(Counter(actions).values()), dtype=float)
    probabilities = counts / counts.sum()
    return float(-(probabilities * np.log(probabilities)).sum())


def describe(values):
    xs = np.asarray([x for x in values if x is not None and math.isfinite(x)], dtype=float)
    if not len(xs):
        return {"n": 0}
    return {"n": len(xs), "mean": float(xs.mean()), "std": float(xs.std()),
            "min": float(xs.min()), "p10": float(np.quantile(xs, .1)),
            "median": float(np.median(xs)), "p90": float(np.quantile(xs, .9)),
            "max": float(xs.max())}


def corr(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    a, b = np.asarray(pairs, dtype=float).T
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-12 and b.std() > 1e-12 else None


def fraction(xs):
    xs = list(xs)
    return float(sum(xs) / len(xs)) if xs else None


def swept_circle(bullet, player, next_player, frames):
    """Linear-motion geometry proxy, explicitly NOT the native collision oracle."""
    rx, ry = bullet["x"] - player["x"], bullet["y"] - player["y"]
    pvx = (next_player["x"] - player["x"]) / max(1, frames)
    pvy = (next_player["y"] - player["y"]) / max(1, frames)
    vx, vy = bullet["vx"] - pvx, bullet["vy"] - pvy
    denominator = vx * vx + vy * vy
    t = max(0., min(float(frames), -(rx * vx + ry * vy) / denominator)) if denominator else 0.
    return math.hypot(rx + vx * t, ry + vy * t) - bullet["radius"] - player["radius"]


def bullet_proxy(bullets, p, q, frames):
    if not bullets:
        return None
    b = min(bullets, key=lambda x: swept_circle(x, p, q, frames))
    return {"id": b["id"], "age": b.get("age"), "active": b.get("active"),
            "invulnerable_frames": b.get("invulnerable_frames"),
            "radius": b["radius"], "x": b["x"], "y": b["y"],
            "vx": b["vx"], "vy": b["vy"],
            "current_clearance": math.hypot(b["x"]-p["x"], b["y"]-p["y"])-b["radius"]-p["radius"],
            "swept_clearance_proxy": swept_circle(b, p, q, frames)}


def movement_summary(features):
    """Keep native respawn states out of voluntary movement interpretation."""
    return {"decisions": len(features), "x": describe(f["x"] for f in features),
            "y": describe(f["y"] for f in features),
            "near_sidewall24_fraction": fraction(f["near_sidewall_24"] for f in features),
            "outward_wall_command_fraction": fraction(f["outward_wall_command"] for f in features),
            "above_poc_fraction": fraction(f["above_poc"] for f in features),
            "displacement_per_decision": describe(f["displacement"] for f in features
                if f["player_state"] == f["next_player_state"] == 1)}


def alternative_geometry(t):
    """18 four-frame linear alternatives, for diagnosis only, never an actor."""
    s=t["snapshot"]; p=s["player"]; frames=t["elapsed_frames"]
    bullets=[b for b in s["bullets"] if b["active"] and not b["intangible"]]
    if not bullets:
        return None
    clearances=[]
    for a in range(18):
        dx,dy,focus,_=decode(a)
        speed=2.0 if focus else 4.5
        scale=speed/max(1.,math.hypot(dx,dy))*frames
        q={"x":max(-184.,min(184.,p["x"]+scale*dx)),
           "y":max(32.,min(432.,p["y"]+scale*dy)), "radius":p["radius"]}
        clearances.append(min(swept_circle(b,p,q,frames) for b in bullets))
    return {"clearances_by_base_action":clearances,
            "positive_clearance_base_actions":[a for a,c in enumerate(clearances) if c>0],
            "best_proxy_base_action":int(np.argmax(clearances)),
            "model":"constant current bullet velocities, constant player speed2/4.5, clamped x+-184/y32..432, current old-active circles only, no safety margin; NOT native collision labels or a guarantee of survival"}


def transition_features(t, index, run, episode):
    s, n = t["snapshot"], t["next_snapshot"]
    p, q = s.get("player"), n.get("player")
    if p is None or q is None:
        raise ValueError("This diagnostic currently requires measured before/after player positions")
    a = t["action"]
    dx, dy, focus, bomb = decode(a)
    frames = t["elapsed_frames"]
    res = s.get("extra", {}).get("resources", {})
    active = sorted((b for b in s["bullets"] if b["active"] and not b["intangible"]),
                    key=lambda b: ((b["x"]-p["x"])**2+(b["y"]-p["y"])**2, b["id"]))
    inactive = [b for b in s["bullets"] if not b["active"] and not b["intangible"]]
    inactive_old = [b for b in inactive if b.get("age", 0) >= 8]
    active_nearest = bullet_proxy(active, p, q, frames)
    if active_nearest:
        active_nearest["current_distance_rank"] = next(i+1 for i,b in enumerate(active) if b["id"] == active_nearest["id"])
    ignored_nearest = bullet_proxy(inactive_old, p, q, frames)
    omitted = bullet_proxy(active[24:], p, q, frames)
    enemies = [e for e in res.get("enemies", []) if e.get("contactable") and not e.get("intangible")]
    nearest_enemy = min(enemies, key=lambda e: math.hypot(e["x"]-p["x"], e["y"]-p["y"]), default=None)
    damageable = [e for e in res.get("enemies", []) if e.get("damageable") and e["hp"] > 0 and e["y"] < p["y"]]
    aligned = any(abs(e["x"]-p["x"]) <= 24 for e in damageable)
    next_enemies = {e["id"]: e for e in n.get("extra", {}).get("resources", {}).get("enemies", [])}
    measured_hp_drop = sum(max(0, e["hp"]-next_enemies[e["id"]]["hp"])
                           for e in res.get("enemies", []) if e["id"] in next_enemies)
    life_drop = max(0, s["lives"]-n["lives"])
    bomb_drop = max(0, s["bombs"]-n["bombs"])
    x, y = p["x"], p["y"]
    f = {"run": run, "episode": episode, "index": index,
         "episode_step": t.get("episode_step", t.get("step", index)),
         "frame": s["frame"], "stage_frame": s["stage_frame"], "stage": s["stage"],
         "action": a, "dx": dx, "dy": dy, "focus_requested": focus, "bomb_requested": bomb,
         "x": x, "y": y, "next_x": q["x"], "next_y": q["y"],
         "displacement": math.hypot(q["x"]-x, q["y"]-y),
         "player_state": p["state"], "next_player_state": q["state"],
         "invulnerable": p["invulnerable_frames"], "deathbomb_timer": p["deathbomb_timer"],
         "death_entry": p["state"] != 4 and q["state"] == 4,
         "life_drop": life_drop, "lives": s["lives"], "bombs": s["bombs"],
         "bomb_inventory_drop": bomb_drop, "bomb_drop_without_life_drop": bomb_drop if not life_drop else 0,
         "power": s["power"], "power_gain_without_life_drop": max(0, n["power"]-s["power"]) if not life_drop else 0,
         "score_gain": n["score"]-s["score"], "frames": frames,
         "eligible_to_shoot": res.get("player_can_shoot"),
         "player_flags": res.get("player_flags"),
         "observed_shot_bit": bool(s.get("extra", {}).get("input", 0) & 1),
         "enemies_above": len(damageable), "aligned_with_enemy_above_proxy": aligned,
         "native_measured_hp_drop_same_id": measured_hp_drop,
         "enemy_count": len(res.get("enemies", [])), "boss_present": any(e.get("boss") for e in res.get("enemies", [])),
         "items": len(res.get("items", [])), "poc_line_y": res.get("poc_line_y"),
         "above_poc": y <= res.get("poc_line_y", -math.inf),
         "near_sidewall_24": abs(x) >= 160,
         "outward_wall_command": p["state"] == 1 and ((x <= -180 and dx < 0) or (x >= 180 and dx > 0)),
         "active_bullets": len(active), "inactive_bullets": len(inactive),
         "inactive_age_ge8": len(inactive_old), "active_overflow24": len(active) > 24,
         "active_min_swept_proxy": active_nearest["swept_clearance_proxy"] if active_nearest else None,
         "inactive_age_ge8_min_swept_proxy": ignored_nearest["swept_clearance_proxy"] if ignored_nearest else None,
         "omitted_active_min_swept_proxy": omitted["swept_clearance_proxy"] if omitted else None,
         "lasers": len(s["lasers"]), "laser_nonzero_speed": sum(abs(l.get("speed", 0)) > 1e-9 for l in s["lasers"]),
         "active_velocity_component_gt16": sum(abs(b["vx"]) > 16 or abs(b["vy"]) > 16 for b in active),
         "surprise": -t["old_log_probability"] if "old_log_probability" in t else None,
         "neural_noise_counter": t.get("neural_noise_counter", t.get("noise_counter_after")),
         "neural_rng_key": t.get("neural_rng_key"),
         "reward": t["reward"], "terminated": t["terminated"], "truncated": t["truncated"],
         "native_gameover": n.get("extra", {}).get("native_gameover"),
         "nearest_enemy": nearest_enemy,
         "active_nearest_proxy": active_nearest, "inactive_nearest_proxy": ignored_nearest,
         "omitted_active_nearest_proxy": omitted,
         "neural": t.get("neural", {}),
         }
    return f


def analyze_episode(transitions, run, episode, config, result):
    validate_transitions(transitions,config["model_config"]["substeps"])
    features = [transition_features(t, i, run, episode) for i, t in enumerate(transitions)]
    if not features:
        raise ValueError("Empty episode")
    actions = [x["action"] for x in features]
    run_lengths, length = [], 1
    for a, b in zip(actions, actions[1:]):
        if a == b:
            length += 1
        else:
            run_lengths.append(length)
            length = 1
    run_lengths.append(length)
    deaths = []
    for i, f in enumerate(features):
        if not f["death_entry"]:
            continue
        prior = features[max(0, i-15):i+1]
        drops = [x for x in features[i:min(len(features), i+30)] if x["life_drop"]]
        deaths.append({"run": run, "episode": episode, "index": i,
                       "stage_frame": f["stage_frame"], "action": f["action"],
                       "x": f["x"], "y": f["y"], "bombs_before_hit": f["bombs"],
                       "invulnerable_before": f["invulnerable"], "deathbomb_timer_before": f["deathbomb_timer"],
                       "bomb_commands_last60frames": sum(x["bomb_requested"] for x in prior),
                       "later_life_drop_step": drops[0]["episode_step"] if drops else None,
                       "active_bullets": f["active_bullets"], "inactive_age_ge8": f["inactive_age_ge8"],
                       "active_nearest_proxy": f["active_nearest_proxy"],
                       "inactive_nearest_proxy": f["inactive_nearest_proxy"],
                       "omitted_active_nearest_proxy": f["omitted_active_nearest_proxy"],
                       "nearest_enemy": f["nearest_enemy"], "neural": f["neural"],
                       "base_action_geometry_alternatives": alternative_geometry(transitions[i]),
                       "deathbomb_window_observed_decisions": [{k:x[k] for k in ("episode_step", "action", "bombs", "player_state", "deathbomb_timer", "life_drop")} for x in features[i+1:min(len(features), i+5)] if x["player_state"] == 4],
                       "context_last60frames": [{k:x[k] for k in ("episode_step", "stage_frame", "action", "x", "y", "player_state", "invulnerable", "bombs", "deathbomb_timer", "active_bullets", "inactive_age_ge8", "active_min_swept_proxy", "inactive_age_ge8_min_swept_proxy")} for x in prior]})
    promotion_candidates = []
    for t in transitions:
        after = {b["id"]: b for b in t["next_snapshot"]["bullets"]}
        p = t["snapshot"]["player"]
        for b in t["snapshot"]["bullets"]:
            c = after.get(b["id"])
            if (not b["active"] and c and c["active"] and c["age"] >= b["age"]
                    and c["age"]-b["age"] <= t["elapsed_frames"]+1):
                promotion_candidates.append({"id": b["id"], "stage_frame": t["snapshot"]["stage_frame"],
                    "before_age": b["age"], "after_age": c["age"], "before_radius": b["radius"],
                    "distance_to_player_before": math.hypot(b["x"]-p["x"], b["y"]-p["y"])})
    neural_keys = ("spikes_in_window", "descending_spikes_in_window", "descending_responding_fraction",
                   "descending_filtered_mean_hz", "descending_filtered_max_hz")
    neural = {}
    for key in neural_keys:
        vals = [f["neural"].get(key) for f in features]
        if any(x is not None for x in vals):
            neural[key] = {**describe(vals), "pearson_with_x": corr(vals, [f["x"] for f in features]),
                          "pearson_with_y": corr(vals, [f["y"] for f in features]),
                          "pearson_with_active_bullet_count": corr(vals, [f["active_bullets"] for f in features]),
                          "mean_by_action": {str(a): describe([f["neural"].get(key) for f in features if f["action"] == a]).get("mean") for a in sorted(set(actions))}}
    noise = [f["neural_noise_counter"] for f in features]
    nonconstant_noise_deltas = Counter(b-a for a,b in zip(noise, noise[1:]) if a is not None and b is not None)
    actionable = [f for f in features if f["player_state"] == 1 and f["invulnerable"] == 0]
    bomb_requests = [f for f in features if f["bomb_requested"]]
    controllable=[f for f in features if f["player_state"] == 1]
    summary = {"run": run, "episode": episode,
        "evidence_role": "independent-held-out-native-evaluation" if run != "wholebrain-ppo-resource-v1" else "training-episode-not-independent-evidence",
        "deterministic_actions": config.get("deterministic_actions", False),
        "game_rng_seed_applied": config.get("game_rng_seed_applied"),
        "result": {k:result.get(k) for k in ("steps", "native_frames", "score", "life_decrements", "visited_stages", "completed_stages", "native_gameover", "terminated", "truncated", "teacher_action_calls", "safety_takeovers")},
        "trace_decisions": len(features), "trace_frames": sum(f["frames"] for f in features),
        "actions": {"frequency": dict(sorted(Counter(actions).items())), "realized_frequency_entropy_nats": entropy(actions),
            "frequency_entropy_meaning": "entropy of realized action histogram, NOT conditional policy entropy",
            "effective_frequency_actions": math.exp(entropy(actions)),
            "top_action": Counter(actions).most_common(1)[0],
            "switch_fraction": fraction(a != b for a,b in zip(actions, actions[1:])),
            "consecutive_same_action_run_length": describe(run_lengths),
            "focus_fraction": fraction(f["focus_requested"] for f in features),
            "stationary_request_fraction": fraction(f["dx"] == 0 and f["dy"] == 0 for f in features),
            "sampled_surprise_nats": describe(f["surprise"] for f in features),
            "surprise_meaning": "mean -log p(selected action) estimates conditional entropy only for genuinely sampled actions; correlated finite trace, not exact entropy"},
        "position": movement_summary(features),
        "position_controllable_state1": movement_summary(controllable),
        "position_vulnerable_state1_invulnerability0": movement_summary(actionable),
        "native_player_state_frequency":dict(Counter(f["player_state"] for f in features)),
        "shooting": {"shot_held_by_action_decoder_fraction": 1.0,
            "observed_snapshot_shot_bit_fraction": fraction(f["observed_shot_bit"] for f in features),
            "native_eligible_to_shoot_fraction": fraction(f["eligible_to_shoot"] for f in features),
            "native_eligible_to_shoot_fraction_controllable_state1": fraction(f["eligible_to_shoot"] for f in controllable),
            "not_shoot_eligible_by_state_and_flags":dict(Counter(f"state{f['player_state']}:flags{f['player_flags']}" for f in features if not f["eligible_to_shoot"])),
            "alignment_fraction_when_damageable_enemy_above": fraction(f["aligned_with_enemy_above_proxy"] for f in features if f["enemies_above"]),
            "alignment_fraction_when_eligible_and_enemy_above": fraction(f["aligned_with_enemy_above_proxy"] for f in controllable if f["enemies_above"] and f["eligible_to_shoot"]),
            "same_id_measured_enemy_hp_drop_total": sum(f["native_measured_hp_drop_same_id"] for f in features),
            "hp_drop_caveat": "includes all native damage sources; removals are not assigned HP=0; alignment is only a geometry proxy",
            "power_gain_without_life_drop_total": sum(f["power_gain_without_life_drop"] for f in features)},
        "bombs": {"requests": len(bomb_requests), "requests_with_zero_inventory": sum(f["bombs"] <= 0 for f in bomb_requests),
            "requests_during_invulnerability": sum(f["invulnerable"] > 0 for f in bomb_requests),
            "requests_during_state4": sum(f["player_state"] == 4 for f in bomb_requests),
            "inventory_decrements_without_same_step_life_loss": sum(f["bomb_drop_without_life_drop"] for f in features),
            "deaths_with_unused_bombs": sum(d["bombs_before_hit"] > 0 for d in deaths),
            "hit_entries_with_positive_bomb_inventory": sum(d["bombs_before_hit"] > 0 for d in deaths),
            "legacy_hit_counter_is_not_actual_life_loss": True,
            "request_details":[{k:f[k] for k in ("episode_step", "stage_frame", "action", "x", "y", "player_state", "invulnerable", "bombs", "bomb_inventory_drop", "life_drop", "eligible_to_shoot")} for f in bomb_requests],
            "meaning": "requested Bomb is a one-tick press then release; a request does not prove native use; inventory changes can combine pickups and consumption"},
        "coverage": {"active_bullets": describe(f["active_bullets"] for f in features),
            "inactive_bullets": describe(f["inactive_bullets"] for f in features),
            "inactive_age_ge8": describe(f["inactive_age_ge8"] for f in features),
            "active_overflow24_fraction": fraction(f["active_overflow24"] for f in features),
            "actionable_inactive_age_ge8_fraction": fraction(f["inactive_age_ge8"] > 0 for f in actionable),
            "omitted_active_proxy_crossing_fraction": fraction(f["omitted_active_min_swept_proxy"] is not None and f["omitted_active_min_swept_proxy"] <= 0 for f in actionable),
            "active_velocity_component_gt16_records": sum(f["active_velocity_component_gt16"] for f in features),
            "laser_records": sum(f["lasers"] for f in features),
            "laser_nonzero_speed_records": sum(f["laser_nonzero_speed"] for f in features),
            "inactive_to_active_same_id_candidates": len(promotion_candidates),
            "promotion_candidates": promotion_candidates,
            "inactive_caveat": "inactive and age>=8 is only a candidate pool: raw Bullet state enum/collision flags were not recorded; clearing and other states cannot all be relabeled as dangerous"},
        "death_entry_count": len(deaths), "life_decrement_total_in_trace": sum(f["life_drop"] for f in features),
        "neural_scalar_activity": neural, "noise_counter_successive_deltas": dict(nonconstant_noise_deltas),
        "neural_readout_limit": "only scalar activity diagnostics were saved for independent evaluations, not per-DN rates, logits or action probabilities; action changes cannot be uniquely attributed to individual neurons",
        }
    if result.get("steps") is not None and result["steps"] != len(features):
        raise ValueError("Sealed result and trace length disagree")
    return summary, features, deaths


def independent_run(path):
    manifest, proof = verified_manifest(path)
    event_data = stable_bytes(path/"events.jsonl")
    events = verify_events(event_data)
    graph = graph_evidence(events)
    proof["events"] = file_evidence(path/"events.jsonl")
    proof["graph"] = graph
    summary = json.loads(stable_bytes(path/"evaluation_summary.json"))
    proof["evaluation_summary"] = file_evidence(path/"evaluation_summary.json")
    proof["trajectories"] = []
    episodes, all_features, all_deaths = [], [], []
    for i, campaign in enumerate(summary["campaigns"]):
        seal = campaign["acceptance_gate"]["evidence"]["trajectory"]
        trajectory = Path(seal["path"])
        proof["trajectories"].append(file_evidence(trajectory, seal["sha256"]))
        data = rows(trajectory)
        transitions = [x for x in data if x["record_type"] == "transition"]
        metadata = next(x for x in data if x["record_type"] == "metadata")
        # ExperimentLogger adds its acceptance criterion to the manifest's
        # copy. The writer deliberately records the original runtime config.
        runtime_config = {k:v for k,v in manifest["config"].items() if k != "final_acceptance_criterion"}
        if metadata["config"] != runtime_config:
            raise ValueError("Trajectory and manifest config disagree")
        proof["runtime_config_comparison"] = "equal after excluding logger-injected final_acceptance_criterion from manifest"
        result = campaign["result"]
        e, fs, ds = analyze_episode(transitions, path.name, str(i), manifest["config"], result)
        e["acceptance_gate_passed"] = campaign["acceptance_gate"]["reported_measurements_pass_gate"]
        e["checkpoint_sha256"] = manifest["config"]["checkpoint_sha256"]
        episodes.append(e); all_features.extend(fs); all_deaths.extend(ds)
    return episodes, all_features, all_deaths, proof


def training_run(path, out):
    manifest, proof = verified_manifest(path)
    # The producer can append during this read. Retain and verify only the
    # complete prefix actually observed; never claim to seal the live logfile.
    raw = (path/"events.jsonl").read_bytes()
    cutoff = raw.rfind(b"\n")+1
    prefix = raw[:cutoff]
    events = verify_events(prefix)
    captured = out/"training_events_snapshot.jsonl"
    captured.write_bytes(prefix)
    proof["active_log_prefix_snapshot"] = {"source_path": str(path/"events.jsonl"),
        "captured_bytes": cutoff, "ignored_partial_tail_bytes": len(raw)-cutoff,
        "last_sequence": events[-1]["sequence"], "last_utc": events[-1]["utc"],
        **file_evidence(captured)}
    proof["graph"] = graph_evidence(events)
    finished = {e["payload"]["episode_id"]:e["payload"] for e in events if e["event"] == "ppo_training_episode_finished"}
    by_episode = defaultdict(list)
    proof["sealed_native_rollouts"] = []
    for event in events:
        if event["event"] != "ppo_behavior_rollout_sealed":
            continue
        seal = event["payload"]["native_trajectory"]
        fp = Path(seal["path"])
        proof["sealed_native_rollouts"].append({"event_sequence":event["sequence"], **file_evidence(fp, seal["sha256"])})
        for t in rows(fp):
            if t["record_type"] == "transition":
                by_episode[t["episode_id"]].append(t)
    episodes, features, deaths = [], [], []
    excluded = []
    for episode_id, transitions in by_episode.items():
        transitions.sort(key=lambda t:t["episode_step"])
        if episode_id not in finished or not transitions[-1]["terminated"] and not transitions[-1]["truncated"]:
            excluded.append({"episode_id":episode_id, "sealed_decisions":len(transitions), "reason":"no completed episode and terminal sealed trace at captured cutoff"})
            continue
        if [t["episode_step"] for t in transitions] != list(range(len(transitions))):
            raise ValueError("Missing or duplicate native transitions in completed training episode")
        r = finished[episode_id]
        e, fs, ds = analyze_episode(transitions, path.name, episode_id, manifest["config"], r)
        e["acceptance_gate_passed"] = None
        episodes.append(e); features.extend(fs); deaths.extend(ds)
    updates = [{"sequence":e["sequence"], **e["payload"]} for e in events if e["event"] == "epoch"]
    proof["excluded_incomplete_episodes"] = excluded
    return episodes, features, deaths, proof, updates


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def write_csv(path, data):
    if not data:
        return
    keys = [k for k in data[0] if not isinstance(data[0][k], (dict,list))]
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, keys, extrasaction="ignore")
        writer.writeheader(); writer.writerows(data)


def report(audit, deaths):
    lines = ["# 全脑学生原作轨迹诊断", "", "本报告只读既有、哈希校验后的原作轨迹；未启动游戏、GPU或神经重放。训练回合单列，不能用作独立验收。", "",
        "## 独立评测与训练回合", "", "|来源|选动作|决策/原生帧|分数|动作频数熵|侧墙24内占比|上方回收线以上|Bomb请求/库存下降*|中弹入口|未用Bomb中弹|", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for e in audit["episodes"]:
        r=e["result"]; b=e["bombs"]; p=e["position_controllable_state1"]
        label=e["run"].replace("wholebrain-eval-","").replace("wholebrain-ppo-resource-v1","训练")+":"+e["episode"].split(":")[-1]
        lines.append(f"|{label}|{'确定性' if e['deterministic_actions'] else '随机'}|{e['trace_decisions']}/{e['trace_frames']}|{r.get('score')}|{e['actions']['realized_frequency_entropy_nats']:.3f}|{p['near_sidewall24_fraction']:.1%}|{p['above_poc_fraction']:.1%}|{b['requests']}/{b['inventory_decrements_without_same_step_life_loss']}|{e['death_entry_count']}|{b['deaths_with_unused_bombs']}|")
    lines += ["", "位置比例只统计原生player.state=1；state0/2/4的入场、受击和复活运动不当成策略移动。*库存下降只统计同一步无掉残机时；请求不等于原生成功使用，也可能与拾取同时发生。所有独立评测均停在一面 GameOver，没有通过六面验收。", ""]
    ppo=[e for e in audit["episodes"] if e["run"] == "wholebrain-eval-ppo-update16-v1"]
    if ppo:
        e=ppo[0]; a=e["actions"]; b=e["bombs"]; n=e["neural_scalar_activity"].get("descending_filtered_mean_hz",{})
        ds=[d for d in deaths if d["run"]==e["run"] and d["episode"]==e["episode"]]
        death_info="；".join(f"stage_frame{d['stage_frame']} / action{d['action']} / active扫掠间隙{d['active_nearest_proxy']['swept_clearance_proxy']:.2f} / 当前距离第{d['active_nearest_proxy']['current_distance_rank']}弹" for d in ds if d["active_nearest_proxy"])
        bomb_window=sum(x["action"]>=18 for d in ds for x in d["deathbomb_window_observed_decisions"])
        lines += ["## 当前独立PPO评测的直接发现", "",
            f"- 原生{e['trace_frames']}帧/{e['trace_decisions']}决策后GameOver，得分{e['result']['score']}；{b['deaths_with_unused_bombs']}/{len(ds)}次受击带Bomb。动作切换率{a['switch_fraction']:.1%}、停住请求{a['stationary_request_fraction']:.1%}、聚焦{a['focus_fraction']:.1%}，频数熵{a['realized_frequency_entropy_nats']:.3f}。这些量显示控制更分散，但随机策略本来就会提高切换，不能等同更会躲弹。",
            f"- {b['requests']}次Bomb请求中{b['requests_during_invulnerability']}次在已有无敌时间内，观察到{b['inventory_decrements_without_same_step_life_loss']}次无同帧掉残机的库存下降；各受击前60帧Bomb请求为{[d['bomb_commands_last60frames'] for d in ds]}，可见deathbomb窗口共{bomb_window}次Bomb请求。至少此局没有在临界受击点用出预防或救命Bomb。",
            f"- state4入口的可见弹候选：{death_info}。本局中弹都有当前24弹槽内的相交代理，不支持把这些受击直接归因于inactive过滤或槽溢出。具体原生致死弹未被记录，候选仍不是碰撞oracle。",
            f"- Shot动作始终保持，原生可射击比例{e['shooting']['native_eligible_to_shoot_fraction']:.1%}；不许射击的状态/flags分解已保存，不能把原生死亡/复活或Bomb阶段的禁射误读为模型选择停火。可射击且上方有可伤害敌机时，对齐代理比例{e['shooting']['alignment_fraction_when_eligible_and_enemy_above']:.1%}，需要改善目标跟随。",
            f"- DN滤波均值跨步{n.get('min',float('nan')):.2f}–{n.get('max',float('nan')):.2f}Hz，均值{n.get('mean',float('nan')):.2f}Hz、标准差{n.get('std',float('nan')):.2f}，没有活动归零；但只有活动标量，无法直接评价36动作读出概率如何随几何变化。",
            "- 全部分析轨迹没有激光，active弹速度分量也没有超过编码除数16；激光速度缺失和速度裁剪是后续覆盖问题，不是本批早死的已证原因。", ""]
    lines += [
        "## 解释边界", "", "- PPO16随机动作与main12/readout13确定性动作、未控制原作RNG的差别，阻止直接作因果性能排名；只有一次PPO独立回合。", "- 表中熵是实际动作频数的熵；独立评测未保存逐步logits/概率，无法恢复真实条件策略熵。训练的 `-old_log_probability` 是采样熵估计，更新日志的entropy是优化重放口径。", "- 中弹入口按player.state首次进入4识别，掉残机延迟单列，避免把同一次受击计算两遍。前60帧上下文和几何候选见 deaths.json；线性扫掠是诊断代理，不能替代原生碰撞判定。", "- 神经日志仅有全脑/DN活动标量，未保存逐个下降神经元率向量；标量相关不证明读出理解几何。", "",
        "## 读入与动作边界", "", "1. **旧active过滤不是原生碰撞oracle。** native_target封口静态核对显示：state2成长弹timer>=8已进入原生玩家碰撞路径，而转state1还等待ANM条件；旧reader以state==1置active，464编码器及教师再过滤inactive，可能完全漏掉已经危险的弹。旧trace保留age=原生timer，但缺raw state/完整碰撞flags，因此inactive且age>=8只是候选池，不能把全部候选重标为危险弹。state2运动还只有0.5倍stored velocity；本报告inactive扫掠使用原记录速度，故更不能作为精确危险标签。弹体+0x24的iframes保护弹体删除/转换，也不等于保护玩家。原始交叉引用和已封口补充证据的哈希见 provenance.json。", "2. 24个弹槽按当前距离选取，现有字段有当前位置/速度/半径/age，未含碰撞状态和flags；最近并不等于最先相交。应先修正经原生验证的危险语义，再比较最近距离与最短相遇时间选槽。", "3. Laser已有speed/age保存在Snapshot，但464仅编码最近线段端点/width/kind/node_count，且过滤inactive；缺速度/预警状态/倒计时，弯曲激光只保留最近一段。本批若没有激光，不能用它解释这些受击。", "4. 18基础方向/速度与Bomb边沿组成36动作；Shot始终按住，没有独立停火动作。每决策4帧，快速移动最多约18单位、聚焦约8单位；现有观测不显式给上一动作/玩家速度。神经每决策5ms、DN滤波20ms约4决策，需用可控对照查响应延迟；游戏在推理时停驻，不能把墙钟推理延迟算成失控前进。", "",
        "## 可执行的下一轮顺序", "", "1. 先封口并接入原生hazard语义的新版reader/schema/教师，重新采集；保留旧版本作对照，旧模型不能静默套新字段。新增成长弹state2 timer7/8、碰撞flags、转state1和清除态的合同样例。", "2. 在新旧独立轨迹中保存逐步action logits/概率、所选动作logp、DN率向量或其可恢复摘要、原生Bomb事件；这样才能区分动作随机性、脑内噪声和读出状态变化。", "3. 统一随机/确定性各至少多局并列评测相同检查点，记录原作初始RNG；继续记录Shot资格、敌机对齐、墙外无效移动、带Bomb中弹和掉残机，不把训练回合并入验收。", "4. 读入修正后再单独对照4帧与2帧决策、20ms与较短读出滤波；维持完整图与无旁路，不能同时改教师、观测、动作频率后把差异归给某一项。", ""]
    return "\n".join(lines)


def self_test():
    assert decode(18) == (0, 0, True, True)
    assert decode(35) == (1, 1, False, True)
    assert entropy([0,0,0]) == 0
    assert abs(entropy([0,1])-math.log(2)) < 1e-12
    b={"x":-10.,"y":0.,"vx":20.,"vy":0.,"radius":1.}
    p={"x":0.,"y":0.,"radius":1.}
    assert swept_circle(b,p,p,1) == -2.0  # crosses between endpoints
    event={"schema":"fly-touhou-experiment-event-v1","sequence":0,
           "previous_sha256":"0"*64,"run_id":"fixture","event":"fixture","payload":{}}
    event["event_sha256"]=sha(encoded(event))
    assert len(verify_events(encoded(event)+b"\n")) == 1
    event["payload"]={"tampered":True}
    try:
        verify_events(encoded(event)+b"\n")
    except ValueError:
        pass
    else:
        raise AssertionError("Hash verifier accepted tampering")
    print("self-test: action mapping, continuous crossing and event integrity passed; no game/GPU imported")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base",type=Path,default=Path(r"./outputs/legacy-build\experiments"))
    parser.add_argument("--out",type=Path)
    parser.add_argument("--eval",action="append",dest="evaluations")
    parser.add_argument("--training-run",default="wholebrain-ppo-resource-v1")
    parser.add_argument("--native-evidence",action="append",type=Path,default=[])
    parser.add_argument("--native-seal",action="append",type=Path,default=[])
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    if args.self_test:
        self_test(); return
    if args.out is None:
        parser.error("--out is required unless --self-test")
    out=args.out.resolve()
    out.mkdir(parents=True,exist_ok=False)
    try:
        episodes,features,deaths,provenance=[],[],[],{}
        for name in args.evaluations or DEFAULT_EVALS:
            es,fs,ds,p=independent_run(args.base/name)
            episodes+=es; features+=fs; deaths+=ds; provenance[name]=p
        es,fs,ds,p,updates=training_run(args.base/args.training_run,out)
        episodes+=es; features+=fs; deaths+=ds; provenance[args.training_run]=p
        provenance["native_semantics_evidence"]=[file_evidence(x) for x in args.native_evidence]
        provenance["native_semantics_seals"]=[]
        for seal_path in args.native_seal:
            seal=json.loads(stable_bytes(seal_path))
            if seal["original_exe_sha256"] != FORMAL_EXE_SHA:
                raise ValueError("Native semantics evidence refers to another original executable")
            checked=[file_evidence(x["path"],x["sha256"]) for x in seal["files"]]
            provenance["native_semantics_seals"].append({"seal":file_evidence(seal_path),"verified_files":checked})
        provenance["native_semantics_status"]="Verified explicitly supplied native evidence seals" if args.native_seal else "Static cross-reference supplied; no native evidence seal was provided to this audit"
        source=Path(__file__).resolve()
        shutil.copyfile(source,out/source.name)
        provenance["audit_source"]=file_evidence(out/source.name)
        audit={"schema":SCHEMA,"created_utc":datetime.now(timezone.utc).isoformat(),
            "execution":{"cpu_only":True,"game_launched":False,"torch_imported":False,"neural_replay":False},
            "episodes":episodes,"training_update_metrics":updates,
            "evidence_roles":"Independent sealed evaluation vs completed training episodes retained separately; no success or superiority declaration",
            "semantic_limit":"Old Bullet.active state==1 is not a native collision oracle. State2 timer>=8 can already collide; raw enum/flags missing from old traces prevent exact reconstruction."}
        save_json(out/"diagnostic_summary.json",audit)
        save_json(out/"provenance.json",provenance)
        save_json(out/"deaths.json",deaths)
        write_csv(out/"decisions.csv",features)
        write_csv(out/"deaths.csv",deaths)
        (out/"report.md").write_text(report(audit,deaths),encoding="utf-8")
        files={x.name:file_evidence(x) for x in out.iterdir() if x.is_file()}
        save_json(out/"audit_seal.json",{"schema":SCHEMA,"created_utc":datetime.now(timezone.utc).isoformat(),"files":files})
        print(json.dumps({"output":str(out),"episodes":len(episodes),"decisions":len(features),"death_entries":len(deaths),"training_updates":len(updates)},ensure_ascii=False))
    except BaseException as exc:
        save_json(out/"failure.json",{"error":repr(exc),"preserved_partial_output":True})
        raise


if __name__ == "__main__":
    main()
