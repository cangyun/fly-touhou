"""CUDA BC-only closed-loop qualification: 20 argmax and 20 sampled games.

Invoking this entry point starts owned background original-game processes.
It never runs PPO or optimizes any parameter.  A failed/interrupted attempt is
retained, and its qualification remains false.
"""
from __future__ import annotations

import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.vec_env import SubprocVecEnv

from fly_touhou.engineering_contract import sha256_file
from fly_touhou.geometry_observation import GeometryObservationEncoder
from fly_touhou.gru_imitation import NATIVE_BOUNDS, state_subset_sha256
from fly_touhou.spatial_native_features import SpatialNativeFeatures
from fly_touhou.standard_gru_ppo import NativeAvoidanceEnv, NativeGRUPolicy, load_bc_actor


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2, 4), default=2)
    parser.add_argument("--seed", type=int, default=110011)
    parser.add_argument("--wall-cap-seconds", type=float, default=1800.)
    args = parser.parse_args()
    if args.wall_cap_seconds <= 0:
        parser.error("The qualification wall cap must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads(args.base_protocol.read_text(encoding="utf-8"))
    feature_names = list(GeometryObservationEncoder.from_specification(protocol["encoder"]).feature_names) + ["task_progress"]
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    metadata = checkpoint.get("metadata", {})
    hidden_size = metadata.get("hidden_size", 128)
    if hidden_size not in (128, 256):
        raise ValueError("Unknown BC recurrent architecture")
    names_hash = hashlib.sha256(json.dumps(feature_names).encode()).hexdigest()
    if (metadata.get("observation_size", 1912) != 1912
            or metadata.get("task_start_stage_frame") != 12
            or metadata.get("feature_names_sha256", names_hash) != names_hash
            or tuple(metadata.get("bounds", NATIVE_BOUNDS)) != NATIVE_BOUNDS):
        raise ValueError("BC checkpoint observation or movement semantics differ")
    if not torch.cuda.is_available():
        raise RuntimeError("BC qualification requires CUDA; no CPU fallback")
    torch.set_num_threads(2)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    policy = NativeGRUPolicy(gym.spaces.Box(-1., 1., shape=(1912,), dtype=np.float32),
        gym.spaces.Discrete(18), lambda _: 1e-4, lstm_hidden_size=hidden_size,
        features_extractor_class=SpatialNativeFeatures,
        features_extractor_kwargs={"feature_names": feature_names, "bounds": NATIVE_BOUNDS}).to("cuda")
    receipt = load_bc_actor(policy, args.checkpoint)
    policy.set_training_mode(False)
    actor_before = state_subset_sha256(policy, private_critic=False)
    config = {"schema": "BC-closed-loop-qualification-v1", "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint), "base_protocol": str(args.base_protocol.resolve()),
        "base_protocol_sha256": sha256_file(args.base_protocol), "workers": args.workers,
        "episodes_per_mode": 20, "modes": ["argmax", "categorical_sampling"],
        "device": str(policy.device), "native_rng_seed_applied": False, "seed": args.seed,
        "PPO_updates": 0, "teacher_actions": 0, "safety_takeovers": 0,
        "warmstart_receipt": receipt, "wall_cap_seconds": args.wall_cap_seconds,
        "observation_size": 1912, "feature_names_sha256": names_hash,
        "task_start_stage_frame": 12}
    write_json(output / "config.json", config)
    started = time.monotonic()
    groups = {}
    env = None
    try:
        for mode in config["modes"]:
            env = SubprocVecEnv([partial(NativeAvoidanceEnv, str(args.base_protocol.resolve()),
                str(output / mode / f"worker{worker}"), worker, "bc_development",
                task_reward="completion", append_task_time=True, capture_death_windows=True)
                for worker in range(args.workers)], start_method="spawn")
            observations = env.reset()
            state = None
            starts = np.ones(args.workers, dtype=bool)
            # Predetermined per-worker quotas avoid completion-dependent selection.
            quotas = np.array([(20 + worker) // args.workers for worker in range(args.workers)])
            counts = np.zeros(args.workers, dtype=int)
            rows = []
            extra_rows = []
            with (output / f"{mode}_episodes.jsonl").open("x", encoding="utf-8", buffering=1) as stream:
                while (counts < quotas).any():
                    if time.monotonic() - started > args.wall_cap_seconds or (output / "STOP").exists():
                        raise TimeoutError("BC qualification stopped at its wall cap or STOP")
                    actions, state = policy.predict(observations, state=state, episode_start=starts,
                                                    deterministic=mode == "argmax")
                    observations, _, starts, infos = env.step(actions)
                    for worker, done in enumerate(starts):
                        if not done:
                            continue
                        row = dict(infos[worker]["native_episode"], mode=mode,
                            checkpoint_sha256=config["checkpoint_sha256"],
                            counted_in_predeclared_quota=bool(counts[worker] < quotas[worker]))
                        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                        if row.get("native_start_stage_frame") != 12 or row.get("task_start_stage_frame") != 12:
                            raise ValueError("BC closed-loop episode did not start at fixed stage-frame 12")
                        if counts[worker] < quotas[worker]:
                            rows.append(row)
                            counts[worker] += 1
                        else:
                            extra_rows.append(row)
                    write_json(output / "progress.json", {"mode": mode, "completed": len(rows),
                        "target": 20, "successes": sum(row["horizon_completed_without_hit"] for row in rows),
                        "extra_worker_episodes_logged": len(extra_rows), "wall_seconds": time.monotonic() - started})
            env.close()
            env = None
            groups[mode] = {"episodes": len(rows), "successes": sum(
                row["horizon_completed_without_hit"] for row in rows),
                "mean_first_hit_frames": float(np.mean([row["frames"] for row in rows])),
                "episode_results": rows, "extra_worker_episodes_logged": len(extra_rows)}
            print(json.dumps({"mode": mode, "episodes": len(rows), "successes": groups[mode]["successes"]}))
        actor_unchanged = state_subset_sha256(policy, private_critic=False) == actor_before
        if not actor_unchanged or policy.optimizer.state:
            raise ValueError("BC qualification unexpectedly changed learner state")
        qualified = len(groups) == 2 and all(group["episodes"] == 20 and group["successes"] >= 1 for group in groups.values())
        write_json(output / "report.json", {"schema": "BC-closed-loop-qualification-v1", "status": "finished",
            "BC_qualified_for_PPO": qualified, "bc_qualified": qualified,
            "episodes_per_mode": 20, "deterministic_successes": groups["argmax"]["successes"],
            "sampling_successes": groups["categorical_sampling"]["successes"],
            "observation_size": 1912, "feature_names_sha256": names_hash,
            "task_start_stage_frame": 12,
            "groups": groups, "actor_unchanged": actor_unchanged,
            "PPO_updates": 0, "teacher_actions": 0, "safety_takeovers": 0,
            "bomb_actions": 0,
            "checkpoint_sha256": config["checkpoint_sha256"],
            "native_rng_controlled": False, "cross_rng_generalization_proven": False,
            "wall_seconds": time.monotonic() - started,
            "interpretation": "One success per mode is only the minimum RL-entry condition, not reliable avoidance."})
    except BaseException:
        write_json(output / "failure.json", {"status": "failed", "BC_qualified_for_PPO": False,
            "bc_qualified": False,
            "exception": traceback.format_exc(), "completed_groups": groups,
            "all_finished_episode_rows_retained_in_jsonl": True,
            "PPO_updates": 0, "wall_seconds": time.monotonic() - started})
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
