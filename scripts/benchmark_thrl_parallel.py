"""Short THRL original-game throughput benchmark; dry plan by default.

Execute only with --execute-native. No training/update, speed patch or native
RNG mutation. Torch stays inside the parent inference class so spawn workers
remain cold. --vector-steps is bounded to64, warmup to16, total wall to300s.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import statistics
import sys
import threading
import time
import traceback

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fly_touhou.atomic_json_io import atomic_write_json
from fly_touhou.thrl_study_io import read_shared_json


def _timed_native_factory(kwargs):
    """Timing-only subclass inside the worker; inherited lifecycle unchanged."""
    from fly_touhou.thrl_native_env import THRLNativeEnv

    class TimedEnv(THRLNativeEnv):
        def reset(self, **options):
            started = time.perf_counter()
            observation, info = super().reset(**options)
            info["benchmark_env_reset_seconds"] = time.perf_counter() - started
            return observation, info

        def step(self, action):
            started = time.perf_counter()
            observation, reward, terminated, truncated, info = super().step(action)
            info["benchmark_env_step_seconds"] = time.perf_counter() - started
            return observation, reward, terminated, truncated, info

    return TimedEnv(**kwargs)


def vector_factory(kwargs, output, timeout):
    import fly_touhou.thrl_vector as module
    previous = module.worker_factory
    module.worker_factory = _timed_native_factory
    try:
        return module.THRLNativeVector(kwargs, output, timeout=timeout)
    finally:
        module.worker_factory = previous


def file_evidence(path):
    path = Path(path).resolve(strict=True)
    blob = path.read_bytes()
    return {"path": str(path), "bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest()}


def source_evidence():
    names = ("thrl_native_env.py", "thrl_native_events.py", "thrl_event_geometry.py",
             "thrl_observation.py", "thrl_rewards.py", "thrl_vector.py", "thrl_model.py")
    return {name: file_evidence(ROOT / "src/fly_touhou" / name) for name in names}


def distribution(values):
    values = sorted(float(v) for v in values)
    if not values:
        return {"n": 0, "mean": None, "median": None, "p95": None, "max": None}
    return {"n": len(values), "mean": statistics.mean(values), "median": statistics.median(values),
            "p95": values[max(0, int(np.ceil(.95 * len(values))) - 1)], "max": values[-1]}


class CheckpointPolicy:
    """Frozen saved source19 no-attention model, CPU or CUDA batch inference."""
    def __init__(self, checkpoint, expected_sha256, device, count, seed, deterministic):
        import torch
        from fly_touhou.thrl_model import MOActorCritic
        self.torch, self.device, self.deterministic = torch, device, deterministic
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested; CPU fallback is disabled")
        blob = Path(checkpoint).read_bytes()
        self.checkpoint_sha256 = hashlib.sha256(blob).hexdigest()
        if self.checkpoint_sha256 != expected_sha256:
            raise ValueError("Checkpoint differs from the bound benchmark plan")
        # This is the project's own saved training checkpoint, including RNG
        # objects. Consume exactly the bytes hashed above, never a hot reread.
        payload = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=False)
        config = payload["model_config"]
        if (payload.get("schema") != "th14-thrl-training-checkpoint-v1"
                or config.get("action_dim") != 19 or config.get("attention") != "none"):
            raise ValueError("Benchmark requires saved source19 no-attention THRL checkpoint")
        self.model = MOActorCritic(**config)
        self.model.load_state_dict(payload["model"], strict=True)
        self.model = self.model.to(device).eval()
        self.hidden = torch.zeros(1, count, self.model.gru_hidden_size, device=device)
        self.metadata = {"checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_decisions": payload.get("decisions"),
            "checkpoint_adam_minibatch_steps": payload.get("adam_minibatch_steps"),
            "model_config": config, "device": device, "new_optimizer_updates": 0,
            "action_selection": "argmax" if deterministic else "categorical temperature1",
            "inference_seed": seed}
        del payload, blob
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(seed)
            torch.cuda.reset_peak_memory_stats()
        self.before_sha256 = self.fingerprint()

    def fingerprint(self):
        digest = hashlib.sha256()
        for key, value in self.model.state_dict().items():
            digest.update(key.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def select(self, observation):
        torch = self.torch
        with torch.inference_mode():
            logits, _, self.hidden = self.model.forward_step(
                torch.as_tensor(observation["features"], device=self.device),
                torch.as_tensor(observation["maps"], device=self.device), self.hidden)
            if self.deterministic:
                actions = logits.argmax(-1)
            else:
                actions = torch.distributions.Categorical(logits=logits.clamp(-80, 80)).sample()
            result = actions.cpu().numpy()
            if self.device == "cuda":
                torch.cuda.synchronize()
            return result

    def reset_at(self, indices):
        with self.torch.inference_mode():
            self.hidden[:, indices] = 0.

    def resources(self):
        if self.device != "cuda":
            return {"device": "cpu", "cuda_used": False}
        torch = self.torch
        free, total = torch.cuda.mem_get_info()
        return {"device": "cuda", "name": torch.cuda.get_device_name(),
            "allocated_bytes": torch.cuda.memory_allocated(), "reserved_bytes": torch.cuda.memory_reserved(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "device_free_bytes": free, "device_total_bytes": total,
            "device_free_scope": "whole device, includes other applications"}


class ResourceSampler:
    """Short wall-bounded resource sampling; only recorded owned process IDs."""
    def __init__(self, ownership, minimum_available_gib=8., maximum_cpu=80., interval=.5):
        self.ownership, self.minimum_available = Path(ownership), minimum_available_gib * 2**30
        self.maximum_cpu, self.interval = maximum_cpu, interval
        self.phase, self.samples, self.errors = "startup", [], []
        self.violation = None
        self._stop = threading.Event()
        self._cache = {}
        self._high_cpu_samples = 0
        self._thread = threading.Thread(target=self._run, name="short-throughput-resources", daemon=True)

    def start(self):
        psutil.cpu_percent(interval=None)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            memory = psutil.virtual_memory()
            cpu = psutil.cpu_percent(interval=None)
            row = {"unix": time.time(), "phase": self.phase,
                "system_cpu_percent": cpu, "ram_available_bytes": memory.available,
                "ram_total_bytes": memory.total, "ram_percent": memory.percent,
                "owned_processes": []}
            identities = [{"pid": os.getpid(), "create_time": psutil.Process().create_time()}]
            if self.ownership.exists():
                try:
                    owned = read_shared_json(self.ownership)
                    for item in owned.get("workers", []):
                        identities.extend([item.get("worker_identity"), item.get("native_identity")])
                except (OSError, ValueError) as error:
                    self.errors.append(repr(error))
            for identity in identities:
                if not identity or identity.get("create_time") is None:
                    continue
                try:
                    key = (identity["pid"], identity["create_time"])
                    process = self._cache.setdefault(key, psutil.Process(identity["pid"]))
                    if abs(process.create_time() - identity["create_time"]) > .000001:
                        continue  # PID reused: never attribute another program.
                    info = process.memory_info()
                    row["owned_processes"].append({"pid": process.pid,
                        "create_time": identity["create_time"], "rss_bytes": info.rss,
                        "private_bytes": getattr(info, "private", None),
                        "cpu_percent_one_core100": process.cpu_percent(interval=None)})
                except psutil.Error:
                    pass
            row["owned_rss_sum_bytes"] = sum(p["rss_bytes"] for p in row["owned_processes"])
            row["owned_cpu_equivalent_cores"] = sum(p["cpu_percent_one_core100"] for p in row["owned_processes"]) / 100.
            self.samples.append(row)
            if memory.available < self.minimum_available:
                self.violation = "system available RAM below reserved minimum"
            self._high_cpu_samples = self._high_cpu_samples + 1 if cpu > self.maximum_cpu else 0
            if self._high_cpu_samples >= 3:
                self.violation = "sustained system CPU exceeds work-headroom limit"
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=3.)

    def summary(self):
        return {"sample_interval_seconds": self.interval, "samples": len(self.samples),
            "system_cpu_percent": distribution([s["system_cpu_percent"] for s in self.samples]),
            "minimum_available_ram_bytes": min((s["ram_available_bytes"] for s in self.samples), default=None),
            "peak_owned_rss_sum_bytes": max((s["owned_rss_sum_bytes"] for s in self.samples), default=None),
            "owned_cpu_equivalent_cores": distribution([s["owned_cpu_equivalent_cores"] for s in self.samples]),
            "rss_sum_warning": "shared mappings may be counted in multiple process RSS values",
            "system_scope": "includes user's other applications; owned scope excludes unrecorded helpers",
            "resource_limit_violation": self.violation, "sampling_errors": self.errors}


def build_plan(args):
    config = read_shared_json(args.config) if args.config else {}
    game_exe, profile = args.game_exe or config.get("game_exe"), args.profile or config.get("profile")
    if not game_exe or not profile:
        raise ValueError("Supply --config or both --game-exe and --profile")
    evidence = file_evidence(args.checkpoint)
    if args.checkpoint_sha256 and args.checkpoint_sha256 != evidence["sha256"]:
        raise ValueError("Requested checkpoint SHA mismatch")
    return {"schema": "thrl-short-parallel-benchmark-plan-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": evidence, "executable": file_evidence(game_exe), "profile": file_evidence(profile),
        "source": source_evidence(), "workers": args.workers,
        "vector_steps": args.vector_steps, "warmup": args.warmup, "device": args.device,
        "seed": args.seed, "deterministic": args.deterministic, "frame_skip": 4,
        "policy_action_count": 19, "checkpoint_wait_sleep_ms": 10,
        "wall_cap_seconds": args.max_wall_seconds, "native_timeout_seconds": args.timeout,
        "minimum_available_ram_gib": args.minimum_available_ram_gib,
        "maximum_system_cpu_percent": args.maximum_system_cpu_percent,
        "source_reward_unchanged": True, "native_rng_mutated": False,
        "new_training_updates": 0, "formal_test": False,
        "purpose": "short opening-fragment throughput; not a long stability/performance test",
        "checkpoint_architecture": "pending actual load: source19, no attention required"}


def run_benchmark(plan, output, *, make_vector=vector_factory,
                  make_policy=CheckpointPolicy, sampler_class=ResourceSampler):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    atomic_write_json(output / "plan.json", plan)
    sampler = sampler_class(output / "workers/ownership.json",
        plan["minimum_available_ram_gib"], plan["maximum_system_cpu_percent"])
    sampler.start()
    vector = policy = None
    rows, episodes, reset_rows = [], [], []
    started = time.perf_counter()
    deadline = started + plan["wall_cap_seconds"]
    count = plan["workers"]
    result = {"status": "running", "workers": count, "new_optimizer_updates": 0,
              "acknowledged_decisions": 0, "acknowledged_native_frames": 0,
              "measured_decisions": 0, "partial_prefixes": "unknown-unfinished-native-games"}

    def check_budget():
        if sampler.violation:
            raise RuntimeError(sampler.violation)
        if time.perf_counter() >= deadline:
            raise TimeoutError("Short benchmark wall budget reached")
        if vector is not None:
            vector.timeout = min(plan["native_timeout_seconds"], max(.1, deadline-time.perf_counter()))

    try:
        check_budget()
        tick = time.perf_counter()
        policy = make_policy(plan["checkpoint"]["path"], plan["checkpoint"]["sha256"],
            plan["device"], count, plan["seed"], plan["deterministic"])
        result["model_load_seconds"] = time.perf_counter() - tick
        result["policy"] = policy.metadata
        kwargs = [dict(executable=plan["executable"]["path"], profile_path=plan["profile"]["path"],
            private_appdata=str(output / "native" / f"worker{i}" / "appdata"),
            difficulty=3, frame_skip=4, wall_time_limit_seconds=None,
            require_complete_reward_mapping=True, hide_window=True, window_title=None,
            timeout=25, policy_action_count=19) for i in range(count)]
        check_budget()
        tick = time.perf_counter()
        vector = make_vector(kwargs, output / "workers", min(plan["native_timeout_seconds"], deadline-tick))
        result["spawn_constructor_seconds"] = time.perf_counter()-tick
        check_budget()
        tick = time.perf_counter()
        observation, infos = vector.reset()
        result["initial_native_reset_seconds"] = time.perf_counter()-tick
        result["world_initial_reset_seconds"] = [row.get("benchmark_env_reset_seconds") for row in infos]
        result["startup_seconds_before_warmup"] = time.perf_counter()-started
        measured_started = None
        for step in range(plan["warmup"] + plan["vector_steps"]):
            check_budget()
            phase = "warmup" if step < plan["warmup"] else "measured"
            sampler.phase = phase
            if phase == "measured" and measured_started is None:
                measured_started = time.perf_counter()
            tick = time.perf_counter()
            # Shared rows are live. Read/copy ALL policy inputs before any step.
            copied = {key: value.copy() for key, value in observation.items()}
            copy_seconds = time.perf_counter()-tick
            tick = time.perf_counter()
            actions = policy.select(copied)
            policy_seconds = time.perf_counter()-tick
            tick = time.perf_counter()
            observation, rewards, terminated, truncated, infos = vector.step(actions)
            batch_seconds = time.perf_counter()-tick
            result["acknowledged_decisions"] += count
            result["acknowledged_native_frames"] += sum(int(row["elapsed_frames"]) for row in infos)
            if any(row.get("reward_mapping_complete") is not True for row in infos):
                raise RuntimeError("Full native THRL reward mapping became unknown")
            row = {"vector_step": step, "phase": phase,
                "observation_copy_seconds": copy_seconds, "policy_seconds": policy_seconds,
                "environment_batch_seconds": batch_seconds,
                "world_env_step_seconds": [i.get("benchmark_env_step_seconds") for i in infos],
                "actions": np.asarray(actions).tolist(),
                "elapsed_native_frames": [i["elapsed_frames"] for i in infos],
                "native_progress": [{key: i.get(key) for key in
                    ("pid", "stage", "stage_frame", "lives", "bombs", "misses", "bombs_used")} for i in infos]}
            rows.append(row)
            if phase == "measured":
                result["measured_decisions"] += count
            done = np.flatnonzero(terminated | truncated).tolist()
            if done:
                episodes.extend({"worker": i, "phase": phase, **infos[i]["episode_summary"]} for i in done)
                if step + 1 < plan["warmup"] + plan["vector_steps"]:
                    check_budget()
                    tick = time.perf_counter()
                    reset_infos = vector.reset_at(done)
                    reset_rows.append({"after_vector_step": step, "workers": done,
                        "reset_batch_seconds": time.perf_counter()-tick,
                        "world_reset_seconds": [i.get("benchmark_env_reset_seconds") for i in reset_infos]})
                    policy.reset_at(done)
                    observation = vector.observation
        measured_seconds = time.perf_counter() - measured_started
        selected = [row for row in rows if row["phase"] == "measured"]
        result.update(status="finished", measured_wall_seconds=measured_seconds,
            aggregate_decisions_per_second=result["measured_decisions"] / measured_seconds,
            environment_batch_seconds=distribution([r["environment_batch_seconds"] for r in selected]),
            policy_seconds=distribution([r["policy_seconds"] for r in selected]),
            observation_copy_seconds=distribution([r["observation_copy_seconds"] for r in selected]),
            world_latency_seconds=[distribution([r["world_env_step_seconds"][i] for r in selected
                                               if r["world_env_step_seconds"][i] is not None]) for i in range(count)],
            world_latency_scope="env.step only: native/input/read/encoding, excludes shared write/pipe",
            steady_wall_scope="includes copying, batched inference, dispatch, reset overhead and Python",
            completed_native_episodes=len(episodes))
    except BaseException as error:
        result.update(status="failed", error=repr(error), traceback=traceback.format_exc(),
            paid_cost_note="acknowledged full batches are lower bounds; an interrupted batch may have advanced additional worlds")
    finally:
        sampler.phase = "cleanup"
        if policy is not None:
            try:
                result["gpu_resources"] = policy.resources()
                after = policy.fingerprint()
                result.update(model_before_sha256=policy.before_sha256, model_after_sha256=after,
                              model_unchanged=after == policy.before_sha256)
            except BaseException:
                result["policy_final_check_error"] = traceback.format_exc()
        if vector is not None:
            try:
                tick = time.perf_counter()
                vector.close()
                result["cleanup_seconds"] = time.perf_counter()-tick
                result["close_errors"] = vector.close_errors
                result["ownership"] = read_shared_json(vector.ownership_path)
                records = result["ownership"].get("workers", [])
                result["cleanup_exact_identity_passed"] = bool(not vector.close_errors
                    and len(records) == count and all(r.get("worker_alive_after_close") is False for r in records)
                    and all(r.get("native_history") for r in records)
                    and all(not c.get("same_identity_still_live") for r in records for c in r.get("native_exit_checks", [])))
            except BaseException:
                result["cleanup_error"] = traceback.format_exc()
                result["cleanup_exact_identity_passed"] = False
        sampler.stop()
        result["resources"] = sampler.summary()
        result["wall_seconds_including_startup_cleanup"] = time.perf_counter()-started
        result["source_after"] = source_evidence()
        result["source_changed_during_benchmark"] = result["source_after"] != plan["source"]
        if result["status"] == "finished" and (not result.get("cleanup_exact_identity_passed")
                                               or not result.get("model_unchanged")):
            result["status"] = "failed-verification"
        if result["status"] == "finished" and sampler.violation:
            result["status"] = "failed-resource-bound"
        atomic_write_json(output / "result.json", result)
        atomic_write_json(output / "steps.json", rows)
        atomic_write_json(output / "episodes.json", episodes)
        atomic_write_json(output / "resets.json", reset_rows)
        atomic_write_json(output / "resources.json", sampler.samples)
    return result


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path)
    result.add_argument("--game-exe", type=Path)
    result.add_argument("--profile", type=Path)
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--checkpoint-sha256")
    result.add_argument("--workers", type=int, choices=(2, 4, 8, 16), required=True)
    result.add_argument("--vector-steps", type=int, choices=range(1, 65), default=32)
    result.add_argument("--warmup", type=int, choices=range(0, 17), default=4)
    result.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    result.add_argument("--seed", type=int, default=731011)
    result.add_argument("--deterministic", action="store_true")
    result.add_argument("--timeout", type=float, default=90.)
    result.add_argument("--max-wall-seconds", type=float, default=180.)
    result.add_argument("--minimum-available-ram-gib", type=float, default=8.)
    result.add_argument("--maximum-system-cpu-percent", type=float, default=80.)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--execute-native", action="store_true")
    return result


def main():
    args = parser().parse_args()
    if (not 1 <= args.max_wall_seconds <= 300 or not 0 < args.timeout <= 120
            or args.minimum_available_ram_gib < 8 or not 20 <= args.maximum_system_cpu_percent <= 85):
        raise ValueError("Invalid bounded wall/timeout or user-work resource reserve")
    plan = build_plan(args)
    if not args.execute_native:
        args.output.mkdir(parents=True, exist_ok=False)
        atomic_write_json(args.output / "dry_plan.json", plan)
        print(json.dumps({"status": "dry-plan-only", "output": str(args.output), "workers": args.workers,
            "native_started": False, "torch_loaded": "torch" in sys.modules}))
        return
    result = run_benchmark(plan, args.output)
    print(json.dumps({key: result.get(key) for key in
        ("status", "workers", "startup_seconds_before_warmup", "measured_wall_seconds",
         "aggregate_decisions_per_second", "cleanup_exact_identity_passed")}, ensure_ascii=False))
    if result["status"] != "finished":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
