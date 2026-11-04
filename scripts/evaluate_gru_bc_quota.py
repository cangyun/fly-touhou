"""BC development qualification using the reviewed fixed-quota evaluator.

Real evaluation requires CUDA and starts owned original-game processes.
--validate-only checks committed files on CPU and never starts a game.
This is the existing BC RL-entry test, not formal M4/G2 evaluation.
"""
from __future__ import annotations

import argparse
from functools import partial
import hashlib
from importlib.metadata import distribution, version
import json
from pathlib import Path
import random
import signal
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from fly_touhou.engineering_contract import canonical_sha256, file_record, sha256_file
from fly_touhou.atomic_json_io import atomic_write_json as write_json
from fly_touhou.gru_ppo_evaluation import (
    EvaluationStopped, FixedQuotaVecMixin, audit_native_logs, evaluate_vector_group, validate_episode)

SCHEMA = "BC-closed-loop-qualification-v1"
MODES = ("argmax", "categorical_sampling")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def runtime_sources():
    package = ROOT / "src/fly_touhou"
    dependencies = {
        "stable-baselines3": ("stable_baselines3/common/base_class.py",
            "stable_baselines3/common/vec_env/base_vec_env.py",
            "stable_baselines3/common/vec_env/subproc_vec_env.py"),
        "sb3-contrib": ("sb3_contrib/common/recurrent/policies.py",
            "sb3_contrib/ppo_recurrent/ppo_recurrent.py")}
    paths = [Path(__file__), *sorted(path for path in package.rglob("*")
        if path.is_file() and path.suffix in (".py", ".js")),
        *(Path(distribution(name).locate_file(path)) for name, paths in dependencies.items() for path in paths)]
    return [file_record(path) for path in paths]


def inspect_inputs(checkpoint, base_protocol, selection=None):
    """Bind selected weights, optimizer and RNG without restoring training state."""
    import torch
    from fly_touhou.geometry_observation import GeometryObservationEncoder
    from fly_touhou.gru_imitation import NATIVE_BOUNDS, inspect_bc_training_bundle

    checkpoint, base_protocol = Path(checkpoint).resolve(), Path(base_protocol).resolve()
    selection = Path(selection).resolve() if selection else checkpoint.parent / "selected_checkpoint.json"
    receipt_path = checkpoint.with_suffix(".resume.json")
    receipt = read_json(receipt_path)
    contract = receipt["semantic_contract"]
    receipt = inspect_bc_training_bundle(checkpoint, semantic_contract=contract)
    chosen = read_json(selection)
    if (Path(chosen["path"]).resolve() != checkpoint or chosen.get("sha256") != sha256_file(checkpoint)
            or chosen.get("model_epoch") != receipt["model_epoch"]):
        raise ValueError("Selected BC weights do not match the selected checkpoint and resume receipt")
    protocol = read_json(base_protocol)
    names = list(GeometryObservationEncoder.from_specification(protocol["encoder"]).feature_names) + ["task_progress"]
    names_hash = hashlib.sha256(json.dumps(names).encode()).hexdigest()
    metadata = torch.load(checkpoint, map_location="cpu", weights_only=True).get("metadata", {})
    if (len(names) != 1912 or metadata.get("hidden_size") not in (128, 256)
            or metadata.get("observation_size") != 1912 or metadata.get("task_start_stage_frame") != 12
            or metadata.get("feature_names_sha256") != names_hash
            or tuple(metadata.get("bounds", ())) != NATIVE_BOUNDS
            or metadata.get("run_id") != receipt["run_id"]
            or metadata.get("model_epoch") != receipt["model_epoch"]
            or metadata.get("BC_optimizer_steps") != receipt["adam_steps"]
            or metadata.get("semantic_contract_sha256") != receipt["semantic_contract_sha256"]
            or contract.get("protocol_sha256") != sha256_file(base_protocol)
            or contract.get("hidden_size") != metadata["hidden_size"]
            or contract.get("task_start_stage_frame") != 12
            or contract.get("bounds") != list(NATIVE_BOUNDS)
            or contract.get("feature_names_sha256") != canonical_sha256(names)):
        raise ValueError("BC checkpoint observation, movement or selected-bundle metadata differ")
    # Training entry provenance is retained in the receipt; only executable
    # inference modules must match this ROOT. No second package root is mixed in.
    for name in ("gru_imitation.py", "standard_gru_ppo.py", "spatial_native_features.py"):
        key = next((key for key in contract["source_hashes"] if key.replace("\\", "/") == f"src/fly_touhou/{name}"), None)
        if key is None or sha256_file(ROOT / "src/fly_touhou" / name) != contract["source_hashes"][key]:
            raise ValueError(f"BC inference source changed: {name}")
    libraries = {name: version(name) for name in ("torch", "numpy", "stable-baselines3", "sb3-contrib")}
    if any(libraries.get(name) != expected for name, expected in contract["library_versions"].items()):
        raise ValueError("BC inference dependency versions differ from its committed bundle")
    return {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
        "base_protocol": str(base_protocol), "base_protocol_sha256": sha256_file(base_protocol),
        "feature_names": names, "feature_names_sha256": names_hash, "metadata": metadata,
        "selected_bundle": {"selection": file_record(selection), "resume": file_record(receipt_path),
            "files": receipt["files"], "run_id": receipt["run_id"], "model_epoch": receipt["model_epoch"],
            "adam_steps": receipt["adam_steps"], "semantic_contract_sha256": receipt["semantic_contract_sha256"],
            "cumulative_epochs_consumed": chosen.get("cumulative_epochs_consumed"),
            "selected_steps_are_not_total_training_cost": True},
        "runtime_dependencies": libraries, "runtime_sources": runtime_sources()}


def verify_inputs(inputs):
    records = [*inputs["selected_bundle"]["files"].values(), inputs["selected_bundle"]["selection"],
               inputs["selected_bundle"]["resume"], *inputs["runtime_sources"]]
    for record in records:
        if file_record(record["path"]) != record:
            raise ValueError(f"Qualification input changed: {record['path']}")
    if sha256_file(inputs["base_protocol"]) != inputs["base_protocol_sha256"]:
        raise ValueError("Qualification protocol changed")


def load_cuda_policy(inputs, seed):
    import gymnasium as gym
    import torch
    from fly_touhou.gru_imitation import NATIVE_BOUNDS
    from fly_touhou.spatial_native_features import SpatialNativeFeatures
    from fly_touhou.standard_gru_ppo import NativeGRUPolicy, load_bc_actor

    if not torch.cuda.is_available():
        raise RuntimeError("BC qualification requires CUDA; no CPU fallback")
    torch.set_num_threads(2)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    policy = NativeGRUPolicy(gym.spaces.Box(-1., 1., shape=(1912,), dtype=np.float32),
        gym.spaces.Discrete(18), lambda _: 1e-4, lstm_hidden_size=inputs["metadata"]["hidden_size"],
        features_extractor_class=SpatialNativeFeatures,
        features_extractor_kwargs={"feature_names": inputs["feature_names"], "bounds": NATIVE_BOUNDS}).to("cuda")
    receipt = load_bc_actor(policy, inputs["checkpoint"])
    policy.set_training_mode(False)
    return policy, receipt


def policy_state(policy):
    from fly_touhou.gru_imitation import state_subset_sha256
    return {"actor": state_subset_sha256(policy, private_critic=False),
        "private_critic": state_subset_sha256(policy, private_critic=True),
        "optimizer_has_state": bool(policy.optimizer.state)}


def make_vector(inputs, directory, workers):
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from fly_touhou.standard_gru_ppo import NativeAvoidanceEnv

    class BCQuotaVecEnv(FixedQuotaVecMixin, SubprocVecEnv):
        pass
    return BCQuotaVecEnv([partial(NativeAvoidanceEnv, inputs["base_protocol"],
        str(directory / f"worker{worker}"), worker, "bc_development", task_reward="completion",
        append_task_time=True, capture_death_windows=True) for worker in range(workers)], start_method="spawn")


def legacy_group(group, mode, checkpoint_sha):
    group = dict(group)
    for key in ("episode_results", "extra_episode_results"):
        group[key] = [dict(row, mode=mode, checkpoint_sha256=checkpoint_sha) for row in group.get(key, [])]
    rows = group["episode_results"]
    group.update(episodes=len(rows), successes=sum(row["horizon_completed_without_hit"] for row in rows),
        mean_first_hit_frames=float(np.mean([row["frames"] for row in rows])) if rows else None,
        extra_worker_episodes_logged=len(group["extra_episode_results"]))
    return group


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def inspect_recovery(directory, inputs):
    """Recover this interrupted 20+2 attempt; unfinished native play stays cost-only.

    Recovery is intentionally limited to the audited progress-file sharing fault.
    It never writes to the original directory or infers an unfinished hit result.
    """
    directory = Path(directory).resolve()
    config, failure = read_json(directory / "config.json"), read_json(directory / "failure.json")
    exception = failure.get("exception", "")
    if (config.get("schema") != SCHEMA or config.get("workers") != 1
            or config.get("checkpoint_sha256") != inputs["checkpoint_sha256"]
            or Path(config.get("checkpoint", "__missing__")).resolve() != Path(inputs["checkpoint"])
            or config.get("base_protocol_sha256") != inputs["base_protocol_sha256"]
            or config.get("feature_names_sha256") != inputs["feature_names_sha256"]
            or config.get("episodes_per_mode") != 20 or config.get("modes") != list(MODES)
            or config.get("task_start_stage_frame") != 12 or config.get("observation_size") != 1912
            or config.get("PPO_updates") != 0 or config.get("teacher_actions") != 0
            or config.get("safety_takeovers") != 0 or failure.get("status") != "failed"
            or failure.get("PPO_updates") != 0 or "[WinError 5]" not in exception
            or "progress.tmp" not in exception or "progress.json" not in exception
            or "temporary.replace(path)" not in exception):
        raise ValueError("Recovery requires the same selected BC and the audited progress WinError 5 attempt")
    loaded = config.get("warmstart_receipt", {})
    if (not str(config.get("device", "")).startswith("cuda")
            or loaded.get("critic_private_unchanged") is not True
            or loaded.get("PPO_optimizer_steps_at_start") != 0
            or loaded.get("BC_metadata") != inputs["metadata"]):
        raise ValueError("Original qualification does not describe the same immutable CUDA BC policy")
    groups, audits, records, costs = {}, {}, [], {}
    for mode, target in (("argmax", 20), ("categorical_sampling", 2)):
        episode_path = directory / f"{mode}_episodes.jsonl"
        rows = read_jsonl(episode_path)
        if (len(rows) != target or len({(row["worker"], row["episode"]) for row in rows}) != target
                or any(row.get("worker") != 0 or row.get("mode") != mode
                    or row.get("checkpoint_sha256") != inputs["checkpoint_sha256"]
                    or row.get("counted_in_predeclared_quota") is not True for row in rows)):
            raise ValueError("Recovery completed rows do not describe exactly 20 argmax and 2 sampling episodes")
        for row in rows:
            validate_episode(row)
        group = legacy_group({"episode_results": rows, "extra_episode_results": [],
            "worker_quotas": [20]}, mode, inputs["checkpoint_sha256"])
        audit = audit_native_logs(directory / mode, group, require_complete=mode == "argmax")
        if audit["issues"] or audit["native_completed_episodes"] != target:
            raise ValueError("Original qualification JSONL and native completions disagree")
        transitions = read_jsonl(directory / mode / "worker0/transitions.jsonl")
        resets = read_jsonl(directory / mode / "worker0/resets.jsonl")
        complete_ids = {row["episode"] for row in rows}
        reset_ids = {row["episode"] for row in resets}
        if (len(reset_ids) != len(resets) or any(row["episode"] not in reset_ids for row in transitions)
                or any(reset.get("task_start_stage_frame") != 12 for reset in resets)
                or any(row["next_frame"] - row["frame"] != 4 for row in transitions)):
            raise ValueError("Recovery reset/transition frame accounting differs from the fixed task")
        unfinished = [row for row in transitions if row["episode"] not in complete_ids]
        if any(row.get("hit") or row.get("task_terminal") for row in unfinished):
            raise ValueError("A completed native outcome cannot be hidden as an interrupted prefix")
        prefix_decisions = len(unfinished)
        if (mode == "argmax" and prefix_decisions != 0
                or mode == "categorical_sampling" and prefix_decisions != 148
                or len(audit["unfinished_reset_prefixes"]) != 1):
            raise ValueError("Recovery requires the audited argmax reset-only and 148-step sampling prefix")
        costs[mode] = {"completed_episodes": target,
            "completed_decisions": sum(row["decisions"] for row in rows),
            "completed_frames": sum(row["frames"] for row in rows),
            "unfinished_prefix_decisions": prefix_decisions, "unfinished_prefix_frames": 4 * prefix_decisions,
            "unfinished_prefix_counted_as_hit": False, "unfinished_prefix_counted_in_20_completed_quota": False}
        groups[mode], audits[mode] = group, audit
        records.extend([file_record(episode_path), *audit["files"]])
    if groups["argmax"] != legacy_group({**failure["completed_groups"]["argmax"],
            "extra_episode_results": [], "worker_quotas": [20]}, "argmax", inputs["checkpoint_sha256"]):
        raise ValueError("Failure receipt and imported argmax rows disagree")
    return {"directory": str(directory), "files": [file_record(directory / "config.json"),
        file_record(directory / "failure.json"), *records], "groups": groups, "native_log_audits": audits,
        "old_seed": config["seed"], "remaining_sampling_episodes": 18, "retained_cost": costs,
        "original_actor_final_hash_available": False,
        "original_actor_evidence": "The original frozen read-only BC evaluator ran zero optimizer updates; no final state hash was saved before its JSON fault.",
        "RNG_bitwise_resume": False, "native_rng_controlled": False}


def verify_recovery(recovery):
    for record in recovery["files"]:
        if file_record(record["path"]) != record:
            raise ValueError(f"Original recovery evidence changed: {record['path']}")


def merge_recovered_group(recovery, fresh, inputs):
    imported = recovery["groups"]["categorical_sampling"]["episode_results"]
    merged = legacy_group({"episode_results": imported + fresh["episode_results"],
        "extra_episode_results": fresh["extra_episode_results"]}, "categorical_sampling", inputs["checkpoint_sha256"])
    merged.update(imported_episodes=len(imported), new_episodes=fresh["episodes"],
        new_attempt_evaluation_group=fresh,
        imported_native_log_audit=recovery["native_log_audits"]["categorical_sampling"],
        native_audit_scope="The new attempt audit covers 18 new completions; the imported audit covers the original 2.")
    return merged


def run_qualification(inputs, output, *, workers=2, seed=110011, wall_cap_seconds=1800., stopping=None,
                      recovery=None):
    """One policy load, two fixed-quota groups, no learning or model selection."""
    import torch

    if workers not in (1, 2) or wall_cap_seconds <= 0:
        raise ValueError("BC qualification requires one/two workers and a positive wall cap")
    output = Path(output)
    started, run_id = time.monotonic(), uuid.uuid4().hex
    if recovery is not None and (workers != 2 or seed != 111012):
        raise ValueError("This bounded recovery uses two workers and the explicitly redeclared seed 111012")
    groups = {"argmax": dict(recovery["groups"]["argmax"],
        imported_native_log_audit=recovery["native_log_audits"]["argmax"])} if recovery else {}
    current, model_before, policy = None, None, None
    modes = ("categorical_sampling",) if recovery else MODES
    episode_target = 18 if recovery else 20
    stopping = stopping or (lambda: (output / "STOP").exists() or (output.parent / "STOP").exists())
    config = {"schema": SCHEMA, **{key: value for key, value in inputs.items() if key not in ("feature_names", "metadata")},
        "workers": workers, "episodes_per_mode": 20, "modes": list(modes), "seed": seed,
        "new_episodes_per_executed_mode": episode_target,
        "worker_quotas_per_mode": [episode_target // workers] * workers, "run_id": run_id,
        "task_start_stage_frame": 12, "observation_size": 1912, "frame_skip": 4,
        "horizon_frames": 1024, "actions": 18, "bomb": False, "device_required": "cuda",
        "wall_cap_seconds": wall_cap_seconds, "formal_test": False, "formal_M4_accepted": False,
        "native_rng_seed_applied": False, "PPO_updates": 0, "teacher_actions": 0, "safety_takeovers": 0,
        "recovery_provenance": recovery}
    write_json(output / "config.json", config)
    try:
        verify_inputs(inputs)
        if recovery:
            verify_recovery(recovery)
        policy, warmstart = load_cuda_policy(inputs, seed)
        model_before = policy_state(policy)
        if model_before["optimizer_has_state"]:
            raise ValueError("Qualification may not restore a training optimizer")
        config.update(device=str(policy.device), warmstart_receipt=warmstart)
        write_json(output / "config.json", config)
        with (output / "events.jsonl").open("x", encoding="utf-8", buffering=1) as events:
            for current in modes:
                stream_path = output / f"{current}_episodes.jsonl"
                with stream_path.open("x", encoding="utf-8", buffering=1) as stream:
                    last_progress = [-1]
                    def event(kind, **fields):
                        if kind == "episode_finished":
                            row = dict(fields["row"], mode=current, checkpoint_sha256=inputs["checkpoint_sha256"])
                            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                        elif kind == "group_progress":
                            if fields["episodes"] != last_progress[0]:
                                last_progress[0] = fields["episodes"]
                                write_json(output / "progress.json", dict(fields, mode=current, target=episode_target,
                                    completed=fields["episodes"], wall_seconds=time.monotonic() - started))
                        else:
                            events.write(json.dumps(dict(kind=kind, **fields), ensure_ascii=False, allow_nan=False) + "\n")
                    def predict(observations, state, starts):
                        with torch.no_grad():
                            return policy.predict(observations, state=state, episode_start=starts,
                                                  deterministic=current == "argmax")
                    result = evaluate_vector_group(lambda: make_vector(inputs, output / current, workers), predict,
                        policy_kind=current, workers=workers, run_id=run_id, stop_requested=stopping,
                        deadline=started + wall_cap_seconds, event=event, native_output=output / current,
                        episode_target=episode_target)
                    fresh = legacy_group(result, current, inputs["checkpoint_sha256"])
                    groups[current] = merge_recovered_group(recovery, fresh, inputs) if recovery else fresh
        model_after = policy_state(policy)
        if model_after != model_before or model_after["optimizer_has_state"]:
            raise ValueError("BC qualification unexpectedly changed learner state")
        verify_inputs(inputs)
        if recovery:
            verify_recovery(recovery)
        qualified = all(group["episodes"] == 20 and group["successes"] >= 1 for group in groups.values())
        report = {"schema": SCHEMA, "status": "finished", "BC_qualified_for_PPO": qualified, "bc_qualified": qualified,
            "episodes_per_mode": 20, "deterministic_successes": groups["argmax"]["successes"],
            "sampling_successes": groups["categorical_sampling"]["successes"], "groups": groups,
            "observation_size": 1912, "feature_names_sha256": inputs["feature_names_sha256"],
            "task_start_stage_frame": 12, "actor_unchanged": True, "private_critic_unchanged": True,
            "PPO_updates": 0, "teacher_actions": 0, "safety_takeovers": 0, "bomb_actions": 0,
            "checkpoint_sha256": inputs["checkpoint_sha256"], "selected_bundle": inputs["selected_bundle"],
            "recovery_provenance": recovery, "new_completed_episodes": sum(
                group.get("new_episodes", group["episodes"] if not recovery else 0) for group in groups.values()),
            "device": str(policy.device), "model_checks": {"before": model_before, "after": model_after},
            "native_rng_controlled": False, "cross_rng_generalization_proven": False,
            "wall_seconds": time.monotonic() - started, "formal_test": False, "formal_M4_accepted": False,
            "interpretation": "One success per mode is only the minimum RL-entry condition, not reliable avoidance."}
        write_json(output / "report.json", report)
        return report
    except BaseException as caught:
        partial_group = getattr(caught, "partial_evaluation_group", None)
        if partial_group is not None and current is not None:
            fresh = legacy_group(partial_group, current, inputs["checkpoint_sha256"])
            groups[current] = merge_recovered_group(recovery, fresh, inputs) if recovery else fresh
        failure = {"schema": SCHEMA, "status": "interrupted" if isinstance(caught, EvaluationStopped) else "failed",
            "BC_qualified_for_PPO": False, "bc_qualified": False, "exception": traceback.format_exc(),
            "completed_groups": groups, "interrupted_mode": current, "PPO_updates": 0,
            "checkpoint_sha256": inputs["checkpoint_sha256"], "all_finished_episode_rows_retained_in_jsonl": True,
            "recovery_provenance": recovery,
            "system_fault_counted_as_hit": False, "formal_test": False, "formal_M4_accepted": False,
            "wall_seconds": time.monotonic() - started}
        write_json(output / "failure.json", failure)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--base-protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--seed", type=int, default=110011)
    parser.add_argument("--wall-cap-seconds", type=float, default=1800.)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--recover-from-qualification", type=Path)
    args = parser.parse_args()
    if args.wall_cap_seconds <= 0:
        parser.error("The qualification wall cap must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    inputs = inspect_inputs(args.checkpoint, args.base_protocol, args.selection)
    recovery = inspect_recovery(args.recover_from_qualification, inputs) if args.recover_from_qualification else None
    if recovery and (args.workers != 2 or args.seed != 111012):
        parser.error("The audited 18-episode recovery requires --workers 2 --seed 111012")
    if args.validate_only:
        report = {"status": "validated_only", "input_binding": inputs, "recovery_provenance": recovery,
            "original_game_started": False, "CUDA_started": False,
            "BC_qualified_for_PPO": False, "formal_test": False, "formal_M4_accepted": False}
        write_json(output / "report.json", report)
        print(json.dumps({key: value for key, value in report.items()
                          if key not in ("input_binding", "recovery_provenance")}))
        return
    interrupted = [False]
    previous = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.signal(signum, lambda *_: interrupted.__setitem__(0, True))
    try:
        report = run_qualification(inputs, output, workers=args.workers, seed=args.seed,
            wall_cap_seconds=args.wall_cap_seconds, recovery=recovery,
            stopping=lambda: interrupted[0] or (output / "STOP").exists() or (output.parent / "STOP").exists())
        print(json.dumps({key: report[key] for key in ("status", "BC_qualified_for_PPO", "deterministic_successes", "sampling_successes")}))
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    main()
