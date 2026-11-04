"""Seal legacy baseline evidence and a provisional M0 contract, offline only."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fly_touhou.engineering_contract import (
    MetricCounters, compact_trajectory_fingerprint, file_record,
    sha256_file, validate_split_ledger, write_json_new,
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def json_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                yield line_number, json.loads(line)


def audit_checkpoint(path):
    # Trusted local experiment files only. weights_only avoids policy/class
    # unpickling, and CPU mapping neither initializes nor allocates on CUDA.
    import torch
    with zipfile.ZipFile(path) as archive:
        data = json.loads(archive.read("data"))
        optimizer = torch.load(io.BytesIO(archive.read("policy.optimizer.pth")),
                               map_location="cpu", weights_only=True)
        steps = [int(value.get("step", 0)) for value in optimizer["state"].values()]
        policy = torch.load(io.BytesIO(archive.read("policy.pth")),
                            map_location="cpu", weights_only=True)
        return {**file_record(path), "archive_members": archive.namelist(),
                "num_timesteps": data["num_timesteps"],
                "SB3_n_updates_epoch_counter": data["_n_updates"],
                "adam_minibatch_steps_max": max(steps, default=0),
                "optimizer_parameters_with_state": len(steps),
                "optimizer_step_values": sorted(set(steps)),
                "policy_state_tensor_elements": sum(v.numel() for v in policy.values()),
                "contains_full_native_snapshot": False,
                "contains_explicit_external_rng_sidecar": False,
                "loaded_policy_class": False, "device_used_for_audit": "cpu"}


def audit_experiment(experiment_id, directory):
    directory = Path(directory)
    config = read_json(directory / "config.json")
    result = read_json(directory / "training_result.json")
    evaluation = read_json(directory / "development_result.json")
    checkpoint = audit_checkpoint(directory / "final.zip")
    issues = []
    trace_counters = MetricCounters()
    ledger = []
    worker_results = []
    for trace_path in sorted((directory / "train_native").glob("worker*/transitions.jsonl")):
        episode_path = trace_path.with_name("episodes.jsonl")
        completed = {int(row["episode"]): row for _, row in json_rows(episode_path)}
        trajectories = defaultdict(list)
        worker_counters = MetricCounters()
        for line_number, row in json_rows(trace_path):
            elapsed = int(row["next_frame"]) - int(row["frame"])
            if elapsed != config["frame_skip"] or not 0 <= int(row["action"]) < config["actions"]:
                issues.append({"kind": "frame_or_action_mismatch", "path": str(trace_path),
                               "line": line_number})
            if abs(float(row["reward"]) - (elapsed / 60 - 5 * bool(row["hit"]))) > 1e-7:
                issues.append({"kind": "legacy_reward_mismatch", "path": str(trace_path),
                               "line": line_number})
            args = {"elapsed_frames": elapsed, "task_terminal": bool(row["task_terminal"]),
                    "hit": bool(row["hit"]), "success": False}
            trace_counters.transition(**args)
            worker_counters.transition(**args)
            trajectories[int(row["episode"])].append(row)
        for episode, rows in trajectories.items():
            terminal = bool(rows[-1]["task_terminal"])
            native_row = completed.get(episode)
            if terminal != (native_row is not None):
                issues.append({"kind": "trace_episode_terminal_mismatch",
                               "worker": trace_path.parent.name, "episode": episode})
            if native_row is not None:
                if len(rows) != native_row["decisions"] or sum(
                    row["next_frame"] - row["frame"] for row in rows
                ) != native_row["frames"]:
                    issues.append({"kind": "episode_metrics_mismatch", "episode": episode})
                if bool(rows[-1]["hit"]) != bool(native_row["hit"]):
                    issues.append({"kind": "episode_hit_mismatch", "episode": episode})
                ledger.append({"trajectory_uid": f"{experiment_id}/train/{trace_path.parent.name}/{episode}",
                    "role": "train", "experiment_id": experiment_id,
                    "trajectory_fingerprint": compact_trajectory_fingerprint(rows),
                    "fingerprint_kind": "compact_dynamics_v1",
                    "native_rng_initial": native_row.get("native_rng"),
                    "native_rng_controlled": native_row.get("game_rng_seed_applied", False),
                    "data_path": str(trace_path), "complete": True,
                    "used_for_model_selection": True})
        worker_results.append({"worker": trace_path.parent.name,
                               "metrics": worker_counters.to_dict(),
                               "partial_episodes": sorted(set(trajectories) - set(completed)),
                               "trace": file_record(trace_path), "episodes": file_record(episode_path)})
    selected_eval_rows = [row for _, row in json_rows(directory / "development_episodes.jsonl")]
    recomputed_evaluation = {}
    grouped = defaultdict(list)
    for row in selected_eval_rows:
        grouped[row["policy_kind"]].append(row)
        ledger.append({"trajectory_uid": f"{experiment_id}/historical_development/"
                      f"{row['policy_kind']}/{row['worker']}/{row['episode']}",
                      "role": "historical_development", "experiment_id": experiment_id,
                      "native_rng_initial": row.get("native_rng"),
                      "native_rng_controlled": row.get("game_rng_seed_applied", False),
                      "data_path": str(directory / "development_episodes.jsonl"),
                      "complete": True, "used_for_model_selection": False,
                      "fingerprint_kind": None, "trajectory_fingerprint": None,
                      "fingerprint_absence_reason": "not_reconstructed_across_legacy_eval_group_resets"})
    for policy, rows in sorted(grouped.items()):
        frames = [row["frames"] for row in rows]
        actual = {"episodes": len(rows), "no_hit_1024": sum(
            row["horizon_completed_without_hit"] for row in rows),
            "mean_first_hit_frames": sum(frames) / len(frames)}
        recomputed_evaluation[policy] = actual
        expected = evaluation["groups"][policy]
        for key, value in actual.items():
            if abs(value - expected[key]) > 1e-8:
                issues.append({"kind": "evaluation_summary_mismatch", "policy": policy, "metric": key})
    expected_new = result["decisions"] - config["initial_decisions"]
    if trace_counters.decisions != expected_new:
        issues.append({"kind": "decision_count_mismatch", "trace": trace_counters.decisions,
                       "expected_new": expected_new})
    if checkpoint["num_timesteps"] != result["decisions"]:
        issues.append({"kind": "checkpoint_decisions_mismatch"})
    if checkpoint["adam_minibatch_steps_max"] != result["adam_minibatch_steps"]:
        issues.append({"kind": "checkpoint_optimizer_mismatch"})
    if checkpoint["sha256"] != result["final_sha256"]:
        issues.append({"kind": "checkpoint_hash_mismatch"})
    source_records = []
    for label, expected in config["source_hashes"].items():
        source = Path(label)
        record = {"path": label, "recorded_sha256": expected, "exists": source.is_file()}
        if source.is_file():
            record["observed_sha256"] = sha256_file(source)
            record["matches_recorded"] = record["observed_sha256"] == expected
        source_records.append(record)
    rng_values = [row.get("native_rng_initial") for row in ledger]
    rng_counts = Counter(str(value) for value in rng_values)
    return {"experiment_id": experiment_id, "directory": str(directory),
            "status": "historical_failed_baseline", "formal_acceptance": False,
            "historical_observation_dimensions": config["raw_observation"],
            "reward": config["reward"], "algorithm_parameters": {key: config[key] for key in
                ("learning_rate", "gamma", "gae_lambda", "n_epochs", "n_steps", "workers", "batch_size")},
            "network_seed": config["seed"], "native_rng_controlled": False,
            "cumulative_decisions": result["decisions"],
            "initial_decisions": config["initial_decisions"],
            "new_decisions_from_actual_trace": trace_counters.decisions,
            "actual_native_frames_this_segment": trace_counters.native_frames,
            "actual_trace_metrics": trace_counters.to_dict(),
            "training_successes_from_episode_files": sum(
                row["horizon_completed_without_hit"] for worker in worker_results
                for _, row in json_rows(worker["episodes"]["path"])),
            "adam_minibatch_steps_cumulative": result["adam_minibatch_steps"],
            "adam_minibatch_steps_initial": config["initial_adam_minibatch_steps"],
            "checkpoint": checkpoint, "workers": worker_results,
            "evaluation_recomputed_from_episode_rows": recomputed_evaluation,
            "selected_development_episode_count": len(selected_eval_rows),
            "recorded_source_hashes": source_records,
            "rng_initial_repetition_counts_above_one": {key: count for key, count in rng_counts.items() if count > 1},
            "rng_independence_proven": False,
            "source_artifacts": [file_record(directory / name) for name in
                ("config.json", "training_result.json", "development_result.json",
                 "development_episodes.jsonl", "updates.jsonl")],
            "audit_issues": issues, "trace_and_checkpoint_consistent": not issues}, ledger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiments-root", type=Path, default=Path(
        "./outputs/experiments"))
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    baselines = []
    ledger = []
    for name, suffix in (
        ("flat-gru-v1-final65536", "native-standard-gru-ppo-v1/seed11_total65536"),
        ("spatial-gru-v2-final16384", "native-standard-gru-spatial-v2/seed11"),
    ):
        audit, episodes = audit_experiment(name, args.experiments_root / suffix)
        baselines.append(audit)
        ledger.extend(episodes)
    write_json_new(output / "baseline_index.json", {
        "schema": "engineering-baseline-index-v1", "observed_utc": started,
        "scope": "two_recent_standard_PPO_baselines_not_all_historical_fly_runs",
        "read_only_inputs": True, "baselines": baselines})
    write_json_new(output / "historical_episode_ledger.json", ledger)
    write_json_new(output / "historical_split_audit.json", validate_split_ledger(ledger))
    package_versions = {}
    for name in ("numpy", "torch", "gymnasium", "stable-baselines3", "sb3-contrib", "frida", "psutil"):
        try:
            package_versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            package_versions[name] = None
    critical_sources = [ROOT / relative for relative in (
        "pyproject.toml", "src/fly_touhou/engineering_contract.py",
        "scripts/prepare_engineering_m0.py", "src/fly_touhou/standard_gru_ppo.py",
        "scripts/train_standard_gru.py", "src/fly_touhou/native_diagnostic.py",
        "src/fly_touhou/native.py", "src/fly_touhou/hazard_reader.py",
        "src/fly_touhou/native_geometry.py", "src/fly_touhou/geometry_observation.py",
        "src/fly_touhou/spatial_native_features.py", "src/fly_touhou/background.py",
        "src/fly_touhou/frida_controller.py", "src/fly_touhou/responsive_controller.py",
        "src/fly_touhou/parallel_controller.py", "src/fly_touhou/hooks/th14.js",
        "src/fly_touhou/hooks/th14_responsive.js", "src/fly_touhou/hooks/th14_parallel_startup.js",
        "artifacts/plans/gru_v3_task_contract_20261020.json",
        "artifacts/plans/engineering_roadmap_20261020.json",
        "artifacts/diagnostic_protocol/native-list-gru-reuse4-v1.json",
    )]
    config = read_json(args.experiments_root / "native-standard-gru-spatial-v2/seed11/config.json")
    executable = Path(config["game_exe"])
    exe_record = {"path": str(executable), "exists": executable.is_file(),
                  "expected_sha256": config["game_exe_sha256"]}
    if executable.is_file():
        exe_record.update(file_record(executable))
        exe_record["matches_expected"] = exe_record["sha256"] == exe_record["expected_sha256"]
    write_json_new(output / "version_inventory.json", {
        "schema": "engineering-version-observation-v1", "observed_utc": started,
        "status": "inventory_only_not_final_M1_M2_freeze",
        "python": {"executable": sys.executable, "version": platform.python_version()},
        "packages": package_versions, "original_exe": exe_record,
        "sources": [file_record(path, root=ROOT) for path in critical_sources if path.is_file()],
        "missing_critical_paths": [str(path) for path in critical_sources if not path.is_file()],
        "training_launched": False, "native_process_started": False,
        "cuda_operations_performed": False})
    write_json_new(output / "experiment_contract.json", {
        "schema": "engineering-short-task-contract-v1", "observed_utc": started,
        "status": "provisional_until_M1_M2_frozen",
        "inherit": "artifacts/plans/gru_v3_task_contract_20261020.json",
        "task": {"game": "TH14 1.00b original executable", "sha256": config["game_exe_sha256"],
                 "difficulty": "Lunatic", "character": "Reimu A", "native_frame_horizon": 1024,
                 "frames_per_decision": 4, "actions": 18, "shot": "held", "bomb": False,
                 "first_hit_terminal": True, "hit_on_horizon": "failure"},
        "observation": {"proposed_total_dimensions": 1912, "raw_dimensions": 1911,
                        "task_logical_time_dimensions": 1, "version": None,
                        "freeze_requires": ["M2 critical geometry and hazard audit", "M1 chosen backend"]},
        "reward": {"success": 1, "hit": -1, "ordinary": 0,
                   "task_terminal_bootstrap": False, "rollout_boundary_bootstrap": True},
        "model": {"features": "spatial_CNN", "actor": "GRU128", "critic": "separate_GRU128",
                  "training_and_inference_device": "cuda"},
        "PPO": {"learning_rate": 0.0001, "gamma": 1, "gae_lambda": .95, "n_epochs": 3,
                "n_steps_per_env": 128, "batch_size": 128, "clip_range": .2,
                "vf_coef": .5, "max_grad_norm": .5, "ent_coef": 0,
                "clip_range_vf": None, "target_kl": None, "custom_update_rejection": False},
        "execution_blockers": ["M1 backend and workers not frozen", "M2 observation not frozen",
                               "M2 teacher qualification missing", "M3 BC qualification missing"],
        "formal_acceptance": {"seeds": [11, 23, 37], "new_episodes_each": 20,
                              "minimum_success_each": 16, "relative_G2_gain": .2},
        "initial_budget_decisions_all_envs": 102400,
        "network_seed_is_native_RNG_seed": False})
    write_json_new(output / "split_and_rng_contract.json", {
        "schema": "engineering-data-splits-v1", "status": "reserved_no_new_data_collected",
        "split_unit": "whole_native_episode",
        "datasets": [
            {"id": "M2_diagnostics", "role": "diagnostic", "maximum_episodes": 40},
            {"id": "M2_teacher_qualification", "role": "development", "episodes": 20,
             "not_reused_for_BC": True},
            {"id": "M3_teacher_demonstrations_train", "role": "train", "episodes": 64},
            {"id": "M3_teacher_demonstrations_validation", "role": "validation", "episodes": 16},
            {"id": "M3_DAgger", "role": "train", "maximum_rounds": 2, "episodes_per_round": 20},
            {"id": "M3_closed_loop", "role": "development", "episodes_per_execution_mode": 20,
             "modes": ["argmax", "categorical_sampling"]},
            {"id": "M4_periodic_selection", "role": "development", "episodes_per_checkpoint": 10,
             "every_decisions": 25600},
            {"id": "M4_final", "role": "formal_test", "episodes_per_seed": 20,
             "seeds": [11, 23, 37], "used_for_model_selection": False},
        ],
        "episode_record_required": ["trajectory_uid", "role", "experiment_id", "policy_sha256",
                                    "native_rng_initial", "native_rng_controlled", "reset_ordinal",
                                    "trajectory_fingerprint", "fingerprint_kind", "terminal_cause",
                                    "used_for_model_selection", "source_trace_path"],
        "formal_test_reuse_rule": "If results guide a model change, retain the result as development evidence and allocate a new formal test batch.",
        "RNG_scopes": {"learner": "record Python, NumPy, Torch CPU/CUDA and named generators separately",
                       "game": "not controlled; observe initial RNG and fingerprints; a fresh process is not proof of independence"},
        "historical_data": "development/diagnostic only; never relabel as new formal test",
        "independence_proven": False})
    write_json_new(output / "metrics_and_resume_contract.json", {
        "schema": "engineering-metrics-resume-v1",
        "metrics": {
            "decisions": "one environment step; vector step adds active environment count",
            "native_frames": "sum of measured after.frame-before.frame; never inferred from rendered FPS",
            "episodes": "completed native task terminals; partial episodes reported separately",
            "attempts": "all started attempts including interface faults and human interruptions",
            "adam_minibatch_steps": "actual optimizer.step count, not parameter-state count or SB3 epoch counter",
            "SB3_n_updates": "SB3 recurrent PPO epoch counter; distinct from optimizer minibatch steps",
            "training_seed": "learner randomness, distinct from native game RNG",
            "failure_rate": "task failures and interface faults reported with explicit denominators"},
        "terminal_causes": ["hit", "horizon_success", "task_timeout", "boss_defeated", "campaign_clear",
                            "interface_fault", "human_interruption"],
        "event_link_fields": ["run_id", "policy_version", "worker", "episode", "decision",
                              "native_frame_before", "native_frame_after", "observation_version",
                              "observation_ref", "action", "input_mask", "action_log_probability",
                              "scalar_reward", "reward_components", "terminal_cause", "rollout_id"],
        "update_link_fields": ["run_id", "rollout_id", "policy_before_sha256", "policy_after_sha256",
                               "decision_start", "decision_end", "adam_step_start", "adam_step_end"],
        "legacy_trace_limits": ["full raw/encoded observations absent at most terminal windows",
                                "no policy probability in legacy compact rows",
                                "no exact per-transition rollout/update identity"],
        "checkpoint_requirements": ["weights", "optimizer", "cumulative counters", "contract and source hashes",
                                    "Python/NumPy/Torch CPU/CUDA RNG", "named generator states if used"],
        "new_native_resume_semantics": {"full_native_snapshot_available": False,
            "restore_learner_optimizer_and_rng": True, "open_fresh_native_episode": True,
            "reset_recurrent_state": True, "discard_incomplete_rollout": True,
            "log_trajectory_discontinuity": True, "claim_exact_native_continuation": False},
        "BC_to_PPO": "transfer actor features/GRU/head, retain same-seed untrained critic branches, fresh Adam; this is initialization rather than resume",
        "old_baseline_resume": "optimizer exists; explicit full learner RNG sidecars do not; do not claim exact learner/native continuation",
        "runtime_integration_status": "parent_agent_implementation_pending_validation"})
    validation = {
        "schema": "M0-offline-evidence-validation-v1", "observed_utc": started,
        "baseline_count": len(baselines),
        "baseline_trace_and_checkpoint_consistent": all(item["trace_and_checkpoint_consistent"] for item in baselines),
        "original_exe_hash_matches": exe_record.get("matches_expected", False),
        "historical_split_audit": validate_split_ledger(ledger),
        "M0_recording_tool_implemented": True,
        "full_M0_runtime_event_to_update_trace_proven": False,
        "final_protocol_freeze_proven": False,
        "M1_M2_M3_learning_gates_passed": False,
        "training_or_native_execution_in_this_script": False,
        "status": "offline_baseline_seal_complete_runtime_freeze_pending",
    }
    write_json_new(output / "validation.json", validation)
    print(json.dumps({key: value for key, value in validation.items() if key != "historical_split_audit"},
                     ensure_ascii=False, indent=2))
    if not validation["baseline_trace_and_checkpoint_consistent"] or not validation["original_exe_hash_matches"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
