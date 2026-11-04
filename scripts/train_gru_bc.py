"""Bounded CUDA whole-episode BC; does not start the original game.

The default budget includes the 2048-decision recurrent-prefix probe: three
probe epochs followed by at most 27 full-dataset epochs.  Offline improvements
do not qualify the actor for PPO; closed-loop tests remain a separate gate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import gymnasium as gym
import numpy as np
import torch

from fly_touhou.engineering_contract import canonical_sha256, sha256_file, write_json_new
from fly_touhou.geometry_observation import GeometryObservationEncoder
from fly_touhou.gru_imitation import (
    NATIVE_BOUNDS, actor_sequence_logits, configure_actor_training,
    constant_action_baseline, copy_bc_training_bundle, dataset_identity, enforce_epoch_budget,
    evaluate_actor, inspect_bc_training_bundle, load_dagger_demonstrations, load_demonstrations,
    masked_cross_entropy, pack_episodes, prefix_probe, read_epoch_budget_history,
    restore_bc_training_bundle, save_bc_training_bundle, state_subset_sha256,
    validate_dagger_aggregation, validate_teacher_qualification,
)
from fly_touhou.spatial_native_features import SpatialNativeFeatures
from fly_touhou.standard_gru_ppo import NativeGRUPolicy, capture_rng_state, restore_rng_state


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def build_policy(feature_names, seed, device, *, hidden_size=128):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    observation_space = gym.spaces.Box(-np.ones(1912, dtype=np.float32),
                                     np.ones(1912, dtype=np.float32), dtype=np.float32)
    policy = NativeGRUPolicy(observation_space, gym.spaces.Discrete(18), lambda _: 1e-3,
        lstm_hidden_size=hidden_size, features_extractor_class=SpatialNativeFeatures,
        features_extractor_kwargs={"feature_names": feature_names, "bounds": NATIVE_BOUNDS})
    return policy.to(device)


def train_epoch(policy, episodes, optimizer, shuffle_rng, *, batch_episodes,
                batch_callback=None, batch_guard=None):
    policy.set_training_mode(True)
    permutation = shuffle_rng.permutation(len(episodes))
    total_loss = 0.
    decisions = optimizer_steps = 0
    for offset in range(0, len(episodes), batch_episodes):
        if batch_guard is not None:
            batch_guard()
        selected = [episodes[int(index)] for index in permutation[offset:offset + batch_episodes]]
        packed = pack_episodes(selected, policy.device)
        logits = actor_sequence_logits(policy, packed["observations"])
        loss = masked_cross_entropy(logits, packed["actions"], packed["valid"])
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite BC loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad], .5,
                                      error_if_nonfinite=True)
        optimizer.step()
        n = int(packed["valid"].sum().item())
        total_loss += loss.item() * n
        decisions += n
        optimizer_steps += 1
        if batch_callback is not None:
            batch_callback(n)
    return {"training_decisions_this_epoch": decisions, "adam_steps_this_epoch": optimizer_steps,
            "online_batch_ce": total_loss / decisions}


def validate_inputs(args):
    base_train, train_meta = load_demonstrations(args.train_manifest, expected_role="train", expected_episodes=64)
    validation, val_meta = load_demonstrations(args.validation_manifest, expected_role="validation", expected_episodes=16)
    dagger_groups, dagger_meta = [], []
    for index, path in enumerate(args.dagger_manifest, 1):
        group, meta = load_dagger_demonstrations(path, expected_round=index)
        dagger_groups.append(group)
        dagger_meta.append(meta)
    split = validate_dagger_aggregation(base_train, validation, dagger_groups)
    train = base_train + [episode for group in dagger_groups for episode in group]
    for key in ("feature_names", "observation_version", "protocol_sha256"):
        if any(train_meta["manifest"].get(key) != meta["manifest"].get(key) for meta in [val_meta, *dagger_meta]):
            raise ValueError(f"Train/validation/DAgger protocol mismatch: {key}")
    qualification = validate_teacher_qualification(args.teacher_qualification)
    for meta in (train_meta, val_meta, *dagger_meta):
        declared = Path(meta["manifest"].get("qualification", "__missing__"))
        if not declared.is_absolute():
            declared = Path(meta["path"]).parent / declared
        if declared.resolve() != Path(qualification["path"]):
            raise ValueError("Demonstrations refer to a different teacher qualification")
        if meta in dagger_meta and meta["manifest"].get("qualification_sha256") != qualification["sha256"]:
            raise ValueError("DAgger teacher qualification content changed")
    protocol_path = Path(train_meta["manifest"]["base_protocol_path"])
    if not protocol_path.is_absolute():
        protocol_path = Path(train_meta["path"]).parent / protocol_path
    if sha256_file(protocol_path) != train_meta["manifest"]["protocol_sha256"]:
        raise ValueError("Demonstration base protocol changed")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    feature_names = list(GeometryObservationEncoder.from_specification(protocol["encoder"]).feature_names)
    feature_names.append("task_progress")
    if feature_names != train_meta["manifest"]["feature_names"]:
        raise ValueError("Manifest feature names disagree with the actual encoder protocol")
    return train, validation, {
        "train_manifest": {key: value for key, value in train_meta.items() if key != "manifest"},
        "validation_manifest": {key: value for key, value in val_meta.items() if key != "manifest"},
        "base_train_dataset_identity": dataset_identity(base_train),
        "train_dataset_identity": dataset_identity(train), "validation_dataset_identity": dataset_identity(validation),
        "dagger_manifests": [{"path": meta["path"], "sha256": meta["sha256"],
            "round_index": meta["manifest"]["round_index"],
            "student_checkpoint_sha256": meta["manifest"]["student_checkpoint_sha256"]} for meta in dagger_meta],
        "split": split, "qualification": qualification,
        "base_protocol": str(protocol_path.resolve()), "protocol_sha256": sha256_file(protocol_path),
        "observation_version": train_meta["manifest"]["observation_version"],
        "feature_names": feature_names,
        "NPZ_files": [{"path": episode.source_path, "sha256": sha256_file(episode.source_path)}
                      for episode in train + validation]}


def semantic_contract(args, data_spec, source_hashes):
    return {"schema": "ordinary-GRU-BC-semantic-contract-v1", "seed": args.seed,
        "hidden_size": args.hidden_size, "feature_names_sha256": canonical_sha256(data_spec["feature_names"]),
        "observation_version": data_spec["observation_version"], "bounds": list(NATIVE_BOUNDS),
        "task_start_stage_frame": 12,
        "base_train_manifest_sha256": data_spec["train_manifest"]["sha256"],
        "base_train_dataset_identity": data_spec["base_train_dataset_identity"],
        "validation_manifest_sha256": data_spec["validation_manifest"]["sha256"],
        "validation_dataset_identity": data_spec["validation_dataset_identity"],
        "qualification_sha256": data_spec["qualification"]["sha256"],
        "sensor_audit_sha256": data_spec["qualification"]["sensor_audit_sha256"],
        "protocol_sha256": data_spec["protocol_sha256"], "source_hashes": source_hashes,
        "library_versions": {name: importlib.metadata.version(name) for name in
                             ("torch", "numpy", "stable-baselines3", "sb3-contrib")},
        "loss": "hard_action_cross_entropy", "learning_rate": .001, "Adam_eps": 1e-5,
        "gradient_clip": .5, "batch_episodes": args.batch_episodes,
        "initial_probe_epochs": args.probe_epochs, "probe_decisions": 2048,
        "full_recurrent_prefix": True, "maximum_epochs_all_runs": 30}


def check_resume_dataset_extension(receipt, current_refs, checkpoint):
    previous = receipt["dagger_manifests"]
    if current_refs[:len(previous)] != previous or len(current_refs) - len(previous) not in (0, 1):
        raise ValueError("BC resume must retain prior DAgger data and append at most one new round")
    if len(current_refs) > len(previous) and current_refs[-1]["student_checkpoint_sha256"] != sha256_file(checkpoint):
        raise ValueError("The new DAgger round must have been controlled by the resumed student checkpoint")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--teacher-qualification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--probe-epochs", type=int, default=3)
    parser.add_argument("--batch-episodes", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, choices=(128, 256), default=128)
    parser.add_argument("--capacity-evidence", type=Path)
    parser.add_argument("--wall-cap-seconds", type=float, default=3600.)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--dagger-manifest", type=Path, action="append", default=[])
    parser.add_argument("--previous-training-results", type=Path, nargs="+", default=[])
    args = parser.parse_args()
    if (not 1 <= args.max_epochs <= 30 or not 1 <= args.probe_epochs <= 30
            or not 1 <= args.batch_episodes <= 8 or args.wall_cap_seconds <= 0):
        parser.error("Require total epochs <=30, a bounded prefix probe, and episode batch <=8")
    if not args.resume_checkpoint and args.probe_epochs >= args.max_epochs:
        parser.error("An initial BC run must include full-dataset epochs after its probe")
    if len(args.dagger_manifest) > 2 or (args.dagger_manifest and not args.resume_checkpoint):
        parser.error("DAgger permits at most two rounds and requires a resumed student")
    if bool(args.resume_checkpoint) != bool(args.previous_training_results):
        parser.error("Resume requires terminal previous-training-results; fresh runs have no prior results")
    if args.hidden_size == 256 and (not args.capacity_evidence or not args.capacity_evidence.is_file()):
        parser.error("GRU256 is a separately evidenced capacity comparison; supply --capacity-evidence")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    try:
        train, validation, data_spec = validate_inputs(args)
        source_hashes = {str(path.relative_to(ROOT)): sha256_file(path) for path in (
            Path(__file__), ROOT / "src/fly_touhou/gru_imitation.py",
            ROOT / "src/fly_touhou/standard_gru_ppo.py", ROOT / "src/fly_touhou/spatial_native_features.py")}
        contract = semantic_contract(args, data_spec, source_hashes)
        history = read_epoch_budget_history(args.previous_training_results)
        resume = inspect_bc_training_bundle(args.resume_checkpoint, semantic_contract=contract) if args.resume_checkpoint else None
        if resume is not None:
            check_resume_dataset_extension(resume, data_spec["dagger_manifests"], args.resume_checkpoint)
        budget = enforce_epoch_budget(history, additional_epochs=args.max_epochs, resume_receipt=resume)
    except BaseException:
        atomic_json(output / "failure.json", {"status": "input_validation_failed",
            "exception": traceback.format_exc(), "training_started": False,
            "PPO_admission": False})
        raise
    run_id = uuid.uuid4().hex
    config = {"schema": "whole-episode-gru-bc-v2", "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "validated_only" if args.validate_only else "configured_not_yet_trained",
        "dataset": data_spec, "seed": args.seed, "network": "SpatialNativeFeatures + true GRU",
        "hidden_size": args.hidden_size, "bounds": NATIVE_BOUNDS,
        "legacy_raw_coordinate_normalization_unchanged": True,
        "frames_per_action": 4, "task_start_stage_frame": 12,
        "device_required": "cuda", "loss": "hard_action_cross_entropy",
        "target18_soft_loss_used": False, "learning_rate": .001, "gradient_clip": .5,
        "max_epochs_including_probe": args.max_epochs, "probe_epochs": args.probe_epochs,
        "previous_training_results": history["requested_results"], "epoch_budget": budget,
        "resume_checkpoint": str(args.resume_checkpoint.resolve()) if args.resume_checkpoint else None,
        "resume_checkpoint_sha256": sha256_file(args.resume_checkpoint) if args.resume_checkpoint else None,
        "semantic_contract": contract,
        "probe_decisions": 2048, "batch_episodes": args.batch_episodes,
        "full_recurrent_prefix": True, "frame_independent_zero_hidden_states": False,
        "private_critic_training": False, "shared_features_training": True,
        "selection": "lowest_validation_CE_from_full_dataset_epochs_then_earlier_epoch",
        "closed_loop_qualification_proven": False, "PPO_admission": False,
        "source_hashes": source_hashes}
    write_json_new(output / "config.json", config)
    if args.validate_only:
        print(json.dumps({"status": "dataset_validation_only", "train_episodes": len(train),
                          "validation_episodes": len(validation), "training_started": False}))
        return
    if not torch.cuda.is_available():
        atomic_json(output / "failure.json", {"status": "CUDA_unavailable", "training_started": False,
            "PPO_admission": False, "silent_CPU_fallback": False})
        raise RuntimeError("BC requires CUDA; silent CPU fallback is forbidden")
    started = time.monotonic()
    torch.set_num_threads(2)
    policy = build_policy(data_spec["feature_names"], args.seed, "cuda", hidden_size=args.hidden_size)
    actor_parameters = configure_actor_training(policy)
    optimizer = torch.optim.Adam(actor_parameters, lr=.001, eps=1e-5)
    critic_initial = state_subset_sha256(policy, private_critic=True)
    actor_initial = state_subset_sha256(policy, private_critic=False)
    shuffle_rng = np.random.default_rng(args.seed + 10000)
    initial_model_epoch = optimizer_steps = decision_exposures = 0
    if args.resume_checkpoint:
        resume = restore_bc_training_bundle(policy, optimizer, shuffle_rng, args.resume_checkpoint,
            semantic_contract=contract, restore_global_rng=restore_rng_state)
        initial_model_epoch = resume["model_epoch"]
        optimizer_steps, decision_exposures = resume["adam_steps"], resume["decision_exposures"]
    actor_initial = state_subset_sha256(policy, private_critic=False)
    base_train = [episode for episode in train if episode.dataset_kind != "dagger"]
    probe = prefix_probe(base_train, decisions=2048, seed=args.seed)
    constant = constant_action_baseline(train)
    probe_constant = constant_action_baseline(probe)
    config.update(status="training", initial_private_critic_sha256=critic_initial,
                  initial_actor_sha256=actor_initial,
                  trainable_parameters=sum(parameter.numel() for parameter in actor_parameters),
                  gpu=torch.cuda.get_device_name(0),
                  trainable_parameter_names=[name for name, p in policy.named_parameters() if p.requires_grad],
                  constant_baseline=constant, probe_dataset_identity=dataset_identity(probe))
    atomic_json(output / "config.json", config)
    epochs_started = epochs_completed = 0
    initial_steps, initial_exposures = optimizer_steps, decision_exposures
    events_path = output / "epoch_events.jsonl"
    def budget_ledger():
        return [*history["budget_ledger"], {"run_id": run_id, "epochs_charged": epochs_started}]
    def runtime_record(status):
        return {"status": status, "run_id": run_id, "epochs_started_this_run": epochs_started,
            "epochs_completed_this_run": epochs_completed, "epochs_completed": epochs_completed,
            "cumulative_epochs_consumed": history["epochs_consumed"] + epochs_started,
            "budget_ledger": budget_ledger(), "previous_training_results": history["requested_results"],
            "epoch_events_path": str(events_path), "semantic_contract_sha256": canonical_sha256(contract),
            "adam_steps": optimizer_steps, "adam_steps_this_run": optimizer_steps - initial_steps,
            "decision_exposures": decision_exposures,
            "decision_exposures_this_run": decision_exposures - initial_exposures,
            "model_epochs_at_resume": initial_model_epoch,
            "model_epochs_completed": initial_model_epoch + epochs_completed,
            "wall_seconds": time.monotonic() - started, "PPO_admission": False,
            "closed_loop_qualification_proven": False, "game_process_started": False}
    def checkpoint_metadata(epoch, phase):
        return {"schema": "native-gru-bc-actor-state-v1", "epoch": epoch, "phase": phase,
            "seed": args.seed, "hidden_size": args.hidden_size, "observation_size": 1912,
            "task_start_stage_frame": 12,
            "observation_version": data_spec["observation_version"],
            "feature_names_sha256": hashlib.sha256(json.dumps(data_spec["feature_names"]).encode()).hexdigest(),
            "bounds": list(NATIVE_BOUNDS), "private_critic_unchanged": True,
            "BC_optimizer_steps": optimizer_steps, "new_PPO_Adam_required": True,
            "closed_loop_qualification_proven": False}
    def save_bundle(path, model_epoch, phase):
        rng_state = capture_rng_state()
        rng_state["named_generators"] = {"BC_episode_shuffle": shuffle_rng.bit_generator.state}
        return save_bc_training_bundle(policy, path, optimizer=optimizer, rng_state=rng_state,
            metadata=checkpoint_metadata(model_epoch, phase), semantic_contract=contract,
            dagger_refs=data_spec["dagger_manifests"], run_id=run_id, model_epoch=model_epoch,
            budget_ledger=budget_ledger(), adam_steps=optimizer_steps, decision_exposures=decision_exposures)
    def batch_completed(decisions):
        nonlocal optimizer_steps, decision_exposures
        optimizer_steps += 1
        decision_exposures += decisions
        atomic_json(output / "training_state.json", runtime_record("training"))
    def check_batch_budget():
        if time.monotonic() - started > args.wall_cap_seconds or (output / "STOP").exists():
            # The event ledger has already charged this epoch. Only its last
            # completed boundary bundle is resumable, never partial weights.
            raise TimeoutError("BC reached its wall cap or STOP between episode batches")
    save_bundle(output / "initial.pt", initial_model_epoch, "resumed_initial" if resume else "initial")
    initial_metrics = {"phase": "initial", "epoch": 0,
        "probe": evaluate_actor(policy, probe, batch_episodes=args.batch_episodes, constant_baseline=probe_constant),
        "train": evaluate_actor(policy, train, batch_episodes=args.batch_episodes, constant_baseline=constant),
        "validation": evaluate_actor(policy, validation, batch_episodes=args.batch_episodes, constant_baseline=constant)}
    write_json_new(output / "initial_metrics.json", initial_metrics)
    best = None
    last = None
    try:
        with (output / "curve.jsonl").open("x", encoding="utf-8", buffering=1) as curve, events_path.open("x", encoding="utf-8", buffering=1) as events:
            for epoch in range(1, args.max_epochs + 1):
                if time.monotonic() - started > args.wall_cap_seconds or (output / "STOP").exists():
                    break
                epochs_started += 1
                model_epoch = initial_model_epoch + epoch
                phase = "prefix_probe" if model_epoch <= args.probe_epochs else "full_dataset_BC"
                events.write(json.dumps({"kind": "epoch_started", "run_id": run_id,
                    "epoch_this_run": epoch, "model_epoch": model_epoch, "phase": phase}) + "\n")
                atomic_json(output / "training_state.json", runtime_record("training"))
                epoch_train = probe if phase == "prefix_probe" else train
                metrics = train_epoch(policy, epoch_train, optimizer, shuffle_rng,
                                      batch_episodes=args.batch_episodes, batch_callback=batch_completed,
                                      batch_guard=check_batch_budget)
                current_critic = state_subset_sha256(policy, private_critic=True)
                if current_critic != critic_initial:
                    raise ValueError("BC unexpectedly changed a private critic tensor")
                row = {"epoch": epoch, "model_epoch": model_epoch, "phase": phase, **metrics,
                    "adam_steps_cumulative": optimizer_steps, "decision_exposures_cumulative": decision_exposures,
                    "private_critic_unchanged": True, "wall_seconds": time.monotonic() - started,
                    "probe": evaluate_actor(policy, probe, batch_episodes=args.batch_episodes, constant_baseline=probe_constant),
                    "train": evaluate_actor(policy, train, batch_episodes=args.batch_episodes, constant_baseline=constant),
                    "validation": evaluate_actor(policy, validation, batch_episodes=args.batch_episodes, constant_baseline=constant)}
                checkpoint = output / f"bc_epoch_{epoch:03d}.pt"
                bundle = save_bundle(checkpoint, model_epoch, phase)
                row["checkpoint"] = str(checkpoint)
                row["checkpoint_sha256"] = sha256_file(checkpoint)
                row["resume_receipt_sha256"] = bundle["receipt_sha256"]
                curve.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                epochs_completed += 1
                events.write(json.dumps({"kind": "epoch_completed", "run_id": run_id,
                    "epoch_this_run": epoch, "model_epoch": model_epoch,
                    "checkpoint_sha256": row["checkpoint_sha256"]}) + "\n")
                atomic_json(output / "progress.json", row)
                atomic_json(output / "training_state.json", runtime_record("training"))
                print(json.dumps({key: row[key] for key in ("epoch", "phase", "adam_steps_cumulative", "wall_seconds", "probe", "validation")}))
                if phase == "full_dataset_BC" and (best is None or row["validation"]["ce"] < best["validation"]["ce"]):
                    best = row
                last = row
        if best is not None:
            selected = output / "bc_actor.pt"
            copy_bc_training_bundle(best["checkpoint"], selected)
            atomic_json(output / "selected_checkpoint.json", {
                "path": str(selected), "sha256": sha256_file(selected), "source_epoch": best["epoch"],
                "model_epoch": best["model_epoch"], "cumulative_epochs_consumed": history["epochs_consumed"] + epochs_started,
                "validation": best["validation"], "closed_loop_qualification_proven": False,
                "required_next_test": "20 argmax and 20 categorical sampling native episodes; both need success before PPO"})
        result = {**runtime_record("finished" if epochs_completed == args.max_epochs else "bounded_stop"),
            "private_critic_unchanged": state_subset_sha256(policy, private_critic=True) == critic_initial,
            "actor_parameters_changed": state_subset_sha256(policy, private_critic=False) != actor_initial,
            "selected_epoch": best["epoch"] if best else None,
            "last_epoch_metrics": last, "selected_checkpoint": str(output / "bc_actor.pt") if best else None}
        write_json_new(output / "result.json", result)
        atomic_json(output / "training_state.json", result)
    except BaseException:
        failure = {**runtime_record("failed"), "exception": traceback.format_exc(),
            "last_fully_saved_checkpoint": last["checkpoint"] if last else str(output / "initial.pt"),
            "partial_epoch_charged_in_full": epochs_started > epochs_completed}
        atomic_json(output / "failure.json", failure)
        atomic_json(output / "training_state.json", failure)
        raise


if __name__ == "__main__":
    main()
