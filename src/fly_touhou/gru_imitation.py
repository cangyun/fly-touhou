"""Whole-episode behaviour cloning for the ordinary GRU control baseline.

Native collection and closed-loop qualification live outside this module.
Offline CE improvement is not a claim of competent game control.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
from typing import Iterable, Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .engineering_contract import canonical_sha256, file_record, sha256_file, write_json_new


PRIVATE_CRITIC_PREFIXES = ("lstm_critic.", "mlp_extractor.value_net.", "value_net.")
ACTOR_PREFIXES = ("features_extractor.", "pi_features_extractor.", "vf_features_extractor.",
                  "lstm_actor.", "mlp_extractor.policy_net.", "action_net.")
NATIVE_BOUNDS = (-184., 184., 32., 432.)


@dataclass(frozen=True)
class DemonstrationEpisode:
    episode_id: str
    observations: np.ndarray
    actions: np.ndarray
    target18: np.ndarray
    trajectory_sha256: str
    source_path: str
    role: str
    danger: np.ndarray | None = None
    behavior_actions: np.ndarray | None = None
    dataset_kind: str = "teacher_demonstration"
    dagger_round: int = 0

    def __len__(self):
        return len(self.actions)


def load_demonstrations(manifest_path: str | Path, *, expected_role: str,
                        expected_episodes: int | None = None,
                        require_full_success: bool = True) -> tuple[list[DemonstrationEpisode], dict]:
    """Read trusted NPZ arrays without pickle and validate whole-episode data."""
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if expected_role not in ("train", "validation") or manifest.get("split") != expected_role:
        raise ValueError("The manifest split must match train/validation")
    if manifest.get("frames_per_action", 4) != 4:
        raise ValueError("Demonstrations must retain four native frames per action")
    source_rows = manifest.get("episodes", [])
    eligible_rows = [row for row in source_rows if row.get("eligible_successful_demonstration",
        row.get("success", row.get("horizon_completed_without_hit", False))) is True]
    excluded_rows = [row for row in source_rows if row not in eligible_rows]
    if expected_episodes is not None and len(eligible_rows) != expected_episodes:
        raise ValueError(f"Expected exactly {expected_episodes} successful {expected_role} episodes")
    if "successful_demonstrations" in manifest and manifest["successful_demonstrations"] != len(eligible_rows):
        raise ValueError("Successful demonstration count disagrees with retained episode rows")
    if "attempts" in manifest and manifest["attempts"] != len(source_rows):
        raise ValueError("Attempt count disagrees with the retained success/failure ledger")
    if expected_episodes in (64, 16) and len(source_rows) > int(np.ceil(expected_episodes * 1.25)):
        raise ValueError("Demonstration attempts exceeded the declared 80/20 collection cap")
    if manifest.get("task_start_stage_frame") != 12:
        raise ValueError("Demonstrations must freeze native task start at stage-frame 12")
    feature_names = manifest.get("feature_names")
    if (not isinstance(feature_names, list) or len(feature_names) != 1912
            or len(set(feature_names)) != 1912 or feature_names[-1] != "task_progress"):
        raise ValueError("Demonstrations need raw1911 plus trailing task_progress feature names")
    episodes = []
    episode_ids = set()
    fingerprints = set()
    repeated_fingerprints = []
    for row in eligible_rows if require_full_success else source_rows:
        if row.get("native_start_stage_frame") != 12:
            raise ValueError("A demonstration started outside the fixed native stage-frame 12")
        episode_id = str(row["episode_id"])
        fingerprint = row["trajectory_sha256"]
        if episode_id in episode_ids:
            raise ValueError("Duplicate demonstration episode identity")
        if fingerprint in fingerprints:
            repeated_fingerprints.append(fingerprint)
        episode_ids.add(episode_id)
        fingerprints.add(fingerprint)
        path = Path(row["dataset_path"])
        if not path.is_absolute():
            path = manifest_path.parent / path
        path = path.resolve()
        if row.get("dataset_sha256") is not None and sha256_file(path) != row["dataset_sha256"]:
            raise ValueError(f"Demonstration file changed: {path}")
        with np.load(path, allow_pickle=False) as archive:
            observations = archive["observations"]
            actions = archive["actions"]
            targets = archive["target18"]
            danger = archive["danger"].astype(bool) if "danger" in archive.files else None
        if (observations.dtype != np.float32 or observations.ndim != 2
                or observations.shape[1] != 1912 or not np.isfinite(observations).all()):
            raise ValueError(f"Invalid observation data: {path}")
        length = len(observations)
        if not 1 <= length <= 256 or actions.shape != (length,) or actions.dtype.kind not in "iu":
            raise ValueError("Require one contiguous episode of at most 256 action decisions")
        if (actions < 0).any() or (actions >= 18).any():
            raise ValueError("Teacher action is outside the 18 no-Bomb choices")
        if (targets.shape != (length, 18) or not np.isfinite(targets).all()
                or (targets < 0).any() or not np.allclose(targets.sum(1), 1., atol=1e-5)):
            raise ValueError("target18 must contain finite categorical distributions")
        expected_clock = np.arange(length, dtype=np.float32) * (4. / 1024.)
        if not np.allclose(observations[:, -1], expected_clock, atol=1e-6):
            raise ValueError("Task time must follow the full recurrent prefix from episode reset")
        if danger is not None and danger.shape != (length,):
            raise ValueError("Danger labels must have one value per decision")
        success = row.get("success", row.get("horizon_completed_without_hit"))
        if require_full_success and (length != 256 or row.get("frames") != 1024 or success is not True
                                     or row.get("eligible_successful_demonstration", True) is not True):
            raise ValueError("BC base demonstrations must be complete 1024-frame successful episodes")
        observed_fingerprint = hashlib.sha256()
        for observation, action in zip(observations, actions):
            observed_fingerprint.update(np.round(observation, 4).tobytes())
            observed_fingerprint.update(bytes([int(action)]))
        if observed_fingerprint.hexdigest() != fingerprint:
            raise ValueError("Recorded trajectory fingerprint differs from the actual NPZ sequence")
        episodes.append(DemonstrationEpisode(episode_id, observations, actions.astype(np.int64),
                         targets.astype(np.float32), str(fingerprint), str(path), expected_role, danger))
    if not episodes:
        raise ValueError("Empty demonstration dataset")
    return episodes, {"path": str(manifest_path), "sha256": sha256_file(manifest_path),
                      "manifest": manifest, "collection_attempts": len(source_rows),
                      "distinct_encoded_trajectories": len(fingerprints),
                      "repeated_training_trajectory_fingerprints": repeated_fingerprints,
                      "stochastic_independence_proven": False,
                      "excluded_unsuccessful_demonstrations": [{
                          "episode_id": row.get("episode_id"), "dataset_path": row.get("dataset_path"),
                          "frames": row.get("frames"), "reason": "retained_failed_collection_attempt"
                      } for row in excluded_rows]}


def validate_demonstration_split(train: Iterable[DemonstrationEpisode],
                                  validation: Iterable[DemonstrationEpisode]) -> dict:
    train, validation = list(train), list(validation)
    train_fingerprints = {episode.trajectory_sha256 for episode in train}
    validation_fingerprints = {episode.trajectory_sha256 for episode in validation}
    overlap = train_fingerprints & validation_fingerprints
    train_files = {Path(episode.source_path).resolve() for episode in train}
    validation_files = {Path(episode.source_path).resolve() for episode in validation}
    file_overlap = train_files & validation_files
    if overlap or file_overlap:
        raise ValueError("Whole trajectories leak between training and validation")
    return {"split_unit": "whole_native_episode", "train_episodes": len(train),
            "validation_episodes": len(validation), "cross_split_fingerprint_overlap": 0,
            "cross_split_file_overlap": 0, "stochastic_independence_proven": False}


def pack_episodes(episodes: Iterable[DemonstrationEpisode], device: torch.device | str) -> dict:
    episodes = list(episodes)
    if not episodes:
        raise ValueError("Cannot batch an empty episode list")
    lengths = [len(episode) for episode in episodes]
    shape = (len(episodes), max(lengths))
    observations = np.zeros((*shape, episodes[0].observations.shape[1]), dtype=np.float32)
    actions = np.zeros(shape, dtype=np.int64)
    valid = np.zeros(shape, dtype=bool)
    danger = np.zeros(shape, dtype=bool)
    danger_known = np.zeros(shape, dtype=bool)
    for batch, episode in enumerate(episodes):
        observations[batch, :len(episode)] = episode.observations
        actions[batch, :len(episode)] = episode.actions
        valid[batch, :len(episode)] = True
        if episode.danger is not None:
            danger[batch, :len(episode)] = episode.danger
            danger_known[batch, :len(episode)] = True
    return {"observations": torch.as_tensor(observations, device=device),
            "actions": torch.as_tensor(actions, device=device),
            "valid": torch.as_tensor(valid, device=device),
            "danger": torch.as_tensor(danger, device=device),
            "danger_known": torch.as_tensor(danger_known, device=device),
            "lengths": lengths}


def actor_sequence_logits(policy, observations: torch.Tensor) -> torch.Tensor:
    """One zero state per full episode, never one zero state per frame."""
    if observations.ndim != 3:
        raise ValueError("Expected [episode, time, feature] observations")
    batch, time, dimensions = observations.shape
    if not policy.share_features_extractor:
        raise ValueError("BC/PPO contract requires shared spatial features")
    features = policy.extract_features(observations.reshape(batch * time, dimensions))
    features = features.reshape(batch, time, policy.features_dim).transpose(0, 1)
    hidden = features.new_zeros((policy.lstm_actor.num_layers, batch, policy.lstm_actor.hidden_size))
    latent, _ = policy.lstm_actor(features, (hidden, torch.zeros_like(hidden)))
    latent = policy.mlp_extractor.forward_actor(latent)
    return policy.action_net(latent).transpose(0, 1)


def masked_cross_entropy(logits: torch.Tensor, actions: torch.Tensor,
                          valid: torch.Tensor) -> torch.Tensor:
    if logits.shape[:-1] != actions.shape or actions.shape != valid.shape or not valid.any():
        raise ValueError("CE requires matching nonempty sequence masks")
    return F.cross_entropy(logits[valid], actions[valid])


def configure_actor_training(policy) -> list[nn.Parameter]:
    parameters = []
    for name, parameter in policy.named_parameters():
        actor = name.startswith(ACTOR_PREFIXES) and not name.startswith(PRIVATE_CRITIC_PREFIXES)
        parameter.requires_grad_(actor)
        if actor:
            parameters.append(parameter)
    if not parameters:
        raise ValueError("No actor parameters found")
    return parameters


def state_subset_sha256(policy, *, private_critic: bool) -> str:
    digest = hashlib.sha256()
    selected = 0
    for name, tensor in sorted(policy.state_dict().items()):
        if name.startswith(PRIVATE_CRITIC_PREFIXES) != private_critic:
            continue
        selected += 1
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    if not selected:
        raise ValueError("The requested policy subset was empty")
    return digest.hexdigest()


def constant_action_baseline(episodes: Iterable[DemonstrationEpisode], *, smoothing: float = 1.) -> dict:
    if smoothing <= 0:
        raise ValueError("Positive declared smoothing prevents unseen-label infinite CE")
    actions = np.concatenate([episode.actions for episode in episodes])
    counts = np.bincount(actions, minlength=18)
    probabilities = (counts + smoothing) / (len(actions) + 18 * smoothing)
    return {"counts": counts.tolist(), "probabilities": probabilities.tolist(),
            "smoothing": smoothing, "argmax_action": int(np.argmax(probabilities)),
            "training_decisions": len(actions)}


def evaluate_actor(policy, episodes: list[DemonstrationEpisode], *, batch_episodes: int,
                   constant_baseline: Mapping) -> dict:
    was_training = policy.training
    policy.set_training_mode(False)
    device = policy.device
    counts = 0
    ce_sum = correct = entropy_sum = constant_ce_sum = constant_correct = 0.
    danger_count = danger_ce_sum = danger_correct = known_danger_count = 0.
    probabilities = np.asarray(constant_baseline["probabilities"])
    with torch.no_grad():
        for offset in range(0, len(episodes), batch_episodes):
            group = episodes[offset:offset + batch_episodes]
            packed = pack_episodes(group, device)
            logits = actor_sequence_logits(policy, packed["observations"])
            valid = packed["valid"]
            labels = packed["actions"][valid]
            selected = logits[valid]
            n = len(labels)
            ce_sum += F.cross_entropy(selected, labels, reduction="sum").item()
            predictions = selected.argmax(-1)
            correct += (predictions == labels).sum().item()
            log_prob = selected.log_softmax(-1)
            entropy_sum += -(log_prob.exp() * log_prob).sum().item()
            labels_cpu = labels.cpu().numpy()
            constant_ce_sum += float(-np.log(probabilities[labels_cpu]).sum())
            constant_correct += int((labels_cpu == constant_baseline["argmax_action"]).sum())
            counts += n
            known = packed["danger_known"] & valid
            danger = known & packed["danger"]
            known_danger_count += int(known.sum().item())
            if danger.any():
                danger_logits, danger_labels = logits[danger], packed["actions"][danger]
                danger_count += len(danger_labels)
                danger_ce_sum += F.cross_entropy(danger_logits, danger_labels, reduction="sum").item()
                danger_correct += (danger_logits.argmax(-1) == danger_labels).sum().item()
    policy.set_training_mode(was_training)
    return {"decisions": counts, "episodes": len(episodes), "ce": ce_sum / counts,
            "accuracy": correct / counts, "mean_entropy": entropy_sum / counts,
            "constant_ce": constant_ce_sum / counts, "constant_accuracy": constant_correct / counts,
            "danger_labels_available": known_danger_count > 0,
            "danger_labeled_decisions": int(known_danger_count), "danger_decisions": int(danger_count),
            "danger_ce": danger_ce_sum / danger_count if danger_count else None,
            "danger_accuracy": danger_correct / danger_count if danger_count else None,
            "closed_loop_success_proven": False}


def prefix_probe(episodes: list[DemonstrationEpisode], *, decisions: int = 2048,
                 seed: int = 11) -> list[DemonstrationEpisode]:
    """Select full prefixes, so a selected frame always has its recurrent past."""
    if sum(map(len, episodes)) < decisions or decisions <= 0:
        raise ValueError("Insufficient demonstration decisions for the recurrent prefix probe")
    order = np.random.default_rng(seed).permutation(len(episodes))
    selected = []
    remaining = decisions
    for index in order:
        episode = episodes[int(index)]
        count = min(len(episode), remaining)
        selected.append(DemonstrationEpisode(
            episode.episode_id, episode.observations[:count].copy(), episode.actions[:count].copy(),
            episode.target18[:count].copy(), episode.trajectory_sha256, episode.source_path,
            episode.role, episode.danger[:count].copy() if episode.danger is not None else None))
        remaining -= count
        if not remaining:
            break
    return selected


def save_bc_checkpoint(policy, path: str | Path, *, metadata: dict) -> None:
    """CPU tensor state_dict is loadable with torch.load(weights_only=True)."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    torch.save({"state_dict": {name: tensor.detach().cpu().clone()
                              for name, tensor in policy.state_dict().items()},
                "metadata": metadata}, path)


def dataset_identity(episodes: list[DemonstrationEpisode]) -> str:
    return canonical_sha256([{"episode_id": episode.episode_id,
                              "trajectory_sha256": episode.trajectory_sha256,
                              "decisions": len(episode)} for episode in episodes])


def encoded_trajectory_sha256(observations: np.ndarray, actions: np.ndarray) -> str:
    digest = hashlib.sha256()
    for observation, action in zip(observations, actions):
        digest.update(np.round(observation, 4).tobytes())
        digest.update(bytes([int(action)]))
    return digest.hexdigest()


def load_dagger_demonstrations(manifest_path: str | Path, *, expected_round: int) -> tuple[list[DemonstrationEpisode], dict]:
    """All 20 student episodes count, including failures and short prefixes."""
    episodes, meta = load_demonstrations(manifest_path, expected_role="train", require_full_success=False)
    manifest = meta["manifest"]
    if (manifest.get("schema") != "native-shared-teacher-dagger-v1"
            or manifest.get("dataset_kind") != "dagger" or manifest.get("round_index") != expected_round
            or expected_round not in (1, 2) or len(episodes) != 20
            or not isinstance(manifest.get("student_checkpoint_sha256"), str)):
        raise ValueError("DAgger requires exactly 20 student episodes in round 1 or 2")
    enriched = []
    for episode, row in zip(episodes, manifest["episodes"]):
        with np.load(episode.source_path, allow_pickle=False) as archive:
            behavior = archive["executed_actions"]
            named_behavior = archive["behavior_actions"]
        if (behavior.shape != (len(episode),) or behavior.dtype.kind not in "iu"
                or (behavior < 0).any() or (behavior >= 18).any()
                or not np.array_equal(behavior, named_behavior)):
            raise ValueError("DAgger executed/behavior actions must record the same student choices")
        if row.get("frames") != len(episode) * 4:
            raise ValueError("DAgger must retain the complete native prefix from reset")
        if row.get("teacher_label_trajectory_sha256") != episode.trajectory_sha256:
            raise ValueError("DAgger teacher labels disagree with the recorded label fingerprint")
        if encoded_trajectory_sha256(episode.observations, behavior) != row.get("behavior_trajectory_sha256"):
            raise ValueError("DAgger behavior fingerprint differs from actual student actions")
        enriched.append(DemonstrationEpisode(episode.episode_id, episode.observations, episode.actions,
            episode.target18, episode.trajectory_sha256, episode.source_path, episode.role,
            episode.danger, behavior.astype(np.int64), "dagger", expected_round))
    meta["excluded_unsuccessful_demonstrations"] = []
    meta["short_failed_prefixes_retained"] = sum(len(episode) < 256 for episode in enriched)
    return enriched, meta


def validate_dagger_aggregation(base_train: list[DemonstrationEpisode], validation: list[DemonstrationEpisode],
                                 dagger_groups: list[list[DemonstrationEpisode]]) -> dict:
    if len(dagger_groups) > 2 or any(len(group) != 20 for group in dagger_groups):
        raise ValueError("At most two DAgger rounds, each containing exactly 20 student episodes")
    combined = base_train + [episode for group in dagger_groups for episode in group]
    ids = [episode.episode_id for episode in combined]
    if len(set(ids)) != len(ids):
        raise ValueError("Aggregated demonstration episode IDs must remain unique")
    report = validate_demonstration_split(combined, validation)
    # Label changes cannot disguise reuse of the same observations as validation.
    validation_observations = {hashlib.sha256(episode.observations.tobytes()).hexdigest() for episode in validation}
    if any(hashlib.sha256(episode.observations.tobytes()).hexdigest() in validation_observations for episode in combined):
        raise ValueError("DAgger/base observation sequence leaks into validation")
    return {**report, "base_training_episodes": len(base_train), "dagger_rounds": len(dagger_groups),
            "dagger_episodes": sum(map(len, dagger_groups)),
            "dagger_failed_or_short_prefixes_retained": sum(len(episode) < 256 for group in dagger_groups for episode in group)}


def save_bc_training_bundle(policy, checkpoint: str | Path, *, optimizer, rng_state: dict,
                              metadata: dict, semantic_contract: dict, dagger_refs: list[dict],
                              run_id: str, model_epoch: int, budget_ledger: list[dict],
                              adam_steps: int, decision_exposures: int) -> dict:
    """Commit matching actor/Adam/RNG states; the SHA receipt is written last."""
    checkpoint = Path(checkpoint).resolve()
    paths = {"checkpoint": checkpoint, "optimizer": checkpoint.with_suffix(".optimizer.pt"),
             "rng": checkpoint.with_suffix(".rng.pt")}
    receipt_path = checkpoint.with_suffix(".resume.json")
    if any(path.exists() for path in [*paths.values(), receipt_path]):
        raise FileExistsError("Refusing to overwrite a BC training bundle")
    if "BC_episode_shuffle" not in rng_state.get("named_generators", {}):
        raise ValueError("BC resume needs the named episode-shuffle generator")
    private_sha = state_subset_sha256(policy, private_critic=True)
    save_bc_checkpoint(policy, checkpoint, metadata={**metadata, "run_id": run_id,
        "model_epoch": int(model_epoch), "private_critic_sha256": private_sha,
        "semantic_contract_sha256": canonical_sha256(semantic_contract)})
    torch.save(optimizer.state_dict(), paths["optimizer"])
    torch.save(rng_state, paths["rng"])
    receipt = {"schema": "GRU-BC-resume-bundle-v1", "run_id": run_id,
        "files": {name: file_record(path) for name, path in paths.items()},
        "semantic_contract": semantic_contract,
        "semantic_contract_sha256": canonical_sha256(semantic_contract),
        "dagger_manifests": dagger_refs, "model_epoch": int(model_epoch),
        "budget_ledger_at_save": budget_ledger, "adam_steps": int(adam_steps),
        "decision_exposures": int(decision_exposures), "at_epoch_boundary": True,
        "private_critic_sha256": private_sha,
        "trainable_parameter_names": [name for name, value in policy.named_parameters() if value.requires_grad]}
    write_json_new(receipt_path, receipt)
    return {**receipt, "receipt_path": str(receipt_path), "receipt_sha256": sha256_file(receipt_path)}


def inspect_bc_training_bundle(checkpoint: str | Path, *, semantic_contract: dict) -> dict:
    checkpoint = Path(checkpoint).resolve()
    receipt_path = checkpoint.with_suffix(".resume.json")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (receipt.get("schema") != "GRU-BC-resume-bundle-v1"
            or receipt.get("at_epoch_boundary") is not True
            or set(receipt.get("files", {})) != {"checkpoint", "optimizer", "rng"}
            or receipt.get("semantic_contract_sha256") != canonical_sha256(semantic_contract)
            or receipt.get("semantic_contract") != semantic_contract):
        raise ValueError("BC resume semantic contract changed or boundary receipt is missing")
    if Path(receipt["files"]["checkpoint"]["path"]).resolve() != checkpoint:
        raise ValueError("Resume receipt points to a different checkpoint")
    for record in receipt["files"].values():
        path = Path(record["path"])
        if (not path.is_file() or sha256_file(path) != record["sha256"]
                or path.stat().st_size != record["bytes"]):
            raise ValueError("A BC checkpoint/optimizer/RNG artifact changed")
    return {**receipt, "receipt_path": str(receipt_path), "receipt_sha256": sha256_file(receipt_path)}


def restore_bc_training_bundle(policy, optimizer, shuffle_rng, checkpoint: str | Path, *,
                                semantic_contract: dict, restore_global_rng) -> dict:
    receipt = inspect_bc_training_bundle(checkpoint, semantic_contract=semantic_contract)
    if state_subset_sha256(policy, private_critic=True) != receipt["private_critic_sha256"]:
        raise ValueError("BC resume must reconstruct the same-seed untrained private critic")
    current_names = [name for name, value in policy.named_parameters() if value.requires_grad]
    if current_names != receipt["trainable_parameter_names"]:
        raise ValueError("BC resume trainable parameter ordering changed")
    weights = torch.load(receipt["files"]["checkpoint"]["path"], map_location="cpu", weights_only=True)
    metadata = weights.get("metadata", {})
    if (metadata.get("run_id") != receipt["run_id"] or metadata.get("model_epoch") != receipt["model_epoch"]
            or metadata.get("semantic_contract_sha256") != receipt["semantic_contract_sha256"]):
        raise ValueError("BC checkpoint metadata disagrees with its resume receipt")
    policy.load_state_dict(weights["state_dict"], strict=True)
    if state_subset_sha256(policy, private_critic=True) != receipt["private_critic_sha256"]:
        raise ValueError("Saved BC checkpoint changed the private critic")
    optimizer.load_state_dict(torch.load(receipt["files"]["optimizer"]["path"], map_location="cpu", weights_only=True))
    actual_steps = max((int(value.get("step", 0)) for value in optimizer.state.values()), default=0)
    if actual_steps != receipt["adam_steps"]:
        raise ValueError("BC optimizer actual step counter disagrees with its receipt")
    for group in optimizer.param_groups:
        if ("learning_rate" in semantic_contract and group["lr"] != semantic_contract["learning_rate"]
                or "Adam_eps" in semantic_contract and group["eps"] != semantic_contract["Adam_eps"]):
            raise ValueError("BC optimizer settings disagree with the frozen contract")
    state = torch.load(receipt["files"]["rng"]["path"], map_location="cpu", weights_only=True)
    if policy.device.type == "cuda" and len(state.get("torch_cuda", [])) != torch.cuda.device_count():
        raise ValueError("BC resume CUDA RNG device count changed")
    shuffle_state = state["named_generators"]["BC_episode_shuffle"]
    if shuffle_state["bit_generator"] != type(shuffle_rng.bit_generator).__name__:
        raise ValueError("BC shuffle generator algorithm changed")
    # Restore last, after policy/Adam deserialization has completed.
    restore_global_rng(state)
    shuffle_rng.bit_generator.state = shuffle_state
    return receipt


def copy_bc_training_bundle(source: str | Path, destination: str | Path) -> dict:
    source, destination = Path(source).resolve(), Path(destination).resolve()
    receipt = json.loads(source.with_suffix(".resume.json").read_text(encoding="utf-8"))
    destinations = {"checkpoint": destination, "optimizer": destination.with_suffix(".optimizer.pt"),
                    "rng": destination.with_suffix(".rng.pt")}
    if any(path.exists() for path in [*destinations.values(), destination.with_suffix(".resume.json")]):
        raise FileExistsError("Selected BC bundle already exists")
    for name, target in destinations.items():
        original = receipt["files"][name]
        if sha256_file(original["path"]) != original["sha256"]:
            raise ValueError("Selected BC bundle source changed")
        shutil.copyfile(original["path"], target)
    receipt["files"] = {name: file_record(path) for name, path in destinations.items()}
    receipt["selected_copy_of"] = file_record(source.with_suffix(".resume.json"))
    write_json_new(destination.with_suffix(".resume.json"), receipt)
    return receipt


def read_epoch_budget_history(result_paths: Iterable[str | Path]) -> dict:
    """Charge every begun epoch in every linked run, even discarded weights."""
    visited: dict[Path, set[str]] = {}
    visiting: set[Path] = set()
    ledger: dict[str, int] = {}
    records: list[dict] = []

    def visit(path):
        path = Path(path).resolve()
        if path in visited:
            return visited[path]
        if path in visiting:
            raise ValueError("Previous training result ancestry contains a cycle")
        visiting.add(path)
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("status") not in ("finished", "bounded_stop", "failed"):
            raise ValueError("Previous BC runs must have terminal state evidence")
        run_id = report.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("Previous BC run lacks a run identity")
        ancestor_ids: set[str] = set()
        for previous in report.get("previous_training_results", []):
            if sha256_file(previous["path"]) != previous["sha256"]:
                raise ValueError("An earlier training result changed")
            ancestor_ids.update(visit(previous["path"]))
        events_path = Path(report["epoch_events_path"])
        events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if any(row.get("run_id") != run_id for row in events):
            raise ValueError("Epoch events mix different training runs")
        started = [row["epoch_this_run"] for row in events if row.get("kind") == "epoch_started"]
        completed = [row["epoch_this_run"] for row in events if row.get("kind") == "epoch_completed"]
        if (started != list(range(1, len(started) + 1)) or len(set(completed)) != len(completed)
                or not set(completed) <= set(started)
                or report.get("epochs_started_this_run") != len(started)
                or report.get("epochs_completed_this_run") != len(completed)):
            raise ValueError("Previous BC epoch accounting disagrees with the actual event log")
        if run_id in ledger and ledger[run_id] != len(started):
            raise ValueError("Conflicting actual epoch cost for the same run")
        ledger[run_id] = len(started)
        declared = {item["run_id"]: item["epochs_charged"] for item in report["budget_ledger"]}
        if any(key not in ledger or ledger[key] != value for key, value in declared.items()):
            raise ValueError("Declared training costs lack matching ancestor evidence")
        if set(declared) != ancestor_ids | {run_id}:
            raise ValueError("Previous BC run or ancestor is absent from its declared costs")
        if report.get("cumulative_epochs_consumed") != sum(declared.values()):
            raise ValueError("Cumulative BC epoch charge disagrees with the lineage ledger")
        records.append(file_record(path))
        visiting.remove(path)
        visited[path] = ancestor_ids | {run_id}
        return visited[path]

    roots = [Path(path).resolve() for path in result_paths]
    for path in roots:
        visit(path)
    return {"budget_ledger": [{"run_id": key, "epochs_charged": value} for key, value in sorted(ledger.items())],
            "epochs_consumed": sum(ledger.values()), "evidence": records,
            "requested_results": [file_record(path) for path in roots]}


def enforce_epoch_budget(history: Mapping, *, additional_epochs: int, resume_receipt: Mapping | None = None) -> dict:
    if not isinstance(additional_epochs, int) or not 1 <= additional_epochs <= 30:
        raise ValueError("An invocation may request 1 to 30 BC epochs")
    spent = int(history["epochs_consumed"])
    if spent + additional_epochs > 30:
        raise ValueError("Actual prior epoch costs plus this invocation exceed the total 30-epoch budget")
    if resume_receipt is not None:
        observed = {row["run_id"]: row["epochs_charged"] for row in history["budget_ledger"]}
        if resume_receipt["run_id"] not in observed:
            raise ValueError("Resume source run is missing from actual prior training costs")
        for row in resume_receipt["budget_ledger_at_save"]:
            if observed.get(row["run_id"], -1) < row["epochs_charged"]:
                raise ValueError("A saved checkpoint cannot reduce previously consumed epoch costs")
    return {"previous_epochs_consumed": spent, "additional_epochs_requested": additional_epochs,
            "maximum_cumulative_epochs_after_run": spent + additional_epochs,
            "maximum_total_epochs": 30}


def validate_teacher_qualification(path: str | Path) -> dict:
    """Check the completed M2 report and linked sensor audit before BC."""
    path = Path(path).resolve()
    report = json.loads(path.read_text(encoding="utf-8"))
    rows = report.get("episode_results", [])
    if (report.get("schema") != "native-shared-teacher-qualification-v1"
            or report.get("teacher_qualified") is not True
            or report.get("episodes") != 20 or len(rows) != 20
            or report.get("observation_size") != 1912 or report.get("frame_skip") != 4
            or report.get("shared_information") is not True
            or report.get("critical_sensor_issues_resolved") is not True
            or report.get("bomb_actions") != 0 or report.get("safety_takeovers") != 0
            or report.get("desktop_input") is not False):
        raise ValueError("M2 teacher/common-information qualification has not passed")
    if report.get("task_start_stage_frame") != 12 or any(row.get("native_start_stage_frame") != 12 for row in rows):
        raise ValueError("Teacher qualification must use fixed native stage-frame 12 in every episode")
    successes = sum(row.get("horizon_completed_without_hit") is True for row in rows)
    if successes < 19 or successes != report.get("successes"):
        raise ValueError("Teacher must complete at least 19 of exactly 20 recorded new games")
    for row in rows:
        if row.get("bomb_actions") != 0 or row.get("safety_takeovers") != 0:
            raise ValueError("Qualification episode used prohibited assistance")
        if row.get("horizon_completed_without_hit") and row.get("frames") != 1024:
            raise ValueError("Qualification success is not the fixed full task")
    sensor_path = Path(report.get("sensor_audit_report", "__missing__"))
    if not sensor_path.is_absolute():
        sensor_path = path.parent / sensor_path
    sensor = json.loads(sensor_path.read_text(encoding="utf-8"))
    counts = sensor.get("classification_counts", {})
    windows = sensor.get("windows", [])
    distinct = {window.get("fingerprint") for window in windows if window.get("fingerprint")}
    if (sensor.get("at_least20_distinct_windows") is not True
            or sensor.get("distinct_geometric_windows", 0) < 20
            or len(distinct) < 20 or len(distinct) != sensor.get("distinct_geometric_windows")
            or counts.get("potential_critical_object_omission", 0)
            or counts.get("recorded_observation_differs_from_native_snapshot", 0)
            or sensor.get("unresolved_contact_windows", 0)
            or any(window.get("classification") != "covered_geometry_policy_or_timing_failure"
                   or window.get("terminal_geometry", {}).get("critical_omission")
                   or window.get("terminal_geometry", {}).get("contact_candidate_count", 0) == 0
                   or window.get("terminal_geometry", {}).get("live_lasers_require_separate_geometry_review", 0)
                   for window in windows)):
        raise ValueError("Linked sensor audit still contains critical or unresolved windows")
    return {"path": str(path), "sha256": sha256_file(path), "successes": successes,
            "episodes": 20, "sensor_audit_path": str(sensor_path.resolve()),
            "sensor_audit_sha256": sha256_file(sensor_path),
            "runtime_results_rechecked_here": False}
