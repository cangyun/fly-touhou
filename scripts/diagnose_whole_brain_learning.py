"""Replay-overfit diagnostic on 32 real Lunatic decisions through the full brain.

This deliberately evaluates the same training fragment. It tests optimization
and spike-path execution; it does not estimate held-out or native-game skill.
No game is launched and no input is sent by this program.
"""

from pathlib import Path
import argparse
import hashlib
import json
import math
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fly_touhou.connectome import file_sha256, load_connectome  # noqa: E402
from fly_touhou.experiment import ExperimentLogger, NeuralActivityAccumulator  # noqa: E402
from fly_touhou.whole_brain import ShiuParameters, WholeBrainStudent  # noqa: E402
from fly_touhou.whole_brain_learning import (  # noqa: E402
    FORMAL_GAME_SHA256, N_ACTIONS, PreparedEpisode, _Metrics, _clip_gradients_fp64,
    _evaluate, _prime_state, _save_checkpoint, _write_json, graph_fingerprint,
    load_campaign_demonstrations, mixed_distillation_loss, split_campaign_episodes,
)


def select_fragment(episodes, length=32, minimum_actions=4):
    """Select current-only contiguous labels from an already reserved train set."""
    if length < 1 or not 1 <= minimum_actions <= min(N_ACTIONS, length):
        raise ValueError("Invalid diagnostic fragment length/action diversity")
    for episode in episodes:
        if int(episode.native_context["difficulty"]) != 3:
            continue
        labels = episode.probabilities.argmax(-1)
        for first in range(len(episode) - length + 1):
            last = first + length
            if len(np.unique(labels[first:last])) < minimum_actions:
                continue
            context = {**episode.native_context, "diagnostic_source_episode_id": episode.episode_id,
                       "diagnostic_source_steps": [first, last],
                       "diagnostic_fragment_is_not_a_complete_campaign": True}
            return PreparedEpisode(
                f"{episode.episode_id}/steps-{first}-{last - 1}",
                episode.observations[first:last].copy(), episode.probabilities[first:last].copy(),
                episode.actions[first:last].copy(), context, episode.source_path,
                # Exact elapsed native frames are not retained per step by the
                # prepared CPU dataset. Do not invent length*frame_skip here.
                native_frames=0,
            )
    raise ValueError("No reserved Lunatic training fragment meets the requested contiguous action diversity")


def parameter_hashes(model):
    result = {}
    for name, value in model.named_parameters():
        array = value.detach().cpu().contiguous().numpy()
        digest = hashlib.sha256()
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
        result[name] = digest.hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--game-exe", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, default=ROOT / "data" / "connectome")
    parser.add_argument("--ports", type=Path, default=ROOT / "data" / "connectome" / "annotations" / "selected_ports_1024_v783.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--length", type=int, default=32)
    parser.add_argument("--minimum-actions", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-decisions", type=int, default=4)
    parser.add_argument("--warmup-ms", type=float, default=50.0)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    options = parser.parse_args()
    if options.threads < 1 or options.epochs < 1 or options.chunk_decisions < 1:
        parser.error("Thread/epoch/chunk counts must be positive")
    if (not math.isfinite(options.learning_rate) or options.learning_rate <= 0
            or not math.isfinite(options.warmup_ms) or options.warmup_ms < 0):
        parser.error("Use a positive finite learning rate and nonnegative finite warmup")
    if file_sha256(options.game_exe) != FORMAL_GAME_SHA256:
        parser.error("Diagnostic requires the actual supplied formal TH14 executable identity")
    torch.set_num_threads(options.threads)
    torch.manual_seed(options.seed)
    if str(options.device).startswith("cuda"):
        torch.cuda.manual_seed_all(options.seed)
    output = options.output.resolve()
    source = options.source_dir.resolve()
    config = {"experiment_kind": "same-training-fragment-replay-overfit-diagnostic",
        "diagnostic_only": True, "game_exe_sha256": FORMAL_GAME_SHA256,
        "epochs": options.epochs, "length_decisions": options.length,
        "minimum_teacher_argmax_classes": options.minimum_actions,
        "fragment_selection": "first qualifying continuous window in deterministic reserved Lunatic training episodes",
        "seed": options.seed, "torch_num_threads": options.threads,
        "learning_rate": options.learning_rate, "chunk_decisions": options.chunk_decisions,
        "warmup_ms": options.warmup_ms, "substeps": 25, "dt_ms": 0.2,
        "neural_ms_per_decision": 5.0, "encoder_init_scale": 1.0,
        "hard_weight": 0.5, "gradient_clip": options.gradient_clip,
        "scope": "optimization on the exact same measured training fragment; no held-out or native campaign success claimed"}
    files = {"diagnostic_script": Path(__file__), "formal_game_executable": options.game_exe,
             "trace": options.trace, "ports_manifest": options.ports,
             "source_completeness": source / "Completeness_783.csv",
             "source_connectivity": source / "Connectivity_783.parquet",
             "source_model": source / "model.py"}
    for name in ("whole_brain", "whole_brain_learning", "connectome", "observation", "state", "planner", "experiment"):
        files[f"code_{name}"] = ROOT / "src" / "fly_touhou" / f"{name}.py"
    for name in ("flywire_annotations_v2.1.0.tsv", "sources_manifest.json"):
        path = source / "annotations" / name
        if path.exists():
            files[f"annotation_{name}"] = path
    logger = ExperimentLogger(output, config, files=files)
    try:
        logger.record_runtime()
        with logger.phase("diagnostic_select_original_lunatic_training_fragment"):
            dataset = load_campaign_demonstrations(options.trace)
            train, validation = split_campaign_episodes(dataset.episodes, seed=options.seed)
            fragment = select_fragment(train, options.length, options.minimum_actions)
            train_ids, validation_ids = [e.episode_id for e in train], [e.episode_id for e in validation]
            logger.record_split(train_ids, validation_ids)
            logger.record_curriculum("Lunatic fragment diagnostic", difficulty=3,
                stages=fragment.native_context.get("stages_observed", [1]),
                description="Same 32-decision training fragment replay; validation episodes reserved and unused")
            selected = {"source_episode_id": fragment.native_context["diagnostic_source_episode_id"],
                        "source_steps_start_inclusive_stop_exclusive": fragment.native_context["diagnostic_source_steps"],
                        "teacher_argmax_classes": np.unique(fragment.probabilities.argmax(-1)).tolist(),
                        "argmax_counts": np.bincount(fragment.probabilities.argmax(-1), minlength=N_ACTIONS).tolist(),
                        "decisions": len(fragment), "native_frames_for_fragment": None,
                        "encoder": dataset.encoder.specification(), "sources": dataset.sources}
            logger.event("diagnostic_fragment_selected", selected)
            _write_json(output / "fragment.json", selected)
        with logger.phase("diagnostic_load_and_verify_actual_complete_graph"):
            graph = load_connectome(source, max_neurons=None)
            ports = json.loads(options.ports.read_text(encoding="utf-8"))
            model = WholeBrainStudent(dataset.encoder.size, N_ACTIONS, graph,
                ports["input_indices"], ports["output_indices"], device=options.device,
                substeps=25, dt_ms=0.2, encoder_init_scale=1.0,
                parameters=ShiuParameters.from_upstream(source / "model.py"))
            logger.bind_whole_brain(graph, model, source, ports_manifest=options.ports)
            fingerprint = graph_fingerprint(graph)
        optimizer = torch.optim.Adam(model.parameters(), lr=options.learning_rate)
        observations = torch.as_tensor(fragment.observations, device=model.device)
        targets = torch.as_tensor(fragment.probabilities, device=model.device)
        actions = torch.as_tensor(fragment.actions, device=model.device)
        best_score = float("inf")
        history = []
        for epoch in range(options.epochs + 1):
            started = time.monotonic()
            before_hashes = parameter_hashes(model)
            changed = {name: 0 for name, _ in model.named_parameters()}
            gradient_norms, gradients_per_chunk = [], []
            training_activity = NeuralActivityAccumulator(model.output_indices.cpu().tolist())
            online = _Metrics(0.5)
            if epoch:
                model.train()
                with logger.phase(f"diagnostic_epoch_{epoch:04d}_fragment_training"):
                    state = _prime_state(model, observations[0], options.warmup_ms)
                    for first in range(0, len(fragment), options.chunk_decisions):
                        optimizer.zero_grad(set_to_none=True)
                        losses = []
                        for step in range(first, min(first + options.chunk_decisions, len(fragment))):
                            logits, state = model(observations[step], state)
                            logits = logits.unsqueeze(0)
                            losses.append(mixed_distillation_loss(logits, targets[step:step + 1], 0.5))
                            online.add(logits, targets[step:step + 1], actions[step:step + 1])
                            training_activity.add(state.window_spike_counts, model.neural_ms_per_decision)
                        loss = torch.stack(losses).mean()
                        if not bool(torch.isfinite(loss)):
                            raise FloatingPointError("Non-finite full-brain diagnostic loss")
                        loss.backward()
                        norm, named_norms = _clip_gradients_fp64(model, options.gradient_clip)
                        gradient_norms.append(norm)
                        gradients_per_chunk.append({"first_step": first, "global_norm_fp64": norm,
                                                    "parameter_norms": named_norms})
                        previous = {name: value.detach().clone() for name, value in model.named_parameters()}
                        optimizer.step()
                        for name, value in model.named_parameters():
                            if not bool(torch.isfinite(value).all()):
                                raise FloatingPointError(f"Non-finite diagnostic parameter {name}")
                            changed[name] += int(bool(torch.any(value.detach() != previous[name])))
                        state = state.detach()
            with logger.phase(f"diagnostic_epoch_{epoch:04d}_same_fragment_replay"):
                replay, activity = _evaluate(model, [fragment], warmup_ms=options.warmup_ms, hard_weight=0.5)
            after_hashes = parameter_hashes(model)
            metrics = {"same_training_fragment_replay": replay,
                       "online_training": online.summary() if epoch else None,
                       "gradient_norm_fp64_mean": float(np.mean(gradient_norms)) if epoch else 0.0,
                       "gradient_norm_fp64_max": max(gradient_norms, default=0.0),
                       "gradients_per_chunk": gradients_per_chunk,
                       "changed_chunks_per_parameter": changed,
                       "parameter_hashes_before": before_hashes, "parameter_hashes_after": after_hashes,
                       "training_activity": training_activity.summary() if epoch else None,
                       "evaluation_scope": "same training fragment; validation episodes not evaluated"}
            score = replay["mixed_cross_entropy"]
            best_score = min(best_score, score)
            logger.record_epoch(epoch, metrics, elapsed_seconds=time.monotonic() - started, activity=activity)
            checkpoint = output / "epochs" / f"epoch_{epoch:04d}.pt"
            _save_checkpoint(checkpoint, model, optimizer, fingerprint=fingerprint, encoder=dataset.encoder,
                epoch=epoch, metrics=metrics, config=config, train_ids=train_ids,
                validation_ids=validation_ids, best_score=best_score)
            archive = logger.record_checkpoint(checkpoint, tag="diagnostic_training_fragment", epoch=epoch)
            record = {"epoch": epoch, "mixed_cross_entropy": score, "teacher_kl": replay["teacher_kl"],
                      "teacher_argmax_agreement": replay["teacher_argmax_agreement"],
                      "descending_spikes": activity["descending_spikes"],
                      "mean_descending_rate_hz": activity["mean_descending_rate_hz"],
                      "changed_chunks_per_parameter": changed, "parameter_hashes": after_hashes,
                      "gradient_norm_fp64_max": max(gradient_norms, default=0.0),
                      "checkpoint": archive["path"], "seconds": time.monotonic() - started}
            history.append(record)
            _write_json(output / "diagnostic_progress.json", {"history": history, "scope": config["scope"]})
            print(json.dumps(record, ensure_ascii=False, allow_nan=False), flush=True)
        report = {"status": "completed_training_fragment_diagnostic", "history": history,
                  "mixed_ce_reduction_epoch0_to_best": history[0]["mixed_cross_entropy"] - best_score,
                  "graph_fingerprint": fingerprint, "selected_fragment": selected,
                  "final_game_acceptance_achieved": False, "held_out_generalization_measured": False,
                  "scope": config["scope"]}
        _write_json(output / "diagnostic_report.json", report)
        logger.event("diagnostic_finished", report)
    except BaseException as error:
        logger.record_error(error, phase="full_brain_fragment_diagnostic")
        raise


if __name__ == "__main__":
    main()
