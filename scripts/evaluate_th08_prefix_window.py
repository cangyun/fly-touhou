"""Evaluate a frozen policy after a matched, committed-action TH08 prefix."""

import argparse
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time
import traceback
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fly_touhou.nbnm_job_phase import gpu_phase
from fly_touhou.nbnm_run import read
from fly_touhou.th08_controller import TH08InputController
from fly_touhou.th08_game_state import STAGE_NUMBERS
from fly_touhou.th08_native_forecast import _horizon_results, query_live_native_routes
from fly_touhou.th08_prefix import (
    prepare_policy_window, prepare_replay, read_prefix_source, recorded_burn_in, replay_practice_slot,
    restore_observed_attack_age, save_stage_memory,
)
from fly_touhou.th08_sensation import POLICY_MASKS
from fly_touhou.th08_training import portable_spec, training_worlds
from fly_touhou.thrl_durable_records import append_jsonl, write_json


def frozen_step(model, graphs, features, brain):
    import torch

    from fly_touhou.nbnm_numerics import verified_movement_actions

    batch = torch.as_tensor(features[None], device=model.device)
    result, after, _ = graphs.frozen_batch(batch, brain, 1)
    scores = result["logits"][0, :len(POLICY_MASKS)].detach().cpu().numpy()
    action = int(result["actions"][0])
    verified_movement_actions(scores[None], [action])
    return action, after, scores.tolist()


def native_opening(world, observation):
    """Compare the exact proposals to holding each first command at one root."""
    scene = world.encoder.last_result
    hold_paths, stats = query_live_native_routes(world.controller, [[(mask, 96)] for mask in POLICY_MASKS])
    return {
        "info": world.status(), "features": observation["features"].tolist(),
        "proposal_margins": scene["margins"].tolist(),
        "proposal_covered": scene["native_coverage"].tolist(),
        "proposal_first_contact": scene["native_forecast"]["first_contact_by_action"],
        "proposal_first_commands": scene["corridor_courses"][:, :, :4].tolist(),
        "corner_route_actions": scene["native_forecast"].get("corner_route_actions", {}),
        "hold_horizons": [_horizon_results(path) for path in hold_paths],
        "hold_first_contacts": [next((i for i, frame in enumerate(path, start=1) if frame["contact"]), None)
                                for path in hold_paths],
        "hold_time_ms": stats["elapsed_ms"],
    }


def run(args):
    import torch

    from fly_touhou.nbnm_cuda import InferenceGraphs
    from fly_touhou.nbnm_prefix import burn_in
    from fly_touhou.nbnm_run import load_model, runtime
    from fly_touhou.spiking_engineered_control import SpikingState
    from fly_touhou.thrl_job_paths import cleanup_native_temp

    args.output.mkdir(parents=True)
    started = time.monotonic()
    temporary = runtime(args.output)
    graphs = None

    def progress(phase, **detail):
        write_json(args.output / "progress.json", dict(
            phase=phase, elapsed_seconds=time.monotonic() - started, **detail))

    try:
        report, source_episode, source_dir, trace = read_prefix_source(
            args.report, args.episode, through_decision=args.decision)
        practice_slot = replay_practice_slot(report)
        seed = report["native_startup_seed"]
        decision = args.decision
        burn_steps = args.burn_in
        use_archived_history = getattr(args, "archived_burn_in", False)
        saved_memory = None
        stage_entry = getattr(args, "stage_entry", None)
        if stage_entry is not None:
            entry = next((row for row in trace if STAGE_NUMBERS[row["raw_stage"]] == stage_entry), None)
            if entry is None or practice_slot is not None:
                raise ValueError("Choose a stage reached in the recorded original Story")
            decision = entry["decision"]
            memory_path = source_dir / f"stage-{stage_entry}-entry.pt"
            memory_report = report
            continuation = read(args.report)
            if str(stage_entry) in continuation.get("stage_entries", {}):
                memory_report = continuation
                memory_path = args.report.parent / memory_path.name
            same_policy = (args.checkpoint == Path(memory_report["checkpoint"]).resolve()
                           and args.spec == Path(memory_report["policy_spec"]).resolve())
            if same_policy and memory_path.is_file():
                saved_memory = torch.load(memory_path, map_location="cpu", weights_only=True)
                burn_steps = 0
            else:
                burn_steps = min(burn_steps, decision)
        through_ending = getattr(args, "through_ending", False)
        until_attack_end = getattr(args, "until_attack_end", False)
        if through_ending and practice_slot is not None:
            raise ValueError("Continuing through the ending requires an original Story prefix")
        limit = args.decisions if args.decisions is not None else (None if through_ending or until_attack_end else 128)
        start = decision - burn_steps
        if start < 0 or not 0 <= decision < len(trace):
            raise ValueError("Choose a window whose complete burn-in is in the recorded trace")
        source_settings = portable_spec(read(Path(report["policy_spec"])),
                                        native_forecasts=report.get("forecast_native_branches", False))["nbnm"]
        spec = portable_spec(read(args.spec))
        if use_archived_history:
            from fly_touhou.th08_attack_readout import ATTACK_TASK_CONTEXTS

            source_sense = source_settings.copy()
            target_sense = spec["nbnm"].copy()
            contexts = (source_sense.pop("task_context", None), target_sense.pop("task_context", None))
            if source_sense != target_sense or any(context not in ATTACK_TASK_CONTEXTS for context in contexts):
                raise ValueError("Recorded burn-in needs matching sensory settings and observed-attack context")
            archived_features, archived_actions = recorded_burn_in(source_dir, trace, decision + 1)
        model, migrated = load_model(spec, args.checkpoint, apply_native_sensors=False)
        if migrated or not (model.bombs_disabled and model.soft_risk_composition):
            raise ValueError("Use the existing NBNM connectome policy")
        model.eval().requires_grad_(False)
        graphs = InferenceGraphs(model, sample=False, packed_binary_messages=spec.get("packed_binary_messages", False))
        identity = dict(native_seed=seed, source_decision=decision, checkpoint=str(args.checkpoint))
        progress("original-prefix-replay", **identity)
        with TemporaryDirectory(prefix="original-th08-", dir=args.output) as directory:
            slots = () if practice_slot is None else (practice_slot,)
            private, = training_worlds(report["native_executable"], report["native_profile"], report["d3d8to9"],
                                       directory, 1, source_settings, practice_slots=slots)
            with TH08InputController(private["executable"], private["profile"],
                    private_appdata=private["private_appdata"], d3d8to9=private["d3d8to9"],
                    unpaced=True, native_seed=seed, native_branch_checkpoints=True,
                    forecast_spawn_animations=source_settings.get("forecast_spawn_animations", False),
                    forecast_scripted_beams=source_settings.get("forecast_scripted_beams", False)) as controller:
                prefix = SimpleNamespace(decision=start, export_teacher=False, match_recorded_stage_entry=True,
                                         record_native_forecast=True, audit_spells=True)
                world, _, _, _, offsets = prepare_replay(
                    prefix, source_dir, controller, source_settings, source_episode, report, trace)
                observation, info, history = prepare_policy_window(
                    world, spec["nbnm"], report, trace, offsets, start, decision,
                    until_attack_end=until_attack_end and not use_archived_history,
                    carry_native_courses=getattr(args, "carry_native_courses", False))
                if use_archived_history:
                    restore_observed_attack_age(world, observation, archived_features[-1])
                    history = {"features": archived_features[:-1], "actions": archived_actions[:-1]}
                    burn_steps = len(history["actions"])
                root = trace[decision]
                encoder = world.encoder
                opening = native_opening(world, observation)
                opening["damage_forecast"] = world.encoder.last_result["native_forecast"].get("spell_damage")
                if saved_memory is None:
                    brain = burn_in(model, {"policy_burn_in": history}, model.initial_state(1))
                else:
                    if (saved_memory["decision"] != decision or saved_memory["raw_stage"] != root["raw_stage"]
                            or saved_memory["native_frames"] != root["native_frames"]
                            or saved_memory["stage_rng_seeds"] != info["stage_rng_seeds"]
                            or not torch.equal(saved_memory["features"], torch.from_numpy(observation["features"]))):
                        raise ValueError("Stage memory does not match the restored original observation")
                    brain = SpikingState(*(part.to(model.device) for part in saved_memory["brain"]))
                action, brain, scores = frozen_step(model, graphs, observation["features"], brain)
                opening.update(model_action=action, recorded_action=root["action"], logits=scores)
                write_json(args.output / "opening.json", opening)
                np.savez_compressed(args.output / "burn-in.npz", features=np.asarray(history["features"]),
                                    actions=np.asarray(history["actions"], dtype=np.int64))
                decisions = 0
                initial_checks = world.native_forecast_checks
                entry = info
                spell_index = len(world.spell_results)
                attack_ended = False
                stage_entries = {}
                new_stage_entry = False
                progress("autonomous", **identity, decisions=0, info=info)
                while observation is not None:
                    if (args.output / "STOP").exists():
                        info = dict(info, outcome="stopped")
                        break
                    scene = world.encoder.last_result
                    row = {
                        "decision": decisions, "native_frames": info["native_frames"], "stage_frame": info["stage_frame"],
                        "raw_stage": info["raw_stage"], "native_spell_id": info["native_spell_id"],
                        "native_dialogue_edges": True,
                        "position": info["position"],
                        "action": action, "logits": scores, "margins": scene["margins"][:, action].tolist(),
                        "known": scene["native_coverage"][:, action].tolist(),
                        "forecast_contact": scene["native_forecast"]["first_contact_by_action"][action],
                        "forecast_contacts_by_action": scene["native_forecast"]["first_contact_by_action"],
                        "corridor_status_48_96": scene["corridor"][:, 2:4].tolist(),
                        "forecast_elapsed_ms": scene["native_forecast"]["elapsed_ms"],
                        "revalidated_course_action": scene["native_forecast"].get("revalidated_course_action"),
                        "revalidated_course_frames": scene["native_forecast"].get("revalidated_course_frames"),
                        "selected_verified_course_frames": scene["native_forecast"]["verified_course_frames"][action],
                        "held_routes_tested": scene["native_forecast"].get("held_routes_tested", 0),
                        "held_route_actions": scene["native_forecast"].get("held_route_actions", []),
                        "corner_route_actions": scene["native_forecast"].get("corner_route_actions", {}),
                        "final_spark_charge": scene["final_spark_charge"],
                        "final_spark_exits": scene["native_forecast"].get("final_spark_exits"),
                        "spell_damage": scene["native_forecast"].get("spell_damage"),
                        "spell_health": scene["native_forecast"].get("spell_health"),
                    }
                    if getattr(args, "record_observations", False):
                        row["input_features"] = observation["features"].tolist()
                    append_jsonl(args.output / "decisions.jsonl", row)
                    if new_stage_entry:
                        write_json(args.output / f"resume-stage-{info['stage']}.json", dict(
                            start=entry, end=info, source_report=str(args.report),
                            source_episode=source_episode["episode"], source_decision=decision,
                            rollout_scope="remaining-story", resume_through_decision=decision + decisions,
                            checkpoint=str(args.checkpoint), policy_spec=str(args.spec),
                            stage_entries=stage_entries, full_game_NBNM=False))
                        new_stage_entry = False
                    previous_stage = info["raw_stage"]
                    observation, info = world.step(action)
                    decisions += 1
                    if decisions % 128 == 0 or info["raw_stage"] != previous_stage or observation is None:
                        progress("autonomous", **identity, decisions=decisions, info=info)
                    attack_ended = (info["outcome"] in (None, "story-clear", "stage-practice-clear")
                                    and (info["raw_stage"], info["native_spell_id"])
                                    != (entry["raw_stage"], entry["native_spell_id"]))
                    if (observation is None or decisions == limit
                            or (until_attack_end and attack_ended)
                            or (not through_ending and info["raw_stage"] != entry["raw_stage"])):
                        break
                    if through_ending and info["raw_stage"] != previous_stage:
                        path = args.output / f"stage-{info['stage']}-entry.pt"
                        save_stage_memory(path, decision + decisions,
                                          info["native_frames"] + root["native_frames"] - entry["native_frames"],
                                          info, observation, brain)
                        stage_entries[str(info["stage"])] = path.name
                        new_stage_entry = True
                    action, brain, scores = frozen_step(model, graphs, observation["features"], brain)
                result = {
                    "start": entry, "end": info, "source_report": str(args.report), "checkpoint": str(args.checkpoint),
                    "policy_spec": str(args.spec), "stage_entries": stage_entries,
                    "source_episode": source_episode["episode"],
                    "burn_in_decisions": burn_steps, "autonomous_local_decisions": decisions,
                    "archived_burn_in": use_archived_history,
                    "restored_recurrent_state": saved_memory is not None,
                    "source_decision": decision,
                    "rollout_scope": "remaining-story" if through_ending else "attack" if until_attack_end else "stage",
                    "local_attack_ended": attack_ended,
                    "prefix_spell_results": world.spell_results[:spell_index],
                    "spell_results": world.spell_results[spell_index:],
                    "native_forecast_matches": world.native_forecast_checks - initial_checks,
                    "forecast_native_carried_courses": encoder.forecast_native_carried_courses,
                    "forecast_native_hold_fallback": encoder.forecast_native_hold_fallback,
                    "forecast_native_corner_turns": encoder.forecast_native_corner_turns,
                    "forecast_final_spark_cue": encoder.forecast_final_spark_cue,
                    "native_dialogue_edges": True,
                    "original_frames_survived_after_release": info["native_frames"] - entry["native_frames"],
                    "full_game_NBNM": False, "elapsed_seconds": time.monotonic() - started,
                }
                if info["contact"] and world.raw_scene is not None:
                    result["contact_geometry"] = world.contact_objects(world.raw_scene)
                write_json(args.output / "report.json", result)
                progress("complete", **identity, decisions=decisions, info=info)
    except BaseException:
        write_json(args.output / "failure.json", {"error": traceback.format_exc()})
        raise
    finally:
        if graphs is not None:
            graphs.clear()
        cleanup_native_temp(temporary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("report", "spec", "checkpoint", "output"):
        parser.add_argument(f"--{name}", required=True, type=lambda path: Path(path).resolve())
    parser.add_argument("--episode", type=int, default=1)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--decision", type=int)
    source.add_argument("--stage-entry", type=int, choices=range(1, 7))
    parser.add_argument("--burn-in", type=int, default=32)
    parser.add_argument("--archived-burn-in", action="store_true",
                        help="Warm the policy from all original observations through the recorded release decision")
    parser.add_argument("--decisions", type=int)
    parser.add_argument("--record-observations", action="store_true",
                        help="Save the model's input for each committed autonomous action")
    parser.add_argument("--carry-native-courses", action="store_true",
                        help="Revalidate the last selected course after each committed original movement")
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--through-ending", action="store_true")
    scope.add_argument("--until-attack-end", action="store_true",
                       help="Stop at a new spell, the current spell result, a stage transition, or the first contact.")
    args = parser.parse_args()
    if args.burn_in < 0 or (args.decisions is not None and args.decisions < 1):
        parser.error("Use nonnegative burn-in and positive autonomous decision counts")
    if args.archived_burn_in and args.decision is None:
        parser.error("Recorded burn-in needs an explicit release decision")
    with gpu_phase("evaluation"):
        run(args)
