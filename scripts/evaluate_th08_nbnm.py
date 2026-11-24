"""Run a frozen whole-connectome policy on original TH08 Lunatic Story."""

import argparse
from collections import deque
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


from fly_touhou.th08_game_state import capture_goal_outcome, captured_spells


def verified_full_spell_clear(episode, reference):
    required = captured_spells(reference)
    return (required is not None and episode["full_game_NBNM"] and not episode["contact"]
            and episode["route_history"] == [entry["raw_stage"] for entry in reference["stages"]]
            and captured_spells(episode) == required)


def observed_course_suffixes(rows):
    """Measure how much of each forecast command sequence the original played."""
    evidence = []
    for start, row in enumerate(rows):
        courses = row.get("selected_corridor_commands")
        if courses is None:
            continue
        for horizon, course in zip((48, 96), courses, strict=True):
            if course[0] < 0:
                continue
            elapsed = matched = 0
            outcome = "unobserved"
            for command, step in zip(course[:horizon // 4], rows[start:]):
                if step["native_frames"] != row["native_frames"] + 4 * matched:
                    break
                if command != step["action"]:
                    outcome = "deviated"
                    break
                matched += 1
                elapsed = step["after"]["native_frames"] - row["native_frames"]
                if step["after"]["contact"]:
                    outcome = "contact-on-course"
                    break
                if elapsed == horizon:
                    outcome = "cleared-horizon"
                    break
            evidence.append(dict(decision=row["decision"], horizon=horizon, matched_commands=matched,
                                 matched_native_frames=elapsed, outcome=outcome))
    return evidence


def write_contact_evidence(path, recent):
    """Publish the consecutive original frames leading to the first contact."""
    from fly_touhou.thrl_durable_records import write_json

    if not recent[-1]["after"]["contact"]:
        write_json(path, dict(verified_contact_command=False,
                              reason="resource use reported without a native contact marker",
                              last_decision=recent[-1]))
        return str(path)
    selected = []
    for row in reversed(recent):
        after = row["after"]
        if (after["raw_stage"] != row["raw_stage"]
                or not 0 < after["native_frames"] - row["native_frames"] <= 4):
            break
        if selected and after["native_frames"] != selected[-1]["native_frames"]:
            break
        selected.append(row)
    if not selected:
        write_json(path, dict(verified_contact_command=False,
                              reason="terminal observation follows a native transition",
                              last_decision=recent[-1]))
    else:
        selected.reverse()
        terminal = selected[-1]
        action = terminal["action"]
        write_json(path, dict(
            verified_contact_command=True, raw_stage=terminal["raw_stage"],
            observed_contact_frame=terminal["after"]["native_frames"],
            contact_command=action,
            contact_command_clearance4=terminal["geometry"]["clearance"][0][action],
            contact_forecast4_covered=terminal["geometry"]["forecast_covered"][0],
            clearance_horizons=[4, 12, 24], corridor_horizons=[48, 96],
            elapsed_native_frames=terminal["after"]["native_frames"] - selected[0]["native_frames"],
            decisions=selected, observed_corridor_suffixes=observed_course_suffixes(selected),
        ))
    return str(path)


def attempt(args, episode, model, graphs, spec, executable, report_progress, capture_requirements=()):
    import numpy as np
    import torch

    from fly_touhou.nbnm_numerics import verified_movement_actions
    from fly_touhou.th08_controller import TH08InputController
    from fly_touhou.th08_prefix import save_stage_memory
    from fly_touhou.th08_story import TH08StoryWorld
    from fly_touhou.thrl_durable_records import write_json

    episode_dir = args.output / f"episode-{episode:03d}"
    episode_dir.mkdir()
    brain = model.initial_state(1)
    decisions = 0
    verified_movements = 0
    scripted_beam_decisions = 0
    relative_turn_decisions = 0
    started = time.monotonic()
    recent = deque(maxlen=24)
    contact_evidence = None
    settings = spec["nbnm"]
    record_observations = getattr(args, "record_observations", False)

    def save_stage_entry():
        if obs is None or args.practice_stage is not None or args.native_seed is None:
            return
        save_stage_memory(episode_dir / f"stage-{info['stage']}-entry.pt",
                          decisions, info["native_frames"], info, obs, brain)

    with TH08InputController(executable, args.profile, private_appdata=episode_dir / "appdata",
                            d3d8to9=args.d3d8to9,
                            forecast_spawn_animations=settings.get("forecast_spawn_animations", False),
                            forecast_scripted_beams=settings.get("forecast_scripted_beams", False),
                            unpaced=args.unpaced, native_seed=args.native_seed,
                            native_branch_checkpoints=settings["forecast_native_branches"]) as controller:
        world = TH08StoryWorld(controller, task_context=spec["nbnm"].get("task_context", "legacy"),
                               retain_raw_scene=True, practice_slot=args.practice_stage,
                               spell_slot=args.practice_spell,
                               forecast_sweep_laser_births=settings.get("forecast_sweep_laser_births", False),
                               forecast_wide_beam_births=settings.get("forecast_wide_beam_births", False),
                               forecast_orbital_point_births=settings.get("forecast_orbital_point_births", False),
                               forecast_final_b_ring=settings.get("forecast_final_b_ring", False),
                               forecast_native_branches=settings["forecast_native_branches"],
                               forecast_native_narrow_exits=settings.get("forecast_native_narrow_exits", False),
                               forecast_native_carried_courses=settings.get("forecast_native_carried_courses", False),
                               forecast_native_hold_fallback=settings.get("forecast_native_hold_fallback", False),
                               forecast_native_corner_turns=settings.get("forecast_native_corner_turns", False),
                               forecast_final_spark_cue=settings.get("forecast_final_spark_cue", False),
                               align_native_stage_entry=args.native_seed is not None,
                               native_dialogue_edges=True)
        obs, info = world.start()
        expected_spell_id = getattr(args, "expect_spell_id", None)
        if expected_spell_id is not None and info["practice_spell"] != expected_spell_id:
            raise ValueError(f"Original Spell Practice opened ID {info['practice_spell']}, expected {expected_spell_id}")
        seed = world.state.get("native_seed")
        if args.native_seed is not None and (not seed or seed["requested"] != args.native_seed or seed["applied"] != 1):
            raise ValueError("The original TH08 startup did not apply the requested seed")
        if seed is not None:
            write_json(episode_dir / "startup-seed.json", seed)
        if args.native_seed is not None:
            write_json(episode_dir / "paired-entry.json", dict(
                native_seed=args.native_seed, raw_stage=info["raw_stage"], stage_frame=info["stage_frame"],
                stage_rng_seeds=info["stage_rng_seeds"], position=info["position"],
                features=obs["features"].tolist() if obs is not None else None))
        save_stage_entry()
        report_progress(episode, decisions, info)
        with (episode_dir / "decisions.jsonl").open("w", encoding="utf-8", buffering=1) as trace:
            while obs is not None and (args.max_decisions is None or decisions < args.max_decisions):
                if (args.output / "STOP").exists():
                    return dict(info, episode=episode, policy_decisions=decisions, outcome="stopped",
                                verified_movement_decisions=verified_movements,
                                scripted_beam_decisions=scripted_beam_decisions,
                                relative_turn_decisions=relative_turn_decisions,
                                spell_results=world.spell_results,
                                elapsed_seconds=time.monotonic() - started)
                inputs = torch.as_tensor(obs["features"][None], device=model.device)
                pred, brain, _ = graphs.frozen_batch(inputs, brain, 1)
                extra_heads = {name: pred[name][0, :18] for name in
                               ("corridor_logits", "corridor_witness_logits") if name in pred}
                telemetry = torch.cat((pred["logits"][0, :18], pred["route_scores"][0, :18],
                    pred["horizon_scores"][0, :18].flatten(),
                    *(tensor.flatten() for tensor in extra_heads.values()))).cpu().numpy()
                scores = telemetry[:18]
                action = int(pred["actions"][0])
                verified = verified_movement_actions(scores[None], [action])
                scene = world.encoder.last_result
                scripted = scene["live_scripted_beams"]
                scripted_beam_decisions += bool(scripted)
                relative_turn_decisions += bool(scene["active_relative_turns"])
                row = dict(decision=decisions, stage=info["stage"],
                    raw_stage=info["raw_stage"], stage_frame=info["stage_frame"], native_frames=info["native_frames"],
                    native_spell_id=info["native_spell_id"], position=info["position"], native_dialogue_edges=True,
                    action=action, logits=scores.tolist(), scripted_beams=scripted,
                    active_relative_turns=scene["active_relative_turns"],
                    pending_relative_turns=scene["pending_relative_turns"],
                    pending_laser_births=scene["pending_laser_births"],
                    pending_wide_beams=scene["pending_wide_beams"],
                    final_spark_charge=scene["final_spark_charge"],
                    pending_orbital_points=scene["pending_orbital_points"], pending_ring_bullets=scene["pending_ring_bullets"])
                if scene["native_forecast"] is not None:
                    row["native_branch_forecast"] = dict(scene["native_forecast"],
                        covered_candidates=scene["native_coverage"].sum(axis=1).tolist())
                features = obs["features"].tolist()
                if record_observations:
                    row["input_features"] = features
                trace.write(json.dumps(row) + "\n")
                evidence = dict(row, route_scores=telemetry[18:36].tolist(),
                    selected_corridor_commands=scene["corridor_courses"][action].tolist(),
                    neural_risk_by_action=telemetry[36:90].reshape(18, 3).tolist(),
                    input_features=features,
                    geometry=dict(clearance=np.clip(scene["margins"][:3], -64., 64.).tolist(),
                        forecast_covered=scene["known"].tolist(),
                        corridor_status=scene["corridor"][:, 2:4].tolist(),
                        corridor_confidence=scene["corridor"][:, 6].tolist(),
                        objects=[scene["live_bullets"], scene["live_lasers"], scene["live_enemies"]]))
                offset = 90
                for name, tensor in extra_heads.items():
                    end = offset + tensor.numel()
                    evidence[name] = telemetry[offset:end].reshape(tensor.shape).tolist()
                    offset = end
                previous_stage = info["stage"]
                previous_scene = world.raw_scene
                previous_spell_count = len(world.spell_results)
                obs, info = world.step(action)
                if capture_requirements and (len(world.spell_results) != previous_spell_count
                        or info["stage"] != previous_stage or info["outcome"] is not None):
                    if all(capture_goal_outcome(info, world.spell_results, sequence) == "spell-capture-failure"
                           for sequence in capture_requirements):
                        info["outcome"] = "spell-capture-failure"
                        obs = None
                evidence["after"] = {k: info[k] for k in ("native_frames", "raw_stage", "stage_frame",
                    "position", "native_spell_id", "contact", "misses", "Bombs", "continues")}
                if obs is None and info["contact"] and info["stage"] == previous_stage:
                    evidence["native_objects"] = world.contact_objects(previous_scene)
                recent.append(evidence)
                decisions += 1
                verified_movements += verified
                if info["stage"] != previous_stage:
                    save_stage_entry()
                if decisions % 128 == 0 or info["stage"] != previous_stage or obs is None:
                    report_progress(episode, decisions, info)
    if obs is not None:
        info["outcome"] = "decision-budget"
    elif info["outcome"] == "nbnm-failure" and recent:
        contact_evidence = write_contact_evidence(episode_dir / "first-contact-evidence.json", recent)
    return dict(info, episode=episode, policy_decisions=decisions, verified_movement_decisions=verified_movements,
                scripted_beam_decisions=scripted_beam_decisions,
                relative_turn_decisions=relative_turn_decisions,
                native_forecast_matched_decisions=world.native_forecast_checks,
                spell_results=world.spell_results,
                first_contact_evidence=contact_evidence, elapsed_seconds=time.monotonic() - started)


def run(args):
    import psutil

    from fly_touhou.nbnm_corridor import SCHEMA
    from fly_touhou.nbnm_run import load_model, read, runtime
    from fly_touhou.th08_native_forecast import FORECAST_REVISION
    from fly_touhou.th08_training import portable_spec, training_worlds
    from fly_touhou.thrl_durable_records import write_json
    from fly_touhou.thrl_job_paths import cleanup_native_temp

    args.output.mkdir(parents=True)
    temporary = runtime(args.output)
    from fly_touhou.nbnm_cuda import InferenceGraphs

    proc = psutil.Process()
    started = time.monotonic()
    entry_mode = "original-story"
    if args.practice_spell is not None:
        entry_mode = "original-spell-practice"
    elif args.practice_stage is not None:
        entry_mode = "original-stage-practice"
    metadata = dict(schema="th08-original-frozen-nbnm-v1", checkpoint=str(args.checkpoint),
        policy_spec=str(args.spec), native_executable=str(args.executable),
        native_profile=str(args.profile), d3d8to9=str(args.d3d8to9),
        input_schema=SCHEMA, difficulty="Lunatic", decision_ticks=4,
        movement_actions=18, action_selection="network joint argmax", game_process="private original",
        private_game_files=True, native_audio=args.record_audio, native_unpaced=args.unpaced,
        entry_mode=entry_mode, practice_stage_slot=args.practice_stage, practice_spell_slot=args.practice_spell,
        native_rng_controlled=args.native_seed is not None, native_startup_seed=args.native_seed,
        aligned_native_stage_entries=args.native_seed is not None, native_dialogue_edges=True, training_updates=0)
    metadata["native_contact_scope"] = "Player::Die within the committed input command"

    def progress(episode, decisions, info):
        write_json(args.output / "progress.json", dict(metadata, phase="frozen", PID=proc.pid,
            create_time=proc.create_time(), episode=episode, decisions=decisions, info=info,
            elapsed_seconds=time.monotonic() - started))

    graphs = None
    results = []
    try:
        reference_paths = getattr(args, "spell_capture_reference", [])
        references = [read(path) for path in reference_paths]
        if any(reference.get("reference_spell_sequence_verified") is not True
               or captured_spells(reference) is None for reference in references):
            raise ValueError("Use a complete spell sequence verified in original TH08 Replay")
        if reference_paths:
            metadata["spell_capture_references"] = [str(path) for path in reference_paths]
        stop_on_missed_capture = getattr(args, "stop_on_missed_capture", False)
        requirements = tuple(captured_spells(reference) for reference in references) if stop_on_missed_capture else ()
        if requirements:
            metadata["stop_on_missed_capture"] = True
        spec = portable_spec(read(args.spec))
        metadata.update(forecast_native_branches=spec["nbnm"]["forecast_native_branches"],
                        forecast_native_narrow_exits=spec["nbnm"].get("forecast_native_narrow_exits", False),
                        forecast_native_carried_courses=spec["nbnm"].get("forecast_native_carried_courses", False),
                        forecast_native_hold_fallback=spec["nbnm"].get("forecast_native_hold_fallback", False),
                        forecast_native_corner_turns=spec["nbnm"].get("forecast_native_corner_turns", False),
                        observation_forecast_revision=FORECAST_REVISION)
        metadata["forecast_scripted_beams"] = spec["nbnm"].get("forecast_scripted_beams", False)
        metadata["forecast_spawn_animations"] = spec["nbnm"].get("forecast_spawn_animations", False)
        metadata["forecast_sweep_laser_births"] = spec["nbnm"].get("forecast_sweep_laser_births", False)
        metadata["forecast_wide_beam_births"] = spec["nbnm"].get("forecast_wide_beam_births", False)
        metadata["forecast_final_spark_cue"] = spec["nbnm"].get("forecast_final_spark_cue", False)
        metadata["forecast_orbital_point_births"] = spec["nbnm"].get("forecast_orbital_point_births", False)
        metadata["forecast_final_b_ring"] = spec["nbnm"].get("forecast_final_b_ring", False)
        if metadata["forecast_spawn_animations"]:
            from fly_touhou.th08_bullet_spawn import SPAWN_FORECAST_REVISION
            metadata["spawn_forecast_revision"] = SPAWN_FORECAST_REVISION
        model, migrated = load_model(spec, args.checkpoint, apply_native_sensors=False)
        if (migrated or model.input_schema != SCHEMA
                or not (model.bombs_disabled and model.soft_risk_composition)):
            raise ValueError("TH08 transfer requires the existing 407-column NBNM policy")
        corridor = spec["nbnm"]["corridor_risk"]
        metadata["learned_score_weights"] = dict(
            route_temperature=model.temperature, neural_risk_scale=model.neural_risk_scale,
            risk_horizon_weights=list(model.movement_risk_horizon_weights),
            corridor_risk=corridor.get("actor_weight", 0.), corridor_witness=corridor.get("witness_weight", 0.))
        if corridor.get("short_risk_priority") is not None:
            metadata["learned_score_weights"]["short_risk_priority"] = corridor["short_risk_priority"]
        model.eval().requires_grad_(False)
        graphs = InferenceGraphs(model, sample=False, packed_binary_messages=spec.get("packed_binary_messages", False))
        with TemporaryDirectory(prefix="original-th08-", dir=args.output) as directory:
            private = Path(directory).resolve()
            if private.parent != args.output:
                raise ValueError("Private TH08 game files must stay within the evaluation output")
            slots = () if args.practice_stage is None else (args.practice_stage,)
            world = training_worlds(args.executable, args.profile, args.d3d8to9, private, 1, spec["nbnm"],
                                    practice_slots=slots, record_audio=args.record_audio)[0]
            for i in range(1, args.episodes + 1):
                row = attempt(args, i, model, graphs, spec, world["executable"], progress,
                              capture_requirements=requirements)
                if references:
                    row["full_game_NBNMFS"] = any(verified_full_spell_clear(row, reference) for reference in references)
                results.append(row)
                write_json(args.output / "report.json", dict(metadata, episodes=results,
                    attempts=len(results), full_game_NBNM=sum(r["full_game_NBNM"] for r in results),
                    full_game_NBNMFS=sum(r["full_game_NBNMFS"] for r in results) if references else None,
                    stage_practice_NBNM=sum(r["outcome"] == "stage-practice-clear" for r in results),
                    spell_practice_NBNM=sum(r["outcome"] == "spell-practice-survived" for r in results),
                    spell_practice_captures=sum(r["outcome"] == "spell-practice-survived"
                        and any(spell["captured"] is True and spell["spell_id"] == r["practice_spell"]
                                for spell in r["spell_results"]) for r in results),
                    verified_movement_decisions=sum(r["verified_movement_decisions"] for r in results),
                    elapsed_seconds=time.monotonic() - started))
                if row["outcome"] == "stopped":
                    break
        phase = "stopped" if results[-1]["outcome"] == "stopped" else "complete"
        write_json(args.output / "progress.json", dict(metadata, phase=phase, attempts=len(results),
            full_game_NBNM=sum(r["full_game_NBNM"] for r in results), elapsed_seconds=time.monotonic() - started))
    except BaseException:
        write_json(args.output / "failure.json", dict(metadata, error=traceback.format_exc()))
        raise
    finally:
        if graphs is not None:
            graphs.clear()
        cleanup_native_temp(temporary)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("spec", "checkpoint", "executable", "d3d8to9", "output"):
        p.add_argument(f"--{arg}", type=lambda value: Path(value).resolve(), required=True)
    p.add_argument("--profile", type=Path, default=ROOT / "configs" / "th08-1.00d-zh-input.json")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--native-seed", type=int, help="Use this 16-bit startup seed for a paired original-game test")
    pace = p.add_mutually_exclusive_group()
    pace.add_argument("--record-audio", action="store_true", help="Restore original BGM and effects at normal game speed")
    pace.add_argument("--unpaced", action="store_true", help="Skip the original gameplay frame wait in this silent run")
    p.add_argument("--practice-stage", type=int, choices=range(8),
                   help="Start at an original stage practice slot (4 is Stage 4B)")
    p.add_argument("--practice-spell", type=int, choices=range(44),
                   help="Start this unlocked Lunatic spell slot in the selected practice stage")
    p.add_argument("--expect-spell-id", type=int,
                   help="Check the native spell ID selected by the original practice menu")
    p.add_argument("--max-decisions", type=int, help="Optional diagnostic decision limit")
    p.add_argument("--record-observations", action="store_true",
                   help="Save the model's input for each committed autonomous action")
    p.add_argument("--spell-capture-reference", type=Path, action="append", default=[],
                   help="Original replay report defining a verified route's complete spell sequence; repeat for other routes")
    p.add_argument("--stop-on-missed-capture", action="store_true",
                   help="End a Story attempt once every verified spell sequence has missed a required capture")
    args = p.parse_args()
    if args.episodes < 1 or (args.max_decisions is not None and args.max_decisions < 1):
        p.error("Episode count and optional decision limit must be positive")
    if args.native_seed is not None and not 0 <= args.native_seed <= 65535:
        p.error("Use a native startup seed between 0 and 65535")
    if args.practice_spell is not None and args.practice_stage is None:
        p.error("--practice-spell requires --practice-stage")
    if args.expect_spell_id is not None and args.practice_spell is None:
        p.error("--expect-spell-id requires --practice-spell")
    if args.stop_on_missed_capture and (not args.spell_capture_reference or args.practice_stage is not None):
        p.error("Early capture rejection requires a verified spell sequence and original Story mode")
    from fly_touhou.nbnm_job_phase import gpu_phase
    with gpu_phase("evaluation"):
        run(args)
