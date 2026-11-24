"""Reproduce a seeded TH08 policy prefix and test an alternative in the original."""
import argparse
import gzip
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fly_touhou.nbnm_job_phase import gpu_phase
from fly_touhou.nbnm_run import read
from fly_touhou.nbnm_corridor import HORIZONS
from fly_touhou.th08_controller import TH08InputController
from fly_touhou.th08_prefix import (contiguous_safe_move, match_pose, measured_action,
                                    prepare_replay, read_prefix_source, replay_practice_slot)
from fly_touhou.th08_sensation import NORMAL_MOVE_COUNT, POLICY_MASKS
from fly_touhou.th08_story import nearby_native_objects
from fly_touhou.th08_training import portable_spec, training_worlds
from fly_touhou.thrl_durable_records import write_json


def refreshed_command(scene, motion, preferred):
    four, twelve = scene["margins"][:2]
    if not (four > 1.).any():
        return int(four.argmax())
    courses = scene["corridor_courses"]
    ends = scene["ends"]

    def rank(action):
        reach = 2 if courses[action, 1, 0] >= 0 else int(courses[action, 0, 0] >= 0)
        space = min((ends[action] - motion.minimum).min(), (motion.maximum - ends[action]).min())
        return four[action] > 1., reach, min(float(twelve[action]), 32.) + .25 * space
    best = max(range(len(ends)), key=rank)
    return preferred if rank(preferred)[:2] == rank(best)[:2] and four[preferred] > 1. else best


def contact_snapshot(controller, state):
    game = controller.read_game_state(state)
    fields = ("raw_stage", "raw_phase", "native_spell_id", "native_spell_flags", "spell_active",
              "native_spell_captured", "lives", "bombs", "recorded_misses", "recorded_bombs", "continues")
    return dict(input_poll=state["input_poll"], native_mode=state["mode"], player_state=state["player_state"],
                contact=state["native_contact"], misses=state["misses"], bombs_used=state["bombs_used"],
                deathbombs_used=state["deathbombs_used"], **{key: game.get(key) for key in fields})


def observe_contact_settlement(controller, state, frames):
    """Observe native counters after the diagnostic's ordinary contact stop."""
    first = contact_snapshot(controller, state)
    samples = [first]
    offset = 0
    while offset < frames and state["mode"] == 2 and state["stage"] == first["raw_stage"]:
        previous = samples[-1]
        mask = (offset // 4) % 2 if previous["raw_phase"] in (2, 3) else POLICY_MASKS[0]
        window = min(frames - offset, 4)
        state = controller.advance(mask, window)
        samples.append(contact_snapshot(controller, state))
        offset += window
    return samples


def replay(args, source_dir, controller, settings, source_episode, source_report, trace):
    world, obs, info, history, frame_offsets = prepare_replay(
        args, source_dir, controller, settings, source_episode, source_report, trace)
    native_seed = source_report["native_startup_seed"]
    seeds = info["stage_rng_seeds"]
    motion, projectiles, enemies = world.raw_scene
    entry = dict(info=info, objects=nearby_native_objects(motion.position, projectiles, enemies))
    if args.native_branch_move is not None:
        routes = world.encoder.last_result["native_candidate_routes"]
        commands = routes[args.native_branch_move, :args.horizon // world.decision_ticks].tolist()
    elif args.corridor_move is not None:
        horizon = int(args.horizon == 96)
        courses = world.encoder.last_result["corridor_courses"][args.corridor_move]
        steps = args.horizon // world.decision_ticks
        predicted = courses[horizon, :steps]
        if predicted[0] < 0 and horizon and args.refresh_corridor:
            predicted = courses[0, :48 // world.decision_ticks]
        if (predicted < 0).any():
            raise ValueError(f"No valid starting corridor for movement {args.corridor_move}")
        commands = predicted.tolist()
        commands.extend([commands[-1]] * (steps - len(commands)))
    elif args.commands is not None:
        commands = json.loads(args.commands.read_text(encoding="utf-8"))
    else:
        commands = [row["action"] for row in trace[args.decision:]]
    if (not isinstance(commands, list) or not commands
            or any(type(x) is not int or not 0 <= x < NORMAL_MOVE_COUNT for x in commands)):
        raise ValueError("Suffix commands must be a nonempty JSON array of movement indexes 0 through 17")

    if args.record_native_forecast:
        from fly_touhou.th08_native_forecast import capture_native_branch, compare_legacy_geometry

        commands = commands[:args.horizon // world.decision_ticks]
        masks = [POLICY_MASKS[action] for action in commands for _ in range(world.decision_ticks)]
        forecast = capture_native_branch(controller, masks)
        with gzip.open(args.output / "native-forecast.json.gz", "wt", encoding="utf-8") as stream:
            json.dump(forecast, stream, separators=(",", ":"))
        legacy = None
        if len(set(commands)) == 1:
            margins = world.encoder.last_result["margins"][:, commands[0]]
            legacy = {str(horizon): float(gap.clip(-256., 256.))
                      for horizon, gap in zip(HORIZONS, margins, strict=True)}
        return dict(source_report=str(args.report), matched_prefix_decisions=args.decision,
                    native_seed=native_seed, branch_entry=entry, commands=commands,
                    legacy_straight_clearance=legacy,
                    legacy_geometry=compare_legacy_geometry(world.encoder.last_scene, forecast),
                    native_forecast=forecast["summary"],
                    autonomous_policy=False, full_game_NBNM=False)

    verified_actions = 0
    after = []
    contact_geometry = None
    settlement_frames = getattr(args, "contact_settlement_frames", 0)
    before_contact = contact_snapshot(controller, world.state) if settlement_frames else None
    for index, planned_action in enumerate(commands):
        previous_obs, before = obs, info
        previous_scene = world.raw_scene
        forecast = world.encoder.last_result
        action = planned_action
        reference = None
        if args.reference_scores:
            reference = trace[args.decision + index]
            match_pose(reference, before, frame_offsets[before["raw_stage"]])
        if args.refresh_corridor and index:
            action = refreshed_command(forecast, previous_scene[0], planned_action)
        obs, info = world.step(action, observe=args.export_teacher or args.refresh_corridor)
        after.append(dict(action=action, planned_action=planned_action, **info))
        if info["contact"] and previous_obs is not None:
            contact_geometry = dict(four_frame_margin=float(forecast["margins"][0, action].clip(-64., 64.)),
                                    native_objects=world.contact_objects(previous_scene))
        if not contiguous_safe_move(before, info, world.decision_ticks):
            break
        verified_actions += 1
        if args.export_teacher:
            first = index == 0
            history.append(measured_action(previous_obs, before, action, exit_focus=first,
                                           override=first and action != trace[args.decision]["action"],
                                           reference=reference))
    survived = verified_actions == len(commands)
    result = dict(source_report=str(args.report), source_episode=source_episode["episode"],
                  native_seed=native_seed, matched_prefix_decisions=args.decision, branch_entry=entry,
                  native_frame_offset_at_each_stage=frame_offsets,
                  commands=[row["action"] for row in after], planned_commands=commands, after=after,
                  fully_executed_without_contact=survived,
                  verified_contiguous_frames=verified_actions * world.decision_ticks,
                  actual_contact=info["contact"], contact_geometry=contact_geometry,
                  autonomous_policy=False, full_game_NBNM=False)
    if settlement_frames and info["contact"]:
        result["contact_settlement"] = dict(
            before=before_contact, samples=observe_contact_settlement(controller, world.state, settlement_frames))
    if args.export_teacher and survived and len(commands) >= 12 and len(history) >= 16:
        from capture_th08_teacher_windows import save_windows

        teacher = args.output / "teacher"
        teacher.mkdir()
        capture = dict(source_report=str(args.report), source_decision=args.decision,
                       actual_contact=False, verified_suffix_frames=result["verified_contiguous_frames"])
        scene = world.encoder.last_result
        forecast = scene["native_forecast"]
        save_windows(teacher, history, info["raw_stage"], seeds[-1], policy=source_report["checkpoint"],
                     capture_status=capture, forecast_revision=forecast["revision"] if forecast is not None else None)
        result["teacher_manifest"] = str(teacher / "manifest.json")
        result["final_corridor_candidates"] = scene["corridor_courses"].tolist()
        result["final_corridor_features"] = scene["corridor"].tolist()
    return result


def run(args):
    started = time.monotonic()
    args.output.mkdir(parents=True)
    try:
        report, source_episode, source_dir, trace = read_prefix_source(
            args.report, args.episode, reference_scores=args.reference_scores)
        practice_slot = replay_practice_slot(report)
        native_seed = report["native_startup_seed"]
        if not 0 <= args.decision < len(trace) or trace[args.decision]["decision"] != args.decision:
            raise ValueError("Choose a decision index in the original frozen trace")
        settings = portable_spec(read(Path(report["policy_spec"])),
                                 native_forecasts=report.get("forecast_native_branches", False))["nbnm"]
        if args.reference_scores and not settings.get("forecast_native_branches"):
            raise ValueError("Retention scores must come from a frozen live-forecast episode")
        profile = Path(report["native_profile"])
        renderer = Path(report["d3d8to9"])
        with gpu_phase("evaluation"), TemporaryDirectory(prefix="original-th08-", dir=args.output) as root:
            slots = () if practice_slot is None else (practice_slot,)
            game = training_worlds(Path(report["native_executable"]), profile, renderer, Path(root), 1, settings,
                                   practice_slots=slots)[0]
            with TH08InputController(
                game["executable"], profile, private_appdata=game["private_appdata"], d3d8to9=renderer,
                unpaced=report["native_unpaced"], native_seed=native_seed,
                forecast_spawn_animations=settings.get("forecast_spawn_animations", False),
                forecast_scripted_beams=settings.get("forecast_scripted_beams", False),
                native_branch_checkpoints=settings.get("forecast_native_branches", False) or args.live_forecast,
            ) as controller:
                result = replay(args, source_dir, controller, settings, source_episode, report, trace)
        result["elapsed_seconds"] = time.monotonic() - started
        write_json(args.output / "report.json", result)
    except BaseException:
        write_json(args.output / "failure.json", dict(error=traceback.format_exc()))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=lambda path: Path(path).resolve(), required=True,
                        help="Seeded original TH08 Story, practice, or a matched continuation report")
    parser.add_argument("--episode", type=int, default=1)
    parser.add_argument("--decision", type=int, required=True, help="First original decision to replace")
    parser.add_argument("--output", type=lambda path: Path(path).resolve(), required=True)
    suffix = parser.add_mutually_exclusive_group()
    suffix.add_argument("--commands", type=Path, help="JSON array of replacement movement indexes")
    suffix.add_argument("--corridor-move", type=int, choices=range(NORMAL_MOVE_COUNT),
                        help="Execute the current projected corridor for this first movement")
    suffix.add_argument("--native-branch-move", type=int, choices=range(NORMAL_MOVE_COUNT),
                        help="Execute the tape measured by the original native branch for this first movement")
    parser.add_argument("--horizon", type=int, choices=(48, 96), default=96)
    parser.add_argument("--export-teacher", action="store_true",
                        help="Export measured continuous replacement commands with their observed inputs")
    parser.add_argument("--final-spark-cue", action="store_true",
                        help="Use Final Master Spark cues after verifying the recorded original entry")
    parser.add_argument("--refresh-corridor", action="store_true",
                        help="Recheck every movement against the actual next original scene during this diagnostic")
    parser.add_argument("--match-recorded-stage-entry", action="store_true",
                        help="Align older reports at their recorded original stage clocks")
    parser.add_argument("--record-native-forecast", action="store_true",
                        help="Record all occupied pools and native collision tests for the selected continuation")
    parser.add_argument("--contact-settlement-frames", type=int, default=0,
                        help="Observe original resource counters after contact with neutral movement and no Bomb")
    parser.add_argument("--live-forecast", action="store_true",
                        help="Re-encode the final prefix and its continuation with live original forecasts")
    parser.add_argument("--reference-scores", action="store_true",
                        help="Export matching frozen logits with the original live-forecast decisions")
    args = parser.parse_args()
    if args.native_branch_move is not None:
        args.live_forecast = True
    replacement = any(x is not None for x in (args.commands, args.corridor_move, args.native_branch_move))
    if args.refresh_corridor and args.corridor_move is None:
        parser.error("Corridor refresh requires a candidate first move")
    if args.export_teacher and not replacement and not args.reference_scores:
        parser.error("Teacher export requires a replacement command sequence or corridor")
    if args.reference_scores and (not args.export_teacher or replacement):
        parser.error("Reference scores require the recorded sequence and teacher export")
    if args.record_native_forecast and (args.export_teacher or args.refresh_corridor):
        parser.error("Native forecast recording uses a fixed continuation")
    if args.contact_settlement_frames < 0 or (args.contact_settlement_frames and args.record_native_forecast):
        parser.error("Native contact settlement requires a nonnegative duration and a committed continuation")
    run(args)
