"""Capture exact four-frame commands from original TH08 Replay playback."""

from argparse import ArgumentParser
from pathlib import Path
from shutil import copyfile
import sys
from tempfile import TemporaryDirectory

import numpy as np
import psutil


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fly_touhou.nbnm_job_phase import gpu_phase
from fly_touhou.nbnm_rehearsal import SCHEMA
from fly_touhou.th08_controller import TH08InputController
from fly_touhou.th08_human_replay import observed_four_frame_action, read_replay
from fly_touhou.th08_sensation import TH08ObservationEncoder
from fly_touhou.th08_training import training_worlds
from fly_touhou.thrl_durable_records import write_json
from fly_touhou.thrl_job_paths import cleanup_native_temp, configure_native_temp


def playable(state, game, raw_stage):
    return (state["native_replay"] and state["mode"] == 2 and state["player_state"] == 0
            and state["stage"] == raw_stage and game["active"] and game["raw_phase"] == 0
            and game["pause_state"] == 0)


def contacted(state, game):
    return bool(state["native_contact"] or state["misses"] or state["bombs_used"] or state["deathbombs_used"]
                or game.get("recorded_misses") or game.get("recorded_bombs") or game.get("continues"))


REPLAY_ACTION_BITS = 0xF7


def locate_replay_input(controller, stage, state, requested, entry_poll, stop_path):
    """Translate a recorded Replay index to a verified original input poll."""
    before = entry_poll + max(0, requested - 768)
    while state["input_poll"] < before and not stop_path.exists():
        state = controller.advance(0, min(24, before - state["input_poll"]))
        if not state["native_replay"] or state["stage"] != stage.raw_stage:
            raise ValueError("Original Replay ended before input alignment")
    started = state["input_poll"]
    recorded = stage.inputs & REPLAY_ACTION_BITS
    matches = np.arange(len(recorded))
    minimum = min(128, requested)
    for step in range(min(768, requested)):
        if (stop_path.exists() or not state["native_replay"] or state["stage"] != stage.raw_stage
                or state["input_poll"] != started + step):
            raise ValueError("Original input alignment did not have a consecutive Replay sample")
        observed = state["gameplay_input"] & REPLAY_ACTION_BITS
        matches = matches[matches + step < len(recorded)]
        matches = matches[recorded[matches + step] == observed]
        state = controller.advance(0, 1)
        count = step + 1
        # Holding still often repeats the first 128 inputs. Continue only
        # until a longer exact sequence identifies one Replay starting index.
        if not len(matches) or (count >= minimum and len(matches) == 1):
            break
    if len(matches) != 1:
        raise ValueError(f"Original input alignment matched {len(matches)} positions after {count} polls")
    first = int(matches[0])
    remaining = requested - first - count
    if remaining < 0 or state["input_poll"] != started + count:
        raise ValueError("Original Replay moved past the requested input index during alignment")
    return state, state["input_poll"] + remaining, dict(replay_index=first, count=count,
                                                        first_native_poll=started)


def capture(controller, stage, args):
    state = controller.enter_replay((stage.raw_stage,))
    if (state["difficulty"], state["shot_type"]) != (3, 0):
        raise ValueError("Require original Lunatic Border Team in Replay mode")
    entry_poll = state["input_poll"]
    limit = entry_poll + len(stage.inputs) + 4096
    replay_index = args.replay_input_offset
    by_input = args.first_input_poll is not None or replay_index is not None
    clock = "input_poll" if by_input else "stage_timer"
    stop_path = args.output / "STOP"
    alignment = None
    if replay_index is not None:
        requested = replay_index
        state, target, alignment = locate_replay_input(controller, stage, state, requested, entry_poll, stop_path)
    else:
        requested = args.first_input_poll if by_input else args.first_stage_frame
        target = entry_poll + requested if by_input else requested
    while state[clock] < target and state["input_poll"] < limit and not stop_path.exists():
        if not state["native_replay"] or state["stage"] != stage.raw_stage:
            raise ValueError("The requested starting clock was not reached during replay")
        state = controller.advance(0, min(24, target - state[clock], limit - state["input_poll"]))
    if state[clock] < target and not stop_path.exists():
        raise ValueError(f"Original replay stopped at stage frame {state['stage_timer']} and input poll "
                         f"{state['input_poll'] - entry_poll} before the requested start")

    if by_input and alignment is None:
        # Replay can cross a dialogue or transition while the stage clock is
        # stationary. Wait for the first playable checkpoint in this stage.
        while (state["native_replay"] and state["stage"] == stage.raw_stage
               and state["input_poll"] < limit and not stop_path.exists()):
            game = controller.read_game_state(state)
            if playable(state, game, stage.raw_stage) or contacted(state, game):
                break
            state = controller.advance(0, min(4, limit - state["input_poll"]))

    encoder = TH08ObservationEncoder(forecast_sweep_laser_births=args.forecast_sweep_laser_births,
                                     forecast_wide_beam_births=args.forecast_wide_beam_births,
                                     forecast_orbital_point_births=args.forecast_orbital_point_births,
                                     forecast_final_b_ring=args.forecast_final_b_ring,
                                     forecast_final_spark_cue=getattr(args, "forecast_final_spark_cue", False),
                                     item_collection_input=getattr(args, "item_collection_input", False))
    rows = []
    proc = psutil.Process()
    start_frame = state["stage_timer"]
    start_poll = state["input_poll"] - entry_poll
    end_reason = "decision-budget"
    skipped_transition_polls = 0
    while len(rows) < args.decisions and state["input_poll"] < limit:
        if stop_path.exists():
            end_reason = "stopped"
            break
        game = controller.read_game_state(state)
        if not state["native_replay"] or state["stage"] != stage.raw_stage:
            end_reason = "left-original-replay-stage"
            break
        if contacted(state, game):
            end_reason = "native-contact-or-resource-use"
            break
        if not playable(state, game, stage.raw_stage):
            if alignment is not None:
                end_reason = "left-playable-stage"
                break
            count = min(4, limit - state["input_poll"])
            state = controller.advance(0, count)
            skipped_transition_polls += count
            continue
        before = state
        power_before = game["power"]
        motion = controller.read_motion(before)
        feature = encoder.encode(before, game, motion, controller.read_projectiles(),
                                  controller.read_enemies(), controller.read_items())["features"]
        scene = encoder.last_result
        scripted_beams = scene["live_scripted_beams"]
        pending_wide_beams = scene["pending_wide_beams"]
        has_precursor = bool(scene["pending_laser_births"] or pending_wide_beams
                             or scene["pending_orbital_points"] or scene["pending_ring_bullets"]
                             or scene["final_spark_charge"])
        masks, positions = [], []
        for _ in range(4):
            masks.append(state["gameplay_input"])
            state = controller.advance(0, 1)
            positions.append(state["player_position"])
            if state["native_contact"] or state["stage"] != stage.raw_stage:
                break
        game = controller.read_game_state(state)
        if contacted(state, game):
            end_reason = "native-contact-or-resource-use"
            break
        if not playable(state, game, stage.raw_stage):
            if alignment is not None:
                end_reason = "left-playable-stage"
                break
            skipped_transition_polls += state["input_poll"] - before["input_poll"]
            continue
        if state["input_poll"] != before["input_poll"] + 4:
            end_reason = "input-clock-discontinuity"
            break
        if alignment is not None:
            first = replay_index + 4*len(rows)
            expected = stage.inputs[first:first + 4] & REPLAY_ACTION_BITS
            if not np.array_equal(np.asarray(masks) & REPLAY_ACTION_BITS, expected):
                end_reason = "replay-input-discontinuity"
                break
        action = observed_four_frame_action(motion, masks, positions)
        rows.append(dict(features=feature, action=action, native_poll=before["input_poll"],
                         stage_frame=before["stage_timer"], position=positions[-1], stage=game["stage"],
                         scripted_beams=scripted_beams, pending_wide_beams=pending_wide_beams,
                         native_action_frames=len(positions), observed_contact=False,
                         verified_exit_focus=has_precursor))
        if encoder.item_collection_input:
            rows[-1]["native_power_gain"] = max(0., game["power"] - power_before)
        if len(rows) % 64 == 0:
            write_json(args.output / "progress.json", dict(phase="capture_replay", PID=proc.pid,
                create_time=proc.create_time(), native_pid=controller.pid, rows=len(rows),
                raw_stage=stage.raw_stage, stage_frame=state["stage_timer"]))
    if state["input_poll"] >= limit and len(rows) < args.decisions:
        end_reason = "replay-input-limit"
    return rows, dict(requested_decisions=args.decisions, raw_stage=stage.raw_stage,
                      start_basis="replay_input_offset" if alignment else clock, requested_start=requested,
                      input_alignment=alignment, start_input_poll=start_poll,
                      start_stage_frame=start_frame, final_raw_stage=state["stage"],
                      final_stage_frame=state["stage_timer"], end_reason=end_reason,
                      skipped_transition_polls=skipped_transition_polls,
                      final_input_poll=state["input_poll"] - entry_poll,
                      input_schema=encoder.schema, observation_features=encoder.feature_shape[0],
                      native_audio=False, native_unpaced=controller.unpaced,
                      forecast_scripted_beams=controller.forecast_scripted_beams,
                      forecast_spawn_animations=controller.forecast_spawn_animations,
                      forecast_sweep_laser_births=args.forecast_sweep_laser_births,
                      forecast_wide_beam_births=args.forecast_wide_beam_births,
                      forecast_final_spark_cue=getattr(args, "forecast_final_spark_cue", False),
                      forecast_orbital_point_births=args.forecast_orbital_point_births,
                      forecast_final_b_ring=args.forecast_final_b_ring,
                      startup_mutex=state.get("startup_mutex"))


def save_windows(output, rows, raw_stage, native_seed, *, replay=None, policy=None, capture_status=None,
                 forecast_revision=None):
    histories = []
    source_kind = ("native_TH08_teacher_replay_window" if replay is not None
                   else "native_TH08_teacher_rolling_exit_window")
    commands = np.asarray([r["action"] for r in rows], np.int64)
    polls = np.asarray([r["native_poll"] for r in rows], np.int64)
    valid = commands >= 0
    consecutive = (np.diff(polls) == 4) & valid[:-1] & valid[1:]
    first = np.flatnonzero(valid & ~np.r_[False, consecutive][:len(valid)])
    last = np.flatnonzero(valid & ~np.r_[consecutive, False][:len(valid)]) + 1
    for left, right in zip(first, last, strict=True):
        length = int(right - left)
        if length < 16:
            continue
        exact = rows[left:right]
        filename = f"window-{int(left):06d}.npz"
        arrays = dict(features=np.stack([r["features"] for r in exact]),
                      actions=commands[left:right], native_poll=polls[left:right],
                      pending_wide_beams=np.asarray([r.get("pending_wide_beams", 0) for r in exact], np.uint8))
        if all("native_power_gain" in row for row in exact):
            arrays["native_power_gain"] = np.asarray([row["native_power_gain"] for row in exact], np.float32)
        if all("reference_logits" in row for row in exact):
            arrays["reference_logits"] = np.asarray([row["reference_logits"] for row in exact], np.float32)
        measured = all(row.get("native_action_frames") == 4 and row.get("observed_contact") is False for row in exact)
        if measured:
            arrays.update(native_frames=polls[left:right],
                          native_action_frames=np.asarray([r["native_action_frames"] for r in exact], np.uint8),
                          observed_contacts=np.asarray([r["observed_contact"] for r in exact], bool))
            if policy is not None:
                arrays["teacher_override"] = np.asarray([r.get("teacher_override", False) for r in exact], bool)
        np.savez_compressed(output / filename, **arrays)
        # Burn in from the earlier portion of this local replay segment.
        start = min(16, length - 16)
        task = dict(mode="route", stage=exact[0]["stage"],
                    end_event="native-replay-window" if replay is not None else "native-exit-window")
        history = dict(file=filename, task=task, native_seed=native_seed, split="train",
            raw_stage=raw_stage, decisions=length, nbnm_success=True, training_start=start,
            source_kind=source_kind,
            initial_memory="zero then exact measured local prefix",
            native_validation=dict(contacts=0, Bombs=0, stage_clear=False, task_complete=True),
            start_native_poll=exact[0]["native_poll"], end_native_poll=exact[-1]["native_poll"] + 4)
        if forecast_revision is not None:
            history["observation_forecast_revision"] = forecast_revision
        if capture_status is not None:
            history.update(first_stage_frame=exact[0]["stage_frame"], last_observed_stage_frame=exact[-1]["stage_frame"])
        history["scripted_beam_learning_actions"] = sum(row.get("scripted_beams", 0) > 0 for row in exact[start:])
        history["wide_beam_prelude_learning_actions"] = sum(row.get("pending_wide_beams", 0) > 0 for row in exact[start:])
        if measured:
            history["verified_native_exposure_rows"] = length
            if policy is not None:
                history["verified_teacher_overrides"] = int(arrays["teacher_override"][start:].sum())
            starts = [i for i in range(start, length - 11) if exact[i].get("verified_exit_focus")]
            if starts:
                focus = min(starts[0], length - 16)
                history["correction_windows"] = [[focus, focus + 16]]
                history["native_exit_focus_rows"] = len(starts)
        histories.append(history)
    manifest = dict(schema=SCHEMA, histories=histories, PPO_rows=0,
        observation_rows=len(rows), supported_commands=int((commands >= 0).sum()),
        scripted_beam_observation_rows=sum(r.get("scripted_beams", 0) > 0 for r in rows),
        wide_beam_prelude_observation_rows=sum(r.get("pending_wide_beams", 0) > 0 for r in rows),
        proof="four native input polls per command and zero recorded contact")
    if replay is not None:
        manifest.update(source_replay=str(replay), proof="four observed native positions per action, original Replay mode")
    elif policy is not None:
        manifest["source_policy"] = str(policy)
    if capture_status is not None:
        manifest["capture"] = capture_status
    write_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser = ArgumentParser(description=__doc__)
    for option in ("replay", "executable", "d3d8to9", "output"):
        parser.add_argument(f"--{option}", type=lambda path: Path(path).resolve(), required=True)
    parser.add_argument("--profile", type=Path, default=ROOT / "configs/th08-1.00d-zh-input.json")
    parser.add_argument("--raw-stage", type=int, choices=range(8), required=True)
    clock = parser.add_mutually_exclusive_group()
    clock.add_argument("--first-stage-frame", type=int, default=480)
    clock.add_argument("--first-input-poll", type=int,
                       help="Original input polls after entering this replay stage")
    clock.add_argument("--replay-input-offset", type=int,
                       help="Align to the input index in the original Replay file before capture")
    parser.add_argument("--decisions", type=int, default=512)
    parser.add_argument("--unpaced", action="store_true", help="Skip the original gameplay frame wait in Replay")
    parser.add_argument("--forecast-scripted-beams", action="store_true")
    parser.add_argument("--forecast-spawn-animations", action="store_true")
    parser.add_argument("--forecast-sweep-laser-births", action="store_true")
    parser.add_argument("--forecast-wide-beam-births", action="store_true")
    parser.add_argument("--forecast-orbital-point-births", action="store_true")
    parser.add_argument("--forecast-final-b-ring", action="store_true")
    parser.add_argument("--forecast-final-spark-cue", action="store_true")
    parser.add_argument("--item-collection-input", action="store_true",
                        help="Record native collection observations and actual Power gains")
    args = parser.parse_args()
    metadata, stages = read_replay(args.replay)
    stage = next((item for item in stages if item.raw_stage == args.raw_stage), None)
    if stage is None or metadata["team"] != 0 or metadata["difficulty"] != 3 or metadata["spell_practice"]:
        parser.error("Require a matching original Lunatic Border Team Story replay stage")
    if (args.decisions < 16 or args.first_stage_frame < 0
            or (args.first_input_poll is not None and not 0 <= args.first_input_poll < len(stage.inputs))
            or (args.replay_input_offset is not None and not 0 < args.replay_input_offset < len(stage.inputs))):
        parser.error("Select at least 16 decisions from a valid original replay starting clock")
    with gpu_phase("evaluation"):
        args.output.mkdir(parents=True, exist_ok=False)
        with TemporaryDirectory(prefix="private-game-", dir=args.output) as directory:
            workspace = Path(directory).resolve()
            if workspace.parent != args.output:
                raise ValueError("Private replay files must remain in the requested output")
            temporary = configure_native_temp(workspace)
            try:
                owned = training_worlds(args.executable, args.profile, args.d3d8to9, workspace, 1, {})[0]
                target = owned["executable"].parent / "replay"
                target.mkdir()
                copyfile(args.replay, target / "th8_01.rpy")
                with TH08InputController(owned["executable"], args.profile, private_appdata=owned["private_appdata"],
                                        d3d8to9=args.d3d8to9, forecast_scripted_beams=args.forecast_scripted_beams,
                                        forecast_spawn_animations=args.forecast_spawn_animations, unpaced=args.unpaced) as controller:
                    rows, capture_status = capture(controller, stage, args)
                save_windows(args.output, rows, stage.raw_stage, stage.rng_seed, replay=args.replay,
                             capture_status=capture_status)
            finally:
                cleanup_native_temp(temporary)


if __name__ == "__main__":
    main()
