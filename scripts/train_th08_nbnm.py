"""Fine-tune the shared whole-connectome policy on original TH08 Lunatic Story."""

import argparse
import math
from pathlib import Path
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fly_touhou.th08_training import portable_spec

SCHEMA = "th08-original-ready-nbnm-ppo-v1"
PPO_BATCH_DECISIONS = 256
ENCODER_AND_HEADS = frozenset((
    "inherited_sensory_ports", "new_context_ports", "movement", "risk", "native_warning48",
    "corridor_risk", "corridor_witness", "corridor_course", "corridor_endpoint",
))
CAPTURE_READOUTS = frozenset(("movement", "three_critics"))
SURVIVAL_READOUTS = CAPTURE_READOUTS | {"risk", "corridor_risk", "corridor_witness"}
ORIGINAL_GEOMETRY_UPGRADES = ("forecast_spawn_animations", "forecast_scripted_beams",
                             "forecast_sweep_laser_births", "forecast_wide_beam_births",
                             "forecast_orbital_point_births", "forecast_final_b_ring",
                             "forecast_native_branches", "forecast_native_narrow_exits",
                             "forecast_native_carried_courses", "forecast_native_hold_fallback",
                             "forecast_native_corner_turns", "forecast_final_spark_cue")


def stage_rehearsal_tasks(plan, decisions):
    stages = plan.get("teacher_stage_cycle", ())
    if not stages:
        return ()
    segment_size = plan["segment_decisions"]
    slots = segment_size // (PPO_BATCH_DECISIONS * plan["teacher_every"])
    segment_index = (decisions - 1) // segment_size
    first = segment_index * slots
    return tuple(stages[index % len(stages)] for index in range(first, first + slots))


def resolve_course(path, workers):
    from fly_touhou.nbnm_run import read
    from fly_touhou.th08_prefix import read_prefix_source, replay_practice_slot

    source = read(path)
    lanes = source["lanes"]
    kinds = ("full_story", "stage", "nonspell", "spell")
    if len(lanes) != workers or any(not lane for lane in lanes):
        raise ValueError("Give each worker one nonempty original course lane")
    observed = set()
    for lane in lanes:
        kind = lane[0]["kind"]
        group = lane[0].get("quota_group", kind)
        if kind not in kinds or any(row["kind"] != kind or row.get("quota_group", kind) != group for row in lane):
            raise ValueError("Keep a single task category per course lane")
        observed.add(kind)
        for row in lane:
            if kind == "nonspell":
                report_path = Path(row["report"]).resolve(strict=True)
                decision = row["decision"]
                report, _, _, trace = read_prefix_source(report_path, through_decision=decision)
                release = trace[-1]
                if (decision < 16 or len(trace) != decision + 1 or release["native_spell_id"] is not None
                        or type(row["end_spell_id"]) is not int or not 0 <= row["end_spell_id"] < 255):
                    raise ValueError(f"Use a recorded original nonspell entrance from {report_path}")
                slot = replay_practice_slot(report)
                if report["entry_mode"] == "original-spell-practice":
                    raise ValueError("Nonspells require an original Story or Stage Practice entrance")
                row.update(report=str(report_path), seed=report["native_startup_seed"],
                           raw_stage=release["raw_stage"], spell_id=None, practice_slot=slot)
            else:
                if type(row.get("seed")) is not int or not 0 <= row["seed"] < 65536:
                    raise ValueError("Use a 16-bit original seed for each course task")
                if kind in ("stage", "spell") and not 0 <= row["practice_slot"] < 8:
                    raise ValueError("Choose an original practice stage")
                if kind == "spell":
                    if not (0 <= row["spell_slot"] < 44 and 0 <= row["spell_id"] < 255):
                        raise ValueError("Choose an original Spell Practice entry")
                    row["raw_stage"] = row["practice_slot"]
    if observed != set(kinds):
        raise ValueError("Cover full Story, stage, nonspell and spell practice")
    return lanes


def validate_training_parent(stored, plan, spec, *, extending, upgrade_geometry=False, retarget_course=False):
    if stored.get("schema") != SCHEMA:
        raise ValueError("Use an original TH08 training checkpoint")
    from fly_touhou.training.run_contract import restore_contract
    restore_contract(stored, spec.get("run_contract"))
    previous_spec = portable_spec(stored["spec"], native_forecasts=stored["plan"].get("forecast_native_branches", False))
    upgrades = []
    if extending and not previous_spec["nbnm"]["forecast_native_branches"]:
        previous_spec["nbnm"]["forecast_native_branches"] = True
        upgrades.append("forecast_native_branches")
    if extending and upgrade_geometry:
        for key in ORIGINAL_GEOMETRY_UPGRADES:
            if spec["nbnm"].get(key) is True and not previous_spec["nbnm"].get(key, False):
                previous_spec["nbnm"][key] = True
                upgrades.append(key)
        if not upgrades:
            raise ValueError("Select a new original geometry observation to enable")
    power_upgrade = (extending and spec["nbnm"].get("native_capture_power_guidance")
                     and not previous_spec["nbnm"].get("native_capture_power_guidance"))
    if power_upgrade:
        previous_spec["nbnm"]["native_capture_power_guidance"] = True
        upgrades.append("native_capture_power_guidance")
    if previous_spec != spec:
        raise ValueError("Use a TH08 training checkpoint with the same observation and model settings")
    previous = stored["plan"]
    if extending:
        if (stored["pending"] is not None or (not retarget_course and stored["decisions"] != previous["decisions"])
                or plan["decisions"] <= stored["decisions"]):
            raise ValueError("Extend from a fully updated segment to a larger cumulative decision budget")
        previous = dict(previous, decisions=plan["decisions"])
        if retarget_course:
            if not previous.get("course_lanes") or not plan.get("course_lanes"):
                raise ValueError("Retargeting requires an existing original course")
            for key in ("course", "course_lanes", "course_weights", "course_source_outputs", "course_focus_spells",
                        "native_start_retries", "teacher_banks", "teacher_every", "teacher_loss_weight"):
                if key in plan:
                    previous[key] = plan[key]
            for key in ("movement_sampling_temperature", "rehearsal_focus_order"):
                if key in plan:
                    previous[key] = plan[key]
                else:
                    previous.pop(key, None)
        previous.pop("parameter_snapshots", None)
        if plan.get("parameter_snapshots"):
            previous["parameter_snapshots"] = plan["parameter_snapshots"]
        if plan.get("native_unpaced"):
            previous["native_unpaced"] = True
        else:
            previous.pop("native_unpaced", None)
        if upgrades:
            previous["spec"] = plan["spec"]
            if "forecast_native_branches" in upgrades:
                previous["forecast_native_branches"] = True
            for key in ("forecast_native_narrow_exits", "forecast_native_carried_courses",
                        "forecast_native_hold_fallback", "forecast_native_corner_turns", "forecast_final_spark_cue"):
                if key in upgrades:
                    previous[key] = True
        old_windows = previous.get("teacher_windows", [])
        new_windows = plan.get("teacher_windows", [])
        if (len(new_windows) > len(old_windows) and new_windows[:len(old_windows)] == old_windows
                and plan["teacher_histories"] > previous.get("teacher_histories", 0)):
            previous.update(teacher_windows=new_windows, teacher_histories=plan["teacher_histories"])
        if plan.get("practice_stages"):
            previous["practice_stages"] = plan["practice_stages"]
        if plan.get("capture_reference") and plan["capture_reference"] == previous.get("capture_reference"):
            previous["capture_entries"] = plan["capture_entries"]
            previous["native_rng_controlled"] = plan["native_rng_controlled"]
            if power_upgrade:
                previous["capture_power_guidance"] = True
        if "nbnm_entries" in plan and "nbnm_entries" in previous:
            for key in ("nbnm_entries", "native_rng_controlled", "workers"):
                previous[key] = plan[key]
            if "practice_stages" not in plan:
                previous.pop("practice_stages", None)
        if plan.get("teacher_native_exit_weight"):
            for key in ("teacher_native_exit_weight", "teacher_native_exit_histories", "teacher_stage_cycle"):
                if key in plan:
                    previous[key] = plan[key]
    if previous != plan:
        raise ValueError("Retain the collection and learning settings when continuing TH08 training")
    return upgrades


def run(args, plan, spec, teacher):
    if sys.platform == "win32":
        from multiprocessing import set_executable

        # Use the real GUI interpreter so worker PID ownership remains exact.
        # Spawn restores the trainer's sys.path, including its venv packages.
        pythonw = Path(sys._base_executable).with_name("pythonw.exe")
        if pythonw.is_file():
            set_executable(str(pythonw))

    from fly_touhou.nbnm_run import atomic_torch_save, load_cpu_checkpoint, load_model, runtime
    from fly_touhou.thrl_durable_records import append_jsonl, write_json
    from fly_touhou.thrl_job_paths import cleanup_native_temp

    out = args.output
    checkpoint_path = out / "latest-training.pt"
    tape_path = out / "latest-rollout.pt"
    parent = checkpoint_path if args.resume else args.continue_from
    stored = load_cpu_checkpoint(parent) if parent is not None else None
    if stored is not None:
        upgrades = validate_training_parent(stored, plan, spec, extending=args.continue_from is not None,
                                           upgrade_geometry=args.upgrade_original_geometry,
                                           retarget_course=args.retarget_course)
    if not args.resume:
        out.mkdir(parents=True, exist_ok=False)
        write_json(out / "plan.json", plan)
        if stored is not None:
            write_json(out / "continued-from.json", dict(
                checkpoint=str(parent), decisions=stored["decisions"], optimizer_updates=stored["optimizer_updates"],
                original_geometry_upgrades=upgrades, retargeted_course=args.retarget_course))
    temporary = runtime(out)
    import psutil
    import torch

    from fly_touhou.compact_story_ppo import optimizer_for, update
    from fly_touhou.compact_story_ready import ReadyCollector
    from fly_touhou.nbnm_corridor import SCHEMA as OBSERVATION_SCHEMA
    from fly_touhou.nbnm_cuda import InferenceGraphs
    from fly_touhou.th08_native_forecast import FORECAST_REVISION
    from fly_touhou.th08_rehearsal import TH08RouteRehearsal, build_catalog, student_spell_ids, training_coverage
    from fly_touhou.th08_training import factory, training_worlds

    proc = psutil.Process()
    proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    start = time.monotonic()
    decisions = stored["decisions"] if stored is not None else 0
    updates = stored["optimizer_updates"] if stored is not None else 0
    pending = stored["pending"] if stored is not None else None
    collector = graphs = None
    parameter_snapshots = frozenset(plan.get("parameter_snapshots", ()))
    course_lanes = plan.get("course_lanes", ())
    teacher_banks = plan.get("teacher_banks", ())
    sources = tuple(Path(path) for path in plan.get("course_source_outputs", ()))
    focus_spells = plan.get("course_focus_spells", ())
    focus_order = plan.get("rehearsal_focus_order", ())
    sampling_temperature = plan.get("movement_sampling_temperature", 1.)
    retries = plan.get("native_start_retries", 0)
    student_bank = out / "route-library" / "student"
    inherited_student_banks = [path / "route-library" / "student" for path in sources]
    spell_ids = {row["spell_id"] for lane in course_lanes for row in lane if row["kind"] == "spell"}
    nonspell_ids = {row["end_spell_id"] for lane in course_lanes for row in lane if row["kind"] == "nonspell"}
    initial_catalog = out / "rehearsal-catalogs" / "initial.json"

    def save_coverage():
        if course_lanes:
            coverage = training_coverage((*sources, out), spell_ids, nonspell_ids)
            write_json(out / "training-coverage.json", dict(coverage, completed_decisions=decisions,
                optimizer_updates=updates, expected_spells=sorted(spell_ids), expected_nonspells=sorted(nonspell_ids)))

    def progress(phase, **detail):
        write_json(out / "progress.json", dict(phase=phase, PID=proc.pid, create_time=proc.create_time(),
            original_game="TH08", difficulty="Lunatic", completed_decisions=decisions, optimizer_updates=updates,
            budget=args.decisions, elapsed_seconds=time.monotonic() - start,
            native_audio=False, native_unpaced=args.unpaced,
            movement_sampling_temperature=sampling_temperature,
            forecast_native_branches=plan["forecast_native_branches"],
            observation_forecast_revision=FORECAST_REVISION, **detail))

    def save():
        parameters = model.parameter_payload(dict(TH08_decisions=decisions, TH08_PPO_updates=updates))
        from fly_touhou.training.run_contract import mark_parameters
        mark_parameters(parameters, spec.get("run_contract"))
        if pending is None and decisions in parameter_snapshots:
            atomic_torch_save(parameters, out / f"parameters-{decisions:06d}.pt")
        atomic_torch_save(dict(schema=SCHEMA, plan=plan, spec=spec, model_payload=parameters, optimizer=optimizer.state_dict(),
            decisions=decisions, optimizer_updates=updates, pending=pending,
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()), checkpoint_path)

    try:
        progress("loading")
        torch.manual_seed(spec["seed"])
        torch.cuda.manual_seed_all(spec["seed"])
        from fly_touhou.games.item_collection import input_schema as collection_schema
        model, migrated = load_model(spec, args.checkpoint,
            payload=stored["model_payload"] if stored is not None else None, apply_native_sensors=False,
            reset_collection_line=True)
        expected_schema = collection_schema(OBSERVATION_SCHEMA, spec["nbnm"].get("item_collection_input", False))
        if migrated or model.input_schema != expected_schema or not model.soft_risk_composition:
            raise ValueError("Supply a compatible NBNM whole-connectome policy")
        if course_lanes and not model.bombs_disabled:
            raise ValueError("The original NBNM course needs Bomb disabled")
        model.native_contact_risk_calibration = True
        groups = {"encoder-heads": ENCODER_AND_HEADS, "capture-readouts": CAPTURE_READOUTS,
                  "survival-readouts": SURVIVAL_READOUTS}.get(args.train_scope)
        if spec.get("run_contract"):
            identity = spec["run_contract"]
            if identity["train_scope"] != args.train_scope:
                raise ValueError("The public configuration and TH08 training scope differ")
            groups = identity["optimizer_groups"]
        optimizer = optimizer_for(model, trainable_groups=groups, include_native_attack_readout=bool(course_lanes))
        if groups is not None:
            trained = {name: param.numel() for name, param in model.named_parameters() if param.requires_grad}
            frozen = {name: param.numel() for name, param in model.named_parameters() if not param.requires_grad}
            write_json(out / "parameter-scope.json", dict(scope=args.train_scope,
                optimizer_groups=[group["name"] for group in optimizer.param_groups],
                trained=trained, frozen=frozen))
        if stored is not None:
            optimizer.load_state_dict(stored["optimizer"])
            torch.set_rng_state(stored["torch_rng"])
            torch.cuda.set_rng_state_all(stored["cuda_rng"])
            del stored
        if not args.resume:
            save()
        if teacher_banks and not initial_catalog.exists():
            build_catalog(initial_catalog, teacher_banks, student_bank, inherited_student_banks)
        save_coverage()
        stop_path = out / "STOP"
        while (pending is not None or decisions < args.decisions) and not stop_path.exists():
            if pending is None:
                if collector is None:
                    progress("starting_private_story_workers")
                    worlds = training_worlds(args.executable, args.profile, args.d3d8to9,
                                             out / "world-games", args.workers,
                                             dict(spec["nbnm"], native_unpaced=args.unpaced, native_start_retries=retries,
                                                  course_focus_spells=focus_spells),
                                             practice_slots=plan.get("practice_stages", []),
                                             capture_entries=plan.get("capture_entries", plan.get("nbnm_entries", [])),
                                             course_lanes=course_lanes, route_output=student_bank if teacher_banks else None)
                    collector = ReadyCollector(model, {}, out / f"native-{proc.pid}-{int(start * 1000)}",
                                               workers=args.workers, world_factory=factory, world_kwargs=worlds,
                                               movement_sampling_temperature=sampling_temperature)
                    if course_lanes:
                        collector.worker_groups = [lane[0].get("quota_group", lane[0]["kind"]) for lane in course_lanes]
                        collector.group_weights = plan["course_weights"]
                    collector.actions = decisions
                    graphs = InferenceGraphs(model, movement_temperature=sampling_temperature,
                                             packed_binary_messages=spec.get("packed_binary_messages", False))
                    collector.inference_graphs = graphs
                count = min(args.segment_decisions, args.decisions - decisions)
                progress("sampling")
                tape = collector.rollout(model, total_decisions=count,
                                         progress=lambda row: progress("sampling", sampling=row))
                graphs.clear()
                atomic_torch_save(tape, tape_path)
                previous_decisions = decisions
                decisions += count
                pending = dict(logs=[], collection=tape["metadata"])
                if teacher_banks:
                    catalog = out / "rehearsal-catalogs" / f"after-{decisions:07d}.json"
                    rows = build_catalog(catalog, teacher_banks, student_bank, inherited_student_banks)
                    pending["teacher_catalog"] = str(catalog)
                    if focus_spells:
                        from fly_touhou.thrl_study_io import read_shared_json
                        previous_catalog = catalog.with_name(f"after-{previous_decisions:07d}.json")
                        prior = read_shared_json(previous_catalog if previous_catalog.exists() else initial_catalog)
                        new_spells = student_spell_ids(rows) - student_spell_ids(prior["histories"])
                        pending["first_success_spells"] = sorted(new_spells.intersection(focus_spells))
                save()
                save_coverage()
            else:
                tape = load_cpu_checkpoint(tape_path)
            if stop_path.exists():
                break
            origin = updates - len(pending["logs"])
            rehearsal_tasks = stage_rehearsal_tasks(plan, decisions)
            if teacher_banks:
                first = pending.get("first_success_spells", ())
                teacher = TH08RouteRehearsal(pending["teacher_catalog"], focus_spells=focus_spells,
                                             first_successes=first, focus_order=focus_order)
                segment_index = (decisions - 1) // args.segment_decisions
                slots = args.segment_decisions // (PPO_BATCH_DECISIONS * plan["teacher_every"])
                rehearsal_tasks = teacher.scheduled_tasks(segment_index, slots)
                progress("teacher-library-ready", teacher_routes=sum("teacher_certificate" in row for row in teacher.rows),
                         student_routes=sum(row["source_kind"] == "native_TH08_student_success" for row in teacher.rows),
                         catalog=pending["teacher_catalog"], first_success_spell_ids=list(first))

            def learned(row):
                nonlocal updates
                if "trained_student_rows" in row:
                    pending["logs"].append(row)
                    updates = origin + row["update"]
                    if row["update"] % 8 == 0 or stop_path.exists():
                        save()
                progress("updating", learning=row)

            report = update(model, optimizer, tape, epochs=1, micro_scenes=16, seed=spec["seed"] + decisions,
                            checkpoint_chunks=True, target_kl=None, native_only_corridors=True,
                            rule_archive_replay=teacher, rule_archive_coefficient=plan.get("teacher_loss_weight", 0.),
                            rule_archive_every=plan.get("teacher_every", 8),
                            rule_archive_tasks=rehearsal_tasks,
                            completed_update_prefix=tuple(pending["logs"]), progress=learned)
            segment = dict(decisions=decisions, optimizer_updates=updates, sampling=pending["collection"],
                           learning=report)
            if teacher_banks:
                segment["teacher_catalog"] = pending["teacher_catalog"]
                segment["first_success_spell_ids"] = list(pending.get("first_success_spells", ()))
            pending = None
            save()
            append_jsonl(out / "segments.jsonl", segment)
            save_coverage()
            del tape
        finished = decisions == args.decisions and pending is None
        parameters = model.parameter_payload(dict(TH08_decisions=decisions, TH08_PPO_updates=updates))
        atomic_torch_save(parameters, out / "latest-parameters.pt")
        summary = dict(completed=finished, sampled_decisions=decisions, optimizer_updates=updates,
                       rollout_pending=pending is not None, native_rng_controlled=plan["native_rng_controlled"],
                       forecast_native_branches=plan["forecast_native_branches"],
                       observation_forecast_revision=FORECAST_REVISION,
                       train_scope=args.train_scope)
        write_json(out / "result.json", summary)
        progress("complete" if finished else "paused", summary=summary)
    except BaseException as error:
        failures = [{k: row[k] for k in ("index", "operation", "error", "pending_workers", "commands") if k in row}
                    for row in getattr(error, "failures", ())]
        write_json(out / "failure.json", dict(error=traceback.format_exc(), PID=proc.pid, native_failures=failures))
        raise
    finally:
        if graphs is not None:
            graphs.clear()
        if collector is not None:
            collector.close()
        cleanup_native_temp(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("spec", "checkpoint", "executable", "d3d8to9", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--profile", type=Path, default=ROOT / "configs" / "th08-1.00d-zh-input.json")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--decisions", type=int, default=8192)
    parser.add_argument("--segment-decisions", type=int, default=2048)
    parser.add_argument("--movement-sampling-temperature", type=float, default=1.,
                        help="Temperature for student movement sampling during PPO")
    parser.add_argument("--teacher-windows", type=Path, action="append", default=[],
                        help="Optional manifest from capture_th08_teacher_windows.py")
    parser.add_argument("--teacher-bank", type=Path, action="append", default=[],
                        help="Scan completed original teacher route deposits at each PPO segment")
    parser.add_argument("--course", type=Path, help="Rotating original Story, Stage, nonspell and spell lanes")
    parser.add_argument("--teacher-loss-weight", type=float, default=.25)
    parser.add_argument("--teacher-native-exit-weight", type=float, default=0.,
                        help="Train the exit head on verified contact-free 48/96-frame native Replay suffixes")
    parser.add_argument("--teacher-every", type=int,
                        help="Optimizer updates between teacher windows, cycling through their original stages")
    parser.add_argument("--practice-stage", type=int, choices=range(8), action="append", default=[],
                        help="Assign one worker to this native Practice Start slot, 0 through 7")
    parser.add_argument("--capture-reference", type=Path,
                        help="Learn the full native spell sequence from its verified original replay receipt")
    parser.add_argument("--capture-entry", action="append", nargs=2, metavar=("REPORT", "DECISION"), default=[],
                        help="Assign one worker to a spell checkpoint in a recorded original Story trace")
    parser.add_argument("--capture-practice", action="append", nargs=4, type=int, default=[],
                        metavar=("STAGE", "SLOT", "SPELL", "SEED"),
                        help="Assign one worker to a required card in native Spell Practice")
    parser.add_argument("--capture-power-guidance", action="store_true",
                        help="Learn native Power pickup preferences among contact-free opening routes")
    parser.add_argument("--nbnm-entry", action="append", nargs=2, metavar=("REPORT", "DECISION"), default=[],
                        help="Resume a recorded original Story or practice at a verified decision")
    parser.add_argument("--nbnm-spell", action="append", nargs=4, type=int, default=[],
                        metavar=("STAGE", "SLOT", "SPELL", "SEED"),
                        help="Train survival to the end of an original spell, including a safe timeout")
    parser.add_argument("--native-survival-guidance", action="store_true",
                        help="Train on decisive collision contrasts from original candidate tapes")
    parser.add_argument("--train-scope", choices=("joint", "encoder-heads", "capture-readouts", "survival-readouts"),
                        default="joint", help="Select the trainable parameter groups")
    parser.add_argument("--unpaced", action="store_true", help="Skip the original gameplay frame wait in silent workers")
    parser.add_argument("--save-parameters-at", type=int, action="append", default=[], metavar="DECISIONS",
                        help="Keep frozen parameters after a complete update segment")
    parser.add_argument("--execute", action="store_true")
    continuation = parser.add_mutually_exclusive_group()
    continuation.add_argument("--resume", action="store_true")
    continuation.add_argument("--continue-from", type=Path,
                              help="Extend a run with its model and optimizer in a new output directory")
    parser.add_argument("--retarget-course", action="store_true",
                        help="Continue a saved segment with new course quotas and original-route priorities")
    parser.add_argument("--upgrade-original-geometry", action="store_true",
                        help="Enable verified native forecasts in a fresh extension of a completed run")
    args = parser.parse_args()
    if args.upgrade_original_geometry and args.continue_from is None:
        parser.error("--upgrade-original-geometry requires a completed --continue-from checkpoint")
    if args.retarget_course and (args.continue_from is None or args.course is None):
        parser.error("--retarget-course requires --continue-from and --course")
    if not 1 <= args.workers <= 100 or args.segment_decisions < max(256, args.workers * 16):
        parser.error("Use 1 to 100 workers and at least 16 decisions per worker in a segment")
    prefix_entries = args.capture_entry + args.nbnm_entry
    practice_spells = args.capture_practice + args.nbnm_spell
    if args.course and (args.practice_stage or prefix_entries or practice_spells or args.capture_reference):
        parser.error("The course file defines all original starts")
    if args.teacher_bank and (not args.course or args.teacher_windows or args.teacher_native_exit_weight):
        parser.error("Live teacher banks use course rehearsal with measured successful movement")
    if len(args.practice_stage) + len(prefix_entries) + len(practice_spells) > args.workers:
        parser.error("The number of practice and capture workers cannot exceed the total")
    if (args.capture_entry or args.capture_practice) and args.capture_reference is None:
        parser.error("Original capture entries require their verified spell sequence")
    if args.capture_power_guidance and args.capture_reference is None:
        parser.error("Power guidance requires the capture objective")
    if args.capture_reference is not None and (args.nbnm_entry or args.nbnm_spell or args.native_survival_guidance):
        parser.error("Choose either the NBNM survival or full spell capture objective")
    if args.segment_decisions % PPO_BATCH_DECISIONS or args.decisions <= 0 or args.decisions % args.segment_decisions:
        parser.error("Use whole segments made up of 256-decision PPO batches")
    if any(not 0 < step < args.decisions or step % args.segment_decisions for step in args.save_parameters_at):
        parser.error("Save parameter snapshots at intermediate completed segment boundaries")
    if args.teacher_every is not None and (
            not (args.teacher_windows or args.teacher_bank)
            or not 1 <= args.teacher_every <= args.segment_decisions // PPO_BATCH_DECISIONS):
        parser.error("Stage-balanced rehearsal requires teacher windows and at least one teacher update per segment")
    if not math.isfinite(args.movement_sampling_temperature) or args.movement_sampling_temperature <= 0:
        parser.error("Movement sampling temperature must be finite and positive")
    if not math.isfinite(args.teacher_loss_weight) or args.teacher_loss_weight < 0:
        parser.error("Teacher movement loss weight must be finite and nonnegative")
    if not 0. <= args.teacher_native_exit_weight <= 1. or (args.teacher_native_exit_weight and not args.teacher_windows):
        parser.error("Specify a weight from zero to one and provide measured teacher windows for exit supervision")
    paths = ("spec", "checkpoint", "executable", "d3d8to9", "profile")
    for name in paths:
        setattr(args, name, getattr(args, name).resolve(strict=True))
    if args.continue_from is not None:
        args.continue_from = args.continue_from.resolve(strict=True)
    args.output = args.output.resolve()
    from fly_touhou.nbnm_run import read

    source = read(args.spec)
    entries = []
    sequence = None
    if args.capture_reference is not None:
        from fly_touhou.th08_game_state import captured_spells

        args.capture_reference = args.capture_reference.resolve(strict=True)
        reference = read(args.capture_reference)
        sequence = captured_spells(reference)
        if reference.get("reference_spell_sequence_verified") is not True or sequence is None:
            parser.error("Use a complete spell sequence verified in original TH08 Replay")
        source["nbnm"]["native_capture_sequence"] = [list(pair) for pair in sequence]
        if args.capture_power_guidance:
            source["nbnm"]["native_capture_power_guidance"] = True
        if any(not any(raw == slot for raw, _ in sequence) for slot in args.practice_stage):
            parser.error("Select a practice stage with spells in the verified route")
    if args.native_survival_guidance:
        source["nbnm"]["native_survival_guidance"] = True
    if prefix_entries:
        from fly_touhou.th08_prefix import read_prefix_source, replay_practice_slot

    for report_path, decision in prefix_entries:
        path = Path(report_path).resolve(strict=True)
        index = int(decision)
        if index < 16:
            parser.error("Select an original decision after the burn-in")
        report, _, _, trace = read_prefix_source(path, through_decision=index)
        practice = replay_practice_slot(report)
        if sequence is not None and practice is not None:
            parser.error("A capture-sequence prefix must come from a seeded original Story trace")
        if len(trace) != index + 1:
            parser.error("The recorded original trace does not reach the selected decision")
        row = trace[-1]
        if sequence is not None and (row["raw_stage"], row["native_spell_id"]) not in sequence:
            parser.error("The capture entry must be inside an observed required spell")
        entry = dict(report=str(path), decision=index, raw_stage=row["raw_stage"],
                     spell_id=row["native_spell_id"], seed=report["native_startup_seed"])
        if practice is not None:
            entry.update(practice_slot=practice, prefix_spell_practice=report["entry_mode"] == "original-spell-practice")
        entries.append(entry)
    for stage, slot, spell, seed in practice_spells:
        if not (0 <= stage < 8 and 0 <= slot < 44 and spell >= 0 and 0 <= seed <= 65535):
            parser.error("Select an original spell ID, its native stage and menu slot, and a 16-bit seed")
        if sequence is not None and (stage, spell) not in sequence:
            parser.error("Select a required spell in the verified route")
        entries.append(dict(raw_stage=stage, spell_slot=slot, spell_id=spell, seed=seed))
    spec = portable_spec(source)
    plan = dict(schema=SCHEMA, **{key: str(getattr(args, key)) for key in paths},
                workers=args.workers, decisions=args.decisions, segment_decisions=args.segment_decisions,
                native_rng_controlled=len(entries) == args.workers, resume_restarts_native_games=True,
                action_source="student", corridor_labels="executed native outcomes only")
    if args.movement_sampling_temperature != 1.:
        plan["movement_sampling_temperature"] = args.movement_sampling_temperature
    if args.capture_reference is not None:
        plan.update(capture_reference=str(args.capture_reference), capture_entries=entries,
                    capture_goal="original sequence with no missed capture",
                    attack_supervision="damage preference among verified contact-free 96-frame routes")
    elif entries:
        plan.update(nbnm_entries=entries, completion_goal="original NBNM without required spell captures")
    if args.native_survival_guidance:
        plan["native_survival_guidance"] = "verified collision contrasts among original candidate tapes"
    if args.capture_power_guidance:
        plan["capture_power_guidance"] = True
    if args.unpaced:
        plan["native_unpaced"] = True
    plan["forecast_native_branches"] = spec["nbnm"]["forecast_native_branches"]
    for key in ("forecast_native_narrow_exits", "forecast_native_carried_courses",
                "forecast_native_hold_fallback", "forecast_native_corner_turns", "forecast_final_spark_cue"):
        if spec["nbnm"].get(key, False):
            plan[key] = True
    if args.save_parameters_at:
        plan["parameter_snapshots"] = sorted(set(args.save_parameters_at))
    if args.practice_stage:
        plan["practice_stages"] = args.practice_stage
    if args.train_scope != "joint":
        plan["train_scope"] = args.train_scope
    if args.course:
        path = args.course.resolve(strict=True)
        lanes = resolve_course(path, args.workers)
        settings = read(path)
        groups = [lane[0].get("quota_group", lane[0]["kind"]) for lane in lanes]
        weights = settings.get("quota_weights", {kind: groups.count(kind) / args.workers for kind in sorted(set(groups))})
        from fly_touhou.nbnm_sampling import group_block_budget
        group_block_budget(groups, weights, args.segment_decisions // 16)
        plan.update(course=str(path), course_lanes=lanes, native_rng_controlled=True,
                    course_weights=weights,
                    completion_goal="original NBNM without required spell captures",
                    train_native_attack_readout=True)
        if settings.get("source_outputs"):
            plan["course_source_outputs"] = [str(Path(folder).resolve(strict=True)) for folder in settings["source_outputs"]]
        if settings.get("focus_spells"):
            plan["course_focus_spells"] = settings["focus_spells"]
        if settings.get("rehearsal_focus_order"):
            focus_order = settings["rehearsal_focus_order"]
            if (not args.teacher_bank or not isinstance(focus_order, list)
                    or any(type(spell) is not int or spell not in plan.get("course_focus_spells", ())
                           for spell in focus_order)):
                parser.error("Choose focused spell IDs from the course for the teacher rehearsal order")
            plan["rehearsal_focus_order"] = focus_order
        if "native_start_retries" in settings:
            retries = settings["native_start_retries"]
            if type(retries) is not int or not 0 <= retries <= 2:
                parser.error("Use at most two retries per original reset")
            plan["native_start_retries"] = retries
    if args.teacher_bank:
        plan.update(teacher_banks=[str(path.resolve(strict=True)) for path in args.teacher_bank],
                    teacher_loss_weight=args.teacher_loss_weight, teacher_every=args.teacher_every or 4)
    teacher = None
    if args.teacher_windows:
        from fly_touhou.nbnm_rehearsal import SuccessRehearsal

        manifests = [path.resolve(strict=True) for path in args.teacher_windows]
        teacher = SuccessRehearsal(manifests[0], additional_manifests=manifests[1:],
                                   native_exit_coefficient=args.teacher_native_exit_weight,
                                   include_supported_native_exits=bool(args.teacher_native_exit_weight))
        sources = {"native_TH08_teacher_replay_window", "native_TH08_teacher_rolling_exit_window"}
        if any(row.get("source_kind") not in sources for row in teacher.rows):
            parser.error("Use measured teacher windows from original TH08")
        plan.update(teacher_windows=[str(path) for path in manifests], teacher_histories=len(teacher.rows),
                    teacher_loss_weight=args.teacher_loss_weight)
        measured = [row for row in teacher.rows if row.get("verified_native_exposure_rows")]
        if args.teacher_native_exit_weight:
            if not measured:
                parser.error("Re-capture original Replay windows with measured action durations and contact outcomes")
            plan.update(teacher_native_exit_weight=args.teacher_native_exit_weight,
                        teacher_native_exit_histories=len(measured))
        if args.teacher_every is not None:
            for row in teacher.rows:
                exit_focus = (args.teacher_native_exit_weight and row.get("verified_native_exposure_rows")
                              and row.get("native_exit_focus_rows"))
                stage = row["task"]["stage"]
                task = f"TH08 verified exit stage {stage}" if exit_focus else f"TH08 stage {stage}"
                group = row.get("rehearsal_group") if row.get("verified_native_exposure_rows") else None
                row["task_key"] = f"{task} ({group})" if group else task
            stage_cycle = sorted({row["task_key"] for row in teacher.rows})
            plan.update(teacher_every=args.teacher_every, teacher_stage_cycle=stage_cycle)
    if not args.execute:
        import json
        print(json.dumps(plan, indent=2))
        return
    from fly_touhou.nbnm_job_phase import gpu_phase

    with gpu_phase("training"):
        run(args, plan, spec, teacher)


if __name__ == "__main__":
    main()
