"""Collect measured escape windows from the student's own original TH08 play."""
import argparse
from collections import deque
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from capture_th08_teacher_windows import save_windows
from fly_touhou.nbnm_job_phase import gpu_phase
from fly_touhou.nbnm_rehearsal import SCHEMA
from fly_touhou.th08_native_forecast import FORECAST_REVISION
from fly_touhou.th08_sensation import POLICY_MASKS
from fly_touhou.th08_training import factory, portable_spec, training_worlds
from fly_touhou.thrl_durable_records import write_json
from fly_touhou.thrl_job_paths import cleanup_native_temp


def ranked_exit(scene, scores):
    """Choose a supported continuation when the policy's first choice has none."""
    courses = scene["corridor_courses"]
    proposed = int(scores.argmax())
    for horizon in (1, 0):
        exists = courses[:, horizon, 0] >= 0
        if not exists.any():
            continue
        if exists[proposed]:
            return ()
        choices = np.flatnonzero(exists)
        selected = choices[scores[choices].argmax()]
        return tuple(int(command) for command in courses[selected, horizon] if command >= 0)
    immediate, next_twelve = scene["margins"][:2]
    choices = np.flatnonzero(immediate > 1.)
    if immediate[proposed] <= 0. and len(choices):
        selected = max(choices, key=lambda i: (next_twelve[i], immediate[i], scores[i]))
        return (int(selected),)
    return ()


def collect(env, model, graphs, args, retain, progress):
    import torch

    count = 0
    leads = 0
    earlier_overrides = 0
    observation, info = env.reset()
    brain = model.initial_state(1)
    previous = deque(maxlen=16)
    course = deque()
    candidate = None
    recent = deque(maxlen=16)

    def summary(outcome, state):
        fields = ("stage", "raw_stage", "stage_frame", "native_spell_id", "native_frames", "contact", "misses", "Bombs")
        return dict(decisions=count, teacher_opportunities=leads, earlier_teacher_overrides=earlier_overrides,
                    outcome=outcome,
                    native_state={name: state.get(name) for name in fields},
                    last_decisions=list(recent), first_contact=state.get("compact_contact_evidence"))

    while observation is not None and count < args.max_decisions:
        if (args.output / "STOP").exists():
            return summary("stopped", info)
        world = env.world
        scene = world.encoder.last_result
        features = observation["features"]
        inputs = torch.as_tensor(features[None], device=model.device)
        before = brain
        pred, brain, _ = graphs.frozen_batch(inputs, brain, 1)
        proposed = int(pred["actions"][0])
        action = proposed
        precursor = bool(scene["pending_laser_births"] or scene["pending_wide_beams"]
                         or scene["pending_orbital_points"] or scene["pending_ring_bullets"])
        matching_spell = args.spell_id is None or info["native_spell_id"] == args.spell_id
        eligible = (matching_spell and info["stage_frame"] >= args.first_stage_frame
                    and (precursor or args.include_ordinary_closures))
        imminent_contact = scene["margins"][0, proposed] <= 0.
        can_assist = eligible or (args.assist_before_recording and imminent_contact)
        if candidate is None and not course and can_assist and len(previous) == previous.maxlen:
            candidate_course = ranked_exit(scene, pred["logits"][0, :18].cpu().numpy())
            if candidate_course:
                course.extend(candidate_course if eligible else candidate_course[:1])
                if eligible:
                    candidate = list(previous)
                    leads += 1
        if course and scene["margins"][0, course[0]] <= 1.:
            # Take a new supported turn if an actual new hazard invalidates the route.
            course.clear()
        if candidate is not None and not course:
            course.extend(ranked_exit(scene, pred["logits"][0, :18].cpu().numpy()))
        if course:
            action = course.popleft()
        if action != proposed:
            if candidate is None:
                earlier_overrides += 1
            action_tensor = torch.as_tensor([action], device=model.device)
            _, brain, _ = model.step(inputs, before, actions=action_tensor)
        motion, _, _ = world.raw_scene
        native_poll = world.state["input_poll"]
        observation, _, terminal, _, after = env.step(action)
        count += 1
        courses = scene["corridor_courses"]
        clearance = scene["margins"][0].clip(-64., 64.)
        recent.append(dict(stage_frame=info["stage_frame"], native_spell_id=info["native_spell_id"],
                           native_frames=info["native_frames"], position=info["position"], policy_action=proposed,
                           executed_action=action, proposed_clearance4=float(clearance[proposed]), best_clearance4=float(clearance.max()),
                           first_moves_with_exits=(courses[:, :, 0] >= 0).sum(axis=0).tolist(), eligible=eligible))
        if terminal:
            return summary(after["outcome"], after)
        precise = (world.state["input_poll"] == native_poll + 4
                   and after["raw_stage"] == info["raw_stage"]
                   and np.allclose(motion.paths([POLICY_MASKS[action]], 4)[0, -1],
                                   after["position"], rtol=0., atol=.02))
        if precise:
            row = dict(features=features, action=action, native_poll=native_poll,
                       stage=info["stage"], stage_frame=info["stage_frame"],
                       native_action_frames=4, observed_contact=False,
                       verified_exit_focus=precursor or action != proposed, teacher_override=action != proposed)
            previous.append(row)
            if candidate is not None:
                candidate.append(row)
                if len(candidate) == 40:
                    if retain(candidate):
                        return summary("verified-96-frame-exit", after)
                    candidate = None
        else:
            previous.clear()
            course.clear()
            candidate = None
        info = after
        if count % 256 == 0:
            progress(count, leads, info)
    return summary("decision-budget", info)


def run(args):
    import psutil
    import torch

    from fly_touhou.nbnm_cuda import InferenceGraphs
    from fly_touhou.nbnm_run import load_model, read, runtime

    args.output.mkdir(parents=True)
    temp = runtime(args.output)
    graphs = None
    started = time.monotonic()
    process = psutil.Process()
    records = []
    histories = []
    try:
        spec = portable_spec(read(args.spec))
        model, migrated = load_model(spec, args.checkpoint, apply_native_sensors=False)
        if migrated or not (model.bombs_disabled and model.soft_risk_composition):
            raise ValueError("Use the TH08 transfer's existing NBNM connectome checkpoint")
        model.eval().requires_grad_(False)
        graphs = InferenceGraphs(model, sample=False, packed_binary_messages=spec.get("packed_binary_messages", False))
        with TemporaryDirectory(prefix="private-original-", dir=args.output) as directory:
            folder = Path(directory).resolve(strict=True)
            if folder.parent != args.output:
                raise ValueError("Keep private TH08 game files in the requested output")
            world = training_worlds(args.executable, args.profile, args.d3d8to9, folder, 1,
                dict(spec["nbnm"], native_unpaced=True), practice_slots=(args.practice_stage,))[0]
            env = factory(world)
            try:
                for episode in range(1, args.episodes + 1):
                    def progress(decisions, opportunities, info):
                        write_json(args.output / "progress.json", dict(
                            phase="collect-original-exit", PID=process.pid, create_time=process.create_time(), episode=episode,
                            student_and_teacher_decisions=decisions, teacher_opportunities=opportunities,
                            native_stage=info["stage"], native_spell_id=info["native_spell_id"],
                            stage_frame=info["stage_frame"], native_audio=False, autonomous_evaluation=False))

                    def retain(rows):
                        output = args.output / f"episode-{episode:03d}-window-{len(histories) + 1:03d}"
                        output.mkdir()
                        bank = save_windows(output, rows, args.practice_stage, None, policy=args.checkpoint,
                                            forecast_revision=FORECAST_REVISION)
                        for row in bank["histories"]:
                            row = dict(row, file=str((output / row["file"]).relative_to(args.output)))
                            histories.append(row)
                        if not bank["histories"]:
                            raise RuntimeError("The measured exit did not produce a contiguous teacher window")
                        return len(histories) >= args.max_windows

                    with torch.inference_mode():
                        result = collect(env, model, graphs, args, retain, progress)
                    records.append(dict(episode=episode, **result))
                    if len(histories) >= args.max_windows or result["outcome"] == "stopped":
                        break
            finally:
                env.close()
        write_json(args.output / "manifest.json", dict(schema=SCHEMA, histories=histories, PPO_rows=0,
                    source_policy=str(args.checkpoint), native_audio=False, autonomous_evaluation=False))
        write_json(args.output / "report.json", dict(cases=records, accepted_windows=len(histories),
                   game="TH08", raw_stage=args.practice_stage, native_audio=False,
                   elapsed_seconds=time.monotonic()-started, autonomous_evaluation=False))
    finally:
        if graphs is not None:
            graphs.clear()
        cleanup_native_temp(temp)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("spec", "checkpoint", "executable", "d3d8to9", "output"):
        parser.add_argument(f"--{name}", type=lambda path: Path(path).resolve(), required=True)
    parser.add_argument("--profile", type=Path, default=ROOT / "configs/th08-1.00d-zh-input.json")
    parser.add_argument("--practice-stage", type=int, choices=range(8), required=True)
    parser.add_argument("--spell-id", type=int, help="Collect only during this original spell")
    parser.add_argument("--first-stage-frame", type=int, default=0,
                        help="Begin collecting at this native stage clock")
    parser.add_argument("--include-ordinary-closures", action="store_true",
                        help="Include changing exits among visible bullets in teacher collection")
    parser.add_argument("--assist-before-recording", action="store_true",
                        help="Avoid predicted immediate contact on the way to the selected phase")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-decisions", type=int, default=12000)
    parser.add_argument("--max-windows", type=int, default=2)
    args = parser.parse_args()
    if min(args.episodes, args.max_decisions, args.max_windows) < 1 or args.first_stage_frame < 0:
        parser.error("Use positive collection budgets and a nonnegative starting frame")
    with gpu_phase("evaluation"):
        run(args)


if __name__ == "__main__":
    main()
