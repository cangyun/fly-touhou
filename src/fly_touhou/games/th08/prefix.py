"""Restore a recorded original TH08 position and committed movement history."""

import json
from itertools import islice
from pathlib import Path

from ...th08_story import TH08StoryWorld


POSE_FIELDS = ("raw_stage", "stage_frame", "native_frames", "position", "native_spell_id")


def save_stage_memory(path, decision, native_frames, info, observation, brain):
    """Save the brain state immediately before the first action of a stage."""
    import torch

    from ...nbnm_run import atomic_torch_save

    atomic_torch_save(dict(
        decision=decision, native_frames=native_frames, raw_stage=info["raw_stage"],
        stage_rng_seeds=info["stage_rng_seeds"], features=torch.from_numpy(observation["features"]),
        brain=[state.detach().cpu() for state in brain.tensors()]), path)


def read_trace(episode_dir, *, reference_scores=False, through_decision=None, default_raw_stage=None):
    keys = ("decision", "action", *POSE_FIELDS, *(("logits",) if reference_scores else ()))
    with (episode_dir / "decisions.jsonl").open(encoding="utf-8") as stream:
        lines = stream if through_decision is None else islice(stream, through_decision + 1)
        rows = []
        for row in map(json.loads, lines):
            if default_raw_stage is not None:
                row.setdefault("raw_stage", default_raw_stage)
            record = {key: row[key] for key in keys}
            if "native_dialogue_edges" in row:
                record["native_dialogue_edges"] = row["native_dialogue_edges"]
            rows.append(record)
        return rows


def recorded_burn_in(source_dir, trace, count):
    """Read observations recorded for the same committed original movement history."""
    import numpy as np

    with (source_dir / "decisions.jsonl").open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in islice(stream, count)]
    for expected, recorded in zip(trace[:count], rows, strict=True):
        if any(recorded[key] != expected[key] for key in ("decision", "action", *POSE_FIELDS)):
            raise ValueError("The saved observations came from a different original movement prefix")
    features = [np.asarray(row["input_features"], dtype=np.float32) for row in rows]
    actions = [row["action"] for row in rows]
    return features, actions


def restore_observed_attack_age(world, observation, recorded_features):
    """Restore the attack clock when re-observation begins partway through an attack."""
    encoded_age = float(recorded_features[276])
    if 0 <= encoded_age < 1.:
        age = round(600. * encoded_age / (1. - encoded_age))
        world.encoder._attack_start = world.state["input_poll"] - age
        observation["features"][276] = encoded_age


def read_prefix_source(path, episode=1, *, reference_scores=False, through_decision=None):
    """Resolve a recorded original start and optionally extend its committed trace."""
    return _read_prefix_source(Path(path).resolve(), episode, reference_scores, through_decision, set())


def _read_prefix_source(path, episode, reference_scores, through_decision, visited):
    if path in visited:
        raise ValueError("The TH08 continuation references itself")
    report = json.loads(path.read_text(encoding="utf-8-sig"))
    native_route = (report.get("autonomous_policy") is False and "matched_prefix_decisions" in report
                    and "after" in report)
    if not native_route and report.get("rollout_scope") not in ("attack", "stage", "remaining-story"):
        source_dir = path.parent / f"episode-{episode:03d}"
        source_episode = next(row for row in report["episodes"] if row["episode"] == episode)
        trace = read_trace(source_dir, reference_scores=reference_scores, through_decision=through_decision)
        return report, source_episode, source_dir, trace

    if reference_scores:
        raise ValueError("A combined diagnostic trace has no single frozen policy for retention scores")
    splice = report["matched_prefix_decisions"] if native_route else report["source_decision"]
    visited.add(path)
    original, source_episode, source_dir, trace = _read_prefix_source(
        Path(report["source_report"]).resolve(), report.get("source_episode", episode), False, splice, visited)
    practice = replay_practice_slot(original)
    if len(trace) != splice + 1 or (practice is not None and report.get("rollout_scope") == "remaining-story"):
        raise ValueError("The continuation needs its recorded seeded original entry")
    start = report["branch_entry"]["info"] if native_route else report["start"]
    entry_stage = start["raw_stage"]
    end = report["after"][-1] if native_route else report["end"]
    if source_episode["stage_rng_seeds"][:len(start["stage_rng_seeds"])] != start["stage_rng_seeds"]:
        raise ValueError("TH08 continuation has different native stage seeds at the join")
    if through_decision is not None and through_decision < splice:
        return original, source_episode, source_dir, trace[:through_decision + 1]
    if native_route:
        states = [start, *report["after"]]
        actions = [row["action"] for row in report["after"]]
        if end["outcome"] is None and not end["contact"]:
            actions.append(None)
        suffix = [
            {"decision": index, "action": action, **{key: state[key] for key in POSE_FIELDS},
             "native_dialogue_edges": replay_dialogue(original, state)}
            for index, (state, action) in enumerate(zip(states, actions))
        ]
        if through_decision is not None:
            suffix = suffix[:through_decision - splice + 1]
    else:
        end_decision = report.get("resume_through_decision")
        if through_decision is not None:
            end_decision = through_decision if end_decision is None else min(through_decision, end_decision)
        suffix = read_trace(path.parent, through_decision=None if end_decision is None else end_decision - splice,
                            default_raw_stage=entry_stage if entry_stage == end["raw_stage"] else None)
    if not suffix or suffix[0]["native_frames"] != start["native_frames"] or any(
        suffix[0][key] != record[key]
        for record in (start, trace[-1]) for key in POSE_FIELDS if key != "native_frames"
    ):
        raise ValueError("TH08 continuation does not match its recorded Story join")
    if any(row["decision"] != i for i, row in enumerate(suffix)):
        raise ValueError("TH08 continuation decisions are not contiguous")
    frame_offset = trace[-1]["native_frames"] - suffix[0]["native_frames"]
    for row in suffix:
        row["decision"] += splice
        row["native_frames"] += frame_offset
    trace[-1:] = suffix
    source_episode = dict(source_episode, stage_rng_seeds=end["stage_rng_seeds"])
    return original, source_episode, source_dir, trace


def replay_practice_slot(report):
    if report["native_rng_controlled"]:
        if report["entry_mode"] == "original-story":
            return None
        if report["entry_mode"] in ("original-stage-practice", "original-spell-practice"):
            slot = report["practice_stage_slot"]
            spell = report.get("practice_spell_slot")
            valid_spell = (type(spell) is int and 0 <= spell < 44
                           if report["entry_mode"] == "original-spell-practice" else spell is None)
            if type(slot) is int and slot in range(8) and valid_spell:
                return slot
    raise ValueError("Use a seeded original TH08 Story, Stage Practice, or Spell Practice report")


def replay_dialogue(report, row):
    return row.get("native_dialogue_edges", report.get("native_dialogue_edges", False))


def match_pose(row, info, frame_offset, *, check_spell=True):
    observed = dict(info, position=list(info["position"]), native_frames=info["native_frames"] - frame_offset)
    difference = {key: (row[key], observed[key]) for key in POSE_FIELDS
                  if (check_spell or key != "native_spell_id") and row[key] != observed[key]}
    if difference or info["outcome"] is not None:
        raise RuntimeError(f"Original prefix differs at decision {row['decision']}: {difference}, {info['outcome']}")


def contiguous_safe_move(before, after, ticks):
    return (after["outcome"] is None and not after["contact"]
            and after["raw_stage"] == before["raw_stage"]
            and after["native_frames"] - before["native_frames"] == ticks)


def measured_action(obs, before, action, *, exit_focus=False, override=False, reference=None):
    row = dict(features=obs["features"], native_poll=before["native_frames"], action=action,
               stage=before["stage"], stage_frame=before["stage_frame"], native_action_frames=4,
               observed_contact=False, verified_exit_focus=exit_focus, teacher_override=override)
    if reference is not None:
        row["reference_logits"] = reference["logits"]
    return row


def match_practice_intro(row, info, offset, pending_spell):
    """Match movement throughout the title animation, then require the exact attack."""
    expected, observed = row["native_spell_id"], info["native_spell_id"]
    introductory = pending_spell is not None and {expected, observed} <= {None, pending_spell}
    match_pose(row, info, offset, check_spell=not introductory)
    return None if expected == observed == pending_spell else pending_spell


def prepare_replay(args, source_dir, controller, settings, source_episode, source_report, trace):
    native_seed = source_report["native_startup_seed"]
    practice_slot = replay_practice_slot(source_report)
    entry_clocks = {}
    if args.match_recorded_stage_entry:
        for row in trace:
            entry_clocks.setdefault(row["raw_stage"], row["stage_frame"])
    if practice_slot is not None:
        entry_clocks[practice_slot] = trace[0]["stage_frame"]
    align_entry = args.match_recorded_stage_entry or source_report.get("aligned_native_stage_entries", False)
    world = TH08StoryWorld(controller, task_context=settings.get("task_context", "legacy"), retain_raw_scene=True,
                           practice_slot=practice_slot, spell_slot=source_report.get("practice_spell_slot"),
                           forecast_sweep_laser_births=settings.get("forecast_sweep_laser_births", False),
                           forecast_wide_beam_births=settings.get("forecast_wide_beam_births", False),
                           forecast_orbital_point_births=settings.get("forecast_orbital_point_births", False),
                           forecast_final_b_ring=settings.get("forecast_final_b_ring", False),
                           forecast_native_branches=settings.get("forecast_native_branches", False),
                           forecast_native_narrow_exits=settings.get("forecast_native_narrow_exits", False),
                           forecast_native_carried_courses=settings.get("forecast_native_carried_courses", False),
                           forecast_native_hold_fallback=settings.get("forecast_native_hold_fallback", False),
                           forecast_native_corner_turns=settings.get("forecast_native_corner_turns", False),
                           forecast_final_spark_cue=settings.get("forecast_final_spark_cue", False),
                           align_native_stage_entry=align_entry, stage_entry_clocks=entry_clocks,
                           native_dialogue_edges=replay_dialogue(source_report, trace[0]))
    world.encoder.retain_scene = args.record_native_forecast
    obs, info = world.start()
    startup = world.state.get("native_seed")
    if not startup or startup["requested"] != native_seed or startup["applied"] != 1:
        raise RuntimeError("Original replay did not use its specified seed")
    expected_entry = json.loads((source_dir / "paired-entry.json").read_text(encoding="utf-8"))
    features = obs["features"].tolist() if obs is not None else None
    if features != expected_entry["features"] or info["stage_rng_seeds"] != expected_entry["stage_rng_seeds"]:
        keys = ("raw_stage", "stage_frame", "stage_rng_seeds", "position")
        observed_entry = dict(info, position=list(info["position"]))
        differences = {key: (expected_entry[key], observed_entry[key]) for key in keys
                       if expected_entry[key] != observed_entry[key]}
        indices = ([] if features is None or expected_entry["features"] is None else
                   [i for i, (a, b) in enumerate(zip(expected_entry["features"], features)) if a != b])
        raise RuntimeError(f"Original replay first observation differs: {differences}, "
                           f"feature columns {indices[:12]}")
    history = []
    frame_offsets = {trace[0]["raw_stage"]: 0}
    context_start = max(0, args.decision - (16 if args.export_teacher else 2))
    enable_final_spark_cue = getattr(args, "final_spark_cue", False) and not world.encoder.forecast_final_spark_cue
    spell_intro = info["practice_spell"] if world.spell_slot is not None else None
    index = 0
    while index <= args.decision:
        row = trace[index]
        world.native_dialogue_edges = replay_dialogue(source_report, row)
        raw_stage = row["raw_stage"]
        if raw_stage not in frame_offsets:
            frame_offsets[raw_stage] = info["native_frames"] - row["native_frames"] if entry_clocks else 0
        if index < args.decision:
            spell_intro = match_practice_intro(row, info, frame_offsets[raw_stage], spell_intro)
        else:
            match_pose(row, info, frame_offsets[raw_stage])
        if index >= context_start:
            lift_forecast = getattr(args, "live_forecast", False) and world.encoder.native_forecast_controller is None
            if enable_final_spark_cue or lift_forecast:
                if enable_final_spark_cue:
                    world.encoder.forecast_final_spark_cue = True
                    enable_final_spark_cue = False
                if lift_forecast:
                    world.encoder.native_forecast_controller = controller
                obs, info = world._prepare()
        if index == args.decision:
            break
        if (args.record_native_forecast or getattr(args, "live_forecast", False)) and index < context_start - 1:
            stop = index
            limit = min(context_start - 1, index + 512)
            while stop < limit:
                current, following = trace[stop:stop + 2]
                if (following["raw_stage"] != raw_stage or
                        following["native_frames"] - current["native_frames"] != world.decision_ticks or
                        replay_dialogue(source_report, following) != world.native_dialogue_edges or
                        (getattr(args, "audit_spells", False)
                         and following["native_spell_id"] != current["native_spell_id"])):
                    break
                stop += 1
            if stop > index + 1:
                obs, info = world.replay_movement([row["action"] for row in trace[index:stop]])
                index = stop
                continue
        previous_obs, before = obs, info
        obs, info = world.step(row["action"], observe=index + 1 >= context_start)
        if args.export_teacher and index >= context_start:
            if contiguous_safe_move(before, info, world.decision_ticks):
                history.append(measured_action(previous_obs, before, row["action"],
                                               reference=row if getattr(args, "reference_scores", False) else None))
            else:
                history.clear()
        index += 1
    seeds = info["stage_rng_seeds"]
    if seeds != source_episode["stage_rng_seeds"][:len(seeds)]:
        raise RuntimeError("Original replay did not match the native RNG state at stage starts")

    return world, obs, info, history, frame_offsets


def prepare_policy_window(world, settings, report, trace, offsets, start, decision, *,
                          until_attack_end=False, carry_native_courses=False):
    """Reproduce the sensory history used to warm a frozen TH08 local policy."""
    encoder = world.encoder
    encoder.native_forecast_controller = world.controller
    encoder.forecast_native_narrow_exits = settings.get("forecast_native_narrow_exits", False)
    encoder.forecast_final_spark_cue = settings.get("forecast_final_spark_cue", False)
    encoder.task_context = settings.get("task_context", "legacy")
    observation, info = world._prepare()
    history = {"features": [], "actions": []}
    hold_fallback = settings.get("forecast_native_hold_fallback", False)
    corner_turns = settings.get("forecast_native_corner_turns", False)
    pending_spell = info.get("practice_spell") if start == 0 else None
    for recorded in trace[start:decision]:
        pending_spell = match_practice_intro(recorded, info, offsets[recorded["raw_stage"]], pending_spell)
        world.native_dialogue_edges = replay_dialogue(report, recorded)
        history["features"].append(observation["features"])
        history["actions"].append(recorded["action"])
        if recorded["decision"] == decision - 1:
            encoder.forecast_native_hold_fallback = hold_fallback
            encoder.forecast_native_corner_turns = corner_turns
        observation, info = world.step(recorded["action"])
    release = trace[decision]
    match_pose(release, info, offsets[release["raw_stage"]])
    world.native_dialogue_edges = True
    encoder.measure_spell_damage = (until_attack_end and info["spell_active"]
        and not (encoder.forecast_final_spark_cue and info["native_spell_id"] == 99))
    refresh = (hold_fallback != encoder.forecast_native_hold_fallback
               or corner_turns != encoder.forecast_native_corner_turns)
    encoder.forecast_native_hold_fallback = hold_fallback
    encoder.forecast_native_corner_turns = corner_turns
    if encoder.measure_spell_damage or refresh or world.controller.read_dialogue_wait() is not None:
        observation, info = world._prepare()
    encoder.forecast_native_carried_courses = (carry_native_courses
                                               or settings.get("forecast_native_carried_courses", False))
    return observation, info, history
