"""Durable Story checkpoints that replay raw input without rerunning the policy."""

from pathlib import Path

from .thrl_durable_records import write_json

SCHEMA = "frozen-NBNM-story-stage-bookmark-v1"
PREFIX_STATUS = "nbnm_frozen_stage_entry"


def _encode_carry(carry, frame):
    if carry is None:
        return None
    result = dict(frame_offset=int(carry["frame"] - frame), stage=int(carry["stage"]),
                  position=carry["position"].tolist(), course=carry["course"].tolist())
    if carry.get("issun_lure"):
        result["issun_lure"] = True
    if "attack" in carry:
        result["attack"] = carry["attack"]
    if carry.get("alternate") is not None:
        result["alternate"] = _encode_carry(carry["alternate"], frame)
    return result


def _decode_carry(carry, frame):
    if carry is None:
        return None
    import numpy as np

    result = dict(frame=frame + carry["frame_offset"], stage=carry["stage"],
                  position=np.asarray(carry["position"], dtype=float),
                  course=np.asarray(carry["course"], dtype=np.int8))
    if carry.get("issun_lure"):
        result["issun_lure"] = True
    if "attack" in carry:
        result["attack"] = carry["attack"]
    if carry.get("alternate") is not None:
        result["alternate"] = _decode_carry(carry["alternate"], frame)
    return result


def capture(env, record, info, done):
    """Take the first settled checkpoint after a stage has ended."""
    if not env.frozen_story_bookmarks or done:
        return
    s = env.snapshot
    stage = s.stage
    if (not 2 <= stage <= 6 or stage in env._saved_bookmark_stages or
            s.stage_frame < 32 or s.screen != 7 or s.dialogue or
            s.player is None or s.player.state != 1 or s.player.invulnerable_frames or s.lasers or
            env.stage_history != list(range(1, stage+1))):
        return
    # TH14 can announce the new stage while the departing stage clock
    # remains visible. Require a low clock from this stage before recording.
    if not any(row["after"]["stage"] == stage and row["after"]["stage_frame"] < 32
               for row in record["transitions"]):
        return
    from .nbnm_stage_bookmark import replay_stage_record
    from .native_replay import semantic_state

    native = replay_stage_record(env.controller._memory, stage)
    event_keys = ("misses", "bombs_used", "point_items")
    events = {key: s.extra["thrl_events"][key] for key in event_keys}
    if native is None or native["continues"] != 0 or events["misses"] or events["bombs_used"]:
        return
    path = Path(env.stage6_prefix_capture)
    root = path.parent.parent / "stage-bookmark-prefixes"
    root.mkdir(parents=True, exist_ok=True)
    prefix = root / f"stage-{stage}-{path.name}"
    native_path = prefix.with_suffix(".native-stage.json")
    encoder = env.encoder
    encoder_state = dict(attack=list(encoder._attack), attack_age_frames=s.frame-encoder._attack_start,
                         corridor_carry=_encode_carry(encoder.corridor_carry, s.frame))
    if getattr(encoder, "frozen_exit_scoring", False) and encoder.exit48_carry is not None:
        encoder_state["exit48_carry"] = _encode_carry(encoder.exit48_carry, s.frame)
    write_json(prefix, dict(record, schema="frozen-SNN-actual-prefix-commands-v1", status=PREFIX_STATUS,
        raw_input_segments=list(env.controller._input_tape), landing=semantic_state(s),
        native_events=events, teacher_actions=0, optimizer_updates=0,
        policy_steps=env.policy_steps, initial_episode_frames=env.episode_frames,
        encoder_state=encoder_state,
        prefix_was_played_live_by_frozen_network=not bool(getattr(env, '_frozen_prefix_replay_receipt', None)),
        restored_and_verified_ancestors=bool(getattr(env, '_frozen_prefix_replay_receipt', None))))
    write_json(native_path, native)
    env._saved_bookmark_stages.add(stage)
    info["frozen_story_bookmark"] = dict(stage=stage, prefix=str(prefix),
                                          native_stage_record=str(native_path))


def save(output, model, brain, index, features, info, spec, checkpoint, split, previous_action=None):
    """Publish a manifest only after both the native and neural data are durable."""
    import os
    from .nbnm_run import atomic_torch_save
    from .nbnm_stage_bookmark import save_checkpoint
    from .th14_stage_replay import write_stage_replay

    entry = info["frozen_story_bookmark"]
    stage, seed = entry["stage"], info["native_seed"]
    root = Path(output) / f"stage{stage}-bookmarks"
    dest = root / f'seed-{seed}-episode-{info["native_episode"]}-world-{index}'
    dest.mkdir(parents=True, exist_ok=False)
    weights = Path(output) / "frozen-story-parameters.pt"
    save_checkpoint(checkpoint, weights, model)
    write_json(dest / "configuration.json", spec)
    atomic_torch_save(dict(schema="frozen-SNN-stage-entry-brain-v2", opening_seed=seed,
        features=features.copy(), state=[v[index:index+1].detach().cpu().clone() for v in brain.tensors()],
        stage_frame=info["stage_frame"], context=dict(stage=stage, stage_frame=info["stage_frame"]),
        optimizer_updates=0), dest / "brain.pt")
    from .nbnm_run import read
    native = read(entry["native_stage_record"])
    write_json(dest / "native-stage.json", native)
    if stage == 6:
        write_stage_replay(dest / "stage6-start.rpy", native)
    relative = lambda path: os.path.relpath(path, dest)
    manifest = dict(schema=SCHEMA, stage=stage, seed=seed, stage_frame=info["stage_frame"],
        split=split, policy_steps=info["policy_steps"], previous_action=previous_action,
        native_input_prefix=relative(entry["prefix"]), native_stage_record="native-stage.json",
        brain="brain.pt", checkpoint=relative(weights), configuration="configuration.json",
        source="frozen full Story", brain_timing="immediately before the bookmarked decision")
    write_json(dest / "bookmark.json", manifest)
    write_json(Path(output) / f"latest-bookmark-{seed}-world-{index}.json",
        dict(stage=stage, seed=seed, stage_frame=info["stage_frame"],
             bookmark=os.path.relpath(dest / "bookmark.json", output)))
    return dest


def load(bookmark):
    from .nbnm_run import read

    path = Path(bookmark).resolve()
    if path.is_dir():
        path = path / "bookmark.json"
    entry = read(path)
    if "bookmark" in entry and entry.get("schema") is None:
        path = (path.parent / entry["bookmark"]).resolve()
        entry = read(path)
    if entry.get("schema") != SCHEMA:
        raise ValueError("Resume requires an automatically saved frozen Story bookmark")
    return {**entry, "manifest": str(path), **{
        key: str((path.parent / entry[key]).resolve())
        for key in ("native_input_prefix", "brain", "configuration", "checkpoint")
    }}


def restore_encoder(env, snapshot, record):
    encoder = env.encoder
    saved = record["encoder_state"]
    encoder._attack = tuple(saved["attack"])
    encoder._attack_start = snapshot.frame - saved["attack_age_frames"]
    encoder.corridor_carry = _decode_carry(saved["corridor_carry"], snapshot.frame)
    encoder.exit48_carry = _decode_carry(saved.get("exit48_carry"), snapshot.frame)
    return env._encode(snapshot)


def resume_tape(env, path):
    """Reopen a verified tape with the same recorded prefix for future saves."""
    from .nbnm_run import read
    record = read(path)
    env.controller.begin_input_tape()
    env.controller._input_tape.extend(tuple(item) for item in record["raw_input_segments"])
    env._frozen_prefix = dict(start=record["start"], transitions=list(record["transitions"]),
        ticks=record["ticks"], native_seed=record["native_seed"], status="recording")
    env._saved_bookmark_stages = set(range(2, record["landing"]["stage"]+1))
