"""Keep replayable stage entries from the deepest seeded original TH08 runs."""

import argparse
from itertools import islice
import json
from pathlib import Path
from shutil import copy2
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fly_touhou.nbnm_run import read
from fly_touhou.th08_game_state import STAGE_NUMBERS
from fly_touhou.thrl_durable_records import write_json
from fly_touhou.th08_prefix import POSE_FIELDS


def source_record(root, allow_live):
    path = root / "report.json"
    if not path.is_file():
        path = root / "progress.json"
        if not allow_live or not path.is_file():
            return None
    report = read(path)
    if report["entry_mode"] != "original-story" or not report["native_rng_controlled"]:
        return None
    rows = report.get("episodes")
    if rows is None:
        rows = [dict(report["info"], episode=report["episode"], policy_decisions=report["decisions"],
                     outcome="live-prefix", full_game_NBNM=False)]
    if not rows:
        return None

    def depth(row):
        return row["full_game_NBNM"], row["stage"], row["stage_frame"], row["native_frames"]

    row = max(rows, key=depth)
    return depth(row), path, report, row


def archive_source(source, output, models):
    _, path, report, row = source
    source_dir = path.parent / f"episode-{row['episode']:03d}"
    keys = ("decision", "action", *POSE_FIELDS, "logits")
    with (source_dir / "decisions.jsonl").open(encoding="utf-8") as stream:
        trace = [{key: move[key] for key in keys}
                 for move in map(json.loads, islice(stream, row["policy_decisions"]))]
    if (not trace or len(trace) != row["policy_decisions"]
            or any(i != move["decision"] for i, move in enumerate(trace))):
        raise ValueError(f"Missing a continuous committed Story trace in {source_dir}")
    saved = output / path.parent.name
    episode_dir = saved / f"episode-{row['episode']:03d}"
    episode_dir.mkdir(parents=True)
    write_json(saved / "source-record.json", report)
    copy2(source_dir / "paired-entry.json", episode_dir)
    stage_entries = {}
    with (episode_dir / "decisions.jsonl").open("w", encoding="utf-8") as stream:
        for move in trace:
            stage = STAGE_NUMBERS[move["raw_stage"]]
            stream.write(json.dumps(dict(move, stage=stage), separators=(",", ":")) + "\n")
            if stage in stage_entries:
                continue
            memory = f"stage-{stage}-entry.pt"
            if (source_dir / memory).is_file():
                copy2(source_dir / memory, episode_dir)
            else:
                memory = None
            stage_entries[stage] = dict(
                decision=move["decision"], raw_stage=move["raw_stage"],
                stage_frame=move["stage_frame"], native_frames=move["native_frames"],
                position=move["position"], stage_rng_seed=row["stage_rng_seeds"][stage - 1],
                recurrent_state=memory)
    frozen = {key: value for key, value in report.items() if key not in ("info", "episodes", "phase")}
    for key in ("checkpoint", "policy_spec"):
        original = Path(report[key]).resolve()
        if original not in models:
            exported = output / "policy" / f"{len(models):02d}-{original.name}"
            exported.parent.mkdir(exist_ok=True)
            copy2(original, exported)
            models[original] = str(exported)
        frozen[key] = models[original]
    exported_report = saved / "report.json"
    write_json(exported_report, dict(frozen, episodes=[row], attempts=1,
        archived_source=str(path), full_game_NBNM=int(row["full_game_NBNM"]),
        stage_entries=stage_entries, archived_committed_decisions=len(trace)))
    return dict(report=str(exported_report), seed=report["native_startup_seed"], outcome=row["outcome"],
                stages=stage_entries, decisions=len(trace), source=str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--keep", type=int, default=3)
    parser.add_argument("--include-live", action="store_true")
    args = parser.parse_args()
    if args.keep < 1:
        parser.error("--keep must be at least one")
    sources = [row for root in args.runs if (row := source_record(root.resolve(), args.include_live))]
    if not sources:
        parser.error("No eligible seeded original Story results")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True)
    models = {}
    selected = {}
    for source in sorted(sources, key=lambda row: row[0], reverse=True):
        seed = source[2]["native_startup_seed"]
        if seed not in selected:
            selected[seed] = archive_source(source, args.output, models)
        if len(selected) >= args.keep:
            break
    write_json(args.output / "manifest.json", dict(
        schema="th08-original-story-stage-entries-v1", sources=list(selected.values())))
    print(json.dumps([dict(seed=seed, stages=list(row["stages"]), decisions=row["decisions"])
                      for seed, row in selected.items()]))


if __name__ == "__main__":
    main()
