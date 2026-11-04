"""Finite CPU profile of frozen movement labels on four archived observations.

Never attaches to a live process, constructs a network or opens a native game.
This measures only the observable movement branch, not the native planner or
the running workers. The three stage4 samples are deliberately selected from
three bounded source shards and are not a representative live timing sample.
"""
from __future__ import annotations

import argparse
import cProfile
import hashlib
import io
import json
import os
from pathlib import Path
import pstats
import sys
import time
import zlib

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

RUNTIME = Path("./outputs/source_snapshots/20261026-B1-D95-T025-fresh-Adam-prepared-v3")
sys.path.insert(0, str(RUNTIME / "src"))
from fly_touhou.thrl_durable_records import write_json
from fly_touhou.thrl_study_io import read_shared_json as read, utc_now


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run():
    import numpy as np
    import psutil
    from fly_touhou.thrl_movement_teacher import MovementSafetyTeacher, normal_unprotected

    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    started = time.monotonic()

    def guard():
        if time.monotonic() - started > 60 or psutil.virtual_memory().available < 3 * 1024**3:
            raise RuntimeError("Finite offline profile wall/RAM limit reached")

    bank = Path("./outputs/distillation/B1-d95-final19-labels-v1")
    source = read(bank / "episodes/world000/episode_index.json")["source_episode"]
    if sha(source["index"]) != source["index_sha256"]:
        raise ValueError("Archived source index identity differs")
    numeric = Path("./outputs/engineering/B1-autonomous-dodge-20261024/full-teacher-numeric-coverage-v1")
    receipt = next(read(path) for path in sorted(numeric.glob("game*-receipt.json"))
                   if Path(read(path)["source_index"]).parent.name == "world000")
    if (receipt["source_index_sha256"] != source["index_sha256"]
            or sha(receipt["compact"]) != receipt["compact_sha256"]):
        raise ValueError("Independent numeric archive identity differs")
    with np.load(receipt["compact"], allow_pickle=False) as archive:
        stage = archive["stage"]
    stage4 = np.flatnonzero(stage == 4)
    shard_numbers = [0, *[int(stage4[int((len(stage4) - 1) * q)]) // 32 for q in (.15, .55, .9)]]
    if len(set(shard_numbers)) != 4:
        raise ValueError("Require one opening and three distinct stage4 shards")
    teacher, results = MovementSafetyTeacher(), []
    for number in shard_numbers:
        guard()
        spec = source["shards"][number]
        path = Path(source["index"]).parent / spec["name"]
        if (not 1 <= spec["rows"] <= 32 or spec["uncompressed_bytes"] > 20 * 1024**2
                or path.stat().st_size != spec["bytes"] or sha(path) != spec["sha256"]):
            raise ValueError("Require a bounded immutable source shard")
        decoder = zlib.decompressobj()
        payload = decoder.decompress(path.read_bytes(), spec["uncompressed_bytes"] + 1)
        if (len(payload) != spec["uncompressed_bytes"] or not decoder.eof
                or decoder.unused_data or decoder.unconsumed_tail
                or hashlib.sha256(payload).hexdigest() != spec["uncompressed_sha256"]):
            raise ValueError("Archived observations lost source identity")
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            features, objects, maps, stages = [archive[key] for key in ("features", "objects", "maps", "stage")]
        if number == 0:
            offset = 0
        else:
            valid = np.flatnonzero((stages == 4) & normal_unprotected(features))
            if not len(valid):
                results.append(dict(shard=number, skipped="no normal-unprotected stage4 sample"))
                continue
            laser_counts = ((objects[:, :, 0] > .5) & ((objects[:, :, 3] > .5) | (objects[:, :, 4] > .5))).sum(1)
            offset = int(valid[np.argmax(laser_counts[valid])])
        f, o, m = features[offset], objects[offset], maps[offset]
        present = o[:, 0] > .5
        lasers = present & ((o[:, 3] > .5) | (o[:, 4] > .5))
        profiler = cProfile.Profile()
        guard()
        before = time.perf_counter()
        profiler.enable()
        safe, detail = teacher.label(f, o, m, force=True)
        profiler.disable()
        label_seconds = time.perf_counter() - before
        before = time.perf_counter()
        short_reason = teacher._uncertainty(f, o, m, horizon=4)
        uncertainty4_seconds = time.perf_counter() - before
        statistics = pstats.Stats(profiler)
        top = sorted(statistics.stats.items(), key=lambda row: row[1][3], reverse=True)[:12]
        results.append(dict(shard=number, source_row=spec["start"] + offset, native_seed=source["native_seed"],
            source_shard_sha256=spec["sha256"], stage=int(stages[offset]),
            present_tokens=int(present.sum()), laser_tokens=int(lasers.sum()),
            unknown_laser_tokens=int((lasers & (o[:, 26] <= .5)).sum()),
            laser_occupancy_cells=int((m[10] > 0).sum()),
            label_seconds_with_profiler=label_seconds, uncertainty4_seconds_without_profiler=uncertainty4_seconds,
            reason=detail["reason"], safe_actions=int(safe.sum()), uncertainty4_reason=short_reason,
            inclusive_profile=[dict(file=key[0], line=key[1], function=key[2], primitive_calls=value[0],
                calls=value[1], exclusive_seconds=value[2], inclusive_seconds=value[3]) for key, value in top]))
    return dict(schema="B1-four-archived-observation-movement-profile-v1", utc=utc_now(),
        runtime=str(RUNTIME), source_episode="world000", samples=results,
        frozen_movement_source_sha256=sha(RUNTIME / "src/fly_touhou/thrl_movement_teacher.py"),
        source_script_sha256=sha(__file__), elapsed_seconds=time.monotonic() - started,
        limitations=["One profiled invocation per selected sample;not a steady-state or live-worker time share.",
            "Only movement label and separate four-frame uncertainty guard;native planner not executed.",
            "Nested inclusive timings overlap and must not be added;short guard timing is separate.",
            "No execution teacher,policy,game,reward or training configuration changed."],
        torch_imported="torch" in sys.modules, child_processes_created=len(psutil.Process().children()),
        GPU_or_network_or_native_models_started=False, optimizer_updates=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve existing evidence")
    report = run()
    if report["torch_imported"] or report["child_processes_created"]:
        raise RuntimeError("Unexpected model import or child process in CPU-only profile")
    write_json(args.output, report)
    print(json.dumps(dict(output=str(args.output), samples=[{k: row[k] for k in row if k != "inclusive_profile"}
        for row in report["samples"]], elapsed_seconds=report["elapsed_seconds"])))


if __name__ == "__main__":
    main()
