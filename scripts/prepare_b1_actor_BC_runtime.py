"""Freeze pinned BC additions over an unchanged existing B1 runtime.

No model, CUDA, native process, optimizer, training or scheduler is started.
Existing baseline files cannot be replaced, and destinations are exclusive.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fly_touhou.thrl_durable_records import write_json
from fly_touhou.thrl_study_io import read_shared_json as read, utc_now

SCHEMA = "B1-full-actor-path-BC-pinned-additions-v1"
REQUIRED = {"src/fly_touhou/thrl_full_actor_path_bc.py", "src/fly_touhou/thrl_full_actor_bc_stream.py",
            "src/fly_touhou/thrl_full_teacher_data.py", "src/fly_touhou/thrl_teacher_archive.py",
            "scripts/train_b1_actor_path_BC_contrast.py", "scripts/qualify_b1_full_actor_BC_CUDA.py",
            "scripts/prepare_b1_actor_BC_runtime.py"}


def digest(value):
    return hashlib.sha256(value).hexdigest()


def child(root, name):
    relative = Path(name)
    path = (root / relative).resolve()
    if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(root.resolve()):
        raise ValueError("Source path escapes its declared root")
    return path


def freeze(baseline, baseline_sha, additions, output):
    if output.exists():
        raise FileExistsError("Never overwrite or reuse an existing source snapshot")
    if output.resolve().is_relative_to(baseline.resolve()):
        raise ValueError("New source snapshots cannot be written inside the unchanged baseline")
    raw = (baseline / "source_manifest.json").read_bytes()
    if digest(raw) != baseline_sha:
        raise ValueError("Pinned baseline manifest identity differs")
    base = json.loads(raw)
    proposal = read(additions)
    if proposal.get("schema") != SCHEMA:
        raise ValueError("Require an explicit pinned additions manifest")
    names = [row["path"] for row in proposal["files"]]
    if len(names) != len(set(names)) or not REQUIRED.issubset(names):
        raise ValueError("Missing or duplicated frozen BC entry/helper dependencies")
    content = {}
    for row in base["files"]:
        name = Path(row["path"]).as_posix()
        data = child(baseline, name).read_bytes()
        if name in content or len(data) != row["bytes"] or digest(data) != row["sha256"]:
            raise ValueError("Existing baseline source bytes differ: " + name)
        content[name] = data
    for row in proposal["files"]:
        name = Path(row["path"]).as_posix()
        if name in content or not name.endswith(".py"):
            raise ValueError("BC additions cannot replace existing model/PPO/native files")
        data = child(ROOT, name).read_bytes()
        if not row.get("role") or digest(data) != row["sha256"]:
            raise ValueError("Final tested additions differ: " + name)
        content[name] = data
    for name, data in content.items():
        if name.endswith(".py"):
            compile(data, name, "exec")
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for name, data in sorted(content.items()):
        target = child(output, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        if digest(target.read_bytes()) != digest(data):
            raise ValueError("Frozen copy verification failed")
        records.append(dict(path=name, bytes=len(data), sha256=digest(data)))
    manifest = dict(schema="B1-full-actor-path-BC-additions-runtime-v1", utc=utc_now(),
        origin=str(ROOT), baseline=str(baseline.resolve()), baseline_manifest_sha256=baseline_sha,
        pinned_additions_manifest=str(additions.resolve()), pinned_additions_sha256=digest(additions.read_bytes()),
        unchanged_baseline_files=len(base["files"]), replaced_existing_files=0, additions=names,
        files=records, model_PPO_environment_reward_unchanged=True,
        native_worlds_started=0, optimizer_steps=0)
    write_json(output / "source_manifest.json", manifest)
    (output / "snapshot_manifest.json").write_bytes((output / "source_manifest.json").read_bytes())
    return dict(runtime=str(output.resolve()), manifest_sha256=digest((output / "source_manifest.json").read_bytes()),
                baseline_files=len(base["files"]), added_files=len(names), total_files=len(records))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--baseline-sha256", required=True)
    parser.add_argument("--additions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.baseline, args.baseline_sha256, args.additions, args.output)))
