"""Offline experiment bookkeeping for the engineering roadmap.

This module never starts a game or imports a CUDA runtime.  Its checks reject
inconsistent records; passing them is not evidence of competent game control.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


SPLIT_ROLES = frozenset({
    "train", "validation", "development", "formal_test", "diagnostic",
    "historical_development",
})


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_record(path: str | Path, *, root: str | Path | None = None) -> dict:
    path = Path(path).resolve()
    label = str(path)
    if root is not None:
        try:
            label = path.relative_to(Path(root).resolve()).as_posix()
        except ValueError:
            pass
    return {"path": label, "bytes": path.stat().st_size,
            "sha256": sha256_file(path)}


def write_json_new(path: str | Path, value: Any) -> None:
    """Create an immutable observation of a run; never overwrite old evidence."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


@dataclass
class MetricCounters:
    """Counts for one run segment, not aliases for an ambiguous 'iteration'."""
    decisions: int = 0
    native_frames: int = 0
    completed_episodes: int = 0
    successes: int = 0
    hits: int = 0
    interface_faults: int = 0
    human_interruptions: int = 0

    def transition(self, *, elapsed_frames: int, task_terminal: bool,
                   hit: bool, success: bool = False) -> None:
        if isinstance(elapsed_frames, bool) or not isinstance(elapsed_frames, int) or elapsed_frames <= 0:
            raise ValueError("elapsed_frames must be a positive actual native-frame count")
        if success and (not task_terminal or hit):
            raise ValueError("A successful terminal cannot also be a hit")
        self.decisions += 1
        self.native_frames += elapsed_frames
        self.completed_episodes += int(task_terminal)
        self.successes += int(success)
        self.hits += int(hit)

    def fault(self, *, human_interruption: bool = False) -> None:
        if human_interruption:
            self.human_interruptions += 1
        else:
            self.interface_faults += 1

    def to_dict(self) -> dict:
        return asdict(self)


def compact_trajectory_fingerprint(rows: Iterable[Mapping[str, Any]]) -> str:
    """Hash logged dynamics, ignoring PID/absolute clock/episode numbering.

    Legacy traces lack full object states.  Equality under this fingerprint
    is a duplicate warning, not a claim of full native-state equivalence.
    """
    values = [{"action": int(row["action"]),
               "elapsed_frames": int(row["next_frame"]) - int(row["frame"]),
               "player": row["player"], "next_player": row["next_player"],
               "bullets": row.get("bullets"), "hit": bool(row["hit"]),
               "terminal": bool(row["task_terminal"])} for row in rows]
    if not values:
        raise ValueError("An empty trajectory has no useful fingerprint")
    return canonical_sha256({"schema": "compact_dynamics_v1", "steps": values})


def validate_split_ledger(rows: Iterable[Mapping[str, Any]]) -> dict:
    """Detect trajectory leakage and formal-test reuse without hiding repeats."""
    rows = list(rows)
    issues: list[dict] = []
    by_uid: dict[str, list[Mapping[str, Any]]] = {}
    by_fingerprint: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        uid = row.get("trajectory_uid")
        role = row.get("role")
        if not isinstance(uid, str) or not uid:
            issues.append({"kind": "missing_trajectory_uid", "row": dict(row)})
            continue
        by_uid.setdefault(uid, []).append(row)
        if role not in SPLIT_ROLES:
            issues.append({"kind": "unknown_split_role", "trajectory_uid": uid})
        if role == "formal_test" and row.get("used_for_model_selection", False):
            issues.append({"kind": "formal_test_consumed_for_selection", "trajectory_uid": uid})
        fingerprint = row.get("trajectory_fingerprint")
        kind = row.get("fingerprint_kind")
        if fingerprint and kind:
            by_fingerprint.setdefault((str(kind), str(fingerprint)), []).append(row)
    for uid, group in by_uid.items():
        if len(group) > 1:
            issues.append({"kind": "duplicate_trajectory_uid", "trajectory_uid": uid,
                           "roles": sorted({str(row.get("role")) for row in group})})
    repeats = []
    for (kind, fingerprint), group in by_fingerprint.items():
        if len(group) > 1:
            roles = sorted({str(row.get("role")) for row in group})
            item = {"fingerprint_kind": kind, "fingerprint": fingerprint,
                    "trajectories": [row["trajectory_uid"] for row in group], "roles": roles}
            repeats.append(item)
            if len(roles) > 1:
                issues.append(dict(item, kind="cross_split_duplicate_trajectory"))
    return {"schema": "episode-split-audit-v1", "rows": len(rows),
            "issues": issues, "repeated_fingerprints": repeats,
            "valid": not issues,
            "independence_proven": False,
            "interpretation": "No detected overlap does not prove stochastic independence."}


def make_resume_receipt(*, checkpoint: str | Path, rng_sidecar: str | Path,
                        contract: Mapping[str, Any], cumulative_decisions: int,
                        adam_minibatch_steps: int) -> dict:
    """Bind saved files to a contract; the runner owns state serialization.

    Without a complete native snapshot the resume must open a fresh episode,
    clear recurrent/rollout state and report a trajectory discontinuity.  The
    receipt does not assert that RNG or optimizer state was actually restored.
    """
    if cumulative_decisions < 0 or adam_minibatch_steps < 0:
        raise ValueError("Cumulative counters cannot be negative")
    return {"schema": "engineering-resume-receipt-v1",
            "checkpoint": file_record(checkpoint), "rng_sidecar": file_record(rng_sidecar),
            "contract_sha256": canonical_sha256(contract),
            "cumulative_decisions": int(cumulative_decisions),
            "adam_minibatch_steps": int(adam_minibatch_steps),
            "native_full_snapshot": False,
            "restart_episode": True, "clear_recurrent_state": True,
            "discard_partial_rollout": True, "bitwise_native_continuation": False}


def validate_resume_receipt(receipt: Mapping[str, Any], *, contract: Mapping[str, Any]) -> dict:
    issues = []
    if receipt.get("schema") != "engineering-resume-receipt-v1":
        issues.append("unknown_receipt_schema")
    if receipt.get("contract_sha256") != canonical_sha256(contract):
        issues.append("contract_changed")
    for name in ("checkpoint", "rng_sidecar"):
        record = receipt.get(name, {})
        path = Path(record.get("path", "__missing__"))
        if not path.is_file():
            issues.append(f"{name}_missing")
        elif sha256_file(path) != record.get("sha256") or path.stat().st_size != record.get("bytes"):
            issues.append(f"{name}_changed")
    if not receipt.get("native_full_snapshot", False):
        for required in ("restart_episode", "clear_recurrent_state", "discard_partial_rollout"):
            if receipt.get(required) is not True:
                issues.append(f"{required}_required")
        if receipt.get("bitwise_native_continuation") is not False:
            issues.append("cannot_claim_bitwise_native_continuation")
    else:
        issues.append("native_snapshot_resume_not_implemented_by_this_contract")
    return {"valid": not issues, "issues": issues,
            "runtime_restore_verified": False}
