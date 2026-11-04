"""Append-only, tamper-evident records for full-brain original-game experiments.

The pinned source contains 138,639 indexed neurons and 15,091,983 signed
connections. A failed whole-brain check is an error, never permission to reduce
the graph or replace the actor by an ordinary MLP. Checkpoints, failures and
unsuccessful epochs remain in the same event history.
"""
from __future__ import annotations

from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import sys
import threading
import time
import traceback
from typing import Any, Iterable
import uuid

import numpy as np

FULL_BRAIN_NEURONS = 138_639
FULL_BRAIN_EDGES = 15_091_983
SYNAPTIC_MV_PER_COUNT = 0.275
TRIAL_TH14_SHA256 = "6c62bff1e6be644086c5811c36eb938818f14012f94ad5fce784ca7889c66d48"
EVENT_SCHEMA = "fly-touhou-experiment-event-v1"
MANIFEST_SCHEMA = "fly-touhou-experiment-manifest-v1"
MAX_EVENT_BYTES = 2 * 1024 * 1024
SOURCE_ARCHIVE_MAX_BYTES = 2 * 1024 * 1024
SOURCE_ARCHIVE_SUFFIXES = frozenset({
    ".py", ".js", ".cjs", ".mjs", ".json", ".toml", ".md", ".ps1", ".yaml", ".yml",
    ".ini", ".cfg", ".conf", ".txt", ".c", ".h", ".cpp", ".hpp", ".ts", ".tsx",
})
FINAL_CAMPAIGN_CRITERION = {
    "game": "TH14", "edition": "formal", "difficulty": "Lunatic", "stages": [1, 2, 3, 4, 5, 6],
    "finite_native_lives": True, "continues_used": 0, "teacher_action_calls": 0,
    "safety_takeovers": 0, "brain_state_resets_between_stages": 0,
    "native_resource_resets_between_stages": 0,
    "run_mode": 0, "replay_playback": 0,
    "actor": "complete pinned 138639-neuron,15091983-connection brain",
}


class ExperimentContractError(RuntimeError):
    pass


class EventIntegrityError(ExperimentContractError):
    pass


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_stamp(path: Path) -> tuple[int, ...]:
    stat = path.stat()
    if not path.is_file():
        raise ExperimentContractError(f"Pinned input is not a regular file: {path}")
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu()
        return value.item() if value.numel() == 1 else value.tolist()
    return value


def _encoded(value: Any) -> bytes:
    return json.dumps(_plain(value), sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_encoded(value)).hexdigest()


def _array_digest(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode())
        digest.update(_encoded(list(contiguous.shape)))
        digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


@contextmanager
def _file_lock(path: Path, timeout: float = 10):
    """Serialize writers without changing the append-only event file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as lock:
        if lock.seek(0, os.SEEK_END) == 0:
            lock.write(b"0")
            lock.flush()
        deadline = time.monotonic() + timeout
        while True:
            lock.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except (BlockingIOError, OSError):
                if time.monotonic() >= deadline:
                    raise TimeoutError("Experiment event writer lock timed out") from None
                time.sleep(0.025)
        try:
            yield
        finally:
            lock.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def verify_events(path: str | Path, *, run_id: str | None = None) -> list[dict]:
    """Verify every event and link, refusing truncated tails without rewriting."""
    path = Path(path)
    if not path.exists():
        return []
    previous = "0" * 64
    events = []
    with path.open("rb") as stream:
        for index, line in enumerate(stream):
            if not line.endswith(b"\n"):
                raise EventIntegrityError("Interrupted event tail is preserved; create a linked recovery run")
            try:
                event = json.loads(line)
                recorded = event.pop("event_sha256")
                if (event["schema"] != EVENT_SCHEMA or event["sequence"] != index
                        or event["previous_sha256"] != previous or _digest(event) != recorded
                        or (run_id is not None and event["run_id"] != run_id)):
                    raise ValueError("Event chain or run identity mismatch")
            except (ValueError, KeyError, TypeError) as exc:
                raise EventIntegrityError(f"Invalid experiment event {index}: {exc}") from exc
            event["event_sha256"] = recorded
            events.append(event)
            previous = recorded
    return events


def _last_event(path: Path) -> dict | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    with path.open("rb") as stream:
        length = stream.seek(0, os.SEEK_END)
        width = min(length, 8192)
        while True:
            stream.seek(length - width)
            tail = stream.read(width)
            if not tail.endswith(b"\n"):
                raise EventIntegrityError("Partial event tail cannot be overwritten")
            lines = tail.splitlines()
            if len(lines) >= 2 or width == length:
                line = lines[-1]
                break
            width = min(length, width * 2)
            if width > MAX_EVENT_BYTES + 1:
                raise EventIntegrityError("Experiment event exceeds the bounded record size")
    try:
        event = json.loads(line)
        recorded = event.pop("event_sha256")
        if _digest(event) != recorded:
            raise ValueError("Last event hash mismatch")
        event["event_sha256"] = recorded
        return event
    except (ValueError, KeyError, TypeError) as exc:
        raise EventIntegrityError(f"Invalid experiment tail: {exc}") from exc


def validate_whole_brain(graph: Any, model: Any, source_dir: str | Path) -> dict:
    """Check the actual graph arrays and recurrent buffer against pinned bytes."""
    from .connectome import POST_COLUMN, PRE_COLUMN, WEIGHT_COLUMN, verify_source_files
    from .whole_brain import WholeBrainStudent
    import pyarrow.parquet as pq
    import torch

    if graph.n_neurons != FULL_BRAIN_NEURONS or graph.n_edges != FULL_BRAIN_EDGES:
        raise ExperimentContractError("Whole-brain experiment cannot use a neuron/edge subset")
    if not isinstance(model, WholeBrainStudent):
        raise ExperimentContractError("Actor must be WholeBrainStudent; MLP/reservoir fallback is prohibited")
    if model.graph is not graph or model.n_neurons != FULL_BRAIN_NEURONS:
        raise ExperimentContractError("Actor does not retain the audited complete graph")
    indices = np.asarray(graph.source_indices)
    if not np.array_equal(indices, np.arange(FULL_BRAIN_NEURONS)):
        raise ExperimentContractError("Full graph source indices were reordered or filtered")
    source_dir = Path(source_dir)
    verified = verify_source_files(source_dir)
    with (source_dir / "Completeness_783.csv").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        next(reader)
        ids = tuple(row[0] for row in reader)
    if tuple(graph.node_ids) != ids:
        raise ExperimentContractError("Actor neuron IDs do not match the pinned completeness row order")
    parquet = pq.ParquetFile(source_dir / "Connectivity_783.parquet")
    if parquet.metadata.num_rows != FULL_BRAIN_EDGES:
        raise ExperimentContractError("Pinned source edge count changed")
    offset = 0
    for batch in parquet.iter_batches(columns=[PRE_COLUMN, POST_COLUMN, WEIGHT_COLUMN], batch_size=262144):
        count = batch.num_rows
        expected = (batch.column(0).to_numpy().astype(np.int32),
                    batch.column(1).to_numpy().astype(np.int32),
                    batch.column(2).to_numpy().astype(np.float32))
        actual = (graph.pre[offset:offset + count], graph.post[offset:offset + count],
                  graph.signed_weights[offset:offset + count])
        if not all(np.array_equal(a, b) for a, b in zip(actual, expected)):
            raise ExperimentContractError("Graph connections or signed counts differ from the complete pinned source")
        offset += count
    recurrent = model.recurrent
    if recurrent.layout != torch.sparse_coo or tuple(recurrent.shape) != (FULL_BRAIN_NEURONS, FULL_BRAIN_NEURONS):
        raise ExperimentContractError("Actor has no full sparse recurrent matrix")
    recurrent = recurrent.coalesce()
    if recurrent._nnz() != FULL_BRAIN_EDGES or recurrent.requires_grad:
        raise ExperimentContractError("Recurrent edges were dropped or made unconstrained trainable weights")
    expected_matrix = graph.incoming_matrix(normalize=False, gain=SYNAPTIC_MV_PER_COUNT).tocoo(copy=False)
    if expected_matrix.nnz != FULL_BRAIN_EDGES:
        raise ExperimentContractError("Source edges collapse or disappear in recurrent assembly")
    for start in range(0, FULL_BRAIN_EDGES, 262144):
        stop = min(start + 262144, FULL_BRAIN_EDGES)
        row = recurrent.indices()[0, start:stop].detach().cpu().numpy()
        col = recurrent.indices()[1, start:stop].detach().cpu().numpy()
        values = recurrent.values()[start:stop].detach().cpu().numpy()
        if (not np.array_equal(row, expected_matrix.row[start:stop])
                or not np.array_equal(col, expected_matrix.col[start:stop])
                or not np.allclose(values, expected_matrix.data[start:stop], rtol=1e-6, atol=1e-7)):
            raise ExperimentContractError("Actor recurrence is pruned, normalized, reversed or incorrectly scaled")
    input_ids = model.input_indices.detach().cpu().numpy()
    output_ids = model.output_indices.detach().cpu().numpy()
    for label, values in (("input", input_ids), ("output", output_ids)):
        if (values.ndim != 1 or not len(values) or len(np.unique(values)) != len(values)
                or values.min() < 0 or values.max() >= FULL_BRAIN_NEURONS):
            raise ExperimentContractError(f"Invalid biological {label} port indices")
    if np.intersect1d(input_ids, output_ids).size:
        raise ExperimentContractError("Sensory inputs directly overlap descending output ports")
    proof = {"neurons": FULL_BRAIN_NEURONS, "directed_edges": FULL_BRAIN_EDGES,
            "neuron_coverage": 1.0, "connection_record_coverage": 1.0,
            "source_files": verified, "synaptic_mv_per_signed_count": SYNAPTIC_MV_PER_COUNT,
            "incoming_normalization": False, "recurrent_weights_frozen": True,
            "neuron_ids_sha256": _digest(list(graph.node_ids)),
            "graph_arrays_sha256": _array_digest(graph.pre, graph.post, graph.signed_weights),
            "input_neurons": len(input_ids), "output_neurons": len(output_ids),
            "input_indices_sha256": _array_digest(input_ids), "output_indices_sha256": _array_digest(output_ids),
            "model_class": f"{type(model).__module__}.{type(model).__name__}",
            "coverage_meaning": "all source nodes/edges are represented; not a claim that all neurons fired"}
    if model.recurrent_gain_limit:
        import math
        limit = model.recurrent_gain_limit
        gain_parameter = model.recurrent_gain_logits
        if (not math.isfinite(limit) or not 0 < limit < 1 or gain_parameter is None
                or tuple(gain_parameter.shape) != (FULL_BRAIN_NEURONS,)
                or not bool(torch.isfinite(gain_parameter).all())):
            raise ExperimentContractError("Invalid bounded internal plasticity configuration")
        proof.update(recurrent_weights_frozen=False, base_recurrent_weights_frozen=True,
                     internal_plasticity=model.recurrent_plasticity_specification(),
                     internal_plasticity_state=model.recurrent_plasticity_summary())
    if model.propagation_backend != 'coo-mm':
        from .whole_brain import numerical_runtime_specification
        model.verify_propagation_buffers()
        proof['numerical_runtime']=numerical_runtime_specification(model.propagation_backend)
        proof['actual_forward_and_adjoint_full_source_matrices_verified']=True
        implementation = (model.numerical_implementation_metadata()
                          if model.propagation_backend == 'coo-cusparse-cached' else None)
        if implementation is not None:
            proof['native_sparse_implementation'] = implementation
    return proof


class NeuralActivityAccumulator:
    """Aggregate real per-decision spike counts; preserve silent-neuron evidence."""
    def __init__(self, output_indices: Iterable[int], *, neurons: int = FULL_BRAIN_NEURONS):
        if neurons != FULL_BRAIN_NEURONS:
            raise ExperimentContractError("Whole-brain activity requires the complete indexed neuron population")
        self.outputs = np.asarray(list(output_indices), dtype=np.int64)
        if (self.outputs.ndim != 1 or not len(self.outputs) or len(np.unique(self.outputs)) != len(self.outputs)
                or self.outputs.min() < 0 or self.outputs.max() >= neurons):
            raise ValueError("Invalid descending output indices")
        self.neurons = neurons
        self.ever_active = np.zeros(neurons, dtype=bool)
        self.output_ever_active = np.zeros(len(self.outputs), dtype=bool)
        self.spikes = self.output_spikes = self.decision_windows = self.batch_windows = 0
        self.total_window_ms = 0.0

    def add(self, counts: Any, window_ms: float) -> None:
        if not math.isfinite(window_ms) or window_ms <= 0:
            raise ValueError("Use the actual positive neural integration window, in milliseconds")
        counts = counts.detach().cpu().numpy() if hasattr(counts, "detach") else np.asarray(counts)
        if counts.ndim == 1:
            counts = counts[None, :]
        if (counts.ndim != 2 or counts.shape[1] != self.neurons or counts.shape[0] < 1
                or not np.isfinite(counts).all() or (counts < 0).any()
                or not np.allclose(counts, np.round(counts))):
            raise ValueError("Activity requires real nonnegative whole-window spike counts [batch,neurons]")
        active = (counts > 0).any(axis=0)
        self.ever_active |= active
        self.output_ever_active |= active[self.outputs]
        self.spikes += int(counts.sum(dtype=np.float64))
        self.output_spikes += int(counts[:, self.outputs].sum(dtype=np.float64))
        self.decision_windows += 1
        self.batch_windows += counts.shape[0]
        self.total_window_ms += window_ms * counts.shape[0]

    def summary(self) -> dict:
        seconds = self.total_window_ms / 1000
        return {"scope": "accumulated measured neural decision windows", "represented_neurons": self.neurons,
                "decision_windows": self.decision_windows, "batch_windows": self.batch_windows,
                "simulated_batch_seconds": seconds, "total_spikes": self.spikes,
                "ever_active_neurons": int(self.ever_active.sum()),
                "ever_active_neuron_fraction": float(self.ever_active.mean()),
                "mean_neuron_rate_hz": self.spikes / (seconds * self.neurons) if seconds else 0.0,
                "descending_neurons": len(self.outputs), "descending_spikes": self.output_spikes,
                "ever_active_descending_neurons": int(self.output_ever_active.sum()),
                "mean_descending_rate_hz": self.output_spikes / (seconds * len(self.outputs)) if seconds else 0.0}


def neural_activity_summary(state: Any, *, window_ms: float, output_indices: Iterable[int]) -> dict:
    if not hasattr(state, "window_spike_counts"):
        raise ExperimentContractError("Do not infer window activity from only the final timestep's spikes")
    accumulator = NeuralActivityAccumulator(output_indices)
    accumulator.add(state.window_spike_counts, window_ms)
    summary = accumulator.summary()
    summary["scope"] = "one explicitly measured neural decision window"
    for name in ("membrane", "synaptic"):
        values = getattr(state, name, None)
        if values is not None:
            values = values.detach().cpu().numpy() if hasattr(values, "detach") else np.asarray(values)
            if not np.isfinite(values).all():
                raise FloatingPointError(f"Non-finite whole-brain {name} state")
            summary[f"{name}_min_mv"] = float(values.min())
            summary[f"{name}_max_mv"] = float(values.max())
    return summary


class ExperimentLogger:
    def __init__(self, run_dir: str | Path, config: dict, *, files: dict[str, str | Path], resume: bool = False):
        self.directory = Path(run_dir).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "events.jsonl"
        self.manifest_path = self.directory / "manifest.json"
        self.lock_path = self.directory / ".events.lock"
        self._lock = threading.RLock()
        self._coverage: dict | None = None
        self._split_registry: dict[str, dict] = {}
        immutable_config = json.loads(_encoded(config))
        if ("final_acceptance_criterion" in immutable_config
                and immutable_config["final_acceptance_criterion"] != FINAL_CAMPAIGN_CRITERION):
            raise ExperimentContractError("A curriculum experiment cannot narrow the user's final Lunatic six-stage goal")
        immutable_config["final_acceptance_criterion"] = FINAL_CAMPAIGN_CRITERION
        file_paths = {str(k): str(Path(v).resolve()) for k, v in files.items()}
        if self.manifest_path.exists():
            if not resume:
                raise FileExistsError("Experiment exists; choose a new run or explicitly resume identical inputs")
            self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if self.manifest.get("schema") != MANIFEST_SCHEMA:
                raise ExperimentContractError("Unsupported experiment manifest")
            self.run_id = self.manifest["run_id"]
            events = verify_events(self.path, run_id=self.run_id)
            if (not events or events[0]["event"] != "experiment_created"
                    or events[0]["payload"]["manifest_sha256"] != file_sha256(self.manifest_path)
                    or _digest(self.manifest["config"]) != self.manifest["config_sha256"]):
                raise EventIntegrityError("The immutable manifest differs from its creation event")
            for event in events:
                if event["event"] == "dataset_split":
                    self._split_registry[event["payload"].get("name", "main")] = event["payload"]
            if (immutable_config != self.manifest["config"]
                    or file_paths != {k: v["path"] for k, v in self.manifest["files"].items()}):
                self.event("resume_rejected", {"reason": "configuration or input paths changed"})
                raise ExperimentContractError("Cannot resume an experiment with changed configuration or data paths")
            try:
                self.verify_inputs()
            except BaseException as exc:
                self.record_error(exc, phase="resume_input_validation")
                raise
            self.event("experiment_resumed", {"manifest_sha256": file_sha256(self.manifest_path)})
        else:
            if resume:
                raise FileNotFoundError("No existing experiment manifest to resume")
            if self.path.exists():
                raise EventIntegrityError("An orphan event stream is preserved; choose a new recovery directory")
            self.run_id = uuid.uuid4().hex
            # Capture all sources before hashing any of them, so a later input
            # changing while earlier files are copied also rejects creation.
            stamps = {key: _file_stamp(Path(path)) for key, path in file_paths.items()}
            entries = {key: self._pin_input(Path(path), stamps[key]) for key, path in file_paths.items()}
            for key, path in file_paths.items():
                if _file_stamp(Path(path)) != stamps[key]:
                    raise ExperimentContractError(f"Pinned input changed during experiment initialization: {key}")
            self.manifest = {"schema": MANIFEST_SCHEMA, "run_id": self.run_id,
                "created_utc": datetime.now(timezone.utc).isoformat(), "config": immutable_config,
                "config_sha256": _digest(immutable_config),
                "input_archive_policy": {"max_bytes": SOURCE_ARCHIVE_MAX_BYTES,
                                         "suffixes": sorted(SOURCE_ARCHIVE_SUFFIXES)},
                "files": entries}
            self.verify_archives()
            with self.manifest_path.open("xb") as stream:
                stream.write(_encoded(self.manifest) + b"\n")
                stream.flush()
                os.fsync(stream.fileno())
            self.event("experiment_created", {"manifest_sha256": file_sha256(self.manifest_path)})

    def _pin_input(self, path: Path, expected_stamp: tuple[int, ...]) -> dict:
        """Keep small source bytes under their hash; never copy game/data binaries."""
        if _file_stamp(path) != expected_stamp:
            raise ExperimentContractError(f"Pinned input changed during experiment initialization: {path}")
        archive_source = path.suffix.lower() in SOURCE_ARCHIVE_SUFFIXES and expected_stamp[2] <= SOURCE_ARCHIVE_MAX_BYTES
        if archive_source:
            # This read is bounded even if another writer unexpectedly grows it.
            with path.open("rb") as stream:
                content = stream.read(SOURCE_ARCHIVE_MAX_BYTES + 1)
            if len(content) != expected_stamp[2] or _file_stamp(path) != expected_stamp:
                raise ExperimentContractError(f"Pinned source changed while being read: {path}")
            digest = hashlib.sha256(content).hexdigest()
        else:
            content = None
            digest = file_sha256(path)
        if _file_stamp(path) != expected_stamp:
            raise ExperimentContractError(f"Pinned input changed while being hashed: {path}")
        entry = {"path": str(path), "sha256": digest, "bytes": expected_stamp[2]}
        if content is not None:
            relative = Path("inputs") / f"{digest}{path.suffix.lower()}"
            archive = self.directory / relative
            archive.parent.mkdir(exist_ok=True)
            try:
                with archive.open("xb") as stream:
                    stream.write(content)
                    stream.flush()
                    os.fsync(stream.fileno())
            except FileExistsError:
                pass  # Identical sources can share an already verified hash.
            if (archive.stat().st_size != len(content) or file_sha256(archive) != digest):
                raise EventIntegrityError(f"Pinned source archive was modified or interrupted: {archive}")
            if _file_stamp(path) != expected_stamp or file_sha256(path) != digest:
                raise ExperimentContractError(f"Pinned source changed while being archived: {path}")
            entry["archive_path"] = relative.as_posix()
        return entry

    def event(self, name: str, payload: dict | None = None) -> dict:
        if not isinstance(name, str) or not name:
            raise ValueError("Event name must be nonempty text")
        with self._lock, _file_lock(self.lock_path):
            last = _last_event(self.path)
            if last and last["run_id"] != self.run_id:
                raise EventIntegrityError("Run IDs changed inside an event stream")
            event = {"schema": EVENT_SCHEMA, "run_id": self.run_id,
                     "sequence": last["sequence"] + 1 if last else 0,
                     "utc": datetime.now(timezone.utc).isoformat(), "monotonic_ns": time.monotonic_ns(),
                     "event": name, "payload": _plain(payload or {}),
                     "previous_sha256": last["event_sha256"] if last else "0" * 64}
            event["event_sha256"] = _digest(event)
            line = _encoded(event) + b"\n"
            if len(line) > MAX_EVENT_BYTES:
                raise ValueError("Store large arrays as hashed artifacts, not oversized JSONL events")
            with self.path.open("ab") as stream:
                stream.write(line)
                stream.flush()
                os.fsync(stream.fileno())
            return event

    def verify_inputs(self) -> None:
        for label, entry in self.manifest["files"].items():
            path = Path(entry["path"])
            if not path.is_file() or path.stat().st_size != entry["bytes"] or file_sha256(path) != entry["sha256"]:
                raise ExperimentContractError(f"Immutable experiment input changed: {label}")
        self.verify_archives()

    def verify_archives(self) -> dict[str, str]:
        """Verify saved source bytes independently of the current source paths.

        Older manifests without archive fields remain valid and are never
        rewritten. A changed/deleted source can be recovered manually from the
        returned archive, while continuing the old run still rejects that source.
        """
        verified = {}
        for label, entry in self.manifest["files"].items():
            if "archive_path" not in entry:
                continue
            path = (self.directory / entry["archive_path"]).resolve()
            if (not path.is_relative_to((self.directory / "inputs").resolve())
                    or not path.is_file() or path.stat().st_size != entry["bytes"]
                    or file_sha256(path) != entry["sha256"]):
                raise EventIntegrityError(f"Immutable source archive changed or is missing: {label}")
            verified[label] = str(path)
        return verified

    def archived_input(self, label: str) -> Path:
        """Return verified historic bytes without overwriting a live source."""
        archives = self.verify_archives()
        if label not in archives:
            raise KeyError(f"No archived small source/configuration for {label!r}; legacy and large inputs are hash-only")
        return Path(archives[label])

    def bind_whole_brain(self, graph: Any, model: Any, source_dir: str | Path,
                         *, ports_manifest: str | Path | None = None) -> dict:
        with self.phase("whole_brain_coverage_validation"):
            self.verify_inputs()
            pinned_paths = {entry["path"] for entry in self.manifest["files"].values()}
            for filename in ("Completeness_783.csv", "Connectivity_783.parquet"):
                if str((Path(source_dir) / filename).resolve()) not in pinned_paths:
                    raise ExperimentContractError("Pin both original graph source files in the experiment manifest")
            proof = validate_whole_brain(graph, model, source_dir)
            if ports_manifest is not None:
                path = Path(ports_manifest).resolve()
                if str(path) not in {entry["path"] for entry in self.manifest["files"].values()}:
                    raise ExperimentContractError("Biological port mapping was not pinned at experiment creation")
                ports = json.loads(path.read_text(encoding="utf-8"))
                allowed_inputs = set(ports["input_indices"])
                expected_outputs = set(ports["output_indices"])
                inputs = set(model.input_indices.detach().cpu().tolist())
                outputs = set(model.output_indices.detach().cpu().tolist())
                if not inputs.issubset(allowed_inputs) or outputs != expected_outputs:
                    raise ExperimentContractError("Model ports diverge from the fixed visual/descending annotation mapping")
                proof["ports_manifest_sha256"] = file_sha256(path)
            self._coverage = proof
            self.event("whole_brain_verified", proof)
            return proof

    def record_split(self, train_ids: Iterable[str], validation_ids: Iterable[str], test_ids: Iterable[str] = (),
                     *, name: str = "main") -> None:
        groups = {"train": list(map(str, train_ids)), "validation": list(map(str, validation_ids)),
                  "test": list(map(str, test_ids))}
        if not groups["train"] or not groups["validation"]:
            raise ExperimentContractError("Train and validation must each contain complete episodes")
        for label, values in groups.items():
            if len(values) != len(set(values)):
                raise ExperimentContractError(f"Duplicate {label} episode IDs")
        if (set(groups["train"]) & set(groups["validation"]) or set(groups["train"]) & set(groups["test"])
                or set(groups["validation"]) & set(groups["test"])):
            raise ExperimentContractError("Episode leakage across training, validation or final test")
        existing = self._split_registry.get(name)
        if existing and any(groups[role] != existing[role] for role in groups):
            raise ExperimentContractError("A named experimental split is immutable")
        for prior in self._split_registry.values():
            for first in groups:
                for second in groups:
                    if first != second and set(groups[first]) & set(prior[second]):
                        raise ExperimentContractError("An episode reserved for validation/test cannot enter another curriculum's training")
        payload = {"name": name, "unit": "complete_episode", **groups, "split_sha256": _digest(groups)}
        self.event("dataset_split", payload)
        self._split_registry[name] = payload

    def record_curriculum(self, name: str, *, difficulty: int, stages: Iterable[int], description: str) -> None:
        stages = list(stages)
        if difficulty not in range(4) or not stages or any(s not in range(1, 7) for s in stages):
            raise ValueError("Record the actual TH14 difficulty 0..3 and story stages 1..6")
        self.event("curriculum_phase", {"name": name, "difficulty": difficulty, "stages": stages,
            "description": description, "counts_as_final_acceptance": False,
            "final_acceptance_criterion": FINAL_CAMPAIGN_CRITERION})

    def record_epoch(self, epoch: int, metrics: dict, *, elapsed_seconds: float, activity: dict) -> None:
        if self._coverage is None:
            raise ExperimentContractError("Verify the complete graph/model before recording a whole-brain training epoch")
        if epoch < 0 or not math.isfinite(elapsed_seconds) or elapsed_seconds < 0:
            raise ValueError("Invalid epoch index or elapsed time")
        if activity.get("represented_neurons") != FULL_BRAIN_NEURONS:
            raise ExperimentContractError("Epoch activity excludes part of the full indexed brain")
        # _encoded rejects NaN/Inf, preserving a valid event stream on failures.
        _encoded(metrics)
        _encoded(activity)
        self.event("epoch", {"epoch": epoch, "metrics": metrics, "elapsed_seconds": elapsed_seconds,
                             "activity": activity, "graph_arrays_sha256": self._coverage["graph_arrays_sha256"]})

    def record_checkpoint(self, path: str | Path, *, tag: str, epoch: int) -> dict:
        if self._coverage is None:
            raise ExperimentContractError("Cannot tag an unverified subset/MLP as a whole-brain checkpoint")
        path = Path(path).resolve(strict=True)
        self.verify_inputs()
        digest = file_sha256(path)
        archive_dir = self.directory / "checkpoints"
        archive_dir.mkdir(exist_ok=True)
        safe_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", tag).strip(".") or "checkpoint"
        archive = archive_dir / f"epoch_{epoch:06d}_{safe_tag}_{digest[:16]}{path.suffix or '.bin'}"
        if archive.exists():
            if file_sha256(archive) != digest:
                raise EventIntegrityError("An immutable checkpoint archive was modified")
        else:
            partial = archive.with_name(archive.name + f".partial_{uuid.uuid4().hex}")
            try:
                with path.open("rb") as source, partial.open("xb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
                    target.flush()
                    os.fsync(target.fileno())
                if file_sha256(partial) != digest or file_sha256(path) != digest:
                    raise ExperimentContractError("Checkpoint changed while it was being archived")
                # Exclusive final creation preserves earlier failed/best models.
                with archive.open("xb") as target, partial.open("rb") as source:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
                    target.flush()
                    os.fsync(target.fileno())
                partial.unlink()
            except BaseException as error:
                self.record_error(error, phase="checkpoint_archive", context={"partial_path": str(partial)})
                raise
        entry = {"path": str(archive), "source_path": str(path), "sha256": digest, "bytes": archive.stat().st_size,
                 "tag": tag, "epoch": epoch, "config_sha256": self.manifest["config_sha256"],
                 "graph_arrays_sha256": self._coverage["graph_arrays_sha256"]}
        self.event("checkpoint", entry)
        return entry

    def record_campaign_result(self, result: dict, *, evidence_files: dict[str, str | Path]) -> dict:
        """Keep failures as well as successes under the unchanged final criterion.

        These checks audit reported native measurements. The linked trajectories
        still require review; a passed metadata gate alone is not proof of the
        thread's overall objective.
        """
        expected_sha = self.manifest["config"].get("game_exe_sha256")
        checks = {
            "formal_executable": bool(expected_sha and result.get("exe_sha256") == expected_sha
                                      and expected_sha != TRIAL_TH14_SHA256),
            "lunatic": result.get("difficulty") in (3, "Lunatic"),
            "six_stages_in_order": result.get("completed_stages") == [1, 2, 3, 4, 5, 6],
            "ending_verified": result.get("ending_verified") is True,
            "not_game_over": result.get("native_gameover") is False,
            "finite_native_lives": result.get("finite_native_lives") is True,
            "unmodified_native_resources": result.get("resource_overrides") is False,
            "no_continues": result.get("continues_used") == 0,
            "no_teacher": result.get("teacher_action_calls") == 0,
            "no_safety_takeovers": result.get("safety_takeovers") == 0,
            "no_brain_resets_between_stages": result.get("brain_state_resets_between_stages") == 0,
            "no_resource_resets_between_stages": result.get("native_resource_resets_between_stages") == 0,
            "native_story_run": result.get("run_mode") == 0,
            "not_replay_playback": result.get("replay_playback") == 0,
            "not_control_intervention": self.manifest["config"].get("intervention") in (None, "none")
                                        and self.manifest["config"].get("control_only") is not True,
            "whole_brain_bound": self._coverage is not None,
            "trajectory_evidence_present": bool(evidence_files),
        }
        artifacts = {name: {"path": str(Path(path).resolve()), "sha256": file_sha256(path),
                            "bytes": Path(path).stat().st_size} for name, path in evidence_files.items()}
        payload = {"result": result, "criterion": FINAL_CAMPAIGN_CRITERION, "checks": checks,
                   "reported_measurements_pass_gate": all(checks.values()), "evidence": artifacts,
                   "evidence_scope": "Reported measurements and hashed native evidence; not an automatic declaration that the entire user goal is complete"}
        self.event("campaign_result", payload)
        return payload

    def record_native_context(self, snapshot: Any, info: dict) -> dict:
        snapshot.validate()
        digest = info.get("exe_sha256")
        expected = self.manifest["config"].get("game_exe_sha256")
        if (not expected or digest != expected or digest == TRIAL_TH14_SHA256
                or expected not in {entry["sha256"] for entry in self.manifest["files"].values()}):
            raise ExperimentContractError("Final-target runs require the pinned supplied formal TH14 executable, not the trial")
        if (not str(snapshot.extra.get("backend_kind", "")).startswith("original-th14")
                or snapshot.extra.get("desktop_input") is not False):
            raise ExperimentContractError("Expected background original-game process-local control")
        context = {"exe_sha256": digest, "version": snapshot.version, "stage": snapshot.stage,
                   "difficulty": snapshot.difficulty, "character": snapshot.character,
                   "subshot": snapshot.subshot, "native_extra": snapshot.extra,
                   "game_rng_seed_applied": bool(info.get("game_rng_seed_applied", False))}
        self.event("formal_original_game_context", context)
        return context

    def record_proposal(self, name: str, configuration_changes: dict, *, rationale: str,
                        parent_run: str | None = None) -> None:
        """Preserve candidate changes without mutating this run's fixed config."""
        self.event("proposal", {"name": name, "configuration_changes": configuration_changes,
            "rationale": rationale, "parent_run": parent_run,
            "execution_rule": "Actual changed settings require a new immutable experiment run"})

    def record_runtime(self) -> None:
        import torch
        versions = {}
        for name in ("numpy", "scipy", "pandas", "pyarrow", "frida", "pywin32", "gymnasium", "stable-baselines3"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        self.event("runtime", {"python": sys.version, "executable": sys.executable,
            "platform": platform.platform(), "torch": str(torch.__version__),
            "packages": versions,
            "cuda_available": torch.cuda.is_available(), "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "determinism_note": "Seeds and software are recorded; bitwise GPU reproducibility is not presumed"})

    def record_error(self, error: BaseException, *, phase: str, context: dict | None = None) -> None:
        self.event("error", {"phase": phase, "exception_type": type(error).__name__, "message": str(error),
            "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
            "context": context or {}})

    @contextmanager
    def phase(self, name: str):
        start = time.monotonic()
        self.event("phase_started", {"phase": name})
        try:
            yield
        except BaseException as error:
            self.record_error(error, phase=name, context={"elapsed_seconds": time.monotonic() - start})
            self.event("phase_failed", {"phase": name, "elapsed_seconds": time.monotonic() - start})
            raise
        else:
            self.event("phase_finished", {"phase": name, "elapsed_seconds": time.monotonic() - start})
