"""Read the reviewed CPU Phi opening prefix, without constructing a policy.

This bounded reader accepts only source-row-zero prefixes. It returns fixed
9721 FP32 sensory vectors, source metadata and immutable teacher-file pointers.
No teacher tensor, trainable encoder, brain, optimizer or game is loaded. The
CLI is limited to the two approved world000 opening packets (64 rows).
"""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "2"

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import sys
import zipfile

from . import fly_sensory_cache as producer

READER_SCHEMA = "fly-v5-sensory9721-continuous-prefix-reader-v1"
PINNED_PRODUCER_SOURCE = "c36188635d63b341b9daec3738a7ed7b685b9c6587d7c31ceab0f2d0468c8420"
PINNED_ENCODER_CLASS_SOURCE = "dc689534b7ed37dc3a3b6acae0c41261424e289e7b7c3496bc2cf997c0dfebc9"
DEFAULT_PLAN = (Path(__file__).resolve().parents[2] / "artifacts/engineering/B1-autonomous-dodge-20261024/"
                "fly-port-feasibility-v1/student-prototype/sensory-cache-preparation-v1/continuous-prefix64-plan.json")
MAX_ROWS = 64
MAX_PACKET_BYTES = 2 * 1024 ** 2


@dataclass(frozen=True)
class PrefixContract:
    """Identity pins. Non-production identities require explicit fixture scope."""

    plan_sha256: str
    packet_sha256: tuple[str, ...]
    original_episode_rows: int
    native_seed: int
    bank_identity_sha256: str
    bank_identity_file_sha256: str
    source_index_sha256: str
    bank_episode_commit_sha256: str
    synthetic_fixture: bool = False


REAL_CONTRACT = PrefixContract(
    "b5bad91cb91e5330bfcbe3dbedf324d193a7b71abcbcf0a1f643e9d012021a48",
    ("04f93709e62d5425fc8aeaa4e1b2a79489e15911f02b17f21505f6d90ac100b3",
     "7eaa084d10f9373fdc7b3f52708a9ef809cccf42a2fd0f17dff783585c298a29"),
    24755, 11655, producer.PINNED_BANK_IDENTITY,
    "7f5b83283055ab7c5ac830e77ded314a8468fcc732c377731d6d6e8c70e29f03",
    "f289a401ed208d1430c0b3cd30d9ed5d6cb999218d64ca00b41b5fc95df8487e",
    "cb332928917cae2e8888014c12bd903e24116dfeece169084df32a1d1987a2f9",
)

EXPECTED_ADAPTER = dict(
    input_schema="semantic-separated-observed-laser-segments-v5", features=[25],
    maps=[16, 92, 96], objects=[96, 29], map_pool_shape=[12, 12],
    map_statistics_in_order=["mean", "maximum", "minimum"],
    vector_order=["globals", "all96_object_tokens_row_major", "map_statistics_channel_major"],
    output_dim=9721, trainable_parameters=0, map_compression_lossless=False,
    retina_reconstruction_claimed=False,
)
EXPECTED_ENCODER = dict(
    cache_kind=producer.CACHE_KIND, adapter=EXPECTED_ADAPTER, dtype="float32",
    encoder_parameter_count=0, student_source_sha256=producer.PINNED_STUDENT_SOURCE,
    encoder_class_source_sha256=PINNED_ENCODER_CLASS_SOURCE,
    producer_source_sha256=PINNED_PRODUCER_SOURCE,
    torch_version="2.8.0+cu128", numpy_version="1.26.4", torch_CPU_threads=2,
    trainable_input_encoder_cached=False, teacher_hidden_cached=False,
    brain_or_action_head_cached=False, quantization=False,
)


@dataclass(frozen=True)
class SourceRowMetadata:
    episode: str
    source_row: int
    before_frames: int
    after_observation_frames: int
    episode_starts: bool
    dones: bool
    native_seeds: int
    stage: int
    stage_frame: int
    native_misses: int
    native_bombs_used: int
    behaviour_actions: int


@dataclass(frozen=True)
class TeacherLabelReference:
    """A verified file/row pointer, never a deserialized teacher tensor."""

    bank_root: str
    bank_identity_sha256: str
    episode: str
    path: str
    sha256: str
    file_bytes: int
    shard_start: int
    shard_rows: int
    local_row: int
    source_row: int


@dataclass(frozen=True)
class CachedSensoryRow:
    phi: object  # immutable bytes-backed numpy float32[9721], CPU only
    metadata: SourceRowMetadata
    teacher: TeacherLabelReference
    packet_sha256: str


def verify_reviewed_sources():
    """Hash source text; importing Torch or the student is unnecessary."""
    if producer.file_sha(producer.__file__) != PINNED_PRODUCER_SOURCE:
        raise ValueError("Cache producer source SHA differs from reviewed c361 version")
    student = Path(__file__).with_name("fly_distillation_student.py")
    if producer.file_sha(student) != producer.PINNED_STUDENT_SOURCE:
        raise ValueError("Student fixed-Phi source SHA differs from reviewed e97 version")


def _headers(path, n):
    """Bound allocations and reject extra/object fields before deserialization."""
    import numpy as np
    with zipfile.ZipFile(path) as archive:
        expected = {name + ".npy" for name in ("phi", "provenance_utf8", *producer.ROW_KEYS)}
        names = archive.namelist()
        if len(names) != len(expected) or set(names) != expected:
            raise ValueError("Packet must contain only Phi, provenance and original row metadata")
        if sum(item.file_size for item in archive.infolist()) > MAX_PACKET_BYTES:
            raise ValueError("Packet exceeds bounded32-row allocation limit")
        for name in names:
            with archive.open(name) as stream:
                version = np.lib.format.read_magic(stream)
                if version != (1, 0):
                    raise ValueError("Unsupported packet NPY version")
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
            key = name[:-4]
            if key == "provenance_utf8":
                valid = len(shape) == 1 and 1 <= shape[0] <= 32768 and dtype == np.uint8
            else:
                expected_dtype = np.float32 if key == "phi" else (
                    np.bool_ if key in ("episode_starts", "dones") else np.int64)
                valid = shape == ((n, 9721) if key == "phi" else (n,)) and dtype == expected_dtype
            if not valid or fortran or dtype.hasobject:
                raise ValueError("Packet shape/dtype/layout differs: " + key)


class ContinuousPrefixReader:
    """Strict, bounded source-zero stream; it does not own or reset brain state.

    Inspection validates hashes, references, headers and all source metadata.
    Phi finiteness/range is checked when explicitly iterating a prefix. A caller
    must start its own student state at source0 and retain it across packets,
    stage changes and misses. A late packet cannot be an initial-state prefix.
    """

    def __init__(self, plan=DEFAULT_PLAN, guard=None, *, contract=REAL_CONTRACT):
        self.plan_path = Path(plan)
        self.guard = guard or producer.CpuGuard(self.plan_path, max_seconds=120.)
        self.contract = contract
        self.guard.check("reader_before_identity")
        verify_reviewed_sources()
        if not contract.synthetic_fixture and contract != REAL_CONTRACT:
            raise ValueError("Custom identities require explicit synthetic_fixture=True")
        if producer.file_sha(self.plan_path) != contract.plan_sha256:
            raise ValueError("Approved continuous prefix plan SHA differs")
        self.plan = producer.read_json(self.plan_path)
        self._validate_plan()
        self._packets = []
        self.bank = self.episode = None
        self._open()

    def _validate_plan(self):
        p = self.plan
        files = p.get("cache_files", [])
        n = p.get("rows")
        if (type(n) is not int or not 1 <= n <= MAX_ROWS or p.get("episode") != "world000"
                or p.get("original_episode_rows") != self.contract.original_episode_rows
                or p.get("source_rows") != [0, n - 1]
                or p.get("complete_cached_episode") is not False
                or p.get("requires_own_student_state_from_source_row0") is not True
                or p.get("late_probe_files_excluded") is not True
                or p.get("late_zero_state_or_cross_episode_generalization_claim") is not False
                or len(files) != len(self.contract.packet_sha256) or not files):
            raise ValueError("Require a bounded source0 prefix; late/partial-history zero-state entry is forbidden")
        offset = 0
        for spec, expected_sha in zip(files, self.contract.packet_sha256):
            if (type(spec.get("start")) is not int or spec["start"] != offset
                    or type(spec.get("rows")) is not int or not 1 <= spec["rows"] <= 32
                    or spec.get("sha256") != expected_sha):
                raise ValueError("Prefix packets must start at source0 and remain consecutive with pinned packet SHA")
            offset += spec["rows"]
        if offset != n:
            raise ValueError("Prefix packet row count differs")

    def _verify_packet_file(self, spec):
        self.guard.check("reader_packet_hash")
        path = Path(spec["path"])
        if path.stat().st_size > MAX_PACKET_BYTES or producer.file_sha(path) != spec["sha256"]:
            raise ValueError("Cache packet SHA/size differs: " + str(path))
        _headers(path, spec["rows"])
        return path

    def _open(self):
        import numpy as np
        if np.__version__ != EXPECTED_ENCODER["numpy_version"]:
            raise ValueError("Reader NumPy version differs from the reviewed CPU cache version")
        previous_after = None
        for number, spec in enumerate(self.plan["cache_files"]):
            path = self._verify_packet_file(spec)
            with np.load(path, allow_pickle=False) as archive:
                provenance = json.loads(archive["provenance_utf8"].tobytes())
                metadata = {key: archive[key] for key in producer.ROW_KEYS}
            self._validate_provenance(provenance)
            if self.bank is None:
                self.bank = producer.SensoryBank(
                    provenance["bank_root"], self.guard,
                    expected_identity=self.contract.bank_identity_sha256,
                    require_real_cohort=not self.contract.synthetic_fixture)
                self.episode = self.bank.episode("world000")
                if (self.bank.identity_file_sha != self.contract.bank_identity_file_sha256
                        or self.episode.commit_sha != self.contract.bank_episode_commit_sha256
                        or self.episode.source["index_sha256"] != self.contract.source_index_sha256
                        or self.episode.source["rows"] != self.contract.original_episode_rows
                        or self.episode.source["native_seed"] != self.contract.native_seed):
                    raise ValueError("Immutable bank/episode/source identity differs")
            episode = self.episode
            original, label = episode.source["shards"][number], episode.metadata["label_shards"][number]
            if (provenance["bank_root"] != str(self.bank.root.resolve())
                    or provenance["source_index"] != episode.source["index"]
                    or provenance["source_shard"] != original or provenance["teacher_label_shard"] != label
                    or original["start"] != spec["start"] or original["rows"] != spec["rows"]):
                raise ValueError("Packet source/teacher pointer differs from committed original shard")
            # Hash original compressed bytes only; never decompress maps here.
            raw_path = producer.safe_child(Path(episode.source["index"]).parent, original["name"])
            producer.verify_file(raw_path, original, self.guard)
            labels_metadata = self.bank.label_rows(episode, number)
            if any(not np.array_equal(metadata[key], labels_metadata[key]) for key in producer.ROW_KEYS):
                raise ValueError("Cache row/reset/clock/seed metadata differs from immutable labels")
            if (np.any(metadata["before_frames"] < 0) or np.any(metadata["stage_frame"] < 0)
                    or np.any(metadata["native_misses"] < 0) or np.any(metadata["native_bombs_used"] < 0)
                    or np.any((metadata["behaviour_actions"] < 0) | (metadata["behaviour_actions"] >= 19))):
                raise ValueError("Source row clocks/counters/action metadata is invalid")
            if previous_after is not None and metadata["before_frames"][0] != previous_after:
                raise ValueError("Original source clock gap across prefix packets")
            previous_after = int(metadata["after_observation_frames"][-1])
            self._packets.append((dict(spec), metadata, dict(label)))

    def _validate_provenance(self, p):
        c = self.contract
        expected = dict(cache_kind=producer.CACHE_KIND, probe_sample_only=True,
                        complete_cached_episode=False, original_episode="world000",
                        original_episode_rows=c.original_episode_rows,
                        bank_identity_sha256=c.bank_identity_sha256,
                        bank_identity_file_sha256=c.bank_identity_file_sha256,
                        bank_episode_commit_sha256=c.bank_episode_commit_sha256,
                        source_index_sha256=c.source_index_sha256,
                        teacher_sha256=producer.TEACHER_SHA256, encoder=EXPECTED_ENCODER,
                        neuron_state_created=False, teacher_hidden_read_or_cached=False,
                        source_row_order="original source episode and native clocks; never reset at stage/miss")
        if any(p.get(key) != value for key, value in expected.items()):
            raise ValueError("Cache provenance/version/source SHA/fixed encoder contract differs")
        if set(p) != set(expected) | {"bank_root", "source_index", "source_shard", "teacher_label_shard"}:
            raise ValueError("Unexpected cache provenance fields")

    def inspect(self):
        return dict(schema=READER_SCHEMA, synthetic_fixture=self.contract.synthetic_fixture,
                    episode="world000", source_rows=list(self.plan["source_rows"]), rows=self.plan["rows"],
                    packets=len(self._packets), metadata_and_header_verified=True,
                    phi_payload_read=False, phi_finiteness_checked=False,
                    plan_sha256=self.contract.plan_sha256, packet_sha256=list(self.contract.packet_sha256),
                    requires_own_initial_state_at_source0=True, complete_cached_episode=False,
                    teacher_arrays_deserialized=False, raw_maps_decompressed=False,
                    cross_CPU_CUDA_or_real_full_brain_equivalence_tested=False)

    def iter_prefix(self, *, max_rows):
        """Yield an explicit finite prefix, in original complete-episode order."""
        import numpy as np
        if type(max_rows) is not int or not 1 <= max_rows <= self.plan["rows"]:
            raise ValueError("Explicit max_rows must be within this approved source0 prefix")
        count = 0
        for spec, rows, label in self._packets:
            if count == max_rows:
                return
            path = self._verify_packet_file(spec)
            self.guard.check("reader_Phi_load")
            with np.load(path, allow_pickle=False) as archive:
                phi = archive["phi"]
            if not np.isfinite(phi).all() or np.any(np.abs(phi) > 1):
                raise ValueError("Expected finite normalized FP32 fixed sensory Phi")
            # Immutable bytes backing prevents consumers re-enabling writeable.
            phi = np.frombuffer(phi.tobytes(order="C"), dtype=np.float32).reshape(-1, 9721)
            for local in range(len(phi)):
                if count == max_rows:
                    return
                self.guard.check("reader_row_yield")
                values = {key: bool(value[local]) if key in ("episode_starts", "dones") else int(value[local])
                          for key, value in rows.items()}
                metadata = SourceRowMetadata(episode="world000", **values)
                reference = TeacherLabelReference(
                    str(self.bank.root.resolve()), self.bank.identity_sha, "world000",
                    str(producer.safe_child(self.episode.folder, label["name"]).resolve()),
                    label["sha256"], label["bytes"], label["start"], label["rows"], local, metadata.source_row)
                yield CachedSensoryRow(phi[local], metadata, reference, spec["sha256"])
                count += 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("inspect", "prefix"), default="inspect")
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--max-seconds", type=float, default=120.)
    parser.add_argument("--receipt", type=Path)
    arguments = sys.argv[1:] if argv is None else argv
    if not arguments:
        parser.print_help()
        return 0
    args = parser.parse_args(arguments)
    if args.receipt is None or args.receipt.exists():
        parser.error("Choose a new --receipt path; existing evidence is preserved")
    if ((args.mode == "prefix" and (args.max_rows is None or not 1 <= args.max_rows <= MAX_ROWS))
            or args.mode == "inspect" and args.max_rows is not None):
        parser.error("Only explicit --mode prefix --max-rows 1..64 consumes Phi")
    if not 0 < args.max_seconds <= 120:
        parser.error("Need a finite wall budget in (0,120] seconds")
    guard = producer.CpuGuard(args.plan, max_seconds=args.max_seconds)
    report = dict(schema=READER_SCHEMA, mode=args.mode, status="not_started",
                  cache_or_bank_or_student_modified=False, new_cache_created=False,
                  GPU_native_brain_optimizer_started=False, teacher_hidden_read=False,
                  complete_episode_history_learning_or_gameplay_claim=False,
                  cross_CPU_CUDA_or_real_full_brain_equivalence_tested=False)
    code = 0
    try:
        report["runtime"] = producer.configure_CPU()
        guard.check("before_reader_load")
        reader = ContinuousPrefixReader(args.plan, guard)
        report["inspection"] = reader.inspect()
        if args.mode == "prefix":
            digest = hashlib.sha256()
            rows = []
            refs = {}
            for row in reader.iter_prefix(max_rows=args.max_rows):
                digest.update(row.phi.tobytes())
                rows.append(asdict(row.metadata))
                refs[row.teacher.path] = dict(path=row.teacher.path, sha256=row.teacher.sha256,
                                             bytes=row.teacher.file_bytes, start=row.teacher.shard_start,
                                             rows=row.teacher.shard_rows)
            report["prefix"] = dict(rows=len(rows), source_rows=[rows[0]["source_row"], rows[-1]["source_row"]],
                                    Phi_shape=[len(rows), 9721], dtype="float32", Phi_payload_SHA256=digest.hexdigest(),
                                    finite_bounded=True, original_row_metadata=rows, teacher_references=list(refs.values()),
                                    teacher_arrays_deserialized=False, own_brain_state_constructed=False)
        report["status"] = "completed_" + args.mode
    except (Exception, KeyboardInterrupt) as error:
        code = 2
        report.update(status="stopped_or_failed", stop_type=type(error).__name__, stop_reason=str(error),
                      no_automatic_retry_or_resource_relaxation=True)
    finally:
        report["resources"] = guard.report()
        report["torch_imported"] = "torch" in sys.modules
        report["reader_source_sha256"] = producer.file_sha(__file__)
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        with args.receipt.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps(dict(status=report["status"], receipt=str(args.receipt)), ensure_ascii=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
