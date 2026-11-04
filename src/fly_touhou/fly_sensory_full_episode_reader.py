"""CPU-only consumer of the sealed formal full-episode fixed-Phi format.

No Torch, model, teacher tensor, hidden/value state, or original game is loaded.
Opening64's reader stays unchanged: only its validation helpers and immutable
row/reference dataclasses are reused. Audit cursors are never resume inputs.
"""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "2"

import argparse
from contextlib import closing
import copy
import hashlib
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace
import zipfile

from .atomic_json_io import atomic_write_json
from .windows_physical_memory import available_physical_memory_bytes
from . import fly_sensory_cache as producer
from . import fly_sensory_cache_reader as reviewed

SCHEMA = "fly-v5-sensory9721-full-episode-CPU-consumer-v1"
PINNED_READER_SOURCE = "83679a21595caa33acaee3897a3717b8305a80021f36f7036f965b20ae1f9583"


class IncompleteOrUnknownEpisode(ValueError):
    pass


def _inventory(folder, expected):
    folder = Path(folder)
    if (not folder.is_dir() or folder.is_symlink()
            or set(p.name for p in folder.iterdir()) != set(expected)
            or any(p.is_symlink() for p in folder.iterdir())):
        raise IncompleteOrUnknownEpisode("Unknown/incomplete formal episode inventory")


def _separate_output(path, cache_root, bank_root, sources=()):
    target = Path(path).resolve()
    protected = [Path(cache_root).resolve(), Path(bank_root).resolve(),
                 *(Path(p["index"]).resolve().parent for p in sources)]
    if any(target == p or p in target.parents or target in p.parents for p in protected):
        raise ValueError("Audit/receipt output must be separate from immutable cache/bank/source")


def _teacher19_headers(path, rows):
    """Read only three NPY headers, never any teacher tensor or hidden/value."""
    import numpy as np
    with zipfile.ZipFile(path) as archive:
        for key, dtype in (("teacher_final_logits", np.float32), ("teacher_probabilities", np.float32),
                           ("legal_action_mask", np.bool_)):
            with archive.open(key + ".npy") as stream:
                if np.lib.format.read_magic(stream) != (1, 0):
                    raise ValueError("Unsupported final19 teacher NPY version")
                shape, fortran, actual = np.lib.format.read_array_header_1_0(stream)
            if shape != (rows, 19) or actual != dtype or actual.hasobject or fortran:
                raise ValueError("Immutable teacher pointer must name final19 FP32/bool fields")


class FullEpisodeSensoryReader:
    """Stream one complete source episode from row0, with bounded packet IO.

    max_rows limits rows read/yielded using whole original packets. A remaining
    budget smaller than the next packet stops before reading it. A later bad
    packet can stop an already yielded prefix; complete is true only after the
    verified true terminal row was yielded. Each invocation starts at row0.
    """

    def __init__(self, root, bank_root=producer.DEFAULT_BANK, guard=None, *, synthetic_fixture=False):
        self.root, self.bank_root = Path(root), Path(bank_root)
        self.guard = guard or producer.CpuGuard(self.root, max_seconds=120.,
                                              available_ram=available_physical_memory_bytes)
        self.synthetic_fixture = synthetic_fixture
        self.bank = None
        self._audit = None
        self._active = False
        self.guard.check("before_full_episode_reader")
        reviewed.verify_reviewed_sources()
        if producer.file_sha(reviewed.__file__) != PINNED_READER_SOURCE:
            raise ValueError("Frozen opening reader helper source differs")
        self.bank = producer.SensoryBank(self.bank_root, self.guard,
            expected_identity=None if self.synthetic_fixture else producer.PINNED_BANK_IDENTITY,
            require_real_cohort=not self.synthetic_fixture)

    def _formal_identity(self):
        self.guard.check("formal_cache_identity")
        if not self.root.is_dir() or self.root.is_symlink():
            raise IncompleteOrUnknownEpisode("Formal cache root is missing or unknown")
        if set(p.name for p in self.root.iterdir()) - {"identity.json", "episodes", "_resume"}:
            raise IncompleteOrUnknownEpisode("Unknown formal cache root files")
        identity = producer.read_json(self.root / "identity.json")
        if self.bank is None:
            self.bank = producer.SensoryBank(self.bank_root, self.guard,
                expected_identity=None if self.synthetic_fixture else producer.PINNED_BANK_IDENTITY,
                require_real_cohort=not self.synthetic_fixture)
        if producer.file_sha(self.bank.root / "identity.json") != self.bank.identity_file_sha:
            raise IncompleteOrUnknownEpisode("Immutable teacher bank identity file changed")
        expected = dict(cache_kind=producer.CACHE_KIND, bank_identity_sha256=self.bank.identity_sha,
                        bank_identity_file_sha256=self.bank.identity_file_sha, encoder=reviewed.EXPECTED_ENCODER)
        if identity != expected:
            raise IncompleteOrUnknownEpisode("Formal cache_kind/bank/FP32 Phi identity differs")
        return identity

    def _metadata(self, name):
        identity = self._formal_identity()
        folder = producer.safe_child(self.root / "episodes", name)
        if not (folder / "COMMITTED.json").is_file() or not (folder / "episode_index.json").is_file():
            raise IncompleteOrUnknownEpisode("Episode has no complete formal COMMITTED/index; inspect only")
        episode = self.bank.episode(name)
        if (producer.file_sha(episode.folder / "COMMITTED.json") != episode.commit_sha
                or producer.file_sha(episode.source["index"]) != episode.source["index_sha256"]):
            raise IncompleteOrUnknownEpisode("Immutable source/teacher episode identity changed")
        index = producer.read_json(folder / "episode_index.json")
        commit = producer.read_json(folder / "COMMITTED.json")
        if (commit != dict(cache_kind=producer.CACHE_KIND, identity_sha256=producer.json_sha(identity),
                           episode_index_sha256=producer.file_sha(folder / "episode_index.json"), rows=episode.source["rows"])
                or set(index) != {"cache_kind", "complete_cached_episode", "rows", "source_episode",
                                 "bank_episode_commit_sha256", "cache_shards"}
                or index.get("cache_kind") != producer.CACHE_KIND or index.get("complete_cached_episode") is not True
                or index.get("rows") != episode.source["rows"] or index.get("source_episode") != episode.source
                or index.get("bank_episode_commit_sha256") != episode.commit_sha
                or not isinstance(index.get("cache_shards"), list)
                or len(index["cache_shards"]) != len(episode.source["shards"])):
            raise IncompleteOrUnknownEpisode("Complete formal manifest/source identity differs")
        offset = 0
        for number, (spec, original) in enumerate(zip(index["cache_shards"], episode.source["shards"])):
            if (set(spec) != {"name", "rows", "start", "bytes", "sha256"}
                    or spec["name"] != f"phi-{number:06d}.npz"
                    or type(spec["rows"]) is not int or not 1 <= spec["rows"] <= 32
                    or type(spec["start"]) is not int or spec["start"] != offset
                    or spec["start"] != original["start"] or spec["rows"] != original["rows"]
                    or type(spec["bytes"]) is not int or not 1 <= spec["bytes"] <= reviewed.MAX_PACKET_BYTES
                    or not isinstance(spec["sha256"], str) or not re.fullmatch("[0-9a-f]{64}", spec["sha256"])):
                raise IncompleteOrUnknownEpisode("Formal packets must cover consecutive source0->terminal")
            if producer.safe_child(folder, spec["name"]).stat().st_size != spec["bytes"]:
                raise IncompleteOrUnknownEpisode("Formal packet is missing or has a different size")
            offset += spec["rows"]
        if offset != episode.source["rows"]:
            raise IncompleteOrUnknownEpisode("Formal episode does not cover the complete original source")
        _inventory(folder, {"COMMITTED.json", "episode_index.json", *(s["name"] for s in index["cache_shards"])})
        return folder, episode, index, producer.file_sha(folder / "COMMITTED.json")

    def inspect(self, episode=None):
        """Metadata only; unknown/partial episodes never become yieldable."""
        try:
            if episode is None:
                self._formal_identity()
                entries = sorted(p.name for p in (self.root / "episodes").iterdir())
                return dict(schema=SCHEMA, root=str(self.root), metadata_only=True,
                            payload_or_teacher_tensors_read=False, episodes=[self.inspect(n) for n in entries])
            folder, source, index, commit_sha = self._metadata(episode)
            return dict(schema=SCHEMA, episode=episode, source_rows=[0, source.source["rows"] - 1],
                        rows=source.source["rows"], packets=len(index["cache_shards"]), native_seed=source.source["native_seed"],
                        complete_producer_episode=True, eligible=True, metadata_only=True,
                        payload_SHA_or_finiteness_verified=False, formal_commit_sha256=commit_sha,
                        consumption_complete=False, synthetic_fixture=self.synthetic_fixture)
        except (OSError, ValueError, KeyError, TypeError) as error:
            return dict(schema=SCHEMA, episode=episode, root=str(self.root), eligible=False,
                        status="unknown_or_incomplete_inspect_only", reason=str(error), metadata_only=True,
                        payload_SHA_or_finiteness_verified=False, consumption_complete=False)

    def _audit_write(self, path, *, final=False):
        report = {**self._audit, "resources": self.guard.report(),
                  "hard_kill_yield_cursor_is_lower_bound": not final,
                  "hard_kill_read_cost_counters_are_lower_bounds": not final,
                  "maximum_uncheckpointed_yield_rows": 0 if final else 32}
        atomic_write_json(path / ("RESULT.json" if final else "progress.json"), report)

    def audit_report(self):
        return copy.deepcopy(self._audit)

    def _packet(self, folder, episode, number, spec, previous_after):
        import numpy as np
        self.guard.check("before_full_packet_read")
        path = producer.safe_child(folder, spec["name"])
        self._audit["packet_attempts"] += 1
        self._audit["last_packet_attempted"] = number
        producer.verify_file(path, spec, self.guard)
        self._audit["payload_bytes_SHA_verified"] += spec["bytes"]
        reviewed._headers(path, spec["rows"])
        with np.load(path, allow_pickle=False) as archive:
            phi = archive["phi"]
            self._audit["payload_rows_read"] += len(phi)
            rows = {key: archive[key] for key in producer.ROW_KEYS}
            provenance = json.loads(archive["provenance_utf8"].tobytes())
        expected = producer.provenance(self.bank, episode, number,
                                      SimpleNamespace(identity=reviewed.EXPECTED_ENCODER), probe=False)
        if provenance != expected:
            raise ValueError("Formal Phi packet source/teacher/Phi provenance differs")
        original = episode.source["shards"][number]
        producer.verify_file(producer.safe_child(Path(episode.source["index"]).parent, original["name"]), original, self.guard)
        self._audit["raw_source_compressed_bytes_SHA_verified"] += original["bytes"]
        original_rows = self.bank.label_rows(episode, number)
        self._audit["teacher_metadata_rows_verified"] += spec["rows"]
        label = episode.metadata["label_shards"][number]
        teacher_path = producer.safe_child(episode.folder, label["name"])
        _teacher19_headers(teacher_path, spec["rows"])
        if (not np.isfinite(phi).all() or np.any(np.abs(phi) > 1)
                or any(not np.array_equal(rows[key], original_rows[key]) for key in producer.ROW_KEYS)
                or previous_after is not None and int(rows["before_frames"][0]) != previous_after):
            raise ValueError("Phi finite/FP32/source row/clock/seed/reset/true-terminal contract differs")
        self._audit["payload_rows_verified"] += spec["rows"]
        self._audit["last_verified_source_row"] = spec["start"] + spec["rows"] - 1
        self._audit["last_verified_after_frame"] = int(rows["after_observation_frames"][-1])
        immutable = np.frombuffer(phi.tobytes(order="C"), dtype=np.float32).reshape(-1, 9721)
        return immutable, rows, label

    def iter_episode(self, episode, *, max_rows, audit_dir):
        """Fresh row0 stream only. No start-row, saved-cursor, or resume option."""
        if self._active:
            raise ValueError("One active audited episode stream per reader")
        self._audit = None
        folder, source, index, commit_sha = self._metadata(episode)
        if type(max_rows) is not int or not 1 <= max_rows <= source.source["rows"]:
            raise ValueError("Explicit max_rows must be within this original complete episode")
        audit = Path(audit_dir)
        _separate_output(audit, self.root, self.bank_root, self.bank.sources)
        if audit.exists():
            raise ValueError("Choose a new audit directory separate from immutable cache/bank/source")
        self.guard.destination = audit
        self.guard.check("before_new_consumer_audit")
        audit.mkdir(parents=True, exist_ok=False)
        self._audit = dict(schema=SCHEMA, status="started", episode=episode, source_rows=source.source["rows"],
            source_start_row=0, max_payload_and_yield_rows=max_rows, packet_attempts=0, last_packet_attempted=None,
            payload_rows_read=0, payload_rows_verified=0, teacher_metadata_rows_verified=0, rows_yielded=0,
            next_source_row_to_yield=0, last_yielded_source_row=None, last_verified_source_row=None,
            last_verified_after_frame=None, terminal_yielded=False, consumption_complete=False,
            payload_bytes_SHA_verified=0, raw_source_compressed_bytes_SHA_verified=0,
            formal_commit_sha256=commit_sha, cache_root=str(self.root.resolve()), bank_identity_sha256=self.bank.identity_sha,
            teacher_hidden_value_or_target_tensors_loaded=False, raw_maps_decompressed=False,
            brain_Torch_GPU_or_optimizer_started=False, numerical_training_resume=False,
            saved_cursor_is_input=False, synthetic_fixture=self.synthetic_fixture)
        self._active = True
        previous_after = None
        try:
            self._audit_write(audit)
            bank_root_text = str(self.bank.root.resolve())
            for number, spec in enumerate(index["cache_shards"]):
                if self._audit["payload_rows_read"] + spec["rows"] > max_rows:
                    self._audit.update(status="row_budget_before_next_packet", next_packet_rows=spec["rows"])
                    break
                self._audit_write(audit)  # latest persisted yielded cursor before a new packet
                phi, metadata, label = self._packet(folder, source, number, spec, previous_after)
                previous_after = int(metadata["after_observation_frames"][-1])
                self._audit_write(audit)
                teacher_path_text = str(producer.safe_child(source.folder, label["name"]).resolve())
                for local in range(len(phi)):
                    self.guard.check("before_full_episode_row_yield")
                    values = {key: bool(value[local]) if key in ("episode_starts", "dones") else int(value[local])
                              for key, value in metadata.items()}
                    row = reviewed.SourceRowMetadata(episode=episode, **values)
                    if row.source_row != self._audit["rows_yielded"]:
                        raise ValueError("Never skip to a late source row or reuse an audit cursor")
                    reference = reviewed.TeacherLabelReference(bank_root_text, self.bank.identity_sha,
                        episode, teacher_path_text, label["sha256"],
                        label["bytes"], label["start"], label["rows"], local, row.source_row)
                    self._audit["rows_yielded"] += 1
                    self._audit["next_source_row_to_yield"] = row.source_row + 1
                    self._audit["last_yielded_source_row"] = row.source_row
                    self._audit["terminal_yielded"] = row.dones
                    yield reviewed.CachedSensoryRow(phi[local], row, reference, spec["sha256"])
                self._audit_write(audit)
            if self._audit["terminal_yielded"]:
                self._audit["status"] = "complete_true_terminal_yielded"
        except GeneratorExit:
            self._audit["status"] = "consumer_closed"
            raise
        except (Exception, KeyboardInterrupt) as error:
            self._audit.update(status="stopped_or_failed", error_type=type(error).__name__, reason=str(error))
            raise
        finally:
            self._audit["consumption_complete"] = bool(self._audit["terminal_yielded"]
                and self._audit["rows_yielded"] == source.source["rows"])
            self._active = False
            self._audit_write(audit, final=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("inspect", "consume"), default="inspect")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--bank", type=Path, default=producer.DEFAULT_BANK)
    parser.add_argument("--episode")
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--max-seconds", type=float, default=120.)
    parser.add_argument("--audit-dir", type=Path)
    parser.add_argument("--receipt", type=Path, required=True)
    arguments = sys.argv[1:] if argv is None else argv
    if not arguments:
        parser.print_help()
        return 0
    args = parser.parse_args(arguments)
    if args.receipt.exists() or not 0 < args.max_seconds <= 86400:
        parser.error("Choose a new receipt and finite wall budget (0,86400]")
    try:
        _separate_output(args.receipt, args.cache_root, args.bank)
    except ValueError as error:
        parser.error(str(error))
    if args.mode == "consume" and (not args.episode or args.max_rows is None or not 1 <= args.max_rows <= 741839 or args.audit_dir is None):
        parser.error("Consume needs explicit episode, finite max_rows and a new audit_dir")
    if args.mode == "inspect" and (args.max_rows is not None or args.audit_dir is not None):
        parser.error("Inspect does not consume rows or create audit directories")
    guard = producer.CpuGuard(args.audit_dir or args.receipt, max_seconds=args.max_seconds,
                             available_ram=available_physical_memory_bytes)
    report = dict(schema=SCHEMA, status="not_started", mode=args.mode,
                  cache_bank_core_or_old_reader_modified=False, Torch_GPU_brain_training_or_game_started=False)
    code = 0
    reader = None
    receipt_safe = True
    try:
        report["runtime"] = producer.configure_CPU()
        guard.check("before_consumer_metadata")
        reader = FullEpisodeSensoryReader(args.cache_root, args.bank, guard)
        try:
            _separate_output(args.receipt, args.cache_root, args.bank, reader.bank.sources)
        except ValueError:
            receipt_safe = False
            raise
        report["inspection"] = reader.inspect(args.episode)
        if args.mode == "consume":
            digest = hashlib.sha256()
            with closing(reader.iter_episode(args.episode, max_rows=args.max_rows, audit_dir=args.audit_dir)) as stream:
                for row in stream:
                    digest.update(row.phi.tobytes())
            report.update(audit=reader.audit_report(), yielded_Phi_SHA256=digest.hexdigest())
        report["status"] = "completed_" + args.mode
    except (Exception, KeyboardInterrupt) as error:
        code = 2
        report.update(status="stopped_or_failed", error_type=type(error).__name__, reason=str(error))
        if reader is not None:
            report["audit"] = reader.audit_report()
    finally:
        report.update(resources=guard.report(), Torch_imported="torch" in sys.modules,
                      consumer_source_sha256=producer.file_sha(__file__))
        if receipt_safe:
            args.receipt.parent.mkdir(parents=True, exist_ok=True)
            with args.receipt.open("x", encoding="utf-8") as stream:
                json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
        print(json.dumps(dict(status=report["status"], receipt=str(args.receipt))))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
