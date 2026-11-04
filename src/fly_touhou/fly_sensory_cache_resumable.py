"""Explicit CPU-only, source-bound resumable fixed-Phi export preparation.

Inspection and recovery validation do not import Torch. Explicit export lazily
reuses the unchanged FixedPhi Torch CPU adapter. A published shard contains an
immutable packet and commit. Interrupted private attempts are retained, never
treated as committed history, overwritten, or retried within the same run.
"""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "2"

import argparse
import json
import math
from pathlib import Path
import re
import sys
from types import SimpleNamespace
import uuid

import psutil

from . import fly_sensory_cache as producer
from . import fly_sensory_cache_reader as reviewed

SCHEMA = "fly-v5-sensory9721-resumable-episodes-v1"
SHARD_SCHEMA = "fly-v5-sensory9721-resumable-shard-v1"
PINNED_READER_SOURCE = "83679a21595caa33acaee3897a3717b8305a80021f36f7036f965b20ae1f9583"
DEFAULT_ROOT = Path("./outputs/distillation/B1-d95-v5-sensory9721-resumable-cache-v1")


def _json_x(path, value):
    path = Path(path)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _inside(path, root):
    result, boundary = Path(path).resolve(), Path(root).resolve()
    if boundary not in result.parents:
        raise ValueError("Transaction path must remain inside its new cache root")
    return result


def _publish(source, destination, root):
    """Publish own new transaction without replacing any existing target."""
    source, destination = _inside(source, root), _inside(destination, root)
    if destination.exists():
        raise ValueError("Refuse to overwrite an existing/unknown/incomplete transaction: " + str(destination))
    if os.name != "nt" and source.is_file():
        os.link(source, destination)  # atomic no-replace; retain the private copy
    else:
        source.rename(destination)  # Windows rename rejects an existing target


def _marker(path, value, scratch, root):
    temporary = Path(scratch) / ("marker-" + uuid.uuid4().hex + ".json")
    _json_x(temporary, value)
    _publish(temporary, path, root)


def _inventory(folder, allowed):
    folder = Path(folder)
    if folder.is_symlink() or any(path.is_symlink() for path in folder.iterdir()):
        raise ValueError("Cache transactions must not contain symbolic links")
    if set(path.name for path in folder.iterdir()) - set(allowed):
        raise ValueError("Unknown cache inventory; nothing is overwritten: " + str(folder))


def _metadata_digest(rows):
    import hashlib
    digest = hashlib.sha256()
    for key in producer.ROW_KEYS:
        digest.update(rows[key].tobytes(order="C"))
    return digest.hexdigest()


def _clock(rows):
    return dict(first_before_frame=int(rows["before_frames"][0]),
                last_after_frame=int(rows["after_observation_frames"][-1]),
                native_seed=int(rows["native_seeds"][0]),
                first_episode_start=bool(rows["episode_starts"][0]), last_done=bool(rows["dones"][-1]))


class ResumableSensoryCache:
    """One writer, source0->terminal per episode, append-only published shards."""

    def __init__(self, bank, root, guard, *, synthetic_fixture=False):
        self.bank, self.root, self.guard = bank, Path(root), guard
        self.resume_root = self.root / "_resume"
        self.synthetic_fixture = synthetic_fixture
        self.last_run = None
        guard.check("resumable_source_identity")
        reviewed.verify_reviewed_sources()
        if producer.file_sha(reviewed.__file__) != PINNED_READER_SOURCE:
            raise ValueError("Reviewed reader source SHA differs")
        if not synthetic_fixture and (bank.identity_sha != producer.PINNED_BANK_IDENTITY
                                      or len(bank.sources) != 30 or sum(p["rows"] for p in bank.sources) != 741839):
            raise ValueError("Production export requires the pinned complete30 D95 source bank")
        target, source = self.root.resolve(), bank.root.resolve()
        protected = [source, producer.DEFAULT_CACHE.resolve()]
        protected.extend(Path(p["index"]).resolve().parent for p in bank.sources)
        if any(target == p or p in target.parents or target in p.parents for p in protected):
            raise ValueError("New resumable root must be separate from immutable bank/source/old cache")
        self.formal_identity = dict(cache_kind=producer.CACHE_KIND,
                                    bank_identity_sha256=bank.identity_sha,
                                    bank_identity_file_sha256=bank.identity_file_sha,
                                    encoder=reviewed.EXPECTED_ENCODER)
        self.formal_identity_sha = producer.json_sha(self.formal_identity)
        self.identity = dict(
            schema=SCHEMA, payload_kind=producer.CACHE_KIND, synthetic_fixture=synthetic_fixture,
            formal_identity_sha256=self.formal_identity_sha,
            bank_root=str(bank.root.resolve()), bank_identity_sha256=bank.identity_sha,
            bank_identity_file_sha256=bank.identity_file_sha,
            source_episode_order=[dict(name=p["name"], rows=p["rows"], native_seed=p["native_seed"],
                                      source_index_sha256=p["index_sha256"]) for p in bank.sources],
            encoder=reviewed.EXPECTED_ENCODER,
            writer_source_sha256=producer.file_sha(__file__), reviewed_reader_source_sha256=PINNED_READER_SOURCE,
            cache_contains_trainable_encoder_or_teacher_hidden_or_brain_state=False,
            state_history_claim=False)
        self.identity_sha = producer.json_sha(self.identity)

    def _check_root(self):
        self.guard.check("resumable_root_validation")
        _inventory(self.root, ("identity.json", "episodes", "_resume"))
        if (producer.read_json(self.root / "identity.json") != self.formal_identity
                or producer.read_json(self.resume_root / "identity.json") != self.identity):
            raise ValueError("Cache identity/source/encoder/writer conflict; choose a new root")
        _inventory(self.resume_root, ("identity.json", "episodes", "attempts", "leases", "COMPLETED.json"))
        for path in [self.root / "episodes", *(self.resume_root / name for name in ("episodes", "attempts", "leases"))]:
            if not path.is_dir() or path.is_symlink():
                raise ValueError("Incomplete or unknown root inventory; no implicit repair")

    def _provenance(self, episode, number):
        return producer.provenance(self.bank, episode, number,
                                   SimpleNamespace(identity=reviewed.EXPECTED_ENCODER), probe=False)

    def _read_shard(self, episode, number, folder, previous_after, *, verify_payloads=True):
        self.guard.check("validate_committed_shard")
        _inventory(folder, ("payload.npz", "COMMITTED.json"))
        if {p.name for p in folder.iterdir()} != {"payload.npz", "COMMITTED.json"}:
            raise ValueError("Incomplete canonical shard; preserved without overwrite")
        commit = producer.read_json(folder / "COMMITTED.json")
        source, label = episode.source["shards"][number], episode.metadata["label_shards"][number]
        expected = dict(schema=SHARD_SCHEMA, identity_sha256=self.identity_sha,
                        episode=episode.source["name"], shard_number=number, start=source["start"], rows=source["rows"],
                        source_shard=source, teacher_label_shard=label)
        if (any(commit.get(k) != v for k, v in expected.items())
                or set(commit) != set(expected) | {"payload", "Phi_bytes_sha256", "row_metadata_sha256", "provenance_sha256", "source_clock"}
                or set(commit.get("payload", {})) != {"name", "rows", "start", "bytes", "sha256"}
                or commit["payload"]["name"] != "payload.npz"
                or commit["payload"]["rows"] != source["rows"] or commit["payload"]["start"] != source["start"]):
            raise ValueError("Committed shard source identity/row range differs")
        payload = folder / "payload.npz"
        clock = commit["source_clock"]
        if (set(clock) != {"first_before_frame", "last_after_frame", "native_seed", "first_episode_start", "last_done"}
                or type(clock["first_before_frame"]) is not int or clock["first_before_frame"] < 0
                or type(clock["last_after_frame"]) is not int or clock["last_after_frame"] < clock["first_before_frame"]
                or clock["native_seed"] != episode.source["native_seed"]
                or clock["first_episode_start"] is not (source["start"] == 0)
                or clock["last_done"] is not (source["start"] + source["rows"] == episode.source["rows"])
                or previous_after is not None and clock["first_before_frame"] != previous_after):
            raise ValueError("Committed source clocks/seed/reset/terminal metadata differs")
        if payload.stat().st_size != commit["payload"]["bytes"]:
            raise ValueError("Committed payload size differs")
        if not verify_payloads:
            return commit, clock["last_after_frame"]
        import hashlib
        import numpy as np
        producer.verify_file(payload, commit["payload"], self.guard)
        reviewed._headers(payload, source["rows"])
        producer.verify_file(producer.safe_child(Path(episode.source["index"]).parent, source["name"]), source, self.guard)
        labels_metadata = self.bank.label_rows(episode, number)
        with np.load(payload, allow_pickle=False) as archive:
            phi = archive["phi"]
            rows = {key: archive[key] for key in producer.ROW_KEYS}
            provenance = json.loads(archive["provenance_utf8"].tobytes())
        if (not np.isfinite(phi).all() or np.any(np.abs(phi) > 1)
                or any(not np.array_equal(rows[key], labels_metadata[key]) for key in producer.ROW_KEYS)
                or provenance != self._provenance(episode, number)
                or producer.json_sha(provenance) != commit["provenance_sha256"]
                or hashlib.sha256(phi.tobytes(order="C")).hexdigest() != commit["Phi_bytes_sha256"]
                or _metadata_digest(rows) != commit["row_metadata_sha256"] or _clock(rows) != clock):
            raise ValueError("Committed payload Phi/provenance/source row metadata differs")
        if previous_after is not None and int(rows["before_frames"][0]) != previous_after:
            raise ValueError("Committed source clocks are not continuous across shards")
        return commit, int(rows["after_observation_frames"][-1])

    def _episode_summary(self, episode, commits):
        return dict(cache_kind=producer.CACHE_KIND, complete_cached_episode=True, rows=episode.source["rows"],
                    source_episode=episode.source,
                    bank_episode_commit_sha256=episode.commit_sha,
                    cache_shards=[dict(c["payload"], name=f"phi-{i:06d}.npz") for i, c in enumerate(commits)])

    def _finalize_episode(self, episode, commits, run):
        """Atomic old-format complete episode; hard links do not duplicate Phi."""
        folder = run / "scratch" / ("final-" + episode.source["name"])
        folder.mkdir(exist_ok=False)
        for i, commit in enumerate(commits):
            self.guard.check("before_final_episode_link")
            source = self.resume_root / "episodes" / episode.source["name"] / "shards" / f"{i:06d}" / "payload.npz"
            destination = folder / f"phi-{i:06d}.npz"
            os.link(_inside(source, self.root), _inside(destination, self.root))
        _json_x(folder / "episode_index.json", self._episode_summary(episode, commits))
        _json_x(folder / "COMMITTED.json", dict(cache_kind=producer.CACHE_KIND, identity_sha256=self.formal_identity_sha,
                 episode_index_sha256=producer.file_sha(folder / "episode_index.json"), rows=episode.source["rows"]))
        self.guard.check("before_atomic_complete_episode_publish")
        _publish(folder, self.root / "episodes" / episode.source["name"], self.root)

    def _check_final_episode(self, episode, commits, *, verify_payloads):
        folder = self.root / "episodes" / episode.source["name"]
        index = self._episode_summary(episode, commits)
        expected_names = {"episode_index.json", "COMMITTED.json", *(p["name"] for p in index["cache_shards"])}
        _inventory(folder, expected_names)
        if (set(p.name for p in folder.iterdir()) != expected_names
                or producer.read_json(folder / "episode_index.json") != index
                or producer.read_json(folder / "COMMITTED.json") != dict(
                    cache_kind=producer.CACHE_KIND, identity_sha256=self.formal_identity_sha,
                    episode_index_sha256=producer.file_sha(folder / "episode_index.json"), rows=episode.source["rows"])):
            raise ValueError("Complete old-format episode manifest differs")
        for spec in index["cache_shards"]:
            if (folder / spec["name"]).stat().st_size != spec["bytes"]:
                raise ValueError("Complete episode payload size differs")
            if verify_payloads:
                producer.verify_file(folder / spec["name"], spec, self.guard)

    def _completion(self):
        return dict(schema=SCHEMA, identity_sha256=self.identity_sha,
                    complete_episodes=[dict(episode=p["name"], rows=p["rows"],
                                           episode_commit_sha256=producer.file_sha(self.root / "episodes" / p["name"] / "COMMITTED.json"))
                                       for p in self.bank.sources],
                    rows=sum(p["rows"] for p in self.bank.sources))

    def inspect(self, *, verify_payloads=False):
        """Default metadata-only; explicit resume validates every payload SHA."""
        if not self.root.exists():
            return dict(schema=SCHEMA, root_exists=False, committed_rows=0, complete_episodes=[],
                        bank_rows=sum(p["rows"] for p in self.bank.sources), episodes=[], pending_attempts=0,
                        payload_SHA_and_Phi_validated=False, metadata_only=True)
        self._check_root()
        known = {p["name"] for p in self.bank.sources}
        if (set(p.name for p in (self.root / "episodes").iterdir()) - known
                or set(p.name for p in (self.resume_root / "episodes").iterdir()) - known):
            raise ValueError("Unknown cached episode; no files are overwritten")
        results, complete, total = [], [], 0
        earlier_incomplete = False
        for source in self.bank.sources:
            folder = self.resume_root / "episodes" / source["name"]
            if not folder.exists():
                if (self.root / "episodes" / source["name"]).exists():
                    raise ValueError("Complete output lacks its resumable source-bound shard history")
                earlier_incomplete = True
                continue
            if earlier_incomplete:
                raise ValueError("Cached episodes cannot skip the original source episode order")
            _inventory(folder, ("shards",))
            shards = folder / "shards"
            if not shards.is_dir():
                raise ValueError("Incomplete canonical episode; no implicit repair")
            names = sorted(p.name for p in shards.iterdir())
            if names != [f"{i:06d}" for i in range(len(names))] or len(names) > len(source["shards"]):
                raise ValueError("Unknown, missing, or nonconsecutive canonical shard")
            episode = self.bank.episode(source["name"])
            commits, previous_after = [], None
            for i in range(len(names)):
                commit, previous_after = self._read_shard(episode, i, shards / names[i], previous_after,
                                                          verify_payloads=verify_payloads)
                commits.append(commit)
            n = sum(c["rows"] for c in commits)
            all_shards = len(commits) == len(source["shards"])
            is_complete = (self.root / "episodes" / source["name"]).exists()
            if is_complete and (not all_shards or n != source["rows"]):
                raise ValueError("Complete episode marker differs from source0->terminal committed payloads")
            if is_complete:
                self._check_final_episode(episode, commits, verify_payloads=verify_payloads)
                complete.append(source["name"])
            else:
                earlier_incomplete = True
            results.append(dict(episode=source["name"], rows=n, commits=commits, previous_after=previous_after,
                                complete=is_complete, ready_to_finalize=all_shards and not is_complete))
            total += n
        if (self.resume_root / "COMPLETED.json").exists():
            if len(complete) != len(self.bank.sources) or producer.read_json(self.resume_root / "COMPLETED.json") != self._completion():
                raise ValueError("Whole-bank completion marker differs")
        return dict(schema=SCHEMA, root_exists=True, committed_rows=total, complete_episodes=complete,
                    episodes=results, pending_attempts=sum(not (p / "FINISHED").exists() for p in (self.resume_root / "attempts").iterdir()),
                    all_episodes_complete=len(complete) == len(self.bank.sources), metadata_only=not verify_payloads,
                    payload_SHA_and_Phi_validated=verify_payloads)

    def _start_run(self, *, resume, max_rows):
        leases = self.resume_root / "leases"
        names = sorted(p.name for p in leases.iterdir())
        if names != [f"{i:08d}.json" for i in range(len(names))]:
            raise ValueError("Unknown/incomplete writer lease; preserve it and choose a reviewed recovery")
        if names:
            owner = producer.read_json(leases / names[-1])
            if (set(owner) != {"schema", "identity_sha256", "run_id", "pid", "process_created"}
                    or owner["schema"] != SCHEMA or owner["identity_sha256"] != self.identity_sha
                    or not isinstance(owner["run_id"], str) or not re.fullmatch("[0-9a-f]{32}", owner["run_id"])
                    or type(owner["pid"]) is not int or owner["pid"] <= 0
                    or not isinstance(owner["process_created"], (int, float)) or not math.isfinite(owner["process_created"])):
                raise ValueError("Unknown writer lease identity")
            finished = (self.resume_root / "attempts" / owner["run_id"] / "FINISHED").exists()
            alive = False
            try:
                alive = abs(psutil.Process(owner["pid"]).create_time() - owner["process_created"]) < .01
            except psutil.Error:
                pass
            if not finished and (alive or not resume):
                raise ValueError("Another writer is active or explicit resume is missing")
        run_id = uuid.uuid4().hex
        folder = self.resume_root / "attempts" / run_id
        process = psutil.Process()
        # Exclusive creation serializes writers. A collision is an error, not a retry.
        _json_x(leases / f"{len(names):08d}.json", dict(schema=SCHEMA, identity_sha256=self.identity_sha,
                 run_id=run_id, pid=process.pid, process_created=process.create_time()))
        folder.mkdir(exist_ok=False)
        (folder / "scratch").mkdir()
        _json_x(folder / "STARTED.json", dict(schema=SCHEMA, identity_sha256=self.identity_sha, resume=resume,
                 max_new_rows=max_rows, wall_budget_seconds=self.guard.max_seconds,
                 same_fixed_Torch_CPU_encoder=True, CUDA_VISIBLE_DEVICES=os.environ["CUDA_VISIBLE_DEVICES"]))
        self.last_run = folder
        return folder

    def export(self, *, max_rows, resume=False, encoder_factory=None, after_commit=None):
        """Explicit finite invocation. max_rows counts NEW rows, whole source shards only.

        All previous published shards are validated before any new run writes.
        after_commit exists only for deterministic crash-contract injection.
        """
        import hashlib
        if type(max_rows) is not int or not 1 <= max_rows <= sum(p["rows"] for p in self.bank.sources):
            raise ValueError("Explicit finite max_rows must be within the complete source cohort")
        if not self.synthetic_fixture and (encoder_factory is not None or after_commit is not None):
            raise ValueError("Failure injection hooks are limited to explicit synthetic fixtures")
        self.guard.check("before_export_root")
        if self.root.exists():
            if not resume:
                raise ValueError("Existing root requires explicit resume; no overwrite")
            before = self.inspect(verify_payloads=True)
        else:
            if resume:
                raise ValueError("Resume requires an existing identity-matched root")
            self.root.mkdir(parents=True, exist_ok=False)
            _json_x(self.root / "identity.json", self.formal_identity)
            (self.root / "episodes").mkdir()
            self.resume_root.mkdir()
            _json_x(self.resume_root / "identity.json", self.identity)
            for name in ("episodes", "attempts", "leases"):
                (self.resume_root / name).mkdir()
            before = self.inspect(verify_payloads=True)
        self.guard.check("before_writer_lease")
        run = self._start_run(resume=resume, max_rows=max_rows)
        new_rows, published, encoder = 0, [], None
        report = dict(schema=SCHEMA, status="started", resume=resume,
                      committed_rows_before=before["committed_rows"], max_new_rows=max_rows,
                      run=str(run), new_rows=0, no_automatic_retry=True)
        try:
            states = {p["episode"]: p for p in before["episodes"]}
            stop_budget = False
            for source in self.bank.sources:
                state = states.get(source["name"], dict(commits=[], previous_after=None, complete=False))
                if state["complete"]:
                    continue
                episode = self.bank.episode(source["name"])
                folder = self.resume_root / "episodes" / source["name"]
                folder.mkdir(exist_ok=True)
                (folder / "shards").mkdir(exist_ok=True)
                commits, previous_after = list(state["commits"]), state["previous_after"]
                for number in range(len(commits), len(source["shards"])):
                    spec = source["shards"][number]
                    if new_rows + spec["rows"] > max_rows:
                        stop_budget = True
                        break
                    self.guard.check("before_new_source_shard")
                    if encoder is None:
                        encoder = (encoder_factory or producer.FixedPhi)()
                        if encoder.identity != reviewed.EXPECTED_ENCODER:
                            raise ValueError("Only unchanged parameter-free FixedPhi Torch CPU identity is permitted")
                    observations, rows = self.bank.shard(episode, number)
                    if previous_after is not None and int(rows["before_frames"][0]) != previous_after:
                        raise ValueError("Source clocks differ at resumable boundary")
                    phi = encoder.encode(observations)
                    scratch = run / "scratch" / f"{source['name']}-{number:06d}"
                    scratch.mkdir(exist_ok=False)
                    provenance = self._provenance(episode, number)
                    packet = producer.write_packet(scratch / "payload.npz", phi, rows, provenance, self.guard)
                    commit = dict(schema=SHARD_SCHEMA, identity_sha256=self.identity_sha,
                                  episode=source["name"], shard_number=number, start=spec["start"], rows=spec["rows"],
                                  source_shard=spec, teacher_label_shard=episode.metadata["label_shards"][number],
                                  payload=packet, Phi_bytes_sha256=hashlib.sha256(phi.tobytes(order="C")).hexdigest(),
                                  row_metadata_sha256=_metadata_digest(rows), provenance_sha256=producer.json_sha(provenance),
                                  source_clock=_clock(rows))
                    _json_x(scratch / "COMMITTED.json", commit)
                    # Recheck the exact persisted packet before making it visible.
                    checked, previous_after = self._read_shard(episode, number, scratch, previous_after)
                    self.guard.check("before_atomic_shard_publish")
                    _publish(scratch, folder / "shards" / f"{number:06d}", self.root)
                    commits.append(checked)
                    new_rows += spec["rows"]
                    published.append(dict(episode=source["name"], shard=number, start=spec["start"], rows=spec["rows"],
                                          payload_sha256=packet["sha256"]))
                    if after_commit is not None:
                        after_commit(published[-1])
                if len(commits) == len(source["shards"]):
                    self.guard.check("before_complete_episode_marker")
                    self._finalize_episode(episode, commits, run)
                else:
                    break
                if stop_budget:
                    break
            # No second full-bank scan: all old packets were checked before the
            # run and every new packet was checked before publishing.
            complete = all((self.root / "episodes" / p["name"] / "COMMITTED.json").exists() for p in self.bank.sources)
            if complete and not (self.resume_root / "COMPLETED.json").exists():
                self.guard.check("before_complete_bank_marker")
                _marker(self.resume_root / "COMPLETED.json", self._completion(), run / "scratch", self.root)
            report.update(status="complete" if complete else "new_row_budget_exhausted",
                          all_episodes_complete=complete)
        except (Exception, KeyboardInterrupt) as error:
            report.update(status="stopped_or_failed", error_type=type(error).__name__, reason=str(error))
            raise
        finally:
            report.update(new_rows=new_rows, published=published, resources=self.guard.report(),
                          torch_imported="torch" in sys.modules,
                          previous_committed_payloads_overwritten=False, uncommitted_attempts_preserved=True)
            _json_x(run / "RESULT.json", report)
            (run / "FINISHED").mkdir()
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("inspect", "export"), default="inspect")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--bank", type=Path, default=producer.DEFAULT_BANK)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--max-seconds", type=float, default=120.)
    parser.add_argument("--receipt", type=Path)
    arguments = sys.argv[1:] if argv is None else argv
    if not arguments:
        parser.print_help()
        return 0
    args = parser.parse_args(arguments)
    if args.receipt is None or args.receipt.exists():
        parser.error("Choose a new --receipt; existing evidence is preserved")
    if args.execute != (args.mode == "export") or args.resume and not args.execute:
        parser.error("Writing/resuming requires --mode export --execute explicitly")
    if ((args.mode == "export" and (args.max_rows is None or not 1 <= args.max_rows <= 741839))
            or args.mode == "inspect" and args.max_rows is not None or not 0 < args.max_seconds <= 86400):
        parser.error("Export needs explicit finite max_rows; wall budget must be in (0,86400] seconds")
    guard = producer.CpuGuard(args.cache_root, max_seconds=args.max_seconds)
    report = dict(schema=SCHEMA, mode=args.mode, status="not_started", cache_root=str(args.cache_root),
                  GPU_native_brain_optimizer_started=False, original_bank_core_reader_exporter_modified=False,
                  full_export_automatically_started=False)
    code = 0
    try:
        report["runtime"] = producer.configure_CPU()
        guard.check("before_bank_load")
        bank = producer.SensoryBank(args.bank, guard)
        store = ResumableSensoryCache(bank, args.cache_root, guard)
        report["resumable_identity_sha256"] = store.identity_sha
        report["formal_identity_sha256"] = store.formal_identity_sha
        report["result"] = store.inspect() if args.mode == "inspect" else store.export(max_rows=args.max_rows, resume=args.resume)
        report["status"] = "completed_" + args.mode
    except (Exception, KeyboardInterrupt) as error:
        code = 2
        report.update(status="stopped_or_failed", error_type=type(error).__name__, reason=str(error), no_automatic_retry=True)
    finally:
        report.update(resources=guard.report(), torch_imported="torch" in sys.modules,
                      writer_source_sha256=producer.file_sha(__file__))
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        _json_x(args.receipt, report)
        print(json.dumps(dict(status=report["status"], receipt=str(args.receipt))))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
