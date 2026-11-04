"""CPU-only, lossless cache of the student's existing fixed V5 -> 9721 Phi.

Default mode only inspects immutable bank indices. Probe writes a small new
sample directory. Full export requires two explicit options and is not an
automatic continuation of inspect/probe. No brain, teacher, optimizer, native
process, trainable input encoder, or hidden-state cache is constructed here.
"""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import argparse
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import shutil
import time
from typing import Any, Callable, Mapping
import uuid
import zipfile
import zlib

import psutil

BANK_SCHEMA = "B1-rule-history-final19-streamed-labels-v1"
CACHE_KIND = "fly-v5-fixed-sensory9721-FP32-cache-v1"
PINNED_BANK_IDENTITY = "1c6006dc7fbbbe5465e7afcef5f43afc8ffebfc4aba8e7833f4ad2b55eebb4f5"
PINNED_STUDENT_SOURCE = "e97ab80cf9a31c96af24fe50d9d6150914214634aa46fb0d9ae8a765baa988c2"
TEACHER_SHA256 = "d95fe710068483fc81d8525d89bccf72e61d209c7e1bb442ea66d022b9c1d06c"
DEFAULT_BANK = Path("./outputs/distillation/B1-d95-final19-labels-v1")
DEFAULT_CACHE = Path("./outputs/distillation/B1-d95-v5-sensory9721-cache-v1")
MIN_RAM = 3 * 1024 ** 3
MIN_DISK = 4 * 1024 ** 3
RAW_SHARD_LIMIT = 20 * 1024 ** 2
ROW_KEYS = ("source_row", "before_frames", "after_observation_frames", "episode_starts", "dones",
            "native_seeds", "stage", "stage_frame", "native_misses", "native_bombs_used", "behaviour_actions")
SHAPES = {"features": (25,), "maps": (16, 92, 96), "objects": (96, 29)}


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def safe_child(folder, name):
    folder = Path(folder)
    if not isinstance(name, str) or Path(name).name != name or name in ("", ".", ".."):
        raise ValueError("Unsafe cache/source filename")
    child = folder / name
    if child.resolve().parent != folder.resolve():
        raise ValueError("Cache/source path escapes its declared directory")
    return child


def existing_ancestor(path):
    path = Path(path).resolve()
    while not path.exists():
        if path.parent == path:
            raise FileNotFoundError("No existing storage ancestor: " + str(path))
        path = path.parent
    return path


class CacheStop(RuntimeError):
    pass


class CpuGuard:
    def __init__(self, destination, *, max_seconds=120., available_ram=None, free_disk=None):
        self.destination = Path(destination)
        self.max_seconds = max_seconds
        self.available_ram = available_ram or (lambda: psutil.virtual_memory().available)
        self.free_disk = free_disk or (lambda: shutil.disk_usage(existing_ancestor(self.destination)).free)
        self.started = time.monotonic()
        self.minimum_ram = self.minimum_disk = None
        self.peak_RSS = 0

    def check(self, stage):
        ram, disk = int(self.available_ram()), int(self.free_disk())
        self.minimum_ram = ram if self.minimum_ram is None else min(ram, self.minimum_ram)
        self.minimum_disk = disk if self.minimum_disk is None else min(disk, self.minimum_disk)
        self.peak_RSS = max(self.peak_RSS, psutil.Process().memory_info().rss)
        if ram < MIN_RAM:
            raise CacheStop("available_RAM_below_3_GiB:" + stage)
        if disk < MIN_DISK:
            raise CacheStop("destination_free_disk_below_4_GiB:" + stage)
        if time.monotonic() - self.started >= self.max_seconds:
            raise CacheStop("finite_wall_budget_reached:" + stage)

    def report(self):
        return dict(minimum_available_RAM_bytes=self.minimum_ram, minimum_destination_free_bytes=self.minimum_disk,
                    peak_RSS_bytes=self.peak_RSS, max_seconds=self.max_seconds,
                    elapsed_seconds=time.monotonic() - self.started)


def configure_CPU():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "2"
    process = psutil.Process()
    if os.name == "nt":
        process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        if process.nice() != psutil.BELOW_NORMAL_PRIORITY_CLASS:
            raise CacheStop("belowNormal_priority_not_applied")
        priority = "belowNormal"
    else:
        process.nice(max(5, process.nice()))
        priority = "nice>=5"
    return dict(CUDA_VISIBLE_DEVICES=os.environ["CUDA_VISIBLE_DEVICES"], priority=priority, CPU_threads_limit=2)


def verify_file(path, spec, guard):
    guard.check("immutable_file_hash")
    if Path(path).stat().st_size != spec["bytes"] or file_sha(path) != spec["sha256"]:
        raise ValueError("Immutable bank/source bytes differ: " + str(path))


@dataclass(frozen=True)
class Episode:
    source: Mapping[str, Any]
    folder: Path
    metadata: Mapping[str, Any]
    commit_sha: str
    files: Mapping[str, Any]


class SensoryBank:
    def __init__(self, root, guard, *, expected_identity=PINNED_BANK_IDENTITY, require_real_cohort=True):
        self.root, self.guard = Path(root), guard
        guard.check("bank_identity")
        self.identity = read_json(self.root / "identity.json")
        self.identity_sha = json_sha(self.identity)
        self.identity_file_sha = file_sha(self.root / "identity.json")
        context = self.identity.get("context", {})
        self.sources = context.get("episodes", [])
        if (self.identity.get("schema") != BANK_SCHEMA or self.identity.get("save_final_logits") is not True
                or context.get("teacher_sha256") != TEACHER_SHA256
                or self.identity.get("teacher_was_behaviour_policy") is not False
                or expected_identity is not None and self.identity_sha != expected_identity
                or require_real_cohort and (len(self.sources) != 30 or sum(row["rows"] for row in self.sources) != 741839)):
            raise ValueError("Require the pinned committed D95 final19 off-policy label bank")
        self._episodes = {}

    def episode(self, name):
        if name in self._episodes:
            return self._episodes[name]
        source = next((row for row in self.sources if row["name"] == name), None)
        if source is None:
            raise ValueError("Unknown original source episode")
        folder = safe_child(self.root / "episodes", name)
        commit = read_json(folder / "COMMITTED.json")
        if (commit.get("schema") != BANK_SCHEMA or commit.get("identity_sha256") != self.identity_sha
                or commit.get("source_index_sha256") != source["index_sha256"] or commit.get("rows") != source["rows"]):
            raise ValueError("Episode is not an identity-matched complete committed label transaction")
        files = {row["name"]: row for row in commit["files"]}
        if set(files) | {"COMMITTED.json"} != {path.name for path in folder.iterdir()}:
            raise ValueError("Committed episode file inventory differs")
        verify_file(folder / "episode_index.json", files["episode_index.json"], self.guard)
        verify_file(folder / "source_index.json", files["source_index.json"], self.guard)
        self.guard.check("source_index_identity")
        if file_sha(source["index"]) != source["index_sha256"] or file_sha(folder / "source_index.json") != source["index_sha256"]:
            raise ValueError("Original episode index changed")
        metadata = read_json(folder / "episode_index.json")
        if (metadata.get("identity_sha256") != self.identity_sha or metadata.get("source_episode") != source
                or metadata.get("natural_behaviour_episode_complete") is not True
                or metadata.get("label_generation_complete") is not True
                or len(metadata.get("label_shards", [])) != len(source["shards"])):
            raise ValueError("Source/label complete episode provenance differs")
        offset = 0
        for number, (original, label) in enumerate(zip(source["shards"], metadata["label_shards"])):
            if (original["start"] != offset or label["start"] != offset or label["rows"] != original["rows"]
                    or label["source_shard"] != original or label["name"] != f"labels-{number:06d}.npz"
                    or label.get("final19_logits_saved") is not True or not 1 <= original["rows"] <= 32
                    or not 1 <= original["uncompressed_bytes"] <= RAW_SHARD_LIMIT
                    or files.get(label["name"]) != {key: label[key] for key in ("name", "bytes", "sha256")}):
                raise ValueError("Original shard/label mapping is not complete consecutive source order")
            offset += original["rows"]
        if offset != source["rows"]:
            raise ValueError("Source episode rows differ from shard inventory")
        result = Episode(source, folder, metadata, file_sha(folder / "COMMITTED.json"), files)
        self._episodes[name] = result
        return result

    def label_rows(self, episode, number):
        import numpy as np
        spec = episode.metadata["label_shards"][number]
        path = safe_child(episode.folder, spec["name"])
        verify_file(path, spec, self.guard)
        with np.load(path, allow_pickle=False) as archive:
            if not {"teacher_final_logits", "teacher_probabilities", "legal_action_mask"}.issubset(archive.files):
                raise ValueError("Final19 teacher labels are not present")
            # Target arrays, teacher hidden, values and learned embeddings are
            # deliberately not deserialized or copied into this cache.
            rows = {key: archive[key] for key in ROW_KEYS}
        n, start = spec["rows"], spec["start"]
        for key, value in rows.items():
            dtype = np.bool_ if key in ("episode_starts", "dones") else np.int64
            if value.shape != (n,) or value.dtype != dtype:
                raise ValueError("Source row metadata shape/dtype differs: " + key)
        indices = np.arange(start, start + n, dtype=np.int64)
        if (not np.array_equal(rows["source_row"], indices)
                or not np.array_equal(rows["episode_starts"], indices == 0)
                or not np.array_equal(rows["dones"], indices == episode.source["rows"] - 1)
                or np.any(rows["native_seeds"] != episode.source["native_seed"])
                or np.any(rows["after_observation_frames"] < rows["before_frames"])
                or not np.array_equal(rows["before_frames"][1:], rows["after_observation_frames"][:-1])):
            raise ValueError("Source row/reset/seed/clock continuity differs")
        return rows

    def shard(self, episode, number):
        import numpy as np
        self.guard.check("source_shard_load")
        spec = episode.source["shards"][number]
        rows = self.label_rows(episode, number)
        if number:
            previous = self.label_rows(episode, number - 1)
            if rows["before_frames"][0] != previous["after_observation_frames"][-1]:
                raise ValueError("Clock continuity differs across original source shards")
        source_path = safe_child(Path(episode.source["index"]).parent, spec["name"])
        verify_file(source_path, spec, self.guard)
        decoder = zlib.decompressobj()
        payload = decoder.decompress(source_path.read_bytes(), spec["uncompressed_bytes"] + 1)
        if (len(payload) != spec["uncompressed_bytes"] or not decoder.eof or decoder.unused_data
                or decoder.unconsumed_tail or hashlib.sha256(payload).hexdigest() != spec["uncompressed_sha256"]):
            raise ValueError("Original source lossless decompression identity differs")
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            if sum(item.file_size for item in archive.infolist()) > RAW_SHARD_LIMIT:
                raise ValueError("Source NPZ exceeds the existing bounded32-row raw size")
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            observations = {key: archive[key] for key in SHAPES}
            for key in ROW_KEYS:
                if key == "source_row":
                    continue
                original = "actions" if key == "behaviour_actions" else key
                if not np.array_equal(archive[original], rows[key]):
                    raise ValueError("Label and raw source metadata differ: " + key)
        for key, shape in SHAPES.items():
            value = observations[key]
            if (value.dtype != np.float32 or value.shape != (spec["rows"], *shape)
                    or not np.isfinite(value).all() or np.any(np.abs(value) > 1)):
                raise ValueError("Expected original finite normalized FP32 V5 modality: " + key)
        self.guard.check("source_shard_ready")
        return observations, rows

    def inspect(self):
        total = shards = compressed = uncompressed = 0
        episodes = []
        for source in self.sources:
            self.episode(source["name"])
            total += source["rows"]
            shards += len(source["shards"])
            compressed += sum(row["bytes"] for row in source["shards"])
            uncompressed += sum(row["uncompressed_bytes"] for row in source["shards"])
            episodes.append(dict(name=source["name"], rows=source["rows"], shards=len(source["shards"])))
        return dict(bank_identity_sha256=self.identity_sha, bank_identity_file_sha256=self.identity_file_sha,
                    episodes=episodes, episode_count=len(episodes), rows=total, source_shards=shards,
                    original_compressed_bytes=compressed, original_uncompressed_bytes=uncompressed,
                    Phi_FP32_bytes_without_metadata=total * 9721 * 4,
                    scope="complete committed indices verified; only consumed probe/export label and raw shards are individually hashed/read")


class FixedPhi:
    def __init__(self):
        import inspect
        import numpy as np
        import torch
        from .fly_distillation_student import V5SensoryAdapter
        torch.set_num_threads(2)
        module = Path(inspect.getfile(V5SensoryAdapter))
        if file_sha(module) != PINNED_STUDENT_SOURCE:
            raise ValueError("Student source differs from the fixed reviewed Phi implementation")
        self.adapter = V5SensoryAdapter().cpu().eval()
        if (self.adapter.output_dim != 9721 or self.adapter.map_pool_shape != (12, 12)
                or sum(parameter.numel() for parameter in self.adapter.parameters()) != 0):
            raise ValueError("Cache must contain only the unchanged fixed parameter-free Phi")
        self.identity = dict(cache_kind=CACHE_KIND, adapter=self.adapter.specification(), dtype="float32",
                             encoder_parameter_count=0, student_source_sha256=file_sha(module),
                             encoder_class_source_sha256=hashlib.sha256(inspect.getsource(V5SensoryAdapter).encode()).hexdigest(),
                             torch_version=str(torch.__version__), numpy_version=np.__version__,
                             torch_CPU_threads=torch.get_num_threads(), producer_source_sha256=file_sha(__file__),
                             trainable_input_encoder_cached=False, teacher_hidden_cached=False,
                             brain_or_action_head_cached=False, quantization=False)

    def encode(self, observations):
        import numpy as np
        import torch
        with torch.inference_mode():
            result = self.adapter(*(torch.from_numpy(observations[key]) for key in ("features", "maps", "objects")))
        phi = result.contiguous().numpy()
        if phi.dtype != np.float32 or phi.shape != (len(observations["features"]), 9721) or not np.isfinite(phi).all():
            raise ValueError("Fixed Phi output is not finite9721 FP32")
        return phi

    def verify_rowwise(self, observations, phi):
        import numpy as np
        for index in range(len(phi)):
            reference = self.encode({key: value[index:index + 1] for key, value in observations.items()})
            if not np.array_equal(phi[index].view(np.uint32), reference[0].view(np.uint32)):
                raise ValueError("Batched cached Phi differs bitwise from ordinary rowwise student Phi")


def coverage(observations, rows):
    import numpy as np
    from .thrl_observation import MAP_CHANNELS
    if MAP_CHANNELS[10] != "projectile_occupancy":
        raise ValueError("Unexpected V5 projectile occupancy channel; do not guess laser coverage")
    f, objects = observations["features"], observations["objects"]
    present = objects[:, :, 0] > .5
    laser = present & ((objects[:, :, 3] > .5) | (objects[:, :, 4] > .5))
    return dict(stages=sorted(int(value) for value in np.unique(rows["stage"])),
                stage6_rows=int((rows["stage"] == 6).sum()),
                circle_component_gt3_proxy_rows=int((f[:, 13] * 64 > 3).sum()),
                circle_component_gt8_rows=int((f[:, 13] * 64 > 8).sum()),
                rows_with_selected_laser=int(laser.any(axis=1).sum()), selected_laser_tokens=int(laser.sum()),
                rows_with_laser_map_values=int(np.any(observations["maps"][:, 10] > 0, axis=(1, 2)).sum()),
                laser_map_presence_definition="MAP_CHANNELS[10]=projectile_occupancy >0; empty closest-distance defaults are excluded",
                limits="current source-sample coverage only; no neural history, future motion, learning or gameplay claim")


def write_packet(path, phi, rows, provenance, guard):
    import numpy as np
    guard.check("cache_packet_write")
    if phi.dtype != np.float32 or phi.shape != (len(rows["source_row"]), 9721) or not np.isfinite(phi).all():
        raise ValueError("Cache packet must retain exact finite9721 FP32, without quantization")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(provenance, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, phi=phi, provenance_utf8=np.frombuffer(encoded, dtype=np.uint8), **rows)
        stream.flush()
        os.fsync(stream.fileno())
    guard.check("cache_packet_reload")
    with np.load(path, allow_pickle=False) as archive:
        if (set(archive.files) != {"phi", "provenance_utf8", *ROW_KEYS}
                or archive["phi"].dtype != np.float32
                or not np.array_equal(archive["phi"].view(np.uint32), phi.view(np.uint32))
                or any(not np.array_equal(archive[key], rows[key]) for key in ROW_KEYS)
                or json.loads(archive["provenance_utf8"].tobytes()) != provenance):
            raise ValueError("Lossless FP32 cache roundtrip changed payload or provenance")
    return dict(name=Path(path).name, rows=len(phi), start=int(rows["source_row"][0]),
                bytes=Path(path).stat().st_size, sha256=file_sha(path))


def provenance(bank, episode, number, encoder, *, probe):
    return dict(cache_kind=CACHE_KIND, probe_sample_only=probe, complete_cached_episode=False,
                bank_root=str(bank.root.resolve()), bank_identity_sha256=bank.identity_sha,
                bank_identity_file_sha256=bank.identity_file_sha,
                original_episode=episode.source["name"], original_episode_rows=episode.source["rows"],
                bank_episode_commit_sha256=episode.commit_sha,
                source_index=episode.source["index"], source_index_sha256=episode.source["index_sha256"],
                source_shard=dict(episode.source["shards"][number]),
                teacher_label_shard=dict(episode.metadata["label_shards"][number]),
                teacher_sha256=TEACHER_SHA256, encoder=encoder.identity,
                neuron_state_created=False, teacher_hidden_read_or_cached=False,
                source_row_order="original source episode and native clocks; never reset at stage/miss")


def probe(bank, encoder, selections, output, guard, *, max_shards=5):
    import numpy as np
    output = Path(output)
    if output.exists() or output.resolve() == DEFAULT_CACHE.resolve() or DEFAULT_CACHE.resolve() in output.resolve().parents:
        raise ValueError("Probe requires a new sample directory outside the future full cache root")
    if not 1 <= len(selections) <= max_shards <= 8 or len(set((row["episode"], row["shard"]) for row in selections)) != len(selections):
        raise ValueError("Probe needs bounded unique source shards")
    source_order = {row["name"]: index for index, row in enumerate(bank.sources)}
    selections = sorted(selections, key=lambda row: (source_order[row["episode"]], row["shard"]))
    guard.check("probe_directory_create")
    output.mkdir(parents=True, exist_ok=False)
    samples = []
    for selected in selections:
        episode = bank.episode(selected["episode"])
        number = selected["shard"]
        if type(number) is not int or not 0 <= number < len(episode.source["shards"]):
            raise ValueError("Invalid original source shard number")
        begin = time.perf_counter()
        observations, rows = bank.shard(episode, number)
        read_seconds = time.perf_counter() - begin
        guard.check("fixed_Phi_encode")
        begin = time.perf_counter()
        phi = encoder.encode(observations)
        encode_seconds = time.perf_counter() - begin
        encoder.verify_rowwise(observations, phi)
        begin = time.perf_counter()
        path = safe_child(output, f"{episode.source['name']}-shard{number:06d}.npz")
        packet = write_packet(path, phi, rows, provenance(bank, episode, number, encoder, probe=True), guard)
        write_seconds = time.perf_counter() - begin
        packet.update(episode=episode.source["name"], source_shard_number=number, reason=selected.get("reason", "opening_probe"),
                      raw_observation_FP32_bytes=sum(value.nbytes for value in observations.values()), Phi_FP32_bytes=phi.nbytes,
                      read_and_verify_seconds=read_seconds, encode_seconds=encode_seconds,
                      lossless_write_and_reload_seconds=write_seconds, coverage=coverage(observations, rows),
                      batch_vs_rowwise_bitwise_equal=True, cache_roundtrip_bitwise_equal=True)
        samples.append(packet)
    count = sum(row["rows"] for row in samples)
    size = sum(row["bytes"] for row in samples)
    timing = sum(row["read_and_verify_seconds"] + row["encode_seconds"] + row["lossless_write_and_reload_seconds"] for row in samples)
    total_rows = sum(row["rows"] for row in bank.sources)
    bytes_per_row = [row["bytes"] / row["rows"] for row in samples]
    rates = [row["rows"] / (row["read_and_verify_seconds"] + row["encode_seconds"] + row["lossless_write_and_reload_seconds"]) for row in samples]
    result = dict(probe_only=True, full_cache_root_created=False, output=str(output), rows=count, samples=samples,
                  encoded_shape=[count, 9721], dtype="float32", encoder_parameters=0,
                  sample_encoded_compressed_bytes=size, sample_bytes_per_row=size / count,
                  projected_full_compressed_bytes_from_sample=size / count * total_rows,
                  sample_envelope_full_compressed_bytes=[min(bytes_per_row) * total_rows, max(bytes_per_row) * total_rows],
                  planning_upper_bytes_with_50percent_sample_max_margin=max(bytes_per_row) * total_rows * 1.5,
                  measured_pipeline_rows_per_second=count / timing,
                  measured_shard_pipeline_rows_per_second_range=[min(rates), max(rates)],
                  projected_full_pipeline_seconds_at_sample_mean=total_rows / (count / timing),
                  sample_envelope_full_pipeline_seconds=[total_rows / max(rates), total_rows / min(rates)],
                  planning_seconds_with_2x_slowest_sample_margin=2 * total_rows / min(rates),
                  projection_limit="small heterogeneous sample only; not a measured full export size or ETA; rowwise QA time excluded from pipeline timing",
                  student_history_or_learning_or_gameplay_claim=False)
    with (output / "probe_index.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def export_all(bank, encoder, target, guard):
    """Explicit future finite export: one transaction per whole source episode.

    No implicit resume/overwrite/deletion. A failed attempt remains uncommitted
    and cannot be consumed as a complete episode. This path is not run by prep.
    """
    target = Path(target)
    guard.check("full_export_root_create")
    if target.exists():
        raise ValueError("Full export target exists; no implicit resume or overwrite")
    target.mkdir(parents=True, exist_ok=False)
    identity = dict(cache_kind=CACHE_KIND, bank_identity_sha256=bank.identity_sha,
                    bank_identity_file_sha256=bank.identity_file_sha, encoder=encoder.identity)
    with (target / "identity.json").open("x", encoding="utf-8") as stream:
        json.dump(identity, stream, ensure_ascii=False, indent=2)
    completed = []
    for source in bank.sources:
        episode = bank.episode(source["name"])
        staging = target / "attempts" / (source["name"] + "-" + uuid.uuid4().hex)
        guard.check("episode_attempt_create")
        staging.mkdir(parents=True)
        packets, previous_after = [], None
        for number, spec in enumerate(source["shards"]):
            observations, rows = bank.shard(episode, number)
            if previous_after is not None and rows["before_frames"][0] != previous_after:
                raise ValueError("Cache stream cannot skip/reorder source history")
            guard.check("fixed_Phi_encode")
            phi = encoder.encode(observations)
            packet = write_packet(staging / f"phi-{number:06d}.npz", phi, rows,
                                  provenance(bank, episode, number, encoder, probe=False), guard)
            packets.append(packet)
            previous_after = rows["after_observation_frames"][-1]
        if sum(row["rows"] for row in packets) != source["rows"]:
            raise ValueError("Complete cached episode lost source rows")
        index = dict(cache_kind=CACHE_KIND, complete_cached_episode=True, rows=source["rows"],
                     source_episode=source, bank_episode_commit_sha256=episode.commit_sha, cache_shards=packets)
        guard.check("episode_commit")
        with (staging / "episode_index.json").open("x", encoding="utf-8") as stream:
            json.dump(index, stream, ensure_ascii=False, indent=2)
        with (staging / "COMMITTED.json").open("x", encoding="utf-8") as stream:
            json.dump(dict(cache_kind=CACHE_KIND, identity_sha256=json_sha(identity),
                           episode_index_sha256=file_sha(staging / "episode_index.json"), rows=source["rows"]), stream, indent=2)
        final = safe_child(target / "episodes", source["name"])
        final.parent.mkdir(parents=True, exist_ok=True)
        if final.exists() or target.resolve() not in staging.resolve().parents or final.resolve().parent != (target / "episodes").resolve():
            raise ValueError("Unsafe or occupied complete-episode commit target")
        staging.rename(final)
        completed.append(dict(episode=source["name"], rows=source["rows"]))
    return dict(full_export_completed=True, target=str(target), complete_episodes=completed)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("inspect", "probe", "export"), default="inspect")
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--execute-full-export", action="store_true")
    parser.add_argument("--probe-output", type=Path)
    parser.add_argument("--selections", type=Path)
    parser.add_argument("--max-seconds", type=float, default=120.)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.receipt.exists():
        parser.error("Choose a new receipt path; existing evidence is preserved")
    if args.mode == "export" and not args.execute_full_export or args.execute_full_export and args.mode != "export":
        parser.error("Full export requires --mode export AND --execute-full-export explicitly")
    if args.mode == "probe" and args.probe_output is None:
        parser.error("Probe needs a new --probe-output outside the future full cache root")
    if not 0 < args.max_seconds <= 86400:
        parser.error("Need a finite wall budget in (0,86400] seconds")
    destination = args.probe_output if args.mode == "probe" else args.cache_root
    guard = CpuGuard(destination, max_seconds=args.max_seconds)
    report = dict(cache_kind=CACHE_KIND, mode=args.mode, status="not_started", full_cache_root_created=False,
                  original_bank_modified=False, student_core_modified=False, GPU_or_native_or_optimizer_started=False)
    full_cache_existed_before = args.cache_root.exists()
    exit_code = 0
    try:
        report["runtime"] = configure_CPU()
        guard.check("before_bank_or_encoder_load")
        bank = SensoryBank(args.bank, guard)
        encoder = FixedPhi()
        report["encoder"] = encoder.identity
        if args.mode == "inspect":
            report["inspection"] = bank.inspect()
        elif args.mode == "probe":
            selections = read_json(args.selections) if args.selections else [dict(episode="world000", shard=0), dict(episode="world000", shard=1)]
            report["probe"] = probe(bank, encoder, selections, args.probe_output, guard)
        else:
            report["export"] = export_all(bank, encoder, args.cache_root, guard)
        report["status"] = "completed_" + args.mode
    except (Exception, KeyboardInterrupt) as error:
        exit_code = 2
        report.update(status="stopped_or_failed", stop_type=type(error).__name__, stop_reason=str(error),
                      no_automatic_retry_or_resource_relaxation=True)
    finally:
        report["resources"] = guard.report()
        report["full_cache_root_created"] = args.mode == "export" and not full_cache_existed_before and args.cache_root.exists()
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        with args.receipt.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps(dict(status=report["status"], mode=args.mode, receipt=str(args.receipt)), ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
