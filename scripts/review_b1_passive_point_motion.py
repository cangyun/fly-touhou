"""Read bounded passive telemetry; never load models or access original games.

Counts describe a selected, throttled event sample, not all native bullets.
The files are read with Windows replace/delete sharing enabled. Each file's
complete-line prefix is hashed separately; the cohort is not an atomic cut.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fly_touhou.thrl_dodge_diagnostics import _shared_binary
from fly_touhou.thrl_durable_records import write_json
from fly_touhou.thrl_study_io import utc_now


def review(source: Path) -> dict:
    import numpy as np
    import psutil

    if os.name == "nt":
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    sources, examples, kinematic_examples, kinematics = [], [], [], []
    totals, reasons = Counter(), Counter()
    per_stage = defaultdict(Counter)
    source_pids, worlds = set(), set()
    for path in sorted(source.glob("worker*.jsonl")):
        if path.stat().st_size > 8 * 1024**2:
            raise ValueError("Telemetry exceeds its declared eight-MiB bound")
        with _shared_binary(path) as stream:
            raw = stream.read(8 * 1024**2 + 1)
        if len(raw) > 8 * 1024**2:
            raise ValueError("Telemetry grew beyond its declared bound")
        complete = raw[:raw.rfind(b"\n") + 1]
        seen = set()
        count = 0
        for line in complete.splitlines():
            record = json.loads(line)
            if record.get("record_kind") != "passive_point_motion":
                raise ValueError("Unexpected telemetry record kind")
            if any(record.get(key) is not False for key in
                   ("future_motion_guaranteed", "native_collision_truth_claimed",
                    "policy_input_changed", "actions_changed")):
                raise ValueError("Require the declared passive-only record contract")
            if len(record["objects"]) > 8:
                raise ValueError("Telemetry contains more than eight sampled points")
            count += 1
            totals["records"] += 1
            totals["forced_context_records"] += bool(record["forced_context_record"])
            source_pids.add(record["native_pid"])
            worlds.add(record["world_id"])
            per_stage[str(record["stage"])]["records"] += 1
            for event in record["objects"]:
                key = (record["world_id"], event["native_id"], event["measured_frame"])
                if key in seen:
                    totals["duplicate_sample_references"] += 1
                    continue
                seen.add(key)
                totals["unique_sampled_point_observations"] += 1
                lag = record["frame"] - event["measured_frame"]
                if lag < 0:
                    raise ValueError("Event cannot have been measured after its record")
                totals["same_frame_sampled_observations"] += lag == 0
                if not event["known"]:
                    reasons[event["unknown_reason"]] += 1
                    totals["unknown_sampled_observations"] += 1
                    continue
                if event["elapsed_native_frames"] != 4:
                    raise ValueError("Known derivative did not use four original frames")
                totals["known_four_tick_sampled_observations"] += 1
                radius_rate = event["dr_dt"]
                if radius_rate is not None and not math.isfinite(radius_rate):
                    raise ValueError("Nonfinite measured radius derivative")
                active = event["native_state"] == 1
                circle = event["shape"] == "circle"
                growing = circle and radius_rate is not None and radius_rate > 1e-4
                totals["known_active_circle_samples"] += active and circle
                totals["known_active_growing_circle_samples"] += active and growing
                totals["known_active_accelerating_samples"] += active and any(
                    abs(value) > 1e-4 for value in (event["dv_dt"] or ()))
                before = event["prior_same_address_sample"]
                if active and event["native_flags"] & 2 and before is not None:
                    if event["measured_frame"] - before["frame"] != 4:
                        raise ValueError("Stored prior reference is not the known four-frame pair")
                    observed = event["observed_center_velocity"]
                    previous, current = before["stored_velocity"], event["stored_velocity"]
                    prior_error = 4 * math.hypot(*(x - y for x, y in zip(observed, previous)))
                    current_error = 4 * math.hypot(*(x - y for x, y in zip(observed, current)))
                    velocity_rate = math.hypot(*event["dv_dt"])
                    if not all(math.isfinite(value) for value in (prior_error, current_error, velocity_rate)):
                        raise ValueError("Nonfinite past-interval kinematic diagnostic")
                    kinematics.append((prior_error, current_error, velocity_rate))
                    kinematic_examples.append(dict(worker=path.name, world_id=record["world_id"],
                        measured_frame=event["measured_frame"], record_stage=record["stage"],
                        stage_at_measurement_verified_by_same_frame=lag == 0,
                        record_lag_native_frames=lag, native_kind=event["native_kind"],
                        past4_prior_velocity_center_error=prior_error,
                        past4_current_velocity_center_error=current_error,
                        previous_stored_velocity=previous, current_stored_velocity=current,
                        observed_past4_center_velocity=observed))
                    kinematic_examples = sorted(kinematic_examples,
                        key=lambda row: row["past4_prior_velocity_center_error"], reverse=True)[:4]
                if lag == 0:
                    per_stage[str(record["stage"])]["same_frame_known_samples"] += 1
                    per_stage[str(record["stage"])]["same_frame_active_growing_samples"] += active and growing
                if active and growing:
                    examples.append(dict(worker=path.name, world_id=record["world_id"],
                        native_episode=record["native_episode"], native_pid=record["native_pid"],
                        record_frame=record["frame"], record_stage=record["stage"],
                        stage_at_measurement_verified_by_same_frame=lag == 0,
                        record_lag_native_frames=lag, **event))
                    examples = sorted(examples, key=lambda row: row["dr_dt"], reverse=True)[:8]
        sources.append(dict(path=str(path), complete_line_prefix_bytes=len(complete),
            prefix_sha256=hashlib.sha256(complete).hexdigest(), records=count,
            omitted_incomplete_tail_bytes=len(raw) - len(complete)))
    if not sources:
        raise FileNotFoundError("No worker-local point telemetry files")
    past = dict(samples=len(kinematics),
        definition="For a verified PAST four-frame pair: ||4*observed_center_velocity-4*stored_velocity_at_endpoint||;not a forward prediction error.",
        selected_sample_is_not_representative=True,
        future_motion_or_native_collision_prediction_validated=False)
    if kinematics:
        values = np.asarray(kinematics, dtype=np.float64)
        quantiles = [0, .5, .9, .95, .99, 1]
        past.update(quantiles=quantiles,
            prior_velocity_center_error=np.quantile(values[:, 0], quantiles).tolist(),
            current_velocity_center_error=np.quantile(values[:, 1], quantiles).tolist(),
            stored_velocity_rate_norm=np.quantile(values[:, 2], quantiles).tolist(),
            prior_velocity_error_above2=int((values[:, 0] > 2).sum()),
            prior_velocity_error_above4=int((values[:, 0] > 4).sum()),
            examples=kinematic_examples)
    return dict(schema="B1-passive-point-motion-readonly-review-v2", utc=utc_now(),
        source=str(source), independently_read_file_prefixes=sources,
        producer_is_live_and_cohort_snapshot_is_not_atomic=True,
        source_native_pids=sorted(source_pids), world_generations=len(worlds),
        counts=dict(totals), same_frame_stage_counts={k: dict(v) for k, v in per_stage.items()},
        past_interval_kinematic_diagnostic=past,
        unknown_reasons=dict(reasons), strongest_sampled_active_growth_examples=examples,
        interpretation=[
            "Selected at-most-eight changed objects per throttled record;not a census or frequency estimate of all bullets.",
            "Object events can precede their containing record;only same-frame examples establish the recorded stage.",
            "Known finite differences describe past four-frame observations;they do not establish future linear motion.",
            "Past displacement versus endpoint stored velocity quantifies a selected history mismatch;it is not a calibrated future-acceleration model.",
            "Native addresses plus continuity are practical identity assumptions;undetectable address reuse is not excluded.",
            "No player trajectory or exhaustive collision objects are saved here;these events cannot establish why a particular native death occurred.",
            "Current policy inputs,rewards,teacher logic and frozen training configuration were not changed."],
        source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        new_models_loaded=0, native_worlds_started=0, optimizer_updates=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve existing evidence;choose a new output")
    report = review(args.source)
    write_json(args.output, report)
    print(json.dumps(dict(output=str(args.output), counts=report["counts"],
        stages=report["same_frame_stage_counts"], growth_examples=len(report["strongest_sampled_active_growth_examples"]))))


if __name__ == "__main__":
    main()
