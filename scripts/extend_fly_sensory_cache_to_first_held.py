"""Explicit finite extension through world010 using the unchanged cache writer.

Imports CPU-only modules. No teacher/student instantiation, CUDA or game launch.
Default inspects metadata; --execute-resume is required to write new packets.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "2"

from fly_touhou.fly_sensory_cache import CpuGuard, SensoryBank, configure_CPU
from fly_touhou.fly_sensory_cache_resumable import ResumableSensoryCache
from fly_touhou.thrl_durable_records import write_json
from fly_touhou.windows_physical_memory import available_physical_memory_bytes
import psutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-resume", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new receipt directory")
    args.output.mkdir(parents=True)
    cache = Path("./outputs/distillation/B1-d95-v5-sensory9721-resumable-cache-v1")
    configure_CPU()
    guard = CpuGuard(cache, max_seconds=3600, available_ram=available_physical_memory_bytes)
    bank = SensoryBank("./outputs/distillation/B1-d95-final19-labels-v1", guard)
    store = ResumableSensoryCache(bank, cache, guard)
    before = store.inspect()
    if before["committed_rows"] != 24755:
        raise ValueError("This one-off plan expects exactly the completed world000; inspect a changed root first")
    target_rows = sum(e["rows"] for e in bank.sources[:11])
    report = dict(schema="fly-sensory-cache-first-train-held-expansion-v1", utc=datetime.now(timezone.utc).isoformat(),
                  actual_process_identity=dict(pid=os.getpid(), create_time=psutil.Process().create_time()),
                  cache=str(cache), script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  budget_new_rows=target_rows-24755, target_total_formal_rows=target_rows,
                  final_source_episode="world010", split_seed=31, first_held_episode="world010",
                  CUDA_native_brain_optimizer_started=False, original_writer_modified=False,
                  source_order_unchanged=True, automatic_retry=False, new_rows=0, new_shards=0)
    if not args.execute_resume:
        report.update(status="CPU_metadata_inspected_only", before=before, Torch_imported="torch" in sys.modules)
        write_json(args.output / "result.json", report)
        return 0
    write_json(args.output / "progress.json", dict(report, status="validating_old_complete_cache"))

    try:
        # Production writer forbids fixture hooks. Its own per-shard COMMITTED
        # records remain the durable live progress; do not weaken that boundary.
        result = store.export(max_rows=report["budget_new_rows"], resume=True)
        report.update(status="finished_finite_extension", result=result, resources=guard.report(),
                      new_rows=result["new_rows"], new_shards=len(result["published"]), Torch_imported="torch" in sys.modules)
        write_json(args.output / "result.json", report)
        print(json.dumps({k: report[k] for k in ("status", "new_rows", "new_shards")}), flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as error:
        report.update(status="stopped_or_failed", reason=str(error), resources=guard.report(),
                      last_run=str(store.last_run), committed_prefix_preserved=True)
        write_json(args.output / "failure.json", report)
        print(json.dumps({"status": report["status"], "reason": str(error)}), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
