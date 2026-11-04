"""Apply only the bounded JSON writer repair to an unchanged learning entry.

The weights, optimizer, reward, rollout and learning functions remain those of
the hash-checked entry. A separate receipt records the explicit IO overlay.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fly_touhou.atomic_json_io import atomic_write_json
import fly_touhou.atomic_json_io as atomic_module

ENTRIES = {
    "train_standard_gru.py": ("e8e90087a7a2583beb511d6c2c83d847b2535b6c492f908f74344f31d4af48ad", "write_json"),
    "train_gru_bc.py": ("26f01bf68cda1c2ece421cd6a44310a01588919e400dfbc74b1d7b9d93b3657c", "atomic_json"),
    "collect_gru_dagger.py": ("9cf21974f6a5971a5866f2a2e3571ac03ea58d5fe0432b3c07bb479cce45f105", "write_json"),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entry", choices=tuple(ENTRIES), required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    options, forwarded = parser.parse_known_args()
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    entry = ROOT / "scripts" / options.entry
    expected_sha, writer_name = ENTRIES[options.entry]
    if digest(entry) != expected_sha:
        raise ValueError("The learning entry differs from the reviewed unchanged source")
    receipt = options.receipt.resolve()
    receipt.parent.mkdir(parents=True, exist_ok=True)
    record = {"schema": "unchanged-learning-entry-JSON-IO-overlay-v1", "status": "starting",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "wrapper": str(Path(__file__).resolve()), "wrapper_sha256": digest(__file__),
        "entry": str(entry), "entry_sha256": expected_sha, "forwarded_arguments": forwarded,
        "patched_function": writer_name, "IO_helper": str(Path(atomic_module.__file__).resolve()),
        "IO_helper_sha256": digest(atomic_module.__file__),
        "replace_timeout_seconds": 2.0, "retry_interval_seconds": 0.01,
        "learning_functions_or_hyperparameters_modified": False, "auto_restart": False}
    # Preserve exclusive creation before allowing the entry to create outputs.
    with receipt.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
    try:
        specification = importlib.util.spec_from_file_location("reviewed_learning_entry", entry)
        module = importlib.util.module_from_spec(specification)
        sys.modules[specification.name] = module
        specification.loader.exec_module(module)
        if not callable(getattr(module, writer_name, None)):
            raise ValueError("The reviewed JSON writer is unavailable")
        setattr(module, writer_name, atomic_write_json)
        sys.argv = [str(entry), *forwarded]
        module.main()
        record["status"] = "finished"
    except SystemExit as error:
        record.update(status="finished" if error.code in (None, 0) else "failed", exit_code=error.code)
        raise
    except BaseException as error:
        record.update(status="failed", exception=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_write_json(receipt, record)


if __name__ == "__main__":
    main()
