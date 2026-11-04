"""Fetch immutable, checksum-verified FlyWire v783 data and prepare a graph.

Run with this project's Python: python scripts/fetch_connectome.py
The source repository is MIT licensed; its LICENSE and reference model.py
are downloaded alongside the graph for provenance and comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fly_touhou.connectome import (  # noqa: E402
    SOURCE_FILES,
    SOURCE_RAW_BASE,
    SOURCE_REPOSITORY,
    SOURCE_REVISION,
    file_sha256,
    load_connectome,
)


def fetch_file(directory: Path, filename: str, *, force: bool = False) -> dict:
    expected = SOURCE_FILES[filename]
    destination = directory / filename
    if destination.exists() and not force:
        actual = file_sha256(destination)
        if actual != expected["sha256"]:
            raise ValueError(
                f"Existing {destination} failed SHA256 verification. "
                "Use --force only if you want to replace that local copy."
            )
        print(f"Verified existing {filename} ({destination.stat().st_size:,} bytes)", flush=True)
    else:
        url = f"{SOURCE_RAW_BASE}/{filename}"
        temporary = directory / (filename + ".part")
        for attempt in range(3):
            try:
                print(f"Downloading {filename}, attempt {attempt + 1}", flush=True)
                request = Request(url, headers={"User-Agent": "fly-touhou-research/0.1"})
                with urlopen(request, timeout=90) as response, temporary.open("wb") as stream:
                    while block := response.read(1024 * 1024):
                        stream.write(block)
                actual = file_sha256(temporary)
                if actual != expected["sha256"]:
                    raise ValueError(f"Downloaded {filename} checksum mismatch: {actual}")
                if temporary.stat().st_size != expected["size_bytes"]:
                    raise ValueError(f"Downloaded {filename} byte count mismatch")
                temporary.replace(destination)
                break
            except (URLError, HTTPError, TimeoutError, OSError):
                if attempt == 2:
                    raise
                time.sleep(1 + attempt)
        print(f"Verified {filename}: {expected['sha256']}", flush=True)
    return {
        "filename": filename,
        "url": f"{SOURCE_RAW_BASE}/{filename}",
        "size_bytes": destination.stat().st_size,
        "sha256": file_sha256(destination),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=PROJECT_ROOT / "data" / "connectome")
    parser.add_argument("--max-neurons", type=int, default=2048)
    parser.add_argument("--seed-index", type=int, default=None)
    parser.add_argument("--full", action="store_true", help="Prepare all 138,639 neurons in the upstream model files")
    parser.add_argument("--skip-prepare", action="store_true", help="Only fetch and verify source files")
    parser.add_argument("--force", action="store_true", help="Replace existing local download copies")
    parser.add_argument(
        "--manifest", type=Path,
        default=PROJECT_ROOT / "artifacts" / "connectome_manifest.json",
    )
    options = parser.parse_args()
    options.directory.mkdir(parents=True, exist_ok=True)
    files = [fetch_file(options.directory, filename, force=options.force) for filename in SOURCE_FILES]
    manifest = {
        "dataset": "FlyWire FAFB materialization v783",
        "repository": SOURCE_REPOSITORY,
        "revision": SOURCE_REVISION,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": files,
        "license": "MIT; upstream LICENSE preserved in download directory",
    }
    if not options.skip_prepare:
        start = time.perf_counter()
        graph = load_connectome(
            options.directory,
            max_neurons=None if options.full else options.max_neurons,
            seed_index=options.seed_index,
        )
        manifest["graph"] = graph.summary()
        manifest["preparation_seconds"] = time.perf_counter() - start
        print(
            f"Prepared {graph.n_neurons:,} real neurons and {graph.n_edges:,} "
            "directed connections; this is a topological selection, "
            "not an anatomically identified circuit.",
            flush=True,
        )
    options.manifest.parent.mkdir(parents=True, exist_ok=True)
    options.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Manifest: {options.manifest}", flush=True)


if __name__ == "__main__":
    main()
