"""Fetch the pinned graph and the two port-order inputs for the offline model."""
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from fetch_connectome import SOURCE_FILES, fetch_file
from setup_release import download
from fly_touhou.continuous_connectome import LEGACY_DN_MANIFEST_SHA256
from fly_touhou.connectome import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "data/connectome")
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    for filename in SOURCE_FILES:
        fetch_file(args.directory, filename)
    release = ROOT / "release-data"
    annotations = args.directory / "annotations"
    annotations.mkdir(exist_ok=True)
    sources = json.loads((release / "annotation-sources.json").read_text(encoding="utf-8"))
    for item in sources["files"]:
        target = annotations / item["name"]
        if not target.exists() or file_sha256(target) != item["sha256"]:
            download(item["url"], target, item["sha256"])
    name = "selected_ports_1024_v783.json"
    shipped = release / name
    if file_sha256(shipped) != LEGACY_DN_MANIFEST_SHA256:
        raise ValueError("The DN port-order file differs from the pinned version")
    target = annotations / name
    if not target.exists() or file_sha256(target) != LEGACY_DN_MANIFEST_SHA256:
        shutil.copyfile(shipped, target)
    print(f"Graph and port inputs ready in {args.directory}")


if __name__ == "__main__":
    main()
