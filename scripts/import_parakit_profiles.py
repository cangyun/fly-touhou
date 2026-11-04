"""Convert pinned reference offsets; these are not executable-verified profiles."""
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
source = root / "third_party/parakit/offsets.py"
spec = importlib.util.spec_from_file_location("parakit_reference_offsets", source)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
revision = (source.parent / "UPSTREAM_REVISION.txt").read_text().strip()
out = root / "profiles"
out.mkdir(exist_ok=True)
for game in (14, 16):
    data = {
        "name": f"th{game}-full-reference-unverified",
        "game_id": game,
        "executable_names": [f"th{game}.exe"],
        "sha256": [],
        "offsets": asdict(module.offsets[f"th{game}.exe"]),
        "verification": "reference-only; trial/full executable identity must be pinned before control",
        "provenance": {
            "url": "https://github.com/Guy-L/parakit",
            "revision": revision,
            "source": "offsets.py",
            "note": "Offsets describe supported full versions; do not apply them to demos without verification.",
        },
    }
    path = out / f"th{game}_full_reference.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(path)
