"""Preserve a complete v2 brain and Adam history while adding measured v3 inputs."""
from pathlib import Path
import argparse
import json
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

from fly_touhou.checkpoint_migration import migrate_resource_checkpoint
from fly_touhou.connectome import file_sha256
from fly_touhou.hazard_observation import HazardObservationConfig, HazardObservationEncoder
from fly_touhou.planner import PlannerConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--target-specification', type=Path)
    options = parser.parse_args()
    parent = options.parent.resolve(strict=True)
    destination = options.output.resolve()
    if destination.exists() or destination.with_suffix('.migration.json').exists():
        raise FileExistsError('Never overwrite a parent or previous migration')
    checkpoint = torch.load(parent, map_location='cpu', weights_only=True)
    specification = (json.loads(options.target_specification.read_text(encoding='utf-8-sig'))
        if options.target_specification else HazardObservationEncoder(
            HazardObservationConfig(**checkpoint['encoder']['observation']),
            PlannerConfig(**checkpoint['encoder']['planner'])).specification())
    migrated = migrate_resource_checkpoint(checkpoint, specification,
        parent_path=parent, parent_sha256=file_sha256(parent))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix+'.tmp')
    with temporary.open('xb') as stream:
        torch.save(migrated, stream)
    temporary.replace(destination)
    report = dict(migrated['migration'], output=str(destination),
        output_sha256=file_sha256(destination), migration_source_sha256=file_sha256(
            ROOT/'src/fly_touhou/checkpoint_migration.py'))
    destination.with_suffix('.migration.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
