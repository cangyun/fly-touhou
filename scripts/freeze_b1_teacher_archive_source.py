"""Add the new collector over a hash-pinned native runtime, never over live code."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from fly_touhou.atomic_json_io import atomic_write_json as write
from fly_touhou.thrl_study_io import utc_now

OVERLAYS = (
    'src/fly_touhou/thrl_teacher_archive.py',
    'src/fly_touhou/thrl_full_teacher_collection.py',
    'scripts/collect_b1_full_story_teacher.py',
    'scripts/qualify_b1_teacher_archive.py',
    'scripts/freeze_b1_teacher_archive_source.py',
    'tests/test_thrl_teacher_archive.py',
    'tests/test_thrl_full_teacher_collection.py',
    'tests/test_b1_full_teacher_launch_guard.py')


def freeze(args):
    manifest = args.baseline/'source_manifest.json'
    raw = manifest.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.baseline_sha256:
        raise ValueError('Hash-pinned baseline runtime differs')
    original = json.loads(raw)
    if args.output.exists():
        raise FileExistsError('Never overwrite a frozen runtime')
    files = {}
    for row in original['files']:
        relative = Path(row['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe baseline file path')
        data = (args.baseline/relative).read_bytes()
        if len(data) != row['bytes'] or hashlib.sha256(data).hexdigest() != row['sha256']:
            raise ValueError('Baseline file no longer matches its manifest: '+str(relative))
        files[relative.as_posix()] = data
    for relative in OVERLAYS:
        if relative in files:
            raise ValueError('This release must not replace an existing native/training file: '+relative)
        files[relative] = (ROOT/relative).read_bytes()
    for relative, data in files.items():
        if relative.endswith('.py'):
            compile(data, relative, 'exec')
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for relative, data in sorted(files.items()):
        target = args.output/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        records.append(dict(path=relative, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
    value = dict(schema='B1-full-natural-teacher-source-additions-v1', utc=utc_now(),
        origin=str(ROOT), base_runtime=str(args.baseline.resolve()), base_manifest_sha256=args.baseline_sha256,
        new_files=list(OVERLAYS), unchanged_baseline_files=len(original['files']),
        existing_native_input_reward_PPO_and_model_files_replaced=0,
        new_component_role='natural teacher trajectory collection and lossless storage only',
        no_native_job_started_by_freeze=True, frozen_runtime_tests_required_before_launch=True,
        files=records)
    write(args.output/'source_manifest.json', value)
    (args.output/'snapshot_manifest.json').write_bytes((args.output/'source_manifest.json').read_bytes())
    print(json.dumps(dict(runtime=str(args.output.resolve()), files=len(records),
        manifest_sha256=hashlib.sha256((args.output/'source_manifest.json').read_bytes()).hexdigest(),
        unchanged_baseline_files=len(original['files']), replaced_existing_files=0)))


if __name__ == '__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--baseline-sha256',required=True)
    p.add_argument('--output',type=Path,required=True)
    freeze(p.parse_args())
