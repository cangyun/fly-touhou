"""Freeze the actor-warmup implementation over the unchanged collector runtime."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.atomic_json_io import atomic_write_json as write
from fly_touhou.thrl_study_io import utc_now

NEW_FILES=(
    'src/fly_touhou/thrl_cached_actor_fit.py',
    'src/fly_touhou/thrl_full_teacher_data.py',
    'src/fly_touhou/thrl_teacher_latents.py',
    'scripts/qualify_b1_cached_actor_fit.py',
    'scripts/train_b1_full_teacher_actor.py',
    'scripts/freeze_b1_full_teacher_fit_source.py',
    'tests/test_thrl_cached_actor_fit.py',
    'tests/test_thrl_full_teacher_data.py',
    'tests/test_thrl_teacher_latents.py',
    'tests/test_b1_full_teacher_actor_fit.py')


def freeze(args):
    raw=(args.baseline/'source_manifest.json').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=args.baseline_sha256:raise ValueError('Collector baseline hash differs')
    baseline=json.loads(raw);files={}
    if args.output.exists():raise FileExistsError('Never overwrite a frozen runtime')
    for entry in baseline['files']:
        relative=Path(entry['path'])
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe source path')
        data=(args.baseline/relative).read_bytes()
        if len(data)!=entry['bytes'] or hashlib.sha256(data).hexdigest()!=entry['sha256']:
            raise ValueError('Collector baseline content differs: '+str(relative))
        files[relative.as_posix()]=data
    for name in NEW_FILES:
        if name in files:raise ValueError('Warmup must not replace existing native/PPO/model code: '+name)
        files[name]=(ROOT/name).read_bytes()
    for name,data in files.items():
        if name.endswith('.py'):compile(data,name,'exec')
    args.output.mkdir(parents=True)
    records=[]
    for name,data in sorted(files.items()):
        path=args.output/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        records.append(dict(path=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    manifest=dict(schema='B1-full-teacher-actor-fit-source-additions-v1',utc=utc_now(),origin=str(ROOT),
        baseline=str(args.baseline.resolve()),baseline_sha256=args.baseline_sha256,
        unchanged_baseline_files=len(baseline['files']),added_files=list(NEW_FILES),replaced_existing_files=0,
        role='temporary supervised actor warmup;full-network optimization and native efficacy remain required',
        files=records)
    write(args.output/'source_manifest.json',manifest)
    (args.output/'snapshot_manifest.json').write_bytes((args.output/'source_manifest.json').read_bytes())
    print(json.dumps(dict(runtime=str(args.output.resolve()),files=len(records),
        manifest_sha256=hashlib.sha256((args.output/'source_manifest.json').read_bytes()).hexdigest(),
        unchanged_baseline_files=len(baseline['files']),replaced_existing_files=0)))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--baseline-sha256',required=True)
    p.add_argument('--output',type=Path,required=True);freeze(p.parse_args())
