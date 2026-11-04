"""Freeze the explicit actor-to-full-policy transfer over the qualified fitter."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.atomic_json_io import atomic_write_json as write
from fly_touhou.thrl_study_io import utc_now

NEW_FILES=('src/fly_touhou/thrl_actor_transfer.py','scripts/train_b1_after_full_teacher.py',
    'scripts/freeze_b1_full_teacher_PPO_source.py','tests/test_thrl_actor_transfer.py',
    'tests/test_b1_after_full_teacher.py')


def freeze(args):
    raw=(args.baseline/'source_manifest.json').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=args.baseline_sha256:raise ValueError('Qualified fitter manifest differs')
    baseline=json.loads(raw);files={}
    if args.output.exists():raise FileExistsError('Never overwrite a frozen runtime')
    for entry in baseline['files']:
        relative=Path(entry['path'])
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe relative source path')
        data=(args.baseline/relative).read_bytes()
        if len(data)!=entry['bytes'] or hashlib.sha256(data).hexdigest()!=entry['sha256']:
            raise ValueError('Qualified fitter source changed: '+str(relative))
        files[relative.as_posix()]=data
    for name in NEW_FILES:
        if name in files:raise ValueError('This transfer must not replace qualified native,model,reward or PPO code')
        files[name]=(ROOT/name).read_bytes()
    for name,data in files.items():
        if name.endswith('.py'):compile(data,name,'exec')
    args.output.mkdir(parents=True);records=[]
    for name,data in sorted(files.items()):
        target=args.output/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        records.append(dict(path=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    manifest=dict(schema='B1-full-teacher-to-full-policy-PPO-source-additions-v1',utc=utc_now(),origin=str(ROOT),
        baseline=str(args.baseline.resolve()),baseline_sha256=args.baseline_sha256,
        unchanged_baseline_files=len(baseline['files']),added_files=list(NEW_FILES),replaced_existing_files=0,
        role='explicit BC-to-PPO transfer;fresh PPO Adam;full base trainable;same native/model/reward/PPO',files=records)
    write(args.output/'source_manifest.json',manifest)
    (args.output/'snapshot_manifest.json').write_bytes((args.output/'source_manifest.json').read_bytes())
    print(json.dumps(dict(runtime=str(args.output.resolve()),files=len(records),unchanged_baseline_files=len(baseline['files']),
        manifest_sha256=hashlib.sha256((args.output/'source_manifest.json').read_bytes()).hexdigest())))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--baseline-sha256',required=True);p.add_argument('--output',type=Path,required=True);freeze(p.parse_args())
