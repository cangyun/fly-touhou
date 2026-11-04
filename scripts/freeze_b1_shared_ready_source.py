"""Freeze the optional copy reduction without touching running source."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.atomic_json_io import atomic_write_json as write
from fly_touhou.thrl_study_io import utc_now

CHANGED=('src/fly_touhou/thrl_vector.py','src/fly_touhou/thrl_ready_collection.py',
    'scripts/train_thrl_native.py')
ADDED=('tests/test_thrl_shared_ready_observation.py','scripts/benchmark_b1_shared_observation_copy.py',
    'scripts/freeze_b1_shared_ready_source.py','tests/test_thrl_vector.py',
    'tests/test_thrl_ready_batching.py','tests/test_thrl_ready_collection.py','tests/test_thrl_vector_wait.py')

def run(a):
    raw=(a.baseline/'source_manifest.json').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=a.baseline_sha256:raise ValueError('Qualified baseline changed')
    baseline=json.loads(raw);files={}
    if a.output.exists():raise FileExistsError('Do not overwrite frozen source')
    for entry in baseline['files']:
        relative=Path(entry['path'])
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe source path')
        data=(a.baseline/relative).read_bytes()
        if len(data)!=entry['bytes'] or hashlib.sha256(data).hexdigest()!=entry['sha256']:
            raise ValueError('Qualified baseline file changed:'+str(relative))
        files[relative.as_posix()]=data
    for name in CHANGED:
        if name not in files:raise ValueError('Expected baseline source missing')
        files[name]=(ROOT/name).read_bytes()
    for name in ADDED:
        if name in files:raise ValueError('Expected a new source file')
        files[name]=(ROOT/name).read_bytes()
    for name,data in files.items():
        if name.endswith('.py'):compile(data,name,'exec')
    a.output.mkdir(parents=True,exist_ok=False);records=[]
    for name,data in sorted(files.items()):
        target=a.output/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        records.append(dict(path=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    manifest=dict(schema='B1-optional-acknowledged-shared-observation-source-v1',utc=utc_now(),
        baseline=str(a.baseline),baseline_sha256=a.baseline_sha256,origin=str(ROOT),
        changed_files=list(CHANGED),added_files=list(ADDED),unchanged_baseline_files=len(baseline['files'])-len(CHANGED),
        default_enabled=False,inputs_model_rewards_PPO_equations_unchanged=True,
        native_qualification_required_before_use=True,files=records)
    write(a.output/'source_manifest.json',manifest)
    (a.output/'snapshot_manifest.json').write_bytes((a.output/'source_manifest.json').read_bytes())
    print(json.dumps(dict(runtime=str(a.output),files=len(records),changed=len(CHANGED),
        manifest_sha256=hashlib.sha256((a.output/'source_manifest.json').read_bytes()).hexdigest())))

if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--baseline-sha256',required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
