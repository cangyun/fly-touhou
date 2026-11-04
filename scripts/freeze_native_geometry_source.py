"""Freeze V5 teacher changes over a verified immutable V4 teacher baseline."""
import argparse
import hashlib
import json
from pathlib import Path

OVERLAYS=(
    'src/fly_touhou/hazard_reader.py','src/fly_touhou/native_hazard_planner.py',
    'src/fly_touhou/native_geometry.py','src/fly_touhou/geometry_environment.py',
    'src/fly_touhou/resource_teacher_v5.py','scripts/collect_dynamic_teacher_curriculum.py',
    'scripts/audit_stage6_geometry.py','scripts/probe_native_collision_geometry.py',
    'scripts/benchmark_native_geometry.py','scripts/freeze_native_geometry_source.py')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    output=a.output.resolve()
    if output.exists():raise FileExistsError('Never replace a frozen source snapshot')
    manifest_path=a.baseline/'snapshot_manifest.json'
    manifest_bytes=manifest_path.read_bytes()
    baseline=json.loads(manifest_bytes)
    files={}
    for row in baseline['files']:
        relative=Path(row['path'])
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe baseline path')
        data=(a.baseline/relative).read_bytes()
        if hashlib.sha256(data).hexdigest()!=row['sha256'] or len(data)!=row['bytes']:
            raise ValueError('Baseline content differs: '+str(relative))
        files[relative]=data
    for relative in OVERLAYS:files[Path(relative)]=(root/relative).read_bytes()
    for relative,data in files.items():
        if relative.suffix=='.py':compile(data,str(relative),'exec')
    output.mkdir(parents=True,exist_ok=False)
    records=[]
    for relative,data in sorted(files.items()):
        target=output/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(data)
        records.append({'path':str(relative),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
    result={'schema':'fly-touhou-frozen-source-v1','origin':str(root),
        'baseline':str(a.baseline.resolve()),'baseline_manifest_sha256':hashlib.sha256(manifest_bytes).hexdigest(),
        'scope':'Only listed V5 teacher geometry overlays; no student/reward/training changes from current worktree included',
        'overlays':list(OVERLAYS),'files':records}
    target=output/'snapshot_manifest.json'
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'snapshot':str(output),'files':len(records),
        'manifest_sha256':hashlib.sha256(target.read_bytes()).hexdigest()}))


if __name__=='__main__':main()
