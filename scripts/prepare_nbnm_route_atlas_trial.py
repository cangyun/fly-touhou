"""Prepare a bounded continuation with native route priors; never launch it."""
from pathlib import Path
import argparse
import copy
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('base-config','checkpoint','atlas','output','run-output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--additional-segments',type=int,default=2)
    p.add_argument('--rehearsal',type=Path)
    a=p.parse_args()
    if a.additional_segments<1:p.error('Use a positive segment count')
    import torch
    from fly_touhou.nbnm_run import read,validate_spec
    from fly_touhou.nbnm_resume import validate_resume
    from fly_touhou.nbnm_route_atlas import RouteAtlas
    from fly_touhou.thrl_durable_records import write_json
    atlas=RouteAtlas(a.atlas)
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    spec=copy.deepcopy(read(a.base_config))
    spec['nbnm']['route_prior']=dict(enabled=True,manifest=str(a.atlas.resolve()),phase_landmarks=True)
    migration=validate_resume(saved,spec)
    spec['resume_checkpoint']=str(a.checkpoint.resolve())
    spec['output']=str(a.run_output.resolve())
    spec['tensorboard_directory']=str(a.run_output.resolve()/'tensorboard')
    spec['decision_budget']=saved['completed_decisions']+a.additional_segments*spec['segment_decisions']
    spec['resume_extension_segments']=a.additional_segments
    spec['reference_checkpoint']=str(a.checkpoint.resolve())
    spec['evaluation_tasks']=['full_story']
    if a.rehearsal is not None:spec['rehearsal']['manifest']=str(a.rehearsal.resolve())
    validation=validate_spec(spec)
    a.output.mkdir(parents=True,exist_ok=True)
    write_json(a.output/'route.json',spec)
    report=dict(status='prepared_not_started',network_training=False,
        source_decisions=saved['completed_decisions'],source_updates=saved['optimizer_updates'],
        additional_decisions=a.additional_segments*spec['segment_decisions'],
        source_checkpoint=str(a.checkpoint),route_rows=atlas.manifest['rows'],
        covered_phases=len(atlas.manifest['phases']),migration=migration,validation=validation)
    write_json(a.output/'preparation.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
