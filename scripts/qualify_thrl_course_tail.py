"""Qualify the completed-span preference without repeating unchanged PPO."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ('course','prior','tests','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    from fly_touhou.thrl_study_io import read_shared_json
    from fly_touhou.atomic_json_io import atomic_write_json as write
    from fly_touhou.thrl_course_env import THRLCourseEnv
    from fly_touhou.thrl_course_store import CourseStore
    from fly_touhou.thrl_teacher import make_teacher,native36_to_source19
    from fly_touhou.thrl_curriculum import SCHEMA
    import numpy as np
    prior=read_shared_json(a.prior)
    hashes={name:hashlib.sha256((ROOT/'src/fly_touhou'/name).read_bytes()).hexdigest()
            for name in prior['scientific_source_hashes']}
    changed=[name for name,sha in hashes.items() if sha!=prior['scientific_source_hashes'][name]]
    if prior['status']!='passed' or changed!=['thrl_course_store.py']:
        raise ValueError('Only the declared store preference may differ from the actual CUDA proof')
    store=CourseStore(a.course);assignment=dict(schema=SCHEMA,mode='current',lesson='s1_midboss',
        lesson_index=1,frontier=1,depth=300)
    rng=np.random.default_rng(71);selected=set()
    for _ in range(1000):
        trace,index=store.choose(assignment,rng,'train',32)
        span=next(s for s in store.spans('s1_midboss','train') if s['trace']['id']==trace['id'])
        assert span['complete'] and span['sampling_end']-300<=trace['records'][index]['frame']<=span['sampling_end']-32
        selected.add((trace['id'],index))
    cfg=read_shared_json(a.prior.parent/'functional_config.json')
    env=THRLCourseEnv(cfg['game_exe'],cfg['profile'],private_appdata=a.output/'native/userdata',
        curriculum_manifest=a.course,curriculum_seed=31,headless=True,binary_reader=True,
        policy_action_count=19,input_schema='semantic-separated-v1')
    try:
        _,info=env.reset(options=dict(assignment=assignment));prefix=info['prefix_reset'];start=env.snapshot.stage_frame
        teacher=make_teacher()
        for step in range(1500):
            action=int(native36_to_source19(teacher.probabilities(env.snapshot)).argmax())
            _,_,term,trunc,info=env.step(action)
            if term or trunc:break
        if term or not trunc or not info['episode_summary']['segment_reached'] or not info['bootstrap_allowed']:
            raise ValueError('Completed-span midboss takeover did not reach measured latter boundary')
        delta=dict(status='passed',changed_sources=changed,reason='Prefer complete phase spans when available; partial captures cannot masquerade as a measured phase tail',
            random_draws=1000,distinct_landings=len(selected),native_start_frame=start,
            native_end_frame=env.snapshot.stage_frame,prefix=prefix,summary=info['episode_summary'],
            PPO_updates_in_delta_probe=0,model_PPO_reward_sources_unchanged=True)
    finally:env.close()
    suites=list(ET.parse(a.tests).getroot().iter('testsuite'))
    if any(int(s.get('failures',0))+int(s.get('errors',0)) for s in suites):raise ValueError('CPU qualification failed')
    report=copy.deepcopy(prior)
    report.update(schema='thrl-progressive-composite-qualification-v1',
        retained_real_CUDA_proof=dict(path=str(a.prior.resolve()),sha256=hashlib.sha256(a.prior.read_bytes()).hexdigest(),
            scientific_source_hashes=prior['scientific_source_hashes'],actual_PPO_updates=prior['PPO_updates']),
        source_delta=delta,scientific_source_hashes=hashes,
        current_CPU_proof=dict(path=str(a.tests.resolve()),tests=sum(int(s.get('tests',0)) for s in suites)))
    write(a.output/'report.json',report)
    print(json.dumps(dict(status='passed',retained_actual_PPO_updates=prior['PPO_updates'],delta_native_steps=step+1)))


if __name__=='__main__':main()
