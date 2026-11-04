"""Measure original-game movement and clamps using owned process-local inputs."""
from pathlib import Path
import argparse
import gzip
import json
import math
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.connectome import file_sha256
from fly_touhou.hazard_environment import HazardResourceCampaignEnv
from fly_touhou.learning import _context


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe',type=Path,required=True)
    parser.add_argument('--profile',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    options=parser.parse_args()
    output=options.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    results=[]
    for name,action,axis,limit in (('left',10,'x',40),('right',11,'x',40),
            ('top',12,'y',40),('bottom',13,'y',40),('focused',2,'x',3)):
        env=HazardResourceCampaignEnv(options.exe,options.profile,difficulty=3,max_episode_frames=200,
            private_appdata=output/'userdata'/name,dialogue_skip=True,
            window_title=f'原作移动校准 · {name}')
        trace=output/(name+'.jsonl.gz')
        row={'name':name,'action':action,'axis':axis,'status':'incomplete','speeds':[]}
        try:
            _,info=env.reset()
            row['native_context']=_context(env.snapshot,info)
            row['start_player']=env.snapshot.player.__dict__
            stationary=0
            with gzip.open(trace,'xt',encoding='utf-8') as stream:
                for step in range(limit):
                    before=env.snapshot
                    _,_,terminated,truncated,info=env.step(action)
                    after=env.snapshot
                    stream.write(json.dumps({'step':step,'action':action,'snapshot':before.to_dict(),
                        'elapsed_frames':info['elapsed_frames'],'next_snapshot':after.to_dict()},
                        ensure_ascii=False,allow_nan=False)+'\n')
                    p,q=before.player,after.player
                    if (p is None or q is None or p.state!=1 or q.state!=1
                            or p.invulnerable_frames or q.invulnerable_frames or terminated or truncated):
                        row['status']='interrupted-by-native-state'
                        break
                    distance=math.hypot(q.x-p.x,q.y-p.y)
                    if step<3:
                        row['speeds'].append(distance/info['elapsed_frames'])
                    stationary=stationary+1 if distance==0 else 0
                    if stationary>=2:
                        row.update(status='measured',bound=getattr(q,axis))
                        break
                else:
                    if name=='focused':
                        row['status']='measured'
            row['last_player']=env.snapshot.player.__dict__ if env.snapshot.player else None
        finally:
            env.close()
        row['trace_sha256']=file_sha256(trace)
        row['trace']=str(trace)
        results.append(row)
        print(json.dumps(row,ensure_ascii=False),flush=True)
    measured=all(r['status']=='measured' for r in results)
    settings=None
    if measured:
        by_name={r['name']:r for r in results}
        settings={name:by_name[name]['bound'] for name in ('left','right','top','bottom')}
        settings['normal_speed']=statistics.median(v for r in results[:4] for v in r['speeds'])
        settings['focused_speed']=statistics.median(by_name['focused']['speeds'])
    report={'schema':'original-th14-motion-calibration-v1','exe_sha256':file_sha256(options.exe),
        'profile_sha256':file_sha256(options.profile),'source_sha256':file_sha256(__file__),
        'desktop_input':False,'gameplay_resource_overrides':False,'difficulty':3,
        'all_measurements_complete':measured,'planner_overrides':settings,'cases':results,
        'scope':'Native fresh normal campaigns; bounds and initial movement of selected character/shot only.'}
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    if not measured:
        raise RuntimeError('Some movement cases did not finish; do not use incomplete calibration')


if __name__=='__main__':
    main()
