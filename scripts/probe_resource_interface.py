"""Short original-game read-only resource sensor verification, not training."""
from pathlib import Path
import argparse
import gzip
import json

from fly_touhou.campaign import FULL_GAME_SHA256
from fly_touhou.resource_environment import ResourceCampaignEnv
from fly_touhou.campaign_teacher import CampaignTeacher
from fly_touhou.experiment import ExperimentLogger

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--frames',type=int,default=600)
    parser.add_argument('--exe',type=Path,default=Path(
        r'games/th14/th14.exe'))
    args=parser.parse_args()
    profile=ROOT/'profiles/th14_1.00b_original.json'
    config={'phase':'native-resource-sensor-probe','game_exe_sha256':FULL_GAME_SHA256,
            'difficulty':3,'frames':args.frames,'gameplay_resource_writes':False,'training':False}
    files={'original_game':args.exe,'profile':profile,'probe':Path(__file__)}
    for name in ('resources','resource_environment','resource_observation','native','background',
                 'frida_controller','campaign','planner','campaign_teacher','state','observation','experiment'):
        files[name]=ROOT/'src/fly_touhou'/f'{name}.py'
    logger=ExperimentLogger(args.output,config,files=files)
    env=ResourceCampaignEnv(args.exe,profile,difficulty=3,max_episode_frames=args.frames)
    maximum=0
    kinds=set()
    try:
        with logger.phase('original-game-resource-read'),gzip.open(
            args.output/'snapshots.jsonl.gz','xt',encoding='utf-8') as stream:
            _,info=env.reset()
            logger.record_native_context(env.snapshot,info)
            teacher=CampaignTeacher(env.encoder.planner)
            for step in range(args.frames//4+1):
                resources=env.snapshot.extra['resources']
                maximum=max(maximum,len(resources['items']))
                kinds.update(item['kind'] for item in resources['items'])
                stream.write(json.dumps(env.snapshot.to_dict(),ensure_ascii=False)+'\n')
                # Produce ordinary native fast-shot item drops during the first
                # seconds; remaining probe movement uses the retained teacher.
                action=9 if step<100 else int(teacher.probabilities(env.snapshot).argmax())
                _,_,terminated,truncated,_=env.step(action)
                if terminated or truncated:break
            report={'maximum_measured_items':maximum,'measured_item_kinds':sorted(kinds),
                'observations':step+1,'observation_dimension':env.encoder.size,
                'exe_sha256':FULL_GAME_SHA256,'desktop_input':False,
                'passed':maximum>0,'final_snapshot':env.snapshot.to_dict()}
            with (args.output/'report.json').open('x',encoding='utf-8') as stream:
                json.dump(report,stream,ensure_ascii=False,indent=2)
            logger.event('resource_sensor_result',{k:v for k,v in report.items() if k!='final_snapshot'})
            print(json.dumps({k:v for k,v in report.items() if k!='final_snapshot'}),flush=True)
    finally:
        env.close()


if __name__=='__main__':main()
