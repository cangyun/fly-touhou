"""One bounded real stage1->2 teacher transition; no training/checkpoint edit."""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.atomic_json_io import atomic_write_json as write
from fly_touhou.thrl_segment_env import load_start_bank
from fly_touhou.thrl_teacher import teacher_env_factory


def main():
    parser=argparse.ArgumentParser(__doc__)
    for name in ('output','bank','executable','profile'):parser.add_argument('--'+name,required=True,type=Path)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    entry=next(row for row in load_start_bank(args.bank) if row['target']=='boss' and row['native_seed']==12345)
    write(args.output/'bank.json',{'schema':'th14-native-start-bank-aligned-v1','entries':[entry]})
    env=teacher_env_factory({'executable':str(args.executable),'profile_path':str(args.profile),
        'private_appdata':str(args.output/'userdata'),'start_bank':str(args.output/'bank.json'),
        'input_schema':'semantic-separated-v1','policy_action_count':19,'headless':True,
        'binary_reader':True,'full_story_fraction':0.,'curriculum_seed':11})
    report={'status':'running','scope':'one real stage1->stage2 transition','training_updates':0};trace=[]
    try:
        obs,info=env.reset();report['prefix']=info['prefix_reset'];begun=time.monotonic()
        for index in range(3000):
            before=env.snapshot
            obs,reward,terminated,truncated,info=env.step(info['teacher_label']['action'])
            after=env.snapshot
            if index%128==0 or after.stage!=before.stage:
                trace.append({'decision':index+1,'before_stage':before.stage,'after_stage':after.stage,
                    'stage_frame':after.stage_frame,'native_frame':after.frame,
                    'info':{key:info.get(key) for key in ('lives','bombs','episode_frames','policy_steps',
                        'teacher_calls','teacher_actions_executed','nonplay_ticks','loading_bridge_ticks')}})
                write(args.output/'progress.json',{'decisions':index+1,'stage':after.stage,'stage_frame':after.stage_frame})
            if after.stage==2 and after.stage_frame>0:
                report.update(status='passed',decisions=index+1,stage=after.stage,stage_frame=after.stage_frame,
                    elapsed_seconds=time.monotonic()-begun,final_visibility=after.extra.get('native_visibility'))
                break
            if terminated or truncated:raise RuntimeError('Teacher ended before the declared transition')
        else:raise RuntimeError('Bounded transition quota exhausted')
    except BaseException as error:
        report.update(status='failed',error=repr(error),traceback=traceback.format_exc());raise
    finally:
        env.close();write(args.output/'report.json',report);write(args.output/'trace.json',{'events':trace})
        print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
