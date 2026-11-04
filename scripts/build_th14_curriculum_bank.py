"""Capture original first-stage opening/midboss/boss starts with V6 teacher.

Only independently reproduced landings enter the published bank. This is data
preparation, not a PPO run or a success-rate benchmark. No resource injection.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--executable',required=True,type=Path);p.add_argument('--profile',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);p.add_argument('--seed',type=int,default=12345)
    p.add_argument('--wall-seconds',type=int,default=900);p.add_argument('--max-decisions',type=int,default=6000)
    a=p.parse_args()
    if not 0<=a.seed<=65535 or not 1<=a.max_decisions<=10000 or not 1<=a.wall_seconds<=3600:
        raise ValueError('Explicit finite capture budget required')
    a.output.mkdir(parents=True,exist_ok=False)
    from fly_touhou.atomic_json_io import atomic_write_json
    from fly_touhou.native import sha256_file
    from fly_touhou.native_replay import START_SCHEMA,state_fingerprint,replay_entry,validate_tape
    from fly_touhou.thrl_native_env import THRLNativeEnv
    from fly_touhou.thrl_native_events import HiddenTHRLDialogueGeometryController
    from fly_touhou.thrl_teacher import make_teacher,native36_to_source19
    from fly_touhou.thrl_study_io import utc_now
    status={'schema':'th14-curriculum-bank-build-v1','status':'starting','seed':a.seed,
        'formal_test':False,'PPO_training':False,'GPU_training':False,'decisions':0,
        'targets':['opening','midboss','boss'],'captured':[],'verified':[],'errors':[]}
    begun=time.monotonic();last=0.;entries=[]
    env=None
    def publish(force=False):
        nonlocal last
        now=time.monotonic()
        if force or now-last>=5:
            status.update(updated_utc=utc_now(),wall_seconds=now-begun)
            atomic_write_json(a.output/'progress.json',status);last=now
    try:
        env=THRLNativeEnv(a.executable,a.profile,private_appdata=a.output/'capture'/'userdata',
            headless=True,binary_reader=True,native_seed=a.seed,policy_action_count=19)
        env.reset();s=env.snapshot
        while s.stage_frame<64:s=env.controller.advance(1,1)
        env.snapshot=s;env._encode(s);env.reward_model.reset(s)
        base=state_fingerprint(s);env.controller.begin_input_tape();teacher=make_teacher()
        status['status']='capturing';publish(True)
        for i in range(a.max_decisions):
            if time.monotonic()-begun>a.wall_seconds or (a.output/'STOP').exists():
                status['capture_stop']='budget_or_operator';break
            s=env.snapshot
            roles=s.extra.get('thrl_events',{}).get('ecl_roles',{})
            candidates=[]
            if s.stage==1 and s.stage_frame>=128:candidates.append('opening')
            if roles.get('midbosses'):candidates.append('midboss')
            if roles.get('main_bosses'):candidates.append('boss')
            for target in candidates:
                if target in status['captured'] or s.dialogue or s.player.state!=1:continue
                tape=validate_tape(list(env.controller._input_tape))
                entry={'schema':START_SCHEMA,'id':f'stage1-{target}-seed{a.seed}',
                    'target':target,'game_exe_sha256':sha256_file(a.executable),
                    'native_seed':a.seed,'base_stage_frame':64,'base_state_sha256':base,
                    'landing_state_sha256':state_fingerprint(s),'raw_input_segments':tape,
                    'landing':{'stage':s.stage,'stage_frame':s.stage_frame,'lives':s.lives,'bombs':s.bombs,'power':s.power},
                    'source':'ResourceTeacherV6 source19; native finite lives, autonomous Bomb',
                    'reward_tracking':{'schema':'thrl-prefix-reward-state-v1',
                        'cfg_bombs':env.reward_model.cfg_bombs,
                        'rem_bombs_internal':env.reward_model.previous.rem_bombs_internal,
                        **{k:s.extra['thrl_events'][k] for k in ('misses','bombs_used','point_items')}},
                    'resources_injected':False,'native_deaths':True,'verified_independent_resets':0}
                entries.append(entry);status['captured'].append(target)
                atomic_write_json(a.output/f'captured_{target}.json',entry);publish(True)
            status.update(decisions=i,stage=s.stage,stage_frame=s.stage_frame,lives=s.lives,
                bombs=s.bombs,power=s.power,bullets=len(s.bullets),pid=env.pid)
            publish()
            if len(entries)==3:break
            distribution=native36_to_source19(teacher.probabilities(s))
            _,_,terminated,truncated,info=env.step(int(distribution.argmax()))
            if terminated or truncated:
                status['capture_terminal']=info.get('episode_summary');break
        env.controller.end_input_tape();env.close();env=None
        status['status']='verifying_landings';publish(True)
        verified=[]
        for entry in entries:
            receipts=[]
            for repeat in range(2):
                if (a.output/'STOP').exists():break
                directory=a.output/'verification'/entry['target']/str(repeat)
                c=HiddenTHRLDialogueGeometryController(a.executable,a.profile,private_appdata=directory/'userdata',
                    headless=True,binary_reader=True,native_seed=a.seed,timeout=25,worker_name=f"verify-{entry['target']}-{repeat}")
                try:
                    c.start();c.ensure_run(3);started=time.perf_counter();s=replay_entry(c,entry)
                    receipts.append({'pid':c.pid,'state_sha256':state_fingerprint(s),
                                     'prefix_seconds':time.perf_counter()-started})
                except Exception as error:
                    status['errors'].append({'target':entry['target'],'repeat':repeat,'error':repr(error),
                        'traceback':traceback.format_exc()})
                    break
                finally:c.close()
            entry['verified_independent_resets']=len(receipts)
            entry['verification_receipts']=receipts
            atomic_write_json(a.output/f"entry_{entry['target']}.json",entry)
            if len(receipts)==2:verified.append(entry);status['verified'].append(entry['target'])
            publish(True)
        bank={'schema':'th14-native-start-bank-v2','entries':verified,'native_seed':a.seed,
              'resources_injected':False,'formal_test':False,'instant_stage_jump':False}
        atomic_write_json(a.output/'bank.json',bank)
        status['status']='complete' if len(verified)==3 else 'partial'
        status['bank']=str((a.output/'bank.json').resolve());publish(True)
    except BaseException as error:
        status.update(status='failed',error=repr(error),traceback=traceback.format_exc());publish(True);raise
    finally:
        if env is not None:env.close()
        publish(True)
    print(json.dumps(status,ensure_ascii=False))


if __name__=='__main__':main()
