"""Record native inputs and independently verify two batched prefix landings."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--executable',type=Path,required=True);p.add_argument('--profile',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    from fly_touhou.background import action_mask
    from fly_touhou.native_replay import START_SCHEMA,state_fingerprint,replay_entry,validate_tape
    from fly_touhou.thrl_native_events import HiddenTHRLDialogueGeometryController
    from fly_touhou.native import sha256_file
    from fly_touhou.thrl_rewards import THRLRewardModel
    def controller(name):
        c=HiddenTHRLDialogueGeometryController(a.executable,a.profile,private_appdata=a.output/name/'userdata',
            worker_name=name,headless=True,binary_reader=True,native_seed=12345,timeout=25)
        c.start();s=c.ensure_run(3)
        while s.stage_frame<64:s=c.advance(1,1)
        if s.stage_frame!=64:raise ValueError('Prefix base requires stage1 frame64')
        return c,s
    c,s=controller('record');entry=None;report={'schema':'th14-prefix-reset-proof-v1','status':'running'}
    try:
        reward_model=THRLRewardModel((-184.,184.,32.,432.));reward_model.reset(s)
        base_hash=state_fingerprint(s);c.begin_input_tape();started=time.perf_counter()
        for i in range(256):
            action=(0,1,0,2,0,5,0,8)[(i//32)%8]
            s=c.advance(action_mask(action),4)
            reward_model.transition(s)
        elapsed=time.perf_counter()-started
        entry={'schema':START_SCHEMA,'id':'stage1-frame1088-seed12345-functional',
            'game_exe_sha256':sha256_file(a.executable),'native_seed':12345,'base_stage_frame':64,
            'base_state_sha256':base_hash,'landing_state_sha256':state_fingerprint(s),
            'landing':{'stage':s.stage,'stage_frame':s.stage_frame,'lives':s.lives,'bombs':s.bombs,'power':s.power},
            'raw_input_segments':validate_tape(c.end_input_tape()),'verified_independent_resets':0,
            'source':'fixed-input functional test; not a trained curriculum bank',
            'resources_injected':False,'native_deaths':True,'record_seconds':elapsed}
        entry['reward_tracking']={'schema':'thrl-prefix-reward-state-v1','cfg_bombs':reward_model.cfg_bombs,
            'rem_bombs_internal':reward_model.previous.rem_bombs_internal,
            **{k:s.extra['thrl_events'][k] for k in ('misses','bombs_used','point_items')}}
    finally:c.close()
    verifications=[]
    for name in ('replay1','replay2'):
        c,_=controller(name)
        try:
            started=time.perf_counter();s=replay_entry(c,entry)
            verifications.append({'name':name,'pid':c.pid,'landing_state_sha256':state_fingerprint(s),
                'prefix_seconds':time.perf_counter()-started,'native_prefix_ticks':1024})
        finally:c.close()
    entry['verified_independent_resets']=len(verifications)
    (a.output/'entry.json').write_text(json.dumps(entry,indent=2,ensure_ascii=False),encoding='utf-8')
    report.update(status='passed',verifications=verifications,entry='entry.json',
        bounded_functional_test=True,arbitrary_memory_savestate=False,stage_practice_ported=False)
    (a.output/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
