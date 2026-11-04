"""Verify exact prefix reset and curriculum-cut bootstrap in the real env."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--executable',type=Path,required=True);p.add_argument('--profile',type=Path,required=True)
    p.add_argument('--entry',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    from fly_touhou.thrl_segment_env import THRLSegmentEnv
    from fly_touhou.thrl_geometry_observation import FEATURE_DIM
    entry=json.loads(a.entry.read_text(encoding='utf-8'))
    end={'stage':entry['landing']['stage'],'frame':entry['landing']['stage_frame']+8}
    env=THRLSegmentEnv(a.executable,a.profile,private_appdata=a.output/'native'/'userdata',
        start_bank=a.entry,segment_end=end,policy_action_count=19,headless=True,binary_reader=True)
    report={'schema':'thrl-real-segment-env-functional-v1','status':'running'}
    try:
        observation,info=env.reset(seed=11)
        if observation['features'].shape!=(FEATURE_DIM,):raise ValueError('Old observation shape at curriculum start')
        if env.snapshot.stage_frame!=entry['landing']['stage_frame']:raise ValueError('Wrong native curriculum landing')
        receipt=info['prefix_reset'];pid=env.pid
        for _ in range(2):observation,reward,terminated,truncated,info=env.step(9)
        if terminated or not truncated or not info['bootstrap_allowed'] or info['native_terminal_kind']!='ongoing':
            raise ValueError('Curriculum cut was incorrectly treated as native terminal')
        if info['episode_summary']['outcome']!='curriculum-cut':raise ValueError('Curriculum cut mislabeled')
        report.update(pid=pid,prefix_reset=receipt,cut_frame=env.snapshot.stage_frame,
            terminated=terminated,truncated=truncated,bootstrap_allowed=info['bootstrap_allowed'],
            native_terminal_kind=info['native_terminal_kind'],reward_components=info.get('reward_components'),
            learning_efficacy_test=False)
        _,second=env.reset(options={'start_index':0,'force_prefix':True})
        if env.pid==pid or env.snapshot.stage_frame!=entry['landing']['stage_frame']:
            raise ValueError('Repeated native starting-point reset did not reach a fresh exact landing')
        report['second_reset']={'pid':env.pid,'prefix_reset':second['prefix_reset']}
        report['status']='passed'
    except BaseException as error:
        report.update(status='failed',error=repr(error))
        raise
    finally:
        env.close();(a.output/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
