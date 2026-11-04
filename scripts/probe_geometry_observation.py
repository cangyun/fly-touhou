"""Bounded original-game sensor/reward integration probe, not student acceptance."""
import argparse
import gzip
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np  # noqa: E402
from fly_touhou.geometry_environment import GeometryObservationCampaignEnv  # noqa: E402
from fly_touhou.campaign_rewards import attach_training_reward,GEOMETRY_EMERGENCY_REWARD  # noqa: E402
from fly_touhou.resource_teacher_v5 import ResourceTeacherV5  # noqa: E402
from fly_touhou.native_hazard_planner import GeometryNativeHazardPlanner  # noqa: E402
from fly_touhou.experiment import file_sha256  # noqa: E402
from fly_touhou.whole_brain_ppo import _native_observation  # noqa: E402


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe',type=Path,required=True)
    parser.add_argument('--profile',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--frames',type=int,default=256)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    env=GeometryObservationCampaignEnv(args.exe,args.profile,difficulty=3,frame_skip=4,
        private_appdata=args.output/'userdata',dialogue_skip=True,max_episode_frames=args.frames,
        window_title='V4几何感觉短测 · 非果蝇验收')
    attach_training_reward(env,GEOMETRY_EMERGENCY_REWARD,gamma=.995)
    teacher=ResourceTeacherV5(GeometryNativeHazardPlanner(env.encoder.planner.config))
    report={'schema':'original-geometry-observation-integration-v1','status':'running',
        'native_game_started':True,'whole_brain_started':False,'student_trained':False,
        'desktop_input':False,'actions_from':'V5 geometry teacher',
        'excluded_from_training_and_clear_acceptance':True,'control_intervention':False,
        'scope':'normal original-game input and read-only dimensions; no geometry/resource manipulation'}
    try:
        obs,info=env.reset()
        report.update(pid=env.pid,encoder=env.encoder.specification(),reward=env.training_reward.specification(),
                      exe_sha256=info['exe_sha256'])
        count=frames=with_bullets=0
        trajectory=args.output/'native_geometry.jsonl.gz'
        with gzip.open(trajectory,'xt',encoding='utf-8') as stream:
            stream.write(json.dumps({'record_type':'metadata','schema':'geometry-sensor-integration-fixture-v1',
                'excluded_from_training_and_clear_acceptance':True,'encoder':report['encoder']})+'\n')
            while frames<args.frames:
                before=env.snapshot
                _native_observation(obs,env.encoder,before,info)
                np.testing.assert_array_equal(obs[:env.encoder.prefix_size],env.encoder.prefix.encode(before))
                action=int(teacher.probabilities(before).argmax())
                obs,reward,terminated,truncated,info=env.step(action)
                assert info['training_reward_diagnostics']['schema']==GEOMETRY_EMERGENCY_REWARD
                assert info['reward_components']['survival']==0.
                stream.write(json.dumps({'record_type':'transition','step':count,'snapshot':before.to_dict(),
                    'next_snapshot':env.snapshot.to_dict(),'action':action,'elapsed_frames':info['elapsed_frames'],
                    'reward':reward,'reward_components':info['reward_components'],
                    'training_reward_diagnostics':info['training_reward_diagnostics'],
                    'terminated':terminated,'truncated':truncated},allow_nan=False)+'\n')
                count+=1;frames+=info['elapsed_frames'];with_bullets+=bool(before.bullets)
                if terminated or truncated:break
        report.update(status='passed',decisions=count,native_frames=frames,decisions_with_bullets=with_bullets,
            prefix_exact=True,trajectory={'path':str(trajectory.resolve()),'sha256':file_sha256(trajectory)})
    except BaseException as error:
        report.update(status='failed',error=repr(error))
        raise
    finally:
        env.close()
        (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'status':report['status'],'decisions':report['decisions'],'frames':report['native_frames'],
        'old_prefix_exact':True,'new_sensor_size':env.encoder.size},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
