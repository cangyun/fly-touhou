"""Keep native behavior unchanged and recompute v4 labels from current states.

These are off-policy corrective labels, not demonstrations that v4 executed
the recorded actions. The original failed trace remains immutable.
"""
from collections import Counter
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.connectome import file_sha256
from fly_touhou.hazard_observation import HazardObservationEncoder
from fly_touhou.hazard_reader import FULL_GAME_SHA256
from fly_touhou.motion_calibration import calibrated_planner
from fly_touhou.native_hazard_planner import NativeHazardPlanner
from fly_touhou.resource_teacher_v4 import ResourceTeacherV4
from fly_touhou.state import Snapshot
from fly_touhou.whole_brain_learning import load_campaign_demonstrations


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--motion-calibration',type=Path,required=True)
    options=parser.parse_args()
    source=options.input.resolve(strict=True)
    output=options.output.resolve()
    if output.exists():
        raise FileExistsError('Original/corrected traces are never overwritten')
    output.parent.mkdir(parents=True,exist_ok=True)
    source_sha=file_sha256(source)
    teacher=None
    previous_id=None
    modes=Counter()
    changes=transitions=0
    episodes=[]
    with gzip.open(source,'rt',encoding='utf-8') as incoming, gzip.open(output,'xt',encoding='utf-8') as outgoing:
        def emit(row):
            outgoing.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        header=json.loads(next(incoming))
        if header.get('teacher_kind')!='ResourceTeacherV3' or header.get('schema')!='whole-brain-campaign-demo-v1':
            raise ValueError('This correction explicitly targets the sealed failed native v3 teacher trace')
        encoder=HazardObservationEncoder.from_specification(header['encoder'])
        planner_config,calibration=calibrated_planner(encoder.planner.config,options.motion_calibration,
            exe_sha256=FULL_GAME_SHA256)
        teacher=ResourceTeacherV4(NativeHazardPlanner(planner_config))
        provenance={'schema':'native-off-policy-relabel-v1','source':str(source),'source_sha256':source_sha,
            'source_teacher_kind':header['teacher_kind'],'source_code_sha256':file_sha256(__file__),
            'motion_calibration':calibration,'states_actions_rewards_and_episode_ids_unchanged':True,
            'labels_use_current_snapshot_only':True,'future_snapshot_used_for_labels':False,
            'behavior_policy':'failed ResourceTeacherV3; v4 did not execute these trajectories',
            'corrected_teacher':teacher.config()}
        header=dict(header,teacher_kind='ResourceTeacherV4',relabel=provenance,
            config=dict(header['config'],phase='v4-off-policy-hazard-corrections',teacher='native-v4-relabel',
                role='offline corrected labels on preserved failed-policy native states; not v4 behavior performance',
                teacher_config=teacher.config(),motion_calibration=calibration))
        emit(header)
        for number,line in enumerate(incoming,2):
            record=json.loads(line)
            kind=record.get('record_type')
            if kind=='transition':
                if record['episode_id']!=previous_id:
                    teacher.reset()
                    previous_id=record['episode_id']
                    episodes.append(previous_id)
                snapshot=Snapshot.from_dict(record['snapshot'])
                if (snapshot.character,snapshot.subshot,snapshot.difficulty)!=(0,0,3):
                    raise ValueError('Recorded native identity differs from calibrated teacher')
                encoder.encode(snapshot)
                probabilities=teacher.probabilities(snapshot)
                diagnostic=teacher.diagnostics()
                original=record['teacher_probabilities']
                record['parent_label_reference']={'raw_line_1based':number,
                    'raw_line_sha256':hashlib.sha256(line.encode('utf-8')).hexdigest(),
                    'teacher_probabilities':original,'recorded_behavior_action':record['action']}
                record['teacher_probabilities']=probabilities.tolist()
                record['teacher_diagnostic']=diagnostic
                modes[diagnostic['mode']]+=1
                transitions+=1
                changes+=int(probabilities.argmax()!=max(range(len(original)),key=original.__getitem__))
                emit(record)
                if transitions%250==0:
                    outgoing.flush()
                    print('RELABEL',transitions,'changed_argmax',changes,flush=True)
            elif kind=='episode_summary':
                emit({'record_type':'episode_summary','episode_id':record['episode_id'],
                    'steps':record['steps'],'native_frames':record['native_frames'],
                    'summary_role':'native behavior preserved from failed source; this is not corrected teacher survival'})
            elif kind=='collection_summary':
                emit(dict(record,summary_role='native behavior unchanged; only labels corrected offline'))
            else:
                raise ValueError('Unexpected source record')
    # Runs all provenance, contiguity, complete-episode and encoding guards.
    validated=load_campaign_demonstrations(output)
    if len(validated.episodes)!=len(episodes):
        raise ValueError('Relabelled episode count differs')
    report={'schema':'native-off-policy-relabel-report-v1','provenance':provenance,
        'output':str(output),'output_sha256':file_sha256(output),'episodes':episodes,
        'transitions':transitions,'changed_teacher_argmax':changes,'new_label_modes':dict(modes),
        'complete_native_episodes_validated':len(validated.episodes)}
    output.with_suffix('.relabel.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
