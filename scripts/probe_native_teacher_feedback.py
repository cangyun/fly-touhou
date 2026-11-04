"""Shadow-label sealed on-policy training data; no game input or model learning."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from fly_touhou.experiment import _file_lock,file_sha256
from fly_touhou.native_teacher_feedback import (NativeV6TrainingFeedback,annotate_teacher_targets,
    verify_closed_teacher_reference)
from fly_touhou.whole_brain_ppo import WholeBrainRollout,load_rollout
from fly_touhou.state import Snapshot


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--first-updates',type=int,default=16)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--rare-data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2);assert not a.output.exists() and a.first_updates>0
    reference=verify_closed_teacher_reference(a.reference)
    with _file_lock(a.run/'.events.lock'):
        lines=(a.run/'events.jsonl').read_text(encoding='utf-8').splitlines()
    events=[json.loads(x) for x in lines]
    ends=[i for i,e in enumerate(events) if e['event']=='checkpoint' and e['payload']['tag']=='ppo-latest']
    assert len(ends)>=a.first_updates
    end=ends[a.first_updates-1]+1;events=events[:end]
    a.output.mkdir(parents=True,exist_ok=False)
    prefix=a.output/'events_prefix.jsonl';prefix.write_text('\n'.join(lines[:end])+'\n',encoding='utf-8')
    sealed={e['payload']['path']:e['payload'] for e in events if e['event']=='ppo_behavior_rollout_sealed'}
    feedback=NativeV6TrainingFeedback();report={'schema':'native-v6-training-feedback-shadow-probe-v1',
        'status':'running','reference':reference,'updates':[],'native_game_started':False,'learning_performed':False,
        'runtime_action_override':False,'independent_evaluation_used':False}
    modes=Counter();targets=Counter();counts=Counter()
    for e in events:
        if e['event']!='epoch':continue
        epoch=e['payload']['epoch'];path=(a.run/'rollouts'/f'{epoch:06d}.states.pt.gz').resolve()
        seal=sealed[str(path)];assert file_sha256(path)==seal['sha256']
        saved=load_rollout(path)
        rollout=WholeBrainRollout(**{k:(saved[k].numpy() if torch.is_tensor(saved[k]) else saved[k])
            for k in WholeBrainRollout.__dataclass_fields__})
        native=seal['native_trajectory'];assert file_sha256(native['path'])==native['sha256']
        started=time.perf_counter();metadata=annotate_teacher_targets(rollout,native['path'],feedback)
        seconds=time.perf_counter()-started
        verifier=NativeV6TrainingFeedback();verifier.load_state_dict(metadata['before_teacher_state'])
        expected=rollout.teacher_targets.copy();weights=rollout.teacher_target_weights.copy()
        other=annotate_teacher_targets(rollout,native['path'],verifier)
        np.testing.assert_array_equal(expected,rollout.teacher_targets)
        np.testing.assert_array_equal(weights,rollout.teacher_target_weights)
        assert metadata==other
        active=weights>0
        for d in metadata['details']:
            modes[d['expert']['mode']]+=1
            target=d['expert']['target'];targets[target.get('kind','none') if isinstance(target,dict) else 'none']+=1
        counts['queries']+=len(rollout);counts['active_targets']+=int(active.sum())
        counts['student_argmax_matches_expert_target']+=int(((expected.argmax(-1)==rollout.actions)&active).sum())
        counts['admissible_expert_conflicts']+=sum(d['expert_admissible_mass']<.5 for d in metadata['details'])
        out=a.output/f'{epoch:06d}.targets.pt'
        with out.open('xb') as stream:torch.save({'targets':torch.as_tensor(expected),
            'weights':torch.as_tensor(weights),'metadata':metadata},stream)
        report['updates'].append({'epoch':epoch,'targets':{'path':str(out.resolve()),'sha256':file_sha256(out)},
            'native_trajectory':native,'source_rollout_sha256':seal['sha256'],'queries':len(rollout),
            'active_targets':int(active.sum()),'annotation_seconds':seconds,'restored_mode_recomputation_error':0.})
    assert len(report['updates'])==a.first_updates
    rare=json.loads((a.rare_data/'report.json').read_text(encoding='utf-8'))
    raw=Path(rare['raw_rows']['path']);assert file_sha256(raw)==rare['raw_rows']['sha256']
    positive=[]
    with gzip.open(raw,'rb') as stream:
        for line in stream:
            row=json.loads(line);s=row['snapshot'];p=s['player']
            if not (p is not None and s['bombs']>0 and p['state']==4 and p['deathbomb_timer']>0):continue
            teacher=NativeV6TrainingFeedback();probabilities,_=teacher.query(0,'positive_native_fixture',Snapshot.from_dict(s))
            assert probabilities[18:].sum()>.99
            positive.append({'frame':s['frame'],'stage_frame':s['stage_frame'],'timer':p['deathbomb_timer'],
                'stock':s['bombs'],'expert_Bomb_mass':float(probabilities[18:].sum()),'raw_line_sha256':hashlib.sha256(line).hexdigest()})
    assert len(positive)==len(rare['positive_observation_steps'])==4
    report.update(status='passed',counts=dict(counts),teacher_modes=dict(modes),target_kinds=dict(targets),
        actual_rare_stocked_window_checks=positive,event_prefix_sha256=file_sha256(prefix),
        active_teacher_state_after_prefix=feedback.state_dict(),scope='shadow labels only; no teacher loss deployed or student ability claim')
    for name,source in {'probe.py':Path(__file__),'native_teacher_feedback.py':ROOT/'src/fly_touhou/native_teacher_feedback.py'}.items():
        out=a.output/'source_archive'/name;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(source.read_bytes())
    with (a.output/'report.json').open('x',encoding='utf-8') as out:json.dump(report,out,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({'status':'passed','counts':report['counts'],'modes':report['teacher_modes'],
        'target_kinds':report['target_kinds'],'real_positive_checks':len(positive)},ensure_ascii=False))


if __name__=='__main__':main()
