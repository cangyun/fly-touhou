"""Compare versioned sensors on exact original replays; no fitting or games."""
from pathlib import Path
import argparse,gzip,hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.thrl_study_io import read_shared_json as read,utc_now
from fly_touhou.atomic_json_io import atomic_write_json as write
from fly_touhou.state import Snapshot
from fly_touhou.thrl_inputs import SEPARATED_OBJECT_PRIORITY_INPUT as OLD,SEPARATED_CONTACT_INPUT as NEW,make_encoder
from fly_touhou.thrl_native_trace_replay import RecordedAvoidanceCase
from fly_touhou.thrl_observation_audit import audit_snapshot
from fly_touhou.thrl_movement_teacher import MovementSafetyTeacher
from fly_touhou.thrl_teacher import make_teacher,native36_to_source19
from fly_touhou.resource_teacher_v7 import REVISION as V7
from fly_touhou.resource_teacher_v8 import REVISION as V8


def run(a):
    source=read(a.report);replay=read(a.raw_replay/'report.json')
    if replay.get('original_native_replay_verified') is not True:
        raise ValueError('Exact native replay prerequisite missing')
    rows={r['evaluation_seed']:r for r in source['episode_results'] if r['policy']=='trained_greedy'}
    cases={seed:RecordedAvoidanceCase(a.report,rows[seed]) for seed in
        {r['evaluation_seed'] for r in replay['results']}}
    encoders={OLD:make_encoder(OLD),NEW:make_encoder(NEW)};results=[]
    movement=MovementSafetyTeacher()
    for path in sorted(a.raw_replay.rglob('*.json.gz')):
        receipt=read(path.with_suffix('.receipt.json'));compressed=path.read_bytes()
        if hashlib.sha256(compressed).hexdigest()!=receipt['sha256']:
            raise ValueError('Raw capture receipt differs')
        raw=gzip.decompress(compressed)
        if hashlib.sha256(raw).hexdigest()!=receipt['uncompressed_sha256']:
            raise ValueError('Raw decompressed capture differs')
        record=json.loads(raw);snapshot=Snapshot.from_dict(record['snapshot'])
        case=cases[record['evaluation_seed']];index=record['decision']
        obs={version:e.encode(snapshot) for version,e in encoders.items()}
        if record['side']=='before':
            expected={k:case.arrays[k][index-case.tail_begin] for k in ('features','objects')}
            if index==len(case.actions)-1:expected['maps']=case.arrays['last_preaction_maps']
        else:expected={k:case.arrays['after_'+k] for k in ('features','objects','maps')}
        for k,v in expected.items():
            if not np.array_equal(obs[OLD][k],v):raise ValueError('Legacy V3 input changed: '+str(path)+':'+k)
        audit={v:audit_snapshot(snapshot,input_schema=v) for v in encoders}
        detail=dict(path=str(path),sha256=receipt['sha256'],game=case.row['game'],
            decision=index,side=record['side'],original_V3_exact=True,
            fields_changed={k:int(np.count_nonzero(obs[OLD][k]!=obs[NEW][k])) for k in obs[OLD]},
            selection={v:r['selection'] for v,r in audit.items()},
            retained_zero_HP_bodies=[{'id':r['id'],'hp':r['hp'],'x':r['x'],'y':r['y']}
                for r in snapshot.extra['resources']['enemies'] if r['hp']<=0 and r['contactable'] and not r.get('intangible',False)])
        if record['side']=='before' and index==len(case.actions)-1:
            detail['movement_teacher']={}
            for v,o in obs.items():
                safe,why=movement.label(o['features'],o['objects'],o['maps'],force=True)
                detail['movement_teacher'][v]=dict(reason=why['reason'],supported_actions=np.flatnonzero(safe).tolist())
            detail['resource_teacher']={}
            for version in (V7,V8):
                teacher=make_teacher(version);p=native36_to_source19(teacher.probabilities(snapshot))
                detail['resource_teacher'][version]=dict(argmax=int(p.argmax()),Bomb_probability=float(p[18]),
                    chosen_source_action=case.actions[-1],movement_probabilities=p[:18].tolist())
        results.append(detail)
    if len(results)!=20:raise ValueError('Expected exactly20 bounded raw snapshots')
    interventions=read(a.interventions/'report.json')
    if interventions.get('status')!='completed' or interventions['native_runs']!=108:
        raise ValueError('Original action verification incomplete')
    result=dict(status='qualified',utc=utc_now(),schema='B1-HP-independent-contact-sensor-native-qualification-v1',
        snapshots=len(results),all_historical_V3_inputs_exact=True,new_input_schema=NEW,
        native_action_interventions=interventions['summary'],results=results,
        native_replay_source=str(a.raw_replay/'report.json'),native_action_source=str(a.interventions/'report.json'),
        new_original_games=0,optimizer_updates=0,fit_on_diagnostic_cases=False,
        diagnostic_seeds_retired_from_future_independent_acceptance=True,
        scope='Correct measured observation omission; no claim the migrated policy is already trained or reliably evasive')
    if a.output.exists():raise ValueError('Preserve previous qualification output')
    write(a.output,result)
    print(json.dumps({k:result[k] for k in ('status','snapshots','all_historical_V3_inputs_exact','optimizer_updates')},ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for key in ('report','raw-replay','interventions','output'):p.add_argument('--'+key,type=Path,required=True)
    run(p.parse_args())
