"""Offline V9 parity and V10 support checks on recorded native states.

No original worlds, no optimizer and no changed model/checkpoint. The observed
adaptive paths are real; distributions used to isolate the old mask are
explicit controlled references, not claimed raw-snapshot teacher outputs.
"""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def probabilities(args):
    sys.path.insert(0,str(args.runtime/'src'))
    import numpy as np
    from fly_touhou.thrl_study_io import read_shared_json as read
    from fly_touhou.atomic_json_io import atomic_write_json as write
    from fly_touhou.thrl_teacher import make_teacher
    from fly_touhou.resource_teacher_v9 import REVISION as V9
    from fly_touhou.state import Snapshot
    revision=V9
    if args.teacher=='V10':
        from fly_touhou.resource_teacher_v10 import REVISION
        revision=REVISION
    result=[]
    for case in read(args.dataset)['cases']:
        if not case['name'].startswith('recorded-native-'):continue
        teacher=make_teacher(revision);rows=[]
        for raw in case['snapshots']:
            p=teacher.probabilities(Snapshot.from_dict(raw))
            if p.shape!=(36,) or not np.isfinite(p).all() or (p<0).any() or abs(float(p.sum())-1)>2e-6:
                raise ValueError('Invalid native36 teacher reference')
            rows.append(dict(values=p.tolist(),sha256=hashlib.sha256(p.tobytes()).hexdigest(),
                dtype=str(p.dtype),soft_detail=getattr(teacher,'_soft_detail',None)))
        result.append(dict(name=case['name'],rows=rows))
    write(args.output,dict(runtime=str(args.runtime),teacher=revision,cases=result))


def run(args):
    sys.path.insert(0,str(ROOT/'src'))
    import numpy as np
    from fly_touhou.thrl_study_io import read_shared_json as read,utc_now
    from fly_touhou.atomic_json_io import atomic_write_json as write
    from fly_touhou.thrl_native_trace_replay import RecordedAvoidanceCase
    from fly_touhou.thrl_native_env import SOURCE19_TO_NATIVE36
    from fly_touhou.thrl_movement_teacher import MovementSafetyTeacher
    from fly_touhou.resource_teacher_v10 import soften_route_reference,REVISION
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    definitions=[('old_V9',args.runtime,'V9'),('current_V9',ROOT,'V9'),('current_V10',ROOT,'V10')]
    for name,runtime,teacher in definitions:
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'--probabilities','--runtime',str(runtime),
            '--dataset',str(args.dataset),'--teacher',teacher,'--output',str(args.output/(name+'.json'))],
            check=True,capture_output=True,text=True)
    old,current,new=[read(args.output/(name+'.json')) for name,_,_ in definitions]
    if old['cases']!=current['cases']:raise ValueError('Historical V9 reference bytes changed')
    checks=[]
    for a,b in zip(current['cases'],new['cases']):
        if a['name']!=b['name'] or len(a['rows'])!=len(b['rows']):raise ValueError('Native case plan changed')
        for x,y in zip(a['rows'],b['rows']):
            if x['values'][18:]!=y['values'][18:]:raise ValueError('V10 changed inherited Bomb marginal variants')
            checks.append(dict(case=a['name'],Bomb_variants_exact=True,soft_detail=y['soft_detail']))
    certificates=read(args.certificates)
    report=read(args.probe/'report.json');episodes={(r['policy'],r['game']):r for r in report['episode_results']}
    teacher=MovementSafetyTeacher();native=np.asarray(SOURCE19_TO_NATIVE36[:18]);restored=[]
    for row in certificates['results']:
        if not row['restrictive_teacher_mask_excludes_observed_buffered_adaptive_path']:continue
        case=RecordedAvoidanceCase(args.probe/'report.json',episodes[row['policy'],row['game']]);t=row['tail_index']
        if case.source_trace_sha256!=row['source_trace_sha256']:raise ValueError('Observed certificate trace differs')
        observation={k:case.arrays[k][t] for k in ('features','objects','maps')}
        safe,detail=teacher.label(observation['features'],observation['objects'],observation['maps'],force=True)
        chosen=row['observed_next_six_actions'][0]
        if not safe.any() or safe[chosen]:raise ValueError('Original restrictive-mask conflict no longer reproduces')
        p=np.zeros(36);p[native[safe]]=.8/int(safe.sum());p[18]=.2
        q,new_detail=soften_route_reference(p,observation,teacher)
        if q[native[chosen]]<=0 or not new_detail['guidance_usable'] or not np.array_equal(q[18:],p[18:]):
            raise ValueError('Actual buffered adaptive first action was not restored')
        origin=teacher._scene(observation['features'],observation['objects'])[0]
        margins=teacher.clearances(observation['features'],observation['objects'],np.repeat(origin[None],18,axis=0),teacher.velocities,duration=4)
        if np.any(q[native[margins<=1]]):raise ValueError('Restoration introduced a modeled unsafe first action')
        restored.append(dict(policy=row['policy'],game=row['game'],tail_index=t,
            source_trace_sha256=case.source_trace_sha256,chosen=chosen,
            old_controlled_probability=float(p[native[chosen]]),new_controlled_probability=float(q[native[chosen]]),
            actual_adaptive_current_motion_margin=row['conservative_buffer_after_path_error']))
    if len(restored)!=certificates['summary']['restrictive_mask_conflicts']:raise ValueError('Not all declared conflicts were checked')
    result=dict(schema='B1-soft-route-reference-native-record-offline-qualification-v1',utc=utc_now(),status='passed',
        teacher_revision=REVISION,recorded_native_snapshots=len(checks),historical_V9_reference_bytes_exact=True,
        historical_native_Bomb_variants_exact=True,actual_adaptive_first_choices_restored=len(restored),
        restored_cases=restored,raw_native_checks=checks,source_certificates=str(args.certificates),
        source_certificates_sha256=sha(args.certificates),source_probe_report_sha256=sha(args.probe/'report.json'),
        controlled_mask_distribution_used_for_path_isolation=True,actual_V10_native_efficacy_demonstrated=False,
        native_worlds_started=0,optimizer_updates=0,model_parameters_or_reward_changed=False,
        script_sha256=sha(__file__))
    write(args.output/'report.json',result)
    (args.output/'qualification_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:result[k] for k in ('status','recorded_native_snapshots','historical_V9_reference_bytes_exact',
        'actual_adaptive_first_choices_restored','native_worlds_started','optimizer_updates')}))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for key in ['runtime','dataset','certificates','probe','output']:p.add_argument('--'+key,type=Path)
    p.add_argument('--probabilities',action='store_true');p.add_argument('--teacher',choices=['V9','V10'])
    a=p.parse_args();probabilities(a) if a.probabilities else run(a)
