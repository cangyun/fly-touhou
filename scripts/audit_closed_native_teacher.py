"""Read a closed original-game teacher trajectory; validate resources and geometry."""
import argparse
from collections import Counter
import ctypes
import gzip
import hashlib
import json
from pathlib import Path
import sys

PIN=Path('./outputs/legacy-build/source_snapshots/20261019-v4-63')
sys.path.insert(0,str(PIN/'src'))
from fly_touhou.experiment import file_sha256
from fly_touhou.hazard_reader import FULL_GAME_SHA256
from fly_touhou.native_geometry import collision_geometry
from fly_touhou.state import Snapshot


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--job',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    kernel=ctypes.WinDLL('kernel32');kernel.SetPriorityClass(kernel.GetCurrentProcess(),0x40)
    assert not a.output.exists()
    status=json.loads((a.job/'status.json').read_text(encoding='utf-8'))
    assert status['status']=='finished' and status['exit_code']==0
    job=json.loads((a.job/'job.json').read_text(encoding='utf-8'));argv=job['arguments']
    trace=Path(argv[argv.index('--output')+1]);assert trace.parent.resolve()==a.run.resolve()
    exe=Path(argv[argv.index('--exe')+1]);assert file_sha256(exe)==FULL_GAME_SHA256
    source=Path(job['script']).parents[1]
    source_manifest=json.loads((source/'snapshot_manifest.json').read_text(encoding='utf-8'))
    for entry in source_manifest['files']:
        assert file_sha256(source/entry['path'])==entry['sha256']
    for name in ('native_geometry.py','hazard_reader.py','state.py'):
        assert file_sha256(PIN/'src/fly_touhou'/name)==file_sha256(source/'src/fly_touhou'/name)
    sidecar=trace.with_suffix('.report.json');closed=json.loads(sidecar.read_text(encoding='utf-8'))
    assert len(closed['episodes'])==1 and closed['config']['teacher']=='native-v6'
    assert closed['config']['difficulty']==3 and closed['config']['episodes']==1
    a.output.mkdir(parents=True,exist_ok=False)
    digest=file_sha256(trace);decoded=hashlib.sha256();stages={};counts=Counter();history=[]
    previous=None;summary=None;collection=None;first=None;samples=[]
    with gzip.open(trace,'rb') as stream:
        for number,raw in enumerate(stream,1):
            decoded.update(raw);row=json.loads(raw);kind=row.get('record_type')
            if kind=='metadata':
                assert row['config']==closed['config'] and row['teacher_kind']=='ResourceTeacherV6'
                continue
            if kind=='episode_summary':summary=row;continue
            if kind=='collection_summary':collection=row;continue
            if kind!='transition':raise ValueError('Unexpected closed teacher record')
            assert summary is None and row['step']==counts['decisions']
            s=row['snapshot'];t=row['next_snapshot'];p=s['player'];q=t['player']
            if previous is not None:assert previous==s,'Native snapshot sequence is discontinuous'
            previous=t
            if first is None:first=s
            assert s['difficulty']==3 and s['character']==0 and s['subshot']==0
            for snapshot in (s,t):
                x=snapshot['extra']
                assert x['desktop_input'] is False and x['exe_sha256']==FULL_GAME_SHA256
                assert x['run_mode']==0 and x['replay_playback']==0 and x['continues']==0
            probs=row['teacher_probabilities'];assert len(probs)==36
            assert abs(sum(probs)-1)<1e-5 and all(0<=x<=1 for x in probs)
            assert row['action']==max(range(36),key=probs.__getitem__)
            assert row['teacher_diagnostic']['teacher']=='ResourceTeacherV6'
            assert row['elapsed_frames']==4
            counts['decisions']+=1;counts['native_frames']+=row['elapsed_frames']
            counts['net_life_decrements']+=max(0,s['lives']-t['lives'])
            counts['observed_life_gains']+=max(0,t['lives']-s['lives'])
            counts['Bomb_requests']+=row['action']>=18
            window=p is not None and s['bombs']>0 and p['state']==4 and p['deathbomb_timer']>0
            counts['stocked_deathbomb_observation_steps']+=window
            counts['requests_in_stocked_window']+=window and row['action']>=18
            recovered=bool(window and row['action']>=18 and q and q['state']==1 and t['lives']==s['lives'])
            counts['observed_deathbomb_recoveries']+=recovered
            if window:samples.append({'step':row['step'],'stage':s['stage'],'stage_frame':s['stage_frame'],
                'stock':s['bombs'],'timer':p['deathbomb_timer'],'action':row['action'],
                'recovery_observed':recovered,'raw_line':number,'raw_line_sha256':hashlib.sha256(raw).hexdigest()})
            stage=s['stage']
            if not history or history[-1]!=stage:history.append(stage)
            d=stages.setdefault(str(stage),{'decisions':0,'max_player_scale':0.,'max_player_circle_component':0.,
                'large_player_flag_steps':0,'max_bullet_scale':0.,'scaled_collision_bullet_records':0,
                'max_collision_bullet_radius':0.,'net_life_decrements':0,'Bomb_requests':0})
            d['decisions']+=1;d['net_life_decrements']+=max(0,s['lives']-t['lives']);d['Bomb_requests']+=row['action']>=18
            snapshot=Snapshot.from_dict(s);metadata={b['id']:b for b in s['extra']['hazard_metadata']['bullets']}
            shapes,player=collision_geometry(snapshot,metadata)
            if player:
                d['large_player_flag_steps']+=player['large_player_flag']
                d['max_player_scale']=max(d['max_player_scale'],s['extra']['player_scale'])
                d['max_player_circle_component']=max(d['max_player_circle_component'],player['circle_component'])
            for value in metadata.values():
                d['max_bullet_scale']=max(d['max_bullet_scale'],value['native_scale'])
                d['scaled_collision_bullet_records']+=bool(value['native_flags']&0x40 and value['native_scale']!=1)
            for shape in shapes.values():
                if shape['shape']=='circle':d['max_collision_bullet_radius']=max(d['max_collision_bullet_radius'],shape['bullet_radius'])
            if counts['decisions']%2500==0:print(json.dumps({'checked':counts['decisions'],'stage':stage}),flush=True)
    assert summary==closed['episodes'][0] and collection['episodes']==1
    c=summary['campaign'];final=summary['final_snapshot']
    assert history==c['stage_history']==[1,2,3,4,5,6]
    assert c['episode_mode']=='finite-life-campaign' and c['continues']==0 and c['run_mode']==0 and c['replay_playback']==0
    assert c['native_ending_flag'] is True and c['lunatic_six_stage_no_continue'] is True
    assert not final['extra']['native_gameover'] and final==previous
    assert first['lives']==2 and first['bombs']==3 and first['power']==100
    assert counts['decisions']==summary['steps']==25010 and counts['native_frames']==summary['native_frames']==100040
    assert counts['net_life_decrements']==c['life_decrements']==3
    assert counts['Bomb_requests']==c['bomb_actions'] and file_sha256(trace)==digest
    report={'schema':'closed-original-native-teacher-success-audit-v1','status':'passed',
        'role':'engineered ResourceTeacherV6, not fly-brain student acceptance',
        'trace':{'path':str(trace.resolve()),'sha256':digest,'bytes':trace.stat().st_size},
        'sidecar_sha256':file_sha256(sidecar),'decoded_complete_stream_sha256':decoded.hexdigest(),
        'source_manifest_sha256':file_sha256(source/'snapshot_manifest.json'),'counts':dict(counts),
        'all_geometry_before_states_validated':True,'all_native_state_sequences_contiguous':True,
        'all_teacher_actions_recomputed_from_recorded_probabilities':True,'stage_history':history,'stages':stages,
        'final':{'lives':final['lives'],'bombs':final['bombs'],'power':final['power'],'score':final['score'],'native_ending':True},
        'actual_stocked_deathbomb_observations':samples,'student_goal_complete':False,
        'native_game_started_by_auditor':False,'training_performed':False}
    (a.output/'audit_script.py').write_bytes(Path(__file__).read_bytes())
    with (a.output/'report.json').open('x',encoding='utf-8') as out:json.dump(report,out,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({'status':'passed','counts':report['counts'],'final':report['final'],'stage6':stages['6']}),flush=True)


if __name__=='__main__':main()
