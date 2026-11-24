"""Read-only compact111 SNN learning and original-game telemetry."""
from pathlib import Path
import json
from . import tensorboard_continuous as common
from .thrl_dodge_diagnostics import _shared_binary


def export(exporter,label,root,*,launch,read_json):
    root=Path(root);source=next((root/n for n in ('failure.json','result.json','progress.json') if (root/n).exists()),root/'launch.json')
    row=read_json(source);step=row.get('joint_updates',0);prefix='fly_iterations/'+label+'/compact111_SNN/'
    cursor=exporter.state.setdefault('compact_spiking_cursors',{}).setdefault(label,{})
    if not cursor.get('scope') and (root/'algorithm.json').exists():
        common._text(exporter,prefix+'scope',read_json(root/'algorithm.json'),0);cursor['scope']=True
    if (root/'learning.jsonl').exists():
        with _shared_binary(root/'learning.jsonl') as stream:
            stream.seek(cursor.get('offset',0))
            while True:
                before=stream.tell();line=stream.readline()
                if not line or not line.endswith(b'\n'):cursor['offset']=before;break
                item=json.loads(line);n=item['step']
                if n>cursor.get('step',0):
                    for key,value in item.items():
                        if key not in ('step','phase') and isinstance(value,(int,float)):common._scalar(exporter,prefix+item['phase']+'/'+key,value,n)
                    for key,value in item['gradients'].items():common._scalar(exporter,prefix+'gradient/'+key,value,n)
                    cursor['step']=n
                cursor['offset']=stream.tell()
    common._scalar(exporter,prefix+'updates',step,step)
    common._scalar(exporter,prefix+'new_student_actions',row.get('new_student_actions',0),step)
    baseline=root/'stationary-baseline20/report.json'
    if baseline.exists() and not cursor.get('baseline'):
        report=read_json(baseline)
        common._text(exporter,prefix+'stationary_original_baseline',report,0);cursor['baseline']=True
    path=root/'evaluations.json'
    if path.exists():
        for node in read_json(path)['nodes']:
            n=node['updates']
            if n in cursor.get('native_nodes',[]):continue
            for key,value in node['heldout'].items():
                if isinstance(value,(int,float)):common._scalar(exporter,prefix+'heldout/'+key,value,n)
            for role in ('development','confirmation'):
                report=node.get(role)
                if report:
                    common._scalar(exporter,prefix+'native/'+role+'/first_stage_clean',report['first_stage_without_observed_contact_count'],n)
                    for key in ('mean','median','min','max'):common._scalar(exporter,prefix+'native/'+role+'/contact_'+key,report['observed_contact_frames'].get(key),n)
            cursor.setdefault('native_nodes',[]).append(n)
    exporter.state.setdefault('fly_online_jobs',{})[label]=dict(source=str(root),family='compact111_full_connectome_spiking',
        phase=row.get('phase',row.get('status','initializing')),producer_terminal=source.name in ('failure.json','result.json'),
        producer_file=source.name,CUDA_started=launch.get('CUDA_started'),CUDA_device=launch.get('CUDA_device'),
        actual_executor=dict(pid=launch.get('PID'),create_time=launch.get('create_time')),joint_updates=step,
        new_student_actions=row.get('new_student_actions',0),input_features=111,maps=False,read_error=None,
        candidate_conditioned=launch.get('candidate_conditioned',False),
        neural_features_per_candidate=8 if launch.get('candidate_conditioned') else 111,
        learning=row.get('learning'),evaluation=row.get('evaluation'),acquisition=row.get('acquisition'))
    return True


def export_calibration(exporter,label,root,*,launch,read_json):
    root=Path(root)
    source=next((root/n for n in ('failure.json','result.json','progress.json') if (root/n).exists()),root/'launch.json')
    row=read_json(source);full_story=bool(launch.get('full_story'))
    prefix='fly_iterations/'+label+('/full_story_PPO/' if full_story else '/DN_calibration/')
    cursor=exporter.state.setdefault('compact_spiking_cursors',{}).setdefault(label,{})
    if full_story and not cursor.get('scope') and (root/'algorithm.json').exists():
        common._text(exporter,prefix+'scope',read_json(root/'algorithm.json'),0);cursor['scope']=True
    if full_story:
        position_files=[*root.glob('position-cycle*.json'),*root.glob('development-cycle*/position-metrics.json')]
        for path in sorted(position_files):
            key=str(path.relative_to(root));stamp=path.stat().st_mtime_ns
            if cursor.setdefault('position_exports',{}).get(key)==stamp:continue
            frozen=path.name=='position-metrics.json'
            cycle=int(path.parent.name.removeprefix('development-cycle')) if frozen else int(path.stem.removeprefix('position-cycle'))
            role='frozen_position' if frozen else 'training_position'
            for stage,values in read_json(path)['stages'].items():
                for name in ('upper_fraction','unprotected_upper_fraction','mean_y','contacts_per_10000_frames',
                             'above_boss_frames','boss_frames','top112_frames','bottom400_frames','misses','upper_misses'):
                    if name in values:common._scalar(exporter,prefix+role+'/stage'+stage+'/'+name,values[name],cycle)
            cursor['position_exports'][key]=stamp
    if full_story and (root/'learning.jsonl').exists():
        with _shared_binary(root/'learning.jsonl') as stream:
            stream.seek(cursor.get('offset',0))
            while True:
                before=stream.tell();line=stream.readline()
                if not line or not line.endswith(b'\n'):cursor['offset']=before;break
                item=json.loads(line);n=item['step']
                if n>cursor.get('step',0):
                    for k,v in item.items():
                        if k not in ('step','cycle') and isinstance(v,(int,float)):common._scalar(exporter,prefix+'learning/'+k,v,n)
                    cursor['step']=n
                cursor['offset']=stream.tell()
    if full_story and (root/'numerical-updates.jsonl').exists():
        with _shared_binary(root/'numerical-updates.jsonl') as stream:
            stream.seek(cursor.get('numerical_offset',0))
            while True:
                before=stream.tell();line=stream.readline()
                if not line or not line.endswith(b'\n'):cursor['numerical_offset']=before;break
                item=json.loads(line);n=item['cycle']
                if n>cursor.get('numerical_cycle',-1):
                    for parameter,values in item['parameters'].items():
                        for key,value in values.items():
                            common._scalar(exporter,prefix+'numerical_updates/'+parameter+'/'+key,value,n)
                    if not cursor.get('numerical_scope'):
                        common._text(exporter,prefix+'numerical_updates/scope',
                            dict(precision=item['precision'],scope=item['scope']),n)
                        cursor['numerical_scope']=True
                    cursor['numerical_cycle']=n
                cursor['numerical_offset']=stream.tell()
    if full_story and (root/'learning-rate.jsonl').exists():
        with _shared_binary(root/'learning-rate.jsonl') as stream:
            stream.seek(cursor.get('lr_offset',0))
            while True:
                before=stream.tell();line=stream.readline()
                if not line or not line.endswith(b'\n'):cursor['lr_offset']=before;break
                item=json.loads(line);n=item['PPO_updates']
                if n>cursor.get('lr_step',0):
                    for k,v in item.items():
                        if isinstance(v,(int,float)):common._scalar(exporter,prefix+'learning_rate/'+k,v,n)
                    for k,v in item['learning_rates'].items():common._scalar(exporter,prefix+'learning_rate/group/'+k,v,n)
                    cursor['lr_step']=n;cursor['latest_learning_rates']=item
                cursor['lr_offset']=stream.tell()
    if full_story:
        for path in sorted(root.glob('development-cycle*/report.json')):
            role=path.parent.name
            if cursor.get(role):continue
            report=read_json(path);n=int(role.removeprefix('development-cycle'))
            common._scalar(exporter,prefix+'full_game/clears',report['clears'],n)
            common._scalar(exporter,prefix+'full_game/episodes',report['episodes'],n)
            for stage,count in report['stage_counts'].items():common._scalar(exporter,prefix+'full_game/reached_stage'+stage,count,n)
            episodes=report.get('episode_results',[])
            if episodes:
                common._scalar(exporter,prefix+'full_game/Bomb_used_mean',sum(e['bombs_used'] for e in episodes)/len(episodes),n)
                common._scalar(exporter,prefix+'full_game/final_power_mean',sum(e['endpoints']['end']['power'] for e in episodes)/len(episodes),n)
                for metric in ('confirmed_deathbombs','deathbomb_usable_decision_count','deathbomb_usable_bomb_requests'):
                    common._scalar(exporter,prefix+'full_game/'+metric,sum(e['reward_diagnostics'][metric] for e in episodes),n)
            common._text(exporter,prefix+'full_game/scope',dict(frozen=True,full_six_stage=True,episodes=report['episodes'],formal20=False),n)
            cursor[role]=True
    acquisition=row.get('acquisition')
    acquired=(max(row.get('new_student_actions',0),launch.get('inherited_student_actions',0)+acquisition.get('total_student_decisions',0))
        if full_story and acquisition else (acquisition or {}).get('student_actions',0))
    if acquisition and acquired>cursor.get('mining_actions',0):
        n=acquired
        for k,v in acquisition.items():
            if isinstance(v,(int,float)):common._scalar(exporter,prefix+'own_state_mining/'+k,v,n)
        for h,value in zip((4,12,24),acquisition.get('trusted_normal_rows_4_12_24',())):
            common._scalar(exporter,prefix+'own_state_mining/trusted_normal_rows_'+str(h)+'f',value,n)
            denominator=acquisition.get('normal_teacher_rows',0)
            if denominator:common._scalar(exporter,prefix+'own_state_mining/trusted_normal_fraction_'+str(h)+'f',value/denominator,n)
        cursor['mining_actions']=n
    coverage=None
    if full_story:
        coverage=(acquisition or {}).get('job_stage_coverage') or row.get('stage_coverage')
        if coverage is None and (root/'stage-coverage.json').exists():
            coverage=read_json(root/'stage-coverage.json')
        count=sum(v.get('decisions',0) for v in (coverage or {}).get('stages',{}).values())
        if coverage is not None and count>cursor.get('stage_coverage_decisions',-1):
            for stage,values in coverage['stages'].items():
                for key,value in values.items():
                    tag=prefix+'training_stage_coverage/stage'+stage+'/'+key
                    if isinstance(value,(int,float)):common._scalar(exporter,tag,value,acquired)
                    elif isinstance(value,list) and len(value)==3:
                        for h,v in zip((4,12,24),value):common._scalar(exporter,tag+'/'+str(h)+'f',v,acquired)
            common._scalar(exporter,prefix+'training_stage_coverage/complete_games',coverage.get('complete_games',0),acquired)
            common._text(exporter,prefix+'training_stage_coverage/scope',dict(
                scope=coverage['scope'],zero_sample_stages=coverage.get('zero_sample_stages'),
                completed_games_per_world=coverage.get('completed_games_per_world')),acquired)
            cursor['stage_coverage_decisions']=count
    path=root/'calibration.json'
    if path.exists() and not cursor.get('calibration'):
        report=read_json(path)
        common._text(exporter,prefix+'scope',report,1)
        for stage in ('before','after'):
            for fold,values in report[stage].items():
                for key,value in values.items():common._scalar(exporter,prefix+fold+'/'+key,value,0 if stage=='before' else 1)
        cursor['calibration']=True
    for role in ('development20','independent20'):
        path=root/role/'report.json'
        if path.exists() and not cursor.get(role):
            report=read_json(path)
            common._scalar(exporter,prefix+'native/'+role+'/first_stage_clean',report['first_stage_without_observed_contact_count'],1)
            common._scalar(exporter,prefix+'native/'+role+'/observed_contact_median',report['observed_contact_frames']['median'],1)
            common._text(exporter,prefix+'native/'+role+'/scope',dict(first_stage_only=True,full_game_clear_claim=False,
                stage_boundary_censored=report.get('stage_boundary_right_censored')),1)
            cursor[role]=True
    exporter.state.setdefault('fly_online_jobs',{})[label]=dict(source=str(root),family='candidate8_shared_full_connectome_spiking',
        phase=row.get('phase',row.get('status','initializing')),producer_terminal=source.name in ('failure.json','result.json'),
        producer_file=source.name,actual_executor=dict(pid=launch.get('PID'),create_time=launch.get('create_time')),
        input_features=launch.get('observation_features',111),neural_features_per_candidate=launch.get('neural_input_per_candidate',8),maps=False,candidate_conditioned=True,
        evaluation=row.get('evaluation'),fit=row.get('fit'),chunks=row.get('chunks'),read_error=None)
    exporter.state['fly_online_jobs'][label].update(acquisition=acquisition,
        CUDA_started=launch.get('CUDA_started'),CUDA_device=launch.get('CUDA_device'),
        PPO_updates=row.get('PPO_updates',0),new_student_actions=max(row.get('new_student_actions',0),acquired),
        learning=row.get('learning'),learning_rate=cursor.get('latest_learning_rates'),
        full_story=full_story,stage_coverage=coverage)
    if full_story:
        from .tensorboard_compact_story_dodge import export_dodge
        analysis=export_dodge(exporter,label,root,
            settings=getattr(exporter,'compact_story_dodge_settings',{}),read_json=read_json)
        if analysis is not None:
            exporter.state['fly_online_jobs'][label]['avoidance_analysis']=analysis
    return True
