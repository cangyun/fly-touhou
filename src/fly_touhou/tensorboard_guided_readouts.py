"""Shared-read JSON telemetry for serial frozen readout comparisons; no Torch."""
from pathlib import Path
import json
from . import tensorboard_continuous as common
from .thrl_dodge_diagnostics import _shared_binary

SCHEMA='frozen-guided-connectome-readout-comparison-v1'


def export_readouts(exporter,label,root,*,launch,read_json):
    root=Path(root)
    source=next((root/name for name in ('failure.json','result.json','progress.json') if (root/name).exists()),root/'launch.json')
    row=read_json(source);costs=row.get('costs') or {};step=costs.get('supervised_Adam',0)
    prefix='fly_iterations/'+label+'/frozen_readout_comparison/'
    cursor=exporter.state.setdefault('readout_comparison_cursors',{}).setdefault(label,{})
    if not cursor.get('scope'):
        common._text(exporter,prefix+'scope',dict(frozen_checkpoint=True,optimizer_updates=0,teacher_actions=0,
            variants=['composed baseline','risk_only','raw'],native_workers=20,concurrent_native_jobs=1,
            every_cohort='original Lunatic,greedy,NoBomb,uncapped first contact',
            development='same checkpoint and initial RNG seed plan;different policies visit different states',
            gate='>=8/20 first-stage clean and a new independent20 also>=8/20'),0)
        cursor['scope']=True
    tag=row.get('current_tag')
    child=row.get('current_evaluation') or {}
    evaluation=child.get('evaluation') or {}
    if tag:
        n=evaluation.get('student_decisions',0);p=prefix+'live/'+tag+'/'
        for key in ('completed','target','student_decisions','elapsed_seconds'):
            common._scalar(exporter,p+key,evaluation.get(key),n)
    for record in row.get('evaluations',[]):
        if record['tag'] in cursor.get('cohorts',[]):continue
        report=record['report'];p=prefix+'cohorts/'+record['tag']+'/'
        for key in ('episodes','first_stage_without_observed_contact_count','first_stage_without_observed_contact_fraction','actual_Bombs'):
            common._scalar(exporter,p+key,report.get(key),step)
        for key in ('median','mean','q25'):
            common._scalar(exporter,p+'first_contact_frames/'+key,(report.get('observed_contact_frames') or {}).get(key),step)
        cursor.setdefault('cohorts',[]).append(record['tag'])
    error=row.get('error');phase=row.get('phase',row.get('status'))
    if error=="RuntimeError('Operator STOP')":phase='operator_stopped';error=None
    signature=common._signature(dict(phase=phase,error=error,tag=tag))
    if cursor.get('status')!=signature:
        common._text(exporter,prefix+'status',dict(phase=phase,error=error,tag=tag),step);cursor['status']=signature
    common._scalar(exporter,prefix+'counts/source_Adam',step,step)
    common._scalar(exporter,prefix+'counts/additional_Adam',0,step)
    actual={}
    if tag and (root/tag/'launch.json').exists():actual=read_json(root/tag/'launch.json')
    exporter.state.setdefault('fly_online_jobs',{})[label]=dict(source=str(root),family='frozen_connectome_readout_comparison',
        phase=phase,producer_terminal=source.name in ('result.json','failure.json'),producer_file=source.name,
        student_decisions=costs.get('student_actions'),supervised_Adam=step,additional_optimizer_updates=0,
        actual_executor=dict(pid=actual.get('PID'),create_time=actual.get('create_time')),
        controller_executor=dict(pid=launch.get('PID'),create_time=launch.get('create_time')),
        app_goal_remains_paused=False,movement_readout=row.get('movement_readout'),evaluation=evaluation,
        read_error=None,parser_patch='serial-frozen-readout-comparison-v1')
    return True


def export_decoder_calibration(exporter,label,root,*,launch,read_json):
    root=Path(root)
    actor=launch.get('schema')=='current-DN-final-policy-actor-alignment-v1'
    source=next((root/name for name in ('failure.json','result.json','progress.json') if (root/name).exists()),root/'launch.json')
    row=read_json(source);costs=row.get('costs') or {}
    prefix='fly_iterations/'+label+('/final_actor_alignment/' if actor else '/decoder_identification/')
    cursor=exporter.state.setdefault('decoder_identification_cursors',{}).setdefault(label,{})
    if not cursor.get('scope'):
        common._text(exporter,prefix+'scope',dict(diagnostic=True,body_updates_in_this_phase=0,
            body='our current trained fullbrain and sensory encoder;temporary fixed representation',
            views=['DN1299 final composed policy'] if actor else ['encoded384','port_normalized384','DN1299'],input_probes_deployed=False,
            runtime='DN actor+calibrated risk;raw CE weight0' if actor else 'only learned DN risk decoder;no teacher or input shortcut',
            physical_validation='separate six whole teacher records;not native avoidance',
            next='reopen internal/sensory training after locating the bottleneck;not a goal-completion claim'),0)
        cursor['scope']=True
    path=root/'learning.jsonl'
    if path.exists():
        with _shared_binary(path) as stream:
            stream.seek(cursor.get('offset',0))
            while True:
                before=stream.tell();line=stream.readline()
                if not line or not line.endswith(b'\n'):cursor['offset']=before;break
                item=json.loads(line);n=item['step']
                views={'final_composed_policy':item['heldout']} if actor else item['heldout']
                for name,metrics in views.items():
                    for key,value in metrics.items():common._scalar(exporter,prefix+'heldout/'+name+'/'+key,value,n)
                cursor['step']=n;cursor['offset']=stream.tell()
    step=row.get('calibration_step',row.get('actor_step',cursor.get('step',0)))
    for key in ('records_done','records_total','batches_done','batches_total'):
        common._scalar(exporter,prefix+'cache/'+key,row.get(key),step)
    common._scalar(exporter,prefix+'counts/source_joint_Adam',costs.get('supervised_Adam'),step)
    common._scalar(exporter,prefix+'counts/current_head_Adam',step,step)
    common._scalar(exporter,prefix+'counts/current_body_updates',0,step)
    evaluation=row.get('evaluation') or {}
    for key in ('completed','target','student_decisions','elapsed_seconds'):
        common._scalar(exporter,prefix+'live_native20/'+key,evaluation.get(key),evaluation.get('student_decisions',0))
    report=row.get('report') if actor else row.get('native_report')
    report=report or {}
    if report and not cursor.get('native_completed'):
        for key in ('episodes','first_stage_without_observed_contact_count','actual_Bombs'):
            common._scalar(exporter,prefix+'frozen_DN_native20/'+key,report.get(key),step)
        for key in ('median','mean','q25'):
            common._scalar(exporter,prefix+'frozen_DN_native20/first_contact_frames/'+key,
                (report.get('observed_contact_frames') or {}).get(key),step)
        cursor['native_completed']=True
    phase=row.get('phase',row.get('status'));error=row.get('error')
    signature=common._signature(dict(phase=phase,error=error))
    if cursor.get('status')!=signature:
        common._text(exporter,prefix+'status',dict(phase=phase,error=error,temporary_frozen_body=True),step)
        cursor['status']=signature
    family='temporary_final_policy_actor_alignment' if actor else 'temporary_frozen_body_decoder_identification'
    exporter.state.setdefault('fly_online_jobs',{})[label]=dict(source=str(root),family=family,
        phase=phase,producer_terminal=source.name in ('result.json','failure.json'),producer_file=source.name,
        student_decisions=costs.get('student_actions'),supervised_Adam=costs.get('supervised_Adam'),
        decoder_Adam=step,body_updates_in_this_phase=0,temporary_frozen_body=True,
        actual_executor=dict(pid=launch.get('PID'),create_time=launch.get('create_time')),
        app_goal_remains_paused=False,evaluation=evaluation,read_error=None,
        parser_patch='three-coordinate-DN-decoder-identification-v1')
    return True


def export_direct_teacher_reference(exporter,label,root,*,launch,read_json):
    root=Path(root)
    source=next((root/name for name in ('failure.json','report.json','result.json','progress.json') if (root/name).exists()),root/'launch.json')
    row=read_json(source);prefix='fly_iterations/'+label+'/engineered_teacher_NoBomb_reference/'
    cursor=exporter.state.setdefault('direct_teacher_reference_cursors',{}).setdefault(label,{})
    if not cursor.get('scope'):
        common._text(exporter,prefix+'scope',dict(teacher='original ResourceV10 conditioned on18 movement actions',
            neural_model_test=False,optimizer_updates=0,CUDA_used=False,student_training_modified=False,
            original_native_RNG='same20 native seeds as preceding frozen student cohort',
            evaluation='uncapped;first native accepted state4/Miss;continuous shooting;no Bomb',
            teacher_is_not_native_future_oracle=True),0);cursor['scope']=True
    step=row.get('teacher_decisions',row.get('teacher_actions',0))
    for key in ('completed','target','native_worlds','elapsed_seconds'):
        common._scalar(exporter,prefix+'live/'+key,row.get(key),step)
    common._scalar(exporter,prefix+'live/teacher_decisions',step,step)
    if source.name=='report.json' and not cursor.get('completed'):
        for key in ('episodes','first_stage_without_observed_contact_count','actual_Bombs'):
            common._scalar(exporter,prefix+'reference_result/'+key,row.get(key),step)
        for key in ('median','mean','q25'):
            common._scalar(exporter,prefix+'reference_result/first_contact_frames/'+key,(row.get('observed_contact_frames') or {}).get(key),step)
        cursor['completed']=True
    phase=row.get('phase',row.get('status'));terminal=source.name in ('failure.json','report.json','result.json')
    signature=common._signature(dict(phase=phase,error=row.get('error')))
    if cursor.get('status')!=signature:
        common._text(exporter,prefix+'status',dict(phase=phase,error=row.get('error'),neural_model_test=False),step);cursor['status']=signature
    exporter.state.setdefault('fly_online_jobs',{})[label]=dict(source=str(root),family='engineered_ResourceV10_NoBomb_reference',
        phase=phase,producer_terminal=terminal,producer_file=source.name,neural_model_test=False,CUDA_used=False,
        teacher_decisions=step,student_training_modified=False,actual_executor=dict(pid=launch.get('PID'),create_time=launch.get('create_time')),
        completed=row.get('completed',row.get('episodes')),target=row.get('target',20),read_error=None)
    return True


def export_single_frozen(exporter,label,root,*,launch,read_json):
    root=Path(root)
    source=next((root/name for name in ('failure.json','result.json','progress.json') if (root/name).exists()),root/'launch.json')
    row=read_json(source);costs=row.get('costs') or launch.get('source_counters') or {}
    prefix='fly_iterations/'+label+'/frozen_original_NoBomb/'
    cursor=exporter.state.setdefault('single_frozen_cursors',{}).setdefault(label,{})
    if not cursor.get('scope'):
        common._text(exporter,prefix+'scope',dict(frozen=True,optimizer_updates=0,teacher_actions=0,
            movement_readout=launch.get('movement_readout'),episodes=launch.get('episodes'),
            native_frame_cap=None,source_checkpoint=launch.get('source_sha256'),
            one_cohort_does_not_confirm_basic_avoidance=True),0);cursor['scope']=True
    evaluation=row.get('evaluation') or {};n=evaluation.get('student_decisions',0)
    for key in ('completed','target','student_decisions','elapsed_seconds'):
        common._scalar(exporter,prefix+'live/'+key,evaluation.get(key),n)
    report=row.get('report') or {}
    if report and not cursor.get('completed'):
        for key in ('episodes','first_stage_without_observed_contact_count','actual_Bombs'):
            common._scalar(exporter,prefix+'result/'+key,report.get(key),costs.get('supervised_Adam',0))
        for key in ('median','mean','q25'):
            common._scalar(exporter,prefix+'result/first_contact_frames/'+key,
                (report.get('observed_contact_frames') or {}).get(key),costs.get('supervised_Adam',0))
        cursor['completed']=True
    phase=row.get('phase',row.get('status'));error=row.get('error')
    signature=common._signature(dict(phase=phase,error=error))
    if cursor.get('status')!=signature:
        common._text(exporter,prefix+'status',dict(phase=phase,error=error),n);cursor['status']=signature
    exporter.state.setdefault('fly_online_jobs',{})[label]=dict(source=str(root),family='frozen_original_NoBomb_evaluation',
        phase=phase,producer_terminal=source.name in ('result.json','failure.json'),producer_file=source.name,
        student_decisions=costs.get('student_actions'),supervised_Adam=costs.get('supervised_Adam'),
        additional_optimizer_updates=0,actual_executor=dict(pid=launch.get('PID'),create_time=launch.get('create_time')),
        app_goal_remains_paused=False,evaluation=evaluation,movement_readout=launch.get('movement_readout'),
        read_error=None,parser_patch='single-frozen-NoBomb-evaluation-v1')
    return True
