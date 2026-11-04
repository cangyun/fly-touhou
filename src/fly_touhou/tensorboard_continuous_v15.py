"""Read-only direct-teacher DAgger telemetry in the existing TensorBoard."""
from pathlib import Path
import json
from . import tensorboard_continuous as common
from . import tensorboard_continuous_v14 as previous
from .thrl_dodge_diagnostics import _shared_binary

SCHEMA='continuous-calibrated-fullbrain-direct-teacher-DAgger-v1'
current_algorithm_jobs=common.current_algorithm_jobs


def export_continuous_connectome(exporter,label,root,*,read_json):
    root=Path(root)
    launch=read_json(root/'launch.json') if (root/'launch.json').exists() else {}
    if launch.get('schema','').startswith('candidate-SNN-'):
        from .tensorboard_compact_spiking import export_calibration
        return export_calibration(exporter,label,root,launch=launch,read_json=read_json)
    if launch.get('schema')=='compact111-fullbrain-spiking-short-cycle-v1':
        from .tensorboard_compact_spiking import export
        return export(exporter,label,root,launch=launch,read_json=read_json)
    if launch.get('schema')=='engineered-fullbrain-LIF-supervised-native-v1':
        from .tensorboard_spiking_engineered import export
        return export(exporter,label,root,launch=launch,read_json=read_json)
    if launch.get('schema')=='current-fullbrain-student-controlled-DAgger-acquisition-v1' and launch.get('neural_family')=='full-connectome-hard-spiking':
        from .tensorboard_spiking_engineered import export_acquisition
        return export_acquisition(exporter,label,root,launch=launch,read_json=read_json)
    if launch.get('schema')=='chronological-fullbrain-original-GRU-avoidance-supervision-v1':
        from .tensorboard_chronological_avoidance import export
        return export(exporter,label,root,launch=launch,read_json=read_json)
    if launch.get('schema')=='ResourceV10-matched-native-NoBomb-reference-v1':
        from .tensorboard_guided_readouts import export_direct_teacher_reference
        return export_direct_teacher_reference(exporter,label,root,launch=launch,read_json=read_json)
    if launch.get('schema')=='frozen-guided-connectome-readout-comparison-v1':
        from .tensorboard_guided_readouts import export_readouts
        return export_readouts(exporter,label,root,launch=launch,read_json=read_json)
    if launch.get('schema') in ('current-fullbrain-three-coordinate-risk-readout-calibration-v1',
            'current-DN-final-policy-actor-alignment-v1'):
        from .tensorboard_guided_readouts import export_decoder_calibration
        return export_decoder_calibration(exporter,label,root,launch=launch,read_json=read_json)
    if launch.get('frozen') is True and launch.get('Bomb_forbidden') is True and 'movement_readout' in launch:
        from .tensorboard_guided_readouts import export_single_frozen
        return export_single_frozen(exporter,label,root,launch=launch,read_json=read_json)
    if launch.get('schema')!=SCHEMA:
        return previous.export_continuous_connectome(exporter,label,root,read_json=read_json)
    source=next((root/name for name in ('failure.json','result.json','progress.json') if (root/name).exists()),root/'launch.json')
    row=read_json(source);costs=row.get('costs') or {};step=costs.get('supervised_Adam',0)
    prefix='fly_iterations/'+label+'/direct_teacher_avoidance/'
    cursor=exporter.state.setdefault('guided_DAgger_cursors',{}).setdefault(label,{})
    if not cursor.get('scope'):
        common._text(exporter,prefix+'scope',dict(teacher='ResourceV10+18-direction geometry,not D95 actions',
            input_path='raw -> calibrated8053VP -> fullconnectome ->1299DN -> actor/risk',
            actor_and_risk='joint supervision;current phase and trainable parameters recorded in algorithm.json',
            replay='persistent raw expert/recent student/historical hard;20/40/40',
            raw_burn_in=16,BPTT=16,burn_in_is_approximate=True,
            native_opening_learning_probability=launch.get('native_opening_learning_probability',0.),
            native_opening_history='actual reset;selected opening prefixes learn without burn-in',
            evaluation='separate frozen20 NoBomb cohorts;training labels are geometric proxies',
            app_goal_remains_paused=launch.get('app_goal_remains_paused',True)),0);cursor['scope']=True
    path=root/'learning.jsonl'
    if path.exists():
        with _shared_binary(path) as stream:
            stream.seek(cursor.get('offset',0))
            while True:
                before=stream.tell();line=stream.readline()
                if not line or not line.endswith(b'\n'):
                    cursor['offset']=before;break
                item=json.loads(line);n=item['step']
                if n>cursor.get('Adam',0):
                    for key in ('total_loss','actor_CE','composed_CE','safe_set','physical_risk','risk_ordering','actual_contact','actor_native_hit','gradient_norm'):
                        common._scalar(exporter,prefix+'loss/'+key,item.get(key),n)
                    for key in ('VP_abs_gt095','VP_abs_gt099','port_drive_RMS','DN_RMS'):
                        common._scalar(exporter,prefix+'activity/'+key,item.get(key),n)
                    for key in ('attention_KL','attention_queries','attention_multi_object_queries',
                                'attention_normalized_entropy','attention_critical_object_top1',
                                'attention_query_norm','attention_coefficient'):
                        common._scalar(exporter,prefix+'attention_teaching/'+key,item.get(key),n)
                    for key in ('trusted4_rows','teacher_rows','mixed_rows','raw_unsafe','composed_unsafe','risk_minimum_unsafe','contact_rows','contact_false_safe'):
                        common._scalar(exporter,prefix+'replay_proxy/'+key,item.get(key),n)
                    for name,value in item.get('sample_mix',{}).items():
                        common._scalar(exporter,prefix+'replay_sample_mix/'+name,value,n)
                    common._scalar(exporter,prefix+'replay/native_opening_learning_sequences',item.get('native_opening_learning_sequences'),n)
                    common._scalar(exporter,prefix+'time/elapsed_seconds',item.get('elapsed_seconds'),n)
                    cursor['Adam']=n
                cursor['offset']=stream.tell()
    for key,value in costs.items():common._scalar(exporter,prefix+'counts/'+key,value,step)
    # Completed training episodes can span parameter updates and exclude live
    # long-surviving games. They are a progress diagnostic, never a dev score.
    from .continuous_training_dodge_summary import training_contact_summary
    for path in sorted(root.glob('round-*-collection.json')):
        if path.name in cursor.get('training_collections',[]):continue
        round_number=int(path.name.split('-')[1]);summary=training_contact_summary(read_json(path))
        for group,metrics in summary['groups'].items():
            for key,value in metrics.items():
                common._scalar(exporter,prefix+'training_completed_games_not_frozen/'+group+'/'+key,value,round_number)
        common._text(exporter,prefix+'training_completed_games_not_frozen/scope',
            {k:v for k,v in summary.items() if k!='groups'},round_number)
        cursor.setdefault('training_collections',[]).append(path.name)
        cursor['latest_training_contacts']=dict(round=round_number,**summary)
    native={}
    if not source.name in ('failure.json','result.json') and (root/'native-live.json').exists():
        native=read_json(root/'native-live.json')
        for key in ('acknowledged_student_actions','pending_actions','world_count'):
            common._scalar(exporter,prefix+'live_current_native_pool/'+key,native.get(key),step)
    common._scalar(exporter,prefix+'replay/persistent_sequences',row.get('replay_sequences'),step)
    for folder in sorted(root.glob('development-round*'))+[root/'confirmation20']:
        path=folder/'report.json'
        if not path.exists() or folder.name in cursor.get('evaluations',[]):continue
        report=read_json(path);p=prefix+'frozen_NoBomb/'+folder.name+'/'
        for key in ('episodes','first_stage_without_observed_contact_count','first_stage_without_observed_contact_fraction','actual_Bombs'):
            common._scalar(exporter,p+key,report.get(key),step)
        for key in ('median','mean','q25'):
            common._scalar(exporter,p+'first_contact_frames/'+key,(report.get('observed_contact_frames') or {}).get(key),step)
        cursor.setdefault('evaluations',[]).append(folder.name)
    phase=row.get('phase',row.get('status'));terminal=source.name in ('result.json','failure.json')
    operator_stopped=row.get('error')=="RuntimeError('Operator STOP')" and (root/'STOP').exists()
    error=None if operator_stopped else row.get('error')
    if operator_stopped:phase='operator_stopped_for_assessment'
    signature=common._signature(dict(phase=phase,terminal=terminal,error=error))
    if cursor.get('status')!=signature:
        common._text(exporter,prefix+'status',dict(phase=phase,terminal=terminal,error=error,
            operator_stopped=operator_stopped),step)
        cursor['status']=signature
    exporter.state.setdefault('fly_online_jobs',{})[label]=dict(source=str(root),family='direct_original_teacher_DAgger',
        phase=phase,producer_terminal=terminal,producer_file=source.name,student_decisions=costs.get('student_actions'),
        supervised_Adam=step,replay_sequences=row.get('replay_sequences'),
        body_updates_in_this_phase=costs.get('brain_updates_in_this_phase'),
        native_opening_learning_probability=launch.get('native_opening_learning_probability',0.),
        completed_training_contacts=cursor.get('latest_training_contacts'),
        actual_executor=dict(pid=launch.get('PID'),create_time=launch.get('create_time')),
        app_goal_remains_paused=row.get('app_goal_remains_paused',launch.get('app_goal_remains_paused',True)),
        current_pool_acknowledged_student_actions=native.get('acknowledged_student_actions'),
        evaluation=row.get('evaluation'),read_error=None,parser_patch='v15-guided-DAgger')
    return True
