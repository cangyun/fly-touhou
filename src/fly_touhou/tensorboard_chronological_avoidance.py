"""Read-only chronology/original-guidance telemetry;no producer mutation."""
from pathlib import Path
import json
from . import tensorboard_continuous as common
from .thrl_dodge_diagnostics import _shared_binary


def export(exporter,label,root,*,launch,read_json):
    root=Path(root)
    source=next((root/name for name in ('failure.json','result.json','progress.json')
                 if (root/name).exists()),root/'launch.json')
    row=read_json(source);costs=row.get('costs') or {};step=costs.get('supervised_Adam',0)
    algorithm=read_json(root/'algorithm.json') if (root/'algorithm.json').exists() else {}
    prefix='fly_iterations/'+label+'/chronological_avoidance/'
    cursor=exporter.state.setdefault('chronological_avoidance_cursors',{}).setdefault(label,{})
    if cursor.get('scope_version')!='risk-first-phase-aware-v2':
        common._text(exporter,prefix+'scope',dict(training=algorithm.get('losses','original GRU guidance plus signed4f;not PPO'),
            input='raw maps/features/objects ->all8053 VP ->complete trainable brain ->1299 DN ->heads',
            history='chronological16-decision differentiation;carry at intermediate windows',
            proxies='recorded teacher/student states;not an autonomous gameplay score',
            risk_substeps='optional signed4f fitting on this same-forward detached DN;then original joint loss through sensory and brain',
            phase_order=algorithm.get('phase_order'),
            actor_readout_fitting='bounded readout-only phase AFTER raw-input whole-brain risk learning;current-body complete histories are replayed for the DN cache',
            goal_test='separate frozen original20 NoBomb;independent20 confirmation when first-stage clean>=8'),0)
        cursor['scope']=True;cursor['scope_version']='risk-first-phase-aware-v2'
    file=root/'learning.jsonl'
    if file.exists():
        with _shared_binary(file) as stream:
            stream.seek(cursor.get('offset',0))
            while True:
                before=stream.tell();line=stream.readline()
                if not line or not line.endswith(b'\n'):
                    cursor['offset']=before;break
                item=json.loads(line);n=item['step']
                if n>cursor.get('step',0):
                    for key in ('loss','safe_loss','rule_loss','future_loss','hit_loss','signed4f_loss','gradient_norm',
                                'risk_readout_substep_loss_first','risk_readout_substep_loss_last'):
                        common._scalar(exporter,prefix+'losses/'+key,item.get(key),n)
                    for key in ('joint_step','risk_readout_substeps_paid','risk_readout_learning_rate'):
                        common._scalar(exporter,prefix+'risk_substeps/'+key,item.get(key),n)
                    for key,value in item.items():
                        if key.startswith('proxy_'):
                            common._scalar(exporter,prefix+'recorded_state_proxies/'+item['stream']+'/'+key,value,n)
                    common._scalar(exporter,prefix+'stream_epochs/'+item['stream'],item.get('stream_epochs_completed'),n)
                    common._scalar(exporter,prefix+'elapsed_seconds',item.get('elapsed_seconds'),n)
                    cursor['step']=n
                cursor['offset']=stream.tell()
    for key,value in costs.items():common._scalar(exporter,prefix+'counts/'+key,value,step)
    actor_file=root/'actor-calibration-result.json'
    if actor_file.exists() and not cursor.get('actor_calibration_result'):
        actor_result=read_json(actor_file)
        for role in ('held_before','held_after'):
            for key,value in actor_result.get(role,{}).items():
                if isinstance(value,(int,float)):
                    common._scalar(exporter,prefix+'recorded_held_states/'+role+'/'+key,value,step)
        cursor['actor_calibration_result']=True
    folders=['development','confirmation20']+[p.name for p in root.glob('development_*') if p.is_dir()]+[
        p.name for p in root.glob('confirmation_*') if p.is_dir()]
    for folder in folders:
        file=root/folder/'report.json'
        if file.exists() and folder not in cursor.get('evaluations',[]):
            report=read_json(file)
            for key in ('episodes','first_stage_without_observed_contact_count','first_stage_without_observed_contact_fraction','actual_Bombs'):
                common._scalar(exporter,prefix+'frozen_NoBomb/'+folder+'/'+key,report.get(key),step)
            for key,value in report.get('observed_contact_frames',{}).items():
                if isinstance(value,(int,float)):
                    common._scalar(exporter,prefix+'frozen_NoBomb/'+folder+'/contact_frames/'+key,value,step)
            cursor.setdefault('evaluations',[]).append(folder)
    phase=row.get('phase',row.get('status'));terminal=source.name in ('failure.json','result.json')
    exporter.state.setdefault('fly_online_jobs',{})[label]=dict(source=str(root),
        family='chronological_risk_first_avoidance' if algorithm.get('risk_only_pretraining') else 'chronological_original_GRU_avoidance',phase=phase,producer_terminal=terminal,
        producer_file=source.name,supervised_Adam=step,PPO_Adam=0,trainable_brain=True,
        brain_updates_active_now=phase in ('chronological_physical_risk_pretraining','chronological_original_GRU_supervision'),
        risk_only_Adam=costs.get('risk_only_Adam'),actor_readout_Adam=costs.get('actor_readout_Adam'),
        CUDA_started=launch.get('CUDA_started'),CUDA_device=launch.get('CUDA_device'),
        teacher_learning_rows=costs.get('teacher_learning_rows'),own_state_learning_rows=costs.get('own_state_learning_rows'),
        actual_executor=dict(pid=launch.get('PID'),create_time=launch.get('create_time')),
        evaluation=row.get('evaluation'),read_error=None,parser_patch='chronological-avoidance-v1')
    return True
