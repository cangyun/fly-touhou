"""Read-only SNN telemetry, with no Torch/model/native imports."""
import json
from pathlib import Path
from . import tensorboard_continuous as common
from .thrl_dodge_diagnostics import _shared_binary


def export(exporter,label,root,*,launch,read_json):
    root=Path(root)
    source=next((root/x for x in ('failure.json','result.json','progress.json') if (root/x).exists()),root/'launch.json')
    current=read_json(source);costs=current.get('costs',{});step=costs.get('joint_Adam',0)
    prefix='fly_iterations/'+label+'/spiking_avoidance/'
    cursor=exporter.state.setdefault('spiking_engineered_cursors',{}).setdefault(label,{})
    if not cursor.get('scope') and (root/'algorithm.json').exists():
        common._text(exporter,prefix+'scope',read_json(root/'algorithm.json'),0);cursor['scope']=True
    path=root/'learning.jsonl'
    if path.exists():
        with _shared_binary(path) as stream:
            stream.seek(cursor.get('offset',0))
            while True:
                before=stream.tell();line=stream.readline()
                if not line or not line.endswith(b'\n'):cursor['offset']=before;break
                item=json.loads(line)
                if item['step']>cursor.get('step',0):
                    for key,value in item.items():
                        if key not in ('step','stream') and isinstance(value,(int,float)):
                            common._scalar(exporter,prefix+item['stream']+'/'+key,value,item['step'])
                    cursor['step']=item['step']
                cursor['offset']=stream.tell()
    for key,value in costs.items():common._scalar(exporter,prefix+'counts/'+key,value,step)
    evaluations=root/'evaluations.json'
    if evaluations.exists():
        for node in read_json(evaluations)['nodes']:
            n=node['updates']
            if n in cursor.get('evaluations',[]):continue
            for role in ('development','confirmation'):
                report=node.get(role)
                if report:
                    common._scalar(exporter,prefix+'native/'+role+'/first_stage_clean',report['first_stage_without_observed_contact_count'],n)
                    for key,value in report['observed_contact_frames'].items():
                        if isinstance(value,(int,float)):common._scalar(exporter,prefix+'native/'+role+'/'+key,value,n)
            cursor.setdefault('evaluations',[]).append(n)
    phase=current.get('phase',current.get('status',launch.get('status')))
    exporter.state.setdefault('fly_online_jobs',{})[label]=dict(source=str(root),family='fullbrain_trainable_spiking',phase=phase,
        producer_terminal=source.name in ('failure.json','result.json'),producer_file=source.name,supervised_Adam=step,
        PPO_Adam=costs.get('PPO_Adam',0),trainable_brain=True,brain_updates_active_now=phase=='joint_spiking_supervision',
        CUDA_started=launch.get('CUDA_started'),CUDA_device=launch.get('CUDA_device'),
        actual_executor=dict(pid=launch.get('PID'),create_time=launch.get('create_time')),
        evaluation=current.get('evaluation'),learning=current.get('learning'),read_error=None)
    return True


def export_acquisition(exporter,label,root,*,launch,read_json):
    """Fresh student-controlled training trajectories; never report them as tests."""
    root=Path(root)
    source=next((root/x for x in ('failure.json','result.json','progress.json') if (root/x).exists()),root/'launch.json')
    current=read_json(source);step=current.get('student_actions',0)
    prefix='fly_iterations/'+label+'/SNN_current_student_DAgger/'
    common._scalar(exporter,prefix+'student_actions',step,step)
    common._scalar(exporter,prefix+'teacher_actions',0,step)
    common._scalar(exporter,prefix+'optimizer_updates',0,step)
    exporter.state.setdefault('fly_online_jobs',{})[label]=dict(source=str(root),family='fullbrain_spiking_student_state_acquisition',
        phase=current.get('phase',current.get('status','initializing')),producer_terminal=source.name in ('failure.json','result.json'),
        producer_file=source.name,student_actions=step,PPO_Adam=0,brain_updates_active_now=False,
        CUDA_started=launch.get('CUDA_started'),CUDA_device=launch.get('CUDA_device'),
        actual_executor=dict(pid=launch.get('PID'),create_time=launch.get('create_time')),read_error=None,
        scope='student training-state acquisition,not frozen test;teacher annotates,student acts')
    return True
