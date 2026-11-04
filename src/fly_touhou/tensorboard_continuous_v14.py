"""Prepared optional risk-stage extension;does not install/reload a bridge.

An explicitly maintained bridge may import this module instead of v13. Other
families delegate unchanged;this new phase never travels through PPO tags.
"""
from __future__ import annotations
from pathlib import Path
import re

from . import tensorboard_continuous as v13
from .continuous_risk_optimizer_transition import validate_risk_optimizer_transition

ENTRY_SCHEMA='continuous-full-sensory-risk-only-NoBomb-stage-entry-v1'
UPDATE_SCHEMA='continuous-full-sensory-risk-only-signed4f-update-v1'
PATCH='continuous-full-sensory-risk-only-v14-prepared'
current_algorithm_jobs=v13.current_algorithm_jobs


def export_continuous_connectome(exporter,label,root,*,read_json):
    root=Path(root)
    launch=v13._read(root/'launch.json',read_json) if (root/'launch.json').is_file() else {}
    paths=[root/name for name in ('failure.json','result.json','progress.json') if (root/name).is_file()]
    source=paths[0] if paths else root/'launch.json'
    row=v13._read(source,read_json) if paths else launch
    schema=launch.get('schema') or row.get('schema')
    if schema!=ENTRY_SCHEMA:
        return v13.export_continuous_connectome(exporter,label,root,read_json=read_json)
    if row.get('schema') not in (None,ENTRY_SCHEMA):raise ValueError('Risk-stage schema conflict')
    terminal=source.name if source.name in ('failure.json','result.json') else None
    phase=row.get('phase') or ('failed_or_stopped' if terminal=='failure.json' else row.get('status'))
    costs=row.get('costs') or {};step=v13._count(costs.get('supervised_Adam')) or 0
    cursor=exporter.state.setdefault('continuous_risk_only_cursors',{}).setdefault(label,{})
    if cursor.get('source') not in (None,str(root.resolve())):raise ValueError('Risk cursor source changed')
    cursor['source']=str(root.resolve())
    prefix='fly_iterations/'+label+'/continuous_connectome/'
    p=prefix+'NoBomb_risk_only_supervision/'
    if not cursor.get('scope'):
        v13._text(exporter,p+'scope','Prepared risk-only phase:raw-sensory/K8/T16 physical signed4f loss coefficient1. '
            'Student acts;physical4f geometry targets,D95 model queries0. PPO/value/guidance updates0;actor and three critic weights/Adam retained. '
            'Changing-policy collection and frozen NoBomb20 have separate axes;not Bomb-allowed completion success.',0)
        cursor['scope']=True
    for key in ('supervised_Adam','student_actions','completed_rollouts'):
        v13._scalar(exporter,p+'counts/'+key,costs.get(key),step)
    for key in ('PPO_updates','value_updates','teacher_actions_executed'):
        v13._scalar(exporter,p+'not_PPO_or_value/'+key,row.get(key),step)
    v13._scalar(exporter,p+'time/elapsed_seconds',row.get('elapsed_seconds'),step)
    v13._scalar(exporter,p+'target/student_decisions',launch.get('student_decisions'),step)
    v13._scalar(exporter,p+'target/supervised_Adam_max',launch.get('supervised_Adam_max'),step)
    algorithm=row.get('algorithm') or launch.get('algorithm') or {}
    actual=row.get('actual_transition') or launch.get('actual_transition')
    if actual is not None:
        transition=validate_risk_optimizer_transition(actual,expected_mode=algorithm.get('optimizer_mode'))
        if cursor.get('optimizer_transition')!=v13._signature(transition):
            v13._text(exporter,p+'optimizer_transition',transition,step)
            cursor['optimizer_transition']=v13._signature(transition)
        for name,group in transition['groups'].items():
            for key in ('source_lr','target_lr','source_state_count','target_state_count'):
                v13._scalar(exporter,p+'optimizer_transition/'+name+'/'+key,group[key],step)
    v13._scalar(exporter,p+'settings/max_grad_norm',algorithm.get('max_grad_norm',launch.get('max_grad_norm')),step)
    for key,value in (launch.get('perception_branches') or {}).items():
        if type(value) is bool:v13._scalar(exporter,p+'perception_trainable/'+key,int(value),step)
    for name in ('actor_critic_parameters_and_Adam_exact',):
        if type(row.get(name)) is bool:v13._scalar(exporter,p+'identity/'+name,int(row[name]),step)
    for path in sorted((root/'traces').glob('update-*.json')):
        match=re.fullmatch(r'update-(\d+)',path.stem)
        if not match:continue
        number=int(match[1])
        if number<=cursor.get('update',0):continue
        update=v13._read(path,read_json);learning=update.get('learning') or {};paid=update.get('costs') or {}
        adam=v13._count(paid.get('supervised_Adam'));n=v13._count(learning.get('optimizer_updates'))
        if (number>(v13._count(costs.get('completed_rollouts')) or 0)
                or learning.get('schema')!=UPDATE_SCHEMA or adam is None or n is None
                or learning.get('optimizer_transaction_complete') is not True):continue
        if n>0:
            for key,value in v13._mean_terms(learning.get('terms') or []).items():
                v13._scalar(exporter,p+'minibatch_mean/'+key,value,adam)
            for field in ('first_gradient_groups','first_perception_gradient_branches'):
                for key,value in v13._flatten_terms(learning.get(field) or {}).items():
                    v13._scalar(exporter,p+field+'/'+key,value,adam)
        for key in ('optimizer_updates','optimizer_step_attempts','backward_passes','empty_label_minibatches',
                'forward_real_rows','trained_real_rows','loss_real_rows','epochs_completed',
                'actor_gradient_tensors','critic_gradient_tensors'):
            v13._scalar(exporter,p+'update/'+key,learning.get(key),adam)
        cursor['update']=number
    status=dict(phase=phase,terminal=terminal,error=row.get('error'),supervised_Adam=costs.get('supervised_Adam'))
    identity=v13._signature(status)
    if cursor.get('status')!=identity:
        v13._text(exporter,p+'status',status,step);cursor['status']=identity
    frozen=row.get('development') if terminal=='result.json' else row.get('evaluation')
    native={}
    if isinstance(frozen,dict):
        wrapper=dict(row,evaluation=frozen) if terminal=='result.json' else frozen
        native=v13._no_bomb_native(exporter,prefix,wrapper,launch,terminal,cursor)
    exporter.state.setdefault('fly_online_jobs',{})[label]=dict(source=str(root),
        family='NoBomb_risk_only_supervision',phase=phase,producer_file=source.name,
        producer_schema=ENTRY_SCHEMA,producer_terminal=terminal is not None,
        supervised_Adam=v13._count(costs.get('supervised_Adam')),
        PPO_updates=row.get('PPO_updates'),value_updates=row.get('value_updates'),
        student_decisions=v13._count(costs.get('student_actions')),
        actual_executor=dict(pid=launch.get('PID'),create_time=launch.get('create_time')),
        perception_branches=launch.get('perception_branches'),NoBomb_progress=native,
        actor_critic_parameters_and_Adam_exact=row.get('actor_critic_parameters_and_Adam_exact'),
        parser_patch=PATCH,read_error=None)
    return True
