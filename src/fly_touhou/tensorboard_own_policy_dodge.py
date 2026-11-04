"""Once-per-committed-rollout NumPy reduction for the existing TB bridge.

No model,checkpoint,Torch,neural replay,game control or independent poller.
Only required observation/head/task arrays are opened;never maps or activity.
"""
from __future__ import annotations
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile
from .atomic_json_io import atomic_write_json
from .thrl_dodge_diagnostics import _shared_binary

SCHEMA='continuous-own-policy-dodge-TB-derived-v1'
PROTOCOL='signed4f-trusted-normal-unprotected-clamp80-T1-v1'
FIELDS=('features','legal_actions','actions','deathbomb_mask','raw_logits','composed_logits',
    'predicted_costs','action_probs','task_margins','task_trusted','task_outcomes','task_unprotected_mask')
SCOPE=('Own-policy T1 behavior at each rollout theta;not fixed held or formal clear performance. '
    'Dense trusted labels prioritize observed contact/closing distance;unlabeled is unknown. '
    'Selected-action outcome contradictions are vetoed by the producer,so agreement is not independent validation. '
    'Geometric margins do not establish all unexecuted native outcomes. '
    'Outcome1 is observed contact,not necessarily Miss. Bomb/state4/window requests are not confirmed native DB. '
    'One CPU reduction per immutable paid update;no model or extra native action.')


def _div(a,b):return float(a/b) if b else None


def _softmax(scores,legal):
    import numpy as np
    x=np.where(legal,np.clip(scores.astype(np.float64),-80,80),-np.inf)
    x=np.exp(x-x.max(-1,keepdims=True));return x/x.sum(-1,keepdims=True)


def _metrics(d):
    import numpy as np
    actions=d['actions'];shape=actions.shape;f=d['features']
    if (shape!=(256,30) or f.shape!=shape+(25,) or d['task_margins'].shape!=shape+(3,18)
            or np.any((actions<0)|(actions>=19))):raise ValueError('Logical256x30 source19/V5 required')
    normal=(f[...,21]>.5)&(f[...,20]<=0)&(f[...,5]<=0)
    eligible=normal&d['task_unprotected_mask']&d['task_trusted'][...,0]
    target=d['task_margins'][...,0,:];bad=target<=0;safe=target>1
    mixed=eligible&bad.any(-1)&safe.any(-1);cost=d['predicted_costs']
    if not np.isfinite(cost).all() or np.any(cost<0):raise ValueError('Finite positive neural costs required')
    threshold=np.float32(np.logaddexp(0.,1.));predbad=cost>=threshold
    danger=eligible[...,None]&bad;safeq=eligible[...,None]&safe
    risk=dict(danger_queries=int(danger.sum()),danger_recall_hits=int((danger&predbad).sum()),
        safe_queries=int(safeq.sum()),safe_false_positive=int((safeq&predbad).sum()))
    risk.update(danger_recall=_div(risk['danger_recall_hits'],risk['danger_queries']),
        safe_FP=_div(risk['safe_false_positive'],risk['safe_queries']))
    mixed_result=dict(rows=int(mixed.sum()))
    for name,key in (('raw','raw_logits'),('composed','composed_logits')):
        probability=_softmax(d[key][...,:18],d['legal_actions'][...,:18])
        mixed_result[name]=dict(conditional18_danger_mass_mean=
            float((probability*bad).sum(-1)[mixed].mean()) if mixed.any() else None)
    chosen=np.take_along_axis(target,np.minimum(actions,17)[...,None],-1)[...,0]
    selected=mixed&(actions<18)
    mixed_result.update(actual_sampled_movement_rows=int(selected.sum()),
        actual_sampled_Bomb_rows=int((mixed&(actions==18)).sum()),
        sampled_movement_proxy_danger=int((selected&(chosen<=0)).sum()),
        sampled_movement_proxy_danger_fraction=_div((selected&(chosen<=0)).sum(),selected.sum()))
    before_state4=f[...,24]>.5;bomb=actions==18;db=d['deathbomb_mask'];outcome=d['task_outcomes']
    return dict(rows=int(actions.size),normal_unprotected_rows=int(normal.sum()),
        eligible_rows=int(eligible.sum()),mixed_rows=int(mixed.sum()),risk=risk,mixed=mixed_result,
        actual_outcomes_all_rows={str(v):int((outcome==v).sum()) for v in (-1,0,1)},
        actual_observed_contact_rows_without_eligible_geometry=int(((outcome==1)&~eligible).sum()),
        Bomb=dict(sampled_requests=int(bomb.sum()),mean_stored_probability=float(d['action_probs'][...,18].mean()),
            measured_DB_opportunity_rows=int(db.sum()),requests_in_measured_DB_window=int((bomb&db).sum()),
            requests_in_before_state4=int((bomb&before_state4).sum()),
            requests_in_normal_unprotected=int((bomb&normal).sum()),
            other_requests=int((bomb&~before_state4&~normal).sum()),per_row_native_confirmed_DB=None),
        threshold_FP32=float(threshold))


def derive_paid_rollout(root,number,update,*,cache_root,read_json):
    """A complete update is the paid boundary;each immutable NPZ is reduced once."""
    root=Path(root);identity=hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:16]
    folder=Path(cache_root)/identity;path=folder/f'rollout-{number-1:06}.json'
    costs=update.get('costs') or {};learning=update.get('learning') or {};x=costs.get('student_actions')
    paid=(type(x) is int and x>0 and type(learning.get('optimizer_updates')) is int
        and learning['optimizer_updates']>0 and learning.get('optimizer_transaction_complete') is True
        and costs.get('completed_rollouts')==number)
    if not paid:return dict(status='not_a_paid_completed_update',learning_point_written=False)
    if path.is_file():
        cached=read_json(path)
        if cached.get('schema')!=SCHEMA or cached.get('source')!=str(root.resolve()) or cached.get('x_student_actions')!=x:
            raise ValueError('Derived cache belongs to another source/paid boundary')
        return cached
    begun=time.monotonic();sidepath=root/'traces'/f'rollout-{number-1:03}.json'
    arraypath=sidepath.with_suffix('.npz')
    if not sidepath.is_file() or not arraypath.is_file():return dict(status='missing_paired_trace',learning_point_written=False)
    record=dict(schema=SCHEMA,protocol=PROTOCOL,source=str(root.resolve()),update_number=number,
        rollout=number-1,x_student_actions=x,paid_optimizer_updates=learning['optimizer_updates'],
        cumulative_PPO_Adam=costs.get('PPO_Adam'),scope=SCOPE,NPZ_decompressions=0)
    try:
        side=read_json(sidepath);meta=side.get('metadata') or {}
        if (meta.get('policy_version')!=number-1 or meta.get('synthetic_fixture') is not False
                or meta.get('teacher_actions_executed')!=0 or meta.get('safety_takeovers')!=0):
            raise ValueError('Missing actual own-policy source metadata')
        with _shared_binary(arraypath) as f:
            with zipfile.ZipFile(f) as archive:
                available={Path(n).stem for n in archive.namelist()}
        missing=sorted(set(FIELDS)-available)
        if missing:
            record.update(status='unavailable_missing_fields',missing_fields=missing,learning_point_written=False)
        else:
            import numpy as np
            with _shared_binary(arraypath) as f:blob=f.read()
            digest=hashlib.sha256(blob).hexdigest()
            if digest!=side.get('trace_sha256'):raise ValueError('Immutable rollout/sidecar SHA differs')
            with np.load(io.BytesIO(blob),allow_pickle=False) as arrays:d={k:arrays[k] for k in FIELDS}
            record.update(status='completed',metrics=_metrics(d),trace_sha256=digest,trace_bytes=len(blob),
                NPZ_decompressions=1,learning_point_written=True)
            del d,blob
    except (OSError,ValueError,KeyError,zipfile.BadZipFile) as error:
        record.update(status='diagnostic_unavailable',error=str(error)[:500],learning_point_written=False)
    record['CPU_reduction_seconds']=time.monotonic()-begun
    folder.mkdir(parents=True,exist_ok=True);atomic_write_json(path,record)
    return record


def export_paid_dodge(exporter,prefix,root,number,update,*,read_json,cursor):
    """Runs inside the existing bridge tick;never starts a worker or poller."""
    try:
        row=derive_paid_rollout(root,number,update,cache_root=exporter.cursor_path.parent/'tb-derived-dodge',read_json=read_json)
    except (OSError,ValueError,KeyError) as error:
        row=dict(status='diagnostic_unavailable',error=str(error)[:500],learning_point_written=False)
    cursor['own_policy_dodge_status']=dict(update_number=number,status=row['status'])
    if row['status']!='completed':return
    m=row['metrics'];step=row['x_student_actions'];tag=prefix+'own_policy_dodge/'
    scalars=dict(eligible_rows=m['eligible_rows'],mixed_rows=m['mixed_rows'],
        risk_recall=m['risk']['danger_recall'],risk_safe_FP=m['risk']['safe_FP'],
        risk_danger_queries=m['risk']['danger_queries'],risk_safe_queries=m['risk']['safe_queries'],
        mixed_raw_conditional18_danger=m['mixed']['raw']['conditional18_danger_mass_mean'],
        mixed_composed_conditional18_danger=m['mixed']['composed']['conditional18_danger_mass_mean'],
        mixed_selected_move_proxy_danger_fraction=m['mixed']['sampled_movement_proxy_danger_fraction'],
        mixed_selected_move_rows=m['mixed']['actual_sampled_movement_rows'],
        mixed_selected_move_proxy_danger_rows=m['mixed']['sampled_movement_proxy_danger'],
        observed_contacts=m['actual_outcomes_all_rows']['1'],observed_clean=m['actual_outcomes_all_rows']['0'],
        unknown_outcomes=m['actual_outcomes_all_rows']['-1'],Bomb_requests=m['Bomb']['sampled_requests'],
        DB_opportunity_rows=m['Bomb']['measured_DB_opportunity_rows'],
        Bomb_requests_in_DB_window=m['Bomb']['requests_in_measured_DB_window'],
        Bomb_requests_in_state4=m['Bomb']['requests_in_before_state4'],
        Bomb_requests_in_normal_unprotected=m['Bomb']['requests_in_normal_unprotected'])
    for name,value in scalars.items():
        if value is not None:exporter.scalar(tag+name,value,step)
    if not cursor.get('own_policy_dodge_scope'):
        if hasattr(exporter,'text'):exporter.text(tag+'scope',SCOPE,0)
        cursor['own_policy_dodge_scope']=True
