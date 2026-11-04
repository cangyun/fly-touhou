"""Reproduce native continuous states and quantify zero-state T16 burn-in error.

Uses recorded actual encoded384 and full continuous activity. No optimizer,
teacher query, native action or encoder refitting occurs here.
"""
from __future__ import annotations
import numpy as np
import torch
from .continuous_connectome import ContinuousConnectomeState


@torch.no_grad()
def audit_history_reconstruction(model,trace_path,*,check_stop=lambda:None):
    with np.load(trace_path,allow_pickle=False) as source:
        d={k:source[k] for k in ('encoded384','brain_activity_following','DN_activity',
            'noBomb_policy_logits','predicted_costs','features','action','before_episode_frames')}
    n=len(d['features'])
    if n<32 or int(d['before_episode_frames'][0])!=0:
        raise ValueError('At least32 consecutive actual observations from the original opening required')
    starts=sorted({s for s in (0,8,16,24,n-32) if 0<=s<=n-32})
    if len(starts)>8:raise ValueError('At most eight paired reference/zero windows in canonical B16')
    device=model.device;h=torch.zeros((16,model.core.n_neurons),device=device)
    for w,start in enumerate(starts):
        if start:h[2*w]=torch.as_tensor(d['brain_activity_following'][start-1],device=device)
    state=ContinuousConnectomeState(h);records=[]
    for t in range(32):
        check_stop();encoded=torch.zeros((16,384),device=device);features=torch.zeros((16,25),device=device)
        indices=[start+t for start in starts]
        selected=torch.as_tensor(d['encoded384'][indices],device=device)
        measured=torch.as_tensor(d['features'][indices],device=device)
        encoded[:2*len(starts)]=selected.repeat_interleave(2,dim=0)
        features[:2*len(starts)]=measured.repeat_interleave(2,dim=0)
        DN,state=model.core(encoded,state)
        raw=model.actor(DN);cost=model.risk_head(DN)
        logits=model.policy_logits(model.compose(raw,cost,features),features)[:,:18]
        expected_DN=torch.as_tensor(d['DN_activity'][indices],device=device)
        expected_h=torch.as_tensor(d['brain_activity_following'][indices],device=device)
        expected_logits=torch.as_tensor(d['noBomb_policy_logits'][indices,:18],device=device)
        denominator=expected_DN.norm(dim=-1).clamp_min(1e-8)
        for w,start in enumerate(starts):
            exact=2*w;zero=exact+1;actual_action=int(d['action'][indices[w]])
            records.append(dict(start=start,offset=t,learning_row=t>=16,native_frame=int(d['before_episode_frames'][indices[w]]),
                reference_DN_relative_L2=float((DN[exact]-expected_DN[w]).norm()/denominator[w]),
                zero_DN_relative_L2=float((DN[zero]-expected_DN[w]).norm()/denominator[w]),
                reference_full_activity_relative_L2=float((state.activity[exact]-expected_h[w]).norm()/expected_h[w].norm().clamp_min(1e-8)),
                zero_full_activity_relative_L2=float((state.activity[zero]-expected_h[w]).norm()/expected_h[w].norm().clamp_min(1e-8)),
                reference_logits_RMSE=float((logits[exact]-expected_logits[w]).square().mean().sqrt()),
                zero_logits_RMSE=float((logits[zero]-expected_logits[w]).square().mean().sqrt()),
                actual_action=actual_action,reference_action=int(logits[exact].argmax()),zero_action=int(logits[zero].argmax())))
    learning=[r for r in records if r['learning_row']]
    def summarize(rows):
        return dict(rows=len(rows),reference_DN_relative_L2_max=max(r['reference_DN_relative_L2'] for r in rows),
            reference_action_agreement=float(np.mean([r['reference_action']==r['actual_action'] for r in rows])),
            zero_DN_relative_L2_mean=float(np.mean([r['zero_DN_relative_L2'] for r in rows])),
            zero_DN_relative_L2_max=max(r['zero_DN_relative_L2'] for r in rows),
            zero_action_agreement=float(np.mean([r['zero_action']==r['actual_action'] for r in rows])),
            zero_logits_RMSE_mean=float(np.mean([r['zero_logits_RMSE'] for r in rows])))
    summary=summarize(learning)
    if summary['reference_DN_relative_L2_max']>1e-3 or summary['reference_action_agreement']<.99:
        interpretation='Reference replay does not match native;resolve numerical/path difference before attributing error to burn-in.'
    elif summary['zero_DN_relative_L2_mean']>.05 and summary['zero_action_agreement']<.9:
        interpretation='Zero-initialized16-observation burn-in is materially different from actual native history in this prefix.'
    elif summary['zero_DN_relative_L2_mean']<.01 and summary['zero_action_agreement']>=.99:
        interpretation='No material T16 burn-in mismatch found in this native prefix.'
    else:interpretation='Small or mixed T16 burn-in discrepancy;not established as a unique failure cause.'
    return dict(schema='actual-native-continuous-history-reconstruction-v1',trace=str(trace_path),
        recorded_native_rows=n,starts=starts,canonical_neural_batch=16,burn_in=16,learning_rows=16,
        summary=summary,windows=[dict(start=s,summary=summarize([r for r in learning if r['start']==s])) for s in starts],
        records=records,interpretation=interpretation,optimizer_updates=0,native_actions_during_replay=0,
        encoder_recomputed=False,actual_encoded384_replayed=True,full_neural_state_not_evaluator_cursor=True,
        scope='Computational train/inference-history comparison on a recorded prefix;not another policy performance cohort.')
