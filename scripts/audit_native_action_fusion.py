"""Read-only action-fusion alternatives on already recorded native states.

These are same-state geometric proxies, never alternate gameplay results.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--native-report',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    import numpy as np
    import torch
    from fly_touhou.thrl_study_io import read_shared_json as read
    from fly_touhou.thrl_durable_records import write_json
    from fly_touhou.geometry_attention_teaching import per_object_first4
    from fly_touhou.fly_dn_cost_native_policy import compose_from_active_mask,source_policy_logits
    torch.set_num_threads(2)
    report=read(a.native_report);rows=[]
    for episode in report['episode_results']:
        with np.load(episode['first_contact_tail']['path'],allow_pickle=False) as d:
            for t in range(len(d['features'])):
                label,_=per_object_first4(d['features'][t],d['objects'][t],d['maps'][t])
                if label is None:continue
                individual,valid=label;margin=individual[:,valid].min(-1)
                if not (margin<=0).any() or not (margin>1).any():continue
                rows.append(dict(game=episode['evaluation_game'],f=d['features'][t],margin=margin,
                    raw=d['raw_logits'][t],cost=d['predicted_costs'][t],actual=d['noBomb_policy_logits'][t,:18],
                    action=int(d['action'][t]),final=t==len(d['features'])-1))
    if not rows:raise ValueError('No known mixed point-geometry states')
    raw=torch.from_numpy(np.stack([r['raw'] for r in rows]));cost=torch.from_numpy(np.stack([r['cost'] for r in rows]))
    f=torch.from_numpy(np.stack([r['f'] for r in rows]));margin=np.stack([r['margin'] for r in rows])
    final=np.asarray([r['final'] for r in rows]);ids=np.arange(len(rows));results={}

    def summarize(actions,mask):
        indices=ids[mask];selected=margin[indices,actions[mask]]
        return dict(rows=len(indices),selected_proxy_safe=int((selected>1).sum()),
            selected_proxy_unsafe=int((selected<=0).sum()),selected_border=int(((selected>0)&(selected<=1)).sum()))

    def add(name,actions):
        results[name]=dict(all=summarize(actions,np.ones(len(rows),bool)),final_contacts=summarize(actions,final))

    actual=np.stack([r['actual'] for r in rows]).argmax(-1);add('actual_composed',actual)
    add('risk_only',cost.numpy().argmin(-1))
    max_beta8_error=None;beta8_same_actions=None
    with torch.no_grad():
        for beta in (0.,1.,2.,4.,6.,8.):
            composed=compose_from_active_mask(raw,cost,torch.ones(len(rows),dtype=torch.bool),beta=beta)
            logits=source_policy_logits(composed,f)[:,:18].numpy();actions=logits.argmax(-1)
            add('beta'+str(int(beta)),actions)
            if beta==8:
                max_beta8_error=float(np.abs(logits-np.stack([r['actual'] for r in rows])).max())
                beta8_same_actions=int((actions==actual).sum())
    write_json(a.output,dict(schema='same-state-recorded-native-action-fusion-audit-v1',
        native_report=str(a.native_report),source_checkpoint_SHA256=report['checkpoint_sha256'],
        known_mixed_rows=len(rows),known_final_contact_rows=int(final.sum()),
        logged_beta8_reconstruction_max_error=max_beta8_error,beta8_same_actions=beta8_same_actions,
        results=results,policy_or_weights_modified=False,native_or_CUDA_started=False,
        scope='Unchosen actions have geometric labels only;no alternate native survival or validation score is measured.',
        selection_bias='States were visited by the composed policy and final rows are conditioned on its failures;these counts cannot establish that an alternate readout is better.'))
    print(json.dumps(dict(known_mixed_rows=len(rows),beta8_same_actions=beta8_same_actions,
        logged_beta8_reconstruction_max_error=max_beta8_error,results=results)))


if __name__=='__main__':main()
