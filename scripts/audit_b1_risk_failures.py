"""Replay a sealed risk-policy development trace; no fitting or native actions."""
from pathlib import Path
import argparse, hashlib, json, sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from fly_touhou.thrl_study_io import read_shared_json as read, utc_now
from fly_touhou.atomic_json_io import atomic_write_json as write


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run(a):
    import torch, psutil
    from fly_touhou.thrl_risk_policy import model_from_config
    from fly_touhou.thrl_action_risk_readout import adjust_movement_logits
    report = read(a.report)
    if report['status'] != 'completed' or report['native_seed_split'] != 'development':
        raise ValueError('Use completed development evidence for method selection')
    if read(a.report.parent/'cleanup.json') != dict(closed=True, close_errors=[]):
        raise ValueError('Native pool has not closed cleanly')
    pin = report['pins']['trained']; checkpoint = Path(pin['copy'])
    if sha(checkpoint) != pin['sha256']:
        raise ValueError('Checkpoint provenance differs')
    runtime = Path(read(Path(pin['source']).parent/'config.json')['source_runtime'])
    for name in ['thrl_model.py', 'thrl_object_transformer.py', 'thrl_policy_actions.py',
                 'thrl_risk_policy.py', 'thrl_action_risk_readout.py']:
        if sha(ROOT/'src/fly_touhou'/name) != sha(runtime/'src/fly_touhou'/name):
            raise ValueError('Inference implementation differs from executed policy: '+name)
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 4*1024**3:
        raise RuntimeError('CUDA with four GiB reserve required')
    torch.cuda.set_per_process_memory_fraction(.12); torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    if sys.platform == 'win32':
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = model_from_config(**payload['model_config']).cuda().eval()
    model.load_state_dict(payload['model'], strict=True)
    categories = {(x['policy'],x['game']): x for x in read(a.report.parent/'failure_trace_audit.json')['rows']}
    rows = []
    with torch.inference_mode():
        for episode in report['episode_results']:
            pointer = episode.get('failure_trace')
            if not pointer:
                continue
            receipt = read(Path(pointer['receipt'])); path = Path(pointer['arrays'])
            if sha(path) != pointer['sha256'] or receipt['checkpoint_sha256'] != pin['sha256']:
                raise ValueError('Actual trace provenance differs')
            with np.load(path, allow_pickle=False) as z:
                f = torch.as_tensor(z['features'][-1:].copy(), device='cuda')
                m = torch.as_tensor(z['last_preaction_maps'][None].copy(), device='cuda')
                o = torch.as_tensor(z['objects'][-1:].copy(), device='cuda')
                h = torch.as_tensor(z['hidden'][-1:].copy(), device='cuda')[None].contiguous()
                gold = z['probabilities'][-1].copy(); action = int(z['action'][-1])
            encoded = model.base._encode(f, m, o)
            state, _ = model.base.gru(encoded[None], h); state = state[0]
            base = model.base._policy_logits(state, f)
            risk = model.risk(torch.cat((encoded, state), -1))
            adjusted = adjust_movement_logits(base, risk, model.beta)
            probabilities = adjusted[:,:18].softmax(-1)[0].cpu().numpy()
            error = float(np.max(np.abs(probabilities-gold)))
            if error > 3e-5:
                raise ValueError('Actual executed probabilities failed to replay')
            score = risk.sigmoid()[0].cpu().numpy()
            geometry = categories[(episode['policy'],episode['game'])]
            row = dict(policy=episode['policy'],game=episode['game'],action=action,
                category=geometry['category'],replay_error=error,executed_risk_score=float(score[action]),
                actual_collision=bool(episode['first_collision_observed']),
                collision_classified_dangerous=bool(score[action]>.5),
                selected_probability=float(probabilities[action]),
                minimum_risk_score=float(score.min()),all_actions_classified_dangerous=bool((score>.5).all()),
                predicted_risk_scores=score.tolist(),base_movement_argmax=int(base[0,:18].argmax()),
                adjusted_movement_argmax=int(adjusted[0,:18].argmax()))
            margin = geometry.get('modeled_clearances_clipped64')
            if margin is not None:
                margin = np.asarray(margin); safe = margin>1
                row['modeled_buffered_alternative']=bool(safe.any())
                row['risk_scores_for_modeled_safe']=score[safe].tolist()
                row['argmax_modeled_clearance_by_beta']={str(beta):float(margin[int(adjust_movement_logits(base,risk,beta)[0,:18].argmax())])
                    for beta in (0.,4.,8.,16.)}
                row['risk_only_argmin_modeled_clearance']=float(margin[int(score.argmin())])
            rows.append(row)
    def summarize(selected):
        collisions=[x for x in selected if x['actual_collision']]
        buffered=[x for x in collisions if x.get('modeled_buffered_alternative')]
        return dict(cases=len(selected),actual_collisions=len(collisions),
            collision_danger_recall=sum(x['collision_classified_dangerous'] for x in collisions)/max(1,len(collisions)),
            collision_score_quantiles=np.quantile([x['executed_risk_score'] for x in collisions],[0,.25,.5,.75,1]).tolist(),
            all18_dangerous_cases=sum(x['all_actions_classified_dangerous'] for x in collisions),
            modeled_buffered_cases=len(buffered),
            buffered_risk_only_argmin_safe=sum(x['risk_only_argmin_modeled_clearance']>1 for x in buffered),
            buffered_beta_argmax_safe={str(beta):sum(x['argmax_modeled_clearance_by_beta'][str(beta)]>1 for x in buffered)
                                     for beta in (0.,4.,8.,16.)})
    result=dict(status='completed',utc=utc_now(),source_report=str(a.report),checkpoint_sha256=pin['sha256'],
        source_partition='development',rows=rows,summary=summarize(rows),
        by_policy={key:summarize([x for x in rows if x['policy']==key]) for key in ['trained_sampled','trained_greedy']},
        maximum_replay_error=max(x['replay_error'] for x in rows),native_instances=0,optimizer_updates=0,
        risk_scores_calibrated_probabilities=False,
        alternatives_and_beta_changes_native_verified=False,
        interpretation='Actual executed collisions are facts; alternative action geometry and beta logits are offline hypotheses, not future native performance.')
    a.output.mkdir(parents=True,exist_ok=False);write(a.output/'report.json',result)
    print(json.dumps({k:result[k] for k in ['summary','by_policy','maximum_replay_error']}))


if __name__ == '__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--report',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
