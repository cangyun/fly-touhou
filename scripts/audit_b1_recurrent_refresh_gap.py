"""Compare training's bounded GRU refresh with actual frozen episode context.

Recorded no-Bomb endpoints include both collisions and right-censored games.
No training, fresh native worlds, invented history or zero-memory attribution.
"""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.thrl_study_io import read_shared_json as read,utc_now
from fly_touhou.atomic_json_io import atomic_write_json as write
from fly_touhou.thrl_native_trace_replay import RecordedAvoidanceCase


def run(args):
    import torch,psutil
    from fly_touhou.thrl_pair_policy import model_from_config
    if args.output.exists():raise FileExistsError(args.output)
    report=read(args.report)
    if report.get('status')!='completed' or read(args.report.parent/'cleanup.json')!=dict(closed=True,close_errors=[]):
        raise ValueError('Require closed completed observations')
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0]<2*1024**3:raise RuntimeError('CUDA diagnostic reserve unavailable')
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    pin=report['pins']['trained'];checkpoint=Path(pin['copy'])
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=pin['sha256']:raise ValueError('Checkpoint identity differs')
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=model_from_config(**payload['model_config']).cuda().eval().requires_grad_(False)
    model.load_state_dict(payload['model'],strict=True);del payload
    rows=[];replay_errors=[];selected=[r for r in report['episode_results'] if r['policy'] in ('trained_sampled','trained_greedy')]
    with torch.inference_mode():
        for episode in selected:
            case=RecordedAvoidanceCase(args.report,episode);z=case.arrays
            if case.receipt['checkpoint_sha256']!=pin['sha256']:raise ValueError('Trace belongs to different weights')
            t=case.tail_count-1
            if t<96:raise ValueError('Predeclared96-decision burn-in needs enough actual paired history')
            encoded=[]
            for lo in range(0,len(z['features']),32):
                b={k:torch.from_numpy(z[k][lo:lo+32]).cuda() for k in ('features','objects','maps')}
                encoded.append(model._encode(b['features'],b['maps'],b['objects']))
            encoded=torch.cat(encoded)
            f=torch.from_numpy(z['features'][t:t+1]).cuda();o=torch.from_numpy(z['objects'][t:t+1]).cuda()
            actual_h=torch.from_numpy(z['hidden'][t])[None,None].cuda()
            def evaluate(hidden):
                out,_=model.gru(encoded[t:t+1,None],hidden)
                out=out.squeeze(0);base=model.base._policy_logits(out,f)
                adjusted=model._adjust(base,f,o)
                return base[0,:18].softmax(-1),adjusted[0,:18].softmax(-1),adjusted[0].softmax(-1),model.base._critics(out)[0]
            actual=evaluate(actual_h)
            error=float(np.abs(actual[1].cpu().numpy()-z['probabilities'][t]).max())
            if error>3e-5:raise ValueError('Recorded full-history probability replay differs: '+str(error))
            replay_errors.append(error)
            for length in (8,16,32,64,96):
                _,warm=model.gru(encoded[t-length:t,None],torch.zeros_like(actual_h))
                changed=evaluate(warm)
                tv=lambda i:float(.5*(changed[i]-actual[i]).abs().sum())
                rows.append(dict(policy=episode['policy'],game=episode['game'],
                    observed_collision=episode['first_collision_observed'],right_censored=episode['right_censored_at_cap'],
                    history_decisions=length,hidden_max_abs_difference=float((warm-actual_h).abs().max()),
                    hidden_L2_difference=float((warm-actual_h).norm()),body_movement_TV=tv(0),
                    adjusted_movement_TV=tv(1),full19_TV=tv(2),
                    adjusted_movement_argmax_changed=bool(changed[1].argmax()!=actual[1].argmax()),
                    full19_argmax_changed=bool(changed[2].argmax()!=actual[2].argmax()),
                    value_absolute_difference=(changed[3]-actual[3]).abs().cpu().tolist(),
                    source_trace_sha256=case.source_trace_sha256))
    summary={}
    for length in (8,16,32,64,96):
        subset=[r for r in rows if r['history_decisions']==length]
        summary[str(length)]=dict(states=len(subset),
            mean_full19_TV=float(np.mean([r['full19_TV'] for r in subset])),
            maximum_full19_TV=max(r['full19_TV'] for r in subset),
            mean_adjusted_movement_TV=float(np.mean([r['adjusted_movement_TV'] for r in subset])),
            maximum_adjusted_movement_TV=max(r['adjusted_movement_TV'] for r in subset),
            full19_argmax_changed=sum(r['full19_argmax_changed'] for r in subset),
            maximum_hidden_abs_difference=max(r['hidden_max_abs_difference'] for r in subset))
    args.output.mkdir(parents=True)
    result=dict(schema='B1-actual-episode-vs-bounded-recurrent-refresh-gap-v1',utc=utc_now(),
        source_report=str(args.report),source_report_sha256=hashlib.sha256(args.report.read_bytes()).hexdigest(),
        checkpoint_sha256=pin['sha256'],script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        selected_games=len(selected),observed_collision_games=sum(r['first_collision_observed'] for r in selected),
        right_censored_games=sum(r['right_censored_at_cap'] for r in selected),
        maximum_actual_context_probability_replay_error=max(replay_errors),summary=summary,rows=rows,
        current_training_refresh_decisions=64,optimizer_updates=0,native_worlds_started=0,
        recorded_actual_history_used=True,global_context_equivalence_or_native_efficacy_claimed=False)
    write(args.output/'report.json',result);(args.output/'audit_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(maximum_actual_context_probability_replay_error=max(replay_errors),summary=summary)))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--report',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
