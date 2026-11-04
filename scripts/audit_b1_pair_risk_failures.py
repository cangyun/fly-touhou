"""Read a sealed development failure set; validate neural entity-risk extrapolation."""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.thrl_study_io import read_shared_json as read,utc_now
from fly_touhou.atomic_json_io import atomic_write_json as write


def run(a):
    import torch
    from fly_touhou.thrl_pair_risk import PairRiskNetwork,action_clearances,QUERY_SCHEMA
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    if checkpoint.get('query_schema')!=QUERY_SCHEMA:raise ValueError('Do not reinterpret an old feature contract')
    net=PairRiskNetwork(checkpoint['width']).cuda().eval();net.load_state_dict(checkpoint['model'],strict=True)
    report=read(a.report)
    if report['status']!='completed' or report['native_seed_split']!='development':raise ValueError('Use completed development evidence only')
    if read(a.report.parent/'cleanup.json')!=dict(closed=True,close_errors=[]):raise ValueError('Source native cleanup incomplete')
    geometry={(x['policy'],x['game']):x for x in read(a.report.parent/'failure_trace_audit.json')['rows']};rows=[]
    with torch.inference_mode():
        for episode in report['episode_results']:
            pointer=episode.get('failure_trace')
            if not pointer:continue
            path=Path(pointer['arrays']);receipt=read(Path(pointer['receipt']))
            if hashlib.sha256(path.read_bytes()).hexdigest()!=pointer['sha256'] or receipt['checkpoint_sha256']!=report['pins']['trained']['sha256']:
                raise ValueError('Actual source trace provenance differs')
            with np.load(path) as z:
                f=torch.as_tensor(z['features'][-1:].copy(),device='cuda');o=torch.as_tensor(z['objects'][-1:].copy(),device='cuda');action=int(z['action'][-1])
            predicted=action_clearances(net,f,o)[0].cpu().numpy();g=geometry[(episode['policy'],episode['game'])]
            row=dict(policy=episode['policy'],game=episode['game'],actual_hit=episode['first_collision_observed'],action=action,
                selected_predicted_margin=float(predicted[action]),category=g['category'],predicted_margins=predicted.tolist(),
                normalized_player_xy=f[0,:2].cpu().tolist())
            if 'modeled_clearances_clipped64' in g:
                gold=np.array(g['modeled_clearances_clipped64']);safe=gold>1
                row.update(known_geometry=True,has_buffered_alternative=bool(safe.any()),
                    neural_best_modeled_margin=float(gold[predicted.argmax()]),
                    mean_margin_error=float(np.abs(np.clip(predicted,-64,64)-gold).mean()),
                    modeled_danger_pairs=int((gold<=0).sum()),modeled_danger_predicted_safe=int(((gold<=0)&(predicted>=1)).sum()),
                    modeled_safe_pairs=int((gold>1).sum()),modeled_safe_predicted_danger=int(((gold>1)&(predicted<1)).sum()))
            rows.append(row)
    known=[x for x in rows if x.get('known_geometry')];buffered=[x for x in rows if x.get('has_buffered_alternative')]
    summary=dict(cases=len(rows),actual_collision_danger_recall=sum(x['selected_predicted_margin']<1 for x in rows)/len(rows),
        known_geometry_cases=len(known),known_geometry_danger_recall=sum(x['selected_predicted_margin']<1 for x in known)/len(known),
        buffered_cases=len(buffered),neural_best_has_modeled_buffer=sum(x['neural_best_modeled_margin']>1 for x in buffered),
        selected_margin_quantiles=np.quantile([x['selected_predicted_margin'] for x in rows],[0,.25,.5,.75,1]).tolist(),
        modeled_danger_pairs=sum(x['modeled_danger_pairs'] for x in known),modeled_danger_predicted_safe=sum(x['modeled_danger_predicted_safe'] for x in known),
        modeled_safe_pairs=sum(x['modeled_safe_pairs'] for x in known),modeled_safe_predicted_danger=sum(x['modeled_safe_predicted_danger'] for x in known))
    result=dict(status='completed',utc=utc_now(),summary=summary,rows=rows,query_schema=QUERY_SCHEMA,
        checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),native_instances=0,optimizer_steps=0,
        source_report=str(a.report),fit_on_these_cases=False,source_partition='development',native_alternatives_verified=False)
    a.output.mkdir(parents=True,exist_ok=False);write(a.output/'report.json',result);print(json.dumps(summary))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for key in ['checkpoint','report','output']:p.add_argument('--'+key,type=Path,required=True)
    run(p.parse_args())
