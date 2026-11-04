"""Replay measured failure inputs/GRU context; compare conditional18 and full19.

Zero native instances and optimizer steps. Model predictions do not validate
unexecuted original-game alternatives, even when geometry has positive margin.
"""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.thrl_study_io import read_shared_json as read,utc_now
from fly_touhou.atomic_json_io import atomic_write_json as write


def run(a):
    import torch,psutil
    r=read(a.report);folder=a.report.parent
    if r['status']!='completed' or read(folder/'cleanup.json')!=dict(closed=True,close_errors=[]):
        raise ValueError('Require completed native diagnostic and clean pool shutdown')
    pin=r['pins']['trained'];checkpoint=Path(pin['copy'])
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=pin['sha256']:raise ValueError('Frozen policy changed')
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    runtime=Path(payload['config']['source_runtime']);sys.path.insert(0,str(runtime/'src'))
    from fly_touhou.thrl_model import MOActorCritic
    from fly_touhou.thrl_policy_actions import policy_logits
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0]<4*1024**3:
        raise RuntimeError('Explicit CUDA inference with four GiB reserve required')
    torch.cuda.set_per_process_memory_fraction(.12);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if sys.platform=='win32':psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    a.output.mkdir(parents=True,exist_ok=False)
    model=MOActorCritic(**payload['model_config']).cuda().eval();model.load_state_dict(payload['model']);del payload
    categories={(x['policy'],x['game']):x for x in read(folder/'failure_trace_audit.json')['rows']}
    data=[]
    for case in r['episode_results']:
        pointer=case.get('failure_trace')
        if not pointer:continue
        receipt=read(Path(pointer['receipt']));path=Path(pointer['arrays'])
        if (receipt['checkpoint_sha256']!=pin['sha256'] or receipt['sha256']!=pointer['sha256']
                or hashlib.sha256(path.read_bytes()).hexdigest()!=receipt['sha256']):raise ValueError('Trace identity differs')
        if (receipt['policy'],receipt['game'])!=(case['policy'],case['game']):raise ValueError('Episode identity differs')
        with np.load(path,allow_pickle=False) as z:
            item=dict(features=z['features'][-1].copy(),objects=z['objects'][-1].copy(),maps=z['last_preaction_maps'].copy(),
                hidden=z['hidden'][-1].copy().reshape(-1),probabilities=z['probabilities'][-1].copy(),action=int(z['action'][-1]),
                case=case,geometry=categories[(case['policy'],case['game'])])
        if any(item[k].dtype!=np.float32 for k in ['features','objects','maps','hidden','probabilities']):
            raise ValueError('Replay must retain original observation and memory precision')
        data.append(item)
    if len(data)!=40:raise ValueError('Need both complete20-game trained conditions')
    rows=[]
    with torch.inference_mode():
        for start in range(0,len(data),16):
            group=data[start:start+16]
            batch={k:torch.as_tensor(np.stack([x[k] for x in group]),device='cuda') for k in ['features','objects','maps']}
            hidden=torch.as_tensor(np.stack([x['hidden'] for x in group]),device='cuda')[None].contiguous()
            logits,_,_=model.forward_step(batch['features'],batch['maps'],hidden,objects=batch['objects'])
            logits=policy_logits(logits);full=logits.softmax(-1).cpu().numpy();movement=logits[:,:18].softmax(-1).cpu().numpy()
            for i,item in enumerate(group):
                error=float(np.max(np.abs(movement[i]-item['probabilities'])))
                if error>3e-5:raise ValueError('Captured actual probabilities do not replay: '+str(error))
                case=item['case'];g=item['geometry'];margin=g.get('modeled_clearances_clipped64');action=item['action']
                row=dict(policy=case['policy'],game=case['game'],native_frames=case['native_frames'],
                    category=g['category'],action=action,replay_max_probability_error=error,
                    Bomb_probability=float(full[i,18]),full19_argmax=int(full[i].argmax()),movement_argmax=int(movement[i].argmax()),
                    movement_probability_of_executed_action=float(movement[i,action]),
                    full19_probability_of_executed_action=float(full[i,action]),
                    player_normalized_xy=item['features'][:2].tolist(),buffered_actions=g.get('buffered_actions'))
                if margin is not None:
                    margin=np.asarray(margin)
                    row.update(predicted_unsafe4_conditional_mass=float(movement[i,margin<=1].sum()),
                        predicted_collision4_conditional_mass=float(movement[i,margin<=0].sum()),
                        chosen_predicted_clearance=float(margin[action]))
                rows.append(row)
    def summary(selected):
        b=np.array([x['Bomb_probability'] for x in selected]);risk=[x['predicted_unsafe4_conditional_mass'] for x in selected if 'predicted_unsafe4_conditional_mass' in x]
        return dict(cases=len(selected),mean_Bomb_probability=float(b.mean()),Bomb_probability_quantiles=np.quantile(b,[0,.5,.9,1]).tolist(),
            full19_Bomb_argmax_cases=sum(x['full19_argmax']==18 for x in selected),Bomb_probability_over_half=int((b>.5).sum()),
            geometry_covered_cases=len(risk),mean_predicted_unsafe4_conditional_mass=float(np.mean(risk)) if risk else None)
    result=dict(schema='B1-replayed-failure-full19-probability-audit-v1',status='completed',utc=utc_now(),
        source_report=str(a.report),checkpoint_sha256=pin['sha256'],candidate_optimized_decisions=pin['optimized_decisions'],
        summaries={p:summary([x for x in rows if x['policy']==p]) for p in ['trained_sampled','trained_greedy']},
        buffered_alternative_cases=summary([x for x in rows if x['category']=='modeled_buffered_alternative_not_selected']),
        rows=rows,max_replay_probability_error=max(x['replay_max_probability_error'] for x in rows),
        native_instances=0,optimizer_updates=0,training_modified=False,
        alternatives_are_native_counterfactuals=False,full19_natural_game_performance_measured=False)
    write(a.output/'report.json',result);print(json.dumps({k:result[k] for k in ['candidate_optimized_decisions','summaries','buffered_alternative_cases','max_replay_probability_error']}))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--report',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
