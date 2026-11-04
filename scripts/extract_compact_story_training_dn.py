"""Replay sealed student training states to collect all18 candidate DN activities.

Uses recorded actions and carried whole-brain states. No native game launch,
teacher action substitution,optimizer update or development-game fitting.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import psutil

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.thrl_durable_records import write_json


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--rollout',type=Path,required=True)
    ap.add_argument('--connectome',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--execute',action='store_true')
    a=ap.parse_args()
    if not a.execute:raise SystemExit('Explicit --execute required for CUDA replay')
    a.output.mkdir(parents=True,exist_ok=False);proc=psutil.Process();begun=time.monotonic()
    launch=dict(PID=proc.pid,create_time=proc.create_time(),argv=sys.argv,
        checkpoint=str(a.checkpoint),rollout=str(a.rollout),CUDA_started=False,
        optimizer_updates=0,native_games_launched=0,development_data_used=False)
    write_json(a.output/'launch.json',launch)
    try:
        import torch
        from fly_touhou.compact_story_model import from_story_checkpoint
        from fly_touhou.spiking_engineered_control import SpikingState
        torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        payload=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
        data=torch.load(a.rollout,map_location='cpu',weights_only=False)
        model=from_story_checkpoint(a.connectome,payload).eval()
        launch.update(CUDA_started=True,CUDA_device=torch.cuda.get_device_name(),source_costs=payload.get('costs'),
            world_slots=data['actions'].shape[1],recorded_source_actions_only=True)
        write_json(a.output/'launch.json',launch)
        starts=sorted(data['block_states']);length=starts[1]-starts[0]
        contacts=(data['outcomes']==1).nonzero().tolist()
        contact_blocks={t//length*length for t,w in contacts}
        saved={k:[] for k in ('DN','margins','trusted','normal','actions','outcomes','world','time')}
        max_dn_error=max_logp_error=0.;times=0
        with torch.inference_mode():
            for block in starts:
                if (a.output/'STOP').exists():raise RuntimeError('Operator STOP')
                state=data['block_states'][block].to('cuda:0')
                target=range(block,min(block+length,len(data['actions']))) if block in contact_blocks else (block,)
                for t in target:
                    state=state.reset(data['episode_starts'][t].to(model.device))
                    actions=data['actions'][t].to(model.device)
                    out,state,selected=model.step(data['features'][t].to(model.device),state,actions=actions,return_all_DN=True)
                    dn_error=float((selected.cpu()-data['recorded_selected_DN'][t]).abs().max())
                    logp=out['logits'].log_softmax(-1).gather(-1,actions[:,None])[:,0].cpu()
                    logp_error=float((logp-data['old_log_probs'][t]).abs().max())
                    max_dn_error=max(max_dn_error,dn_error);max_logp_error=max(max_logp_error,logp_error)
                    if dn_error>1e-4 or logp_error>1e-4:
                        raise RuntimeError(f'Checkpoint/tape replay mismatch at{t}:DN={dn_error},logp={logp_error}')
                    saved['DN'].append(out['candidate_DN'].cpu())
                    for k in ('margins','trusted','actions','outcomes'):saved[k].append(data[k][t])
                    saved['normal'].append(data['unprotected_mask'][t])
                    saved['world'].append(torch.arange(len(actions)));saved['time'].append(torch.full_like(actions.cpu(),t))
                    times+=1
                write_json(a.output/'progress.json',dict(phase='replaying_training_brain_states',block=block,
                    temporal_points=times,student_states=times*len(actions),max_DN_replay_error=max_dn_error,
                    max_logp_replay_error=max_logp_error,elapsed_seconds=time.monotonic()-begun))
        packed={k:torch.cat(v,0) for k,v in saved.items()}
        packed['metadata']=dict(checkpoint=str(a.checkpoint),rollout=str(a.rollout),source_costs=payload.get('costs'),
            all18_candidates=True,selected_actual_successor_only=True,DN_dtype='float32',development_data_used=False,
            sample_design='Every saved16-step block start;complete16-step blocks containing observed native contacts',
            max_DN_replay_error=max_dn_error,max_logp_replay_error=max_logp_error)
        torch.save(packed,a.output/'training-DN.pt')
        write_json(a.output/'result.json',dict(complete=True,student_states=len(packed['DN']),candidate_directions=len(packed['DN'])*18,
            contact_blocks=sorted(contact_blocks),optimizer_updates=0,native_games_launched=0,
            output=str(a.output/'training-DN.pt'),**packed['metadata'],elapsed_seconds=time.monotonic()-begun))
    except BaseException as error:
        write_json(a.output/'failure.json',dict(error=repr(error),no_automatic_retry=True));raise


if __name__=='__main__':main()
