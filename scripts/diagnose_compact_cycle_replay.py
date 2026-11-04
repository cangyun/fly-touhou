"""Locate a pre-update policy replay mismatch from saved tensors; no native game."""
from pathlib import Path
import argparse,json,os,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cycle',type=int,required=True)
    p.add_argument('--stable-sensory',action='store_true');a=p.parse_args()
    from fly_touhou.thrl_study_io import read_shared_json as read
    from fly_touhou.thrl_durable_records import write_json
    launch=read(a.run/'launch.json');cache=Path(launch['reuse_kernel_cache_from'])
    config=read(a.run/'environment-config.json')
    os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',FLY_TOUHOU_DETERMINISTIC_SPMM='1',
        FLY_TOUHOU_SPMM_CACHE=str(cache/'spmm-kernel-cache'),FLY_TOUHOU_FUSED_EDGE_GRAD='1',
        FLY_TOUHOU_EDGE_GRAD_CACHE=str(cache/'kernel-cache'))
    import numpy as np,torch,psutil
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    proc=psutil.Process();proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);proc.cpu_affinity(proc.cpu_affinity()[-2:])
    from fly_touhou.compact_story_model import from_story_checkpoint,sequence
    from fly_touhou.compact_story_behavior import story_behavior_logits
    from fly_touhou.compact_story_batch_selection import select_sequences,interleaved_groups
    from fly_touhou.spiking_engineered_control import SpikingState
    a.output.parent.mkdir(parents=True,exist_ok=True)
    begun=time.monotonic()
    def status(phase):
        write_json(a.output.with_suffix('.status.json'),dict(phase=phase,PID=proc.pid,created=proc.create_time(),seconds=time.monotonic()-begun))
        print(phase,flush=True)
    status('loading_saved_case_or_tape')
    case_path=a.output.with_suffix('.case.pt')
    if case_path.exists():case=torch.load(case_path,map_location='cpu',weights_only=False)
    else:
        tape=torch.load(a.run/f'rollout-{a.cycle:03}.pt',map_location='cpu',weights_only=False)
        rng=np.random.default_rng(int(config['seed'])+a.cycle);options=launch['PPO_sample_selection']
        selected,_=select_sequences(tape['features'],tape['valid_mask'],16,options['rows'],rng,
            {int(k):v for k,v in options['stage_weights'].items()})
        chosen=interleaved_groups(selected,rng)[0]
        times=torch.tensor([t for t,w in chosen])[None]+torch.arange(16)[:,None]
        worlds=torch.tensor([w for t,w in chosen])[None].expand(16,-1)
        batch={k:tape[k][times,worlds] for k in ['features','actions','episode_starts','old_log_probs',
            'recorded_policy_logits','recorded_neural_horizon_scores','recorded_selected_DN']}
        case=dict(chosen=chosen,batch=batch,before=SpikingState.pack([tape['block_states'][t].row(w) for t,w in chosen]),
            mt=tape['metadata']['movement_sampling_temperature'],bt=tape['metadata']['bomb_sampling_temperature'])
        torch.save(case,case_path);del tape
    payload=torch.load(a.run/'interrupted-parameters.pt',map_location='cpu',weights_only=False)
    status('loading_saved_model')
    from fly_touhou import connectome as graph_module
    graph_loader=graph_module.load_connectome
    graph_module.load_connectome=lambda path,*,max_neurons=None,verify=True:graph_loader(path,max_neurons=max_neurons,verify=False)
    try:model=from_story_checkpoint(ROOT/'data/connectome',payload,device='cuda:0').eval()
    finally:graph_module.load_connectome=graph_loader
    model.core.sparse.deterministic_spmm=True
    model.core.input_encoder.batch_invariant=a.stable_sensory
    chosen=case['chosen'];batch={k:v.to(model.device) for k,v in case['batch'].items()}
    before=case['before'].to(model.device);mt=case['mt'];bt=case['bt']
    results=[]
    def evaluate(label,result,columns):
        logits=result['logits'];actions=batch['actions'][:,columns]
        behavior=story_behavior_logits(logits,mt,bt)
        new=behavior.log_softmax(-1).gather(-1,actions[...,None])[...,0]
        error=(new-batch['old_log_probs'][:,columns]).abs()
        old=batch['recorded_policy_logits'][:,columns]
        finite=torch.isfinite(logits)&torch.isfinite(old)
        delta=torch.where(finite,logits-old,0.).abs()
        risk=(result['horizon_scores']-batch['recorded_neural_horizon_scores'][:,columns]).abs()
        item=dict(mode=label,max_logp_error=float(error.max()),mean_logp_error=float(error.mean()),
            per_tick_max_error=error.max(-1).values.tolist(),worst_row=error.argmax().item(),
            raw_logit_error=float(delta.max()),risk_score_error=float(risk.max()))
        results.append(item);write_json(a.output,dict(checkpoint_costs=payload['costs'],chosen=chosen,
            temperatures=[mt,bt],batch_invariant_sensory=a.stable_sensory,
            results=results,optimizer_updates=0,native_actions=0));print(json.dumps(item),flush=True)
        return error
    status('shared_checkpoint_forward')
    with torch.no_grad():
        result=sequence(model,batch['features'],batch['episode_starts'],before,batch['actions'],checkpoint_chunks=True)
        errors=evaluate('shared_batch16',result,slice(None));del result
        worst=int(errors.max(0).values.argmax())
        status('same_recorded_sequence_single_world')
        result=sequence(model,batch['features'][:,worst:worst+1],batch['episode_starts'][:,worst:worst+1],
            before.row(worst),batch['actions'][:,worst:worst+1],checkpointed=False)
        evaluate('nested_batch1_worst_world',result,slice(worst,worst+1));del result
        status('nested_batch16')
        result=sequence(model,batch['features'],batch['episode_starts'],before,batch['actions'],checkpointed=False)
        evaluate('nested_batch16',result,slice(None));del result
        status('same_world_batch_widths')
        for width in range(2,8):
            one=before.row(worst)
            state=SpikingState(*(v.repeat(width,1) for v in one.tensors()))
            result=sequence(model,batch['features'][:,worst:worst+1].repeat(1,width,1),
                batch['episode_starts'][:,worst:worst+1].repeat(1,width),state,
                batch['actions'][:,worst:worst+1].repeat(1,width),checkpointed=False)
            result={k:v[:,:1] for k,v in result.items() if torch.is_tensor(v)}
            evaluate(f'repeated_world_batch{width}',result,slice(worst,worst+1));del result
    status('completed')


if __name__=='__main__':main()
