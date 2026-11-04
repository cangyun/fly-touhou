"""Private smooth-GRU derivative probe; no training, optimizer or native game."""
from pathlib import Path
import argparse
import gzip
import json
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from fly_touhou.dense_geometry_ppo import dense_geometry_ppo_loss
from fly_touhou.diagnostic_gru import DiagnosticGRU
from fly_touhou.experiment import file_sha256
from fly_touhou.native_diagnostic import write_json
from fly_touhou.right_aligned_credit import right_aligned_segments


def full_outputs(model, observations, ids, *, states=False):
    outputs, initial = [], []
    for i, identity in enumerate(ids):
        observation = observations[i:i+1]
        if i == 0 or ids[i-1] != identity:
            state = torch.zeros(1, model.hidden_size, dtype=observations.dtype)
            for _ in range(3):
                _, state = model(observation, state)
        if states: initial.append(state.detach().clone())
        logits, state = model(observation, state)
        outputs.append(logits)
    return torch.cat(outputs), torch.cat(initial) if states else None


def vector(model):
    return torch.cat([p.grad.flatten() for p in model.parameters()])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--update',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError('Preserve earlier private probes')
    torch.set_num_threads(2)
    behavior_path=args.run/'training'/f'behavior_{args.update:04d}.pt.gz'
    with gzip.open(behavior_path,'rb') as stream: behavior=torch.load(stream,map_location='cpu',weights_only=True)
    parent_path=Path(behavior['parent_checkpoint']['path'])
    assert file_sha256(parent_path)==behavior['parent_checkpoint']['sha256']
    parent=torch.load(parent_path,map_location='cpu',weights_only=True)
    model=DiagnosticGRU().double()
    model.load_state_dict(parent['actor'])
    original={k:p.detach().clone() for k,p in model.named_parameters()}
    observations=behavior['observations'].double()
    ids=behavior['episode_ids']
    arguments=[behavior[k] for k in ('actions','old_selected_logp','advantages','forecast_costs','danger_weights')]
    def objective(outputs,first=None,last=None):
        values=arguments if first is None else [v[first:last] for v in arguments]
        return dense_geometry_ppo_loss(outputs,*values,auxiliary_coefficient=5.,entropy_coefficient=.001)
    logits,_=full_outputs(model,observations,ids)
    full_loss=objective(logits)
    full_loss.backward()
    full_gradient=vector(model).detach().clone()
    model.zero_grad(set_to_none=True)
    with torch.no_grad(): _,initial=full_outputs(model,observations,ids,states=True)
    loss_sum=0.
    for first,last in right_aligned_segments(ids,8):
        state=initial[first:first+1].detach()
        outputs=[]
        for i in range(first,last):
            output,state=model(observations[i:i+1],state);outputs.append(output)
        loss=objective(torch.cat(outputs),first,last)
        (loss*((last-first)/len(observations))).backward()
        loss_sum+=float(loss.detach())*(last-first)
    truncated=vector(model).detach().clone()
    assert abs(loss_sum/len(observations)-float(full_loss.detach()))<1e-12
    cosine=float(torch.dot(full_gradient,truncated)/(full_gradient.norm()*truncated.norm()))
    directions={'full_history_gradient':full_gradient/full_gradient.norm(),
                'truncated8_gradient':truncated/truncated.norm()}
    finite_differences=[]
    for name,direction in directions.items():
        expected=float(torch.dot(full_gradient,direction))
        for epsilon in (.001,.0001,.00001):
            values=[]
            with torch.no_grad():
                for sign in (-1.,1.):
                    offset=0
                    for key,p in model.named_parameters():
                        count=p.numel()
                        p.copy_(original[key]+sign*epsilon*direction[offset:offset+count].reshape_as(p))
                        offset+=count
                    output,_=full_outputs(model,observations,ids)
                    values.append(float(objective(output)))
                for key,p in model.named_parameters():p.copy_(original[key])
            measured=(values[1]-values[0])/(2*epsilon)
            finite_differences.append({'direction':name,'epsilon':epsilon,
                'full_autograd_directional_derivative':expected,
                'centered_finite_difference':measured,
                'relative_error':abs(measured-expected)/max(1e-12,abs(expected))})
    assert all(v['relative_error']<1e-4 for v in finite_differences)
    assert all(torch.equal(p.detach(),original[k]) for k,p in model.named_parameters())
    result={'schema':'private-GRU-recorded-loss-credit-probe-v1','status':'passed',
        'parent_checkpoint':behavior['parent_checkpoint'],
        'behavior':{'path':str(behavior_path),'sha256':file_sha256(behavior_path)},
        'probe_script_sha256':file_sha256(Path(__file__)),
        'samples':len(observations),'complete_episodes':len(set(ids)),
        'reference_precision':'private CPU float64;production CUDA float32 unchanged',
        'same_forward_loss':float(full_loss.detach()),'full_history_gradient_norm':float(full_gradient.norm()),
        'truncated8_gradient_norm':float(truncated.norm()),'gradient_cosine':cosine,
        'finite_differences':finite_differences,'private_parameters_restored':True,
        'live_learner_parameters_changed':False,'optimizer_steps':0,'native_actions':0,
        'scope':'smooth GRU differentiation and truncated-credit approximation only;not game counterfactual or avoidance success',
        'G2_or_G3_achieved':False,'final_acceptance':False}
    write_json(args.output,result,exclusive=True)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
