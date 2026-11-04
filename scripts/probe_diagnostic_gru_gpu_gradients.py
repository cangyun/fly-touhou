"""Compare production float32 CUDA GRU gradients with a private CPU copy."""
from pathlib import Path
import argparse
import gzip
import json
import os
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import torch
from fly_touhou.dense_geometry_ppo import dense_geometry_ppo_loss
from fly_touhou.diagnostic_gru import DiagnosticGRU
from fly_touhou.experiment import file_sha256
from fly_touhou.native_diagnostic import write_json
from fly_touhou.right_aligned_credit import right_aligned_segments


def run(model,data,first,last):
    model.zero_grad(set_to_none=True)
    state=data['states_before'][first:first+1].to(model.device)
    outputs=[]
    for i in range(first,last):
        output,state=model(data['observations'][i:i+1].to(model.device),state)
        outputs.append(output)
    logits=torch.cat(outputs)
    values=[data[k][first:last].to(model.device) for k in ('actions','old_selected_logp','advantages','forecast_costs','danger_weights')]
    loss=dense_geometry_ppo_loss(logits,*values,auxiliary_coefficient=5.,entropy_coefficient=.001)
    loss.backward()
    return float(loss.detach()),logits.detach().cpu(),state.detach().cpu(),{
        k:p.grad.detach().cpu().clone() for k,p in model.named_parameters()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--update',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Preserve prior probes')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    assert torch.cuda.is_available()
    path=args.run/'training'/f'behavior_{args.update:04d}.pt.gz'
    with gzip.open(path,'rb') as f:data=torch.load(f,map_location='cpu',weights_only=True)
    parent=Path(data['parent_checkpoint']['path'])
    assert file_sha256(parent)==data['parent_checkpoint']['sha256']
    checkpoint=torch.load(parent,map_location='cpu',weights_only=True)
    models=[DiagnosticGRU(),DiagnosticGRU().cuda()]
    for model in models:model.load_state_dict(checkpoint['actor'])
    segments=right_aligned_segments(data['episode_ids'],8)
    chosen=[]
    for index,(first,last) in enumerate(segments):
        if last-first<8 or index==0 or last==len(data['actions']) or data['episode_ids'][last-1]!=data['episode_ids'][last]:
            chosen.append((first,last))
    results=[]
    for first,last in chosen:
        cpu=run(models[0],data,first,last);gpu=run(models[1],data,first,last)
        gradients={}
        for name,reference in cpu[3].items():
            error=gpu[3][name].double()-reference.double()
            relative=float(error.norm()/reference.double().norm().clamp_min(1e-12))
            maximum=float(error.abs().max())
            assert relative<1e-4 and maximum<2e-6
            gradients[name]={'relative_L2_error':relative,'maximum_absolute_error':maximum}
        loss_error=abs(cpu[0]-gpu[0])
        logit_error=float((cpu[1]-gpu[1]).abs().max())
        state_error=float((cpu[2]-gpu[2]).abs().max())
        assert loss_error<2e-6 and logit_error<2e-6 and state_error<2e-6
        results.append({'segment':[first,last],'loss_error':loss_error,'logit_error':logit_error,
            'state_error':state_error,'all8_parameter_gradients':gradients})
    value={'schema':'private-native-GRU-CUDA-CPU-gradient-parity-v1','status':'passed',
        'behavior':{'path':str(path),'sha256':file_sha256(path)},'parent':data['parent_checkpoint'],
        'source_script_sha256':file_sha256(Path(__file__)),
        'production_model_source_sha256':file_sha256(ROOT/'src/fly_touhou/diagnostic_gru.py'),
        'GPU':torch.cuda.get_device_name(),'precision':'production float32 parameters;float64 objective',
        'cases':results,'all_parameter_tensors_compared':8,'optimizer_steps':0,'native_actions':0,
        'live_actor_parameters_changed':False,'scope':'private forward and backward parity;not avoidance success',
        'G2_or_G3_achieved':False,'final_acceptance':False}
    write_json(args.output,value,exclusive=True)
    print(json.dumps({'status':'passed','cases':len(results),'all8_parameter_gradients_checked':True,
        'maximum_loss_error':max(v['loss_error'] for v in results),'output':str(args.output)},ensure_ascii=False))


if __name__=='__main__':main()
