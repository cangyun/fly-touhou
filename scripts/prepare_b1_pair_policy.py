"""Package fixed B1 + learned entity readout and qualify real CUDA interfaces."""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.thrl_study_io import read_shared_json as read,utc_now
from fly_touhou.atomic_json_io import atomic_write_json as write


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def run(a):
    import torch
    from fly_touhou.thrl_pair_policy import EntityRiskPolicy,model_from_config
    from fly_touhou.thrl_pair_risk import QUERY_SCHEMA
    if not torch.cuda.is_available():raise RuntimeError('CUDA qualification required')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if sha(a.base_checkpoint)!=a.expected_base_sha:raise ValueError('Frozen original B1 checkpoint differs')
    fit=read(a.readout/'result.json')
    if fit['status']!='completed' or fit['native_instances']!=0:raise ValueError('Readout fit incomplete')
    base=torch.load(a.base_checkpoint,map_location='cpu',weights_only=False)
    head=torch.load(a.readout/'best.pt',map_location='cpu',weights_only=False)
    if head.get('query_schema')!=QUERY_SCHEMA or base.get('optimizer_transaction_complete') is False:
        raise ValueError('Mismatched query schema or incomplete original policy')
    model=EntityRiskPolicy(base['model_config'],width=head['width'],beta=0.).cuda().eval()
    model.base.load_state_dict(base['model'],strict=True);model.risk.load_state_dict(head['model'],strict=True)
    gold=[]
    for name in ['teacher-horizon-targets-v1','student-horizon-targets-v1']:
        with np.load(a.data/name/'targets.npz') as z:gold.append(z['margins'][:,0].copy())
    gold=np.concatenate(gold);held=np.load(a.readout/'held/source_state_indices.npy')
    choices=held[(gold[held].min(-1)<=0)&(gold[held].max(-1)>1)][:2]
    if len(choices)!=2:raise ValueError('Need measured critical held states for interface qualification')
    examples=[]
    with torch.inference_mode():
        for index in choices:
            world,frame=divmod(int(index),1024);folder=a.data/('teacher30-v1' if world<30 else 'student-aggregation-v1')/'demonstrations';world%=30
            arrays={k:np.load(folder/(k+'.npy'),mmap_mode='r') for k in ['features','maps','objects','dones']}
            hidden=torch.zeros(1,1,model.gru_hidden_size,device='cuda')
            for begin in range(0,frame,32):
                last=min(frame,begin+32)
                batch={k:torch.as_tensor(arrays[k][begin:last,world].copy(),device='cuda') for k in ['features','maps','objects']}
                encoded=model.base._encode(batch['features'],batch['maps'],batch['objects'])
                for offset in range(last-begin):
                    _,hidden=model.base.gru(encoded[offset:offset+1,None],hidden.contiguous())
                    if arrays['dones'][begin+offset,world]:hidden.zero_()
            observation={k:torch.as_tensor(arrays[k][frame:frame+1,world].copy(),device='cuda') for k in ['features','maps','objects']}
            examples.append((observation,hidden,int(index)))
        bomb_errors=[];changed=[]
        for obs,h,index in examples:
            f,m,o=(obs[k] for k in ['features','maps','objects'])
            expected,ev,eh=model.base.forward_step(f,m,h,objects=o)
            model.beta=0;zero,zv,zh=model.forward_step(f,m,h,objects=o)
            for x,y in [(expected,zero),(ev,zv),(eh,zh)]:torch.testing.assert_close(x,y,rtol=0,atol=0)
            model.beta=8.;actual,av,ah=model.forward_step(f,m,h,objects=o)
            torch.testing.assert_close(av,ev,rtol=0,atol=0);torch.testing.assert_close(ah,eh,rtol=0,atol=0)
            error=float((actual.softmax(-1)[:,18]-expected.softmax(-1)[:,18]).abs().max());bomb_errors.append(error)
            if error>1e-6 or not torch.isfinite(actual[:,:18]).all():raise ValueError('Bomb marginal or legal movements changed')
            changed.append(float((actual[:,:18]-expected[:,:18]).abs().max()))
            protected=f.clone();protected[:,20]=1
            pl,pv,ph=model.base.forward_step(protected,m,h,objects=o)
            ql,qv,qh=model.forward_step(protected,m,h,objects=o)
            for x,y in [(pl,ql),(pv,qv),(ph,qh)]:torch.testing.assert_close(x,y,rtol=0,atol=0)
    if max(changed)<1e-4:raise ValueError('Qualification did not exercise the learned movement branch')
    cpu_state={k:v.detach().cpu() for k,v in model.state_dict().items()}
    if any(not torch.equal(cpu_state['base.'+k],v) for k,v in base['model'].items()):
        raise RuntimeError('Original B1 tensors changed during qualification')
    if any(not torch.equal(cpu_state['risk.'+k],v) for k,v in head['model'].items()):
        raise RuntimeError('Learned readout tensors changed during qualification')
    clone=model_from_config(**model.model_config());clone.load_state_dict(cpu_state,strict=True)
    cfg=read(a.config);cfg.update(source_runtime=str(ROOT),source_manifest_sha256=sha(ROOT/'source_manifest.json'),
        model_config=model.model_config(),evaluation_only=True,
        initial_baseline_semantics='Same finalB1 already optimized1259520 lifetime PPO decisions; zero new experiment updates; not random initialization',
        diagnostic_checkpoint_semantics='Final B1 plus learned action/entity signed-clearance readout; native efficacy not yet demonstrated')
    a.output.mkdir(parents=True,exist_ok=False);write(a.output/'config.json',cfg)
    lineage=dict(base_checkpoint=str(a.base_checkpoint),base_sha256=a.expected_base_sha,
        base_original_candidate_optimized_decisions=base.get('candidate_optimized_decisions'),
        base_lifetime_PPO_decisions=1259520,head_checkpoint_sha256=sha(a.readout/'best.pt'),
        head_selected_Adam_steps=head['Adam_steps'],head_training_cost_Adam_steps=fit['Adam_steps'],query_schema=QUERY_SCHEMA)
    common=dict(optimizer_transaction_complete=True,optimizer_recovery_eligible=False,evaluation_only=True,
        candidate_optimized_decisions=0,inherited_decisions=1259520,created_utc=utc_now(),lineage=lineage)
    torch.save(dict(**common,model=base['model'],model_config=base['model_config'],config=cfg),a.output/'initial.pt')
    torch.save(dict(**common,model=cpu_state,model_config=model.model_config(),config=cfg),a.output/'latest.pt')
    qualification=dict(status='qualified_for_native_comparison',utc=utc_now(),device=torch.cuda.get_device_name(0),
        native_instances=0,base_weights_unchanged=True,baseline_source_tensor_count=len(base['model']),
        beta0_logits_values_hidden_exact=True,beta8_hidden_values_exact=True,
        protected_logits_values_hidden_exact=True,maximum_Bomb_marginal_error=max(bomb_errors),
        movement_logit_change_on_actual_states=changed,qualification_states=list(map(int,choices)),
        qualification_scope='real native train-partition observations; base hidden replayed on their recorded prefixes; interface equality only',
        strict_checkpoint_roundtrip=True,all18_movements_legal=True,
        model_parameters=model.parameter_counts(),risk_settings=dict(beta=8.,buffer=1.,risk_width=.5),
        latest_checkpoint_sha256=sha(a.output/'latest.pt'),initial_checkpoint_sha256=sha(a.output/'initial.pt'),
        lineage=lineage,native_efficacy_demonstrated=False)
    write(a.output/'CUDA-qualification.json',qualification);print(json.dumps(qualification))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for key in ['base-checkpoint','readout','data','config','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--expected-base-sha',required=True)
    run(p.parse_args())
