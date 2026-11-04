"""Read-only first-actual-minibatch DN-interface gradient reconstruction.

No full brain, neuron dynamics, teacher, native/GPU or Adam. Cached DN leaves
have no temporal/neural Jacobian; their gradients are not input/gain gradients.
Default is a NoTorch plan; --run-cpu makes one finite reconstruction attempt.
"""
from pathlib import Path
from dataclasses import asdict
from types import SimpleNamespace
import argparse
import hashlib
import json
import os
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
RUN=Path('./outputs/distillation/D95-fly-DN-cost26-joint783360-v1')
SCHEMA='fly-first-joint-minibatch-DN-interface-three-objective-gradients-v1'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=RUN)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-seconds',type=float,default=120.)
    p.add_argument('--run-cpu',action='store_true')
    a=p.parse_args()
    if not 0<a.max_seconds<=120:p.error('Finite CPU budget in(0,120] required')
    os.environ.update(CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1')
    if not a.run_cpu:
        print(json.dumps(dict(schema=SCHEMA,status='prepared',Torch_imported='torch' in sys.modules,
            source=str(a.run/'initial-training.pt'),rollout='traces/rollout-000.npz',max_seconds=a.max_seconds,
            brain_forwards=0,GPU=False,Adam=0)));return
    if a.output.exists():raise FileExistsError('Fresh diagnostic output required')
    a.output.mkdir(parents=True)
    import psutil
    process=psutil.Process();process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);process.cpu_affinity(process.cpu_affinity()[-2:])
    from fly_touhou.thrl_dodge_diagnostics import _shared_binary
    from fly_touhou.thrl_study_io import read_shared_json
    from fly_touhou.thrl_durable_records import write_json
    begun=time.monotonic();peak=0
    def guard():
        nonlocal peak
        peak=max(peak,process.memory_info().rss)
        if time.monotonic()-begun>a.max_seconds:raise TimeoutError('Finite CPU reconstruction budget reached')
    def sha(path):
        with _shared_binary(path) as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
    try:
        import numpy as np
        import torch
        from torch.nn import functional as F
        torch.set_num_threads(2)
        from fly_touhou.fly_dn_cost_checkpoint import read_training_source
        from fly_touhou.fly_gru_aligned_model import ThreeDNValueCritics
        from fly_touhou.fly_dn_cost_model import DNCostHead
        from fly_touhou.fly_gru_aligned_ppo import aligned_ppo_config,prepare_three_value_targets,three_value_ppo_loss_tensor_only
        from fly_touhou.fly_gru_aligned_update import sequence_locations,GUIDANCE_CONFIG,_numbers,USED_TASK_TERMS
        from fly_touhou.fly_gru_risk26_update import minibatch_risk26_normalizers,continuous_cost_terms
        from fly_touhou.fly_gru_risk26_contract import ContinuousCostConfig
        from fly_touhou.fly_gru_aligned_guidance import gru_aligned_guidance_loss
        from fly_touhou.fly_dn_cost_native_policy import compose_from_active_mask,normal_unprotected_from_features,source_policy_logits
        initial=a.run/'initial-training.pt';initial_sha=sha(initial)
        source=read_training_source(initial,expected_sha256=initial_sha,source_kind='dn_cost26')
        algorithm=source['recovery']['algorithm'];cfg=aligned_ppo_config(**algorithm['PPO'])
        if algorithm['guidance']!=asdict(GUIDANCE_CONFIG):raise ValueError('Saved actual guidance differs from imported source')
        cost_cfg=ContinuousCostConfig(**algorithm['continuous_cost'])
        parameters=source['recovery']['model_payload'];base=parameters['base_payload'];c=parameters['head_configuration']
        actor=base['student_payload']['brain_parameters']['action_readout.weight'].detach().clone()
        with torch.random.fork_rng(devices=[]):
            critic=ThreeDNValueCritics(1299);head=DNCostHead(c['output_indices'],
                reference_rate_hz=c['reference_rate_hz'],beta=c['beta'],seed=c['initialization_seed'])
        critic.load_state_dict(base['critic_state'],strict=True);critic.eval().requires_grad_(False)
        head.load_state_dict(parameters['head_state'],strict=True);head.eval().requires_grad_(False)
        del source,parameters,base
        path=a.run/'traces/rollout-000.npz';side_path=path.with_suffix('.json');side=read_shared_json(side_path)
        trace_sha=sha(path)
        if trace_sha!=side['trace_sha256']:raise ValueError('Committed initial trace SHA differs')
        meta=side['metadata'];policy_version=int(meta['policy_version']);seed=int(meta['brain_seed'])+policy_version
        update_path=a.run/'traces/update-000001.json';update=read_shared_json(update_path)
        expected=update['learning']['terms'][0]
        names=('DN_features','features','legal_actions','actions','old_log_probs','values','reward_vectors',
            'bootstrap_values','terminated','episode_starts','brain_rng_keys','noise_counters','raw_logits',
            'composed_logits','predicted_costs','combined_normalized_advantages','returns3','GAE3',
            'rule_probabilities','rule_weights','deathbomb_mask','d95_risk_costs','d95_risk_queried_mask','d95_risk_valid_mask',
            'task_margins','task_trusted','task_outcomes','task_unprotected_mask')
        with _shared_binary(path) as stream,np.load(stream,allow_pickle=False) as z:
            arrays={k:z[k].copy() for k in names}
        data_all={k:torch.from_numpy(v) for k,v in arrays.items() if not k.startswith('task_')}
        if data_all['DN_features'].shape!=(256,30,1299):raise ValueError('Exact normalized following DN rollout required')
        adv3,returns,adv=prepare_three_value_targets(data_all['reward_vectors'],data_all['values'],
            data_all['bootstrap_values'],data_all['terminated'],data_all['episode_starts'],cfg=cfg)
        target_checks={name:dict(exact=bool(torch.equal(actual,data_all[key])),
            max_abs=float((actual-data_all[key]).abs().max())) for name,actual,key in
            (('GAE3',adv3,'GAE3'),('returns3',returns,'returns3'),('normalized_advantages',adv,'combined_normalized_advantages'))}
        generator=torch.Generator(device='cpu').manual_seed(seed)
        indices=torch.randperm(1920,generator=generator)[:64]
        times,worlds=sequence_locations(indices,256)
        target=SimpleNamespace(**{k:arrays['task_'+k] for k in ('margins','trusted','outcomes','unprotected_mask')})
        rollout=SimpleNamespace(actions=data_all['actions'],task_targets=target,
            rule_probabilities=data_all['rule_probabilities'],rule_weights=data_all['rule_weights'],
            deathbomb_mask=data_all['deathbomb_mask'],d95_risk_queried_mask=data_all['d95_risk_queried_mask'],
            d95_risk_valid_mask=data_all['d95_risk_valid_mask'])
        normalizers=minibatch_risk26_normalizers(rollout,indices,'cpu')
        DN=data_all['DN_features'][times,worlds].detach().clone().requires_grad_(True)
        totals={k:DN.new_zeros(()) for k in ('PPO','five_guidance','continuous_cost')}
        ppo_numbers=[];guide_numbers=[];cost_numbers=[];checks=[]
        for start in range(0,64,16):
            guard();t,w=times[:,start:start+16],worlds[:,start:start+16]
            local={k:v[t,w].detach() for k,v in data_all.items() if k not in
                ('bootstrap_values','combined_normalized_advantages','returns3','GAE3')}
            local['normalized_advantages']=adv[t,w];local['returns']=returns[t,w]
            local['valid_mask']=torch.ones(4,16,dtype=torch.bool)
            task={k:torch.from_numpy(getattr(target,k)[t.numpy(),w.numpy()]) for k in
                ('margins','trusted','outcomes','unprotected_mask')}
            task.update({k:local[k] for k in ('rule_probabilities','rule_weights','deathbomb_mask')})
            raws=[];combineds=[];costs=[];values=[]
            for tick in range(4):
                x=DN[tick,start:start+16]
                raw=F.linear(x,actor);pred=head.costs_from_features(x)
                combined=compose_from_active_mask(raw,pred,normal_unprotected_from_features(local['features'][tick]),
                    beta=head.beta,check_finite=False)
                # Literal source gradient bridge; no extra value1/2 factor.
                shared=x.detach()+cfg.critic_shared_grad_scale*(x-x.detach())
                raws.append(raw);combineds.append(combined);costs.append(pred);values.append(critic(shared))
            raw,combined,predicted,value=(torch.stack(x) for x in (raws,combineds,costs,values))
            legal=combined.masked_fill(~local['legal_actions'],-torch.inf)
            clean=source_policy_logits(combined,local['features'],check_finite=False)
            logp=clean.log_softmax(-1).gather(-1,local['actions'][...,None]).squeeze(-1)
            checks.append(dict(micro=start//16,old_logp_max_abs=float((logp-local['old_log_probs']).abs().max().detach()),
                raw_max_abs=float((raw-local['raw_logits']).abs().max().detach()),
                composed_max_abs=float((combined-local['composed_logits']).abs().max().detach()),
                cost_max_abs=float((predicted-local['predicted_costs']).abs().max().detach()),
                three_values_max_abs=float((value-local['values']).abs().max().detach())))
            ppo=three_value_ppo_loss_tensor_only(legal,value,local['actions'],local['old_log_probs'],
                local['values'],local['normalized_advantages'],local['returns'],cfg)
            guidance=gru_aligned_guidance_loss(legal,**task,actions=local['actions'],valid_mask=local['valid_mask'],
                config=GUIDANCE_CONFIG,normalizers={k:normalizers[k] for k in ('safe','future','hit','rule','deathbomb')})
            calibration=continuous_cost_terms(predicted,local,normalizers['cost'],cost_config=cost_cfg)
            totals['PPO']=totals['PPO']+ppo['total_loss']*.25
            totals['five_guidance']=totals['five_guidance']+guidance['total_loss']
            totals['continuous_cost']=totals['continuous_cost']+calibration['weighted_loss']
            ppo_numbers.append(_numbers(ppo));guide_numbers.append({k:float(guidance[k].detach()) for k in USED_TASK_TERMS})
            cost_numbers.append({k:float(v.detach()) for k,v in calibration.items()})
        summary={k:sum(row[k] for row in ppo_numbers)/4 for k in ppo_numbers[0]}
        summary.update({'task_'+k:sum(row[k] for row in guide_numbers) for k in USED_TASK_TERMS})
        summary['PPO_total_loss']=summary['total_loss']
        summary['risk_cost_loss']=sum(row['loss'] for row in cost_numbers)
        summary['weighted_risk_cost']=sum(row['weighted_loss'] for row in cost_numbers)
        summary['risk_cost_valid_rows']=sum(row['count'] for row in cost_numbers)
        summary['risk_cost_queried_rows']=sum(row['queried_rows'] for row in cost_numbers)
        summary['risk_cost_mean_abs_gap_error']=sum(row['mean_absolute_gap_contribution'] for row in cost_numbers)
        summary['total_loss']=float(sum(totals.values()).detach())
        differences={k:dict(reconstructed=v,actual=float(expected[k]),abs_error=abs(v-float(expected[k])))
            for k,v in summary.items() if k in expected}
        guard();gradients={}
        for name,loss in totals.items():
            gradients[name]=torch.autograd.grad(loss,DN,retain_graph=True)[0].detach()
        def norm(g):return float(g.double().square().sum().sqrt())
        norms={k:norm(g) for k,g in gradients.items()}
        cosines={}
        for first,second in (('PPO','five_guidance'),('PPO','continuous_cost'),('five_guidance','continuous_cost')):
            denominator=norms[first]*norms[second]
            cosines[first+'__'+second]=(float((gradients[first].double()*gradients[second].double()).sum())/denominator
                if denominator>0 else None)
        max_logp=max(c['old_logp_max_abs'] for c in checks)
        max_term=max(d['abs_error'] for d in differences.values())
        verified=(all(c['exact'] for c in target_checks.values()) and max_logp<=1e-5 and max_term<=2e-5
            and max(c['three_values_max_abs'] for c in checks)<=1e-4)
        guard()
        result=dict(schema=SCHEMA,status='DN_interface_actual_minibatch_verified' if verified else 'reconstruction_unverified',
            sources=dict(initial_training_sha256=initial_sha,trace_sha256=trace_sha,sidecar_sha256=sha(side_path),
                actual_update_sha256=sha(update_path)),policy_version=policy_version,minibatch_seed=seed,
            sequence_indices=indices.tolist(),normalizers={k:float(v) for k,v in normalizers.items()},
            whole_rollout_target_checks=target_checks,readout_checks=checks,term_comparisons=differences,
            loss_values={k:float(v.detach()) for k,v in totals.items()},DN_interface_gradient_L2=norms,
            DN_interface_gradient_cosines=cosines,
            DN_interface_gradient_relative_norms={k:v/norms['five_guidance'] if norms['five_guidance'] else None for k,v in norms.items()},
            source_algorithm=algorithm,interface='cached following DN EMA/reference150,independent leaf per actual observation',
            critic_shared_gradient_scale=cfg.critic_shared_grad_scale,head_parameters_frozen_for_interface_derivatives=True,
            actual_parameter_gradients_computed=False,input_or_gain_parameter_conflict_verified=False,
            limits=['single initial minibatch only','no neural/temporal Jacobian','DN cosine is not input/gain gradient cosine',
                'gradient magnitude is not Adam movement or causal performance evidence','no source update or training rejection'],
            elapsed_seconds=time.monotonic()-begun,peak_RSS_bytes=peak,CPU_threads=2,CPU_affinity=process.cpu_affinity(),
            priority='BelowNormal',GPU_initialized=torch.cuda.is_initialized(),brain_forwards=0,teacher_queries=0,
            native_steps=0,Adam_steps=0,source_module_sha256=sha(Path(__file__)))
        result['dependencies']={name:sha(ROOT/'src/fly_touhou'/name) for name in
            ('fly_gru_risk26_update.py','fly_gru_aligned_update.py','fly_gru_aligned_guidance.py','fly_gru_aligned_ppo.py',
             'fly_dn_cost_model.py','fly_dn_cost_native_policy.py','fly_dn_continuous_risk_policy.py')}
        write_json(a.output/'report.json',result)
        np.savez_compressed(a.output/'DN-interface-gradients.npz',sequence_indices=indices.numpy(),
            time=times.numpy(),world=worlds.numpy(),**{k:v.numpy() for k,v in gradients.items()})
        print(json.dumps(dict(status=result['status'],seconds=result['elapsed_seconds'],max_logp_error=max_logp,
            max_term_error=max_term,losses=result['loss_values'],norms=norms,cosines=cosines,
            normalizers=result['normalizers']),allow_nan=False))
    except BaseException as error:
        write_json(a.output/'failure.json',dict(schema=SCHEMA,error=repr(error),elapsed_seconds=time.monotonic()-begun,
            peak_RSS_bytes=peak,brain_forwards=0,teacher_queries=0,native_steps=0,Adam_steps=0))
        raise


if __name__=='__main__':main()
