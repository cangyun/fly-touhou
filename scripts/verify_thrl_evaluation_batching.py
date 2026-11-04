"""Frozen production-size CUDA inference: serial versus 4/8/16 ready batches.

Uses 16 synthetic, encoder-valid native-shaped snapshots, never opens a game,
never runs an optimizer, and never overwrites any initialization checkpoint.
CUDA event timings cover resident-tensor forward spans, not end-to-end native
evaluation. The sample agreement cannot promise arbitrary future action parity.
"""
from pathlib import Path
import argparse
import hashlib
import math
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def synthetic_snapshots():
    """Generate explicit fixtures through the production encoders, not raw noise."""
    from fly_touhou.state import Snapshot,Player,Bullet,Enemy,Laser
    from fly_touhou.hazard_reader import COLLISION_GEOMETRY_SCHEMA,FULL_GAME_SHA256
    rng=np.random.default_rng(7102311)
    samples=[]
    counts=[0,1,8,32,64,96,140,3,16,48,96,120,0,7,80,160]
    for index,count in enumerate(counts):
        frame=1000+index*117;px=float(rng.uniform(-150,150));py=float(rng.uniform(128,416))
        state=[1,1,4,2,3][index%5]
        invulnerability=0 if state==4 else [0,1,4,60,240][index%5]
        deathbomb=1+index%8 if state==4 else 0
        player=Player(px,py,2.,invulnerability,bool(index%2),state,deathbomb)
        bullets=[];raw=[];geometry=[]
        for j in range(count):
            identity=0x10000+32*j;circle=j%3!=0
            flags=2|(0x10 if circle else 0)|(0x40 if j%5==0 else 0)
            scale=1.+(j%4)*.3;radius=2.+j%7;height=None if circle else 5.+j%11
            x,y=float(rng.uniform(-184,184)),float(rng.uniform(16,432))
            vx,vy=float(rng.uniform(-3,3)),float(rng.uniform(-1,5))
            native_state=2 if j%11==0 else 1;age=j%40
            bullets.append(Bullet(identity,x,y,vx,vy,radius,True,False,0,j%64,age))
            row=dict(id=identity,native_state=native_state,native_flags=flags,native_position=[x,y],
                native_velocity=[vx,vy],native_hitbox_value=radius,native_scale=scale,
                native_speed=math.hypot(vx,vy),native_timer=age,native_cancel_protection_counter=j%12,
                native_animation_gate=0)
            raw.append(row);geometry.append(dict(id=identity,native_flags=flags,native_scale=scale,
                native_hitbox_value=radius,native_height_value=height))
        enemies=[];enemy_rows=[]
        if index%4:
            enemies=[Enemy(0x90000,0.,100.,3000+index*100,12.,16.,True),
                     Enemy(0x91000,-100.,180.,100,12.,12.,False)]
            enemy_rows=[dict(id=e.id,x=e.x,y=e.y,hp=e.hp,vx=.25,vy=.5,contactable=True) for e in enemies]
        items=[dict(id=0xa0000+j,x=float(-100+j*35),y=float(180+j*15),kind=1+j%7,
                    vx=0.,vy=1.,state=1,age=j*5) for j in range(index%5)]
        lasers=() if index%4==0 else (Laser(0xb0000,0,-120.,80.,.9,160.,6.+index%4,True,age=24),)
        extra=dict(player_scale=1.,base_player_radius=2.,
            resources=dict(schema='original-th14-resources-v1',observed=True,items=items,
                enemies=enemy_rows,player_flags=0),
            thrl_events=dict(schema='th14-thrl-events-v1',observed=True,misses=index%3,
                bombs_used=index//4,point_items=index),
            hazard_metadata=dict(frame=frame,bullets=raw,lasers=[]),
            native_collision_geometry=dict(schema=COLLISION_GEOMETRY_SCHEMA,frame=frame,
                exe_sha256=FULL_GAME_SHA256,observed=True,
                player=dict(native_bounds_min=[px-2,py-2],native_bounds_max=[px+2,py+2],
                    native_scale=1.,native_base_radius=2.,native_definition_radius=2.,native_flags=0),
                bullets=geometry))
        samples.append(Snapshot(frame,frame-900,player,tuple(bullets),tuple(enemies),lasers,
            lives=2,bombs=0 if index%2==0 else 3,power=100+index*10,stage=1,difficulty=3,
            source='native-process',extra=extra))
    return samples


def comparison(torch,reference,value,*,atol=2e-6,rtol=2e-5):
    a,b=reference.detach().cpu(),value.detach().cpu()
    matching_inf=torch.equal(torch.isneginf(a),torch.isneginf(b))
    finite=torch.isfinite(a)&torch.isfinite(b)
    diff=(a[finite]-b[finite]).abs()
    scale=a[finite].abs().clamp_min(1e-8)
    close=torch.allclose(a,b,atol=atol,rtol=rtol)
    return dict(passed=bool(close and matching_inf),atol=atol,rtol=rtol,
        matching_negative_infinity_mask=matching_inf,
        max_abs_error=float(diff.max()) if diff.numel() else 0.,
        max_relative_error_reference_floor_1e_8=float((diff/scale).max()) if diff.numel() else 0.,
        max_error_over_combined_tolerance=float((diff/(atol+rtol*a[finite].abs())).max()) if diff.numel() else 0.)


def recurrent_sequence_comparison(torch,model,data,initial_hidden,steps=64):
    """Fixed synthetic observation sequences, independent hidden recursion.

    Every batching mode gets the same external inputs, no action-dependent
    world. Each mode feeds ITS OWN previous next_hidden into its next step.
    Thus differences cannot be erased by sharing the serial reference hidden.
    """
    from fly_touhou.thrl_policy_actions import policy_logits
    sequence_uniforms=np.random.default_rng(7102314).random((steps,16,1000))
    sequence=[]
    for t in range(steps):
        indices=(torch.arange(16,device='cuda')+t)%16
        sequence.append({k:v.index_select(0,indices) for k,v in data.items()})
    results={}
    for batch in (1,8,16):
        hidden=initial_hidden.clone();record={k:[] for k in ('logits','probabilities','next_hidden')}
        for observation in sequence:
            logits_list=[];hidden_list=[]
            for start in range(0,16,batch):
                logits,_,next_hidden=model.forward_step(
                    observation['features'][start:start+batch],observation['maps'][start:start+batch],
                    hidden[:,start:start+batch].contiguous(),
                    **({} if 'objects' not in observation else {'objects':observation['objects'][start:start+batch]}))
                logits_list.append(logits);hidden_list.append(next_hidden)
            logits=torch.cat(logits_list,0);hidden=torch.cat(hidden_list,1)
            record['logits'].append(logits.cpu())
            record['probabilities'].append(policy_logits(logits).softmax(-1).cpu())
            record['next_hidden'].append(hidden.cpu())
        results[batch]={k:torch.stack(v,0) for k,v in record.items()}
    comparisons={};reference=results[1]
    ref_probs=reference['probabilities'].double().numpy()
    for batch,actual in results.items():
        comparisons[str(batch)]={k:comparison(torch,reference[k],actual[k]) for k in reference}
        probs=actual['probabilities'].double().numpy()
        tv=.5*np.abs(probs-ref_probs).sum(-1)
        mismatches=[];maximum_cdf_difference=0.
        for t in range(steps):
            cdf=np.cumsum(probs[t],axis=-1);cdf[:,-1]=1.
            ref_cdf=np.cumsum(ref_probs[t],axis=-1);ref_cdf[:,-1]=1.
            maximum_cdf_difference=max(maximum_cdf_difference,float(np.abs(cdf-ref_cdf).max()))
            actions=np.sum(sequence_uniforms[t,:,:,None]>cdf[:,None,:],axis=-1)
            ref_actions=np.sum(sequence_uniforms[t,:,:,None]>ref_cdf[:,None,:],axis=-1)
            mismatches.append(int((actions!=ref_actions).sum()))
        masks=np.asarray([((np.arange(16)+t)%16)%2==0 for t in range(steps)])
        comparisons[str(batch)].update(
            strict_tensor_allclose=all(comparisons[str(batch)][k]['passed'] for k in reference),
            distribution_max_total_variation=float(tv.max()),distribution_mean_total_variation=float(tv.mean()),
            distribution_max_CDF_difference=maximum_cdf_difference,
            sampled_actions_compared=int(sequence_uniforms.size),sampled_actions_mismatch=sum(mismatches),
            action_mismatches_by_step=mismatches,
            sampled_action_match_rate=1.-sum(mismatches)/sequence_uniforms.size,
            argmax_mismatches=int(np.count_nonzero(probs.argmax(-1)!=ref_probs.argmax(-1))),
            zero_Bomb_probability_exact=bool(np.equal(probs[...,18][masks],0).all()),
            first_step_hidden_max_abs_error=float((actual['next_hidden'][0]-reference['next_hidden'][0]).abs().max()),
            final_step_hidden_max_abs_error=float((actual['next_hidden'][-1]-reference['next_hidden'][-1]).abs().max()))
    return dict(steps=steps,parallel_streams=16,unique_fixture_observations=16,
        input_sequence='Rotate the same 16 encoder-valid fixtures by one stream each time step',
        hidden_recursion='Each batching mode independently carries its own computed hidden; no resets',
        actual_native_trajectory=False,action_uniform_seed=7102314,uniform_draws_per_state=1000,
        comparisons=comparisons,
        distribution_parity_is_not_bitwise=True,
        theoretical_future_action_equality_claimed=False)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--initialization-report',type=Path,
        default=Path('./outputs/engineering/learning-revision-20261023/offline-CUDA-v1/report.json'))
    parser.add_argument('--output',type=Path,
        default=ROOT/'artifacts/engineering/learning-revision-20261023/evaluation-CUDA-v1')
    parser.add_argument('--repeats',type=int,default=20)
    parser.add_argument('--disable-cudnn-tf32',action='store_true',
        help='Diagnostic-only precision comparison; does not edit production settings')
    args=parser.parse_args()
    if not 10<=args.repeats<=30:raise ValueError('Use 10..30 timing repeats')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    import torch
    from fly_touhou.thrl_study_io import read_shared_json,utc_now
    from fly_touhou.atomic_json_io import atomic_write_json
    from fly_touhou.thrl_inputs import make_encoder,ORIGINAL_TIMERS_INPUT,SEPARATED_TIMERS_INPUT
    from fly_touhou.thrl_model import MOActorCritic
    from fly_touhou.thrl_policy_actions import policy_logits
    if not torch.cuda.is_available():raise RuntimeError('Offline verification requires CUDA')
    original_cudnn_tf32=torch.backends.cudnn.allow_tf32
    if args.disable_cudnn_tf32:torch.backends.cudnn.allow_tf32=False
    torch.set_num_threads(2)
    source=read_shared_json(args.initialization_report)
    if source.get('status')!='passed':raise ValueError('Initial models need their passed initialization report')
    snapshots=synthetic_snapshots()
    fixtures={}
    for schema in (ORIGINAL_TIMERS_INPUT,SEPARATED_TIMERS_INPUT):
        encoder=make_encoder(schema);observations=[encoder.encode(s) for s in snapshots]
        fixtures[schema]={k:np.stack([row[k] for row in observations]) for k in observations[0]}
    uniforms=np.random.default_rng(7102312).random((16,1000))
    records={}
    for group in ('A0','B0','M0'):
        spec=source['groups'][group];checkpoint=Path(spec['initial_checkpoint'])
        sha=digest(checkpoint)
        if sha!=spec['initial_sha256']:raise ValueError('Initialization checkpoint digest changed')
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        model=MOActorCritic(**payload['model_config'])
        model.load_state_dict(payload['model'],strict=True)
        config=model.model_config()
        if config!=payload['model_config']:raise ValueError('Model constructor changed checkpoint configuration')
        expected={n:t.clone() for n,t in model.state_dict().items()}
        model=model.to('cuda').eval().requires_grad_(False)
        observation=fixtures[model.input_schema]
        device_data={k:torch.from_numpy(v).to('cuda') for k,v in observation.items()}
        hidden=torch.as_tensor(np.random.default_rng(7102313).normal(0,.1,(1,16,model.gru_hidden_size)),
            dtype=torch.float32,device='cuda')
        # Four zero-state episodes plus nonzero independent GRU states exercise
        # batched state routing. MLP is required to return zero hidden state.
        hidden[:,:4]=0
        chunks={}
        for batch in (1,4,8,16):
            chunks[batch]=[(
                device_data['features'][start:start+batch],device_data['maps'][start:start+batch],
                hidden[:,start:start+batch].contiguous(),
                {} if 'objects' not in device_data else {'objects':device_data['objects'][start:start+batch]})
                for start in range(0,16,batch)]

        def forward(batch,join=False):
            rows=[]
            for features,maps,state,extra in chunks[batch]:
                values=model.forward_step(features,maps,state,**extra)
                if join:rows.append(values)
            if join:
                logits=torch.cat([r[0] for r in rows],0)
                return dict(logits=logits,probabilities=policy_logits(logits).softmax(-1),
                    values=torch.cat([r[1] for r in rows],0),next_hidden=torch.cat([r[2] for r in rows],1))

        differences={};timings={}
        with torch.inference_mode():
            # Warm every size before comparison/timing so kernels and allocators
            # have initialized; no transfer or state construction is timed.
            for batch in (1,4,8,16):
                for _ in range(5):forward(batch)
            torch.cuda.synchronize()
            reference=forward(1,True)
            reference_probs=reference['probabilities'].cpu().numpy().astype(np.float64)
            ref_cdf=np.cumsum(reference_probs,axis=-1);ref_cdf[:,-1]=1.
            ref_actions=np.sum(uniforms[:,:,None]>ref_cdf[:,None,:],axis=-1)
            for batch in (1,4,8,16):
                actual=forward(batch,True)
                differences[str(batch)]={key:comparison(torch,reference[key],actual[key]) for key in reference}
                probs=actual['probabilities'].cpu().numpy().astype(np.float64)
                cdf=np.cumsum(probs,axis=-1);cdf[:,-1]=1.
                actions=np.sum(uniforms[:,:,None]>cdf[:,None,:],axis=-1)
                differences[str(batch)].update(sampled_actions_compared=int(actions.size),
                    sampled_actions_mismatch=int((actions!=ref_actions).sum()),
                    sampled_action_match_rate=float(np.mean(actions==ref_actions)),
                    zero_Bomb_probability_exact=bool(torch.equal(actual['probabilities'][::2,18],
                        torch.zeros(8,device='cuda'))),
                    unmasked_Bomb_probability_positive=bool((actual['probabilities'][1::2,18]>0).all()))
            events={b:[] for b in (1,4,8,16)};walls={b:[] for b in events}
            peaks={b:[] for b in events}
            for repeat in range(args.repeats):
                # Rotate ordering to reduce fixed thermal/load ordering bias.
                order=(1,4,8,16);order=order[repeat%4:]+order[:repeat%4]
                for batch in order:
                    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                    started,ended=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    wall=time.perf_counter();started.record();forward(batch);ended.record();ended.synchronize()
                    walls[batch].append((time.perf_counter()-wall)*1000)
                    events[batch].append(started.elapsed_time(ended))
                    peaks[batch].append(torch.cuda.max_memory_allocated())
            for batch in events:
                values=np.asarray(events[batch]);timings[str(batch)]=dict(
                    states=16,forward_calls=16//batch,repeats=args.repeats,
                    CUDA_event_ms_mean=float(values.mean()),CUDA_event_ms_median=float(np.median(values)),
                    CUDA_event_ms_min=float(values.min()),CUDA_event_ms_max=float(values.max()),
                    CUDA_event_ms_p10=float(np.quantile(values,.1)),CUDA_event_ms_p90=float(np.quantile(values,.9)),
                    synchronized_wall_ms_median=float(np.median(walls[batch])),
                    peak_allocated_bytes=max(peaks[batch]),
                    median_forward_speedup_vs_serial=float(np.median(events[1])/np.median(values)))
            sequence_result=recurrent_sequence_comparison(torch,model,device_data,hidden,64)
        unchanged=all(torch.equal(expected[n],t.detach().cpu()) for n,t in model.state_dict().items())
        checkpoint_unchanged=digest(checkpoint)==sha
        passed=unchanged and checkpoint_unchanged and all(
            all(row[key]['passed'] for key in ('logits','probabilities','values','next_hidden'))
            and row['sampled_actions_mismatch']==0 and row['zero_Bomb_probability_exact']
            and row['unmasked_Bomb_probability_positive'] for row in differences.values())
        records[group]=dict(passed=passed,checkpoint=str(checkpoint),checkpoint_sha256=sha,
            checkpoint_unchanged=checkpoint_unchanged,model_tensors_unchanged=unchanged,
            model_config=config,parameter_counts=model.parameter_counts(),
            shapes={k:list(v.shape) for k,v in observation.items()},
            empty_object_states=(int(np.count_nonzero(observation['objects'][...,0].sum(-1)==0))
                if 'objects' in observation else None),
            full_96_object_states=(int(np.count_nonzero(observation['objects'][...,0].sum(-1)==96))
                if 'objects' in observation else None),
            comparisons=differences,timing_for_all_16_states=timings,
            recurrent_64step_comparison=sequence_result)
        print(group,'passed' if passed else 'FAILED',timings['16']['median_forward_speedup_vs_serial'],flush=True)
        del model,payload,expected,device_data,chunks,hidden,reference,actual
        torch.cuda.empty_cache()
    report=dict(schema='thrl-production-frozen-CUDA-batching-verification-v1',
        status='passed' if all(r['passed'] for r in records.values()) else 'failed',
        created_utc=utc_now(),initialization_report=str(args.initialization_report.resolve()),
        original_game_processes_started=0,training_jobs_started=0,optimizer_steps=0,
        inputs=dict(count=16,kind='synthetic snapshots through production timer/geometry encoders',
            native_measured=False,seed=7102311,raw_bullet_counts=[len(s.bullets) for s in snapshots],
            every_other_state_has_zero_Bombs=True,state_count=16),
        action_comparison=dict(uniform_seed=7102312,uniform_draws_per_state=1000,
            same_uniforms_for_all_batch_sizes=True,probability_dtype_before_CDF='float64 from FP32 softmax'),
        device=torch.cuda.get_device_name(0),torch_version=torch.__version__,CUDA_version=torch.version.cuda,
        precision=dict(dtype='float32',matmul_precision=torch.get_float32_matmul_precision(),
            matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
            original_cudnn_allow_tf32=original_cudnn_tf32,
            diagnostic_override_only=bool(args.disable_cudnn_tf32)),
        groups=records,limitations=[
            'Initialized production-size checkpoints only; not every future learned parameter state.',
            'FP32 agreement is tolerance-based; not bitwise equality.',
            '16000 sampled-action pairs per batching comparison do not guarantee arbitrary future action equality.',
            'Synthetic encoder-valid observations are not held-out original-game trajectories.',
            'CUDA events time resident forward spans including launch gaps; native stepping/reset/IPC/CPU encoding/transfers excluded.',
            'No claim about original-game wall-clock speedup or policy quality.'])
    atomic_write_json(output/'report.json',report)
    print(str(output/'report.json'),flush=True)
    if report['status']!='passed':raise SystemExit(1)


if __name__=='__main__':main()
