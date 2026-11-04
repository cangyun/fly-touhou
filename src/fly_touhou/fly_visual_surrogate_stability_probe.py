"""Prepared theta0 replay: one backward-only .01 versus .003 candidate.

Import/default preparation is NoTorch. CUDA execution is explicit and blocked
until the bound original native job has ended with recorded owned cleanup.
Nothing here trains, samples an action, queries a teacher, or adopts a scale.
"""
from __future__ import annotations

from contextlib import contextmanager
import datetime
import hashlib
import json
import os
from pathlib import Path
import pickletools
import shutil
import time
import zipfile

SCHEMA = 'fly-theta0-backward-only-surrogate01-vs003-diagnostic-v1'
ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = Path('./outputs/distillation/D95-fly-Kickstarting-PPO30-continuous983040-v1')
JOB_PID, JOB_CTIME = 52700, 1793018980.6414654
SCALES = (.01,.003)
SOURCE_NAMES = ('fly_visual_surrogate_stability_probe.py','fly_visual_credit_horizon_probe.py',
    'fly_visual_actor_critic.py','fly_visual_distillation.py','fly_visual_ppo_loss.py','fly_visual_ppo_update.py',
    'fly_visual_ppo_cuda_graph_checkpoint.py','fly_visual_ppo_cuda_graph.py','whole_brain.py','cached_cusparse.py')


def file_sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def write_json(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(temporary,path)


def checkpoint_scalars(path):
    """Inspect pickle opcodes, never unpickle/execute or read tensor payloads."""
    wanted={'surrogate_scale','surrogate_slope_per_mv','substeps','dt_ms','recurrent_gain_limit'}
    result={}
    with zipfile.ZipFile(path) as archive:
        names=[n for n in archive.namelist() if n.endswith('/data.pkl')]
        if len(names)!=1:raise ValueError('One saved parameter metadata pickle required')
        ops=list(pickletools.genops(archive.read(names[0])))
    for i,(op,arg,pos) in enumerate(ops):
        if isinstance(arg,str) and arg in wanted:
            candidates=[a for o,a,_ in ops[i+1:i+5] if o.name in ('BINFLOAT','BININT','BININT1','BININT2')]
            if not candidates:raise ValueError('Unrecognized saved scalar:'+arg)
            if arg in result and result[arg]!=candidates[0]:raise ValueError('Conflicting saved scalar:'+arg)
            result[arg]=candidates[0]
    if result!=dict(surrogate_scale=.01,surrogate_slope_per_mv=.3,substeps=100,dt_ms=.2,recurrent_gain_limit=.1):
        raise ValueError('Require actual unchanged .01/.3,100*.2ms,.1gain theta0 metadata')
    return result


def inspect_trace(path,metadata):
    import numpy as np  # No Torch/model/native import.
    shapes={n:(32,30) for n in ('actions','old_log_probs','values','rewards','terminated','truncated',
        'episode_starts','game_ids','native_episodes','brain_rng_keys','noise_counters')}
    shapes.update(encoded_visual=(32,30,384),teacher_probs=(32,30,19),legal_actions=(32,30,19),
                  raw_logits=(32,30,19),bootstrap_values=(30,))
    with np.load(path,allow_pickle=False) as archive:
        data={k:archive[k] for k in shapes}
    for name,value in data.items():
        if value.shape!=shapes[name]:raise ValueError('Actual first-rollout shape differs:'+name)
        if value.dtype.kind=='f' and not np.isfinite(value).all():raise ValueError('Nonfinite actual trace:'+name)
        expected=np.bool_ if name in ('legal_actions','terminated','truncated','episode_starts') else np.int64 if name in (
            'actions','game_ids','native_episodes','brain_rng_keys','noise_counters') else np.float32
        if value.dtype!=expected:raise ValueError('Actual trace dtype differs:'+name)
    m=metadata['metadata']
    if (m.get('policy_version')!=0 or m.get('neural_batch_size')!=16 or m.get('steps')!=32 or m.get('worlds')!=30
            or m.get('same_neural_batch_in_collection_and_replay') is not True or m.get('PPO_optimizer_updates')!=0
            or m.get('teacher_actions_executed')!=0 or m.get('synthetic_fixture') is not False):
        raise ValueError('Actual initial unchanged-theta B16/30-world record required')
    if not data['episode_starts'][0].all() or (data['noise_counters'][0]!=0).any() or data['truncated'].any():
        raise ValueError('Missing real h0/new natural games; cannot invent intermediate states')
    if not np.array_equal(data['episode_starts'][1:],data['terminated'][:-1]):
        raise ValueError('Only true natural terminal may reset a recorded next row')
    for t in range(1,32):
        starts=data['episode_starts'][t]
        if not np.array_equal(data['noise_counters'][t],np.where(starts,0,data['noise_counters'][t-1]+100)):
            raise ValueError('Recorded row counters do not reconstruct from h0')
        if (data['brain_rng_keys'][t,~starts]!=data['brain_rng_keys'][t-1,~starts]).any():
            raise ValueError('Own game keys changed without a natural new game')
    if not data['legal_actions'].any(-1).all() or (data['actions']<0).any() or (data['actions']>=19).any():
        raise ValueError('Actual source19 inventory support required')
    if not np.take_along_axis(data['legal_actions'],data['actions'][...,None],axis=-1).all():
        raise ValueError('Recorded student action outside legal support')
    p=data['teacher_probs']
    if (p<0).any() or not np.allclose(p.sum(-1),1,rtol=0,atol=2e-6) or p[~data['legal_actions']].any():
        raise ValueError('Actual final legal teacher distribution required')
    return dict(rows=960,steps=32,worlds=30,encoded_shape=list(data['encoded_visual'].shape),
        all_row0_natural_starts=True,initial_counters_zero=True,keys_and_counters_continuous=True,
        all_finite=True,whole_state_independently_saved_in_original_trace=False,
        teacher_temperature=m['teacher_temperature'],student_temperature=m['action_temperature'])


def prepare_pack(run_root,output):
    """Default CPU inspection and immutable byte copies; stops on insufficient data."""
    run_root,output=Path(run_root).resolve(),Path(output).resolve()
    if output.exists():raise FileExistsError('New diagnostic pack directory required; no silent reuse')
    launch=json.loads((run_root/'launch.json').read_text(encoding='utf-8'))
    initial=json.loads((run_root/'initial.json').read_text(encoding='utf-8'))
    trace_meta=json.loads((run_root/'traces/rollout-000.json').read_text(encoding='utf-8'))
    if (launch.get('PID')!=JOB_PID or launch.get('create_time')!=JOB_CTIME
            or launch.get('neural_batch_size')!=16 or launch.get('no_auxiliary') is not True
            or initial.get('costs',{}).get('completed_rollouts')!=0 or initial['costs'].get('Adam_steps')!=0):
        raise ValueError('Require fixed job identity and actual pre-first-update theta0')
    model=run_root/'initial-parameters.pt';trace=run_root/'traces/rollout-000.npz'
    if file_sha(model)!=initial['sha256'] or file_sha(trace)!=trace_meta['trace_sha256']:
        raise ValueError('Initial parameter/actual first-trace SHA mismatch')
    scalars=checkpoint_scalars(model);inspection=inspect_trace(trace,trace_meta)
    inputs={n:dict(source=str(run_root/n),sha256=file_sha(run_root/n),bytes=(run_root/n).stat().st_size)
        for n in ('launch.json','initial.json','initial-parameters.pt','traces/rollout-000.json','traces/rollout-000.npz')}
    output.mkdir(parents=True,exist_ok=False)
    try:
        for name,spec in inputs.items():
            target=output/'inputs'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(spec['source'],target)
            if file_sha(target)!=spec['sha256']:raise IOError('Diagnostic byte-copy SHA mismatch:'+name)
        manifest=dict(schema=SCHEMA,status='prepared_not_executed',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            run_root=str(run_root),original_job=dict(pid=JOB_PID,create_time=JOB_CTIME),inputs=inputs,
            source_sha256={n:file_sha(ROOT/'src/fly_touhou'/n) for n in SOURCE_NAMES},
            entry_sha256=file_sha(ROOT/'scripts/probe_fly_visual_surrogate_stability.py'),
            scalars=scalars,inspection=inspection,arms=[dict(scale=s,horizon=h) for s in SCALES for h in (4,16)],
            loss_rows=[28,29,30,31],shared_rebuilt_boundary=16,T4_rebuilt_boundary=28,
            full_GAE_real_rows=960,neural_width=16,diagnostic_real_group=0,
            teacher_queries=0,native_actions=0,optimizer_updates=0,Torch_imported=False,GPU_executed=False,
            auto_adoption=False,requires_original_job_terminal_and_cleaned=True)
        write_json(output/'manifest.json',manifest)
        write_json(output/'COMMITTED.json',dict(schema=SCHEMA,manifest_sha256=file_sha(output/'manifest.json'),
            complete_diagnostic_inputs=True,GPU_executed=False,optimizer_updates=0))
        return manifest
    except Exception as error:
        write_json(output/'preparation-failure.json',dict(schema=SCHEMA,error=repr(error),no_retry=True))
        raise


def verify_pack(pack):
    pack=Path(pack);commit=json.loads((pack/'COMMITTED.json').read_text(encoding='utf-8'))
    if commit.get('schema')!=SCHEMA or file_sha(pack/'manifest.json')!=commit.get('manifest_sha256'):
        raise ValueError('Prepared diagnostic manifest identity differs')
    manifest=json.loads((pack/'manifest.json').read_text(encoding='utf-8'))
    for name,spec in manifest['inputs'].items():
        if file_sha(pack/'inputs'/name)!=spec['sha256']:raise ValueError('Immutable diagnostic input differs:'+name)
    for name,digest in manifest['source_sha256'].items():
        if file_sha(ROOT/'src/fly_touhou'/name)!=digest:raise ValueError('Pinned diagnostic implementation changed:'+name)
    if file_sha(ROOT/'scripts/probe_fly_visual_surrogate_stability.py')!=manifest['entry_sha256']:
        raise ValueError('Pinned diagnostic entry changed')
    return manifest


def require_original_job_cleaned(manifest):
    import psutil
    job=manifest['original_job']
    try:
        if abs(psutil.Process(job['pid']).create_time()-job['create_time'])<1e-6:
            raise RuntimeError('Bound original job is still active; no extra GPU work is authorized')
    except psutil.NoSuchProcess:pass
    run=Path(manifest['run_root'])
    terminals=[p for p in (run/'result.json',run/'failure.json') if p.exists()]
    if len(terminals)!=1:raise RuntimeError('Original job needs one authoritative completed/failed terminal receipt')
    terminal=json.loads(terminals[0].read_text(encoding='utf-8'))
    if terminal.get('status') not in ('completed','failed','stopped_finite_or_resource_budget'):
        raise RuntimeError('Original job terminal is not complete/failed')
    clean=[p for p in (run/'training-shutdown.json',run/'training-cleanup.json') if p.exists()]
    if not clean or any(json.loads(p.read_text(encoding='utf-8')).get('close_errors')!=[] for p in clean):
        raise RuntimeError('Successful owned-world cleanup has not been recorded')
    return dict(original_job_absent=True,terminal=str(terminals[0]),terminal_sha256=file_sha(terminals[0]),
        cleanup_sha256={str(p):file_sha(p) for p in clean})


@contextmanager
def temporary_surrogate_scale(brain,scale):
    """Backward-only attribute, restored even when a diagnostic fails."""
    if scale not in SCALES:raise ValueError('Only predeclared .01/.003 diagnostic scales')
    previous=brain.surrogate_scale
    try:
        brain.surrogate_scale=scale
        yield
    finally:brain.surrogate_scale=previous


def execute_cuda(pack,*,max_seconds,connectome):
    """Future explicit single read-only GPU process; no native/optimizer work."""
    pack=Path(pack);manifest=verify_pack(pack);cleanup=require_original_job_cleaned(manifest)
    if max_seconds is None or max_seconds<1:raise ValueError('Explicit finite diagnostic wall budget required')
    if os.environ.get('CUDA_VISIBLE_DEVICES') in (None,'','-1'):raise ValueError('Caller-selected CUDA device required')
    output=pack/'execution'
    if output.exists():raise FileExistsError('No implicit diagnostic retry/reuse')
    output.mkdir();begun=time.monotonic()
    import psutil
    process=psutil.Process();process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);process.cpu_affinity(process.cpu_affinity()[:2])
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import numpy as np
    import torch
    from .fly_visual_actor_critic import VisualFlyActorCritic
    from .fly_visual_ppo_loss import categorical_policy,scalar_gae,visual_ppo_loss,VisualPPOLossConfig
    from .fly_visual_ppo_update import _clone_state
    from .fly_visual_ppo_cuda_graph import tensor_reset_state
    from .fly_visual_ppo_cuda_graph_checkpoint import _brain_call
    from .fly_visual_credit_horizon_probe import _parameters,_signature
    if torch.cuda.is_initialized():raise RuntimeError('Fresh future CUDA process required')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True);torch.cuda.init()
    report=dict(schema=SCHEMA,status='initializing',PID=os.getpid(),create_time=process.create_time(),cleanup=cleanup,
        optimizer_updates=0,native_actions=0,teacher_queries=0,auto_adoption=False,original_trace_full_states_saved=False,
        costs={'reconstruction':dict(base_neural_lane_forwards=0,checkpoint_recompute_neural_lane_forwards=0,backward_attempts=0,backwards_completed=0)})
    write_json(output/'launch.json',dict(report,manifest_sha256=file_sha(pack/'manifest.json'),max_seconds=max_seconds,
        GPU_fraction=None,no_artificial_RAM_or_VRAM_cap=True))
    model=None;named=None;signature=None;original_scale=None
    original_cpu_rng=torch.get_rng_state().clone();original_cuda_rng=torch.cuda.get_rng_state().clone()
    def guard():
        if (pack/'STOP').exists():raise RuntimeError('Operator STOP')
        if time.monotonic()-begun>max_seconds:raise TimeoutError('Explicit diagnostic wall budget reached')
    def digest_state(state):
        from dataclasses import fields
        result={}
        for field in fields(state):
            value=getattr(state,field.name);values=value if field.name=='delay_queue' else (value,)
            digests=[]
            for x in values:
                cpu=x.detach().cpu().contiguous();h=hashlib.sha256(str((tuple(cpu.shape),str(cpu.dtype))).encode());h.update(cpu.numpy().tobytes());digests.append(h.hexdigest())
            result[field.name]=digests
        return result
    try:
        guard();initial=pack/'inputs/initial-parameters.pt'
        model=VisualFlyActorCritic.from_actor_critic_checkpoint(initial,connectome,expected_sha256=file_sha(initial),device='cpu').to('cuda:0')
        model.eval();named=_parameters(model);signature=_signature(named);original_scale=model.brain.surrogate_scale
        if original_scale!=.01 or model.brain.surrogate_slope_per_mv!=.3:raise ValueError('Actual loaded baseline scale differs')
        with np.load(pack/'inputs/traces/rollout-000.npz',allow_pickle=False) as archive:
            data={k:torch.from_numpy(archive[k].copy()) for k in archive.files if k not in ('features','reward_vectors','reward_raw_vectors','action_probs','native_episodes')}
        launch=json.loads((pack/'inputs/launch.json').read_text(encoding='utf-8'));cfg=launch['update_config'];loss_cfg=VisualPPOLossConfig(**cfg['loss'])
        adv,ret=scalar_gae(data['rewards'],data['values'],data['bootstrap_values'],data['terminated'],
            truncated=data['truncated'],episode_starts=data['episode_starts'],gamma=cfg['gamma'],gae_lambda=cfg['gae_lambda'])
        adv=(adv-adv.mean())/adv.std(unbiased=False).clamp_min(1e-8)
        group_data={k:v[:,:16].to(model.device).clone() for k,v in data.items() if v.ndim>=2 and v.shape[1]==30}
        group_data.update(advantages=adv[:,:16].to(model.device),returns=ret[:,:16].to(model.device))
        fresh=model.initial_state(16,rng_seed=0);snapshots={};state_hashes={};checks=[]
        # A unique deterministic reconstruction from REAL initial counter0, not
        # invented late zero-state. Both groups verify ALL actual32x30 logits.
        for group,workers in enumerate((list(range(16)),list(range(16,30)))):
            state=model.initial_state(16,rng_seed=0)
            for t in range(32):
                guard();n=len(workers)
                def pad(x,fill=0):return torch.cat((x,x.new_full((16-n,*x.shape[1:]),fill)),0) if n<16 else x
                keys=pad(data['brain_rng_keys'][t,workers]).to(model.device)
                starts=pad(data['episode_starts'][t,workers],True).to(model.device)
                state=tensor_reset_state(state,fresh,starts,keys)
                if not torch.equal(state.rng_keys[:n],keys[:n]) or not torch.equal(state.noise_counter[:n],data['noise_counters'][t,workers].to(model.device)):
                    raise ValueError('Real h0 reconstruction key/counter mismatch')
                if group==0 and t in (16,28):snapshots[t]=_clone_state(state,'cpu')
                with torch.no_grad():raw,state=model.brain._forward_validated(pad(data['encoded_visual'][t,workers]).to(model.device),state)
                report['costs']['reconstruction']['base_neural_lane_forwards']+=16
                actual_raw=data['raw_logits'][t,workers];actual_logp=data['old_log_probs'][t,workers]
                logp=categorical_policy(raw[:n],data['legal_actions'][t,workers].to(model.device)).log_prob(data['actions'][t,workers].to(model.device))
                raw_exact=torch.equal(raw[:n].cpu(),actual_raw);logp_exact=torch.equal(logp.cpu(),actual_logp)
                checks.append(dict(t=t,group=group,raw_exact=raw_exact,old_logp_exact=logp_exact))
                if not raw_exact or not logp_exact:raise ValueError('Actual theta0 recorded forward is not exact; no synthetic boundary allowed')
                if group==0 and t>=16:state_hashes[t]=digest_state(state)
        report['reconstruction']=dict(all960_actual_raw_and_oldlogp_exact=True,checks=checks,
            complete_state_comparison_scope='Reconstructed theta0 reference vs diagnostic arms; original trace has no independent full boundary states')
        arm_results={}
        for scale in SCALES:
            for horizon in (4,16):
                tag=f'scale{scale:g}_T{horizon}';counts=dict(base_neural_lane_forwards=0,checkpoint_recompute_neural_lane_forwards=0,backward_attempts=0,backwards_completed=0)
                report['costs'][tag]=counts
                with temporary_surrogate_scale(model.brain,scale),torch.enable_grad():
                    state=_clone_state(snapshots[32-horizon],model.device);raws=[];values=[];state_exact=[]
                    for t in range(32-horizon,32):
                        guard();state=tensor_reset_state(state,fresh,group_data['episode_starts'][t],group_data['brain_rng_keys'][t])
                        raw,state=_brain_call(model,group_data['encoded_visual'][t],state,counts,True)
                        exact=digest_state(state)==state_hashes[t];state_exact.append(exact)
                        if not exact or not torch.equal(raw.detach().cpu(),data['raw_logits'][t,:16]):
                            raise ValueError('Backward-only candidate changed hardforward/state; stop diagnostic')
                        if t>=28:raws.append(raw);values.append(model.critic(group_data['encoded_visual'][t]))
                    raw,values=torch.stack(raws),torch.stack(values);tail=slice(28,32)
                    logp=categorical_policy(raw,group_data['legal_actions'][tail]).log_prob(group_data['actions'][tail])
                    if not torch.equal(logp.detach().cpu(),data['old_log_probs'][tail,:16]):raise ValueError('Arm actual oldlogp differs')
                    terms=visual_ppo_loss(raw,values,group_data['actions'][tail],group_data['old_log_probs'][tail],group_data['values'][tail],
                        group_data['advantages'][tail],group_data['returns'][tail],group_data['teacher_probs'][tail],group_data['legal_actions'][tail],
                        model.brain.recurrent_gain_logits,config=loss_cfg)
                    guard();counts['backward_attempts']+=1;gradients=torch.autograd.grad(terms['total_loss'],[p for _,p in named]);counts['backwards_completed']+=1
                    if any(not torch.isfinite(g).all() for g in gradients):raise FloatingPointError('Nonfinite diagnostic gradient; no adoption')
                    arm_results[tag]=dict(gradients=tuple(g.detach().cpu().clone() for g in gradients),terms={k:float(v.detach()) for k,v in terms.items()},
                        all_following_state_fields_bitwise_exact=all(state_exact),actual_last4_logp_exact=True,scale=scale,horizon=horizon)
                del state,raw,raws,values,terms,gradients
        def comparison(a,b):
            result={}
            for (name,_),left,right in zip(named,arm_results[a]['gradients'],arm_results[b]['gradients']):
                x,y=left.double().flatten(),right.double().flatten();nx,ny=float(x.norm()),float(y.norm());d=y-x
                result[name]=dict(left_norm=nx,right_norm=ny,difference_norm=float(d.norm()),max_abs_difference=float(d.abs().max()),
                    cosine=None if not nx or not ny else float(torch.dot(x,y)/(nx*ny)),exact=torch.equal(left,right))
            return result
        report['comparisons']={
            'baseline_T4_vs_T16':comparison('scale0.01_T4','scale0.01_T16'),
            'candidate_T4_vs_T16':comparison('scale0.003_T4','scale0.003_T16'),
            'T4_baseline_vs_candidate':comparison('scale0.01_T4','scale0.003_T4'),
            'T16_baseline_vs_candidate':comparison('scale0.01_T16','scale0.003_T16')}
        report['arms']={tag:{k:v for k,v in arm.items() if k!='gradients'} for tag,arm in arm_results.items()}
        if len({arm['terms']['total_loss'] for arm in arm_results.values()})!=1:raise ValueError('The four same-row loss values differ')
        report.update(status='completed_readonly_four_arms',interpretation='Lower surrogate gradient norm is not learned performance; no automatic adoption')
    except Exception as error:
        report.update(status='diagnostic_unavailable',error=repr(error),partial_neural_call_may_be_unmeasured=True,no_retry=True)
    finally:
        try:
            if model is not None:
                if original_scale is not None:model.brain.surrogate_scale=original_scale
                report['surrogate_scale_restored']=original_scale is not None and model.brain.surrogate_scale==original_scale
                report['parameter_and_existing_grad_storage_unchanged']=named is not None and _signature(named)==signature
                model.brain.clear_cuda_capture();model.brain.clear_cached_propagation()
            torch.set_rng_state(original_cpu_rng);torch.cuda.set_rng_state(original_cuda_rng)
            report['process_global_CPU_and_CUDA_rng_restored']=torch.equal(torch.get_rng_state(),original_cpu_rng) and torch.equal(torch.cuda.get_rng_state(),original_cuda_rng)
        except Exception as error:report['cleanup_or_RNG_restoration_error']=repr(error)
        report['elapsed_seconds']=time.monotonic()-begun
        report['extra_neural_lane_forwards']=sum(c['base_neural_lane_forwards']+c['checkpoint_recompute_neural_lane_forwards'] for c in report['costs'].values())
        report['extra_backwards_completed']=sum(c['backwards_completed'] for c in report['costs'].values())
        write_json(output/'result.json',report)
    return report
