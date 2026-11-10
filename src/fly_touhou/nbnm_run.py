"""Run specification, model migration and paired NBNM evaluation helpers."""
from pathlib import Path
import copy
import json
import os
import time


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def task_settings(spec, task):
    """One shared policy can train native cards and real Story failure starts."""
    settings=copy.deepcopy(spec['nbnm']);settings['task']=dict(task)
    for group, candidate in spec.get('tasks',{}).items():
        if candidate==task:
            settings.update(copy.deepcopy(spec.get('task_entries',{}).get(group,{})))
            break
    return settings


def validate_spec(spec):
    from .nbnm_objective import Objective, qualifies_speed, rule_guidance_coefficient
    from .nbnm_sampling import group_block_budget
    from .nbnm_task import TaskTracker,entry_matches,fresh_opening
    from .nbnm_prefix import SCHEMA
    objective=Objective(**spec['nbnm']['objective'])
    rule_guidance_coefficient(spec['nbnm'])
    if spec['nbnm'].get('task_context','legacy') not in ('legacy','critic-only-v1'):
        raise ValueError('Unknown NBNM task-context protocol')
    retention=spec.get('rehearsal',{})
    if retention.get('coefficient',0)>0:
        from .nbnm_rehearsal import SuccessRehearsal
        SuccessRehearsal(retention['manifest'])
        if not 0<retention['coefficient']<=.3 or type(retention.get('every_updates')) is not int or retention['every_updates']<1:
            raise ValueError('Specify a small supervised retention coefficient and positive interval')
    if objective.phase=='speed':
        report=read(spec['speed_qualification_report'])
        if not qualifies_speed(report):raise ValueError('Speed phase requires frozen NBNM qualification')
        if report['task']!=spec['nbnm']['task']:raise ValueError('Qualify the same task before enabling speed')
    workers=spec['workers'];segment=spec['segment_decisions'];budget=spec['decision_budget']
    if not 1<=workers<=100 or segment%256 or budget%segment or budget<=0:
        raise ValueError('Use 1..100 workers and whole 256-row PPO segments')
    if len(spec['world_groups'])!=workers:raise ValueError('One task group per worker required')
    group_block_budget(spec['world_groups'],spec['group_weights'],segment//16)
    for task in spec['tasks'].values():TaskTracker(task)
    pools=spec.get('task_pools')
    if pools:
        if set(spec['world_groups'])!=set(pools):raise ValueError('Task pools must match worker groups')
        for group,keys in pools.items():
            if not keys or len(set(keys))!=len(keys) or any(k not in spec['tasks'] for k in keys):
                raise ValueError('Each pool requires distinct declared task keys')
    for group in spec.get('evaluation_tasks',[]):
        if group not in spec['tasks']:raise ValueError('Unknown final evaluation task '+group)
    from .nbnm_spell_practice import enabled,validate_task
    from . import nbnm_stage_practice as stage_practice
    for task in spec['tasks'].values():
        settings=task_settings(spec,task)
        task_objective=Objective(**settings['objective'])
        if task_objective.phase=='speed':
            report=read(settings.get('speed_qualification_report',spec.get('speed_qualification_report')))
            if not qualifies_speed(report) or report['task']!=task:
                raise ValueError('Speed qualification is per task, not transferable to other cards')
        native_stage=stage_practice.enabled(settings)
        if enabled(settings) or native_stage:
            if enabled(settings):validate_task(task)
            if native_stage and (task['mode']=='full_story' or task.get('end_stage',task['stage'])!=task['stage']):
                raise ValueError('Stage Practice cannot substitute for a cross-stage Story chain')
            if not Path(settings['score_template']).is_file():
                raise ValueError('Supply an existing native score record with the requested spells unlocked')
            from .thrl_training_mode import seed_split
            for split, seeds in settings['practice_evaluation_seeds'].items():
                if not seeds or len(set(seeds)) != len(seeds) or any(seed_split(s)!=split for s in seeds):
                    raise ValueError('Use unique native evaluation seeds belonging to their data split')
            if settings.get('entry_mode')!=stage_practice.PREFIX_MODE:continue
        if fresh_opening(task):continue
        bank=read(settings['start_bank'])
        if bank.get('schema')!=SCHEMA:raise ValueError('Prepare the native attack-start bank first')
        required_splits=(('train',) if native_stage and settings.get('prepare_evaluation_prefixes')
            else ('train','development'))
        for split in required_splits:
            entries=[r for r in bank['entries'] if entry_matches(r['task'],task) and r['split']==split]
            if not entries:raise ValueError(f'Missing {split} task starts: {task}')
        train={r['native_seed'] for r in bank['entries'] if r['split']=='train'}
        dev={r['native_seed'] for r in bank['entries'] if r['split']=='development'}
        if train&dev:raise ValueError('Training and development native seeds overlap')
    return dict(status='ready',network_training=False,workers=workers,
        decision_budget=budget,segment_decisions=segment,optimizer_updates_per_segment=segment//256,
        objective=objective.specification(),task_group_fractions=spec['group_weights'])


def runtime(output):
    from .runtime_cache_paths import configure_generated_caches,generated_kernel_cache_base
    from .thrl_job_paths import configure_native_temp
    configure_generated_caches()
    cache=generated_kernel_cache_base(Path(output))
    os.environ.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',FLY_TOUHOU_DESKTOP_NAME='fly-touhou-job-nbnm-'+str(os.getpid()),
        FLY_TOUHOU_FUSED_EDGE_GRAD='1',FLY_TOUHOU_EDGE_GRAD_CACHE=str(cache/'kernel-cache'),
        FLY_TOUHOU_DETERMINISTIC_SPMM='1',FLY_TOUHOU_SPMM_CACHE=str(cache/'spmm-kernel-cache'))
    temporary=configure_native_temp(output)
    import torch
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    return temporary


def environment(spec, *, evaluation=False):
    cfg=read(spec['environment_config'])
    # These old options describe resource-play, not an NBNM objective.
    for key in ('compact_lower_field_reward','compact_frozen_prefix_replay','compact_frozen_decoding',
                'compact_capture_stage6_prefix','compact_conditional_wall_guidance_coefficient'):
        cfg.pop(key,None)
    cfg.update(seed=spec['seed'],nbnm=copy.deepcopy(spec['nbnm']),
        compact_full_native_teacher_support=spec['nbnm'].get('teacher_mode') in ('resource-movement-v1','nbnm-route-v1'),
        compact_protected_exit_forecast=False)
    if not evaluation:
        from .nbnm_course import worker_settings
        rows=worker_settings(spec)
        cfg['nbnm']['world_tasks']=[r[0] for r in rows]
        cfg['nbnm']['world_entries']=[r[1] for r in rows]
    return cfg


def load_model(spec, checkpoint=None):
    import torch
    from .compact_story_model import from_story_checkpoint
    from .nbnm_model import configure_model
    from .compact_full_story import apply_sensor_patches
    point=torch.load(checkpoint or spec['initial_checkpoint'],map_location='cpu',weights_only=False)
    if 'model_payload' in point:point=point['model_payload']
    migrated=not bool(point.get('nbnm'))
    if not migrated and point['nbnm']['risk_mode']!=spec['nbnm']['risk_mode']:
        raise ValueError('Use matching route/baseline input checkpoints')
    model=from_story_checkpoint(spec['connectome'],point)
    configure_model(model,spec['nbnm'],reset_critics=migrated)
    apply_sensor_patches(model,environment(spec))
    model.core.sparse.deterministic_spmm=True
    return model.eval(),migrated


def atomic_torch_save(value,path):
    import torch
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.writing')
    with temporary.open('wb') as stream:
        torch.save(value,stream);stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)


def evaluation_seeds(spec, *, split='development', task=None):
    from .nbnm_spell_practice import enabled
    task=task or spec['nbnm']['task']
    settings=task_settings(spec,task)
    from .nbnm_stage_practice import ENTRY_MODE,PREFIX_MODE
    if (enabled(settings) or settings.get('entry_mode')==ENTRY_MODE or
            (settings.get('entry_mode')==PREFIX_MODE and settings.get('prepare_evaluation_prefixes'))):
        return list(settings['practice_evaluation_seeds'][split])
    from .nbnm_task import fresh_opening
    if fresh_opening(task):
        return list(spec['full_story_evaluation_seeds'] if split=='development' else spec['full_story_heldout_seeds'])
    bank=read(settings['start_bank'])
    from .nbnm_task import entry_matches
    return sorted({r['native_seed'] for r in bank['entries'] if r['split']==split and entry_matches(r['task'],task)})


def evaluate(model,spec,output,*,task=None,split='development',progress=lambda x:None,guard=lambda:None):
    """Frozen actual-game task attempts; a prefix is never a six-stage clear."""
    import numpy as np
    import torch
    from . import thrl_vector
    from .compact_full_story import factory,story_kwargs
    from .nbnm_prefix import burn_in
    from .spiking_engineered_control import SpikingState
    from .thrl_durable_records import write_json,append_jsonl
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    cfg=environment(spec,evaluation=True);task=dict(task or spec['nbnm']['task'])
    cfg['nbnm']=task_settings(spec,task)
    seeds=evaluation_seeds(spec,split=split,task=task)
    if cfg['nbnm'].get('entry_mode')=='native_stage_prefix' and cfg['nbnm'].get('prepare_evaluation_prefixes'):
        from .nbnm_start_cache import prepare_evaluation_bank
        cfg['nbnm']['start_bank']=prepare_evaluation_bank(spec,cfg['nbnm'],task,seeds,split,output,
            progress=progress,guard=guard)
    budget=len(seeds);workers=min(int(spec.get('evaluation_workers',4)),budget)
    if not 1<=workers<=100:raise ValueError('Evaluation needs 1..100 workers and at least one seed')
    results=[];entries={};begun=time.monotonic()
    decisions=0
    old=thrl_vector.worker_factory;thrl_vector.worker_factory=factory;vector=None;graphs=None
    model.eval()
    try:
        if spec.get('cuda_graph_inference',False) and model.device.type=='cuda':
            from .nbnm_cuda import InferenceGraphs
            graphs=InferenceGraphs(model,sample=False)
        vector=thrl_vector.THRLNativeVector(story_kwargs(cfg,output,workers,split,teacher=False),output/'workers',timeout=180)
        obs,infos=vector.reset(options=[dict(native_seed=s) for s in seeds[:workers]])
        # Vector observations are immutable shared snapshots. The evaluator
        # keeps its own writable staging batch while workers finish out of order.
        obs={key:np.array(value,copy=True) for key,value in obs.items()}
        def record_entry(index,info):
            entries[str(info['native_seed'])]=dict(features=obs['features'][index].tolist(),
                policy_burn_in=info.get('policy_burn_in'))
            write_json(output/'entry-observations.json',entries)
        for index,info in enumerate(infos):record_entry(index,info)
        state=SpikingState.pack([burn_in(model,info,model.initial_state(1)) for info in infos])
        pending=list(range(workers));next_seed=workers
        with torch.no_grad():
            while pending or vector._inflight:
                guard()
                if pending:
                    ids=torch.tensor(pending,device=model.device)
                    sub=SpikingState(*(v.index_select(0,ids) for v in state.tensors()))
                    features=np.stack([obs['features'][i] for i in pending])
                    predict=graphs if graphs is not None else model.step
                    out,following,_=predict(torch.as_tensor(features,device=model.device),sub)
                    for target,value in zip(state.tensors(),following.tensors()):target.index_copy_(0,ids,value)
                    vector.submit_at(pending,'step',out['actions'].cpu().tolist());pending=[]
                messages=vector.receive_ready(max_wait_ms=1,min_batch=4,check_stop=guard)
                for msg in messages:
                    i=msg['index'];info=msg['info']
                    for k,value in msg['observation'].items():obs[k][i]=value
                    if msg['operation']=='reset':
                        record_entry(i,info)
                        fresh=burn_in(model,info,model.initial_state(1))
                        for target,value in zip(state.tensors(),fresh.tensors()):target[i:i+1].copy_(value)
                        pending.append(i);continue
                    decisions+=1
                    if decisions%128==0:
                        progress(dict(phase='frozen_NBNM_evaluation',completed=len(results),target=budget,
                            successes=sum(r['nbnm_success'] for r in results),decisions=decisions,
                            last_stage=info.get('stage'),last_stage_frame=info.get('stage_frame')))
                    if msg['truncated']:raise RuntimeError('Incomplete evaluation is not an NBNM failure or success')
                    if msg['terminated']:
                        row=dict(info['episode_summary'],evaluation_split=split)
                        results.append(row);append_jsonl(output/'episodes.jsonl',row)
                        progress(dict(phase='frozen_NBNM_evaluation',completed=len(results),target=budget,
                            successes=sum(r['nbnm_success'] for r in results)))
                        if next_seed<budget:
                            vector.submit_at([i],'reset',options=[dict(native_seed=seeds[next_seed])]);next_seed+=1
                    else:pending.append(i)
        report=dict(schema='frozen-NBNM-task-evaluation-v1',frozen=True,task=task,
            entry_mode=cfg['nbnm'].get('entry_mode','story_prefix'),
            opening_seeds=seeds,action_selection='joint_mode',evaluation_split=split,
            episode_results=sorted(results,key=lambda r:r['native_seed']),
            attempts=len(results),successes=sum(r['nbnm_success'] for r in results),
            spell_captures=sum(r.get('spell_captured',False) for r in results),
            spell_survived_timeouts=sum(r.get('spell_survived_timeout',False) for r in results),
            successful_mean_native_frames=(float(np.mean([r['task_native_frames'] for r in results if r['nbnm_success']]))
                if any(r['nbnm_success'] for r in results) else None),
            elapsed_seconds=time.monotonic()-begun,training_updates=0)
        report['cuda_graph_inference']=graphs is not None
        report['graph_capture_seconds']=graphs.capture_seconds if graphs is not None else 0.
        write_json(output/'report.json',report)
        return report
    finally:
        if graphs is not None:graphs.clear()
        if vector is not None:vector.close()
        thrl_vector.worker_factory=old
