"""Read-only danger/action/stage diagnostics of the completed pre-PPO BC.

Recompute only its six existing whole-trajectory heldout encodings. No policy
optimization, native game, new seed partition, or new model selection occurs.
Teacher action agreement and input-risk proxies are not native avoidance.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time
import traceback

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def run(a):
    sys.path.insert(0,str(a.runtime/'src'))
    import numpy as np,psutil,torch
    from fly_touhou.thrl_study_io import read_shared_json as read,utc_now
    from fly_touhou.atomic_json_io import atomic_write_json as write
    from fly_touhou.thrl_full_teacher_data import FullTeacherDataset
    from fly_touhou.thrl_teacher_latents import TeacherLatentCache
    from fly_touhou.thrl_cached_actor_fit import begin_cached_actor_phase,forward_cached_actor
    from fly_touhou.thrl_actor_transfer import validate_actor_payload
    from fly_touhou.thrl_pair_policy import model_from_config
    from fly_touhou.thrl_policy_actions import policy_logits
    if a.output.exists():raise FileExistsError(a.output)
    if psutil.virtual_memory().available<3*2**30:raise RuntimeError('Less than3GiB host reserve;defer diagnostic')
    if not torch.cuda.is_available():raise RuntimeError('CUDA diagnostic required;no training CPU fallback')
    if sha(a.runtime/'source_manifest.json')!='3b8c9d6a506e9cc2cae282a095f627882c9beedbb0b0f379e22bdf5d9ce887c0':
        raise ValueError('Qualified model/data-cache runtime changed')
    result=read(a.fit/'result.json');bc_path=Path(result['policy'])
    if result['status']!='completed' or sha(bc_path)!=result['policy_sha256']:
        raise ValueError('Completed selected BC policy required')
    bc=torch.load(bc_path,map_location='cpu',weights_only=False)
    if sha(a.parent)!=bc['parent_sha256']:raise ValueError('Pre-BC ancestor changed')
    parent=torch.load(a.parent,map_location='cpu',weights_only=False)
    validate_actor_payload(torch,bc,result,parent)
    dataset=FullTeacherDataset(a.dataset)
    if dataset.manifest_sha256!=bc['teacher_manifest_sha256']:raise ValueError('Actual teacher archive changed')
    split=read(a.fit/'split.json');expected_split=dataset.split(seed=31)
    if split!=expected_split or len(split['heldout'])!=6:raise ValueError('The original whole-seed split must remain fixed')
    a.output.mkdir(parents=True,exist_ok=False);a.output_created=True
    write(a.output/'launch.json',dict(pid=os.getpid(),create_time=psutil.Process().create_time(),utc=utc_now(),
        CUDA_required=True,optimizer_updates=0,native_worlds_started=0,read_only_diagnostic=True))
    (a.output/'diagnostic_source.py').write_bytes(Path(__file__).read_bytes())
    psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
    torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.06)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    started=time.monotonic()
    def stop():
        if (a.output/'STOP').exists():raise RuntimeError('Operator stopped this read-only diagnostic')
        if time.monotonic()-started>900:raise TimeoutError('Finite900second diagnostic budget exceeded')
        time.sleep(.003)
    def progress(row):write(a.output/'progress.json',dict(row,utc=utc_now(),optimizer_updates=0,native_worlds_started=0))
    class HeldoutView:
        def __init__(self):
            self.original_indices=split['heldout'];self.records=[];offset=0
            for i in self.original_indices:
                r=dataset.records[i]
                self.records.append({**r,'index':len(self.records),'offset':offset});offset+=r['rows']
            self.rows=offset;self.stage=np.empty(offset,np.uint8)
            self.masks={k:np.zeros(offset,bool) for k in (
                'all','normal_unprotected','protected_or_non_normal','teacher_moves','teacher_stays','teacher_Bomb',
                'selected_point_enemy_near32_proxy','selected_point_enemy_incoming24_margin16_proxy',
                'selected_point_spawn_phase_near32','selected_laser_present','selected_laser_motion_unknown')}
            for i in range(1,7):self.masks['stage'+str(i)]=np.zeros(offset,bool)
        def iter_chunks(self,i,*,chunk_rows):
            original=self.original_indices[i];record=self.records[i]
            for first,last,data,legal in dataset.iter_chunks(original,chunk_rows=chunk_rows):
                lo,hi=record['offset']+first,record['offset']+last
                f,o,actions=data['features'],data['objects'],data['actions']
                normal=(f[:,21]>.5)&(f[:,20]<=0)&(f[:,5]<=0)
                present=o[:,:,0]>.5
                hazard=present&((o[:,:,1]>.5)|(o[:,:,2]>.5))&(o[:,:,13]>.5)
                position=o[:,:,6:8]*[384,448];velocity=o[:,:,8:10]*12
                bound=np.linalg.norm(o[:,:,10:12]*128,axis=-1)+np.linalg.norm(f[:,14:16]*64,axis=-1)[:,None]
                current=np.linalg.norm(position,axis=-1)-bound
                times=np.clip(-(position*velocity).sum(-1)/(velocity*velocity).sum(-1).clip(1e-8),0,24)
                closest=np.linalg.norm(position+velocity*times[:,:,None],axis=-1)-bound
                lasers=present&((o[:,:,3]>.5)|(o[:,:,4]>.5))
                values=dict(all=np.ones(last-first,bool),normal_unprotected=normal,protected_or_non_normal=~normal,
                    teacher_moves=(actions!=0)&(actions!=9)&(actions!=18),teacher_stays=(actions==0)|(actions==9),teacher_Bomb=actions==18,
                    selected_point_enemy_near32_proxy=normal&np.any(hazard&(current<32),axis=1),
                    selected_point_enemy_incoming24_margin16_proxy=normal&np.any(hazard&(closest<16),axis=1),
                    selected_point_spawn_phase_near32=normal&np.any(present&(o[:,:,1]>.5)&(o[:,:,20]>.5)&(current<32),axis=1),
                    selected_laser_present=lasers.any(axis=1),selected_laser_motion_unknown=(lasers&(o[:,:,26]<.5)).any(axis=1))
                self.stage[lo:hi]=data['stage']
                for s in range(1,7):values['stage'+str(s)]=data['stage']==s
                for k,v in values.items():self.masks[k][lo:hi]=v
                yield first,last,data,legal
        def close_caches(self):dataset.close_caches()
        def descriptors(self,indices,*,length):
            return [(i,t) for i in indices for t in range(0,self.records[i]['rows']-length+1,length)]
    view=HeldoutView()
    model=model_from_config(**bc['model_config']).cuda();model.load_state_dict(bc['model'],strict=True)
    if any(not torch.equal(v.detach().cpu(),bc['model'][k]) for k,v in model.state_dict().items()):
        raise ValueError('Diagnostic did not load selected BC weights')
    begin_cached_actor_phase(model);model.requires_grad_(False);model.eval()
    cache=TeacherLatentCache(torch,model,view,on_progress=progress,check_stop=stop)
    # The same frozen risk readout serves both parent and selected BC.
    costs=model.beta*torch.nn.functional.softplus(cache.risk)
    view.masks['frozen_risk_critical_proxy']=view.masks['normal_unprotected']&((costs.max(-1).values>4)&(costs.min(-1).values<.1)).cpu().numpy()
    del costs
    descriptors,excluded=cache.valid_descriptors(range(6),length=16)
    if len(descriptors)*16!=result['selected_heldout']['legal_labels'] or excluded:
        raise ValueError('Exact original heldout label count differs')
    comparisons={}
    for label,payload in [('parent_before_BC',parent),('selected_BC',bc)]:
        model.load_state_dict(payload['model'],strict=True);cache.rebind_after_best_restore()
        if any(not torch.equal(v.detach().cpu(),payload['model'][k]) for k,v in model.state_dict().items()):
            raise ValueError('Diagnostic policy identity differs:'+label)
        model.eval();cache.refresh_contexts()
        groups={k:dict(n=0,correct=0,CE_sum=0.,teacher_action_probability_sum=0.,
            label_counts=np.zeros(19,np.int64),correct_by_label=np.zeros(19,np.int64)) for k in view.masks}
        for first in range(0,len(descriptors),64):
            stop();selected=descriptors[first:first+64];batch=cache.batch(selected,length=16)
            with torch.no_grad():
                logits=forward_cached_actor(model,**{k:batch[k] for k in ('encoded','features','risk','initial_hidden','dones')})
                clean=policy_logits(logits);target=batch['actions'];legal=batch['legal']
                losses=torch.nn.functional.cross_entropy(clean,target,reduction='none').cpu().numpy()
                probabilities=clean.softmax(-1).gather(1,target[:,None]).squeeze(1).cpu().numpy()
                prediction=clean.argmax(-1).cpu().numpy();truth=target.cpu().numpy();keep=legal.cpu().numpy()
            positions=np.concatenate([np.arange(view.records[i]['offset']+t,view.records[i]['offset']+t+16) for i,t in selected])
            for name,mask in view.masks.items():
                picked=mask[positions]&keep;n=int(picked.sum())
                if not n:continue
                g=groups[name];correct=prediction[picked]==truth[picked]
                g['n']+=n;g['correct']+=int(correct.sum());g['CE_sum']+=float(losses[picked].sum(dtype=np.float64))
                g['teacher_action_probability_sum']+=float(probabilities[picked].sum(dtype=np.float64))
                g['label_counts']+=np.bincount(truth[picked],minlength=19)
                g['correct_by_label']+=np.bincount(truth[picked][correct],minlength=19)
            progress(dict(phase='reading_'+label,completed=min(first+64,len(descriptors)),total=len(descriptors),wall_seconds=time.monotonic()-started))
        normalized={}
        for name,g in groups.items():
            n=g['n'];counts=g['label_counts'];observed=counts>0
            normalized[name]=dict(samples=n,action_agreement=g['correct']/n if n else None,
                hard_teacher_CE=g['CE_sum']/n if n else None,
                mean_teacher_action_probability=g['teacher_action_probability_sum']/n if n else None,
                label_counts=counts.tolist(),correct_by_label=g['correct_by_label'].tolist(),
                macro_action_recall=float((g['correct_by_label'][observed]/counts[observed]).mean()) if n else None,
                largest_label_class_share=float(counts.max()/n) if n else None)
        comparisons[label]=normalized
        write(a.output/(label+'.json'),normalized)
    references=dict(parent_before_BC=read(a.fit/'initial_fit.json')['heldout'],selected_BC=result['selected_heldout'])
    checks={}
    for label,reference in references.items():
        actual=comparisons[label]['all']
        checks[label]=dict(CE_difference=actual['hard_teacher_CE']-reference['hard_teacher_CE'],
            agreement_difference=actual['action_agreement']-reference['executed_action_top1_agreement'],labels_equal=actual['samples']==reference['legal_labels'])
        if (not checks[label]['labels_equal'] or abs(checks[label]['CE_difference'])>3e-5
                or abs(checks[label]['agreement_difference'])>2/reference['legal_labels']+1e-10):
            raise ValueError('Original heldout aggregate not reproduced within predeclared FP32 tolerance:'+label)
    report=dict(schema='B1-pre-PPO-BC-six-whole-heldout-trajectory-stratification-v1',status='completed',utc=utc_now(),
        runtime=str(a.runtime),source_manifest_sha256=sha(a.runtime/'source_manifest.json'),
        actual_BC_sha256=result['policy_sha256'],parent_sha256=bc['parent_sha256'],
        teacher_manifest_sha256=dataset.manifest_sha256,original_heldout_indices=split['heldout'],
        original_heldout_native_seeds=split['heldout_native_seeds'],heldout_raw_rows=view.rows,
        heldout_scored_labels=len(descriptors)*16,independent_test_seeds_used=False,
        aggregate_reproduction=checks,comparisons=comparisons,cache=cache.stats,
        encoder_and_frozen_risk_weights_equal_between_compared_policies=True,
        full_original_prefix_context_recomputed_for_each_policy=True,
        input_proxies_are_not_native_safety_labels=True,
        proximity_proxy='Selected collidable point/enemy centers minus conservative bounding radii;near32 or stationary-player stored-velocity closest approach24 with margin16',
        frozen_risk_proxy='Same fixed learned physical readout;normal unprotected;max8softplus(logit)>4 and min<.1',
        agreement_is_not_avoidance_or_teacher_action_uniquely_optimal=True,
        model_scope='Completed pre-PPO BC initializer and its parent;not the currently changing full-policy PPO model',
        optimizer_updates=0,native_worlds_started=0,live_training_configuration_changed=False,
        CUDA=True,wall_seconds=time.monotonic()-started)
    write(a.output/'report.json',report);progress(dict(phase='completed',completed=len(descriptors)*16,total=len(descriptors)*16))
    print(json.dumps({k:v for k,v in report.items() if k!='comparisons'},ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for name in ('runtime','dataset','fit','parent','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    try:run(a)
    except BaseException:
        if getattr(a,'output_created',False):
            from fly_touhou.atomic_json_io import atomic_write_json as write
            write(a.output/'failure.json',dict(traceback=traceback.format_exc(),optimizer_updates=0,native_worlds_started=0,no_automatic_restart=True))
        raise
