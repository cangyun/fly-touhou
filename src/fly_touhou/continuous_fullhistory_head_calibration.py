"""Final-theta full-history DN cache and staged readout calibration.

Import is NoTorch/NoNumPy. The complete graph is advanced only in the explicit
cache builder. Risk fitting,actor BC,and held metrics subsequently use DN only.
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path

SCHEMA='continuous-fullhistory-finaltheta-DN-head-calibration-v1'


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def DN_source_identity(model):
    from .continuous_normalized_risk_coordinate import _update_hash
    digest=hashlib.sha256()
    _update_hash(dict(model.core.named_parameters()),digest);_update_hash(model.core.graph_provenance,digest)
    _update_hash(model.sensory_adapter.state_dict(),digest)
    return digest.hexdigest()


def fixed_body_identity(model,optimizer,*,exclude_actor=True,exclude_risk=True,exclude_actor_LR=False):
    """Every non-calibrated tensor/Adam coordinate;not a source-code gate."""
    from .continuous_normalized_risk_coordinate import _update_hash
    omitted=set()
    if exclude_actor:omitted.update(id(p) for p in model.actor.parameters())
    if exclude_risk:omitted.update(id(p) for p in model.risk_head.parameters())
    digest=hashlib.sha256()
    for name,p in model.named_parameters():
        if id(p) in omitted:continue
        digest.update(name.encode());_update_hash(p,digest);_update_hash(optimizer.state.get(p,{}),digest)
    _update_hash(model.sensory_adapter.state_dict(),digest)
    if not exclude_risk:_update_hash(model.risk_head.state_dict(),digest)
    for group in optimizer.param_groups:
        settings={k:v for k,v in group.items() if k!='params'}
        if exclude_actor_LR and group.get('name')=='actor':settings.pop('lr',None)
        _update_hash(settings,digest)
    return digest.hexdigest()


def build_DN_cache(model,optimizer,data_plan,output,*,source_sha256,check_stop=lambda:None,progress=lambda row:None):
    import numpy as np
    import torch
    from torch.nn import functional as F
    from .continuous_connectome import ContinuousConnectomeState
    from .continuous_rule_episode_pretraining import RuleEpisodeReader,assemble_window
    output=Path(output);body=DN_source_identity(model)
    binding=dict(manifest_sha256=data_plan['manifest_sha256'],visual_manifest_sha256=data_plan['visual_manifest_sha256'],
        episodes=[{k:s[k] for k in ('episode','role','rows','source_index_sha256','label_index_sha256','visual_index_sha256')}
            for s in data_plan['specs']])
    if (output/'COMMITTED.json').is_file():
        commit=json.loads((output/'COMMITTED.json').read_text('utf-8'))
        if (commit['DN_source_identity']!=body or commit['data_binding']!=binding
                or sha(output/'index.json')!=commit['index_sha256']
                or sha(output/'DN.npy')!=commit['DN_sha256'] or sha(output/'labels.npz')!=commit['labels_sha256']):
            raise ValueError('Reusable complete DN cache body/data/bytes differ')
        result=json.loads((output/'index.json').read_text('utf-8'));result['reused_without_core_forward']=True
        return result
    if output.exists():raise FileExistsError('Preserve incomplete cache;no automatic restart or late zero state')
    readers=[RuleEpisodeReader(s,check_stop=check_stop) for s in data_plan['specs']]
    if len(readers)!=8:raise ValueError('Exactly the fixed8 complete histories required')
    for reader in readers:
        if (reader.identity['actual_perception_source']!=model.sensory_adapter.perception.source_provenance()
                or reader.identity['actual_statistics']!=model.sensory_adapter.statistics_provenance):
            raise ValueError('Cached frozen perception/statistics differ from source')
    total=sum(r.rows for r in readers);offsets=[];cursor=0
    for r in readers:offsets.append(cursor);cursor+=r.rows
    output.mkdir(parents=True)
    DN=np.lib.format.open_memmap(output/'DN.npy',mode='w+',dtype=np.float32,shape=(total,len(model.core.output_indices)))
    labels=dict(features=np.zeros((total,25),np.float32),margin4=np.zeros((total,18),np.float32),
        trusted4=np.zeros(total,bool),unprotected=np.zeros(total,bool),actions=np.zeros(total,np.int64),
        outcomes=np.full(total,-1,np.int8),source_row=np.zeros(total,np.int64),stage=np.zeros(total,np.int16),
        episode_index=np.zeros(total,np.int8),train_role=np.zeros(total,bool),
        source_rule_targets=np.zeros((total,19),np.float32),source_rule_weights=np.zeros(total,np.float32))
    cursors=[0]*8;activity=model.initial_state(8).activity
    counts=dict(real_rows=0,core_batch_calls=0,physical_lane_forwards=0,padding_lane_forwards=0,
        core_updates=0,optimizer_updates=0,teacher_queries=0,native_actions=0)
    versions=tuple(p._version for p in model.core.parameters());model.eval()
    try:
        with torch.no_grad():
            while any(c<r.rows for c,r in zip(cursors,readers)):
                check_stop();data,used=assemble_window(readers,cursors);window_DN=[]
                for t in range(len(data['encoded_visual'])):
                    check_stop();active=data['valid_mask'][t,:8].to(model.device)
                    starts=data['episode_starts'][t,:8].to(model.device)&active
                    activity=torch.where(starts[:,None],torch.zeros_like(activity),activity)
                    dn,following=model.core(data['encoded_visual'][t].to(model.device),
                        ContinuousConnectomeState(F.pad(activity,(0,0,0,8))))
                    counts['core_batch_calls']+=1;counts['physical_lane_forwards']+=16
                    counts['padding_lane_forwards']+=16-int(active.sum())
                    if not bool(torch.isfinite(following.activity).all()):raise FloatingPointError('Nonfinite final-theta history')
                    activity=torch.where(active[:,None],following.activity[:8],activity)
                    window_DN.append(dn[:8])
                copied=torch.stack(window_DN).cpu().numpy()
                for lane,(reader,start,n,offset) in enumerate(zip(readers,cursors,used,offsets)):
                    if not n:continue
                    sl=slice(offset+start,offset+start+n);DN[sl]=copied[:n,lane]
                    for target,source in (('features','features'),('actions','actions'),('outcomes','outcomes'),
                            ('unprotected','unprotected_mask'),('stage','stage'),('source_rule_targets','rule_probabilities'),
                            ('source_rule_weights','rule_weights')):
                        labels[target][sl]=data[source][:n,lane].numpy()
                    labels['margin4'][sl]=data['margins'][:n,lane,0].numpy()
                    labels['trusted4'][sl]=data['trusted'][:n,lane,0].numpy()
                    labels['source_row'][sl]=np.arange(start,start+n);labels['episode_index'][sl]=lane
                    labels['train_role'][sl]=reader.spec['role']=='train'
                cursors=[c+n for c,n in zip(cursors,used)];counts['real_rows']+=sum(used)
                if counts['core_batch_calls']%512==0:DN.flush();progress(dict(cursors=cursors,counts=copy.deepcopy(counts)))
        if (versions!=tuple(p._version for p in model.core.parameters()) or DN_source_identity(model)!=body):
            raise RuntimeError('DN cache must keep final-theta brain/input/perception unchanged')
        DN.flush();np.savez(output/'labels.npz',**labels)
        result=dict(schema=SCHEMA,status='complete_finaltheta_all8_DN',source_checkpoint_sha256=source_sha256,
            DN_source_identity=body,data_binding=binding,rows=total,DN_shape=list(DN.shape),DN_bytes=int(DN.nbytes),
            episode_offsets=[dict(episode=r.spec['episode'],role=r.spec['role'],offset=o,rows=r.rows) for r,o in zip(readers,offsets)],
            counts=counts,graph=model.core.graph_provenance,core_K=model.core.K,
            true_row0_complete_histories=True,Miss_or_stage_resets=0,teacher_hidden_or_old_DN_read=False,
            full_activity_per_frame_stored=False,reused_without_core_forward=False)
        (output/'index.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf-8')
        commit=dict(DN_source_identity=body,data_binding=binding,index_sha256=sha(output/'index.json'),
            DN_sha256=sha(output/'DN.npy'),labels_sha256=sha(output/'labels.npz'),complete_all8=True)
        (output/'COMMITTED.json').write_text(json.dumps(commit,ensure_ascii=False,indent=2),'utf-8')
        return result
    finally:DN.flush();del DN


def load_DN_cache(path):
    import numpy as np
    import torch
    path=Path(path)
    DN=torch.from_numpy(np.load(path/'DN.npy',mmap_mode='c'))
    with np.load(path/'labels.npz',allow_pickle=False) as z:data={k:torch.from_numpy(z[k].copy()) for k in z.files}
    return DN,data


def install_new_normalized_risk(model,optimizer,signed_net,mean,scale,spec):
    import torch
    from .continuous_normalized_risk_coordinate import NormalizedRiskLayers
    before=fixed_body_identity(model,optimizer,exclude_actor=False)
    old=model.risk_head.layers;model.risk_head.layers=NormalizedRiskLayers(old[0],old[1],old[2],mean,scale)
    state={k:v.detach().clone() for k,v in signed_net.state_dict().items()}
    state['layers.2.weight']*=-16.;state['layers.2.bias']=2.-16.*state['layers.2.bias']
    state['layers.DN_mean']=mean.to(model.device);state['layers.DN_scale']=scale.to(model.device)
    risk=list(model.risk_head.parameters())
    steps=[float(optimizer.state.get(p,{}).get('step',0)) for p in risk]
    removed=sum(p in optimizer.state for p in risk)
    model.risk_head.load_state_dict(state,strict=True)
    for p in risk:p.grad=None;optimizer.state.pop(p,None)
    after=fixed_body_identity(model,optimizer,exclude_actor=False)
    if before!=after:raise RuntimeError('Risk fitting changed a non-risk/actor tensor or Adam coordinate')
    return dict(risk_parameters_retrained=4,risk_coordinates_reset=4,existing_risk_states_removed=removed,
        previous_risk_steps=steps,unaffected_body_Adam_and_LRs_preserved=True,
        input_normalization=spec,training_output_coordinate='z=margin/8',runtime_score='2-16*z',
        private_signed_Adam_transferred=False)


def weighted_teacher_BC(clean,targets,weights):
    """Original teacher_cross_entropy legal-conditioning formula;keep real weights."""
    import torch
    legal=torch.isfinite(clean);selected=targets*legal.to(targets.dtype);mass=selected.sum(-1,keepdim=True)
    if bool((mass<=0).any()):raise ValueError('Active original teacher has no legal mass')
    selected=selected/mass;logp=clean.log_softmax(-1)
    per=-(selected*logp.masked_fill(~legal,0.)).sum(-1)
    return (per*weights).sum()/weights.sum().clamp_min(1e-12)


def calibrate_actor(model,optimizer,DN,data,fixed_cost,*,seed=71932,check_stop=lambda:None,progress=lambda row:None):
    import torch
    before=fixed_body_identity(model,optimizer,exclude_actor=True,exclude_risk=False,exclude_actor_LR=True)
    model.requires_grad_(False);model.actor.requires_grad_(True)
    for p in model.parameters():p.grad=None
    actor=list(model.actor.parameters());ids={id(p) for p in actor}
    if len(actor)!=6:raise ValueError('Original actor LayerNorm+256SiLU+19 six tensors required')
    groups=[]
    for g in optimizer.param_groups:
        hits=[id(p) in ids for p in g['params']]
        if any(hits):
            if not all(hits):raise ValueError('Actor group must not change other learning rates')
            groups.append(dict(name=g.get('name'),previous_lr=g['lr']));g['lr']=3e-4
    train=data['train_role']&(data['source_rule_weights']>0)
    pool=train.nonzero(as_tuple=False).flatten().cpu()
    if not len(pool):raise ValueError('Original positive-weight train rule labels required;world000 is not fabricated')
    generator=torch.Generator(device='cpu').manual_seed(seed);record=dict(optimizer_updates=0,optimizer_step_attempts=0,
        optimizer_transaction_complete=True,backward_passes=0,training_rows=0,eligible_train_rows=len(pool),
        main_Adam_groups=groups,learning_rate=.0003,clip_norm=.5,history=[],brain_updates=0,critic_updates=0,risk_updates=0)
    try:
        for step in range(2000):
            check_stop();index=pool[torch.randint(len(pool),(1024,),generator=generator)]
            x=DN[index].to(model.device);f=data['features'][index].to(model.device)
            target=data['source_rule_targets'][index].to(model.device);weight=data['source_rule_weights'][index].to(model.device)
            optimizer.zero_grad(set_to_none=True)
            raw=model.actor(x);combined=model.compose(raw,fixed_cost[index].to(model.device),f,check_finite=False)
            clean=model.policy_logits(combined,f,check_finite=False);loss=weighted_teacher_BC(clean,target,weight)
            if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite combined-policy teacher BC')
            loss.backward();record['backward_passes']+=1
            if any(p.grad is not None for p in model.parameters() if id(p) not in ids):
                raise RuntimeError('Only actor6 may carry a gradient')
            norm=torch.nn.utils.clip_grad_norm_(actor,.5,error_if_nonfinite=True)
            check_stop();record['optimizer_step_attempts']+=1;record['optimizer_transaction_complete']=False
            optimizer.step();record['optimizer_updates']+=1
            if any(not bool(torch.isfinite(p).all()) for p in actor):raise FloatingPointError('Nonfinite actor parameter')
            record['optimizer_transaction_complete']=True;record['training_rows']+=1024
            if step==0 or (step+1)%100==0:
                record['history'].append(dict(update=step+1,teacher_CE=float(loss.detach()),preclip_grad_norm=float(norm)))
                progress(copy.deepcopy(record))
        if before!=fixed_body_identity(model,optimizer,exclude_actor=True,exclude_risk=False,exclude_actor_LR=True):
            raise RuntimeError('Actor BC changed brain/input/perception/critic/risk or their Adam/LRs')
        record['all_nonactor_parameters_Adam_and_LRs_preserved']=True
        return record
    except BaseException as error:error.actor_BC_diagnostics=copy.deepcopy(record);raise


def predict_cost_cache(model,DN,*,check_stop=lambda:None):
    import torch
    parts=[]
    with torch.no_grad():
        for first in range(0,len(DN),1024):
            check_stop();parts.append(model.risk_head(DN[first:first+1024].to(model.device)).cpu())
    return torch.cat(parts)


def evaluate_cached_heads(model,DN,data,*,check_stop=lambda:None):
    """Only two fit-held roles;stable log-softmax KL and Bomb-separated danger."""
    import numpy as np
    import torch
    held=(~data['train_role']).nonzero(as_tuple=False).flatten();result={}
    def bucket(episode):return result.setdefault(str(episode),dict(rows=0,teacher_rows=0,teacher_weight=0.,
        teacher_KL_sum=0.,teacher_CE_sum=0.,teacher_top1_weighted_correct=0.,
        danger_queries=0,danger_recall_hits=0,safe_queries=0,safe_false_positive=0,
        risk_best_geometry_danger=0,eligible_geometry_rows=0,
        domains={k:dict(rows=0,unconditional_danger_sum=0.,conditional_movement_danger_sum=0.,Bomb_sum=0.,
            greedy_moves=0,greedy_danger=0,greedy_Bomb=0) for k in ('eligible','mixed')}))
    with torch.no_grad():
        for first in range(0,len(held),1024):
            check_stop();ix=held[first:first+1024];x=DN[ix].to(model.device);f=data['features'][ix].to(model.device)
            score=model.risk_head.layers(x);cost=torch.nn.functional.softplus(score);margin=1.-.5*score
            clean=model.policy_logits(model.compose(model.actor(x),cost,f,check_finite=False),f,check_finite=False)
            logp=clean.log_softmax(-1);p=clean.softmax(-1);move=clean[:,:18].softmax(-1)
            target=data['source_rule_targets'][ix].to(model.device);weights=data['source_rule_weights'][ix].to(model.device)
            legal=torch.isfinite(clean);q=target*legal;qmass=q.sum(-1,keepdim=True)
            q=q/qmass.clamp_min(1e-30);qlog=torch.where(q>0,q.clamp_min(1e-30).log(),torch.zeros_like(q))
            ce=-(q*logp.masked_fill(~legal,0.)).sum(-1);kl=(q*(qlog-logp.masked_fill(~legal,0.))).sum(-1)
            correct=q.argmax(-1)==p.argmax(-1)
            pp,mc,mm=(t.cpu().numpy() for t in (p,move,margin))
            yy=data['margin4'][ix].clamp(-64,64).numpy();ff=data['features'][ix].numpy()
            eligible=data['trusted4'][ix].numpy()&data['unprotected'][ix].numpy()&(ff[:,21]>.5)&(ff[:,20]<=0)&(ff[:,5]<=0)
            for row,index in enumerate(ix.tolist()):
                b=bucket(int(data['episode_index'][index]));b['rows']+=1;w=float(weights[row])
                if w>0:
                    if float(qmass[row])<=0:raise ValueError('Held original positive-weight target has no legal mass')
                    b['teacher_rows']+=1;b['teacher_weight']+=w;b['teacher_CE_sum']+=w*float(ce[row])
                    b['teacher_KL_sum']+=w*float(kl[row]);b['teacher_top1_weighted_correct']+=w*int(correct[row])
                if not eligible[row]:continue
                y,m=yy[row],mm[row];b['eligible_geometry_rows']+=1;b['risk_best_geometry_danger']+=int(y[m.argmax()]<=0)
                b['danger_queries']+=int((y<=0).sum());b['danger_recall_hits']+=int(((y<=0)&(m<=.5)).sum())
                b['safe_queries']+=int((y>1).sum());b['safe_false_positive']+=int(((y>1)&(m<=.5)).sum())
                a=int(pp[row].argmax());mixed=bool((y<=0).any() and (y>1).any())
                for domain in ('eligible','mixed') if mixed else ('eligible',):
                    d=b['domains'][domain];d['rows']+=1;d['unconditional_danger_sum']+=float(pp[row,:18][y<=0].sum())
                    d['conditional_movement_danger_sum']+=float(mc[row][y<=0].sum());d['Bomb_sum']+=float(pp[row,18])
                    d['greedy_moves']+=int(a<18);d['greedy_danger']+=int(a<18 and y[min(a,17)]<=0);d['greedy_Bomb']+=int(a==18)
    return dict(by_episode_index=result,held_only=True,source_rows=len(held),brain_forwards=0,optimizer_updates=0,
        prediction_boundary='margin<=.5;true danger<=0,true safe>1',
        conditional_probability='softmax of18 legal movement logits,stable equivalent to q19[:18]/movementmass',
        fit_held_not_independent_native=True,checkpoint_selection=False)
