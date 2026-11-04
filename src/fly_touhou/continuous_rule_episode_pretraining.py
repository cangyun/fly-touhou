"""Complete-rule-episode streaming and continuous-brain supervision.

Module import/metadata inspection is NoTorch/NoNumPy. Array/model/optimizer
operations occur only in explicit execution. No native or teacher is created.
"""
from __future__ import annotations
import bisect
import copy
import hashlib
import json
from pathlib import Path

SCHEMA='continuous-complete-rule-sixlane-supervised-pretraining-v1'
VISUAL_SCHEMA='rule-complete-V5-fixed-standardized-visual384-history-v1'
META=('actions','dones','episode_starts','native_episodes','native_seeds','before_frames',
    'after_observation_frames','stage','stage_frame','before_episode_frames','after_episode_frames')
TBPTT,WIDTH=16,16


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read_json(path):return json.loads(Path(path).read_text('utf-8'))


def inspect_fixed8(manifest,visual_manifest,*,expected_sha256=None,expected_visual_sha256=None):
    """Only JSON/index metadata;missing encoder output is a preparation status."""
    manifest=Path(manifest);visual_manifest=Path(visual_manifest)
    payload=manifest.read_bytes();digest=hashlib.sha256(payload).hexdigest();raw=json.loads(payload)
    if expected_sha256 is not None and digest!=expected_sha256:raise ValueError('Caller fixed8 manifest identity differs')
    entries=raw['episodes']
    if (len(entries)!=8 or len({e['episode'] for e in entries})!=8
            or sum(e['role']=='train' for e in entries)!=6
            or sum(e['role']=='readout_fit_held' for e in entries)!=2):
        raise ValueError('Exactly the fixed six train/two readout-fit-held full episodes required')
    vp=visual_manifest.read_bytes();visual_sha=hashlib.sha256(vp).hexdigest();locations=json.loads(vp)
    if (locations.get('schema')!='fixed-eight-complete-V5-visual384-cache-locations-v1'
            or locations['label_manifest_sha256']!=digest
            or (expected_visual_sha256 is not None and visual_sha!=expected_visual_sha256)):
        raise ValueError('Caller visual locations/label manifest binding differs')
    mapping={e['episode']:e for e in locations['episodes']}
    if set(mapping)!={e['episode'] for e in entries}:raise ValueError('Same fixed8 visual locations required')
    specs=[];pending=[]
    for entry in entries:
        episode=entry['episode'];label=Path(entry['label_cache'])
        place=mapping[episode];visual=Path(place['visual_cache'])
        if (place['role']!=entry['role'] or place['rows']!=entry['rows']
                or place['source_index_sha256']!=entry['source_index_sha256']
                or Path(place['label_cache']).resolve()!=label.resolve()):
            raise ValueError('Same explicit episode/role/label/visual locations required')
        spec=dict(episode=episode,role=entry['role'],rows=entry['rows'],native_seed=entry['native_seed'],
            source_index_sha256=entry['source_index_sha256'],label_root=str(label),visual_root=str(visual),
            original_rule_available=bool(entry.get('original_rule_targets_preserved',False)),
            stage_counts=entry.get('stage_counts',{}))
        if entry.get('status')!='committed' or not (label/'COMMITTED.json').is_file():
            pending.append('label:'+episode);specs.append(spec);continue
        li=read_json(label/'index.json');lc=read_json(label/'COMMITTED.json')
        if (sha(label/'index.json')!=entry['index_sha256'] or lc['index_sha256']!=entry['index_sha256']
                or li['source_index_sha256']!=entry['source_index_sha256'] or li['rows']!=entry['rows']
                or lc.get('complete_episode') is not True or li.get('complete_episode') is not True):
            raise ValueError('Committed original label/source identity differs:'+episode)
        spec.update(label_index_sha256=entry['index_sha256'],label_commit_sha256=sha(label/'COMMITTED.json'))
        if not (visual/'COMMITTED.json').is_file():
            pending.append('visual:'+episode);specs.append(spec);continue
        vi=read_json(visual/'index.json');vc=read_json(visual/'COMMITTED.json');identity=read_json(visual/'identity.json')
        if (vi.get('schema')!=VISUAL_SCHEMA or vi.get('status')!='completed_one_full_rule_visual_history'
                or vc['index_sha256']!=sha(visual/'index.json') or vc['identity_sha256']!=sha(visual/'identity.json')
                or vi['identity_sha256']!=vc['identity_sha256'] or vc.get('complete_episode') is not True
                or vi.get('complete_episode') is not True or vi['rows']!=entry['rows']
                or vi['source_index_sha256']!=entry['source_index_sha256']
                or vi['label_index_sha256']!=entry['index_sha256'] or identity['episode']!=episode):
            raise ValueError('Complete encoder/source/label identity differs:'+episode)
        cursor=0
        if len(vi['packets'])!=len(li['label_packets']):raise ValueError('Visual/label packet counts differ')
        for vp,lp in zip(vi['packets'],li['label_packets']):
            if (vp['start']!=cursor or vp['rows']!=lp['rows'] or vp['start']!=lp['start']
                    or vp['label_reference']['packet']!=lp
                    or Path(vp['label_reference']['root']).resolve()!=label.resolve()
                    or Path(vp['name']).name!=vp['name'] or Path(lp['name']).name!=lp['name']):
                raise ValueError('Original complete packet order/reference differs:'+episode)
            cursor+=vp['rows']
        if cursor!=entry['rows']:raise ValueError('Incomplete encoded history:'+episode)
        spec.update(visual_index_sha256=vc['index_sha256'],visual_identity_sha256=vc['identity_sha256'],
            visual_commit_sha256=sha(visual/'COMMITTED.json'),packet_count=len(vi['packets']),ready=True)
        specs.append(spec)
    return dict(schema=SCHEMA,manifest_path=str(manifest),manifest_sha256=digest,
        visual_manifest_path=str(visual_manifest),visual_manifest_sha256=visual_sha,specs=specs,pending=pending,
        ready=not pending,train_rows=sum(e['rows'] for e in specs if e['role']=='train'),
        held_rows=sum(e['rows'] for e in specs if e['role']=='readout_fit_held'),
        supervision=dict(available='signed4f,safe,source rule where present,measured deathbomb',
            future='12/24 targets absent;trust False,count0;never copy4f',
            collision='all offline outcomes=-1;count0;no invented native facts',
            world000_rule='missing original softp/weight:zero weight,zero placeholder,not behavior imitation'),
        NumPy_imported=False,Torch_imported=False,array_packets_read=0)


class RuleEpisodeReader:
    """At most two paired32-row packets cached;never reads maps/objects."""
    def __init__(self,spec,*,check_stop=lambda:None):
        if not spec.get('ready'):raise ValueError('Completed full visual history required')
        self.spec=copy.deepcopy(spec);self.check_stop=check_stop
        self.visual=Path(spec['visual_root']);self.label=Path(spec['label_root']);self.rows=spec['rows']
        self.index=read_json(self.visual/'index.json');self.identity=read_json(self.visual/'identity.json')
        self.parts=self.index['packets'];self.starts=[p['start'] for p in self.parts];self.cache={}
        self.paid=dict(packet_pairs_read=0,compressed_bytes_read=0)

    def _packet(self,number):
        import numpy as np
        if number in self.cache:return self.cache[number]
        self.check_stop();spec=self.parts[number];lp=spec['label_reference']['packet']
        vp=self.visual/'visual'/spec['name'];yp=self.label/'labels'/lp['name']
        if sha(vp)!=spec['sha256'] or sha(yp)!=lp['sha256']:raise ValueError('Committed encoded/label packet bytes differ')
        with np.load(vp,allow_pickle=False) as z:v={k:z[k].copy() for k in z.files}
        with np.load(yp,allow_pickle=False) as z:y={k:z[k].copy() for k in z.files}
        n=spec['rows'];r=np.arange(spec['start'],spec['start']+n,dtype=np.int64)
        if (v['encoded_visual'].shape!=(n,384) or v['encoded_visual'].dtype!=np.float32
                or not np.isfinite(v['encoded_visual']).all()
                or hashlib.sha256(v['encoded_visual'].tobytes()).hexdigest()!=spec['encoded_raw_sha256']
                or not np.array_equal(v['source_row'],r) or not np.array_equal(y['source_row'],r)
                or any(not np.array_equal(v[k],y[k]) for k in META)
                or not np.array_equal(v['episode_starts'],r==0)
                or not np.array_equal(v['dones'],r==self.rows-1)
                or not np.all(v['native_seeds']==self.spec['native_seed'])
                or y['features'].shape!=(n,25) or y['signed_margin4'].shape!=(n,18)
                or not np.isfinite(y['features']).all() or not np.isfinite(y['signed_margin4']).all()
                or not np.all(y['outcomes']==-1)):
            raise ValueError('Exact paired FP32/history/unknown-fact contract differs')
        rule='source_rule_targets' in y and 'source_teacher_guidance_weight' in y
        if rule!=self.spec['original_rule_available']:raise ValueError('Original rule-label availability differs')
        result={**v,**y};self.cache[number]=result
        while len(self.cache)>2:self.cache.pop(next(iter(self.cache)))
        self.paid['packet_pairs_read']+=1;self.paid['compressed_bytes_read']+=vp.stat().st_size+yp.stat().st_size
        return result

    def fetch(self,start,limit):
        import numpy as np
        if not 0<=start<=self.rows or not 0<=limit<=TBPTT:raise ValueError('Own contiguous cursor/window required')
        end=min(start+limit,self.rows);pieces=[];row=start
        fields=('encoded_visual','features','signed_margin4','trusted4','normal_unprotected_mask','outcomes',*META)
        if self.spec['original_rule_available']:fields+=('source_rule_targets','source_teacher_guidance_weight')
        while row<end:
            part=bisect.bisect_right(self.starts,row)-1;block=self._packet(part);offset=row-self.parts[part]['start']
            n=min(end-row,self.parts[part]['rows']-offset)
            pieces.append({k:block[k][offset:offset+n].copy() for k in fields});row+=n
        if not pieces:return None
        data={k:np.concatenate([p[k] for p in pieces]) for k in fields}
        if (not np.array_equal(data['before_frames'][1:],data['after_observation_frames'][:-1])
                or not np.array_equal(data['before_episode_frames'][1:],data['after_episode_frames'][:-1])):
            raise ValueError('Source clock gap;do not interpolate or reset')
        if start:
            number=bisect.bisect_right(self.starts,start-1)-1;previous=self._packet(number)
            local=start-1-self.parts[number]['start']
            if (data['before_frames'][0]!=previous['after_observation_frames'][local]
                    or data['before_episode_frames'][0]!=previous['after_episode_frames'][local]):
                raise ValueError('Cursor/window clock boundary differs')
        return data


def assemble_window(readers,cursors):
    import numpy as np
    import torch
    from .thrl_deathbomb_learning import deathbomb_opportunity_mask
    length=min(TBPTT,max(r.rows-c for r,c in zip(readers,cursors)))
    if length<=0:return None
    B=WIDTH;arrays=dict(encoded_visual=np.zeros((length,B,384),np.float32),features=np.zeros((length,B,25),np.float32),
        margins=np.zeros((length,B,3,18),np.float32),trusted=np.zeros((length,B,3),bool),
        actions=np.zeros((length,B),np.int64),outcomes=np.full((length,B),-1,np.int64),
        valid_mask=np.zeros((length,B),bool),unprotected_mask=np.zeros((length,B),bool),
        rule_probabilities=np.zeros((length,B,19),np.float32),rule_weights=np.zeros((length,B),np.float32),
        episode_starts=np.zeros((length,B),bool),stage=np.zeros((length,B),np.int16))
    consumed=[]
    for lane,(reader,cursor) in enumerate(zip(readers,cursors)):
        block=reader.fetch(cursor,length);n=0 if block is None else len(block['actions']);consumed.append(n)
        if not n:continue
        sl=(slice(0,n),lane)
        for target,source in (('encoded_visual','encoded_visual'),('features','features'),('actions','actions'),
                ('outcomes','outcomes'),('unprotected_mask','normal_unprotected_mask'),('episode_starts','episode_starts'),('stage','stage')):
            arrays[target][sl]=block[source]
        arrays['margins'][:n,lane,0]=block['signed_margin4'];arrays['trusted'][:n,lane,0]=block['trusted4']
        arrays['valid_mask'][sl]=True
        if reader.spec['original_rule_available']:
            arrays['rule_probabilities'][sl]=block['source_rule_targets'];arrays['rule_weights'][sl]=block['source_teacher_guidance_weight']
    arrays['deathbomb_mask']=np.asarray(deathbomb_opportunity_mask(arrays['features'],
        'semantic-separated-observed-laser-segments-v5'),bool)&arrays['valid_mask']
    return {k:torch.from_numpy(v) for k,v in arrays.items()},consumed


def supervised_forward_window(model,data,activity,counts,*,use_checkpoint=True):
    import torch
    from torch.nn import functional as F
    from torch.utils.checkpoint import checkpoint
    from .continuous_connectome import ContinuousConnectomeState
    from .continuous_signed_clearance import signed_clearance4f_terms
    from .fly_gru_aligned_guidance import gru_aligned_guidance_loss,DEFAULT_CONFIG
    device=model.device;d={k:v.to(device) for k,v in data.items()};lanes=len(activity)
    persistent=activity;logits=[];margins=[];probability=[];raw_probability=[]
    for t in range(len(d['encoded_visual'])):
        active=d['valid_mask'][t,:lanes];starts=d['episode_starts'][t,:lanes]&active
        persistent=torch.where(starts[:,None],torch.zeros_like(persistent),persistent)
        before=F.pad(persistent,(0,0,0,WIDTH-lanes));calls=[0]
        def call(encoded,previous,*,_calls=calls):
            key='forward_physical_lanes' if _calls[0]==0 else 'checkpoint_recompute_physical_lanes'
            counts[key]=counts.get(key,0)+WIDTH;_calls[0]+=1
            DN,following=model.core(encoded,ContinuousConnectomeState(previous))
            return DN,following.activity
        if use_checkpoint and torch.is_grad_enabled():
            DN,following=checkpoint(call,d['encoded_visual'][t],before,use_reentrant=False,preserve_rng_state=False)
        else:DN,following=call(d['encoded_visual'][t],before)
        counts['forward_real_rows']=counts.get('forward_real_rows',0)+int(active.sum())
        persistent=torch.where(active[:,None],following[:lanes],persistent)
        raw=model.actor(DN);score=model.risk_head.layers(DN);cost=F.softplus(score)
        composed=model.compose(raw,cost,d['features'][t],check_finite=False)
        clean=model.policy_logits(composed,d['features'][t],check_finite=False)
        # Receding safe/future guidance keeps original unclamped movement
        # logits;its hit/rule/DB terms perform their own original clamp.
        legal=composed.masked_fill(~torch.isfinite(clean),-torch.inf)
        legal=torch.where(d['valid_mask'][t,:,None],legal,torch.zeros_like(legal))
        logits.append(legal);margins.append(1.-.5*score)
        probability.append(clean.softmax(-1))
        raw_probability.append(model.policy_logits(raw,d['features'][t],check_finite=False).softmax(-1))
    logits=torch.stack(logits);predicted_margin=torch.stack(margins)
    task={k:d[k] for k in ('margins','trusted','outcomes','unprotected_mask')}
    risk=signed_clearance4f_terms(predicted_margin,task,d['features'],d['valid_mask'])
    guidance=gru_aligned_guidance_loss(logits,**task,actions=d['actions'],valid_mask=d['valid_mask'],
        config=DEFAULT_CONFIG,rule_probabilities=d['rule_probabilities'],rule_weights=d['rule_weights'],
        deathbomb_mask=d['deathbomb_mask'])
    # Raw risk objective1;the online PPO coupling.05 is deliberately not used.
    loss=risk['loss']+guidance['total_loss']
    return dict(loss=loss,risk=risk,guidance=guidance,following_activity=persistent,
        probability=torch.stack(probability),raw_probability=torch.stack(raw_probability),predicted_margin=predicted_margin)


def scalar_terms(result):
    def number(value):return float(value.detach().cpu()) if hasattr(value,'detach') else value
    risk=result['risk'];guide=result['guidance']
    return dict(total_loss=number(result['loss']),risk_loss_coefficient=1.,risk_loss=number(risk['loss']),
        risk_Huber=number(risk['huber_loss']),risk_BCE=number(risk['BCE_loss']),
        risk_valid_scenes=number(risk['valid_scenes']),risk_known_queries=number(risk['known_queries']),
        guidance_total=number(guide['total_loss']),
        guidance_losses={k:number(guide[k]) for k in ('safe_loss','future_loss','collision_loss','rule_loss','deathbomb_loss')},
        guidance_counts={k:number(v) for k,v in guide['denominators'].items()})


def construct_source(source,connectome):
    """Explicit raw/normalized typed loading;full dynamic source Adam/RNG."""
    import torch
    from .continuous_connectome import ContinuousConnectomeCore
    from .continuous_connectome_model import ContinuousConnectomeActorCritic
    from .continuous_connectome_training import restore_training_checkpoint,SCHEMA as BASE_TRAINING_SCHEMA
    from .continuous_normalized_risk_coordinate import (TRAINING_SCHEMA as NORMAL_TRAINING_SCHEMA,
        BASE_PARAMETER_SCHEMA,load_normalized_parameter_payload)
    from .fly_frozen_v5_perception import FrozenV5Perception
    from .fly_visual_distillation import FixedStandardizedVisual
    if source['schema'] not in (BASE_TRAINING_SCHEMA,NORMAL_TRAINING_SCHEMA):raise ValueError('Explicit continuous training source required')
    payload=source['model_payload'];cfg=payload['configuration']['core']
    if cfg['K']!=8 or not cfg['trainable_brain']:raise ValueError('This full-brain-learning stage preserves trainable K8')
    core=ContinuousConnectomeCore.from_full_connectome(connectome,K=8,trainable_brain=True,device='cpu',
        seed=cfg['seed'],epsilon=cfg['epsilon'],edge_chunk_size=cfg['sparse']['edge_chunk_size'])
    perception=FrozenV5Perception(payload['perception_configuration'])
    perception._source_provenance=copy.deepcopy(payload['perception_provenance'])
    adapter=FixedStandardizedVisual(perception,payload['fixed_mean'],payload['fixed_std'],payload['statistics_provenance'])
    model=ContinuousConnectomeActorCritic(core,adapter,seed=payload['configuration']['initialization_seed']).to('cuda:0')
    normalized=source['schema']==NORMAL_TRAINING_SCHEMA
    translated=source
    if normalized:
        load_normalized_parameter_payload(model,payload)
        translated=dict(source,schema=BASE_TRAINING_SCHEMA,model_payload=dict(payload,schema=BASE_PARAMETER_SCHEMA,
            configuration=dict(payload['configuration'],schema=BASE_PARAMETER_SCHEMA)))
    optimizer=torch.optim.Adam(model.optimizer_groups())
    inherited=restore_training_checkpoint(model,optimizer,translated,restore_rng=True)
    # Keep critic18 in the SAME optimizer groups/states;grad=None means no step.
    model.critic.requires_grad_(False)
    for p in model.critic.parameters():p.grad=None
    active={id(p) for p in model.parameters() if p.requires_grad}
    for group in optimizer.param_groups:
        if any(id(p) in active for p in group['params']):group['lr']=3e-4
    return model,optimizer,inherited,normalized


def critic_identity(model,optimizer):
    from .continuous_normalized_risk_coordinate import _update_hash
    digest=hashlib.sha256();_update_hash(model.critic.state_dict(),digest)
    for p in model.critic.parameters():_update_hash(optimizer.state.get(p,{}),digest)
    for group in optimizer.param_groups:
        if any(p is q for p in group['params'] for q in model.critic.parameters()):
            _update_hash({k:v for k,v in group.items() if k!='params'},digest)
    return digest.hexdigest()


def evaluate_fit_held(model,readers,*,check_stop=lambda:None,progress=lambda row:None):
    """Final theta,own row0 history;offline labels,never native outcome claims."""
    import numpy as np
    import torch
    cursors=[0]*len(readers);activity=model.initial_state(len(readers)).activity
    counts={};metrics={};windows=0;was_training=model.training;model.eval()
    def slot(episode,stage):return metrics.setdefault(episode,{}).setdefault(str(stage),dict(rows=0,
        trusted_geometry_rows=0,mixed_rows=0,risk_direction_danger=0,composed_greedy_danger=0,
        raw_greedy_danger=0,composed_greedy_Bomb=0,signed_absolute_error_sum=0.,danger_probability_sum=0.,
        eligible_signed_queries=0,danger_recall_hits=0,danger_queries=0,safe_false_positive=0,safe_queries=0,
        Huber_sums={k:0. for k in ('danger','boundary','all')},
        BCE_sums={k:0. for k in ('danger','boundary','all')},
        pool_query_counts={k:0 for k in ('danger','boundary','all')},
        pool_known_counts={k:0 for k in ('danger','boundary','all')},
        policy_domains={k:dict(scene_count=0,unconditional_danger_mass_sum=0.,
            conditional_movement_danger_mass_sum=0.,conditional_movement_count=0,
            zero_movement_mass_rows=0,Bomb_probability_sum=0.,
            composed_greedy_movement_count=0,composed_greedy_danger_count=0,composed_greedy_Bomb_count=0,
            raw_greedy_movement_count=0,raw_greedy_danger_count=0)
            for k in ('eligible','mixed')}))
    try:
        with torch.no_grad():
            while any(c<r.rows for c,r in zip(cursors,readers)):
                check_stop();data,used=assemble_window(readers,cursors)
                result=supervised_forward_window(model,data,activity,counts,use_checkpoint=False)
                activity=result['following_activity'].detach()
                pred=result['predicted_margin'].cpu().numpy();prob=result['probability'].cpu().numpy()
                raw=result['raw_probability'].cpu().numpy();target=data['margins'][:,:,0].numpy().clip(-64,64)
                valid=data['valid_mask'].numpy();f=data['features'].numpy()
                eligible=valid&data['trusted'][:,:,0].numpy()&data['unprotected_mask'].numpy()&(f[:,:,21]>.5)&(f[:,:,20]<=0)&(f[:,:,5]<=0)
                for lane,reader in enumerate(readers):
                    for t in range(used[lane]):
                        row=slot(reader.spec['episode'],int(data['stage'][t,lane]));row['rows']+=1
                        if not eligible[t,lane]:continue
                        y=target[t,lane];p=pred[t,lane];q=prob[t,lane];a=int(q.argmax());ra=int(raw[t,lane].argmax())
                        mixed=bool((y<=0).any() and (y>1).any())
                        row['trusted_geometry_rows']+=1;row['mixed_rows']+=int(mixed)
                        row['risk_direction_danger']+=int(y[p.argmax()]<=0)
                        row['composed_greedy_danger']+=int(a<18 and y[min(a,17)]<=0)
                        row['raw_greedy_danger']+=int(ra<18 and y[min(ra,17)]<=0);row['composed_greedy_Bomb']+=int(a==18)
                        row['signed_absolute_error_sum']+=float(np.abs(p-y).sum());row['eligible_signed_queries']+=18
                        row['danger_probability_sum']+=float(q[:18][y<=0].sum())
                        # BCE logit=-(margin-.5): predicted danger uses the
                        # same margin<=.5 boundary. Truth excludes gray(0,1].
                        row['danger_queries']+=int((y<=0).sum());row['danger_recall_hits']+=int(((y<=0)&(p<=.5)).sum())
                        row['safe_queries']+=int((y>1).sum());row['safe_false_positive']+=int(((y>1)&(p<=.5)).sum())
                        movement=q[:18].astype(np.float64);mass=float(movement.sum())
                        danger_mass=float(movement[y<=0].sum())
                        for domain in ('eligible','mixed') if mixed else ('eligible',):
                            metric=row['policy_domains'][domain];metric['scene_count']+=1
                            metric['unconditional_danger_mass_sum']+=danger_mass
                            metric['Bomb_probability_sum']+=float(q[18])
                            if mass>0:
                                metric['conditional_movement_danger_mass_sum']+=danger_mass/mass
                                metric['conditional_movement_count']+=1
                            else:metric['zero_movement_mass_rows']+=1
                            metric['composed_greedy_movement_count']+=int(a<18)
                            metric['composed_greedy_danger_count']+=int(a<18 and y[min(a,17)]<=0)
                            metric['composed_greedy_Bomb_count']+=int(a==18)
                            metric['raw_greedy_movement_count']+=int(ra<18)
                            metric['raw_greedy_danger_count']+=int(ra<18 and y[min(ra,17)]<=0)
                        diff=np.abs((p-y)/8.);huber=np.where(diff<.1,.5*diff*diff/.1,diff-.05)
                        binary=np.logaddexp(0.,.5-p)-(y<=0)*(.5-p);known=(y<=0)|(y>1)
                        for name,mask in (('danger',y<=0),('boundary',np.abs(y)<8),('all',np.ones(18,bool))):
                            row['Huber_sums'][name]+=float(huber[mask].sum())
                            row['BCE_sums'][name]+=float(binary[mask&known].sum())
                            row['pool_query_counts'][name]+=int(mask.sum());row['pool_known_counts'][name]+=int((mask&known).sum())
                cursors=[c+n for c,n in zip(cursors,used)];windows+=1
                if windows==1 or windows%128==0:progress(dict(windows=windows,cursors=cursors,counts=counts))
        for episode in metrics.values():
            for row in episode.values():
                h=bn=kf=0.
                for name,weight in (('danger',.25),('boundary',.5),('all',.25)):
                    key=name if row['pool_query_counts'][name] else 'all'
                    denominator=max(1,row['pool_query_counts'][key])
                    h+=weight*row['Huber_sums'][key]/denominator
                    bn+=weight*row['BCE_sums'][key]/denominator
                    kf+=weight*row['pool_known_counts'][key]/denominator
                row['signed4f_query_mixture_loss']=h+.2*bn/max(kf,1e-12)
                row['Huber_query_mixture']=h;row['BCE_query_mixture']=bn/max(kf,1e-12)
                row['danger_recall']=row['danger_recall_hits']/row['danger_queries'] if row['danger_queries'] else None
                row['safe_false_positive_rate']=row['safe_false_positive']/row['safe_queries'] if row['safe_queries'] else None
                for metric in row['policy_domains'].values():
                    n=metric['scene_count'];cn=metric['conditional_movement_count']
                    metric['unconditional_danger_mass_mean']=metric['unconditional_danger_mass_sum']/n if n else None
                    metric['conditional_movement_danger_mass_mean']=metric['conditional_movement_danger_mass_sum']/cn if cn else None
                    metric['Bomb_probability_mean']=metric['Bomb_probability_sum']/n if n else None
        return dict(status='completed_frozen_offline_fit_held',episode_cursors=dict(zip((r.spec['episode'] for r in readers),cursors)),
            counts=counts,by_episode_and_stage=metrics,final_theta_own_history_from_row0=True,
            metric_definitions=dict(predicted_danger='margin<=.5;the original BCE boundary',
                true_danger='target<=0',true_safe='target>1;gray(0,1]excluded from classification',
                eligible='valid & trusted4 & unprotected & normal',mixed='eligible & any(target<=0) & any(target>1)',
                unconditional_danger_mass='sum(q19[:18][target<=0])',
                conditional_movement_danger_mass='sum(q19[:18][target<=0])/sum(q19[:18])',
                zero_movement_mass='conditional metric undefined;excluded from conditional count and explicitly counted',
                greedy_danger_denominators='scene_count includes Bomb;greedy_movement_count excludes Bomb;both retained per domain'),
            optimizer_updates=0,native_actions=0,independent_native_test=False,
            no_early_stop_or_checkpoint_selection=True)
    finally:model.train(was_training)
