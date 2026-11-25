"""Real student trajectories, full-story three-value PPO and GRU guidance.

No teacher action substitution, stage resets, truncated games or D95 action
distillation. Brain state follows the sampled action, including neutral Bomb.
"""
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from ..spiking_engineered_control import SpikingState
from ..compact_full_story import factory,story_kwargs
from ..thrl_durable_records import write_json,append_jsonl
from ..fly_gru_aligned_ppo import aligned_ppo_config,prepare_three_value_targets,three_value_ppo_loss
from ..fly_gru_aligned_guidance import gru_aligned_guidance_loss
from ..compact_story_behavior import story_behavior_logits,rollout_sampling_temperature,rollout_bomb_sampling_temperature,sampling_temperature


def invalidate_contradictory_risk(record,index,hit,action,elapsed_frames=4):
    """Do not train either geometric-risk mask on a contradicted outcome."""
    exposed=record['unprotected_mask'][index]
    if 'features' in record:
        from ..compact_native_contact_targets import native_exposure_mask
        exposed=native_exposure_mask(record['features'][index],exposed)
    if not exposed or not 0<=action<18:return False
    gap=record['margins'][index,0,action]
    contradiction=((hit and 0<elapsed_frames<=4 and gap>0)
                   or (not hit and elapsed_frames==4 and gap<0))
    if contradiction:
        record['trusted'][index]=False
        if 'risk_training_trusted' in record:record['risk_training_trusted'][index]=False
        if 'risk_continuation_valid' in record:record['risk_continuation_valid'][index]=False
    return bool(contradiction)


class Collector:
    def __init__(self,model,config,output,*,workers=16,worker_cpus=None,movement_sampling_temperature=1.,bomb_sampling_temperature=1.,continuation=None,
                 world_factory=factory,world_kwargs=None):
        from .. import thrl_vector
        from ..owned_worker_affinity import spread_owned_workers
        import psutil
        self.movement_sampling_temperature=sampling_temperature(movement_sampling_temperature)
        self.bomb_sampling_temperature=sampling_temperature(bomb_sampling_temperature)
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=False)
        self.native_position_hold_forecast=config.get('compact_native_position_hold_forecast')
        self.native_hidden_enemy_contacts=config.get('compact_native_hidden_enemy_contacts')
        self.native_ex8_polar4_forecast=config.get('compact_native_ex8_polar4_forecast')
        from ..compact_full_story import native_wall_revision,native_scale_revision
        self.native_wall_reactive_forecast=native_wall_revision(config)
        self.native_ex_scale_forecast=native_scale_revision(config)
        self.vector=None;self.actions=0;self.completed=0;self.workers=workers
        from ..compact_story_coverage import StoryCoverage
        self.coverage=StoryCoverage(workers)
        try:
            if world_kwargs is None:
                world_kwargs=story_kwargs(config,self.output,workers,'train',teacher=True)
            self.vector=thrl_vector.THRLNativeVector(world_kwargs,self.output/'workers',timeout=180,factory=world_factory)
            if worker_cpus:
                p=psutil.Process();write_json(self.output/'affinity.json',spread_owned_workers(p.pid,p.create_time(),worker_cpus))
            options=([dict(continuation=w) for w in continuation['worlds']] if continuation else None)
            self.obs,self.infos=self.vector.reset(options=options);self.state=model.initial_state(workers);self.starts=np.ones(workers,bool)
            if getattr(model,'nbnm',None) and continuation is None:
                from ..nbnm_prefix import burn_in
                self.state=SpikingState.pack([burn_in(model,info,self.state.row(i)) for i,info in enumerate(self.infos)])
        except BaseException as error:
            try:self.close()
            except BaseException as cleanup_error:
                error.add_note('Native cleanup also reported: '+repr(cleanup_error))
            raise
    @torch.no_grad()
    def rollout(self,model,*,steps=512,length=16,guard=lambda:None,progress=lambda x:None,
            supervision_selection=None):
        from ..compact_native_forecast import native_repair_revisions
        model.eval();rows=[];states={};begun=time.monotonic();bomb_actions=actual_bombs=hits=deathbombs=0
        normal_teacher_rows=normal_teacher_guided_rows=full_native_teacher_restored_rows=0
        trusted_normal_counts=np.zeros(3,np.int64);safe_but_native_contacts=0
        from ..compact_story_coverage import StoryCoverage
        rollout_coverage=StoryCoverage(self.workers)
        selector=None
        if supervision_selection is not None:
            from ..compact_story_action_selection import StoryActionSelector
            if supervision_selection not in ('joint_mode','family_mode'):
                raise ValueError('Deterministic student deployment decoder required for supervision-only collection')
            selector=StoryActionSelector(supervision_selection,self.workers,seed=0)
        for t in range(steps):
            guard()
            if t%length==0:states[t]=self.state.detach().to('cpu')
            f=self.obs['features'].copy();before=self.infos
            label=[x['compact_targets'] for x in before];expert=[x['teacher_label'] for x in before]
            for i in range(self.workers):
                if (label[i]['frame'],label[i]['native_episode'])!=(expert[i]['frame'],expert[i]['native_episode']):
                    raise RuntimeError('Teacher and risk labels are not the current student state')
            if selector is not None:
                out,following,selected_DN=model.step(torch.as_tensor(f,device=model.device),self.state,
                    action_selector=lambda logits:selector(logits,list(range(self.workers))))
            elif self.movement_sampling_temperature==self.bomb_sampling_temperature==1.:
                out,following,selected_DN=model.step(torch.as_tensor(f,device=model.device),self.state,sample=True)
            else:
                def choose(logits):
                    return torch.distributions.Categorical(logits=story_behavior_logits(
                        logits,self.movement_sampling_temperature,self.bomb_sampling_temperature),validate_args=False).sample()
                out,following,selected_DN=model.step(torch.as_tensor(f,device=model.device),self.state,action_selector=choose)
            behavior_logits=story_behavior_logits(out['logits'],self.movement_sampling_temperature,self.bomb_sampling_temperature)
            if selector is not None:
                # Actual behavior is a point mass at the student's own mode.
                # These labelled trajectories are expressly excluded from PPO.
                behavior_logits=torch.full_like(out['logits'],-torch.inf)
                behavior_logits.scatter_(1,out['actions'][:,None],0.)
            action=out['actions'];distribution=torch.distributions.Categorical(logits=behavior_logits,validate_args=False)
            record=dict(features=f,actions=action.cpu().numpy(),old_log_probs=distribution.log_prob(action).cpu().numpy(),
                old_values=out['values'].cpu().numpy(),episode_starts=self.starts.copy(),
                margins=np.asarray([x['margins'] for x in label],np.float32),trusted=np.asarray([x['trusted'] for x in label],bool),
                unprotected_mask=np.asarray([x['normal'] for x in label],bool),
                rule_probabilities=np.asarray([x['probabilities'] for x in expert],np.float32),
                rule_weights=np.asarray([x.get('guidance_weight',1.) for x in expert],np.float32),
                deathbomb_mask=(f[:,4]>0)&(f[:,12]>0)&(f[:,17]>.5),
                lower_field_exception=np.asarray([x.get('lower_field_exception',f[i,111]>.99)
                    or expert[i].get('native_route_guidance',{}).get('applied',False) for i,x in enumerate(label)],bool))
            # Preserve the decision-time mask for unbiased diagnostics. The
            # training mask below can be invalidated after observing outcomes.
            record['recorded_pre_action_trusted']=record['trusted'].copy()
            partial_present=any('partial_point_danger' in x for x in label)
            if partial_present:
                record['partial_point_danger']=np.asarray([x.get('partial_point_danger',np.zeros((3,18),bool)) for x in label],bool)
                record['recorded_pre_action_partial_point_danger']=record['partial_point_danger'].copy()
            record.update(recorded_native_stage=np.asarray([x['stage'] for x in before],np.int16),
                recorded_native_episode=np.asarray([x['compact_targets']['native_episode'] for x in before],np.int64),
                recorded_episode_frames=np.asarray([x['episode_frames'] for x in before],np.int64),
                recorded_teacher_Bomb_corrected=np.asarray([
                    x.get('resource_reference_correction',{}).get('applied',False) for x in expert],bool))
            if any('risk_training_eligible' in x for x in label):
                record.update(risk_training_eligible=np.asarray([x['risk_training_eligible'] for x in label],bool),
                    risk_training_trusted=np.asarray([x['risk_training_trusted'] for x in label],bool))
            # Read-only telemetry from the acting forward, not recomputed
            # outputs or additional learning targets. Small compared with the
            # complete carried-brain archives. Policy inputs remain features.
            record.update(recorded_neural_horizon_scores=out['horizon_scores'].cpu().numpy(),
                recorded_policy_logits=out['logits'].cpu().numpy(),
                recorded_route_scores=out['route_scores'].cpu().numpy(),
                recorded_selected_DN=selected_DN.cpu().numpy())
            if selector is not None or self.movement_sampling_temperature!=1. or self.bomb_sampling_temperature!=1.:
                record['recorded_behavior_policy_logits']=behavior_logits.cpu().numpy()
            if model.phase_valid_control:
                record['rule_weights']*=((f[:,14]>.5)|(f[:,17]>.5)).astype(np.float32)
            normal_teacher_rows+=int(np.count_nonzero(record['unprotected_mask']))
            trusted_normal_counts+=(record['trusted']&record['unprotected_mask'][:,None]).sum(0)
            normal_teacher_guided_rows+=int(np.count_nonzero(record['unprotected_mask']&(record['rule_weights']>0)))
            full_native_teacher_restored_rows+=sum(int(record['unprotected_mask'][i]
                and record['rule_weights'][i]>0 and expert[i].get('full_native_support_attempt',{}).get('applied',False))
                for i in range(self.workers))
            actions=record['actions'];after,_,done,truncated,infos=self.vector.step(actions.tolist())
            if np.asarray(truncated).any():raise RuntimeError('A full-story rollout may end games only at native terminals')
            if any(x.get('reward_mapping_complete') is not True for x in infos):raise RuntimeError('Incomplete original reward mapping')
            rewards=np.asarray([x['reward_vector'] for x in infos],np.float32)
            for reward_key in ('lower_field_potential','avoidable_upper_residence'):
                record[reward_key]=np.asarray([x.get('reward_components_raw',{}).get(reward_key,0.) for x in infos],np.float32)
            if getattr(model,'native_contact_risk_calibration',False) or partial_present:
                record['native_action_frames']=np.asarray([
                    info['episode_frames']-before[i]['episode_frames'] for i,info in enumerate(infos)],np.int32)
            outcomes=np.full(self.workers,-1,np.int8)
            observed_contacts=np.zeros(self.workers,bool)
            for i,info in enumerate(infos):
                if info['policy_action']!=int(actions[i]):raise RuntimeError('Native action differs from the sampled student action')
                if 'compact_contact_evidence' in info:
                    append_jsonl(self.output/'native-contact-evidence.jsonl',dict(world=i,
                        training_student_decisions=self.actions+self.workers,transition=info['compact_contact_evidence'],
                        before_uncertainty=label[i].get('uncertainty')))
                used=info['bombs_used']-before[i]['bombs_used'];actual_bombs+=used;bomb_actions+=actions[i]==18
                deathbombs+=int(used>0 and record['deathbomb_mask'][i])
                hit=bool(info['misses']>before[i]['misses'] or (after['features'][i,17]>.5 and f[i,17]<.5));hits+=hit
                observed_contacts[i]=hit
                if f[i,14]>.5 and actions[i]<18:
                    outcomes[i]=int(hit)
                from ..compact_native_contact_targets import native_exposure_mask
                if native_exposure_mask(f[i],record['unprotected_mask'][i]) and actions[i]<18:
                    gap=record['margins'][i,0,actions[i]]
                    elapsed=info['episode_frames']-before[i]['episode_frames']
                    safe_but_native_contacts+=int(hit and 0<elapsed<=4 and gap>0)
                    invalidate_contradictory_risk(record,i,hit,int(actions[i]),elapsed)
                if done[i]:
                    self.completed+=1;append_jsonl(self.output/'training-episodes.jsonl',dict(info['episode_summary'],
                        statistics_scope='changing-policy training game;not frozen evaluation'))
            salient=(np.ones(self.workers,bool) if selector is not None else
                record['deathbomb_mask']|(actions==18)|(outcomes==1))
            pooled=np.zeros((self.workers,out['DN_mean'].shape[-1]),np.float32)
            if salient.any():pooled[salient]=out['DN_mean'][torch.as_tensor(salient,device=model.device)].cpu().numpy()
            record.update(recorded_pooled_DN_mask=salient,recorded_pooled_DN=pooled,
                reward_vectors=rewards,terminated=np.asarray(done,bool),outcomes=outcomes,
                recorded_observed_contacts=observed_contacts)
            if partial_present:
                from ..compact_partial_risk import veto_selected_clear
                record['partial_point_danger'],veto=veto_selected_clear(record['partial_point_danger'],
                    actions,outcomes,record['unprotected_mask'],record['native_action_frames'])
                record['recorded_partial_point_veto']=veto
            rollout_coverage.observe(before,infos,record,done)
            self.coverage.observe(before,infos,record,done)
            rows.append(record);self.actions+=self.workers;self.state=following;self.obs=after;self.infos=infos
            self.starts=np.asarray(done,bool)
            ended=np.flatnonzero(done).tolist()
            if ended:
                fresh=self.vector.reset_at(ended);self.obs=self.vector.observation
                self.state=self.state.reset(torch.as_tensor(done,device=model.device))
                for i,info in zip(ended,fresh):self.infos[i]=info
            if (t+1)%16==0:progress(dict(phase='student_full_story_collection',rollout_decisions=(t+1)*self.workers,
                target=steps*self.workers,total_student_decisions=self.actions,training_games=self.completed,
                actual_bombs=int(actual_bombs),deathbombs=int(deathbombs),contacts=int(hits),
                normal_teacher_rows=normal_teacher_rows,normal_teacher_guided_rows=normal_teacher_guided_rows,
                full_native_teacher_restored_rows=full_native_teacher_restored_rows,
                trusted_normal_rows_4_12_24=trusted_normal_counts.tolist(),
                safe_forecast_but_native_contact=safe_but_native_contacts,
                stage_coverage=rollout_coverage.summary(),job_stage_coverage=self.coverage.summary(),
                max_stage=max(x['stage'] for x in infos),elapsed_seconds=time.monotonic()-begun))
        # Bootstrap the real next observation without committing this diagnostic
        # forward. A true terminal is masked in GAE even if its slot was reset.
        bootstrap,_,_=model.step(torch.as_tensor(self.obs['features'],device=model.device),self.state)
        data={key:torch.from_numpy(np.stack([r[key] for r in rows])) for key in rows[0]}
        data.update(block_states=states,bootstrap_values=bootstrap['values'].cpu(),
            metadata=dict(student_actions=steps*self.workers,Bomb_requests=int(bomb_actions),actual_bombs=int(actual_bombs),
                native_repair_revisions=native_repair_revisions(
                    reactive_orbs=getattr(self,'native_wall_reactive_forecast',None),
                    bullet_scale=getattr(self,'native_ex_scale_forecast',None)),
                native_wall_reactive_forecast=getattr(self,'native_wall_reactive_forecast',None),
                native_ex_scale_forecast=getattr(self,'native_ex_scale_forecast',None),
                deathbombs=int(deathbombs),contacts=int(hits),teacher_actions=0,full_story=True,
                stage_coverage=rollout_coverage.summary(),job_stage_coverage=self.coverage.summary(),
                normal_teacher_rows=normal_teacher_rows,normal_teacher_guided_rows=normal_teacher_guided_rows,
                full_native_teacher_restored_rows=full_native_teacher_restored_rows,
                trusted_normal_rows_4_12_24=trusted_normal_counts.tolist(),
                safe_forecast_but_native_contact=safe_but_native_contacts,
                supervision_only=selector is not None,action_selection=supervision_selection or 'sample',
                partial_point_danger_revision=label[0].get('partial_point_danger_revision'),
                movement_sampling_temperature=self.movement_sampling_temperature,
                bomb_sampling_temperature=self.bomb_sampling_temperature,
                protected_exit_forecast=any('protected_exit_revision' in x for x in label),
                protected_exit_from_stage=getattr(model,'protected_exit_from_stage',1),
                protected_exit_preserve_long_shield=getattr(model,'protected_exit_preserve_long_shield',False),
                protected_exit_revision=label[0].get('protected_exit_revision'),
                native_contact_risk_calibration=getattr(model,'native_contact_risk_calibration',False),
                native_motion_forecast=getattr(model,'native_motion_forecast',False),
                native_position_hold_forecast=getattr(self,'native_position_hold_forecast',None),
                native_hidden_enemy_contacts=getattr(self,'native_hidden_enemy_contacts',None),
                native_ex8_polar4_forecast=getattr(self,'native_ex8_polar4_forecast',None),
                native_focus_gate=getattr(model,'native_focus_gate',False),
                unknown_laser_player_motion=getattr(model,'unknown_laser_player_motion',False),
                native_curve_laser_geometry=getattr(model,'native_curve_laser_geometry',False),
                primary_attack_target=getattr(model,'primary_attack_target',False),
                forecast_growth_motion=getattr(model,'forecast_growth_motion',False),
                anticipate_point_arming=getattr(model,'anticipate_point_arming',False),
                task_progress_ports=getattr(model,'task_progress_ports',False)))
        return data
    def close(self):
        if self.vector is not None:
            self.vector.close();write_json(self.output/'cleanup.json',dict(errors=self.vector.close_errors))
            if self.vector.close_errors:raise RuntimeError('Training native cleanup unresolved')


def risk_weights(margins,trusted,normal):
    mask=(trusted if trusted.shape==margins.shape else trusted[...,None])&normal[...,None,None]
    mask=mask.expand_as(margins)
    reg=torch.zeros_like(margins);cls=torch.zeros_like(margins)
    for h,hw in enumerate((1.,.3,.1)):
        m=margins[...,h,:];valid=mask[...,h,:];known=(m<=0)|(m>1)
        for fraction,q in ((.25,valid&(m<=0)),(.5,valid&(m.abs()<8)),(.25,valid)):
            q=q if bool(q.any()) else valid
            reg[...,h,:]+=hw*fraction*q/q.sum().clamp_min(1)
            k=q&known;cls[...,h,:]+=hw*fraction*k/k.sum().clamp_min(1)
    return reg,cls


def partial_risk_weights(batch,classification):
    """Keep predictable dangerous directions when other objects are unknown.

    Share the existing per-horizon BCE budget with the partial positives; do
    not add a second full classification loss, safe labels or margin targets.
    """
    mask=batch.get('partial_point_danger')
    if mask is None:return classification,None
    missing=~batch['trusted'][...,None]
    if 'risk_continuation_valid' in batch:
        # A random/release enclosure may censor one action even when the
        # scene-wide forecast flag is true. An independently predicted exact
        # point overlap must not disappear behind that unrelated enclosure.
        missing=missing|~batch['risk_continuation_valid']
    mask=mask&batch['unprotected_mask'][...,None,None]&missing
    if 'risk_continuation_valid' in batch:
        # Partial point predictions concern straight motion. They cannot label
        # a later turn as colliding; retain the identical immediate4f facts.
        mask=mask.clone();mask[...,1:,:]=False
    known=classification.clone();partial=torch.zeros_like(classification)
    for h,weight in enumerate((1.,.3,.1)):
        m=mask[...,h,:]
        if bool(m.any()):
            share=.5 if bool(known[...,h,:].any()) else 1.
            known[...,h,:]*=1.-share
            partial[...,h,:]=weight*share*m/m.sum()
    return known,partial


def risk_loss(scores,margins,reg,cls,*,native=None,partial=None):
    prediction=scores.transpose(-1,-2)
    error=F.smooth_l1_loss(prediction.tanh(),torch.tanh((1-margins)/8),reduction='none',beta=.1)
    bce=F.binary_cross_entropy_with_logits(8*prediction-1,(margins<=0).float(),reduction='none')
    loss=(reg*error).sum()+.2*(cls*bce).sum()
    partial_four=0.
    if partial is not None:
        # Positive-only conditional point forecasts; unknown directions have
        # zero weight. They are not unsampled native collision observations.
        positive=F.softplus(-(8*prediction-1))
        loss=loss+.2*(partial*positive).sum()
        partial_four=(partial[...,0,:]*positive[...,0,:]).sum()
    if native is not None and native['share']:
        from ..compact_native_contact_targets import native_contact_bce
        factual,_=native_contact_bce(scores,native['actions'],native['outcomes'],
            native['unprotected'],native['elapsed_frames'],weights=native['weights'])
        # Reallocate25% of the existing4f classification budget. Keep margin
        # regression and12/24f objectives; invent no unsampled native labels.
        loss=loss+.2*native['share']*(factual-(cls[...,0,:]*bce[...,0,:]).sum()-partial_four)
    return loss


def rollout_risk_weights(batch):
    """Keep actual vulnerability facts distinct from optional exposed forecasts.

    A protected row can calibrate only its measured exposed horizons. It does
    not become an unprotected contact or a safe/hit guidance row.
    """
    if 'risk_continuation_margins' in batch:
        reg,cls=risk_weights(batch['risk_continuation_margins'],batch['risk_continuation_valid'],
            batch.get('risk_training_eligible',batch['unprotected_mask']))
        if 'teacher_native_contact_removed' in batch:
            reg=reg.masked_fill(batch['risk_continuation_source']==6,0.)
        if 'risk_long_classification_only' in batch:
            # Native continuation labels describe observed contacts, not a
            # measured clearance. Never regress a synthetic distance to them.
            reg[...,1:,:]=0
            from ..nbnm_policy_continuation_risk import classification_weights
            cls=classification_weights(batch,cls)
        return reg,cls
    fields=('risk_training_eligible','risk_training_trusted')
    if any(k in batch for k in fields):
        if not all(k in batch for k in fields):raise ValueError('Paired exposed risk eligibility/trust required')
        return risk_weights(batch['margins'],batch[fields[1]],batch[fields[0]])
    return risk_weights(batch['margins'],batch['trusted'],batch['unprotected_mask'])


def optimizer_for(model, *, trainable_groups=None, include_native_attack_readout=False):
    from ..compact_policy_adam import PolicyMasterAdam
    task=getattr(model.core.input_encoder,'task_progress',None)
    context=[*model.core.input_encoder.context.parameters(),*(task.parameters() if task is not None else ())]
    extension=getattr(model.core.input_encoder,'extended_context',None)
    if extension is not None:context.extend(extension.parameters())
    prior=getattr(model.core.input_encoder,'route_prior_context',None)
    if prior is not None:context.extend(prior.parameters())
    future=getattr(model.core.input_encoder,'future_rule_context',None)
    if future is not None:context.extend(future.parameters())
    corridor=getattr(model.core.input_encoder,'corridor_context',None)
    if corridor is not None:context.extend(corridor.parameters())
    collection=getattr(model.core.input_encoder,'collection_context',None)
    if collection is not None:context.extend(collection.parameters())
    movement=list(model.route.parameters())
    if include_native_attack_readout and getattr(model,'native_attack_readout',None) is not None:
        movement.extend(model.native_attack_readout.parameters())
    groups = [
        dict(name='brain',params=list(model.core.internal_parameters()),lr=3e-6,eps=1e-10),
        dict(name='inherited_sensory_ports',params=list(model.core.input_encoder.base.parameters()),lr=3e-5),
        dict(name='new_context_ports',params=context,lr=3e-4),
        dict(name='risk',params=list(model.risk.parameters()),lr=3e-5),
        dict(name='movement',params=movement,lr=1e-4),
        dict(name='Bomb',params=[*model.bomb.parameters(),model.bomb_urgency_gain],lr=1e-3),
        dict(name='three_critics',params=[*model.critic.parameters(),
            *(model.task_critic.parameters() if hasattr(model,'task_critic') else ())],lr=3e-4)]
    if getattr(model, 'warning48_head', None) is not None:
        groups.append(dict(name='native_warning48', params=list(model.warning48_head.parameters()), lr=3e-4))
    if getattr(model,'corridor_head',None) is not None:
        groups.append(dict(name='corridor_risk',params=list(model.corridor_head.parameters()),lr=3e-4))
    if getattr(model,'corridor_witness_head',None) is not None:
        groups.append(dict(name='corridor_witness',params=list(model.corridor_witness_head.parameters()),lr=3e-4))
    if getattr(model,'corridor_course_head',None) is not None:
        groups.append(dict(name='corridor_course',params=list(model.corridor_course_head.parameters()),lr=3e-4))
    if getattr(model,'corridor_endpoint_head',None) is not None:
        groups.append(dict(name='corridor_endpoint',params=list(model.corridor_endpoint_head.parameters()),lr=3e-4))
    if getattr(model,'shared_directions',None) is not None:
        groups.append(dict(name='shared_directions',params=list(model.shared_directions.parameters()),lr=3e-4))
    if trainable_groups is not None:
        groups = [group for group in groups if group['name'] in trainable_groups]
        trainable = {id(parameter) for group in groups for parameter in group['params']}
        for parameter in model.parameters():
            parameter.requires_grad_(id(parameter) in trainable)
    return PolicyMasterAdam(groups,weight_decay=0.)


def restore_optimizer(optimizer,saved):
    """Keep old Adam moments when adding sensory gains or risk parameters.

    Existing ordered groups/parameters are unchanged. The two new root risk
    parameters precede residual parameters;new sensory gains append to the
    context group. Only added parameters start with fresh Adam state.
    """
    groups=[dict(g,params=list(g['params'])) for g in saved['param_groups']]
    next_id=max(i for g in groups for i in g['params'])+1;promoted=False;task_promoted=False
    added=[]
    for new in optimizer.state_dict()['param_groups'][len(groups):]:
        if new['name'] not in ('native_warning48','corridor_risk','corridor_witness','corridor_course','corridor_endpoint'):
            raise ValueError('Unsupported new optimizer group')
        added.append(new['name'])
        count = len(new['params'])
        groups.append(dict(new, params=list(range(next_id, next_id + count))))
        next_id += count
    if len(groups)!=len(optimizer.param_groups):raise ValueError('Optimizer groups differ')
    for old,new in zip(groups,optimizer.param_groups):
        if old.get('name')!=new.get('name'):raise ValueError('Optimizer group order differs')
        if len(old['params'])==len(new['params']):continue
        if old['name']=='three_critics' and len(new['params'])==len(old['params'])+1:
            old['params'].append(next_id);next_id+=1;continue
        if old['name']=='new_context_ports' and len(new['params'])==len(old['params'])+1:
            old['params'].append(next_id);next_id+=1;task_promoted=True;continue
        if old['name']!='risk' or len(new['params'])!=len(old['params'])+2:
            raise ValueError('Unsupported optimizer parameter change')
        old['params']=[next_id,next_id+1]+old['params'];next_id+=2;promoted=True
    optimizer.load_state_dict(dict(saved,param_groups=groups))
    return dict(risk_baseline_parameters_promoted=2 if promoted else 0, native_warning48_added='native_warning48' in added,
        corridor_risk_added='corridor_risk' in added,corridor_course_added='corridor_course' in added,
        corridor_endpoint_added='corridor_endpoint' in added,
        task_progress_port_parameters_added=1 if task_promoted else 0,existing_Adam_moments_preserved=True,
        policy_master_precision=getattr(optimizer,'precision_revision',None))


def update(model,optimizer,rollout,*,epochs=1,micro_scenes=8,seed=1,guard=lambda:None,progress=lambda x:None,critical_replay=None,world_aligned=False,checkpoint_chunks=False,boss_attack_coefficient=0.,conditional_wall_coefficient=0.,guided_wall_coefficient=0.,selective_attack_coefficient=0.,late_replay=None,late_replay_coefficient=0.,rule_archive_replay=None,rule_archive_coefficient=0.,rule_archive_every=8,rule_archive_task=None,native_rule_replay=None,native_rule_coefficient=0.,native_rule_every=8,native_rule_windows=1,max_updates=0,sample_decisions=0,stage_weights=None,target_kl=.03,completed_update_prefix=(),
           native_only_corridors=False, rule_archive_tasks=()):
    if type(max_updates) is not int or max_updates<0:raise ValueError('Nonnegative Adam update quota required')
    if any(row.get('update')!=i+1 for i,row in enumerate(completed_update_prefix)):
        raise ValueError('Resume prefix must contain consecutive committed updates')
    if rollout.get('metadata',{}).get('supervision_only'):
        raise ValueError('Deterministic student teacher-labelled trajectories are supervision only, not PPO samples')
    factual_risk=bool(getattr(model,'native_contact_risk_calibration',False))
    if factual_risk and 'native_action_frames' not in rollout:
        raise ValueError('Measured native_action_frames required;do not assume a four-frame legacy label')
    from ..compact_story_model import sequence
    from ..whole_brain_ppo import clip_gradients_fp64
    if target_kl is not None and (isinstance(target_kl,bool) or not np.isfinite(target_kl) or target_kl<=0):
        raise ValueError('KL early stopping requires a positive finite target or None to disable')
    cfg=aligned_ppo_config(ppo_epochs=epochs,target_kl=target_kl)
    guidance_options={}
    if getattr(model,'nbnm',None):
        from dataclasses import replace
        from ..nbnm_continuation_risk import enabled as continuation_risk_enabled
        if continuation_risk_enabled(model.nbnm) and 'risk_continuation_margins' not in rollout:
            raise ValueError('Continuation-risk learning requires a freshly collected matching PPO tape')
        from ..nbnm_objective import Objective, rule_guidance_coefficient, future_guidance_coefficient
        from ..fly_gru_aligned_guidance import DEFAULT_CONFIG
        objective=Objective(**model.nbnm.get('objective',{}))
        cfg=replace(cfg,gamma=1.,objective_weights=objective.weights,target_kl=None)
        guidance_options['config']=replace(DEFAULT_CONFIG,deathbomb_coefficient=0.,
            rule_coefficient=rule_guidance_coefficient(model.nbnm),
            future_coefficient=future_guidance_coefficient(model.nbnm))
        guidance_options['future_reference_from_rule']=model.nbnm.get('future_reference')=='teacher-route-v1'
    length=cfg.seq_len
    behavior_temperature=rollout_sampling_temperature(rollout)
    bomb_temperature=rollout_bomb_sampling_temperature(rollout)
    # Saved acting logits are telemetry, never a loss target. When present,
    # their full distribution identifies whether drift comes from Bomb or
    # movement, whereas the PPO estimator sees only the sampled action.
    recorded_behavior=rollout.get('recorded_behavior_policy_logits')
    if recorded_behavior is None and behavior_temperature==bomb_temperature==1.:
        recorded_behavior=rollout.get('recorded_policy_logits')
    data={k:v for k,v in rollout.items() if torch.is_tensor(v) and not k.startswith('recorded_')};T,N=data['actions'].shape
    if getattr(model,'nbnm',None):
        from ..nbnm_policy_continuation_risk import REVISION as POLICY_RISK_REVISION,prepare as policy_risk_targets
        if model.nbnm.get('risk_target_revision')==POLICY_RISK_REVISION:
            data.update(policy_risk_targets(rollout))
    native_fact_recovery=None
    if factual_risk:
        from ..compact_native_contact_targets import recover_recorded_native_outcomes
        data['outcomes'],native_fact_recovery=recover_recorded_native_outcomes(rollout)
    warning_settings = (getattr(model, 'nbnm', None) or {}).get('native_warning48')
    if warning_settings:
        from ..nbnm_policy_continuation_risk import observed_futures
        known, hits = observed_futures(dict(rollout, outcomes=data['outcomes']), (48,))
        data['warning48_known'] = known[...,0]
        data['warning48_hits'] = hits[...,0]
    if (getattr(model,'nbnm',None) or {}).get('teacher_outcome_revision'):
        from ..nbnm_teacher_outcomes import prepare as teacher_outcomes
        data.update(teacher_outcomes(dict(rollout,**data),revision=model.nbnm['teacher_outcome_revision']))
    if (getattr(model,'nbnm',None) or {}).get('corridor_risk'):
        from ..nbnm_corridor_loss import witnessed_contacts,targets as corridor_targets,restrict_reference
        observed=dict(rollout,outcomes=data['outcomes'])
        if native_only_corridors:
            from ..nbnm_policy_continuation_risk import observed_futures

            # In a new game, keep partial geometry in the neural observation.
            # Train route witnesses on trajectories the original game executed.
            known, hits = observed_futures(observed, (48, 96), same_stage=True)
            chosen = F.one_hot(data['actions'].clamp(0, 18).long(), 19)[..., :18].bool()
            data['corridor_recovered_exits'] = chosen[..., None] & (known & ~hits)[..., None, :]
            data['corridor_course_hits'] = torch.zeros_like(hits)
            duration = data['native_action_frames']
            data['corridor_label_outcomes'] = torch.where((duration > 0) & (duration <= 4), data['outcomes'], -1)
            data['corridor_label_features'] = data['features'].clone()
            labels = data['corridor_label_features'][..., 281:407]
            if 'native_branch_course_hits' in observed:
                # Each hit belongs to the exact action tape executed from this
                # frame. Use it for training, independent of action sampling.
                cues = labels.unflatten(-1, (18, 7))
                cues[..., 2:4].masked_fill_(observed['native_branch_course_hits'], -1.)
            else:
                labels.zero_()
        elif model.nbnm.get('timed_corridor_supervision'):
            if 'nbnm_verified_corridors' not in data:
                raise ValueError('Timed corridor supervision needs a fresh rollout with measured exit evidence')
            hits, clear = witnessed_contacts(observed, return_clear=True)
            chosen = torch.nn.functional.one_hot(data['actions'].clamp(0, 18).long(), 19)[..., :18].bool()
            data['corridor_verified_exits'] = data['nbnm_verified_corridors'] | (chosen[..., None] & clear[..., None, :])
            from ..nbnm_corridor_loss import native_exit_measurements
            data['corridor_native_exit_margin'], data['corridor_native_exit_room'] = native_exit_measurements(observed)
            data['corridor_course_hits'] = hits
        else:
            data['corridor_course_hits'] = witnessed_contacts(observed)
        if 'recorded_lure_course' in observed:
            data['nbnm_lure_course_hits'] = witnessed_contacts(observed, course_key='recorded_lure_course')
        if not native_only_corridors and model.nbnm['corridor_risk'].get('learn_native_exits'):
            from ..nbnm_policy_continuation_risk import observed_futures
            known, hit = observed_futures(observed, (48, 96), same_stage=True)
            native_survived = known & ~hit
        elif 'corridor_native_exit_margin' in data:
            # An exhausted search is unknown. The witness head can recover
            # an actual played exit without assigning a risk margin to it.
            native_survived = data['corridor_native_exit_margin'] > 0.
        else:
            native_survived = None
        if native_survived is not None:
            from ..nbnm_corridor_loss import recovered_exits
            data['corridor_recovered_exits'] = recovered_exits(data['features'], data['actions'], native_survived)
        ctargets=corridor_targets(data.get('corridor_label_features',data['features']),data['actions'],
            data.get('corridor_label_outcomes',data['outcomes']),data['corridor_course_hits'],
            recovered=data.get('corridor_recovered_exits'))
        if model.nbnm['corridor_risk'].get('prefer_full_horizon'):
            from ..nbnm_corridor_loss import full_horizon_preference
            ctargets,_=full_horizon_preference(data['features'],ctargets,data['corridor_recovered_exits'])
        data['rule_probabilities'],data['rule_weights']=restrict_reference(data['rule_probabilities'],data['rule_weights'],ctargets)
        data['future_replaced_by_corridor']=ctargets[3]
        if model.nbnm.get('continuation_supervision'):
            from ..nbnm_corridor_loss import continuation_preference, continuation_reference
            preferred, strength, wait = continuation_preference(data)
            data['rule_probabilities'] = continuation_reference(data['rule_probabilities'], preferred, strength)
            data['future_replaced_by_corridor'] |= strength > 0
            data['corridor_continuation_target'] = preferred
            data['corridor_continuation_strength'] = strength
            data['corridor_wait_row'] = wait
    if 'valid_mask' in data:
        from ..compact_story_ready_data import prepare_ready_targets
        a3,returns,advantages=prepare_ready_targets(data,cfg,model.device)
    else:
        a3,returns,advantages=prepare_three_value_targets(data['reward_vectors'].to(model.device),data['old_values'].to(model.device),
            data['bootstrap_values'].to(model.device),data['terminated'].to(model.device),data['episode_starts'].to(model.device),cfg)
    target_diagnostics=None
    if getattr(model,'nbnm',None):
        from ..nbnm_learning_diagnostics import target_summary
        target_diagnostics=target_summary(data,a3,returns,cfg.objective_weights)
    data.update(returns=returns.cpu(),advantages=advantages.cpu())
    sequences=[(t,w) for t in range(0,T,length) for w in range(N)];rng=np.random.default_rng(seed)
    selection=None;selected=None
    if sample_decisions:
        from ..compact_story_batch_selection import select_sequences,interleaved_groups
        valid=data.get('valid_mask',torch.ones_like(data['actions'],dtype=torch.bool))
        selected,selection=select_sequences(data['features'],valid,length,sample_decisions,rng,stage_weights)
        selection['selected_sequences']=[list(pair) for pool in selected.values() for pair in pool]
    updates=0;logs=[];model.train();first_replay_error=None;early_stop=None;quota_stop=None
    used_sequences=set()
    late_rng=np.random.default_rng(seed^0x61a7)
    for epoch in range(epochs):
        if selected is not None:
            groups=interleaved_groups(selected,rng)
        elif 'valid_mask' in data:
            from ..compact_story_ready_data import ready_sequence_groups
            groups=ready_sequence_groups(data['valid_mask'],length,rng)
        elif world_aligned:
            if micro_scenes!=16:raise ValueError('World-aligned replay requires16 real sequences per microbatch')
            from ..compact_story_sequence_groups import world_aligned_groups
            groups=world_aligned_groups(T,N,length,rng)
        else:
            order=rng.permutation(len(sequences))
            groups=[[sequences[i] for i in order[first:first+16]] for first in range(0,len(order),16)]
        for chosen in groups:
            if max_updates and updates>=max_updates:
                quota_stop=dict(reason='explicit_Adam_compute_quota',limit=max_updates,
                    shuffled_whole_sequences=True,all_collected_rows_used_for_GAE=True)
                break
            if updates<len(completed_update_prefix):
                # Rebuild the original sampler state without applying these
                # already committed gradients a second time. The saved log
                # also restores each critical window's last measured priority.
                previous=dict(completed_update_prefix[updates])
                if critical_replay is not None and (critical_replay.sequences or critical_replay.normal_sequences):
                    index=critical_replay.sample_index(rng)
                    if critical_replay.balanced and critical_replay.normal_sequences:
                        if previous.get('normal_replay_index') is None:
                            rng.integers(len(critical_replay.normal_sequences))
                        else:
                            normal_index=critical_replay.sample_normal_index(rng)
                            if normal_index!=previous['normal_replay_index']:
                                raise ValueError('Committed normal replay draw did not reconstruct')
                            normal_entry=critical_replay.normal_sequences[normal_index]
                            normal_entry['normal_priority']=previous['normal_Bomb_priority']
                            normal_entry['normal_priority_source']='latest supervised replay'
                    if critical_replay.sequences:
                        if index!=previous['critical_replay_index']:
                            raise ValueError('Committed critical replay draw did not reconstruct')
                        entry=critical_replay.sequences[index]
                        entry['critical_priority']=previous['critical_deathbomb_loss']
                        entry['critical_priority_source']='latest supervised replay'
                if late_replay_coefficient and late_replay is not None and any(late_replay.windows.values()):
                    late_rng.integers(2**31)
                if rule_archive_coefficient and rule_archive_replay is not None and (updates+1)%rule_archive_every==0:
                    late_rng.integers(2**31)
                if native_rule_coefficient and native_rule_replay is not None and (updates+1)%native_rule_every==0:
                    if native_rule_windows==1:native_rule_replay.sample_index(late_rng)
                    else:native_rule_replay.sample_indices(late_rng,native_rule_windows)
                first_replay_error=previous['initial_replay_logp_max_error']
                used_sequences.update(chosen);logs.append(previous);updates+=1
                continue
            guard();update_started=time.monotonic();B=len(chosen)
            time_index=torch.tensor([t for t,w in chosen])[None]+torch.arange(length)[:,None]
            world_index=torch.tensor([w for t,w in chosen])[None].expand(length,-1)
            batch={k:v[time_index,world_index] for k,v in data.items() if k!='bootstrap_values'}
            recorded_batch=(recorded_behavior[time_index,world_index] if recorded_behavior is not None else None)
            before=SpikingState.pack([rollout['block_states'][t].row(w) for t,w in chosen])
            task={k:batch[k] for k in ('margins','trusted','unprotected_mask','outcomes','rule_probabilities','rule_weights','deathbomb_mask')}
            if 'future_replaced_by_corridor' in batch:
                task['future_replaced_by_corridor']=batch['future_replaced_by_corridor']
            # Use whole256-row term denominators even when brain memory requires
            # microbatches. Rare deathbomb labels must not be diluted by splits.
            normalizers=gru_aligned_guidance_loss(torch.zeros(length,B,19),**task,actions=batch['actions'],
                veto_rule_on_contact=True,**guidance_options)['denominators']
            attack_normalizer=None
            position_coefficient=getattr(model,'combat_position_guidance_coefficient',0.)
            position_normalizer=None
            if position_coefficient:
                from ..compact_combat_position import position_targets
                position_kwargs={}
                if getattr(model,'lower_field_default',False):
                    from ..compact_lower_field import position_targets
                    position_kwargs=dict(exceptions=batch.get('lower_field_exception'))
                position_normalizer=position_targets(batch['features'],batch['margins'],batch['trusted'],
                    batch['unprotected_mask'],batch['rule_probabilities'],batch['rule_weights'],**position_kwargs)[0].sum()
            if boss_attack_coefficient:
                from ..compact_boss_attack_guidance import eligible_and_closing
                attack_normalizer=eligible_and_closing(batch['features'],batch['margins'],batch['trusted'],
                    batch['unprotected_mask'],batch['rule_probabilities'],batch['rule_weights'])[0].sum()
            wall_normalizer=None
            if conditional_wall_coefficient:
                from ..compact_boss_attack_guidance import wall_recovery_candidates
                wall_normalizer=wall_recovery_candidates(batch['features'],batch['margins'],
                    batch['unprotected_mask'],batch['rule_weights'])[0].sum()
            guided_wall_normalizer=None
            if guided_wall_coefficient:
                from ..compact_guided_wall_candidate import eligible_and_target
                guided_wall_normalizer=eligible_and_target(batch['features'],batch['margins'],batch['trusted'],
                    batch['unprotected_mask'],batch['rule_probabilities'],batch['rule_weights'])[0].sum()
            selective_attack_normalizer=None
            if selective_attack_coefficient:
                from ..compact_selective_attack_guidance import eligible_and_target as selective_attack_targets
                selective_attack_normalizer=selective_attack_targets(batch['features'],batch['margins'],batch['trusted'],
                    batch['unprotected_mask'],batch['rule_probabilities'],batch['rule_weights'])[0].sum()
            rw,cw=rollout_risk_weights(batch)
            cw,pw=partial_risk_weights(batch,cw)
            native_weights=native_counts=None;native_share=0.
            if factual_risk:
                from ..compact_native_contact_targets import native_contact_weights,native_exposure_mask
                native_weights,native_counts=native_contact_weights(batch['actions'],batch['outcomes'],
                    native_exposure_mask(batch['features'],batch['unprotected_mask']),batch['native_action_frames'])
                native_share=.25 if bool(native_weights.any()) else 0.
            optimizer.zero_grad(set_to_none=True);metrics=dict(loss=0.,PPO=0.,guidance=0.,risk=0.,KL=0.,entropy=0.,
                guidance_safe=0.,guidance_future=0.,guidance_rule=0.,guidance_contact=0.)
            if getattr(model,'nbnm',None):
                metrics['rule_guidance_coefficient']=guidance_options['config'].rule_coefficient
                metrics['effective_future_guidance_coefficient']=(guidance_options['config'].safe_coefficient
                    *guidance_options['config'].future_coefficient)
                if 'teacher_native_contradiction' in batch:
                    metrics['teacher_followed_native_failed_rows']=int(batch['teacher_native_contradiction'].sum())
                    metrics['teacher_immediate_contact_actions_removed']=int(batch['teacher_native_contact_removed'].sum())
                if 'teacher_forecast_dead_end_censored' in batch:
                    metrics['teacher_forecast_dead_end_censored_rows']=int(batch['teacher_forecast_dead_end_censored'].sum())
                    metrics['teacher_abstaining_native_contact_rows']=int(batch['teacher_abstaining_native_contact'].sum())
            if warning_settings:
                warning_count = int(batch['warning48_known'].sum())
                metrics.update(native_warning48_loss=0., native_warning48_rows=warning_count,
                    native_warning48_contacts=int(batch['warning48_hits'].sum()))
            corridor_settings=(getattr(model,'nbnm',None) or {}).get('corridor_risk')
            local_guidance = []
            if 'nbnm_calm_action' in batch and 'corridor_wait_row' not in batch:
                rows = batch['nbnm_calm_action'] & (batch['outcomes'] != 1)
                targets = (torch.arange(18) % 9 == 0).expand(*rows.shape, 18)
                local_guidance.append(('calm', targets, rows, .14 * objective.calm_cost_limit))
            lure = None
            if 'nbnm_lure_moves' in batch and corridor_settings and 'corridor_verified_exits' not in batch:
                from ..nbnm_corridor_loss import lure_targets
                lure = lure_targets(batch)
            exit_batch = None
            exit_settings = corridor_settings or {}
            all_scenes = exit_settings.get('exit_supervision', False)
            road_stages = exit_settings.get('road_exit_supervision_stages')
            if all_scenes or road_stages:
                from ..nbnm_corridor_loss import exit_supervision
                allowed, short_rows, exits, weights = exit_supervision(
                    batch, road_stages if not all_scenes else None,
                    allow_48=exit_settings.get('exit_supervision_48_fallback', False))
                if all_scenes and lure is not None:
                    # Give a validated maneuver the long-horizon target for
                    # this scene, while respecting known first-step collisions.
                    lure &= allowed | ~short_rows[..., None]
                    weights = weights.masked_fill(lure.any(-1), 0.)
                prefix = 'exit' if all_scenes else 'road'
                exit_metric = 'exit_corridor_action' if all_scenes else 'road_exit_action'
                exit_count = (weights > 0).sum()
                metrics.update({f'{exit_metric}_rows': int(exit_count), f'{exit_metric}_loss': 0.})
                if exit_count:
                    exit_batch = allowed, exits, weights, exit_count.clamp_min(8).to(model.device)
            if lure is not None:
                local_guidance.append(('issun', lure, lure.any(-1), corridor_settings['issun_guidance_weight']))
            if all_scenes or road_stages:
                local_guidance.append((f'{prefix}_short', allowed, short_rows, .10))
            guidance_batches = []
            for name, targets, rows, coefficient in local_guidance:
                count = int(rows.sum())
                metrics.update({f'{name}_action_rows': count, f'{name}_action_loss': 0.})
                if count:
                    guidance_batches.append((name, targets, rows, coefficient, rows.sum().clamp_min(8).to(model.device)))
            if corridor_settings:
                from ..nbnm_corridor_loss import targets as corridor_targets,loss as corridor_loss
                recovered=batch.get('corridor_recovered_exits')
                corridor_features = batch.get('corridor_label_features', batch['features'])
                corridor_outcomes = batch.get('corridor_label_outcomes', batch['outcomes'])
                cdata=corridor_targets(corridor_features,batch['actions'],corridor_outcomes,batch['corridor_course_hits'],
                    recovered=recovered)
                if recovered is not None:
                    metrics.update(corridor_native_recovered48=int(recovered[...,0].sum()),
                        corridor_native_recovered96=int(recovered[...,1].sum()))
                cnormal=(cdata[1].sum().to(model.device),cdata[3].sum().to(model.device))
                full_horizon_policy=None
                if corridor_settings.get('prefer_full_horizon'):
                    preferred,mass=full_horizon_preference(batch['features'],cdata,recovered)
                    full_horizon_policy=(preferred,mass,mass.sum().to(model.device))
                    metrics.update(corridor_full_horizon_rows=int(preferred[3].sum()),corridor_full_horizon_mass=float(mass.sum()))
                if 'corridor_continuation_strength' in batch:
                    preferred = batch['corridor_continuation_target']
                    strength = batch['corridor_continuation_strength']
                    active = strength > 0
                    wait = batch['corridor_wait_row']
                    policy_base, mass, _ = full_horizon_policy or (cdata, cdata[3].float(), None)
                    quality = torch.where(active[..., None], preferred.to(cdata[2].dtype), policy_base[2])
                    apply = policy_base[3] | active
                    mass = torch.where(active, strength, mass)
                    full_horizon_policy = ((*policy_base[:2], quality, apply), mass, mass.sum().to(model.device))
                    selected = preferred.gather(-1, batch['actions'].clamp(0,17)[..., None])[..., 0]
                    selected &= (batch['actions'] < 18)
                    moving = active & ~wait
                    metrics.update(corridor_continuation_rows=int(moving.sum()),
                        corridor_continuation_verified_rows=int((moving & (strength == 1.)).sum()),
                        corridor_continuation_followed=int((moving & selected).sum()),
                        corridor_wait_rows=int(wait.sum()), corridor_wait_followed=int((wait & selected).sum()))
                    if 'corridor_verified_exits' in batch:
                        verified = batch['corridor_verified_exits'] & preferred[..., None]
                        native = batch['corridor_native_exit_margin']
                        recorded = preferred.gather(-1, batch['actions'].clamp(0,17)[..., None])[..., 0]
                        exact48 = verified[..., 0].any(-1) | (recorded & (native[..., 0] > 0.))
                        exact96 = verified[..., 1].any(-1) | (recorded & (native[..., 1] > 0.))
                        turns = (batch['nbnm_due_turns'] & preferred).any(-1)
                        metrics.update(corridor_verified48_rows=int((moving & exact48 & ~exact96 & (strength == 1.)).sum()),
                            corridor_verified96_rows=int((moving & exact96 & (strength == 1.)).sum()),
                            corridor_due_turn_rows=int((moving & turns).sum()),
                            corridor_due_turn_followed=int((moving & turns & selected).sum()))
                    if recorded_batch is not None:
                        greedy = recorded_batch[..., :18].argmax(-1, keepdim=True)
                        greedy_supported = preferred.gather(-1, greedy)[..., 0]
                        metrics.update(corridor_continuation_greedy_followed=int((moving & greedy_supported).sum()),
                            corridor_wait_greedy_followed=int((wait & greedy_supported).sum()))
                order_data=None
                if corridor_settings.get('order_supported_exits'):
                    from ..nbnm_corridor_loss import supported_exit_order,supported_exit_order_loss
                    order_data=supported_exit_order(batch['features'],cdata)
                    order_norm=order_data[1].sum().to(model.device)
                    metrics.update(corridor_exit_order_loss=0.,corridor_exit_order_rows=int(order_data[1].any(-1).sum()))
                witness_data=None
                if corridor_settings.get('witness_weight'):
                    from ..nbnm_corridor_loss import witness_targets
                    witness_data=witness_targets(corridor_features,batch['actions'],corridor_outcomes,batch['corridor_course_hits'],
                        recovered=recovered)
                    witness_norm=witness_data[1].sum().clamp_min(1).to(model.device)
                    metrics['corridor_witness_loss']=0.
                course_data=None
                if corridor_settings.get('keep_course'):
                    from ..nbnm_corridor_loss import course_targets,course_loss
                    ct,cr=course_targets(batch['features'],batch['actions'],batch['outcomes'],batch['corridor_course_hits'],
                        allow_replans=corridor_settings.get('revalidated_course_only',False),recovered=recovered)
                    cr=cr & batch['unprotected_mask']
                    course_data=(ct,cr)
                    course_norm=cr.sum().clamp_min(1).to(model.device)
                    metrics.update(corridor_course_loss=0.,corridor_course_rows=int(cr.sum()))
                metrics.update(corridor_risk_loss=0.,corridor_policy_loss=0.,corridor_route_rows=int(cnormal[1]),
                    corridor_matched_path_contacts48=int(batch['corridor_course_hits'][...,0].sum()),
                    corridor_matched_path_contacts96=int(batch['corridor_course_hits'][...,1].sum()),
                    corridor_label_actions48=int((cdata[1][...,0]>0).sum()),corridor_label_actions96=int((cdata[1][...,1]>0).sum()))
            if recorded_batch is not None:
                metrics.update(exact_action_KL=0.,exact_bomb_KL=0.,exact_weighted_movement_KL=0.,bomb_KL_legacy_saturation_rows=0)
            if boss_attack_coefficient:metrics['boss_attack_guidance']=0.
            if position_coefficient:
                metrics.update(combat_position_guidance=0.,combat_position_rows=int(position_normalizer))
            if conditional_wall_coefficient:
                metrics.update(conditional_wall_guidance=0.,conditional_wall_rows=int(wall_normalizer))
            if guided_wall_coefficient:
                metrics.update(guided_wall_guidance=0.,guided_wall_rows=int(guided_wall_normalizer))
            if selective_attack_coefficient:
                metrics.update(selective_attack_guidance=0.,selective_attack_rows=int(selective_attack_normalizer))
            if factual_risk:
                metrics.update(native_risk_clear_rows=int(native_counts[0]),native_risk_contact_rows=int(native_counts[1]),
                    native_risk_4f_BCE_share=native_share)
            if pw is not None:
                metrics['partial_point_conditional_danger_pairs']=int((pw>0).sum())
            if 'risk_continuation_source' in batch:
                from ..nbnm_continuation_risk import conflict_metrics
                # Acting scores remain CPU telemetry, never loss inputs.
                telemetry={k:rollout[k][time_index,world_index] for k in
                    ('recorded_policy_logits','recorded_neural_horizon_scores')}
                metrics.update(conflict_metrics(dict(batch,**telemetry),model))
                sources=batch['risk_continuation_source'][...,1:,:]
                valid_targets=batch['risk_continuation_valid'][...,1:,:]
                metrics.update(continuation_safe_turn_pairs=int(((sources==3)&valid_targets).sum()),
                    continuation_straight_safe_pairs=int(((sources==1)&valid_targets).sum()),
                    continuation_immediate_contact_pairs=int(((sources==2)&valid_targets).sum()),
                    continuation_unknown_pairs=int((~valid_targets).sum()))
                if 'risk_native_future_known' in batch:
                    for h,frames in enumerate((12,24)):
                        known=batch['risk_native_future_known'][...,h]
                        hit=batch['risk_native_future_contact'][...,h]
                        metrics[f'native_continuation_{frames}f_rows']=int(known.sum())
                        metrics[f'native_continuation_{frames}f_contacts']=int((known&hit).sum())
            for start in range(0,B,micro_scenes):
                stop=min(start+micro_scenes,B);count=stop-start
                micro={k:v[:,start:stop].to(model.device) for k,v in batch.items()}
                state=SpikingState(*(v[start:stop].to(model.device) for v in before.tensors()))
                resets=None
                if rollout.get('episode_initial_states'):
                    from ..nbnm_prefix import replay_resets
                    resets=replay_resets(rollout['episode_initial_states'],chosen[start:stop],length,model.device)
                result=sequence(model,micro['features'],micro['episode_starts'],state,micro['actions'],checkpoint_chunks=checkpoint_chunks,reset_states=resets)
                behavior_logits=story_behavior_logits(result['logits'],behavior_temperature,bomb_temperature)
                if recorded_batch is not None:
                    from ..compact_policy_kl import action_kl_parts
                    with torch.no_grad():
                        old=recorded_batch[:,start:stop].to(model.device)
                        total,bomb,movement=action_kl_parts(old,behavior_logits.detach())
                        metrics['exact_action_KL']+=float(total.mean())*count/B
                        metrics['exact_bomb_KL']+=float(bomb.mean())*count/B
                        metrics['exact_weighted_movement_KL']+=float(movement.mean())*count/B
                        old_bomb=old.log_softmax(-1)[...,18].exp()
                        new_bomb=behavior_logits.detach().log_softmax(-1)[...,18].exp()
                        metrics['bomb_KL_legacy_saturation_rows']+=int(((old_bomb<1)&(new_bomb==1)).sum())
                if first_replay_error is None:
                    with torch.no_grad():
                        replay=behavior_logits.log_softmax(-1).gather(-1,micro['actions'][...,None])[...,0]
                        first_replay_error=float((replay-micro['old_log_probs']).abs().max())
                    if first_replay_error>.05:raise RuntimeError('Initial policy replay mismatch: '+str(first_replay_error))
                ppo=three_value_ppo_loss(behavior_logits,result['values'],micro['actions'],micro['old_log_probs'],
                    micro['old_values'],micro['advantages'],micro['returns'],cfg)
                aux={k:micro[k] for k in task}
                guide=gru_aligned_guidance_loss(result['logits'],**aux,actions=micro['actions'],normalizers=normalizers,
                    veto_rule_on_contact=True,
                    position_features=micro['features'] if getattr(model,'lower_field_default',False) else None,
                    position_exceptions=micro.get('lower_field_exception'),**guidance_options)
                native=None if not factual_risk else dict(actions=micro['actions'],outcomes=micro['outcomes'],
                    unprotected=native_exposure_mask(micro['features'],micro['unprotected_mask']),elapsed_frames=micro['native_action_frames'],
                    weights=native_weights[:,start:stop].to(model.device),share=native_share)
                risk=risk_loss(result['horizon_scores'],micro.get('risk_continuation_margins',micro['margins']),rw[:,start:stop].to(model.device),cw[:,start:stop].to(model.device),native=native,
                    partial=None if pw is None else pw[:,start:stop].to(model.device))
                loss=ppo['total_loss']*(count/B)+guide['total_loss']+risk
                if guidance_batches:
                    from ..nbnm_corridor_loss import course_loss as local_action_loss
                    for name, targets, rows, coefficient, normalizer in guidance_batches:
                        value = local_action_loss(result['logits'][..., :18],
                            targets[:, start:stop].to(model.device), rows[:, start:stop].to(model.device), normalizer)
                        loss = loss + coefficient * value
                        metrics[f'{name}_action_loss'] += float(value.detach())
                if exit_batch is not None:
                    from ..nbnm_corridor_loss import course_loss as exit_loss
                    allowed, exits, weights = (part[:, start:stop].to(model.device)
                                                for part in exit_batch[:3])
                    # Compare exit candidates among the admissible first steps
                    # in this loss only. Acting and PPO use all model logits.
                    scores = result['logits'][..., :18].masked_fill(~allowed, -1e8)
                    value = exit_loss(scores, exits, weights > 0, exit_batch[3], row_weights=weights)
                    loss = loss + .30 * value
                    metrics[f'{exit_metric}_loss'] += float(value.detach())
                if corridor_settings:
                    cv=tuple(x[:,start:stop].to(model.device) for x in cdata)
                    policy_preference=None
                    if full_horizon_policy is not None:
                        preferred,mass,policy_norm=full_horizon_policy
                        policy_preference=(tuple(x[:,start:stop].to(model.device) for x in preferred),
                            mass[:,start:stop].to(model.device),policy_norm)
                    cscore,cpolicy=corridor_loss(result['corridor_logits'],result['logits'],cv,cnormal,policy_preference=policy_preference)
                    if order_data is not None:
                        labels=tuple(x[:,start:stop].to(model.device) for x in order_data)
                        ordered=supported_exit_order_loss(result['corridor_logits'][...,1],labels,order_norm)
                        cscore=cscore+.5*ordered
                        metrics['corridor_exit_order_loss']+=float(ordered.detach())
                    loss=loss+corridor_settings['loss_weight']*(cscore+.25*cpolicy)
                    metrics['corridor_risk_loss']+=float(cscore.detach())
                    metrics['corridor_policy_loss']+=float(cpolicy.detach())
                    if witness_data is not None:
                        wy,ww=(a[:,start:stop].to(model.device) for a in witness_data)
                        wl=(torch.nn.functional.binary_cross_entropy_with_logits(result['corridor_witness_logits'],wy,reduction='none')*ww).sum()/witness_norm
                        loss=loss+corridor_settings['loss_weight']*wl
                        metrics['corridor_witness_loss']+=float(wl.detach())
                    if course_data is not None:
                        ct,cr=(part[:,start:stop].to(model.device) for part in course_data)
                        readout=course_loss(5*result['corridor_course_logits'].sigmoid(),ct,cr,course_norm)
                        route=course_loss(result['logits'][...,:18],ct,cr,course_norm)
                        cl=readout+.25*route
                        loss=loss+corridor_settings['loss_weight']*cl
                        metrics['corridor_course_loss']+=float(cl.detach())
                if warning_settings and warning_count:
                    from ..nbnm_native_warning import selected_loss
                    warning_loss = selected_loss(result['warning48_logits'], micro['actions'],
                        micro['warning48_known'], micro['warning48_hits'], warning_count)
                    loss = loss + warning_settings.get('loss_weight', .1) * warning_loss
                    metrics['native_warning48_loss'] += float(warning_loss.detach())
                if position_coefficient:
                    from ..compact_combat_position import combat_position_loss
                    position_kwargs={}
                    if getattr(model,'lower_field_default',False):
                        from ..compact_lower_field import combat_position_loss
                        position_kwargs=dict(exceptions=micro.get('lower_field_exception'))
                    position,_=combat_position_loss(result['logits'],micro['features'],micro['margins'],
                        micro['trusted'],micro['unprotected_mask'],micro['rule_probabilities'],
                        micro['rule_weights'],normalizer=position_normalizer,**position_kwargs)
                    loss=loss+position_coefficient*position
                    metrics['combat_position_guidance']+=float(position.detach())*position_coefficient
                if boss_attack_coefficient:
                    from ..compact_boss_attack_guidance import boss_attack_guidance
                    attack,_=boss_attack_guidance(result['logits'],micro['features'],micro['margins'],micro['trusted'],
                        micro['unprotected_mask'],micro['rule_probabilities'],micro['rule_weights'],normalizer=attack_normalizer)
                    loss=loss+boss_attack_coefficient*attack
                    metrics['boss_attack_guidance']+=float(attack.detach())*boss_attack_coefficient
                if conditional_wall_coefficient:
                    from ..compact_boss_attack_guidance import conditional_wall_guidance
                    wall,_=conditional_wall_guidance(result['logits'],micro['features'],micro['margins'],
                        micro['unprotected_mask'],micro['rule_weights'],normalizer=wall_normalizer)
                    loss=loss+conditional_wall_coefficient*wall
                    metrics['conditional_wall_guidance']+=float(wall.detach())*conditional_wall_coefficient
                if guided_wall_coefficient:
                    from ..compact_guided_wall_candidate import guided_wall_loss
                    guided,_=guided_wall_loss(result['logits'],micro['features'],micro['margins'],micro['trusted'],
                        micro['unprotected_mask'],micro['rule_probabilities'],micro['rule_weights'],
                        normalizer=guided_wall_normalizer)
                    loss=loss+guided_wall_coefficient*guided
                    metrics['guided_wall_guidance']+=float(guided.detach())*guided_wall_coefficient
                if selective_attack_coefficient:
                    from ..compact_selective_attack_guidance import selective_attack_loss
                    attack,_=selective_attack_loss(result['logits'],micro['features'],micro['margins'],micro['trusted'],
                        micro['unprotected_mask'],micro['rule_probabilities'],micro['rule_weights'],
                        normalizer=selective_attack_normalizer)
                    loss=loss+selective_attack_coefficient*attack
                    metrics['selective_attack_guidance']+=float(attack.detach())*selective_attack_coefficient
                if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite full-story SNN loss')
                loss.backward()
                metrics['loss']+=float(loss.detach());metrics['PPO']+=float(ppo['total_loss'].detach())*count/B
                metrics['guidance']+=float(guide['total_loss'].detach());metrics['risk']+=float(risk.detach())
                for term,source in (('safe','safe_loss'),('future','future_loss'),('rule','rule_loss'),('contact','collision_loss')):
                    metrics['guidance_'+term]+=float(guide[source].detach())
                metrics['KL']+=float(ppo['approx_kl'].detach())*count/B;metrics['entropy']+=float(ppo['entropy'].detach())*count/B
                del result,behavior_logits,loss,ppo,guide,risk,micro,state
            # Check BEFORE the next optimizer step, including the first epoch.
            # Auxiliary gradients can move the policy outside PPO's clipped
            # objective too. Keep applied steps; collect fresh on-policy data
            # after stopping this pass. This is not whole-rollout rejection.
            measured_kl=metrics.get('exact_action_KL',metrics['KL'])
            if cfg.target_kl is not None and measured_kl>cfg.target_kl:
                optimizer.zero_grad(set_to_none=True)
                early_stop=dict(reason='policy_KL_before_next_optimizer_step',epoch=epoch+1,
                    measured_KL=measured_kl,target_KL=cfg.target_kl,
                    estimator='exact_behavior_distribution' if recorded_batch is not None else 'sampled_action',
                    optimizer_updates_kept=updates,rolled_back_updates=0,
                    unupdated_minibatch_rows=length*B)
                break
            metrics['on_policy_forward_backward_seconds']=time.monotonic()-update_started
            replay_started=time.monotonic()
            if critical_replay is not None and (critical_replay.sequences or critical_replay.normal_sequences):
                replay_options=dict(index=critical_replay.sample_index(rng))
                if critical_replay.balanced and critical_replay.normal_sequences:
                    replay_options['normal_index']=critical_replay.sample_normal_index(rng)
                replay_loss,replay_info=critical_replay.loss(model,**replay_options)
                if not bool(torch.isfinite(replay_loss)):raise FloatingPointError('Nonfinite real-window teacher replay loss')
                replay_loss.backward();metrics['loss']+=float(replay_loss.detach());metrics.update(replay_info)
                del replay_loss
            metrics['critical_replay_seconds']=time.monotonic()-replay_started
            replay_started=time.monotonic()
            if late_replay_coefficient and late_replay is not None and any(late_replay.windows.values()):
                late_loss,late_info=late_replay.loss(model,index=int(late_rng.integers(2**31)))
                if not bool(torch.isfinite(late_loss)):raise FloatingPointError('Nonfinite late-stage supervised replay loss')
                (late_replay_coefficient*late_loss).backward()
                metrics['loss']+=float(late_loss.detach())*late_replay_coefficient
                metrics.update(late_info)
                del late_loss
            metrics['late_replay_seconds']=time.monotonic()-replay_started
            replay_started=time.monotonic()
            if rule_archive_coefficient and rule_archive_replay is not None and (updates+1)%rule_archive_every==0:
                archive_args = {'index': int(late_rng.integers(2**31))}
                archive_slot = updates // rule_archive_every
                if archive_slot < len(rule_archive_tasks):
                    archive_args['task_key'] = rule_archive_tasks[archive_slot]
                elif rule_archive_task and updates + 1 == rule_archive_every:
                    archive_args['task_key'] = rule_archive_task
                archived_loss, archived_info = rule_archive_replay.loss(model, **archive_args)
                if not bool(torch.isfinite(archived_loss)):raise FloatingPointError('Nonfinite archived rule-teacher movement loss')
                (rule_archive_coefficient*archived_loss).backward()
                metrics['loss']+=float(archived_loss.detach())*rule_archive_coefficient
                metrics.update(archived_info)
                del archived_loss
            metrics['archived_teacher_seconds']=time.monotonic()-replay_started
            replay_started=time.monotonic()
            if native_rule_coefficient and native_rule_replay is not None and (updates+1)%native_rule_every==0:
                if native_rule_windows==1:
                    native_rule_loss,native_rule_info=native_rule_replay.loss(model,index=native_rule_replay.sample_index(late_rng))
                else:
                    native_rule_loss,native_rule_info=native_rule_replay.loss_batch(model,
                        indices=native_rule_replay.sample_indices(late_rng,native_rule_windows))
                if not bool(torch.isfinite(native_rule_loss)):raise FloatingPointError('Nonfinite current-sensor rule-teacher loss')
                (native_rule_coefficient*native_rule_loss).backward()
                metrics['loss']+=float(native_rule_loss.detach())*native_rule_coefficient
                metrics.update(native_rule_info)
                del native_rule_loss
            metrics['native_teacher_seconds']=time.monotonic()-replay_started
            optimizer_started=time.monotonic()
            norm=clip_gradients_fp64([p for g in optimizer.param_groups for p in g['params']],cfg.max_grad_norm)
            if not np.isfinite(norm):raise FloatingPointError('Nonfinite full-brain optimizer gradient')
            optimizer.step();updates+=1
            metrics['optimizer_seconds']=time.monotonic()-optimizer_started
            used_sequences.update(chosen)
            row=dict(update=updates,epoch=epoch+1,**metrics,gradient_norm=norm,initial_replay_logp_max_error=first_replay_error,
                trained_student_rows=length*B,batches_rejected=0,update_seconds=time.monotonic()-update_started)
            observed_stages=(batch['features'][...,111]*6).round().long()
            row.update({f'PPO_stage{s}_rows':int((observed_stages==s).sum()) for s in range(1,7)})
            logs.append(row);progress(row)
        if early_stop is not None or quota_stop is not None:break
    real_rows=int(data['valid_mask'].sum()) if 'valid_mask' in data else T*N
    return dict(optimizer_updates=updates,logs=logs,early_stop=early_stop,compute_quota_stop=quota_stop,
        native_fact_recovery=native_fact_recovery,
        resumed_completed_updates=len(completed_update_prefix),optimizer_updates_performed=updates-len(completed_update_prefix),
        selection=selection,
        selected_student_rows=selection['selected_rows'] if selection is not None else real_rows,
        trained_rows_by_stage={str(s):sum(r[f'PPO_stage{s}_rows'] for r in logs) for s in range(1,7)},
        collected_student_rows=real_rows,trained_student_rows=sum(r['trained_student_rows'] for r in logs),
        unique_trained_student_rows=len(used_sequences)*length,
        unoptimized_student_rows=real_rows-len(used_sequences)*length,
        initial_replay_logp_max_error=first_replay_error,target_diagnostics=target_diagnostics)
