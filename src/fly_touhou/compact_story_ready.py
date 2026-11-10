"""Ready-first full-story collection with a global real-decision budget.

One policy version collects each segment. Native worlds pause at acknowledged
boundaries for PPO; game positions and carried brain states survive updates.
"""
import time
import numpy as np
import torch

from .compact_story_ppo import Collector,invalidate_contradictory_risk
from .compact_story_behavior import story_behavior_logits
from .compact_story_coverage import StoryCoverage
from .compact_story_ready_data import pack_world_records
from .spiking_engineered_control import SpikingState
from .thrl_durable_records import append_jsonl


class ReadyCollector(Collector):
    def __init__(self,*args,ready_batch_size=7,ready_wait_ms=1.,per_world_quota=False,**kwargs):
        if not 1<=ready_batch_size<=7 or not 0<=ready_wait_ms<=2:
            raise ValueError('Ready batching requires1..7 worlds and0..2ms coalescing')
        super().__init__(*args,**kwargs)
        self.ready_batch_size=int(ready_batch_size);self.ready_wait_ms=float(ready_wait_ms)
        self.per_world_quota=bool(per_world_quota)
        self.obs={k:v.copy() for k,v in self.obs.items()};self.policy_version=0

    @torch.no_grad()
    def rollout(self,model,*,steps=128,length=16,guard=lambda:None,progress=lambda x:None,
                supervision_selection=None,total_decisions=None):
        from .compact_native_forecast import native_repair_revisions
        from .compact_partial_risk import veto_selected_clear
        total=steps*self.workers if total_decisions is None else total_decisions
        if length!=16 or total%256 or total<self.workers*length:
            raise ValueError('Global segments need full256-row PPO batches and at least16 decisions/world')
        if self.vector._inflight:raise RuntimeError('Previous collection did not drain its native commands')
        model.eval();self.policy_version+=1;begun=time.monotonic()
        records=[[] for _ in range(self.workers)];states=[{} for _ in range(self.workers)]
        counts=np.zeros(self.workers,np.int64);limits=np.full(self.workers,length,np.int64)
        reserved=self.workers;blocks=total//length;pending={};batch_counts={}
        fixed_quota=bool(getattr(self,'per_world_quota',False))
        if fixed_quota:
            if total%(self.workers*length):raise ValueError('Per-world batches need equal complete16-step quotas')
            limits.fill(total//self.workers);reserved=blocks
        # Optional task-level quotas preserve hard-card exposure while still
        # assigning blocks to the first ready world WITHIN each task group.
        groups=getattr(self,'worker_groups',None)
        remaining=None
        if groups is not None:
            if fixed_quota:raise ValueError('Task quotas use global ready scheduling')
            from .nbnm_sampling import group_block_budget
            remaining=group_block_budget(groups,self.group_weights,blocks)
            for group in groups:remaining[group]-=1
        coverage=StoryCoverage(self.workers);last_report=0;last_reserved=None
        stats=dict(Bomb_requests=0,actual_bombs=0,contacts=0,deathbombs=0,
            normal_teacher_rows=0,normal_teacher_guided_rows=0,full_native_teacher_restored_rows=0,
            safe_forecast_but_native_contact=0)
        trusted_counts=np.zeros(3,np.int64)
        episode_initial_states={}
        selector=None;last_label={}
        if supervision_selection is not None:
            from .compact_story_action_selection import StoryActionSelector
            if supervision_selection not in ('joint_mode','family_mode'):
                raise ValueError('Deterministic supervision-only decoder required')
            selector=StoryActionSelector(supervision_selection,self.workers,seed=0)

        def dispatch(worlds):
            nonlocal last_label
            for first in range(0,len(worlds),self.ready_batch_size):
                guard();indices=worlds[first:first+self.ready_batch_size]
                if any(i in self.vector._inflight or i in pending for i in indices):
                    raise RuntimeError('Cannot dispatch an already busy world')
                idx=torch.tensor(indices,device=model.device)
                before=SpikingState(*(v.index_select(0,idx) for v in self.state.tensors()))
                for j,i in enumerate(indices):
                    if counts[i]%length==0:states[i][int(counts[i])]=before.row(j).detach().to('cpu')
                    if self.starts[i] and self.infos[i].get('policy_burn_in'):
                        episode_initial_states[(int(counts[i]),i)]=before.row(j).detach().to('cpu')
                features=self.obs['features'][indices].copy();infos=[self.infos[i] for i in indices]
                labels=[x['compact_targets'] for x in infos];experts=[x['teacher_label'] for x in infos]
                last_label=labels[0]
                for label,expert in zip(labels,experts):
                    if (label['frame'],label['native_episode'])!=(expert['frame'],expert['native_episode']):
                        raise RuntimeError('Teacher and risk must describe the current student state')
                def choose(logits):
                    if selector is not None:return selector(logits,indices)
                    return torch.distributions.Categorical(logits=story_behavior_logits(logits,
                        self.movement_sampling_temperature,self.bomb_sampling_temperature),validate_args=False).sample()
                inference=getattr(self,'inference_graphs',None)
                if inference is not None and selector is None:
                    out,following,dn=inference(torch.as_tensor(features,device=model.device),before)
                else:
                    out,following,dn=model.step(torch.as_tensor(features,device=model.device),before,action_selector=choose)
                behavior=story_behavior_logits(out['logits'],self.movement_sampling_temperature,self.bomb_sampling_temperature)
                if selector is not None:
                    behavior=torch.full_like(out['logits'],-torch.inf)
                    behavior.scatter_(1,out['actions'][:,None],0.)
                distribution=torch.distributions.Categorical(logits=behavior,validate_args=False)
                actions=out['actions'].cpu().numpy()
                network={k:out[k].cpu().numpy() for k in ('values','logits','route_scores','horizon_scores','DN_mean')}
                logp=distribution.log_prob(out['actions']).cpu().numpy();selected=dn.cpu().numpy()
                behavior_cpu=behavior.cpu().numpy()
                for j,i in enumerate(indices):
                    f=features[j];label=labels[j];expert=experts[j];normal=bool(label['normal'])
                    weight=float(expert.get('guidance_weight',1.))
                    if model.phase_valid_control:weight*=float(f[14]>.5 or f[17]>.5)
                    record=dict(features=f,actions=np.int64(actions[j]),old_log_probs=logp[j],
                        old_values=network['values'][j],episode_starts=bool(self.starts[i]),
                        margins=np.asarray(label['margins'],np.float32),trusted=np.asarray(label['trusted'],bool),
                        unprotected_mask=normal,rule_probabilities=np.asarray(expert['probabilities'],np.float32),
                        rule_weights=np.float32(weight),deathbomb_mask=bool(not model.bombs_disabled and f[4]>0 and f[12]>0 and f[17]>.5),
                        lower_field_exception=bool(label.get('lower_field_exception',f[111]>.99)
                            or expert.get('native_route_guidance',{}).get('applied',False)),
                        partial_point_danger=np.asarray(label.get('partial_point_danger',np.zeros((3,18),bool)),bool),
                        risk_training_eligible=bool(label.get('risk_training_eligible',normal)),
                        risk_training_trusted=np.asarray(label.get('risk_training_trusted',label['trusted']),bool),
                        recorded_native_stage=np.int16(infos[j]['stage']),
                        recorded_native_episode=np.int64(label['native_episode']),
                        recorded_episode_frames=np.int64(infos[j]['episode_frames']),
                        recorded_teacher_Bomb_corrected=bool(expert.get('resource_reference_correction',{}).get('applied',False)),
                        recorded_neural_horizon_scores=network['horizon_scores'][j],
                        recorded_policy_logits=network['logits'][j],recorded_behavior_policy_logits=behavior_cpu[j],
                        recorded_route_scores=network['route_scores'][j],recorded_selected_DN=selected[j])
                    record['recorded_pre_action_trusted']=record['trusted'].copy()
                    record['recorded_pre_action_partial_point_danger']=record['partial_point_danger'].copy()
                    stats['normal_teacher_rows']+=int(normal)
                    stats['normal_teacher_guided_rows']+=int(normal and weight>0)
                    stats['full_native_teacher_restored_rows']+=int(normal and weight>0 and
                        expert.get('full_native_support_attempt',{}).get('applied',False))
                    trusted_counts[:]+=(record['trusted']&normal)
                    pending[i]=(record,infos[j],label,network['DN_mean'][j])
                for target,source in zip(self.state.tensors(),following.tensors()):target.index_copy_(0,idx,source)
                batch_counts[len(indices)]=batch_counts.get(len(indices),0)+1
                self.vector.submit_at(indices,'step',[int(x) for x in actions])

        def report():
            return dict(phase='student_full_story_collection',rollout_decisions=int(counts.sum()),
                target=total,total_student_decisions=self.actions,training_games=self.completed,
                **stats,trusted_normal_rows_4_12_24=trusted_counts.tolist(),
                stage_coverage=coverage.summary(),job_stage_coverage=self.coverage.summary(),
                max_stage=max(x['stage'] for x in self.infos),elapsed_seconds=time.monotonic()-begun,
                scheduling='ready_first_global_blocks',behavior_policy_version=self.policy_version,
                world_decisions=counts.tolist(),policy_batch_size_histogram=dict(batch_counts),
                worlds=[dict(world=i,stage=info['stage'],episode_frames=info['episode_frames'],
                    lives=info.get('lives'),bombs=info.get('bombs'),misses=info.get('misses'),
                    segment_decisions=int(counts[i]),inflight=i in self.vector._inflight)
                    for i,info in enumerate(self.infos)])

        dispatch(list(range(self.workers)))
        while self.vector._inflight:
            guard()
            ready=self.vector.receive_ready(max_wait_ms=self.ready_wait_ms,min_batch=self.ready_batch_size,check_stop=guard)
            next_indices=[]
            for message in ready:
                i=message['index'];observation=message['observation'];info=message['info']
                for key,value in observation.items():self.obs[key][i]=value
                self.infos[i]=info
                if message['operation']=='reset':
                    self.starts[i]=True
                    if getattr(model,'nbnm',None):
                        from .nbnm_prefix import burn_in
                        fresh=burn_in(model,info,model.initial_state(1))
                        for target,value in zip(self.state.tensors(),fresh.tensors()):target[i:i+1].copy_(value)
                    if counts[i]<limits[i]:next_indices.append(i)
                    continue
                record,before,label,pooled=pending.pop(i);action=int(record['actions'])
                done=bool(message['terminated'])
                if message['truncated']:raise RuntimeError('Only native full-game terminals may end a game')
                if info.get('reward_mapping_complete') is not True:raise RuntimeError('Incomplete native rewards')
                if int(info['policy_action'])!=action:raise RuntimeError('Native action differs from sampled student action')
                elapsed=int(info['episode_frames']-before['episode_frames'])
                used=int(info['bombs_used']-before['bombs_used'])
                hit=bool(info['misses']>before['misses'] or (observation['features'][17]>.5 and record['features'][17]<.5))
                # The observed transition is factual even if its first tick
                # was protected. Risk supervision selects exposed rows later.
                outcome=int(hit) if record['features'][14]>.5 and action<18 else -1
                record.update(native_action_frames=np.int32(elapsed),outcomes=np.int8(outcome),
                    recorded_observed_contacts=hit,reward_vectors=np.asarray(info['reward_vector'],np.float32),terminated=done,
                    lower_field_potential=np.float32(info.get('reward_components_raw',{}).get('lower_field_potential',0.)),
                    avoidable_upper_residence=np.float32(info.get('reward_components_raw',{}).get('avoidable_upper_residence',0.)))
                from .compact_native_contact_targets import native_exposure_mask
                if native_exposure_mask(record['features'],record['unprotected_mask']) and action<18:
                    stats['safe_forecast_but_native_contact']+=int(hit and 0<elapsed<=4 and record['margins'][0,action]>0)
                    invalidate_contradictory_risk({k:np.asarray(v)[None] for k,v in record.items()},0,hit,action,elapsed)
                partial,veto=veto_selected_clear(record['partial_point_danger'][None],np.array([action]),
                    np.array([outcome]),np.array([record['unprotected_mask']]),np.array([elapsed]))
                record['partial_point_danger']=partial[0];record['recorded_partial_point_veto']=veto[0]
                salient=selector is not None or record['deathbomb_mask'] or action==18 or outcome==1
                record['recorded_pooled_DN_mask']=bool(salient)
                record['recorded_pooled_DN']=pooled if salient else np.zeros_like(pooled)
                stats['actual_bombs']+=used;stats['Bomb_requests']+=int(action==18)
                stats['contacts']+=int(hit);stats['deathbombs']+=int(used>0 and record['deathbomb_mask'])
                if 'compact_contact_evidence' in info:
                    from .compact_story_contact_evidence import recorded_policy_context
                    append_jsonl(self.output/'native-contact-evidence.jsonl',dict(world=i,
                        training_student_decisions=self.actions+1,transition=info['compact_contact_evidence'],
                        before_uncertainty=label.get('uncertainty'),
                        policy_context=recorded_policy_context(records[i],record,before,
                            risk_scale=getattr(model,'neural_risk_scale',1.))))
                batched={k:np.asarray(v)[None] for k,v in record.items()}
                coverage.observe([before],[info],batched,[done],world_indices=[i])
                self.coverage.observe([before],[info],batched,[done],world_indices=[i])
                records[i].append(record);counts[i]+=1;self.actions+=1;self.starts[i]=done
                if fixed_quota and counts[i]==limits[i] and last_reserved is None:
                    last_reserved=time.monotonic()
                if (counts[i]%length==0 and reserved<blocks
                        and (remaining is None or remaining[groups[i]]>0)):
                    limits[i]+=length;reserved+=1
                    if remaining is not None:remaining[groups[i]]-=1
                    if reserved==blocks:last_reserved=time.monotonic()
                if done:
                    self.completed+=1
                    append_jsonl(self.output/'training-episodes.jsonl',dict(info['episode_summary'],
                        world=i,statistics_scope='changing-policy training game;not frozen evaluation'))
                    for value in self.state.tensors():value[i].zero_()
                    self.vector.submit_at([i],'reset')
                elif counts[i]<limits[i]:next_indices.append(i)
            dispatch(next_indices)
            if int(counts.sum())-last_report>=128:
                progress(report());last_report=int(counts.sum())
        if int(counts.sum())!=total or pending:raise RuntimeError('Incomplete global on-policy segment')
        bootstrap,_,_=model.step(torch.as_tensor(self.obs['features'],device=model.device),self.state)
        metadata=dict(stats,native_repair_revisions=native_repair_revisions(
            reactive_orbs=getattr(self,'native_wall_reactive_forecast',None),
            bullet_scale=getattr(self,'native_ex_scale_forecast',None)),teacher_actions=0,full_story=True,
            native_wall_reactive_forecast=getattr(self,'native_wall_reactive_forecast',None),
            native_ex_scale_forecast=getattr(self,'native_ex_scale_forecast',None),
            stage_coverage=coverage.summary(),job_stage_coverage=self.coverage.summary(),
            trusted_normal_rows_4_12_24=trusted_counts.tolist(),supervision_only=selector is not None,
            action_selection=supervision_selection or 'sample',behavior_policy_version=self.policy_version,
            movement_sampling_temperature=self.movement_sampling_temperature,bomb_sampling_temperature=self.bomb_sampling_temperature,
            partial_point_danger_revision=last_label.get('partial_point_danger_revision'),
            protected_exit_revision=last_label.get('protected_exit_revision'),
            native_position_hold_forecast=self.native_position_hold_forecast,
            native_hidden_enemy_contacts=self.native_hidden_enemy_contacts,native_ex8_polar4_forecast=self.native_ex8_polar4_forecast,
            policy_batch_size_histogram=dict(batch_counts),inflight_at_update_boundary=0,
            per_world_step_quota=fixed_quota,
            sampling_mode='ready-per-world-long-batch-v1' if fixed_quota else 'ready-global-blocks-v1',
            policy_updates_during_collection=0,game_or_brain_reset_at_update=False,
            elapsed_seconds=time.monotonic()-begun,
            final_block_drain_seconds=(time.monotonic()-last_reserved if last_reserved is not None else None))
        for key in ('protected_exit_forecast','native_contact_risk_calibration','native_motion_forecast','native_focus_gate',
                'unknown_laser_player_motion','native_curve_laser_geometry','native_curve_motion_forecast','native_enemy_damage_bound','primary_attack_target','forecast_growth_motion',
                'anticipate_point_arming','task_progress_ports'):
            metadata[key]=bool(getattr(model,key,False))
        metadata.update(protected_exit_from_stage=getattr(model,'protected_exit_from_stage',1),
            protected_exit_preserve_long_shield=getattr(model,'protected_exit_preserve_long_shield',False))
        progress(report())
        result=pack_world_records(records,states,bootstrap['values'].cpu(),metadata)
        if getattr(model,'nbnm',None):
            result['metadata'].update(full_story=False,nbnm_tasks=True,
                task_group_decisions={g:int(sum(counts[i] for i,k in enumerate(groups) if k==g))
                    for g in set(groups)} if groups is not None else {})
        if episode_initial_states:result['episode_initial_states']=episode_initial_states
        return result
