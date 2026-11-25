"""Ready-first full-story collection with a global real-decision budget.

One policy version collects each segment. Native worlds pause at acknowledged
boundaries for PPO; game positions and carried brain states survive updates.
"""
import time
import numpy as np
import torch

from ..compact_story_ppo import Collector,invalidate_contradictory_risk
from ..compact_story_behavior import story_behavior_logits
from ..compact_story_coverage import StoryCoverage
from ..compact_story_ready_data import pack_world_records
from ..spiking_engineered_control import SpikingState
from ..thrl_durable_records import append_jsonl


def _policy_cpu_packet(out, keys, logp, selected, behavior):
    """One synchronous GPU transfer for values and diagnostics of a ready batch."""
    size = len(logp)
    names = [key for key in keys if key in out]
    fields = [out['actions'].float(), logp, selected, behavior, *(out[key] for key in names)]
    blocks = [value.reshape(size, -1) for value in fields]
    packet = torch.cat(blocks, dim=1).cpu().numpy()
    splits = np.cumsum([part.shape[1] for part in blocks[:-1]])
    values = [part.reshape(tensor.shape) for part, tensor in zip(np.split(packet, splits, axis=1), fields)]
    actions, log_probs, selected, behavior, *telemetry = values
    return actions.astype(np.int64), log_probs, selected, behavior, dict(zip(names, telemetry))


class ReadyCollector(Collector):
    def __init__(self,*args,ready_batch_size=7,ready_wait_ms=1.,per_world_quota=False,**kwargs):
        if not 1<=ready_batch_size<=7 or not 0<=ready_wait_ms<=2:
            raise ValueError('Ready batching requires1..7 worlds and0..2ms coalescing')
        super().__init__(*args,**kwargs)
        self.ready_batch_size=int(ready_batch_size);self.ready_wait_ms=float(ready_wait_ms)
        self.per_world_quota=bool(per_world_quota)
        self.obs={k:v.copy() for k,v in self.obs.items()};self.policy_version=0
        if kwargs.get('continuation') is not None:
            from ..nbnm_continuation import restore_collector
            restore_collector(self,kwargs['continuation'],args[0])

    @torch.no_grad()
    def rollout(self,model,*,steps=128,length=16,guard=lambda:None,progress=lambda x:None,
                supervision_selection=None,total_decisions=None):
        from ..compact_native_forecast import native_repair_revisions
        from ..compact_partial_risk import veto_selected_clear
        total=steps*self.workers if total_decisions is None else total_decisions
        if length!=16 or total%256 or total<self.workers*length:
            raise ValueError('Global segments need full256-row PPO batches and at least16 decisions/world')
        if self.vector._inflight:raise RuntimeError('Previous collection did not drain its native commands')
        model.eval();self.policy_version+=1;begun=time.monotonic()
        adaptive=getattr(self,'adaptive_course',None)
        if adaptive is not None:adaptive.refresh()
        stage_course=getattr(self,'adaptive_stage_chain',None)
        if stage_course is not None:stage_course.refresh()
        coverage_weights=getattr(self,'coverage_group_weights',None)
        if adaptive is not None and coverage_weights:
            self.group_weights=dict(coverage_weights if any(not adaptive.history[k] for k in adaptive.keys)
                else self.default_group_weights)
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
            from ..nbnm_sampling import group_block_budget
            remaining=group_block_budget(groups,self.group_weights,blocks)
            for group in groups:remaining[group]-=1
        coverage=StoryCoverage(self.workers);last_report=0;last_reserved=None
        stats=dict(Bomb_requests=0,actual_bombs=0,contacts=0,deathbombs=0,
            normal_teacher_rows=0,normal_teacher_guided_rows=0,full_native_teacher_restored_rows=0,
            safe_forecast_but_native_contact=0)
        trusted_counts=np.zeros(3,np.int64)
        route_stats=dict(teacher_seconds=0.,route_queries=0,three_leg_queries=0,unguided_rows=0,
            atlas_available_rows=0,atlas_local_rows=0,atlas_landmark_rows=0,
            atlas_guided_rows=0,atlas_confidence_sum=0.,
            demonstration_available_rows=0,demonstration_guided_rows=0,demonstration_confidence_sum=0.,
            guided_choice_rows=0,teacher_top_matches_policy_top=0,teacher_top_probability_sum=0.,
            teacher_turn_rows=0,teacher_turn_mass_sum=0.,student_turn_mass_sum=0.,
            border_exit_rows=0,border_teacher_inward_mass_sum=0.,border_student_inward_mass_sum=0.)
        route_stats.update(by_stage={},by_phase={},abstention_reasons={},shorter_turn_queries=0,
            rolling_queries=0,rolling_available_rows=0,rolling_supervised_rows=0,
            rolling_seconds=0.,rolling_teacher_top_matches_policy_top=0,
            rolling_student_supported_mass_sum=0.)
        event_stats=dict(events=0,unweighted_progress=0.,weighted_reward=0.,
            by_kind=dict(road=0,nonspell=0,spell=0,stage_clear=0,power=0),by_stage={},by_task_mode={})
        shaping_totals = {}
        task_counts={}
        episode_initial_states={}
        suspended=set()
        retired_bootstrap={}
        discarded=0
        reset_attempts=0

        def note_reset(i,info):
            nonlocal reset_attempts
            attempts=info.get('native_start_retries',())
            if attempts:
                reset_attempts+=len(attempts)
                append_jsonl(self.output/'native-reset-recovery.jsonl',dict(
                    world=i,task=info.get('nb_nm_task_key'),policy_version=self.policy_version,
                    student_decisions=self.actions,available=not info.get('native_reset_unavailable'),
                    failures=attempts))

        def reserve_block(i):
            nonlocal reserved,last_reserved
            if reserved==blocks or (remaining is not None and remaining[groups[i]]<=0):
                return False
            limits[i]+=length;reserved+=1
            if remaining is not None:remaining[groups[i]]-=1
            if reserved==blocks:last_reserved=time.monotonic()
            return True

        def suspend(i):
            nonlocal reserved,discarded,last_reserved
            if fixed_quota:
                raise RuntimeError('Native restart recovery requires global ready scheduling')
            suspended.add(i)
            complete=int(counts[i])//length*length
            dropped=int(counts[i])-complete
            if dropped:
                retired_bootstrap[i]=records[i][complete]['old_values']
                del records[i][complete:]
                states[i].pop(complete,None)
                for key in [key for key in episode_initial_states if key[1]==i and key[0]>=complete]:
                    del episode_initial_states[key]
                counts[i]=complete
                self.actions-=dropped
                discarded+=dropped
            returned=(int(limits[i])-complete)//length
            limits[i]=complete
            reserved-=returned
            if returned:last_reserved=None
            if remaining is not None:
                group=groups[i]
                remaining[group]+=returned
                available=list(dict.fromkeys(g for j,g in enumerate(groups) if j not in suspended))
                if group not in available and available:
                    # A spell worker's reserved work stays with spells when possible.
                    replacements=[g for g in available if g.split('_')[0]==group.split('_')[0]] or available
                    remaining[replacements[0]]+=remaining[group]
                    remaining[group]=0
        objective_weights = None
        train_calm = train_lure = train_pace = train_continuation = False
        motion_stats = dict(eligible=0, idle=0, reversals=0, distance=0., cost=0., by_stage={})
        lure_stats = dict(card_rows=0, guided_rows=0, followed_rows=0, followed_contacts=0)
        pace_stats = dict(successes=0, reward_sum=0., damageable_target_rows=0, aimed_rows=0)
        if getattr(model, 'nbnm', None):
            from ..nbnm_objective import Objective
            objective = Objective(**model.nbnm.get('objective', {}))
            objective_weights = objective.weights
            train_calm = bool(objective.calm_cost_limit)
            corridor = model.nbnm.get('corridor_risk') or {}
            train_lure = bool(corridor.get('issun_guidance_weight'))
            train_continuation = model.nbnm.get('continuation_supervision', False)
            train_pace = bool(objective.completion_pace_weight)
        selector=None;last_label={}
        if supervision_selection is not None:
            from ..compact_story_action_selection import StoryActionSelector
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
                    if counts[i]%length==0:
                        saved_state=before.row(j).detach().to('cpu')
                        if getattr(model,'nbnm',None):
                            from ..nbnm_numerics import validate_saved_brain_state
                            validate_saved_brain_state(saved_state,world=i,decision=int(counts[i]))
                        states[i][int(counts[i])]=saved_state
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
                logp = distribution.log_prob(out['actions'])
                keys = ('values', 'logits', 'route_scores', 'horizon_scores', 'DN_mean',
                        'warning48_logits', 'corridor_logits', 'corridor_witness_logits', 'corridor_course_logits')
                actions, logp, selected, behavior_cpu, network = _policy_cpu_packet(
                    out, keys, logp, dn, behavior)
                warning48 = None
                if 'warning48_logits' in network:
                    warning48 = 1 / (1 + np.exp(-np.clip(network['warning48_logits'], -60, 60)))
                if getattr(model,'nbnm',None):
                    from ..nbnm_numerics import validate_network_arrays
                    validate_network_arrays(network,features=features,selected_DN=selected,log_probs=logp)
                # Let the original worker advance while the parent records
                # the already-computed action and its diagnostics.
                self.vector.submit_at(indices,'step',[int(x) for x in actions])
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
                        recorded_native_stage_frame=np.int64(infos[j].get('stage_frame',-1)),
                        recorded_native_episode=np.int64(label['native_episode']),
                        recorded_episode_frames=np.int64(infos[j]['episode_frames']),
                        recorded_teacher_Bomb_corrected=bool(expert.get('resource_reference_correction',{}).get('applied',False)),
                        recorded_neural_horizon_scores=network['horizon_scores'][j],
                        recorded_policy_logits=network['logits'][j],recorded_behavior_policy_logits=behavior_cpu[j],
                        recorded_route_scores=network['route_scores'][j],recorded_selected_DN=selected[j])
                    if warning48 is not None:
                        record['recorded_native_warning48'] = warning48[j]
                    if 'corridor_logits' in network:
                        record['recorded_corridor_risk_logits']=network['corridor_logits'][j]
                        courses=infos[j].get('corridor_courses')
                        record['recorded_corridor_course']=(np.asarray(courses[int(actions[j])],np.int8).copy()
                            if courses is not None and 0<=actions[j]<18 else np.full(24,-1,np.int8))
                    if (getattr(model, 'nbnm', None) or {}).get('timed_corridor_supervision'):
                        source = infos[j]
                        record['nbnm_verified_corridors'] = source.get('nbnm_verified_corridors', np.zeros((18, 2), bool))
                        record['nbnm_corridor_room'] = source.get('nbnm_corridor_room', np.zeros(18, np.float32))
                        record['nbnm_due_turns'] = source.get('nbnm_due_turns', np.zeros(18, bool))
                        record['nbnm_previous_heading'] = np.int8(source.get('nbnm_previous_heading', -1))
                    if 'corridor_witness_logits' in network:
                        record['recorded_corridor_witness_logits']=network['corridor_witness_logits'][j]
                    if 'corridor_course_logits' in network:
                        record['recorded_corridor_course_logits']=network['corridor_course_logits'][j]
                    record['recorded_pre_action_trusted']=record['trusted'].copy()
                    if getattr(model,'nbnm',None):
                        from ..nbnm_continuation_risk import enabled,targets
                        if enabled(model.nbnm):record.update(targets(label,expert))
                        record['objective_weights']=np.asarray(infos[j].get('nbnm_actor_weights',
                            objective_weights),np.float32)
                        route_stats['teacher_seconds']+=float(expert.get('query_seconds',0.))
                        route_stats['route_queries']+=int(expert.get('nb_nm_route',{}).get('route_search',False))
                        route_stats['three_leg_queries']+=int(expert.get('nb_nm_route',{}).get('three_leg_paths',0)>0)
                        route_stats['unguided_rows']+=int(weight<=0)
                        rolling=expert.get('nb_nm_route',{}).get('live_replanning') or {}
                        route_stats['rolling_queries']+=int(bool(rolling))
                        route_stats['rolling_available_rows']+=int(bool(rolling.get('available')))
                        route_stats['rolling_seconds']+=float(rolling.get('seconds',0.))
                        stage_key=str(infos[j]['stage'])
                        stage_stats=route_stats['by_stage'].setdefault(stage_key,dict(rows=0,guided=0,trusted4=0,trusted12=0,trusted24=0))
                        stage_stats['rows']+=1;stage_stats['guided']+=int(weight>0)
                        stage_stats['rolling_queries']=stage_stats.get('rolling_queries',0)+int(bool(rolling))
                        applied=bool(rolling.get('direct_student_supervision') and normal and weight>0)
                        stage_stats['rolling_guided_rows']=stage_stats.get('rolling_guided_rows',0)+int(applied)
                        phase=infos[j].get('nb_nm_phase') or 'unknown'
                        phase_stats=route_stats['by_phase'].setdefault(phase,dict(rows=0,guided_rows=0,
                            teacher_top_matches_policy_top=0,teacher_top_probability_sum=0.,
                            teacher_preferred_by_route_but_overridden=0,
                            border_exit_rows=0,border_teacher_inward_mass_sum=0.,border_student_inward_mass_sum=0.))
                        phase_stats['rows']+=1
                        phase_stats['rolling_queries']=phase_stats.get('rolling_queries',0)+int(bool(rolling))
                        phase_stats['rolling_guided_rows']=phase_stats.get('rolling_guided_rows',0)+int(applied)
                        for h,t in zip((4,12,24),label['trusted']):stage_stats['trusted'+str(h)]+=int(t)
                        route_stats['shorter_turn_queries']+=int(expert.get('nb_nm_route',{}).get('route_horizon_frames')==12)
                        if weight<=0:
                            reason=expert.get('guidance_weight_reason','unspecified')
                            route_stats['abstention_reasons'][reason]=route_stats['abstention_reasons'].get(reason,0)+1
                        if normal and weight>0:
                            teacher=np.asarray(expert['probabilities'][:18],float)
                            logits=network['logits'][j,:18].astype(float)
                            probability=np.exp(logits-logits.max());probability/=probability.sum()
                            top=int(teacher.argmax())
                            route_stats['guided_choice_rows']+=1
                            route_stats['teacher_top_matches_policy_top']+=int(top==probability.argmax())
                            route_stats['teacher_top_probability_sum']+=float(probability[top])
                            phase_stats['guided_rows']+=1
                            phase_stats['teacher_top_matches_policy_top']+=int(top==probability.argmax())
                            phase_stats['teacher_top_probability_sum']+=float(probability[top])
                            phase_stats['teacher_preferred_by_route_but_overridden']+=int(
                                top!=probability.argmax() and network['route_scores'][j,top]>
                                network['route_scores'][j,int(probability.argmax())])
                            if rolling.get('available') and rolling.get('direct_student_supervision'):
                                route_stats['rolling_supervised_rows']+=1
                                route_stats['rolling_teacher_top_matches_policy_top']+=int(top==probability.argmax())
                                route_stats['rolling_student_supported_mass_sum']+=float(probability[np.asarray(rolling['supported'],bool)].sum())
                            exits=np.asarray(expert.get('nb_nm_route',{}).get('exits',np.zeros(18)))
                            margins=record['margins'];safe=margins[0]>1
                            turns=safe&(exits>0)&(margins[2]<=0)
                            if turns.any() and teacher[turns].sum()>.05:
                                route_stats['teacher_turn_rows']+=1
                                route_stats['teacher_turn_mass_sum']+=float(teacher[turns].sum())
                                route_stats['student_turn_mass_sum']+=float(probability[turns].sum())
                            x,y=f[0]*368-184,f[1]*400+32
                            if abs(x)>160 or y>408:
                                from ..thrl_movement_teacher import action_velocities
                                from ..nbnm_live_routes import reserve
                                velocity=action_velocities().copy()
                                if f[11]<0:velocity[9:]=velocity[:9]
                                ends=np.clip(np.array([x,y])+4*velocity,[-184,32],[184,432])
                                inward=safe&(reserve(ends)-reserve([x,y])>.05)
                                if inward.any():
                                    route_stats['border_exit_rows']+=1
                                    route_stats['border_teacher_inward_mass_sum']+=float(teacher[inward].sum())
                                    route_stats['border_student_inward_mass_sum']+=float(probability[inward].sum())
                                    phase_stats['border_exit_rows']+=1
                                    phase_stats['border_teacher_inward_mass_sum']+=float(teacher[inward].sum())
                                    phase_stats['border_student_inward_mass_sum']+=float(probability[inward].sum())
                        demonstration=expert.get('nb_nm_route',{}).get('complete_demonstration') or {}
                        matched=bool(demonstration.get('available'))
                        route_stats['demonstration_available_rows']+=int(matched)
                        route_stats['demonstration_guided_rows']+=int(matched and weight>0)
                        route_stats['demonstration_confidence_sum']+=float(demonstration.get('confidence',0.))
                        prior=expert.get('nb_nm_route',{}).get('route_prior') or {}
                        route_stats['atlas_available_rows']+=int(prior.get('available',False))
                        route_stats['atlas_local_rows']+=int(prior.get('match_kind')=='local_continuation')
                        route_stats['atlas_landmark_rows']+=int(prior.get('match_kind')=='phase_landmark')
                        route_stats['atlas_guided_rows']+=int(prior.get('available',False) and weight>0)
                        route_stats['atlas_confidence_sum']+=float(prior.get('confidence',0.))
                        record['recorded_route_atlas_confidence']=np.float32(prior.get('confidence',0.))
                        record['recorded_route_atlas_horizon']=np.int16(prior.get('clean_horizon',0))
                        record['recorded_route_atlas_match']=np.int32(prior.get('matched_row',-1))
                        record['recorded_native_phase_clock']=np.int64(prior.get('clock') if prior.get('clock') is not None else -1)
                        key=infos[j].get('nb_nm_task_key')
                        if key:task_counts[key]=task_counts.get(key,0)+1
                    record['recorded_pre_action_partial_point_danger']=record['partial_point_danger'].copy()
                    stats['normal_teacher_rows']+=int(normal)
                    stats['normal_teacher_guided_rows']+=int(normal and weight>0)
                    stats['full_native_teacher_restored_rows']+=int(normal and weight>0 and
                        expert.get('full_native_support_attempt',{}).get('applied',False))
                    trusted_counts[:]+=(record['trusted']&normal)
                    pending[i]=(record,infos[j],label,network['DN_mean'][j])
                for target,source in zip(self.state.tensors(),following.tensors()):target.index_copy_(0,idx,source)
                batch_counts[len(indices)]=batch_counts.get(len(indices),0)+1

        def report():
            return dict(phase='student_full_story_collection',rollout_decisions=int(counts.sum()),
                target=total,total_student_decisions=self.actions,training_games=self.completed,
                **stats,trusted_normal_rows_4_12_24=trusted_counts.tolist(),
                stage_coverage=coverage.summary(),job_stage_coverage=self.coverage.summary(),
                max_stage=max(x['stage'] for x in self.infos),elapsed_seconds=time.monotonic()-begun,
                scheduling='ready_first_global_blocks',behavior_policy_version=self.policy_version,
                nbnm_route_guidance=dict(route_stats), nbnm_calm_movement=motion_stats,
                nbnm_issun_guidance=lure_stats, nbnm_pace=pace_stats,
                adaptive_course=adaptive.summary() if adaptive is not None else None,
                adaptive_stage_chain=stage_course.summary() if stage_course is not None else None,
                task_decisions=dict(task_counts),
                world_decisions=counts.tolist(),policy_batch_size_histogram=dict(batch_counts),
                native_reset_retries=reset_attempts,native_suspended_worlds=sorted(suspended),
                native_reset_discarded_rows=discarded,
                worlds=[dict(world=i,stage=info['stage'],episode_frames=info['episode_frames'],
                    task_native_frames=info.get('task_native_frames',0),
                    lives=info.get('lives'),bombs=info.get('bombs'),misses=info.get('misses'),
                    segment_decisions=int(counts[i]),inflight=i in self.vector._inflight,
                    task=info.get('nb_nm_task'),task_key=info.get('nb_nm_task_key'),task_group=groups[i] if groups else None)
                    for i,info in enumerate(self.infos)])

        unavailable=[i for i,info in enumerate(self.infos) if info.get('native_reset_unavailable')]
        if self.policy_version==1:
            for i,info in enumerate(self.infos):
                note_reset(i,info)
            for i in unavailable:
                suspend(i)
        elif unavailable:
            self.vector.submit_at(unavailable,'reset')
        dispatch([i for i in range(self.workers) if i not in unavailable])
        while self.vector._inflight:
            guard()
            ready=self.vector.receive_ready(max_wait_ms=self.ready_wait_ms,min_batch=self.ready_batch_size,check_stop=guard)
            next_indices=[]
            for message in ready:
                i=message['index'];observation=message['observation'];info=message['info']
                for key,value in observation.items():self.obs[key][i]=value
                self.infos[i]=info
                if message['operation']=='reset':
                    note_reset(i,info)
                    self.starts[i]=True
                    if info.get('native_reset_unavailable'):
                        suspend(i)
                        continue
                    if getattr(model,'nbnm',None):
                        from ..nbnm_prefix import burn_in
                        fresh=burn_in(model,info,model.initial_state(1))
                        for target,value in zip(self.state.tensors(),fresh.tensors()):target[i:i+1].copy_(value)
                    if counts[i]<limits[i]:next_indices.append(i)
                    continue
                record,before,label,pooled=pending.pop(i);action=int(record['actions'])
                forecast = before.get('native_branch_forecast')
                if forecast is not None:
                    stats['native_branch_forecast_rows'] = stats.get('native_branch_forecast_rows', 0) + 1
                    stats['native_branch_forecast_revision'] = forecast['revision']
                    stats['native_branch_forecast_matched_steps'] = (
                        stats.get('native_branch_forecast_matched_steps', 0)
                        + info['native_forecast_matched_decisions'] - before['native_forecast_matched_decisions'])
                    from ..nbnm_corridor import HORIZONS
                    contacts = np.asarray([frame if frame is not None else np.inf
                                           for frame in forecast['first_contact_by_action']])
                    record['native_branch_course_hits'] = contacts[:, None] <= np.asarray(HORIZONS[-2:])
                for event in info.get('nbnm_progress_events',()):
                    event_stats['events']+=1
                    event_stats['unweighted_progress']+=float(event['progress'])
                    event_stats['by_kind'][event['kind']]+=1
                    stage_key=str(event['stage']);mode=info['nb_nm_task']['mode']
                    event_stats['by_stage'][stage_key]=event_stats['by_stage'].get(stage_key,0)+1
                    event_stats['by_task_mode'][mode]=event_stats['by_task_mode'].get(mode,0)+1
                if 'native_nonspell_success' in info:
                    append_jsonl(self.output/'training-nonspell-successes.jsonl',
                        dict(info['native_nonspell_success'],world=i,student_decisions=self.actions+1))
                event_stats['weighted_reward']+=float(info.get('nbnm_event_reward',0.))
                if terms := info.get('nbnm_reward_shaping'):
                    for key, value in terms.items():
                        shaping_totals[key] = shaping_totals.get(key, 0.) + value
                if adaptive is not None:adaptive.observe_step(i,before.get('nb_nm_task_key'))
                if stage_course is not None:stage_course.observe_step(i,before.get('nb_nm_task_key'))
                done=bool(message['terminated'])
                if message['truncated']:raise RuntimeError('Only native full-game terminals may end a game')
                if info.get('reward_mapping_complete') is not True:raise RuntimeError('Incomplete native rewards')
                if int(info['policy_action'])!=action:raise RuntimeError('Native action differs from sampled student action')
                elapsed=int(info['episode_frames']-before['episode_frames'])
                used=int(info['bombs_used']-before['bombs_used'])
                hit=bool(info.get('policy_contact_event') or info['misses']>before['misses']
                         or (observation['features'][17]>.5 and record['features'][17]<.5))
                # The observed transition is factual even if its first tick
                # was protected. Risk supervision selects exposed rows later.
                outcome=int(hit) if record['features'][14]>.5 and action<18 else -1
                if train_calm:
                    quiet = info.get('nbnm_calm_movement', {})
                    eligible = bool(quiet.get('eligible') and not hit and not done)
                    record['nbnm_calm_action'] = np.bool_(eligible)
                    if train_continuation:
                        certified = np.zeros(18, dtype=bool)
                        if eligible:
                            certified[list(quiet['wait_actions'])] = True
                        record['nbnm_calm_moves'] = certified
                    if eligible:
                        stage = motion_stats['by_stage'].setdefault(str(before['stage']),
                            dict(eligible=0, idle=0, reversals=0, distance=0., cost=0.))
                        for counts_for_scope in (motion_stats, stage):
                            counts_for_scope['eligible'] += 1
                            counts_for_scope['idle'] += int(quiet['idle'])
                            counts_for_scope['reversals'] += int(quiet['reversal'])
                            counts_for_scope['distance'] += quiet['distance']
                            counts_for_scope['cost'] += quiet['cost']
                if train_lure:
                    lure = np.asarray(before.get('issun_lure_guidance', [False]*18), bool)
                    record['nbnm_lure_moves'] = lure
                    course = before.get('issun_lure_courses', {}).get(action)
                    trace = np.full(24, -1, np.int8)
                    if course is not None:
                        trace[:len(course)] = course
                    record['recorded_lure_course'] = trace
                    if before.get('nb_nm_phase') == 's6/BossCard5':
                        lure_stats['card_rows'] += 1
                        lure_stats['guided_rows'] += int(lure.any())
                        lure_stats['followed_rows'] += int(lure[action])
                        lure_stats['followed_contacts'] += int(lure[action] and hit)
                if train_pace:
                    f = record['features']
                    if f[14] > .5 and f[113] > .5 and f[117] > .5:
                        pace_stats['damageable_target_rows'] += 1
                        pace_stats['aimed_rows'] += int(abs(f[114]*368) <= 32 and f[115]*400 <= -64)
                    if done and info.get('nbnm_success'):
                        pace_stats['successes'] += 1
                        pace_stats['reward_sum'] += float(info['reward_vector'][1])
                record.update(native_action_frames=np.int32(elapsed),outcomes=np.int8(outcome),
                    recorded_observed_contacts=hit,reward_vectors=np.asarray(info['reward_vector'],np.float32),terminated=done,
                    lower_field_potential=np.float32(info.get('reward_components_raw',{}).get('lower_field_potential',0.)),
                    avoidable_upper_residence=np.float32(info.get('reward_components_raw',{}).get('avoidable_upper_residence',0.)))
                from ..compact_native_contact_targets import native_exposure_mask
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
                    from ..compact_story_contact_evidence import recorded_policy_context
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
                if counts[i]%length==0:
                    reserve_block(i)
                if done:
                    self.completed+=1
                    append_jsonl(self.output/'training-episodes.jsonl',dict(info['episode_summary'],
                        world=i,statistics_scope='changing-policy training game;not frozen evaluation'))
                    for value in self.state.tensors():value[i].zero_()
                    options=adaptive.finish(i,info['episode_summary']) if adaptive is not None else None
                    if options is None and stage_course is not None:options=stage_course.finish(i,info['episode_summary'])
                    if options is not None:
                        append_jsonl(self.output/'adaptive-assignments.jsonl',dict(world=i,
                            previous_task=before.get('nb_nm_task_key'),next_task=options['task_key'],
                            previous_success=bool(info['episode_summary']['nbnm_success']),
                            policy_version=self.policy_version,student_decisions=self.actions))
                    self.vector.submit_at([i],'reset',options=[options] if options is not None else None)
                elif counts[i]<limits[i]:next_indices.append(i)
            if suspended:
                for i in range(self.workers):
                    if (i not in suspended and i not in self.vector._inflight and i not in next_indices
                            and counts[i]==limits[i] and reserve_block(i)):
                        next_indices.append(i)
            dispatch(next_indices)
            if int(counts.sum())-last_report>=128:
                progress(report());last_report=int(counts.sum())
        if int(counts.sum())!=total or pending:raise RuntimeError('Incomplete global on-policy segment')
        bootstrap,_,_=model.step(torch.as_tensor(self.obs['features'],device=model.device),self.state)
        if retired_bootstrap:
            values=bootstrap['values'].clone()
            for i,old in retired_bootstrap.items():
                values[i]=torch.as_tensor(old,device=values.device,dtype=values.dtype)
            bootstrap['values']=values
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
            native_reset_retries=reset_attempts,native_suspended_worlds=sorted(suspended),
            native_reset_discarded_rows=discarded,
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
        if adaptive is not None:metadata['adaptive_course']=adaptive.summary()
        if stage_course is not None:metadata['adaptive_stage_chain']=stage_course.summary()
        result=pack_world_records(records,states,bootstrap['values'].cpu(),metadata)
        if getattr(model,'nbnm',None):
            result['metadata'].update(full_story=False,nbnm_tasks=True,
                task_decisions=dict(task_counts),
                nbnm_route_guidance=dict(route_stats),
                nbnm_event_progress=event_stats, nbnm_calm_movement=motion_stats,
                nbnm_issun_guidance=lure_stats, nbnm_pace=pace_stats,
                task_group_decisions={g:int(sum(counts[i] for i,k in enumerate(groups) if k==g))
                    for g in set(groups)} if groups is not None else {})
        if shaping_totals:
            result['metadata']['nbnm_reward_shaping'] = shaping_totals
        if episode_initial_states:result['episode_initial_states']=episode_initial_states
        return result
