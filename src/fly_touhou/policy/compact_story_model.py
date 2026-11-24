"""Warm-started whole-connectome LIF actor with resources,Bomb and three values.

Policy and risk read only DN activity after the shared VP input. NBNM can
condition the artificial value heads on task scope separately. The18 candidates share one brain/weights;
only the actual movement (neutral0 for Bomb) commits a following state.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from ..candidate_spiking_model import candidates,build_compact_model,restore_parameters
from ..spiking_engineered_control import SpikingState
from ..compact_story_sensation import SCHEMA as INPUT_SCHEMA

SCHEMA='full-connectome-compact129-spiking-story-PPO-v1'


class ContextPortDrive(nn.Module):
    def __init__(self,base):
        super().__init__();self.base=base
        self.context=nn.Linear(39,base.residual.out_features,bias=False)
        nn.init.zeros_(self.context.weight)
        self.task_progress=None
        self.extended_context=None
        self.route_prior_context=None
        self.future_rule_context=None
        self.corridor_context=None
        self.course_port_gain=0.
        self.batch_invariant=False
    @property
    def residual(self):return self.base.residual
    def forward(self,x):
        if self.batch_invariant:
            # GEMM accumulation can change with ready-batch width. Tiny port
            # differences can cross an LIF threshold and diverge over time.
            # Pad only this small encoder, never the full 138k-neuron core.
            rows=[]
            for first in range(0,len(x),64):
                part=x[first:first+64];count=len(part)
                padded=F.pad(part,(0,0,0,64-count))
                rows.append(self._drive(padded)[:count])
            return torch.cat(rows)
        return self._drive(x)
    def _drive(self,x):
        drive=self.base(x[:,:8])+self.context(x[:,8:47])
        if self.extended_context is not None:
            end=47+self.extended_context.in_features
            drive=drive+self.extended_context(x[:,47:end])
            if self.route_prior_context is not None:
                width=self.route_prior_context.in_features
                drive=drive+self.route_prior_context(x[:,end:end+width]);end+=width
            if self.future_rule_context is not None:
                drive=drive+self.future_rule_context(x[:,end:end+self.future_rule_context.in_features])
            if self.corridor_context is not None:
                corridor=x[:,end:end+self.corridor_context.in_features]
                drive=drive+self.corridor_context(corridor)
                if self.course_port_gain:
                    # The categorical "previous suffix revalidated" flag is
                    # already one of the seven measurements per candidate.
                    # Use distinct VP stimulation instead of treating .75 vs 1
                    # as a monotone safety score.
                    active=(corridor[:,3]==.75) & (corridor[:,2]>.5) & (corridor[:,6]>0)
                    scope=self.course_port_stages
                    if scope is not None:
                        stage=(x[:,26]*6).round()
                        applicable=torch.zeros_like(active)
                        for s in scope:applicable=applicable | (stage==s)
                        active=active & applicable
                    drive=drive+self.course_port_gain*active[:,None]*self.course_port_population
        if self.task_progress is not None:drive=drive+self.task_progress(x)
        return F.relu(drive)


@torch.no_grad()
def prime_critical_event_ports(model,*,amplitude=.25):
    """Salient observed-state4 input on quiet opponent VP populations.

    Only an existing sensory weight column changes. For the same carried
    brain state, inputs with state4=0 are identical. There is no action
    override,DN stimulation,new connection or frozen sensory parameter.
    """
    if not 0<amplitude<=1:raise ValueError('Positive bounded sensory initialization amplitude required')
    encoder=model.core.input_encoder;base=encoder.base
    # Cue indices3..5 encode confidence. Their negative opponent populations
    # are quiet when the forecasts are known, leaving a clear event contrast.
    mask=(base.feature>=3)&(base.feature<=5)&(base.polarity<0)
    if not bool(mask.any()):raise ValueError('Observed opponent VP population required')
    before=encoder.context.weight[:,17].clone()
    encoder.context.weight[:,17].add_(mask.to(before.dtype)*amplitude)
    return dict(observed_feature='native state4',context_column=17,VP_ports=int(mask.sum()),amplitude=amplitude,
        previous_column_abs_max=float(before.abs().max()),other_sensory_columns_changed=False,
        normal_observation_direct_drive_changed=False,policy_action_override=False,
        trainable_after_initialization=True)


def story_candidates(features):
    from ..device_constants import movement_velocities
    cue=candidates(features[...,:111]);shape=features.shape[:-1]
    context=torch.cat((features[...,:18],features[...,111:129]),-1)
    context=context.unsqueeze(-2).expand(*shape,18,36)
    velocities=movement_velocities(features)
    fast=velocities[torch.arange(18,device=features.device)%9]
    velocities=torch.where((features[...,11]<0)[...,None,None],fast,velocities)
    focus=(torch.arange(18,device=features.device)>=9).to(features.dtype)[:,None].expand(*shape,18,1)
    action=torch.cat((velocities,focus),-1).expand(*shape,18,3)
    return torch.cat((cue,context,action),-1)


class CompactStoryPolicy(nn.Module):
    candidate_conditioned=True
    input_schema=INPUT_SCHEMA
    def __init__(self,base,*,temperature=.05):
        super().__init__();self.core=base.core;self.risk=base.risk;self.critic=base.critic;self.seed=base.seed
        self.core.input_encoder=ContextPortDrive(self.core.input_encoder).to(self.core.device)
        self.core.sensory_width=47;self.temperature=temperature;self.horizon_weights=(8.,4.,2.);self.stay_prior=.05
        self.pad_candidate_batch=True;self.phase_valid_control=False;self.soft_risk_composition=False
        self.protection_aware_risk=False
        self.anticipate_point_arming=False
        self.forecast_growth_motion=False
        self.native_motion_forecast=False
        self.native_focus_gate=False
        self.unknown_laser_player_motion=False
        self.native_curve_laser_geometry=False
        self.native_curve_motion_forecast=False
        self.native_enemy_damage_bound=False
        self.task_progress_ports=False
        self.protected_exit_forecast=False
        self.protection_expiry_inclusive=False
        self.primary_attack_target=False
        self.risk_policy_gradient=False
        self.native_contact_risk_calibration=False
        self.neural_risk_scale=1.
        self.idle_prior_scale=1.
        self.combat_position_guidance_coefficient=0.
        self.lower_field_default=False
        self.full_story_action_selection='joint_mode'
        self.shared_directions=None
        self.native_attack_readout = None
        width=2*len(self.core.output_indices)
        self.route=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,128),nn.SiLU(),nn.Linear(128,1))
        self.bomb=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,128),nn.SiLU(),nn.Linear(128,1))
        self.bomb_urgency_gain=nn.Parameter(torch.tensor(12.))
        nn.init.zeros_(self.route[-1].weight);nn.init.zeros_(self.route[-1].bias)
        nn.init.zeros_(self.bomb[-1].weight);nn.init.constant_(self.bomb[-1].bias,-8.)
        self.bombs_disabled=False
        self.to(self.core.device)
    @property
    def device(self):return self.core.device
    def initial_state(self,batch):return self.core.initial_state(batch)
    def candidate_features(self,features):
        base=story_candidates(features)
        if getattr(self,'nbnm',None):
            from ..nbnm_sensation import candidate_extension
            extension=candidate_extension(features,route_prior=bool(self.nbnm.get('route_prior',{}).get('enabled')),
                future_rules=bool(self.nbnm.get('future_rules',{}).get('enabled')),corridor=bool(self.nbnm.get('corridor_risk')))
            from ..th08_attack_readout import CRITIC_TASK_CONTEXTS
            if self.nbnm.get('task_context') in CRITIC_TASK_CONTEXTS:
                extension=torch.cat((extension[...,:12],torch.zeros_like(extension[...,12:16]),extension[...,16:]), -1)
            base=torch.cat((base,extension),-1)
        return base
    def enable_task_progress_ports(self,*,amplitude=.025):
        from ..compact_story_task_drive import TaskProgressPortDrive
        encoder=self.core.input_encoder
        if encoder.task_progress is None:
            encoder.task_progress=TaskProgressPortDrive(encoder.base.feature,encoder.base.polarity,
                initial_amplitude=amplitude).to(self.device)
        self.task_progress_ports=True
        return encoder.task_progress.specification()
    def enable_shared_directions(self):
        from ..compact_shared_directions import SharedDirections
        if not getattr(self,'nbnm',None) or self.core.sensory_width < 63:
            raise ValueError('Single-state directions require configured NBNM sensory cues')
        if self.shared_directions is None:
            self.shared_directions=SharedDirections(self.core.sensory_width,len(self.core.input_indices),
                2*len(self.core.output_indices),device=self.device)

    def advance_shared(self,cue,directional,state,**core_options):
        return self.core(cue,state,port_delta=self.shared_directions.port_delta(directional),**core_options)

    def heads(self,DN,features,*,pooled_DN=None):
        n=len(features);all_DN=DN.reshape(n,18,-1);scores=self.risk(all_DN)
        probability=(8*(scores if self.risk_policy_gradient else scores.detach())-1).sigmoid()
        weights=getattr(self,'movement_risk_horizon_weights',(8.,4.,2.))
        risk_cost=-sum(w*probability[...,h] for h,w in enumerate(weights))
        controllable=(features[:,14]>.5)|(features[:,17]>.5)
        vulnerable=(features[:,14]>.5)&(features[:,5]<=0)&(features[:,13]<=0)
        if self.phase_valid_control:
            if self.protection_aware_risk:
                # A four-tick action can outlast the currently observed shield.
                # Gradually restore the existing neural4/12/24 risk costs as
                # that counter expires. Future Bomb-effect refresh is unknown;
                # this is a finite conservative preference,not action legality.
                u=features[:,13];remaining=60*u/(1-u).clamp_min(1e-6)
                from ..device_constants import constant_like
                horizon=constant_like(features,(4.,12.,24.))
                exposure=((horizon-remaining[:,None])/horizon).clamp(0,1)
                exposure=exposure*(features[:,14]>.5)[:,None]
                risk_cost=-sum(w*probability[...,h]*exposure[:,None,h] for h,w in enumerate(weights))
            else:risk_cost=risk_cost*vulnerable[:,None]
        if self.native_attack_readout is None:
            route = self.route(all_DN)[..., 0]
        else:
            from ..th08_attack_readout import ATTACK_AGE_COLUMN, OBSERVATION_COLUMN
            route_input = self.route[0](all_DN)
            route = (self.route[1:](route_input)[..., 0]
                     + self.native_attack_readout(route_input, features[:, OBSERVATION_COLUMN],
                                                 features[:, ATTACK_AGE_COLUMN]))
        if self.soft_risk_composition:
            # A learned risk forecast is a finite logit penalty, not action
            # legality. Do not amplify it by1/temperature or clamp alternatives
            # into zero-gradient scores. Wrong high-confidence choices must
            # still receive useful PPO/safe/future/rule gradients.
            movement=route/self.temperature+risk_cost
            movement=movement.clone();movement[:,0]+=self.stay_prior/self.temperature
            movement=movement-movement.max(-1,keepdim=True).values
        else:
            movement=risk_cost+route;movement=movement.clone();movement[:,0]+=self.stay_prior
            safe=probability[...,0]<.2;blocked=safe.any(-1,keepdim=True)&~safe
            if self.phase_valid_control:blocked=blocked&vulnerable[:,None]
            movement=movement.masked_fill(blocked,-80.)
            movement=((movement-movement.max(-1,keepdim=True).values)/self.temperature).clamp(-80,0)
        pooled=all_DN.mean(1) if pooled_DN is None else pooled_DN
        # A trainable initialization prior from NEURAL risk outputs, never a
        # raw-geometry shield or forced teacher action. Safe movement remains
        # the default; urgent Bomb can be sampled and learned from the outset.
        urgency=((probability[...,0].min(-1).values-.2)/.05).sigmoid()
        if self.phase_valid_control:urgency=urgency*(vulnerable|(features[:,17]>.5))
        bomb=(self.bomb(pooled)+self.bomb_urgency_gain*urgency[:,None]).clamp(-80,80)
        legal_bomb=(features[:,4]>0)&((features[:,14]>.5)|(features[:,17]>.5))
        if self.bombs_disabled:legal_bomb=torch.zeros_like(legal_bomb)
        logits=torch.cat((movement,bomb.masked_fill(~legal_bomb[:,None],-torch.inf)),-1)
        if self.phase_valid_control:
            neutral=torch.full_like(logits,-torch.inf);neutral[:,0]=0.
            logits=torch.where(controllable[:,None],logits,neutral)
        if self.neural_risk_scale!=1.:
            if not self.soft_risk_composition:raise ValueError('Neural risk scaling requires finite-risk composition')
            from ..compact_story_risk_composition import scale_neural_movement_risk
            logits=scale_neural_movement_risk(logits,risk_cost,self.neural_risk_scale)
        if self.idle_prior_scale!=1.:
            if not self.soft_risk_composition:raise ValueError('Idle-prior sensitivity requires finite-risk composition')
            from ..compact_story_idle_prior import scale_idle_prior
            logits=scale_idle_prior(logits,self.stay_prior/self.temperature,self.idle_prior_scale)
        if getattr(self,'protection_expiry_inclusive',False):
            if not (self.phase_valid_control and self.protection_aware_risk and self.soft_risk_composition):
                raise ValueError('Inclusive expiry requires finite protection-aware risk')
            from ..compact_story_risk_composition import include_protection_expiry_tick
            logits=include_protection_expiry_tick(logits,probability,features,risk_scale=self.neural_risk_scale)
        if self.protected_exit_forecast or getattr(self,'frozen_decoding',None) is not None:
            from ..compact_frozen_decoding import reweight
            logits=reweight(logits,probability,route,features,temperature=self.temperature,
                risk_scale=self.neural_risk_scale,inclusive_expiry=getattr(self,'protection_expiry_inclusive',False),
                options=getattr(self,'frozen_decoding',None),protected_exit=self.protected_exit_forecast,
                protected_from_stage=getattr(self,'protected_exit_from_stage',1))
        warning48 = None
        if getattr(self, 'warning48_head', None) is not None:
            warning48 = self.warning48_head(probability, route, all_DN)
            weight = self.nbnm['native_warning48'].get('actor_weight', 0.)
            if weight:
                cost = weight * warning48.detach().sigmoid() * vulnerable[:, None]
                logits = torch.cat((logits[:, :18] - cost, logits[:, 18:]), -1)
        corridor=witness=course=endpoint_cost=None
        if getattr(self,'corridor_head',None) is not None:
            from ..nbnm_corridor_loss import risk_readout,compose
            route_and_risk = logits[..., :18]
            corridor=risk_readout(self.corridor_head,all_DN)
            logits=compose(logits,corridor,features,vulnerable,self.nbnm['corridor_risk']['actor_weight'])
            if getattr(self,'corridor_witness_head',None) is not None:
                from ..nbnm_corridor_loss import compose_witness
                witness=risk_readout(self.corridor_witness_head,all_DN)
                logits=compose_witness(logits,witness,features,vulnerable,self.nbnm['corridor_risk']['witness_weight'])
            if getattr(self,'corridor_course_head',None) is not None:
                from ..nbnm_corridor_loss import compose_course
                course=risk_readout(self.corridor_course_head,all_DN).squeeze(-1)
                logits=compose_course(logits,course,features,vulnerable,self.nbnm['corridor_risk'])
            if getattr(self,'corridor_endpoint_head',None) is not None:
                from ..nbnm_corridor_loss import endpoint_adjustment
                endpoint=risk_readout(self.corridor_endpoint_head,all_DN)
                correction=endpoint_adjustment(endpoint,features,self.nbnm['corridor_risk']['endpoint_risk'])
                observed=features[...,281:].reshape(*features.shape[:-1],18,7)[...,6].amax(-1)>0
                endpoint_cost=correction*(vulnerable&observed)[...,None]
                logits=torch.cat((logits[...,:18]-endpoint_cost,logits[...,18:]),-1)
            priority = self.nbnm["corridor_risk"].get("short_risk_priority")
            if priority is not None:
                from ..nbnm_corridor_loss import prioritize_short_risk
                logits = prioritize_short_risk(logits, route_and_risk, probability[..., 0],
                                               vulnerable, priority)
        # Mean over every counterfactual is independent of the action sampled.
        value_input=pooled.detach()+.1*(pooled-pooled.detach())
        if getattr(self,'task_critic',None) is not None:
            critic_scope = features[:, 277:281]
            if self.native_attack_readout is not None:
                # TH08's original critic scope has no inside-goal flag on this column.
                critic_scope = torch.cat((critic_scope[:, :3], torch.zeros_like(critic_scope[:, 3:])), dim=1)
            value_input = value_input + self.task_critic(critic_scope)
        values=self.critic(value_input)
        result = dict(logits=logits,values=values,horizon_scores=scores,
            predicted_margin=1-8*scores[...,0],DN_mean=pooled,legal_bomb=legal_bomb,route_scores=route)
        if warning48 is not None:
            result['warning48_logits'] = warning48
        if corridor is not None:
            result['corridor_logits']=corridor
        if witness is not None:
            result['corridor_witness_logits']=witness
        if course is not None:
            result['corridor_course_logits']=course
        if endpoint_cost is not None:
            result['corridor_endpoint_cost']=endpoint_cost
        return result
    def step(self,features,state,*,actions=None,checkpointed=False,sample=False,return_all_DN=False,
             action_selector=None,previous_actions=None,corridor_exit_bias=None):
        if action_selector is not None and (actions is not None or sample):
            raise ValueError('Choose one action source before committing the recurrent state')
        if self.shared_directions is None:
            all_DN,following=self._candidate_brains(features,state,checkpointed)
            actual_DN=None
            out=self.heads(all_DN,features)
        else:
            cue,directional=self.shared_directions.inputs(self.candidate_features(features))
            actual_DN,following=self.advance_shared(cue,directional,state)
            all_DN=self.shared_directions.directions(actual_DN)
            out=self.heads(all_DN,features,pooled_DN=actual_DN)
        chosen=self._choose(out,features,actions=actions,sample=sample,action_selector=action_selector,
            previous_actions=previous_actions,corridor_exit_bias=corridor_exit_bias)
        if actual_DN is not None:
            if return_all_DN:
                out['candidate_DN']=actual_DN[:,None,:].expand(-1,18,-1)
            return out,following,actual_DN
        if return_all_DN:out['candidate_DN']=all_DN.reshape(len(features),18,-1)
        # Bomb holds the same fast-idle movement as source0.
        index=torch.arange(len(features),device=self.device)*18+torch.where(chosen==18,0,chosen)
        after=SpikingState(*(v.index_select(0,index) for v in following.tensors()))
        return out,after,all_DN.index_select(0,index)

    def _candidate_brains(self,features,state,checkpointed):
        width=self.core.sensory_width
        cue=self.candidate_features(features).reshape(-1,width)
        repeated=SpikingState(*(v.repeat_interleave(18,0) for v in state.tensors()))
        DNs=[];following=[]
        for first in range(0,len(cue),128):
            count=min(128,len(cue)-first);x=cue[first:first+count]
            before=SpikingState(*(v[first:first+count] for v in repeated.tensors()))
            if count<128 and self.pad_candidate_batch:
                zero=self.initial_state(128-count);x=torch.cat((x,x.new_zeros(128-count,width)))
                before=SpikingState(*(torch.cat((v,z)) for v,z in zip(before.tensors(),zero.tensors())))
            def call(x,*parts):
                DN,after=self.core(x,SpikingState(*parts));return DN,*after.tensors()
            result=checkpoint(call,x,*before.tensors(),use_reentrant=False) if checkpointed and torch.is_grad_enabled() else call(x,*before.tensors())
            DN,*s=result;DNs.append(DN[:count]);following.append(SpikingState(*(v[:count] for v in s)))
        if len(DNs)==1:return DNs[0],following[0]
        return torch.cat(DNs),SpikingState.pack(following)

    def _choose(self,out,features,*,actions,sample,action_selector,previous_actions,corridor_exit_bias=None):
        idle_bonus=getattr(self,'frozen_safe_idle_bonus',0.)
        if idle_bonus:
            from ..compact_story_idle_prior import prefer_settled_idle,prefer_route_aware_wait,settled_safe24
            out['unadjusted_action']=out['logits'].argmax(-1)
            if getattr(self,'frozen_route_aware_wait',False):
                out['settled_safe24']=settled_safe24(features)
                out['logits'],out['wait_eligible'],out['wait_action']=prefer_route_aware_wait(out,features,idle_bonus)
            else:
                out['logits'],out['settled_safe24']=prefer_settled_idle(out['logits'],features,idle_bonus)
        if previous_actions is not None:
            from ..compact_story_idle_prior import prefer_safe_heading,settled_safe24
            if not idle_bonus:
                out['unadjusted_action']=out['logits'].argmax(-1)
                out['settled_safe24']=settled_safe24(features)
            out['logits'],out['heading_eligible'],out['pre_heading_action']=prefer_safe_heading(
                out,features,previous_actions,self.frozen_heading_steps,self.frozen_safe_heading_bonus)
            out['previous_executed_action']=previous_actions
        if getattr(self,'frozen_wall_escape',False):
            from ..compact_story_idle_prior import prefer_wall_escape,settled_safe24
            out['pre_wall_logits']=out['logits']
            out['logits'],out['wall_escape_rule'],out['pre_wall_action']=prefer_wall_escape(out['logits'],features)
            if 'unadjusted_action' not in out:
                out['unadjusted_action']=out['pre_wall_action']
                out['settled_safe24']=settled_safe24(features)
        if corridor_exit_bias is not None:
            logits = out['logits']
            out['pre_corridor_exit_action'] = logits.argmax(-1)
            out['logits'] = torch.cat((logits[:, :18] + corridor_exit_bias, logits[:, 18:]), dim=-1)
        if action_selector is not None:chosen=action_selector(out['logits'])
        else:chosen=(torch.distributions.Categorical(logits=out['logits'],validate_args=False).sample() if sample else out['logits'].argmax(-1)) if actions is None else actions
        out['actions']=chosen
        return chosen
    @torch.no_grad()
    def infer_candidates(self,features,states,*,committed_actions=None,return_all_DN=False,action_selector=None):
        f=torch.as_tensor(features,device=self.device,dtype=torch.float32)
        out,after,DN=self.step(f,SpikingState.pack(states),actions=committed_actions,return_all_DN=return_all_DN,action_selector=action_selector)
        return out,[after.row(i) for i in range(len(features))],DN
    def parameter_payload(self,costs):
        from ..compact_shared_directions import REVISION as SHARED_REVISION
        from ..compact_protected_exit_candidate import REVISION
        from ..compact_native_contact_targets import REVISION as NATIVE_REVISION
        return dict(schema=SCHEMA,input_schema=self.input_schema,seed=self.seed,costs=dict(costs),temperature=self.temperature,
            state_dict={k:v.detach().cpu().clone() for k,v in self.state_dict().items()},actual_binary_spikes=True,
            nbnm=getattr(self,'nbnm',None),
            critic_task_scope_outside_brain=getattr(self,'task_critic',None) is not None,
            one_shared_brain=True,all_source_nodes_and_signed_edges_retained=True,neural_input_per_candidate=self.core.sensory_width,
            brain_advance=SHARED_REVISION if self.shared_directions is not None else None,
            raw_geometry_or_teacher_action_bypass=False,map_in_policy=False,bombs_disabled=self.bombs_disabled,
            pad_candidate_batch=self.pad_candidate_batch,
            batch_invariant_sensory=self.core.input_encoder.batch_invariant,
            phase_valid_control=self.phase_valid_control,
            protection_aware_risk=self.protection_aware_risk,
            protection_expiry_inclusive=getattr(self,'protection_expiry_inclusive',False),
            anticipate_point_arming=self.anticipate_point_arming,
            forecast_growth_motion=getattr(self,'forecast_growth_motion',False),
            native_motion_forecast=self.native_motion_forecast,
            native_focus_gate=self.native_focus_gate,
            unknown_laser_player_motion=self.unknown_laser_player_motion,
            native_curve_laser_geometry=self.native_curve_laser_geometry,
            native_curve_motion_forecast=getattr(self,'native_curve_motion_forecast',False),
            native_enemy_damage_bound=getattr(self,'native_enemy_damage_bound',False),
            task_progress_ports=self.task_progress_ports,
            task_progress_drive=self.core.input_encoder.task_progress.specification() if self.task_progress_ports else None,
            protected_exit_forecast=self.protected_exit_forecast,
            protected_exit_from_stage=getattr(self,'protected_exit_from_stage',1),
            protected_exit_preserve_long_shield=getattr(self,'protected_exit_preserve_long_shield',False),
            frozen_decoding=getattr(self,'frozen_decoding',None),
            protected_exit_revision=REVISION if self.protected_exit_forecast else None,
            primary_attack_target=self.primary_attack_target,
            risk_policy_gradient=self.risk_policy_gradient,
            native_contact_risk_calibration=self.native_contact_risk_calibration,
            native_contact_target_revision=NATIVE_REVISION if self.native_contact_risk_calibration else None,
            neural_risk_scale=self.neural_risk_scale,
            idle_prior_scale=self.idle_prior_scale,
            combat_position_guidance_coefficient=self.combat_position_guidance_coefficient,
            lower_field_default=self.lower_field_default,
            full_story_action_selection=self.full_story_action_selection,
            soft_risk_composition=self.soft_risk_composition,
            risk_readout_kind=getattr(self.risk,'readout_kind','DN-only-nonlinear-risk-v1'),
            risk_readout_width=getattr(self.risk,'width',128),
            risk_input_soft_clip=getattr(self.risk,'input_soft_clip',None),
            risk_base_input_soft_clip=getattr(self.risk,'base_input_soft_clip',None),
            trainable_risk_baseline=getattr(self.risk,'baseline_trainable',False),
            deterministic_spmm=self.core.sparse.deterministic_spmm,
            physical_phase_handling=('Observed protection expiry gradually restores neural4/12/24 risk costs;ignored-input phases use canonical neutral. No new action mask.'
                if self.protection_aware_risk else 'Observed invulnerability disables unprotected-risk cost;ignored-input phases use canonical neutral. Live movement/Bomb scores still read DN.') if self.phase_valid_control else 'legacy all-phase neural risk',
            critic_input=('actual single-step DN activity;three value heads' if self.shared_directions is not None
                else 'mean DN across all candidates before action selection;three value heads'))


def from_avoidance_checkpoint(connectome,payload,*,device='cuda:0'):
    base=restore_parameters(build_compact_model(connectome,seed=payload['seed'],device=device),payload)
    return CompactStoryPolicy(base)


def from_story_checkpoint(connectome,payload,*,device='cuda:0'):
    base=build_compact_model(connectome,seed=payload['seed'],device=device)
    # Build the same nonlinear DN head shape before restoring the complete state.
    from ..dn_residual_risk import load_head
    risk={k.removeprefix('risk.'):v for k,v in payload['state_dict'].items() if k.startswith('risk.')}
    base.risk=load_head(dict(schema=payload.get('risk_readout_kind','DN-only-nonlinear-risk-v1'),state_dict=risk,
        width=payload.get('risk_readout_width',128),
        input_soft_clip=payload.get('risk_input_soft_clip'),base_input_soft_clip=payload.get('risk_base_input_soft_clip'),
        baseline_trainable=payload.get('trainable_risk_baseline',False)),device=device)
    model=CompactStoryPolicy(base,temperature=payload['temperature'])
    if payload.get('task_progress_ports',False):
        model.enable_task_progress_ports(amplitude=payload['task_progress_drive']['initial_amplitude'])
    if payload.get('nbnm'):
        from ..nbnm_model import configure_model
        configure_model(model,payload['nbnm'],reset_critics=False)
    if model.native_attack_readout is not None:
        prefix = "native_attack_readout.specialists."
        suffix = ".minimum_age"
        for key, value in payload['state_dict'].items():
            if key.startswith(prefix) and key.endswith(suffix):
                context = int(key[len(prefix):-len(suffix)])
                model.native_attack_readout.add_specialist(context, float(value))
    if payload.get('brain_advance') is not None:
        from ..compact_shared_directions import REVISION as SHARED_REVISION
        if payload['brain_advance']!=SHARED_REVISION:raise ValueError('Unknown connectome advance mode')
        model.enable_shared_directions()
    model.load_state_dict(payload['state_dict'],strict=True)
    model.bombs_disabled=payload.get('bombs_disabled',False)
    model.pad_candidate_batch=payload.get('pad_candidate_batch',True)
    model.core.input_encoder.batch_invariant=payload.get('batch_invariant_sensory',False)
    model.phase_valid_control=payload.get('phase_valid_control',False)
    model.protection_aware_risk=payload.get('protection_aware_risk',False)
    model.protection_expiry_inclusive=payload.get('protection_expiry_inclusive',False)
    model.anticipate_point_arming=payload.get('anticipate_point_arming',False)
    model.forecast_growth_motion=payload.get('forecast_growth_motion',False)
    model.native_motion_forecast=payload.get('native_motion_forecast',False)
    model.native_focus_gate=payload.get('native_focus_gate',False)
    model.unknown_laser_player_motion=payload.get('unknown_laser_player_motion',False)
    model.native_curve_laser_geometry=payload.get('native_curve_laser_geometry',False)
    model.native_curve_motion_forecast=payload.get('native_curve_motion_forecast',False)
    model.native_enemy_damage_bound=payload.get('native_enemy_damage_bound',False)
    model.protected_exit_forecast=payload.get('protected_exit_forecast',False)
    model.protected_exit_from_stage=payload.get('protected_exit_from_stage',1)
    model.protected_exit_preserve_long_shield=payload.get('protected_exit_preserve_long_shield',False)
    model.frozen_decoding=payload.get('frozen_decoding')
    if model.protected_exit_forecast:
        from ..compact_protected_exit_candidate import REVISION
        if payload.get('protected_exit_revision')!=REVISION:
            raise ValueError('Unsupported protected exit observation semantics')
    model.primary_attack_target=payload.get('primary_attack_target',False)
    model.risk_policy_gradient=payload.get('risk_policy_gradient',False)
    model.native_contact_risk_calibration=payload.get('native_contact_risk_calibration',False)
    if model.native_contact_risk_calibration:
        from ..compact_native_contact_targets import REVISION as NATIVE_REVISION
        if payload.get('native_contact_target_revision')!=NATIVE_REVISION:
            raise ValueError('Unsupported selected native contact target semantics')
    from ..compact_story_risk_composition import checked_risk_scale
    model.neural_risk_scale=checked_risk_scale(payload.get('neural_risk_scale',1.))
    from ..compact_story_idle_prior import checked_idle_scale
    model.idle_prior_scale=checked_idle_scale(payload.get('idle_prior_scale',1.))
    model.combat_position_guidance_coefficient=payload.get('combat_position_guidance_coefficient',0.)
    model.lower_field_default=payload.get('lower_field_default',False)
    model.full_story_action_selection=payload.get('full_story_action_selection','joint_mode')
    model.soft_risk_composition=payload.get('soft_risk_composition',False)
    model.core.sparse.deterministic_spmm=payload.get('deterministic_spmm',model.core.sparse.deterministic_spmm)
    return model


def sequence(model,features,starts,state,actions,*,checkpointed=True,checkpoint_chunks=False,reset_states=None):
    if checkpointed and checkpoint_chunks and getattr(model,'shared_directions',None) is None:
        from ..compact_story_shared_checkpoint import sequence as shared_sequence
        return shared_sequence(model,features,starts,state,actions,reset_states=reset_states)
    return _sequence_nested(model,features,starts,state,actions,
        checkpointed=checkpointed,checkpoint_chunks=checkpoint_chunks,reset_states=reset_states)


def _sequence_nested(model,features,starts,state,actions,*,checkpointed=True,checkpoint_chunks=False,reset_states=None):
    """Checkpoint the whole scene so18 replicated brain states are not retained."""
    records=[]
    for t in range(len(features)):
        state=state.reset(starts[t])
        if reset_states:
            from ..nbnm_prefix import apply_resets
            state=apply_resets(state,reset_states.get(t))
        def call(f,a,*parts):
            out,after,_=model.step(f,SpikingState(*parts),actions=a,checkpointed=checkpoint_chunks)
            return (out['logits'],out['values'],out['horizon_scores'],out.get('warning48_logits',f.new_empty(0)),
                out.get('corridor_logits',f.new_empty(0)),out.get('corridor_witness_logits',f.new_empty(0)),
                out.get('corridor_course_logits',f.new_empty(0)),*after.tensors())
        result=checkpoint(call,features[t],actions[t],*state.tensors(),use_reentrant=False) if checkpointed else call(features[t],actions[t],*state.tensors())
        logits,values,scores,warning48,corridor,witness,course,*parts=result;state=SpikingState(*parts)
        records.append((logits,values,scores,warning48,corridor,witness,course))
    result = dict(logits=torch.stack([r[0] for r in records]),values=torch.stack([r[1] for r in records]),
        horizon_scores=torch.stack([r[2] for r in records]),state=state)
    if records[0][3].numel():
        result['warning48_logits'] = torch.stack([r[3] for r in records])
    if records[0][4].numel():
        result['corridor_logits'] = torch.stack([r[4] for r in records])
    if records[0][5].numel():
        result['corridor_witness_logits'] = torch.stack([r[5] for r in records])
    if records[0][6].numel():
        result['corridor_course_logits'] = torch.stack([r[6] for r in records])
    return result
