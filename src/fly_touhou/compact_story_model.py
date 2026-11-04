"""Warm-started whole-connectome LIF actor with resources,Bomb and three values.

All task information enters the same VP ports. Policy,risk and values read
only DN activity. The18 counterfactual candidates share one brain/weights;
only the actual movement (neutral0 for Bomb) commits a following state.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .candidate_spiking_model import candidates,build_compact_model,restore_parameters
from .spiking_engineered_control import SpikingState
from .compact_story_sensation import SCHEMA as INPUT_SCHEMA

SCHEMA='full-connectome-compact129-spiking-story-PPO-v1'


class ContextPortDrive(nn.Module):
    def __init__(self,base):
        super().__init__();self.base=base
        self.context=nn.Linear(39,base.residual.out_features,bias=False)
        nn.init.zeros_(self.context.weight)
        self.task_progress=None
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
        drive=self.base(x[:,:8])+self.context(x[:,8:])
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
    from .thrl_movement_teacher import MovementSafetyTeacher
    cue=candidates(features[...,:111]);shape=features.shape[:-1]
    context=torch.cat((features[...,:18],features[...,111:129]),-1)
    context=context.unsqueeze(-2).expand(*shape,18,36)
    velocities=features.new_tensor(MovementSafetyTeacher().velocities)/4.5
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
    def enable_task_progress_ports(self,*,amplitude=.025):
        from .compact_story_task_drive import TaskProgressPortDrive
        encoder=self.core.input_encoder
        if encoder.task_progress is None:
            encoder.task_progress=TaskProgressPortDrive(encoder.base.feature,encoder.base.polarity,
                initial_amplitude=amplitude).to(self.device)
        self.task_progress_ports=True
        return encoder.task_progress.specification()
    def heads(self,DN,features):
        n=len(features);all_DN=DN.reshape(n,18,-1);scores=self.risk(all_DN)
        probability=(8*(scores if self.risk_policy_gradient else scores.detach())-1).sigmoid()
        risk_cost=-8*probability[...,0]-4*probability[...,1]-2*probability[...,2]
        controllable=(features[:,14]>.5)|(features[:,17]>.5)
        vulnerable=(features[:,14]>.5)&(features[:,5]<=0)&(features[:,13]<=0)
        if self.phase_valid_control:
            if self.protection_aware_risk:
                # A four-tick action can outlast the currently observed shield.
                # Gradually restore the existing neural4/12/24 risk costs as
                # that counter expires. Future Bomb-effect refresh is unknown;
                # this is a finite conservative preference,not action legality.
                u=features[:,13];remaining=60*u/(1-u).clamp_min(1e-6)
                horizon=features.new_tensor((4.,12.,24.))
                exposure=((horizon-remaining[:,None])/horizon).clamp(0,1)
                exposure=exposure*(features[:,14]>.5)[:,None]
                risk_cost=(-8*probability[...,0]*exposure[:,None,0]
                    -4*probability[...,1]*exposure[:,None,1]-2*probability[...,2]*exposure[:,None,2])
            else:risk_cost=risk_cost*vulnerable[:,None]
        route=self.route(all_DN)[...,0]
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
        pooled=all_DN.mean(1)
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
            from .compact_story_risk_composition import scale_neural_movement_risk
            logits=scale_neural_movement_risk(logits,risk_cost,self.neural_risk_scale)
        if self.idle_prior_scale!=1.:
            if not self.soft_risk_composition:raise ValueError('Idle-prior sensitivity requires finite-risk composition')
            from .compact_story_idle_prior import scale_idle_prior
            logits=scale_idle_prior(logits,self.stay_prior/self.temperature,self.idle_prior_scale)
        if getattr(self,'protection_expiry_inclusive',False):
            if not (self.phase_valid_control and self.protection_aware_risk and self.soft_risk_composition):
                raise ValueError('Inclusive expiry requires finite protection-aware risk')
            from .compact_story_risk_composition import include_protection_expiry_tick
            logits=include_protection_expiry_tick(logits,probability,features,risk_scale=self.neural_risk_scale)
        if self.protected_exit_forecast or getattr(self,'frozen_decoding',None) is not None:
            from .compact_frozen_decoding import reweight
            logits=reweight(logits,probability,route,features,temperature=self.temperature,
                risk_scale=self.neural_risk_scale,inclusive_expiry=getattr(self,'protection_expiry_inclusive',False),
                options=getattr(self,'frozen_decoding',None),protected_exit=self.protected_exit_forecast,
                protected_from_stage=getattr(self,'protected_exit_from_stage',1))
        # Mean over every counterfactual is independent of the action sampled.
        value_input=pooled.detach()+.1*(pooled-pooled.detach())
        return dict(logits=logits,values=self.critic(value_input),horizon_scores=scores,
            predicted_margin=1-8*scores[...,0],DN_mean=pooled,legal_bomb=legal_bomb,route_scores=route)
    def step(self,features,state,*,actions=None,checkpointed=False,sample=False,return_all_DN=False,action_selector=None):
        if action_selector is not None and (actions is not None or sample):
            raise ValueError('Choose one action source before committing the recurrent state')
        cue=story_candidates(features).reshape(-1,47);n=len(features)
        repeated=SpikingState(*(v.repeat_interleave(18,0) for v in state.tensors()))
        DNs=[];following=[]
        for first in range(0,len(cue),128):
            count=min(128,len(cue)-first);x=cue[first:first+count]
            before=SpikingState(*(v[first:first+count] for v in repeated.tensors()))
            if count<128 and self.pad_candidate_batch:
                zero=self.initial_state(128-count);x=torch.cat((x,x.new_zeros(128-count,47)))
                before=SpikingState(*(torch.cat((v,z)) for v,z in zip(before.tensors(),zero.tensors())))
            def call(x,*parts):
                DN,after=self.core(x,SpikingState(*parts));return DN,*after.tensors()
            result=checkpoint(call,x,*before.tensors(),use_reentrant=False) if checkpointed and torch.is_grad_enabled() else call(x,*before.tensors())
            DN,*s=result;DNs.append(DN[:count]);following.append(SpikingState(*(v[:count] for v in s)))
        all_DN=torch.cat(DNs);out=self.heads(all_DN,features)
        if return_all_DN:out['candidate_DN']=all_DN.reshape(n,18,-1)
        if action_selector is not None:chosen=action_selector(out['logits'])
        else:chosen=(torch.distributions.Categorical(logits=out['logits'],validate_args=False).sample() if sample else out['logits'].argmax(-1)) if actions is None else actions
        out['actions']=chosen
        # Source18 Bomb holds the same fast-idle movement as source0.
        index=torch.arange(n,device=self.device)*18+torch.where(chosen==18,0,chosen)
        all_after=SpikingState.pack(following);after=SpikingState(*(v.index_select(0,index) for v in all_after.tensors()))
        return out,after,all_DN.index_select(0,index)
    @torch.no_grad()
    def infer_candidates(self,features,states,*,committed_actions=None,return_all_DN=False,action_selector=None):
        f=torch.as_tensor(features,device=self.device,dtype=torch.float32)
        out,after,DN=self.step(f,SpikingState.pack(states),actions=committed_actions,return_all_DN=return_all_DN,action_selector=action_selector)
        return out,[after.row(i) for i in range(len(features))],DN
    def parameter_payload(self,costs):
        from .compact_protected_exit_candidate import REVISION
        from .compact_native_contact_targets import REVISION as NATIVE_REVISION
        return dict(schema=SCHEMA,input_schema=INPUT_SCHEMA,seed=self.seed,costs=dict(costs),temperature=self.temperature,
            state_dict={k:v.detach().cpu().clone() for k,v in self.state_dict().items()},actual_binary_spikes=True,
            one_shared_brain=True,all_source_nodes_and_signed_edges_retained=True,neural_input_per_candidate=47,
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
            critic_input='mean DN across all candidates before action selection;three value heads')


def from_avoidance_checkpoint(connectome,payload,*,device='cuda:0'):
    base=restore_parameters(build_compact_model(connectome,seed=payload['seed'],device=device),payload)
    return CompactStoryPolicy(base)


def from_story_checkpoint(connectome,payload,*,device='cuda:0'):
    base=build_compact_model(connectome,seed=payload['seed'],device=device)
    # Build the same nonlinear DN head shape before restoring the complete state.
    from .dn_residual_risk import load_head
    risk={k.removeprefix('risk.'):v for k,v in payload['state_dict'].items() if k.startswith('risk.')}
    base.risk=load_head(dict(schema=payload.get('risk_readout_kind','DN-only-nonlinear-risk-v1'),state_dict=risk,
        width=payload.get('risk_readout_width',128),
        input_soft_clip=payload.get('risk_input_soft_clip'),base_input_soft_clip=payload.get('risk_base_input_soft_clip'),
        baseline_trainable=payload.get('trainable_risk_baseline',False)),device=device)
    model=CompactStoryPolicy(base,temperature=payload['temperature'])
    if payload.get('task_progress_ports',False):
        model.enable_task_progress_ports(amplitude=payload['task_progress_drive']['initial_amplitude'])
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
        from .compact_protected_exit_candidate import REVISION
        if payload.get('protected_exit_revision')!=REVISION:
            raise ValueError('Unsupported protected exit observation semantics')
    model.primary_attack_target=payload.get('primary_attack_target',False)
    model.risk_policy_gradient=payload.get('risk_policy_gradient',False)
    model.native_contact_risk_calibration=payload.get('native_contact_risk_calibration',False)
    if model.native_contact_risk_calibration:
        from .compact_native_contact_targets import REVISION as NATIVE_REVISION
        if payload.get('native_contact_target_revision')!=NATIVE_REVISION:
            raise ValueError('Unsupported selected native contact target semantics')
    from .compact_story_risk_composition import checked_risk_scale
    model.neural_risk_scale=checked_risk_scale(payload.get('neural_risk_scale',1.))
    from .compact_story_idle_prior import checked_idle_scale
    model.idle_prior_scale=checked_idle_scale(payload.get('idle_prior_scale',1.))
    model.combat_position_guidance_coefficient=payload.get('combat_position_guidance_coefficient',0.)
    model.lower_field_default=payload.get('lower_field_default',False)
    model.full_story_action_selection=payload.get('full_story_action_selection','joint_mode')
    model.soft_risk_composition=payload.get('soft_risk_composition',False)
    model.core.sparse.deterministic_spmm=payload.get('deterministic_spmm',model.core.sparse.deterministic_spmm)
    return model


def sequence(model,features,starts,state,actions,*,checkpointed=True,checkpoint_chunks=False):
    if checkpointed and checkpoint_chunks:
        from .compact_story_shared_checkpoint import sequence as shared_sequence
        return shared_sequence(model,features,starts,state,actions)
    return _sequence_nested(model,features,starts,state,actions,
        checkpointed=checkpointed,checkpoint_chunks=checkpoint_chunks)


def _sequence_nested(model,features,starts,state,actions,*,checkpointed=True,checkpoint_chunks=False):
    """Checkpoint the whole scene so18 replicated brain states are not retained."""
    records=[]
    for t in range(len(features)):
        state=state.reset(starts[t])
        def call(f,a,*parts):
            out,after,_=model.step(f,SpikingState(*parts),actions=a,checkpointed=checkpoint_chunks)
            return out['logits'],out['values'],out['horizon_scores'],*after.tensors()
        result=checkpoint(call,features[t],actions[t],*state.tensors(),use_reentrant=False) if checkpointed else call(features[t],actions[t],*state.tensors())
        logits,values,scores,*parts=result;state=SpikingState(*parts)
        records.append((logits,values,scores))
    return dict(logits=torch.stack([r[0] for r in records]),values=torch.stack([r[1] for r in records]),
        horizon_scores=torch.stack([r[2] for r in records]),state=state)
