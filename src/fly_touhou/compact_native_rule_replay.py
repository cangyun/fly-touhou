"""Supervise one full spiking brain from current-sensor rule-teacher games.

Teacher actions provide movement guidance, while selected native transitions
and trusted geometric forecasts train the existing risk head.  These samples
are never used to calculate PPO ratios or return targets.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .compact_rule_teacher_replay import movement_distribution_loss,mask_rule_guidance_on_contact


def demonstrated_movement_loss(logits,data,controllable):
    """Give verified post-shield continuations the normal actor-guidance weight.

    Ordinary behavior fallback remains weak. Protected examples qualify only
    through the archived actual shield expiry plus a clean96f continuation;
    no candidate risk label or PPO trajectory is invented from the example.
    """
    abstained=data['teacher_guidance_weight']<=0
    protected=data['teacher_protected_behavior_valid']&abstained
    ordinary=data['teacher_behavior_valid']&abstained&~protected
    dtype=data.get('targets',logits).dtype
    target=torch.nn.functional.one_hot(data['actions'].clamp(0,18).long(),19).to(dtype)
    weak,weak_rows=movement_distribution_loss(logits,target,ordinary.to(dtype),data['actions'],controllable)
    reposition,reposition_rows=movement_distribution_loss(logits,target,protected.to(dtype),data['actions'],controllable)
    return .25*weak+reposition,dict(
        native_rule_weak_behavior_rows=int(weak_rows.sum()),native_rule_weak_behavior_loss=float(weak.detach()),
        native_rule_protected_behavior_rows=int(reposition_rows.sum()),
        native_rule_protected_behavior_loss=float(reposition.detach()),native_rule_protected_behavior_weight=1.)


def blend_demonstrated_movement(probabilities,actions,behavior_valid,teacher_weights,*,share=.25):
    """Blend an observed clean continuation with the current18-way guidance.

    This is expert behavior supervision, not an unexecuted action's outcome.
    Rows without a reliable demonstrated continuation retain their targets.
    """
    p=np.asarray(probabilities).copy();a=np.asarray(actions)
    selected=np.asarray(behavior_valid,bool)&(np.asarray(teacher_weights)>0)&(a>=0)&(a<18)
    if not selected.any():return p,dict(rows=0,share=share)
    rows=np.flatnonzero(selected);one=np.eye(18,dtype=p.dtype)[a[rows]]
    original=p[rows,:18];mass=original.sum(-1,keepdims=True)
    conditional=np.divide(original,mass,out=one.copy(),where=mass>0)
    p[rows,:18]=(1-share)*conditional+share*one
    # The recorded continuation used no Bomb. Preserve actual deathbomb rows
    # elsewhere; only demonstrated controllable movement qualifies here.
    previous=p[rows,18].copy();p[rows,18]=0.
    return p,dict(rows=len(rows),share=share,ordinary_Bomb_targets_replaced=int((previous>0).sum()),
        alternative_actions_not_labeled_unsafe=True)


def demonstrated_safe_continuation(actions, outcomes, normal, elapsed_frames,
                                   before_frames, misses_before, *, horizon=32):
    """Mark recorded moves followed by a collision/Bomb-free teacher prefix.

    This is a retrospective behavior label, not a guarantee that the chosen
    movement or any alternative is safe. Unknown or incomplete future rows,
    contacts, Bombs and non-four-frame decisions do not qualify.
    """
    actions=np.asarray(actions);outcomes=np.asarray(outcomes);normal=np.asarray(normal,bool)
    elapsed_frames=np.asarray(elapsed_frames);before_frames=np.asarray(before_frames)
    misses_before=np.asarray(misses_before);n=len(actions)
    if np.any(np.diff(before_frames)<0):
        raise ValueError('Teacher episode frames must be chronological')
    stop=np.searchsorted(before_frames,before_frames+horizon,side='left')
    complete=stop<n
    last=np.minimum(stop,max(0,n-1))
    bad=(outcomes==1)|(actions==18)|(elapsed_frames!=4)
    prefix=np.concatenate(([0],np.cumsum(bad,dtype=np.int64)))
    clear=(prefix[np.minimum(stop,n)]-prefix[np.arange(n)])==0
    return (complete&clear&normal&(actions>=0)&(actions<18)&(outcomes==0)
            &(elapsed_frames==4)&(misses_before[last]==misses_before))


def veto_contradicted_forecasts(margins, trusted, actions, outcomes, elapsed_frames=4):
    trusted = np.asarray(trusted, dtype=bool).copy()
    action = np.asarray(actions)
    outcome = np.asarray(outcomes)
    elapsed = np.asarray(elapsed_frames)
    valid = (action >= 0) & (action < 18) & ((outcome == 0) | (outcome == 1))
    chosen = np.asarray(margins)[np.arange(len(action)), 0, np.clip(action, 0, 17)]
    contradiction = valid & (((outcome == 1) & (elapsed>0) & (elapsed<=4) & (chosen > 0))
                              | ((outcome == 0) & (elapsed==4) & (chosen < 0)))
    trusted[contradiction] = False
    return trusted, contradiction


def demonstrated_bomb_family_loss(logits,data,teacher_weights):
    """The Bomb-vs-movement part missing from conditional18 imitation.

    Negative labels use actual controllable teacher movement with either a
    usable no-Bomb reference or a measured32-frame Bomb/contact-free future.
    Positive labels require the recorded native stocked deathbomb window.
    Ordinary heuristic Bomb proposals are not treated as unavoidable danger.
    """
    f=data['features'];a=data['actions']
    control=(f[...,14]>.5)|(f[...,17]>.5)
    stock=(f[...,4]>0)&torch.isfinite(logits[...,18])
    critical=stock&(f[...,12]>0)&(f[...,17]>.5)
    positive=critical&(a==18)
    no_bomb=(data['targets'][...,18]<=1e-6)&(teacher_weights>0)
    observed=data['teacher_behavior_valid']
    negative=stock&control&~critical&(a>=0)&(a<18)&(no_bomb|observed)
    # Observed immediate contacts cannot endorse the proposed move/resource
    # decision, including stale behavior flags in old archives.
    negative &= ~(data['unprotected_mask']&(data['outcome']==1))
    eligible=positive|negative
    if not bool(eligible.any()):
        zero=logits[...,18].masked_fill(~torch.isfinite(logits[...,18]),0.).sum()*0.
        return zero,dict(rows=0,positive=0,negative=0)
    odds=(logits[...,18]-logits[...,:18].logsumexp(-1))[eligible]
    target=positive[eligible].to(odds.dtype)
    loss=torch.nn.functional.binary_cross_entropy_with_logits(odds,target)
    return loss,dict(rows=int(eligible.sum()),positive=int(positive.sum()),negative=int(negative.sum()))


class NativeRuleTeacherReplay:
    def __init__(self, manifest, *, split='train', stage_weights=None):
        self.path = Path(manifest)
        self.manifest = json.loads(self.path.read_text(encoding='utf-8'))
        if self.manifest.get('schema') != 'native-compact129-rule-teacher-windows-v1':
            raise ValueError('Expected current-sensor rule teacher windows')
        self.burn_in = int(self.manifest['burn_in'])
        if split not in ('train', 'heldout'):
            raise ValueError('Rule teacher split must be train or heldout')
        self.split = split
        candidates = [w for w in self.manifest['windows'] if w['split'] == split]
        self.windows = []
        for row in candidates:
            with np.load(self.path.parent/row['file'], allow_pickle=False) as z:
                start=self.burn_in
                f=z['features'][start:]; a=z['actions'][start:]
                normal=z['unprotected_mask'][start:]; y=z['outcome'][start:]
                elapsed=z['native_action_frames'][start:]
                movement=((z['teacher_guidance_weight'][start:]>0)&(a<18)
                    &((f[:,14]>.5)|(f[:,17]>.5))&(z['targets'][start:,:18].sum(-1)>0))
                behavior=(z['teacher_behavior_valid'][start:] if 'teacher_behavior_valid' in z.files
                          else np.zeros_like(a,dtype=bool))
                if 'teacher_protected_behavior_valid' in z.files:
                    behavior |= z['teacher_protected_behavior_valid'][start:]
                behavior &= (z['teacher_guidance_weight'][start:]<=0)
                forecast=normal & z['trusted'][start:].any(-1)
                from .compact_native_contact_targets import native_exposure_mask
                native=native_exposure_mask(f,normal) & (a<18) & (((y==1)&(elapsed>0)&(elapsed<=4))|((y==0)&(elapsed==4)))
                partial=(normal & z['partial_point_danger'][start:].any(axis=(-2,-1))
                    if 'partial_point_danger' in z.files else np.zeros_like(normal))
                deathbomb=(f[:,4]>0)&(f[:,12]>0)&(f[:,17]>.5)&(a==18)
            if any(x.any() for x in (movement,behavior,forecast,native,partial,deathbomb)):
                self.windows.append(row)
        self.empty_windows_excluded=len(candidates)-len(self.windows)
        if not self.windows:
            raise ValueError('No current-sensor teacher windows available')
        self.stage_indices={stage:np.asarray([i for i,w in enumerate(self.windows) if w['stage']==stage])
                            for stage in sorted({w['stage'] for w in self.windows})}
        self.stage_weights=None
        if stage_weights is not None:
            weights={int(stage):float(weight) for stage,weight in stage_weights.items()}
            if (set(weights)!=set(self.stage_indices) or any(not np.isfinite(w) or w<0 for w in weights.values())
                    or sum(weights.values())<=0):
                raise ValueError('Stage weights must cover available stages with nonnegative finite weights')
            total=sum(weights.values())
            self.stage_weights={stage:weights[stage]/total for stage in self.stage_indices}
        self.stage_draws={stage:0 for stage in self.stage_indices}

    def sample_index(self, rng):
        """Meet stage quotas across updates; randomize windows within a stage.

        Weighted random draws could skip stage4 entirely in a short update
        pass. Largest-deficit scheduling carries its counts across rollouts.
        Teacher rows remain supervised examples, never PPO experience.
        """
        if self.stage_weights is None:
            index=int(rng.integers(2**31))%len(self.windows)
            self.stage_draws[self.windows[index]['stage']]+=1
            return index
        total=sum(self.stage_draws.values())+1
        stage=max((s for s,w in self.stage_weights.items() if w>0),
            key=lambda s:total*self.stage_weights[s]-self.stage_draws[s])
        self.stage_draws[stage]+=1
        choices=self.stage_indices[stage]
        return int(choices[int(rng.integers(len(choices)))])

    def sample_indices(self,rng,count):
        """Apply the same stage quotas across a packed teacher minibatch.

        Avoid repeating a window within the batch when its stage has enough
        examples. Stage counts advance once per selected training window.
        """
        if not isinstance(count,int) or isinstance(count,bool) or count<1:
            raise ValueError('Positive teacher-window count required')
        if count==1:return [self.sample_index(rng)]
        selected=[]
        for _ in range(count):
            if self.stage_weights is None:
                choices=[i for i in range(len(self.windows)) if i not in selected]
                if not choices:choices=list(range(len(self.windows)))
                index=choices[int(rng.integers(2**31))%len(choices)]
                stage=self.windows[index]['stage']
            else:
                total=sum(self.stage_draws.values())+1
                stage=max((s for s,w in self.stage_weights.items() if w>0),
                    key=lambda s:total*self.stage_weights[s]-self.stage_draws[s])
                choices=[i for i in self.stage_indices[stage] if i not in selected]
                if not choices:choices=self.stage_indices[stage]
                index=int(choices[int(rng.integers(len(choices)))])
            self.stage_draws[stage]+=1;selected.append(index)
        return selected

    def loss(self, model, *, index, risk_loss_scale=1.):
        return self.loss_batch(model,indices=[index],risk_loss_scale=risk_loss_scale)

    def loss_batch(self, model, *, indices, risk_loss_scale=1.):
        if not 0 <= risk_loss_scale <= 1:
            raise ValueError('Teacher risk loss scale must be in [0,1]')
        if not indices:raise ValueError('At least one teacher window required')
        from .compact_story_model import sequence
        from .compact_story_ppo import risk_weights, risk_loss,partial_risk_weights
        from .compact_native_contact_targets import native_contact_weights,native_exposure_mask
        rows=[self.windows[index % len(self.windows)] for index in indices]
        fields = ('features','targets','actions','episode_starts','teacher_guidance_weight',
                  'margins','trusted','unprotected_mask','outcome','native_action_frames')
        arrays=[];has_partial=False
        for row in rows:
            with np.load(self.path.parent/row['file'], allow_pickle=False) as archive:
                d={k:archive[k].copy() for k in fields}
                d['teacher_behavior_valid']=(archive['teacher_behavior_valid'].copy()
                    if 'teacher_behavior_valid' in archive.files else np.zeros(len(d['actions']),dtype=bool))
                d['teacher_protected_behavior_valid']=(archive['teacher_protected_behavior_valid'].copy()
                    if 'teacher_protected_behavior_valid' in archive.files else np.zeros(len(d['actions']),dtype=bool))
                has_partial=has_partial or 'partial_point_danger' in archive.files
                d['partial_point_danger']=(archive['partial_point_danger'].copy()
                    if 'partial_point_danger' in archive.files else np.zeros_like(d['margins'],dtype=bool))
                arrays.append(d)
        data={k:torch.as_tensor(np.stack([d[k] for d in arrays],axis=1),device=model.device) for k in arrays[0]
              if k!='partial_point_danger' or has_partial}
        state = model.initial_state(len(rows))
        if self.burn_in:
            with torch.no_grad():
                state = sequence(model, data['features'][:self.burn_in],
                    data['episode_starts'][:self.burn_in], state, data['actions'][:self.burn_in],
                    checkpointed=False)['state'].detach()
        result = sequence(model, data['features'][self.burn_in:],
            data['episode_starts'][self.burn_in:], state, data['actions'][self.burn_in:])
        d = {k: v[self.burn_in:] for k,v in data.items()}
        controllable = (d['features'][...,14] > .5) | (d['features'][...,17] > .5)
        exposed=native_exposure_mask(d['features'],d['unprotected_mask'])
        teacher_weights,contact_rows=mask_rule_guidance_on_contact(d['teacher_guidance_weight'],
            d['actions'],d['outcome'],exposed,d['native_action_frames'])
        if getattr(model,'lower_field_default',False):
            from .compact_lower_field import movement_reference
            d['targets'],_=movement_reference(d['features'],d['margins'],d['trusted'],
                d['unprotected_mask'],d['targets'],teacher_weights)
        movement, eligible = movement_distribution_loss(result['logits'], d['targets'],
            teacher_weights, d['actions'], controllable)
        # A teacher may abstain from certifying a distribution yet still have
        # successfully executed an action. Ordinary continuations stay weak;
        # verified post-shield routes receive normal movement-guidance weight.
        # Neither supplies safety labels for the other17 directions.
        behavior, behavior_info = demonstrated_movement_loss(result['logits'],d,controllable)
        family,family_rows=demonstrated_bomb_family_loss(result['logits'],d,teacher_weights)
        regression, classification = risk_weights(d['margins'], d['trusted'], d['unprotected_mask'])
        classification,partial=partial_risk_weights(d,classification)
        weights, counts = native_contact_weights(d['actions'], d['outcome'],
            exposed, d['native_action_frames'])
        native = dict(actions=d['actions'], outcomes=d['outcome'], unprotected=exposed,
            elapsed_frames=d['native_action_frames'], weights=weights, share=.25 if bool(weights.any()) else 0.)
        risk = risk_loss(result['horizon_scores'], d['margins'], regression, classification, native=native,partial=partial)
        with torch.no_grad():
            # The 18-direction comparison below uses modeled clearances, not
            # unexecuted native outcomes. Count pairwise ordering and whether
            # the policy's best movement is in the modeled-danger set.
            margin=d['margins'][...,0,:]
            valid=d['trusted'][...,0]&d['unprotected_mask']
            danger=(margin<=0)&valid[...,None]
            safe=(margin>1)&valid[...,None]
            mixed=danger.any(-1)&safe.any(-1)
            score=result['horizon_scores'][...,0]
            pairs=danger[..., :,None]&safe[...,None,:]
            greater=(score[..., :,None]>score[...,None,:]).to(score.dtype)
            ties=(score[..., :,None]==score[...,None,:]).to(score.dtype)
            correct=((greater+.5*ties)*pairs).sum()
            chosen=result['logits'][...,:18].argmax(-1)
            chosen_danger=danger.gather(-1,chosen[...,None])[...,0]&mixed
        stage_counts={stage:sum(row['stage']==stage for row in rows) for stage in sorted({row['stage'] for row in rows})}
        position_coefficient=getattr(model,'combat_position_guidance_coefficient',0.)
        position=movement.new_zeros(());position_rows=0
        if position_coefficient:
            from .compact_combat_position import combat_position_loss
            if getattr(model,'lower_field_default',False):
                from .compact_lower_field import combat_position_loss
            position,position_rows=combat_position_loss(result['logits'],d['features'],d['margins'],
                d['trusted'],d['unprotected_mask'],d['targets'],teacher_weights)
        return .1*(movement+behavior+family) + risk_loss_scale*risk + position_coefficient*position, dict(
            native_rule_position_loss=float(position.detach()),native_rule_position_rows=int(position_rows),
            native_rule_stage=rows[0]['stage'] if len(stage_counts)==1 else 0,
            native_rule_windows=len(rows),native_rule_stage_windows=stage_counts,
            native_rule_movement_rows=int(eligible.sum()), native_rule_native_clears=int(counts[0]),
            native_rule_native_contacts=int(counts[1]), native_rule_PPO_rows=0,
            native_rule_contact_teacher_veto_rows=int((contact_rows & (d['teacher_guidance_weight']>0)).sum()),
            native_rule_movement_loss=float(movement.detach()), native_rule_risk_loss=float(risk.detach()),
            **behavior_info,
            native_rule_Bomb_family_loss=float(family.detach()),
            native_rule_Bomb_family_rows=family_rows['rows'],
            native_rule_no_Bomb_rows=family_rows['negative'],
            native_rule_confirmed_deathbomb_rows=family_rows['positive'],
            native_rule_partial_point_danger_pairs=int((partial>0).sum()) if partial is not None else 0,
            risk_proxy_mixed_scenes=int(mixed.sum()),risk_proxy_ordered_pairs=int(pairs.sum()),
            risk_proxy_correct_pairs=float(correct),
            policy_proxy_danger_choices=int(chosen_danger.sum()))

    def summary(self):
        return dict(manifest=str(self.path), split=self.split, windows=len(self.windows),
            source_native_repair_revisions=self.manifest.get('source_native_repair_revisions'),
            empty_windows_excluded=self.empty_windows_excluded,
            stages=sorted({w['stage'] for w in self.windows}), teacher='ResourceV10',
            stage_weights=self.stage_weights,
            stage_sampling='weighted_deficit_quota' if self.stage_weights else 'uniform_windows',
            sampled_windows_by_stage=dict(self.stage_draws),
            batching='Independent window histories;18 candidates per window;up to7 windows fit the existing128-candidate padded core call',
            teacher_controlled_data=True, PPO_samples=0, approximate_initial_brain_state=True,
            risk='trusted modeled directions plus native contact for the executed direction only',
            contact_rule_veto='Ignore rule imitation on an observed controllable movement contact within4 frames;retain factual risk target',
            Bomb_family_guidance='No-Bomb on usable teacher movement or measured clean continuation;positive only from actual stocked deathbomb windows;not ordinary heuristic Bomb proposals',
            partial_risk='Conditional discrete point danger only;shares existing BCE budget;no invented safe directions',
            behavior_horizon_frames=self.manifest.get('behavior_horizon_frames',32),
            demonstrated_route_blend=self.manifest.get('demonstrated_route_blend'),
            protected_repositioning=self.manifest.get('protected_repositioning'),
            behavior_fallback=self.manifest.get('behavior_fallback',
                'If rule guidance abstains, weakly imitate its executed move only when followed by32 native frames without contact or Bomb'))
