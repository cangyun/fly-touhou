"""Repeat real student deathbomb windows as teacher supervision, never PPO.

Keep the genuine preceding brain state and recorded movement history. Normal
rows in the same16-step segment retain their rule labels as a contrast to
the critical Bomb targets. No artificial contact or reward is introduced.
"""
from dataclasses import replace
import torch
from .fly_gru_aligned_guidance import DEFAULT_CONFIG,gru_aligned_guidance_loss

CONFIG=replace(DEFAULT_CONFIG,safe_coefficient=0.,future_coefficient=0.,collision_coefficient=0.)
FIELDS=('features','actions','episode_starts','margins','trusted','unprotected_mask','outcomes',
    'rule_probabilities','rule_weights','deathbomb_mask')


class CriticalTeacherReplay:
    def __init__(self,*,balanced=False,protected_exit_forecast=False,native_motion_forecast=False,
                 native_focus_gate=False,unknown_laser_player_motion=False,native_curve_laser_geometry=False,
                 native_repair_revisions=None):
        self.sequences=[];self.sources=[];self.observed_rows=0;self.ignored_rule_rows=0
        self.balanced=balanced;self.normal_sequences=[]
        self.protected_exit_forecast=bool(protected_exit_forecast);self.incompatible_protected_blocks=0
        self.native_motion_forecast=bool(native_motion_forecast);self.outdated_geometry_rule_rows=0
        self.native_repair_revisions=native_repair_revisions
        self.sensor_revisions=dict(native_focus_gate=bool(native_focus_gate),
            unknown_laser_player_motion=bool(unknown_laser_player_motion),
            native_curve_laser_geometry=bool(native_curve_laser_geometry))
    def _normal_priority(self,logits,data):
        # Rank an existing no-Bomb label by its worst current disagreement.
        # A one-frame false alarm must not be diluted by15 already-learned
        # rows. This ranks supervised windows; it is not another loss term.
        mask=((data['rule_weights']>0)&(data['rule_probabilities'][...,18]<=1e-6)
            &~data['deathbomb_mask']&(data['outcomes']!=1))
        if not bool(mask.any()):return 0.
        values=logits.detach().float()
        odds=values[...,18]-values[...,:18].logsumexp(-1)
        return float(torch.nn.functional.softplus(odds)[mask].max())
    def _old_sensor_semantics(self,rollout):
        metadata=rollout.get('metadata',{})
        return ((self.native_repair_revisions is not None
                 and metadata.get('native_repair_revisions')!=self.native_repair_revisions)
                or (self.native_motion_forecast and not metadata.get('native_motion_forecast',False))
                or any(enabled and not metadata.get(name,False)
                       for name,enabled in self.sensor_revisions.items()))
    def _compatible(self,rollout,t,w):
        if 'valid_mask' in rollout and not bool(rollout['valid_mask'][t:t+16,w].all()):return False
        source=bool(rollout.get('metadata',{}).get('protected_exit_forecast',False))
        if source:
            from .compact_protected_exit_candidate import REVISION
            if rollout['metadata'].get('protected_exit_revision')!=REVISION:
                raise ValueError('Unsupported replay protected exit observation semantics')
        if source==self.protected_exit_forecast:return True
        f=rollout['features'][t:t+16,w]
        if bool(((f[...,5]>.5)&(f[...,14]>.5)).any()):
            self.incompatible_protected_blocks+=1
            return False
        return True
    def _copy_block(self,rollout,t,w,source,*,rule_mask=None):
        data={k:rollout[k][t:t+16,w:w+1].detach().cpu().clone() for k in FIELDS}
        if 'lower_field_exception' in rollout:
            data['lower_field_exception']=rollout['lower_field_exception'][t:t+16,w:w+1].detach().cpu().clone()
        control=(data['features'][...,14]>.5)|(data['features'][...,17]>.5)
        self.ignored_rule_rows+=int(((data['rule_weights']>0)&~control).sum())
        data['rule_weights']*=control.to(data['rule_weights'].dtype)
        if self._old_sensor_semantics(rollout):
            # Old safety forecasts can be wrong; keep actual state4 Bomb
            # positives, but never re-teach their obsolete route labels.
            self.outdated_geometry_rule_rows+=int((data['rule_weights']>0).sum())
            data['rule_weights'].zero_()
        if rule_mask is not None:data['rule_weights']*=rule_mask[t:t+16,w:w+1].to(data['rule_weights'].dtype)
        before=rollout['block_states'][t].row(w).detach().to('cpu')
        before=type(before)(*(value.clone() for value in before.tensors()))
        entry=dict(data=data,before=before,
            source=source,source_time=t,source_world=w)
        recorded=rollout.get('recorded_policy_logits')
        critical=data['deathbomb_mask'][:,0]
        if recorded is not None and bool(critical.any()):
            odds=recorded[t:t+16,w].detach().cpu().float().log_softmax(-1)[:,18]
            entry['critical_priority']=float(-odds[critical].mean())
            entry['critical_priority_source']='observed student behavior'
        if recorded is not None and rule_mask is not None:
            entry['normal_priority']=self._normal_priority(recorded[t:t+16,w:w+1].detach().cpu(),data)
            entry['normal_priority_source']='observed student behavior'
        return entry
    def _add_normal_contrast(self,rollout,source):
        if self._old_sensor_semantics(rollout):return
        f=rollout['features'];safe=rollout['margins'][...,0,:]>1
        # Use existing labels from actual stocked student states only. No
        # invented inventory, forced Bomb ban, or new target probability.
        eligible=((f[...,4]>0)&(f[...,14]>.5)&(f[...,17]<=.5)
            &(rollout['rule_weights']>0)&(rollout['rule_probabilities'][...,18]<=1e-6)
            &(~rollout['unprotected_mask']|(rollout['trusted'][...,0]&safe.any(-1))))
        mixed=eligible&rollout['trusted'][...,0]&safe.any(-1)&(rollout['margins'][...,0,:]<=0).any(-1)
        # High Bomb probability with a real no-Bomb teacher label is the error
        # we need to learn. Density alone misses false alarms in easy scenes.
        logits=rollout.get('recorded_policy_logits')
        probability=(logits.softmax(-1)[...,18] if logits is not None else torch.zeros_like(f[...,4]))
        probability=torch.where(eligible,probability,0.)
        best={}
        for t,w in sorted({(int(t)//16*16,int(w)) for t,w in eligible.nonzero().tolist()}):
            if t+16>len(f) or bool(rollout['deathbomb_mask'][t:t+16,w].any()):continue
            if not self._compatible(rollout,t,w):continue
            probs=probability[t:t+16,w]
            score=(int((probs>.5).sum()),float(probs.max()),float(probs.sum()),
                int(mixed[t:t+16,w].sum()),int(eligible[t:t+16,w].sum()))
            if w not in best or score>best[w][0]:best[w]=(score,t)
        # Two hardest worlds plus two rotating worlds preserve both error
        # coverage and diversity. At most one block per world per source.
        offset=(len(self.sources)-1)*4 % f.shape[1]
        hard=sorted(best,key=lambda w:best[w][0],reverse=True)[:2] if logits is not None else []
        worlds=hard+[w for w in sorted(best,key=lambda w:(w-offset)%f.shape[1]) if w not in hard][:4-len(hard)]
        for w in worlds:
            self.normal_sequences.append(self._copy_block(rollout,best[w][1],w,source,rule_mask=eligible))
        self.normal_sequences=self.normal_sequences[-32:]
    def add(self,rollout,*,source):
        if source in self.sources:return 0
        self.sources.append(source)
        mask=rollout['deathbomb_mask'];self.observed_rows+=int(mask.sum())
        blocks=sorted({(t//16*16,w) for t,w in mask.nonzero().tolist()})
        added=0
        for t,w in blocks:
            if not self._compatible(rollout,t,w):continue
            features=rollout['features'][t:t+16,w:w+1]
            selected=mask[t:t+16,w:w+1]
            if not bool(((features[...,4]>0)&(features[...,12]>0)&(features[...,17]>.5))[selected].all()):
                raise ValueError('Critical replay must be an observed stocked native state4 window')
            self.sequences.append(self._copy_block(rollout,t,w,source))
            added+=1
        if self.balanced:self._add_normal_contrast(rollout,source)
        return added
    def loss(self,model,*,index,normal_index=None):
        from .compact_story_model import sequence
        from .spiking_engineered_control import SpikingState
        selected=[];normal_rows=0;normal_choice=None;normal_priority=None
        if self.sequences:selected.append(self.sequences[index%len(self.sequences)])
        if self.balanced and self.normal_sequences:
            normal_choice=(index if normal_index is None else normal_index)%len(self.normal_sequences)
            normal=self.normal_sequences[normal_choice]
            selected.append(normal);normal_rows=int((normal['data']['rule_weights']>0).sum())
        if not selected:raise ValueError('Teacher replay requires an observed student sequence')
        data={k:torch.cat([s['data'][k] for s in selected],1).to(model.device) for k in FIELDS}
        from .compact_lower_field import special_pattern_mask
        exceptions=torch.cat([special_pattern_mask(s['data']) for s in selected],1).to(model.device)
        before=SpikingState.pack([s['before'] for s in selected]).to(model.device)
        result=sequence(model,data['features'],data['episode_starts'],before,data['actions'])
        labels={k:data[k] for k in FIELDS if k not in ('features','episode_starts')}
        loss=gru_aligned_guidance_loss(result['logits'],**labels,config=CONFIG,veto_rule_on_contact=True,
            position_features=data['features'] if getattr(model,'lower_field_default',False) else None,
            position_exceptions=exceptions)
        # Refresh only the sampled critical window, using its current neural
        # prediction. This changes supervised sampling, never PPO weighting.
        if self.sequences:
            entry=self.sequences[index%len(self.sequences)]
            entry['critical_priority']=float(loss['deathbomb_loss'].detach())
            entry['critical_priority_source']='latest supervised replay'
        if normal_choice is not None:
            normal_priority=self._normal_priority(result['logits'][:,-1:],{k:v[:,-1:] for k,v in data.items()})
            normal['normal_priority']=normal_priority
            normal['normal_priority_source']='latest supervised replay'
        return loss['total_loss'],dict(critical_teacher_loss=float(loss['total_loss'].detach()),
            critical_deathbomb_loss=float(loss['deathbomb_loss'].detach()),
            critical_rule_loss=float(loss['rule_loss'].detach()),
            replayed_usable_window_rows=int(data['deathbomb_mask'].sum()),
            replayed_teacher_rows=int(data['features'].shape[0]*data['features'].shape[1]),
            replayed_stocked_normal_rule_rows=normal_rows,
            replayed_PPO_rows=0,
            critical_replay_sampling='half_uniform_half_current_error',
            normal_replay_sampling='half_uniform_half_current_Bomb_error',
            normal_replay_index=normal_choice,normal_Bomb_priority=normal_priority,
            critical_replay_index=int(index%len(self.sequences)) if self.sequences else None)
    def sampling_probabilities(self):
        if not self.sequences:return []
        errors=[float(s.get('critical_priority',1.)) for s in self.sequences]
        if any(not 0<=error<float('inf') for error in errors):
            raise ValueError('Critical replay priorities must be finite nonnegative losses')
        total=sum(errors);uniform=1./len(errors)
        return [.5*uniform+.5*error/total for error in errors] if total>0 else [uniform]*len(errors)
    def sample_index(self,rng):
        probabilities=self.sampling_probabilities()
        return int(rng.choice(len(probabilities),p=probabilities)) if probabilities else 0
    def normal_sampling_probabilities(self):
        if not self.normal_sequences:return []
        errors=[float(s.get('normal_priority',1.)) for s in self.normal_sequences]
        if any(not 0<=error<float('inf') for error in errors):
            raise ValueError('Normal replay priorities must be finite nonnegative errors')
        total=sum(errors);uniform=1./len(errors)
        return [.5*uniform+.5*error/total for error in errors] if total>0 else [uniform]*len(errors)
    def sample_normal_index(self,rng):
        probabilities=self.normal_sampling_probabilities()
        return int(rng.choice(len(probabilities),p=probabilities)) if probabilities else 0
    def summary(self):
        return dict(sequences=len(self.sequences),observed_usable_window_rows=self.observed_rows,
            critical_sampling='half_uniform_half_current_error',
            critical_priority_updates=sum(s.get('critical_priority_source')=='latest supervised replay' for s in self.sequences),
            noncontrollable_rule_rows_excluded=self.ignored_rule_rows,
            protected_exit_forecast=self.protected_exit_forecast,
            native_motion_forecast=self.native_motion_forecast,
            native_repair_revisions=self.native_repair_revisions,
            sensor_revisions=self.sensor_revisions,
            outdated_geometry_rule_rows_excluded=self.outdated_geometry_rule_rows,
            incompatible_protected_blocks_excluded=self.incompatible_protected_blocks,
            source_tapes=list(self.sources),objective='existing rule(.1)+deathbomb(.05) supervised terms only',
            new_PPO_samples=0,artificial_native_contacts=0,teacher_executes_actions=False,
            balanced_stocked_normal_replay=self.balanced,stocked_normal_sequences=len(self.normal_sequences),
            normal_sampling='half_uniform_half_current_Bomb_error',
            normal_priority_updates=sum(s.get('normal_priority_source')=='latest supervised replay' for s in self.normal_sequences),
            stocked_normal_rule_rows=sum(int((s['data']['rule_weights']>0).sum()) for s in self.normal_sequences),
            normal_selection='v2: recorded student Bomb disagreement first; two hardest worlds plus rotating diversity; teacher labels and actual inventory unchanged',
            normal_bank_scope='Existing actual teacher distributions on stocked controllable states; known feasible movement or current protection; no deathbomb window in the contrast block. One critical and one normal sequence share the original rule mean; the deathbomb term keeps its original positive-row denominator.')
