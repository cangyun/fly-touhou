"""Successful NBNM histories for supervised retention, never PPO replay."""
from pathlib import Path
import json
import numpy as np
from functools import lru_cache

SCHEMA='nbnm-successful-history-rehearsal-v1'


@lru_cache(maxsize=1)
def movement_velocities():
    from .thrl_movement_teacher import action_velocities
    return action_velocities()


def verified_turn_weights(features, actions, frames, phases, start, length, training_start):
    """Teach the two commands before a witnessed route bend and its turn command."""
    from .compact_spiking_sensation import FEATURE_NAMES
    weights=np.ones(length,np.float32)
    n=len(frames);offset=len(actions)-n
    if n<2 or phases is None or len(phases)!=n or offset<0:return weights
    a=actions[offset:];f=features[offset:];eligible=(a>=0)&(a<18)
    eligible &= (f[:,14]>.5)&(f[:,5]<=0)&(f[:,13]<=0)&(np.arange(n)+offset>=training_start)
    direction=movement_velocities()[a.clip(0,17)].copy()
    locked=(f[:,FEATURE_NAMES.index('focused')]<0)&(a>=9)&(a<18)
    direction[locked]=movement_velocities()[a[locked]-9]
    valid_pair=(np.diff(frames)==4)&(phases[1:]==phases[:-1])&eligible[:-1]&eligible[1:]
    turns=np.flatnonzero(valid_pair & (direction[1:]!=direction[:-1]).any(axis=1))+1
    for i in turns[(turns+offset>=start)&(turns+offset<start+length+2)]:
        for k in range(i,max(-1,i-3),-1):
            j=k+offset-start
            if 0<=j<length:weights[j]=2.
            if k==0 or not valid_pair[k-1]:break
    return weights


def verified_correction(row):
    proof=row.get('native_validation',{});task=row.get('task',{})
    if proof.get('contacts')!=0 or proof.get('Bombs')!=0:return False
    if task.get('mode')=='attack':
        return bool(row.get('source_kind') in ('successful_rule_teacher','native_verified_teacher_continuation')
            and proof.get('task_complete') and proof.get('native_seed')==row.get('native_seed')
            and proof.get('task')==task and proof.get('goal') in ('native-spell-capture','native-spell-survived-timeout'))
    if row.get('source_kind') not in ('native_verified_teacher_continuation',
                                      'native_verified_student_continuation'):return False
    if proof.get('stage_clear'):return True
    goal=proof.get('goal')
    if task.get('mode')!='route' or not proof.get('task_complete') or task.get('end_event')!=goal:
        return False
    phase=proof.get('final_phase','')
    if goal=='first-native-midboss':
        return str(phase).startswith(f"s{task.get('stage')}/MBoss")
    if goal=='native-frame-arrival':
        target,observed=task.get('end_native_frame'),proof.get('observed_frame')
        return bool(type(target) is int and type(observed) is int and 0 < target <= observed
            and proof.get('observed_stage')==task.get('stage')
            and proof.get('native_seed')==row.get('native_seed') and proof.get('task')==task)
    attack=task.get('end_attack')
    return bool(goal=='first-native-attack-entry' and attack
        and proof.get('native_seed')==row.get('native_seed') and proof.get('task')==task
        and phase==f"s{task.get('stage')}/{attack}")


def continuation_window_starts(first, length, size, row, count, rng):
    """Keep the onset, measured hazard windows, and later demonstrated follow-through."""
    windows=[first]
    for start in row.get('continuation_priority_starts', ()):
        if type(start) is not int or not int(row.get('training_start', 0)) <= start <= size-length:
            raise ValueError('Each continuation priority must start a complete supervised window')
        if len(windows)<count and first+length<=start and all(abs(start-s)>=length for s in windows):
            windows.append(start)
    priority_end=max((int(end) for _,end in row.get('correction_windows',())),default=first+length)
    candidates=np.arange(max(first+length,priority_end),size-length+1,length)
    for start in windows[1:]:
        candidates=candidates[np.abs(candidates-start)>=length]
    groups=min(count-len(windows),len(candidates))
    if groups>0:
        windows.extend(int(rng.choice(group)) for group in np.array_split(candidates,groups))
    return sorted(windows)


def cacheable_window_prefixes(starts):
    from .nbnm_prefix import CUDA_BURN_IN_MIN_STEPS
    # An ordinary multi-window update takes every snapshot from the longest
    # replay. Leave mixed eager/graph lengths on that path.
    return not (starts[0] < CUDA_BURN_IN_MIN_STEPS <= starts[-1])


def export_successes(rollout, episodes, output, *, source, split='train'):
    """Export only complete native-practice episodes with known initial state.

    Partial episodes at a segment boundary cannot reconstruct initial memory.
    Story prefixes need their explicit burn-in history and are not guessed.
    """
    from .thrl_training_mode import seed_split
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    records=[];valid=rollout['valid_mask'].numpy()
    for row in episodes:
        if (not row.get('nbnm_success') or row.get('task_misses')!=0 or row.get('task_Bombs')!=0
            or row.get('nbnm_entry_mode')!='native_spell_practice' or seed_split(row['native_seed'])!=split):continue
        w=int(row['world']);episode=int(row['native_episode'])
        ids=np.flatnonzero(valid[:,w] & (rollout['recorded_native_episode'][:,w].numpy()==episode))
        if (len(ids)<16 or np.any(np.diff(ids)!=1) or not rollout['episode_starts'][ids[0],w]
            or not rollout['terminated'][ids[-1],w] or rollout['reward_vectors'][ids[-1],w,0]!=1):continue
        actions=rollout['actions'][ids,w].numpy()
        if (actions>=18).any() or rollout['recorded_observed_contacts'][ids,w].any():continue
        name=f'world{w}-episode{episode}.npz'
        np.savez_compressed(output/name,features=rollout['features'][ids,w].numpy(),actions=actions,
            starts=rollout['episode_starts'][ids,w].numpy())
        records.append(dict(file=name,task=row['task'],native_seed=row['native_seed'],split=split,
            decisions=len(ids),task_native_frames=row['task_native_frames'],nbnm_success=True,
            source_kind='successful_student_behavior',initial_memory='zero at native card entry'))
    manifest=dict(schema=SCHEMA,source=str(source),histories=records,PPO_rows=0,
        target='executed movements from successful NBNM episodes; not all-direction safety labels',
        native_risk_labels=False,source_policy_changing=True)
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    return manifest


class SuccessRehearsal:
    def __init__(self, manifest, *, sequence_length=16, scope_weights=None, additional_manifests=(),
                 attack_spell_fraction=None,challenging_fraction=0.,phase_priorities=None,priority_fraction=.5,
                 native_short_risk_coefficient=0.,native_correction=None,calibrate_matching_continuations=False,
                 emphasize_verified_turns=False, native_exit_coefficient=0., match_reference_scores=False,
                 prefer_observed_branches=False, balance_native_seeds=False, include_supported_native_exits=False):
        self.path=Path(manifest)
        self.length=int(sequence_length)
        if self.length!=16:raise ValueError('Use the existing16-step neural sequence')
        self.additional_manifests=list(map(Path,additional_manifests))
        self.scope_weights=scope_weights
        self.attack_spell_fraction=attack_spell_fraction;self.challenging_fraction=float(challenging_fraction)
        self.phase_priorities=dict(phase_priorities or {});self.priority_fraction=float(priority_fraction)
        self.native_short_risk_coefficient=float(native_short_risk_coefficient)
        self.native_exit_coefficient=float(native_exit_coefficient)
        self.include_supported_native_exits=bool(include_supported_native_exits)
        self.match_reference_scores=bool(match_reference_scores)
        self.prefer_observed_branches=bool(prefer_observed_branches)
        self.balance_native_seeds=bool(balance_native_seeds)
        self.calibrate_matching_continuations=bool(calibrate_matching_continuations)
        self.emphasize_verified_turns=bool(emphasize_verified_turns)
        if not 0<=self.native_short_risk_coefficient<=1:raise ValueError('Use a bounded native-success calibration weight')
        if not 0<=self.native_exit_coefficient<=1:raise ValueError('Use a bounded native exit calibration weight')
        self.native_correction=dict(native_correction or {})
        self.correction=None
        if self.native_correction:
            self.correction=SuccessRehearsal(self.native_correction['manifest'],
                native_short_risk_coefficient=native_short_risk_coefficient,
                calibrate_matching_continuations=self.native_correction.get('calibrate_matching_continuations',False),
                emphasize_verified_turns=self.native_correction.get('emphasize_verified_turns',False),
                native_exit_coefficient=self.native_exit_coefficient)
            if any(not verified_correction(r) for r in self.correction.rows):
                raise ValueError('Correction replay requires a native-completed spell, stage or declared road endpoint')
            if not 1<=self.native_correction.get('multiplier',1.)<=3:
                raise ValueError('Bound corrective CE relative to ordinary supervised retention')
        if not 0<=self.priority_fraction<=1 or any(not np.isfinite(v) or not 1<=v<=4 for v in self.phase_priorities.values()):
            raise ValueError('Use bounded verified-phase replay priorities and a valid sampling fraction')
        self.refresh()

    def refresh(self):
        """Admit newly completed native successes only at a segment boundary."""
        from .thrl_study_io import read_shared_json
        rows={}
        for path in [self.path]+self.additional_manifests:
            if path!=self.path and not path.exists():continue
            manifest=read_shared_json(path)
            if manifest.get('schema')!=SCHEMA:raise ValueError('NBNM history manifest required')
            for r in manifest['histories']:
                if r['split']!='train' or not r['nbnm_success'] or r.get('decisions',self.length)<self.length:continue
                proof=r.get('native_validation')
                if proof is not None and (proof.get('contacts')!=0 or proof.get('Bombs')!=0
                        or not (proof.get('stage_clear') or proof.get('task_complete'))):continue
                file=str((path.parent/r['file']).resolve())
                rows[file]=dict(r,file=file)
        self.rows=list(rows.values())
        if not self.rows:raise ValueError('No successful training-split NBNM histories available')
        self.manifest=dict(schema=SCHEMA,histories=self.rows,PPO_rows=0)
        self.groups={}
        for i,r in enumerate(self.rows):self.groups.setdefault(json.dumps(r['task'],sort_keys=True),[]).append(i)
        self.correction_stages={}
        if self.rows and all(verified_correction(r) and r.get('correction_windows') for r in self.rows):
            # Full-stage and finite-road successes can teach the same wave.
            # Balance the stages while keeping the recorded scope of each run.
            for i,r in enumerate(self.rows):
                self.correction_stages.setdefault(r['task']['stage'],[]).append(i)

    def window_start(self,features,row,rng,phases=None):
        """Half uniform retention, half challenging parts of verified successes.

        Unknown forecasts are not safe labels; only the demonstrated action
        on the successful native trajectory is supervised here.
        """
        lower=int(row.get('training_start',0));upper=len(features)-self.length
        self.priority_window=False
        preference=row.get('native_preference') if self.prefer_observed_branches else None
        if preference is not None:
            onset=preference['history_index']
            if type(onset) is not int or not lower<=onset<len(features) or lower>upper:
                raise ValueError('A measured native branch needs a complete supervised window')
            self.priority_window=True
            return min(onset,upper),True
        if row.get('correction_windows'):
            windows=[(max(lower,int(a)),min(upper,int(b)-self.length)) for a,b in row['correction_windows']]
            windows=[(a,b) for a,b in windows if b>=a]
            if windows:
                a,b=windows[int(rng.integers(len(windows)))];self.priority_window=True
                return int(rng.integers(a,b+1)),True
        if self.phase_priorities and phases is not None and rng.random()<self.priority_fraction:
            phases=np.asarray(phases)
            if phases.dtype.kind in ('U','S') and len(phases)<=len(features):
                offset=len(features)-len(phases)
                ids=np.arange(offset,len(features))
                weights=np.array([max(0.,self.phase_priorities.get(str(p),1.)-1.) for p in phases])
                # No Boss detected at a spell's entry/exit is not a road task.
                if row.get('task',{}).get('mode')=='attack':
                    weights[np.array([str(p).endswith('/road') for p in phases])]=0.
                weights[ids<lower]=0.
                if weights.sum()>0:
                    center=int(rng.choice(ids,p=weights/weights.sum()))
                    first=int(rng.integers(max(lower,center-self.length+1),min(center,upper)+1))
                    self.priority_window=True
                    return first,False
        challenging=False
        if self.challenging_fraction and rng.random()<self.challenging_fraction:
            from .compact_spiking_sensation import FEATURE_NAMES
            danger=features[:,FEATURE_NAMES.index('danger_hint_4f_action0'):FEATURE_NAMES.index('forecast4_known')]
            trust=features[:,FEATURE_NAMES.index('forecast4_known'):FEATURE_NAMES.index('forecast24_known')+1]
            candidates=np.flatnonzero((danger.max(-1)>0)|(trust.min(-1)<.5))
            candidates=candidates[(candidates>=lower)&(candidates<len(features))]
            if len(candidates):
                center=int(rng.choice(candidates));first=int(rng.integers(max(lower,center-self.length+1),min(center,upper)+1))
                return first,True
        return int(rng.integers(lower,upper+1)),challenging

    def group_probabilities(self,groups):
        tasks=[self.rows[g[0]]['task'] for g in groups];modes=[t['mode'] for t in tasks]
        weights=(np.full(len(groups),1/len(groups)) if not self.scope_weights else
            np.asarray([self.scope_weights.get(mode,1)/modes.count(mode) for mode in modes],float))
        if self.attack_spell_fraction is not None and 'attack' in modes:
            spells=np.array([t['mode']=='attack' and (t.get('spell_id') is not None or 'Card' in t.get('attack','')) for t in tasks])
            other=np.array([t['mode']=='attack' for t in tasks])&~spells
            if spells.any() and other.any():
                mass=weights[spells|other].sum()
                weights[spells]=mass*self.attack_spell_fraction/spells.sum()
                weights[other]=mass*(1-self.attack_spell_fraction)/other.sum()
        # Prioritize actual failure cards inside their existing stratum; do not
        # steal the stage/chain or nonspell retention allocation.
        if self.phase_priorities:
            strata=[(t['mode'],'Card' in t.get('attack','') or t.get('spell_id') is not None) for t in tasks]
            for stratum in set(strata):
                ids=np.array([i for i,s in enumerate(strata) if s==stratum]);mass=weights[ids].sum()
                priority=np.array([self.phase_priorities.get(f"s{tasks[i].get('stage')}/{tasks[i].get('attack')}",1.)
                    if tasks[i]['mode']=='attack' else 1. for i in ids])
                weighted=weights[ids]*priority
                if weighted.sum()>0:weights[ids]=mass*weighted/weighted.sum()
        return weights/weights.sum()

    def loss(self, model, *, index, task_key=None):
        return next(self.iter_losses(model,index=index,task_key=task_key))

    def _select_row(self, index, reserved, windows):
        # Equal task selection prevents the easy card filling the archive.
        rng=np.random.default_rng(index)
        groups=list(self.groups.values())
        if reserved:
            pool = reserved
        elif self.scope_weights or self.phase_priorities:
            pool=groups[int(rng.choice(len(groups),p=self.group_probabilities(groups)))]
        else:
            if self.correction_stages:groups=list(self.correction_stages.values())
            pool=groups[index%len(groups)]
        member=int(rng.integers(len(pool)))
        if ((windows > 1 or self.balance_native_seeds) and self.correction_stages
                and not (reserved or self.scope_weights or self.phase_priorities)):
            step = index // len(groups)
            if self.balance_native_seeds:
                by_seed = {}
                for offset in pool:
                    by_seed.setdefault(self.rows[offset]['native_seed'], []).append(offset)
                choices = list(by_seed.values())
                pool = choices[step % len(choices)]
                step //= len(choices)
            member = step % len(pool)
        return self.rows[pool[member]],rng

    def _window_starts(self, features, row, rng, phases, windows):
        first, challenging = self.window_start(features, row, rng, phases)
        starts = (continuation_window_starts(first, self.length, len(features), row, windows, rng)
                  if windows > 1 else [first])
        return starts, challenging

    def fixed_replay_histories(self, requests):
        """Resolve the exact window prefixes for (rehearsal index, window count) requests."""
        if self.correction is not None:
            raise ValueError('Fixed prefix requests must select their rehearsal pools explicitly')
        histories={}
        archives={}
        for index, windows in requests:
            row,rng=self._select_row(index,(),windows)
            path=row['file']
            if path not in archives:
                with np.load(path,allow_pickle=False) as archive:
                    archives[path]=(archive['features'],archive['actions'],
                                    archive['phase'] if 'phase' in archive else None)
            features,actions,phases=archives[path]
            starts,_=self._window_starts(features,row,rng,phases,windows)
            if not cacheable_window_prefixes(starts):
                continue
            for start in starts:
                histories[(path,start)]=dict(features=features[:start],actions=actions[:start])
        return histories

    def fixed_reference_prefixes(self, model, indices):
        from .nbnm_prefix import fixed_history_snapshots
        return fixed_history_snapshots(model,self.fixed_replay_histories((index,1) for index in indices))

    def iter_losses(self, model, *, index, task_key=None, windows=1, fixed_prefixes=None, movement_activity=None):
        """Build one backward graph at a time, sharing current-weight prefix memory within the update."""
        reserved = [i for i, row in enumerate(self.rows) if row.get('task_key') == task_key] if task_key else []
        # A uniformly sampled index splits verified correction and ordinary
        # retention equally. Keep one whole-brain backward graph in memory.
        if not reserved and self.correction is not None and index%2==0:
            multiplier=float(self.native_correction.get('multiplier',1.))
            for loss,info in self.correction.iter_losses(model,index=index//2,windows=windows):
                info.update(rehearsal_native_correction=True,rehearsal_loss_multiplier=multiplier)
                yield loss*multiplier,info
            return
        if movement_activity is not None and (self.native_short_risk_coefficient or self.native_exit_coefficient):
            raise ValueError('Cached movement activity requires a movement-only rehearsal loss')
        from .nbnm_prefix import burn_in
        row,rng=self._select_row(index,reserved,windows)
        with np.load(self.path.parent/row['file'],allow_pickle=False) as z:
            f=z['features'].copy();actions=z['actions'].copy()
            reference=z['reference_logits'].copy() if self.match_reference_scores else None
            phases=z['phase'].copy() if 'phase' in z else None
            trace_frames=z['frame'].copy() if self.emphasize_verified_turns and 'frame' in z else None
            measured={k:z[k].copy() for k in (
                'native_frames','native_action_frames','observed_contacts','teacher_override') if k in z}
        selected,challenging=self._window_starts(f,row,rng,phases,windows)
        first=selected[0]
        onset_priority=self.priority_window
        last=selected[-1]
        cached = fixed_prefixes is not None and cacheable_window_prefixes(selected)
        if cached:
            states=[fixed_prefixes[(row['file'],start)] for start in selected]
        else:
            saved=dict.fromkeys(selected) if len(selected)>1 else None
            state=burn_in(model,dict(policy_burn_in=dict(features=f[:last],actions=actions[:last])),
                          model.initial_state(1),snapshots=saved)
            states=[state] if saved is None else [saved[start] for start in selected]
        for start,state in zip(selected,states,strict=True):
            loss,info=self._sequence_loss(model,f,actions,row,start,state,
                phases,trace_frames,measured,reserved,challenging and start==first,onset_priority and start==first,
                reference=reference,movement_activity=movement_activity)
            if cached:
                info['rehearsal_prepared_prefix']=True
            if windows>1:
                info.update(rehearsal_windows_in_update=len(selected),rehearsal_shared_prefix_steps=last)
            yield loss,info
            del loss

    def _sequence_loss(self,model,f,actions,row,first,state,phases,trace_frames,measured,reserved,challenging,priority,
                       *,reference=None,movement_activity=None):
        import torch
        from .compact_story_model import sequence

        end=first+self.length
        features=torch.as_tensor(f[first:end],device=model.device)[:,None]
        # Archive integer widths vary; CE targets and gather indices require int64.
        chosen=torch.as_tensor(actions[first:end],device=model.device,dtype=torch.long)[:,None]
        starts=torch.zeros((self.length,1),device=model.device,dtype=torch.bool)
        if first==0:starts[0]=True
        reused_activity = False
        if movement_activity is None:
            out=sequence(model,features,starts,state,chosen,checkpointed=True)
        else:
            from .nbnm_fixed_movement_activity import (
                capture_movement_activity, movement_logits_from_activity,
            )
            key=(row['file'],first)
            reused_activity=key in movement_activity
            if reused_activity:
                activity=movement_activity[key]
            else:
                activity=capture_movement_activity(model,features,starts,chosen,state)
                movement_activity[key]=[(dn.cpu(),None if pooled is None else pooled.cpu())
                                        for dn,pooled in activity]
            out=dict(logits=movement_logits_from_activity(model,features,chosen,activity))
        movement_scores=out['logits'][...,:18].reshape(-1,18)
        planned_turns=0
        if reference is not None:
            target=torch.as_tensor(reference[first:end],device=model.device,dtype=movement_scores.dtype)
            if target.shape!=movement_scores.shape or not bool(torch.isfinite(target).all()):
                raise ValueError('Reference retention needs 18 finite scores for each playable command')
            delta=movement_scores-target
            # A common shift leaves the original preference ordering unchanged.
            movement_loss=(delta-delta.mean(-1,keepdim=True)).square().sum(-1).mean()
        elif trace_frames is not None and verified_correction(row) and row.get('task',{}).get('mode')=='attack':
            weights=verified_turn_weights(f,actions,trace_frames,phases,first,self.length,row.get('training_start',0))
            planned_turns=int((weights>1).sum())
            weights=torch.as_tensor(weights,device=model.device)
            per_command=torch.nn.functional.cross_entropy(movement_scores,chosen.reshape(-1),reduction='none')
            movement_loss=(per_command*weights).sum()/weights.sum()
        elif 'teacher_override' in measured and self.include_supported_native_exits:
            from .nbnm_teacher_outcomes import verified_short_mask
            observed=verified_short_mask(measured,row,len(f),first,end,f[first:end],
                getattr(model,'nbnm',{}).get('native_rule_geometry'))
            repaired=observed & measured['teacher_override'][first:end]
            planned_turns=int(repaired.sum())
            weights=torch.as_tensor(1.+2.*repaired,device=model.device,dtype=movement_scores.dtype)
            per_command=torch.nn.functional.cross_entropy(movement_scores,chosen.reshape(-1),reduction='none')
            movement_loss=(per_command*weights).sum()/weights.sum()
        else:
            movement_loss=torch.nn.functional.cross_entropy(movement_scores,chosen.reshape(-1))
        loss=movement_loss;risk_rows=0;false_alarm_rows=0;risk_loss=loss.detach()*0
        measured_pair=row.get('native_preference') if self.prefer_observed_branches else None
        paired_row=0
        if measured_pair is not None and first<=measured_pair['history_index']<end:
            offset=measured_pair['history_index']-first
            preferred=measured_pair['preferred_action']
            compared=measured_pair['comparison_action']
            if not (verified_correction(row) and 0<=preferred<18 and 0<=compared<18
                    and preferred!=compared and int(actions[first+offset])==preferred):
                raise ValueError('The native preference must match the recorded successful command')
            # Rank two observed continuations at their shared state. Other
            # actions have no counterfactual outcome from this branch.
            gap=movement_scores[offset,preferred]-movement_scores[offset,compared]
            loss=loss+torch.nn.functional.softplus(-gap)
            paired_row=1
        future_verified=[0,0];future_alarms=[0,0]
        if self.native_short_risk_coefficient:
            from .nbnm_teacher_outcomes import verified_short_mask
            verified=verified_short_mask(measured,row,len(f),first,end,f[first:end],
                getattr(model,'nbnm',{}).get('native_rule_geometry'))
            keep=torch.as_tensor(verified,device=model.device);risk_rows=int(keep.sum())
            if risk_rows:
                selected_by_horizon=out['horizon_scores'].reshape(-1,18,3)[
                    torch.arange(len(chosen),device=model.device),chosen.reshape(-1)]
                selected=selected_by_horizon[keep,0]
                logits=8*selected-1
                # Success-only replay corrects false alarms; it must not keep
                # driving already-low risks toward zero. Real contact-positive
                # calibration remains in the balanced on-policy native loss.
                false_alarm=(logits.detach().sigmoid()>.1).to(logits.dtype)
                false_alarm_rows=int(false_alarm.sum())
                risk_loss=(torch.nn.functional.binary_cross_entropy_with_logits(logits,
                    torch.zeros_like(logits),reduction='none')*false_alarm).sum()/false_alarm.sum().clamp_min(1)
                if self.calibrate_matching_continuations:
                    from .nbnm_teacher_outcomes import matched_native_futures
                    # Align the measured suffix to the complete student/teacher
                    # recurrent history, excluding unmeasured prefix commands.
                    clock=np.full(len(verified),-1,dtype=np.int64)
                    offset=len(f)-len(measured['native_frames'])
                    measured_indices=np.flatnonzero(verified)
                    clock[measured_indices]=measured['native_frames'][first+measured_indices-offset]
                    frames=torch.as_tensor(clock,device=model.device)
                    greedy=out['logits'][...,:18].reshape(-1,18).argmax(-1)
                    total_weight=1.
                    for h,commands,weight in ((1,3,.3),(2,6,.1)):
                        witnessed=matched_native_futures(keep,frames,chosen.reshape(-1),greedy,commands)
                        future_verified[h-1]=int(witnessed.sum())
                        values=8*selected_by_horizon[:,h]-1
                        alarms=witnessed & (values.detach().sigmoid()>.1)
                        future_alarms[h-1]=int(alarms.sum())
                        if future_alarms[h-1]:
                            risk_loss=risk_loss+weight*torch.nn.functional.softplus(values[alarms]).mean()
                            total_weight+=weight
                    risk_loss=risk_loss/total_weight
                loss=loss+self.native_short_risk_coefficient*risk_loss
        exit_rows=[0,0]
        if self.native_exit_coefficient and 'corridor_witness_logits' in out:
            from .nbnm_teacher_outcomes import verified_exit_mask
            rows=torch.arange(self.length,device=model.device)
            action_ids=chosen.reshape(-1)
            evidence=verified_exit_mask(measured,row,f,first,end,
                getattr(model,'nbnm',{}).get('native_rule_geometry'))
            predicted=out['corridor_witness_logits'].reshape(self.length,18,2)[rows,action_ids]
            candidates=features[:,0,281:407].reshape(self.length,18,7)
            geometry_status=candidates[rows,action_ids,2:4]
            teach=geometry_status<=0
            if self.include_supported_native_exits:
                # Rehearse the observed safe exit in scenes with competing
                # continuations. No rejected alternative receives a label.
                observed=(candidates[...,6]>0).any(dim=1)
                contested=(candidates[...,2:4]<0).any(dim=1)
                teach=teach|(contested&observed[:,None])
            verified=torch.as_tensor(evidence,device=model.device)&teach
            exit_rows=verified.sum(0).tolist()
            target_error=torch.nn.functional.softplus(-predicted)
            exit_loss=(target_error*verified).sum()/verified.sum().clamp_min(1)
            loss=loss+self.native_exit_coefficient*exit_loss
        live_settings = getattr(model, 'nbnm', {}) or {}
        live_revision = live_settings.get('native_rule_geometry') or 'legacy'
        if live_settings.get('forecast_native_branches'):
            from .th08_native_forecast import FORECAST_REVISION
            live_revision = FORECAST_REVISION
        return loss,dict(rehearsal_movement_loss=float(movement_loss.detach()),rehearsal_rows=self.length,
            rehearsal_matches_reference_scores=reference is not None,
            rehearsal_reused_movement_activity=reused_activity,
            rehearsal_observed_branch_rows=paired_row,
            rehearsal_verified_turn_rows=planned_turns,
            rehearsal_native_correction=False,rehearsal_loss_multiplier=1.,
            rehearsal_geometry_revision=row.get('observation_forecast_revision') or row.get('native_rule_geometry') or 'legacy',
            live_geometry_revision=live_revision,
            archived_geometry_recomputed=False,rehearsal_risk_labels=risk_rows,
            rehearsal_native_short_risk_loss=float(risk_loss.detach()),
            rehearsal_native_exit_labels_48_96=exit_rows,
            rehearsal_native_false_alarm_rows=false_alarm_rows,
            rehearsal_native_long_risk_labels=sum(future_verified),
            rehearsal_matched_native_12_24=future_verified,rehearsal_native_12_24_false_alarms=future_alarms,
            rehearsal_executed_probability=float(out['logits'][...,:18].softmax(-1).gather(-1,chosen[...,None]).mean().detach()),
            rehearsal_executed_top1_fraction=float((out['logits'][...,:18].argmax(-1)==chosen).float().mean().detach()),
            rehearsal_PPO_rows=0,rehearsal_task=row['task'],rehearsal_task_key=row.get('task_key'),
            rehearsal_reserved_task=bool(reserved),rehearsal_burn_in=first,rehearsal_challenging=challenging,
            rehearsal_priority_window=priority,rehearsal_native_seed=row.get('native_seed'),
            rehearsal_memory='current-weight complete committed-action history')

    def summary(self):
        return dict(manifest=str(self.path),histories=len(self.rows),tasks=len(self.groups),
            attack_spell_fraction=self.attack_spell_fraction,challenging_fraction=self.challenging_fraction,
            phase_priorities=self.phase_priorities,priority_fraction=self.priority_fraction,
            native_short_risk_coefficient=self.native_short_risk_coefficient,
            native_exit_coefficient=self.native_exit_coefficient,
            native_correction=self.native_correction,
            additional_manifests=list(map(str,self.additional_manifests)),
            PPO_rows=0,brain_topology_changed=False,history='entire prefix, one candidate per action')
