"""Training-only replay of real student states from stages three and later.

This augments teacher and risk supervision. It never reuses a historical
action, reward, value estimate, or log probability as an on-policy PPO sample.
"""
from __future__ import annotations

import torch

from .spiking_engineered_control import SpikingState

FIELDS=('features','actions','episode_starts','margins','trusted','unprotected_mask','outcomes',
    'rule_probabilities','rule_weights','deathbomb_mask','native_action_frames')
SENSOR_FLAGS=('native_motion_forecast','native_focus_gate','unknown_laser_player_motion',
    'native_curve_laser_geometry','protected_exit_forecast','primary_attack_target',
    'forecast_growth_motion','anticipate_point_arming','native_contact_risk_calibration')


class LateStageReplay:
    def __init__(self,*,sensor_flags,max_per_stage=32,native_repair_revisions=None):
        self.sensor_flags={key:bool(sensor_flags.get(key,False)) for key in SENSOR_FLAGS}
        if self.sensor_flags['protected_exit_forecast'] or not self.sensor_flags['native_contact_risk_calibration']:
            raise ValueError('Late replay currently requires native-contact calibration and the standard protection interface')
        self.max_per_stage=max_per_stage
        self.native_repair_revisions=native_repair_revisions
        self.incompatible_banks=[]
        self.windows={3:[],4:[]}
        self.sources=[];self.incompatible_sources=[]
        self.source_hold_forecasts={};self.source_hidden_contacts={};self.source_ex8_polar4_forecasts={}
        self.examined=set()

    def add(self,rollout,*,source,source_settings=None):
        """Keep student windows with matching feature flags and record the hold revision."""
        if source in self.examined:return 0
        self.examined.add(source)
        metadata=dict(source_settings or {},**rollout.get('metadata',{}))
        if (self.native_repair_revisions is not None
                and metadata.get('native_repair_revisions')!=self.native_repair_revisions):
            self.incompatible_sources.append(source);return 0
        deterministic_student=(metadata.get('supervision_only') and
            metadata.get('action_selection') in ('family_mode','joint_mode'))
        if (not metadata.get('full_story') or (metadata.get('supervision_only') and not deterministic_student)
                or metadata.get('teacher_actions')!=0
                or any(key not in metadata or bool(metadata[key])!=expected
                       for key,expected in self.sensor_flags.items())):
            self.incompatible_sources.append(source);return 0
        if any(key not in rollout for key in FIELDS):
            self.incompatible_sources.append(source);return 0
        features=rollout['features'];T,N=features.shape[:2]
        stages=(features[...,111]*6).round().long()
        added=0
        for group in (3,4):
            best={}
            for t in range(0,T-15,16):
                for world in range(N):
                    if 'valid_mask' in rollout and not bool(rollout['valid_mask'][t:t+16,world].all()):continue
                    stage=stages[t:t+16,world]
                    mask=(stage==3) if group==3 else (stage>=4)
                    if not bool(mask.any()):continue
                    rule=(rollout['rule_weights'][t:t+16,world]>0)&mask
                    hit=(rollout['outcomes'][t:t+16,world]==1)&mask
                    death=rollout['deathbomb_mask'][t:t+16,world]&mask
                    partial=(rollout['partial_point_danger'][t:t+16,world].any((-1,-2))&mask
                        if 'partial_point_danger' in rollout else torch.zeros_like(mask))
                    if not bool(rule.any() or hit.any() or death.any() or partial.any()):continue
                    valid=rollout['trusted'][t:t+16,world,0]&rollout['unprotected_mask'][t:t+16,world]&mask
                    margins=rollout['margins'][t:t+16,world,0]
                    mixed=valid&(margins.le(0).any(-1))&(margins.gt(1).any(-1))
                    score=(int(hit.any()),int(death.any()),int(mixed.sum()),int(rule.sum()),int(mask.sum()))
                    if world not in best or score>best[world][0]:best[world]=(score,t,mask.clone())
            for world in sorted(best):
                _,t,mask=best[world]
                data={key:rollout[key][t:t+16,world:world+1].detach().cpu().clone() for key in FIELDS}
                data['partial_point_danger']=(rollout['partial_point_danger'][t:t+16,world:world+1].detach().cpu().clone()
                    if 'partial_point_danger' in rollout else torch.zeros_like(data['margins'],dtype=torch.bool))
                data['valid_mask']=mask[:,None].cpu()
                before=rollout['block_states'][t].row(world).detach().to('cpu')
                before=SpikingState(*(value.clone() for value in before.tensors()))
                self.windows[group].append(dict(data=data,before=before,source=source,time=t,world=world))
                added+=1
            self.windows[group]=self.windows[group][-self.max_per_stage:]
        if added:
            self.sources.append(source)
            self.source_hold_forecasts[source]=metadata.get('native_position_hold_forecast')
            self.source_hidden_contacts[source]=metadata.get('native_hidden_enemy_contacts')
            self.source_ex8_polar4_forecasts[source]=metadata.get('native_ex8_polar4_forecast')
        retained={window['source'] for values in self.windows.values() for window in values}
        self.sources=[name for name in self.sources if name in retained]
        self.source_hold_forecasts={name:revision for name,revision in self.source_hold_forecasts.items()
                                    if name in retained}
        self.source_hidden_contacts={name:revision for name,revision in self.source_hidden_contacts.items()
                                     if name in retained}
        self.source_ex8_polar4_forecasts={name:revision for name,revision in self.source_ex8_polar4_forecasts.items()
                                         if name in retained}
        return added

    def loss(self,model,*,index):
        """One window from each available late-stage group, supervised only."""
        from .compact_story_model import sequence
        from .compact_story_ppo import rollout_risk_weights,risk_loss,partial_risk_weights
        from .compact_native_contact_targets import native_contact_weights
        from .fly_gru_aligned_guidance import gru_aligned_guidance_loss
        selected=[values[index%len(values)] for values in self.windows.values() if values]
        if not selected:raise ValueError('No late-stage training windows available')
        data={key:torch.cat([window['data'][key] for window in selected],dim=1).to(model.device)
              for key in (*FIELDS,'valid_mask')}
        data['partial_point_danger']=torch.cat([window['data'].get('partial_point_danger',
            torch.zeros_like(window['data']['margins'],dtype=torch.bool)) for window in selected],1).to(model.device)
        before=SpikingState.pack([window['before'] for window in selected]).to(model.device)
        result=sequence(model,data['features'],data['episode_starts'],before,data['actions'])
        data['unprotected_mask']=data['unprotected_mask']&data['valid_mask']
        data['rule_weights']=data['rule_weights']*data['valid_mask'].to(data['rule_weights'].dtype)
        guide=gru_aligned_guidance_loss(result['logits'],data['margins'],data['trusted'],
            data['actions'],data['outcomes'],valid_mask=data['valid_mask'],
            unprotected_mask=data['unprotected_mask'],rule_probabilities=data['rule_probabilities'],
            rule_weights=data['rule_weights'],deathbomb_mask=data['deathbomb_mask'],veto_rule_on_contact=True)
        reg,cls=rollout_risk_weights(data)
        cls,partial=partial_risk_weights(data,cls)
        from .compact_native_contact_targets import native_exposure_mask
        exposed=native_exposure_mask(data['features'],data['unprotected_mask'])&data['valid_mask']
        weights,counts=native_contact_weights(data['actions'],data['outcomes'],
            exposed,data['native_action_frames'])
        native=dict(actions=data['actions'],outcomes=data['outcomes'],unprotected=exposed,
            elapsed_frames=data['native_action_frames'],weights=weights,share=.25 if bool(weights.any()) else 0.)
        risk=risk_loss(result['horizon_scores'],data['margins'],reg,cls,native=native,partial=partial)
        return guide['total_loss']+risk,dict(late_replay_sequences=len(selected),
            late_replay_teacher_rows=int((data['rule_weights']>0).sum()),
            late_replay_native_contacts=int(counts[1]),late_replay_PPO_rows=0,
            late_replay_partial_point_danger_pairs=int((partial>0).sum()) if partial is not None else 0,
            late_replay_guidance=float(guide['total_loss'].detach()),late_replay_risk=float(risk.detach()))

    def summary(self):
        return dict(stage3_windows=len(self.windows[3]),stage4_plus_windows=len(self.windows[4]),
            source_tapes=self.sources,incompatible_sources=self.incompatible_sources,
            sensor_flags=self.sensor_flags,source_hold_forecasts=self.source_hold_forecasts,
            native_repair_revisions=self.native_repair_revisions,incompatible_banks=self.incompatible_banks,
            source_hidden_contacts=self.source_hidden_contacts,source_ex8_polar4_forecasts=self.source_ex8_polar4_forecasts,
            scope='Student training trajectories;teacher and measured risk supervision only;no historical PPO rows.')

    def payload(self):
        return dict(schema='compact-student-late-stage-supervision-bank-v1',
            sensor_flags=self.sensor_flags,windows=self.windows,source_hold_forecasts=self.source_hold_forecasts,
            native_repair_revisions=self.native_repair_revisions,
            source_hidden_contacts=self.source_hidden_contacts,source_ex8_polar4_forecasts=self.source_ex8_polar4_forecasts,
            scope='Student-controlled training data,never PPO samples')

    def merge_payload(self,payload,*,source=None):
        if payload.get('schema')!='compact-student-late-stage-supervision-bank-v1' or payload.get('sensor_flags')!=self.sensor_flags:
            raise ValueError('Late-stage bank requires the same native sensory interface')
        if (self.native_repair_revisions is not None
                and payload.get('native_repair_revisions')!=self.native_repair_revisions):
            self.incompatible_banks.append(dict(source=source,
                reason='risk and route labels precede the native geometry repair'))
            return 0
        self.source_hold_forecasts.update(payload.get('source_hold_forecasts',{}))
        self.source_hidden_contacts.update(payload.get('source_hidden_contacts',{}))
        self.source_ex8_polar4_forecasts.update(payload.get('source_ex8_polar4_forecasts',{}))
        for stage in (3,4):
            existing={(w['source'],w['time'],w['world']) for w in self.windows[stage]}
            for window in payload['windows'][stage]:
                key=(window['source'],window['time'],window['world'])
                if key not in existing:self.windows[stage].append(window);existing.add(key)
            self.windows[stage]=self.windows[stage][-self.max_per_stage:]
        self.sources=list(dict.fromkeys(w['source'] for stage in (3,4) for w in self.windows[stage]))
        self.source_hold_forecasts={name:self.source_hold_forecasts.get(name) for name in self.sources}
        self.source_hidden_contacts={name:self.source_hidden_contacts.get(name) for name in self.sources}
        self.source_ex8_polar4_forecasts={name:self.source_ex8_polar4_forecasts.get(name) for name in self.sources}
