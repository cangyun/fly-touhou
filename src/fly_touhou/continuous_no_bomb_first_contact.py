"""Uncapped original-game avoidance comparison;no job starts on import.

Reuse the existing full-story native environment,source19 movement mapping,
state4/Miss contact evidence and ready-vector protocol. There is no simulator,
teacher action,optimizer,frame cap or synthetic successful episode.
"""
from __future__ import annotations
from collections import deque
from pathlib import Path
import hashlib
import json
import os
import time

from .thrl_full_story_env import THRLFullStoryEnv
from .thrl_avoidance_probe import collision_observed

SCHEMA='th14-NoBomb-uncapped-first-observed-contact-v1'


class UncappedFirstContactEnv(THRLFullStoryEnv):
    """Evaluation-only stop at accepted native contact,including the DB window."""
    def reset(self,*,seed=None,options=None):
        if not options or set(options)!={'evaluation_seed'}:raise ValueError('Explicit paired fresh-opening seed required')
        self._avoidance_decisions=0;self._unprotected_decisions=0
        observation,info=super().reset(seed=int(options['evaluation_seed']),options=None)
        if collision_observed(self.snapshot.player,info['misses']):raise ValueError('Opening already contains contact')
        info.update(diagnostic_only=True,diagnostic_schema=SCHEMA)
        return observation,info

    def step(self,action):
        if not 0<=int(action)<18:raise ValueError('NoBomb evaluation accepts source movement0..17 only')
        before=self.snapshot;first_frame=self.episode_frames;previous_misses=self._info()['misses']
        if before.player is not None and before.player.state==1 and before.player.invulnerable_frames<=0:
            self._unprotected_decisions+=1
        observation,reward,terminated,truncated,info=super().step(action)
        self._avoidance_decisions+=1
        if truncated:raise ValueError('No frame/episode time truncation belongs to this comparison')
        misses=info.get('misses')
        if type(misses) is not int or misses<previous_misses or info.get('bombs_used')!=0:
            raise ValueError('Observed Miss counters and exactly zero native Bombs required')
        player=self.snapshot.player
        fresh=info.get('observation_source')=='current-measured-native-state'
        state4=bool(fresh and player is not None and player.state==4)
        hit=state4 or misses>previous_misses
        native=info.get('episode_summary',{})
        clear=bool(terminated and native.get('outcome')=='clear' and not hit)
        if terminated and not hit and not clear:raise ValueError('Native terminal without observed contact or natural clear')
        if hit or clear:
            endpoint=dict(schema=SCHEMA,diagnostic_only=True,
                outcome='first_observed_contact' if hit else 'natural_clear_without_observed_contact',
                first_collision_observed=hit,right_censored_at_natural_clear=clear,
                right_censored_at_cap=False,native_frame_cap=None,
                native_frames=self.episode_frames,first_contact_frames=self.episode_frames if hit else None,
                first_contact_observation_interval=[first_frame+1,self.episode_frames] if hit else None,
                contact_evidence='current_native_state4' if state4 else 'native_Miss_counter_increase' if hit else None,
                contact_is_native_accepted_not_geometric_prediction=True,
                first_contact_time_is_observed_endpoint_not_exact_native_event=True,
                stage=self.snapshot.stage,stage_frame=self.snapshot.stage_frame,stage_history=list(self.stage_history),
                first_stage_without_observed_contact=max(self.stage_history)>=2,
                native_seed=self.native_seed,seed_applied=info.get('game_rng_seed_applied'),
                native_misses=misses,actual_bombs_used=info['bombs_used'],policy_decisions=self._avoidance_decisions,
                unprotected_preaction_checkpoints=self._unprotected_decisions,
                player_state=player.state if player else None,
                deathbomb_remaining=player.deathbomb_timer if player else None,
                natural_clear_claim=clear,formal_full_game_result=clear,
                collision_is_geometric_proxy=False,
                protection_contact_semantics='native accepted state4/Miss;protected overlaps without native event do not terminate',
                native_clear_summary=native if clear else None)
            info['avoidance_episode']=endpoint
            info.update(terminated=True,truncated=False,hit_ends_episode=hit,
                bootstrap_allowed=False,diagnostic_terminal=True)
            self._needs_reset=True
            # Diagnostic terminal only. No training value/return is constructed.
            return observation,reward,True,False,info
        return observation,reward,False,False,info


def first_contact_factory(options):return UncappedFirstContactEnv(**options)


def make_no_bomb_policy(torch,model,*,family,episodes,movement_readout='composed'):
    """One actual source forward;NoBomb is only the final evaluation mask."""
    from .thrl_policy_actions import policy_logits
    if family not in ('deep','d95'):raise ValueError('Explicit frozen family required')
    if movement_readout not in ('composed','raw','risk_only'):
        raise ValueError('Explicit frozen movement readout required')
    if family=='deep':
        from .continuous_connectome_collector import ContinuousNativePolicy
        backend=ContinuousNativePolicy(model,episodes=episodes,neural_batch_size=16).eval()
    else:backend=model

    class NoBombPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__();self.backend=backend;self.family=family
            self.gru_hidden_size=backend.gru_hidden_size;self.context=None;self.last_policy_rows=None
            self.movement_readout=movement_readout
        def set_native_episode_context(self,context):
            self.context=tuple(context)
            if self.family=='deep':self.backend.set_native_episode_context(context)
        def forward_step(self,features,maps,hidden,*,objects):
            if self.context is None:raise ValueError('Actual worker/game context required')
            if self.family=='deep':
                logits,values,following=self.backend.forward_step(features,maps,hidden,objects=objects)
                self.last_policy_rows=dict(self.backend.last_policy_rows)
            else:
                # Exact EntityRiskPolicy path,split only to retain its already
                # computed risk.No second baseGRU/point/laser forward is added.
                raw,values,following=self.backend.base.forward_step(features,maps,hidden,objects=objects)
                risk=self.backend._risk_scores(features,objects)
                logits=self.backend._adjust(raw,features,objects,precomputed_risk=risk,check_finite=True)
                self.last_policy_rows=dict(raw_logits=raw,composed_logits=logits,
                    predicted_costs=torch.nn.functional.softplus(risk),D95_risk_scores=risk)
            if self.movement_readout=='raw':logits=self.last_policy_rows['raw_logits']
            elif self.movement_readout=='risk_only':
                latent=self.last_policy_rows.get('risk_latent_scores',self.last_policy_rows.get('D95_risk_scores'))
                priority=latent if latent is not None else self.last_policy_rows['predicted_costs']
                # softplus can round several safe costs to exactly zero.
                # Its latent scores retain the intended ordering. Centering
                # also avoids the source's final+80 clamp tying large maxima.
                scores=-(priority-priority.amin(-1,keepdim=True))
                logits=torch.cat((scores,scores.new_full((len(scores),1),-torch.inf)),-1)
            clean=policy_logits(logits).clone();clean[:,18]=-torch.inf
            self.last_policy_rows['noBomb_policy_logits']=clean.detach()
            return clean,values,following
        def actual_population_rows(self):
            if self.family!='deep':return None
            current=self.backend.model;ports=current.core.input_indices
            # Actual committed following activity;the evaluator cursor is NOT a brain state.
            return [self.backend.workers[w].state.activity.index_select(1,ports)[0].detach().cpu().numpy()
                    for w,g in self.context]
        def actual_complete_activity_rows(self):
            if self.family!='deep':raise ValueError('Complete continuous activity requires the deep family')
            return [self.backend.workers[w].state.activity[0].detach().cpu().numpy() for w,g in self.context]
    return NoBombPolicy().eval()


def summarize(rows,np):
    values=np.asarray([r['native_frames'] for r in rows],np.float64)
    hit=np.asarray([r['first_collision_observed'] for r in rows],bool)
    contacts=values[hit]
    def distribution(x):
        return dict(count=len(x),mean=float(x.mean()) if len(x) else None,
            median=float(np.median(x)) if len(x) else None,q25=float(np.quantile(x,.25)) if len(x) else None,
            min=float(x.min()) if len(x) else None,max=float(x.max()) if len(x) else None,values=x.astype(int).tolist())
    return dict(episodes=len(rows),first_observed_contacts=int(hit.sum()),
        natural_clear_right_censored=int((~hit).sum()),no_frame_cap=True,
        observed_contact_frames=distribution(contacts),observed_endpoint_frames=distribution(values),
        endpoint_mean_is_not_uncensored_survival_mean=bool((~hit).any()),
        first_stage_without_observed_contact_count=sum(r['first_stage_without_observed_contact'] for r in rows),
        first_stage_without_observed_contact_fraction=float(np.mean([r['first_stage_without_observed_contact'] for r in rows])),
        contact_stage_counts={str(s):sum(r['first_collision_observed'] and r['stage']==s for r in rows) for s in range(1,7)},
        actual_Bombs=sum(r['actual_bombs_used'] for r in rows),episode_results=rows)


def evaluate_no_bomb(torch,policy,environment,output,*,checkpoint_sha256,episodes=20,workers=20,
        seed=2027431601,check_stop=lambda:None,progress=None,worker_cpus=None,history=8,
        opening_activity_rows=0):
    """Same native async workers/seed draw;only terminal protocol and Bomb support differ."""
    import numpy as np
    from . import thrl_vector
    from .thrl_durable_records import write_json,append_jsonl
    from .thrl_inputs import forward_observation
    from .fly_dn_cost_native_policy import make_dn_cost_evaluation_vector
    import train_thrl_native as native_entry
    if not 1<=workers<=min(20,episodes) or not 1<=episodes<=20 or history!=8:
        raise ValueError('Finite cohort<=20,readyB16 and eight-row actual tails required')
    if type(opening_activity_rows) is not int or opening_activity_rows<0 or (opening_activity_rows and policy.family!='deep'):
        raise ValueError('Optional nonnegative initial activity-recording budget requires continuous brain')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    jobs=[]
    for game in range(episodes):
        e,a=np.random.SeedSequence([seed,game]).generate_state(2)
        jobs.append(dict(game=game,environment_seed=int(e),action_seed=int(a)))
    write_json(output/'episode_plan.json',dict(schema=SCHEMA,jobs=jobs,seed=seed,workers=workers,
        action_selection='greedy',action_temperature=1.,Bomb_forbidden=True,native_frame_cap=None,
        movement_readout=getattr(policy,'movement_readout','composed'),
        readout_revision='greedy-logits-and-latent-risk-v2',
        opening_activity_recording_rows=opening_activity_rows,recording_budget_is_not_an_episode_cap=True,
        episode_stop='first native accepted state4/Miss contact;natural clear is right censoring',
        native_split='development',neural_ready_batch_max=16,teacher_actions=0,optimizer_updates=0))
    kwargs=native_entry.make_env_kwargs(environment,output,workers,'development')
    if any('online_teacher_revision' in k or k['frame_skip']!=4 or k['wall_time_limit_seconds'] is not None
           or k['policy_action_count']!=19 for k in kwargs):raise ValueError('Same source19,original four ticks,no teacher/time cap')
    Vector=make_dn_cost_evaluation_vector(worker_cpus,output)
    old_factory=thrl_vector.worker_factory;vector=None;pending=deque(jobs);slots={};observations={};infos={};tails={};openings={}
    device=next(policy.parameters()).device;hidden=torch.zeros(1,workers,policy.gru_hidden_size,device=device)
    results=[];decisions=batches=0;started=time.monotonic();last_progress=0.
    def publish():
        row=dict(schema=SCHEMA,phase='NoBomb_first_contact',completed=len(results),target=episodes,
            student_decisions=decisions,elapsed_seconds=time.monotonic()-started,
            current_environments=[dict(worker=i,game=slots[i]['game'],**{k:v.get(k) for k in
                ('pid','stage','stage_frame','episode_frames','misses','bombs_used')}) for i,v in infos.items() if i in slots])
        write_json(output/'progress.json',row)
        if progress:progress(row)
    def assign(i):
        slots[i]=dict(pending.popleft(),action_sha=hashlib.sha256(),decisions=0)
        tails[i]=deque(maxlen=history);hidden[:,i]=0
        openings[i]=[]
        vector.submit_at([i],'reset',options=[dict(evaluation_seed=slots[i]['environment_seed'])])
    def finish_trace(i,row,after,info):
        rows=list(tails[i]);folder=output/'first_contact_tails';folder.mkdir(exist_ok=True)
        path=folder/f"game{row['evaluation_game']:03}.npz"
        arrays={k:np.stack([r[k] for r in rows]) for k in rows[0]} if rows else {}
        arrays.update({'after_'+k:np.asarray(after[k]).copy() for k in ('features','objects','maps')})
        np.savez_compressed(path,**arrays)
        pointer=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),rows=len(rows),
            source_checkpoint_sha256=checkpoint_sha256,family=policy.family,
            DN_and_VP_activity_available=policy.family=='deep',
            neural_activity_timing='actual following activity from the same observation used for action;not evaluator cursor',
            extra_brain_forwards=0,extra_teacher_queries=0,actual_contact=row['first_collision_observed'],
            before_frames=arrays.get('before_episode_frames',np.asarray([])).tolist(),
            after_frame=info.get('episode_frames'),actions_are_actual=True)
        if openings[i]:
            initial=output/'opening_activity';initial.mkdir(exist_ok=True)
            opening_path=initial/f"game{row['evaluation_game']:03}.npz"
            np.savez_compressed(opening_path,**{k:np.stack([r[k] for r in openings[i]]) for k in openings[i][0]})
            pointer['opening_activity_trace']=dict(path=str(opening_path),rows=len(openings[i]),
                sha256=hashlib.sha256(opening_path.read_bytes()).hexdigest(),
                activity='actual following full continuous state from the same action-producing brain forward',
                complete_from_native_opening=True,recording_budget=opening_activity_rows,
                recording_budget_is_not_an_episode_cap=True,extra_brain_forwards=0)
        write_json(path.with_suffix('.json'),pointer);return pointer
    try:
        check_stop();thrl_vector.worker_factory=first_contact_factory
        vector=Vector(kwargs,output/'workers')
        for i in range(workers):assign(i)
        with torch.no_grad():
            while vector._inflight:
                check_stop();ready=[]
                for message in vector.receive_ready(max_wait_ms=2,min_batch=min(4,workers),check_stop=check_stop):
                    i=message['index'];job=slots[i];after=message['observation'];info=message['info']
                    if message['operation']=='reset':
                        if info.get('game_rng_seed_applied') is not True:raise ValueError('Original seed was not applied')
                        job['native_seed']=info['full_story_native_seed']
                    else:
                        if int(info['policy_action'])!=job['action'] or info.get('bombs_used')!=0:
                            raise ValueError('Acknowledged movement action or NoBomb fact differs')
                        decisions+=1;job['decisions']+=1
                    observations[i]=after;infos[i]=info
                    if message.get('truncated'):raise ValueError('Unexpected external episode truncation')
                    if message.get('terminated'):
                        row=dict(info['avoidance_episode'],evaluation_game=job['game']+1,
                            evaluation_seed=job['environment_seed'],action_seed=job['action_seed'],
                            native_seed=job['native_seed'],evaluation_decisions=job['decisions'],
                            action_sha256=job['action_sha'].hexdigest(),checkpoint_sha256=checkpoint_sha256)
                        try:row['first_contact_tail']=finish_trace(i,row,after,info)
                        except (OSError,ValueError) as error:
                            row['first_contact_tail']=dict(status='diagnostic_capture_unavailable',error=repr(error),actions_changed=False)
                        results.append(row);append_jsonl(output/'episodes.jsonl',row)
                        if pending:assign(i)
                        else:del slots[i]
                        publish()
                    else:ready.append(i)
                if ready:
                    batch={k:np.stack([observations[i][k] for i in ready]) for k in ('features','maps','objects')}
                    context=[(i,slots[i]['game']) for i in ready];policy.set_native_episode_context(context)
                    logits,_,following=forward_observation(torch,policy,batch,hidden[:,ready]);hidden[:,ready]=following
                    probabilities=logits.softmax(-1).detach().cpu().numpy()
                    rows={k:v.detach().cpu().numpy() for k,v in policy.last_policy_rows.items()}
                    VP=policy.actual_population_rows();actions=[]
                    complete=policy.actual_complete_activity_rows() if opening_activity_rows and any(
                        len(openings[i])<opening_activity_rows for i in ready) else None
                    for j,i in enumerate(ready):
                        action=int(np.argmax(rows['noBomb_policy_logits'][j]));job=slots[i]
                        if action>=18 or probabilities[j,18]!=0:raise ValueError('Bomb escaped evaluation-only mask')
                        item={k:batch[k][j].copy() for k in ('features','objects','maps')}
                        item.update(action=np.int64(action),action_probabilities=probabilities[j].copy(),
                            before_episode_frames=np.int64(infos[i]['episode_frames']),stage=np.int16(infos[i]['stage']),
                            game=np.int16(job['game']),native_episode=np.int64(infos[i]['native_episode']))
                        for k in ('raw_logits','composed_logits','predicted_costs','noBomb_policy_logits'):
                            item[k]=rows[k][j].copy()
                        if policy.family=='deep':
                            item['DN_activity']=rows['DN_activation'][j].copy();item['VP_activity']=VP[j].copy()
                            if 'encoded384' in rows:item['encoded384']=rows['encoded384'][j].copy()
                            if 'risk_latent_scores' in rows:item['risk_latent_scores']=rows['risk_latent_scores'][j].copy()
                            for key in ('DN_membrane','DN_spikes','VP_membrane','VP_spikes'):
                                if key in rows:item[key]=rows[key][j].copy()
                        else:item['D95_risk_scores']=rows['D95_risk_scores'][j].copy()
                        if complete is not None and len(openings[i])<opening_activity_rows:
                            openings[i].append(dict(item,brain_activity_following=complete[j].copy()))
                        tails[i].append(item);job['action']=action;job['action_sha'].update(bytes([action]));actions.append(action)
                    vector.submit_at(ready,'step',actions);batches+=1
                if time.monotonic()-last_progress>=5:publish();last_progress=time.monotonic()
        if len(results)!=episodes:raise ValueError('Incomplete comparison is not a score')
        rows=sorted(results,key=lambda r:r['evaluation_game']);report=summarize(rows,np)
        report.update(schema=SCHEMA,status='completed',source_family=policy.family,
            checkpoint_sha256=checkpoint_sha256,episode_plan=jobs,seed=seed,native_split='development',
            action_selection='greedy',action_temperature=1.,Bomb_forbidden=True,policy_action_projection='source19 mask Bomb18;movement0..17 unchanged',
            movement_readout=getattr(policy,'movement_readout','composed'),
            readout_revision='greedy-logits-and-latent-risk-v2',
            baseline_is_not_historical_Bomb_allowed_score=True,first_contact_resolution='acknowledged native decision endpoint;up to four ticks,not exact native event',
            policy_frozen=True,teacher_actions=0,optimizer_updates=0,
            neural_batches=batches,student_decisions=decisions,elapsed_seconds=time.monotonic()-started)
        write_json(output/'report.json',report);return report
    finally:
        thrl_vector.worker_factory=old_factory
        if vector is not None:
            vector.close();write_json(output/'native_cleanup.json',dict(close_errors=vector.close_errors,workers=workers))
            if vector.close_errors:raise RuntimeError('NoBomb native cleanup has unresolved errors')
