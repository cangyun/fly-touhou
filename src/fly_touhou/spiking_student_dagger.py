"""SNN-owned chronological DAgger acquisition, not PPO or teacher control.

This module starts nothing on import. The live learner and acquisition must
run serially. Original raw maps are streamed losslessly in existing ZIP1 form.
"""
from pathlib import Path
import hashlib
import os
import time
import zipfile

import numpy as np

from .thrl_durable_records import write_json


class StreamingRawMaps:
    """Same original FP32 archive; buffer sixteen real time rows, not a rollout."""
    def __init__(self, target, worlds):
        self.target=Path(target).resolve();self.temporary=self.target.with_name(self.target.name+'.tmp')
        self.index_path=self.target.with_suffix('.zip.index.json')
        if not 1<=worlds<=30 or any(p.exists() for p in (self.target,self.temporary,self.index_path)):
            raise ValueError('New original-map archive and1..30 worlds required')
        self.target.parent.mkdir(parents=True,exist_ok=True)
        self.worlds=worlds;self.rows=self.buffered=0;self.records=[]
        self.buffer=np.empty((16,worlds,16,92,96),np.float32)
        self.digest=hashlib.sha256();self.started=time.monotonic()
        self.archive=zipfile.ZipFile(self.temporary,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True)

    def append(self, maps):
        if maps.shape!=(self.worlds,16,92,96) or maps.dtype!=np.float32 or not np.isfinite(maps).all():
            raise ValueError('Exact finite current original FP32 maps required')
        self.buffer[self.buffered]=maps;self.buffered+=1
        if self.buffered==16:self.flush()

    def flush(self):
        if not self.buffered:return
        n=self.buffered;self.digest.update(memoryview(self.buffer[:n]).cast('B'))
        for w in range(self.worlds):
            block=np.ascontiguousarray(self.buffer[:n,w]);name=f'maps/time{self.rows:03d}-world{w:02d}.npy'
            with self.archive.open(name,'w',force_zip64=True) as stream:np.lib.format.write_array(stream,block,allow_pickle=False)
            info=self.archive.getinfo(name)
            self.records.append(dict(name=name,time_start=self.rows,world=w,rows=n,shape=list(block.shape),dtype=block.dtype.str,
                raw_bytes=block.nbytes,raw_sha256=hashlib.sha256(memoryview(block).cast('B')).hexdigest(),
                npy_bytes=info.file_size,compressed_bytes=info.compress_size))
        self.rows+=n;self.buffered=0

    def finish(self):
        self.flush();self.archive.close()
        with self.temporary.open('r+b') as stream:os.fsync(stream.fileno())
        os.replace(self.temporary,self.target)
        with self.target.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        receipt=dict(schema='full-sensory-original-FP32-map-ZIP1-trace-v1',status='complete',archive=str(self.target),
            archive_sha256=digest,archive_bytes=self.target.stat().st_size,original_shape=[self.rows,self.worlds,16,92,96],
            dtype=np.dtype(np.float32).str,raw_bytes=self.rows*self.worlds*16*92*96*4,raw_sha256=self.digest.hexdigest(),
            chunk_frames=16,member_layout='time-block then individual world;no padding',chunks=len(self.records),members=self.records,
            compression='DEFLATE',compression_level=1,map_quantization=False,training_source='actual SNN student before observations',
            wall_seconds=time.monotonic()-self.started,neural_forwards=0,optimizer_updates=0)
        write_json(self.index_path,receipt);return receipt

    def abort(self):
        # Preserve incomplete traces for diagnosis; never publish a complete index.
        self.archive.close()


class SpikingDAggerCollector:
    """Thirty real worlds. Student acts; observation-bound ResourceV10 labels only."""
    def __init__(self, model, vector, output, *, seed=202610306):
        import torch
        from .spiking_engineered_control import SpikingNativePolicy
        self.model=model;self.policy=SpikingNativePolicy(model).eval();self.vector=vector;self.output=Path(output)
        self.observation=None;self.infos=None;self.starts=np.ones(30,bool)
        self.games=list(range(30));self.next_game=30;self.steps=[0]*30;self.decisions=0
        self.cursor=torch.zeros(1,30,1,device=model.device)
        self.generators=[torch.Generator(device=model.device).manual_seed(seed+w) for w in range(30)]

    def start(self):
        self.observation,self.infos=self.vector.reset()
        if len(self.infos)!=30:raise ValueError('Thirty original training worlds required')

    def collect(self,number,*,steps=256,check_stop=lambda:None,on_progress=lambda row:None):
        import torch
        from .fly_visual_task_collector import _rule_labels_from_source
        from .thrl_movement_learning import transition_outcome
        if self.observation is None or not 1<=steps<=256:raise ValueError('Started collector and finite acquisition block required')
        rows=[];completed=[];trace=self.output/'traces';trace.mkdir(parents=True,exist_ok=True)
        archive=StreamingRawMaps(trace/f'rollout-{number:03}-raw-maps.zip',30)
        try:
            with torch.no_grad():
                for t in range(steps):
                    check_stop();before=self.observation;source_infos=[dict(i) for i in self.infos]
                    rule=_rule_labels_from_source(source_infos);source_weights=np.asarray(rule[1],np.float32)
                    weights=np.where(np.asarray(rule[0])[:,:18].sum(-1)>0,source_weights,0.).astype(np.float32)
                    archive.append(before['maps']);actions=np.empty(30,np.int64);DNs=[]
                    for workers in (list(range(16)),list(range(16,30))):
                        self.policy.set_native_episode_context([(w,self.games[w]) for w in workers])
                        logits,_,following=self.policy.forward_step(before['features'][workers],before['maps'][workers],
                            self.cursor[:,workers],objects=before['objects'][workers])
                        self.cursor[:,workers]=following;probability=logits.softmax(-1)
                        for j,w in enumerate(workers):
                            # A fixed mixed acquisition distribution, never a PPO behavior policy.
                            actions[w]=int(logits[j].argmax()) if w<24 else int(torch.multinomial(probability[j],1,generator=self.generators[w])[0])
                        DNs.append(self.policy.last_policy_rows['DN_activation'].cpu().numpy().copy())
                    if np.any(actions>=18):raise ValueError('Bomb cannot enter NoBomb DAgger')
                    row=dict(features=before['features'].copy(),objects=before['objects'].copy(),actions=actions.copy(),
                        episode_starts=self.starts.copy(),game_ids=np.asarray(self.games,np.int64),
                        native_episodes=np.asarray([i['native_episode'] for i in source_infos],np.int64),
                        episode_steps=np.asarray(self.steps,np.int64),rule_probabilities=np.asarray(rule[0],np.float32).copy(),
                        rule_weights=weights,DN_features=np.concatenate(DNs))
                    after,_,terminated,truncated,infos=self.vector.step(actions.tolist())
                    terminal=np.asarray(terminated,bool)
                    if terminal.shape!=(30,) or np.asarray(truncated,bool).any():raise ValueError('No artificial episode termination allowed')
                    for w,info in enumerate(infos):
                        if (info.get('policy_action')!=int(actions[w]) or info.get('bombs_used')!=0
                                or info.get('teacher_actions_selected_by_worker')!=0 or info.get('safety_takeovers')!=0
                                or info.get('avoidance_task_schema')!='continuous-NoBomb-first-native-contact-training-task-v1'
                                or info.get('full_story_seed_split')!='train' or not info.get('game_rng_seed_applied')):
                            raise ValueError('Original student action, train seed, NoBomb or teacher-only annotation differs')
                        if terminal[w]:completed.append(dict(worker=w,game_id=self.games[w],summary=dict(info['avoidance_episode'])))
                    facts=[dict(info,terminated=bool(terminal[w]),truncated=False) for w,info in enumerate(infos)]
                    row.update(task_terminated=terminal.copy(),task_outcomes=np.asarray([
                        transition_outcome(row['features'][w],source_infos[w],int(actions[w]),after['features'][w],facts[w]) for w in range(30)],np.int8))
                    rows.append(row);self.decisions+=30;self.steps=[n+1 for n in self.steps]
                    self.observation,self.infos,self.starts=after,list(infos),np.zeros(30,bool)
                    done=np.flatnonzero(terminal).tolist()
                    if done:
                        reset_infos=self.vector.reset_at(done);self.observation=self.vector.observation
                        for w,info in zip(done,reset_infos):
                            self.infos[w]=info;self.games[w]=self.next_game;self.next_game+=1;self.steps[w]=0
                            self.starts[w]=True;self.cursor[:,w]=0
                    if (t+1)%16==0:on_progress(dict(phase='SNN_student_own_states',rollout=number,real_steps=t+1,student_actions=self.decisions))
            receipt=archive.finish()
        except BaseException:
            archive.abort();raise
        arrays={key:np.stack([r[key] for r in rows]) for key in rows[0]}
        path=trace/f'rollout-{number:03}.npz';temporary=path.with_suffix('.npz.tmp')
        with temporary.open('wb') as file:np.savez_compressed(file,**arrays);file.flush();os.fsync(file.fileno())
        os.replace(temporary,path)
        write_json(path.with_suffix('.json'),dict(raw_map_archive=receipt,student_actions=steps*30,teacher_actions=0,
            optimizer_updates=0,PPO_samples=False,completed_episodes=completed,recording_budget_is_not_an_episode_cap=True,
            full_neural_history_carried_across_files=True,actor_action_selection='student-greedy24-explore6'))
        return dict(student_actions=steps*30,completed_episodes=len(completed))
