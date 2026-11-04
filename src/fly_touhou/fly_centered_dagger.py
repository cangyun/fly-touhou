"""Student-only native actions, same-observation D95 labels with its own GRU.

New train trajectories, not the old rule-policy label bank. No optimization or
teacher action mixture is implemented. Original native frame stamps are not
available to this policy API and are never inferred from neural counters.
"""
from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path
import time

import numpy as np
import torch

from .fly_centered_native_evaluation import CenteredNativePolicy, native_brain_key
from .thrl_durable_records import write_json
from .thrl_full_story_evaluation import action_probabilities
from .thrl_policy_actions import policy_logits

SCHEMA = 'fly-centered-student-native-D95-DAgger-final19-v1'
D95_SHA = 'd95fe710068483fc81d8525d89bccf72e61d209c7e1bb442ea66d022b9c1d06c'
BODY_SHA = '58e9fd9ba99d6384b94e05a438e8ad7f9199daa47eb32133f8f07edeeab5379f'


def file_sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f, 'sha256').hexdigest()


def train_jobs(seed, episodes, brain_seed):
    from .thrl_training_mode import draw_native_seed
    rows=[]
    for game in range(episodes):
        e,a=map(int,np.random.SeedSequence([seed,game]).generate_state(2))
        rows.append(dict(episode_index=game,environment_seed=e,action_seed=a,
            native_seed=draw_native_seed(np.random.default_rng(e),'train'),
            brain_rng_key=native_brain_key(game,brain_seed),rows=0,complete_episode=False))
    return rows


class DaggerRows:
    def __init__(self, folder, metadata, jobs):
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=False)
        (self.folder/'chunks').mkdir()
        self.index=dict(metadata, schema=SCHEMA,status='collecting',episodes=copy.deepcopy(jobs),chunks=[],rows=0,
            native_frame_available=False,done_representation='terminal_decision_row in complete episode index; no fabricated per-row done',
            teacher_actions=0,teacher_actions_executed=0,optimizer_updates=0,teacher_hidden_or_values_saved=False,
            student_and_teacher_states_independent=True,reset_on_Miss_or_stage=False,
            phi_source='same student.sensory_adapter forward output before trainable input_encoder; no second encoder forward')
        self.pending=[];self.action_hashes=[hashlib.sha256() for _ in jobs]
        self.save()

    def save(self):write_json(self.folder/'index.json',self.index)

    def append(self, row):
        game,rowno=int(row['episode_index']),int(row['source_row'])
        episode=self.index['episodes'][game]
        if (rowno!=episode['rows'] or row['noise_counter_after']!=100*(rowno+1)
                or row['brain_rng_key']!=episode['brain_rng_key'] or row['native_seed']!=episode['native_seed']
                or row['environment_seed']!=episode['environment_seed'] or row['episode_starts']!=(rowno==0)):
            raise ValueError('DAgger original episode/own brain counter continuity differs')
        if self.pending and self.pending[0]['episode_index']!=game:self.flush()
        self.pending.append(row);episode['rows']+=1;self.index['rows']+=1
        self.action_hashes[game].update(bytes([int(row['student_action'])]))
        if len(self.pending)==32:self.flush()

    def flush(self):
        if not self.pending:return
        float_keys=('dn_features','phi','teacher_final_logits','teacher_probabilities')
        bool_keys=('legal_action_mask','episode_starts')
        arrays={k:np.asarray([r[k] for r in self.pending],dtype=np.float32 if k in float_keys else np.bool_ if k in bool_keys else np.int64)
                for k in self.pending[0]}
        n=len(self.pending);episode=int(arrays['episode_index'][0]);start=int(arrays['source_row'][0])
        shapes={'dn_features':(n,1299),'phi':(n,9721),'teacher_final_logits':(n,19),
                'teacher_probabilities':(n,19),'legal_action_mask':(n,19)}
        for k,v in arrays.items():
            if v.shape!=shapes.get(k,(n,)):raise ValueError('DAgger payload shape differs:'+k)
            if k!='teacher_final_logits' and not np.isfinite(v).all():raise ValueError('Nonfinite DAgger payload:'+k)
        legal=arrays['legal_action_mask'];logits=arrays['teacher_final_logits'];p=arrays['teacher_probabilities']
        if (not legal[:,:18].all() or not np.isfinite(logits[legal]).all() or not np.isneginf(logits[~legal]).all()
                or np.any(p<0) or np.any(p[~legal]!=0) or not np.allclose(p.sum(-1),1,rtol=0,atol=2e-6)
                or np.any(arrays['student_action']<0) or np.any(arrays['student_action']>=19)
                or not np.array_equal(arrays['source_row'],np.arange(start,start+n))
                or not np.all(arrays['episode_index']==episode)):
            raise ValueError('Invalid final19 legal distribution/greedy action/row alignment')
        relative='chunks/'+str(len(self.index['chunks'])).zfill(6)+'.npz';path=self.folder/relative
        with path.open('xb') as f:np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
        self.index['chunks'].append(dict(name=relative,bytes=path.stat().st_size,sha256=file_sha(path),
            rows=n,episode_index=episode,source_row_start=start))
        self.pending=[];self.save()

    def reconcile_complete(self, report):
        self.flush()
        if len(report['episode_results'])!=len(self.index['episodes']):raise ValueError('Require all natural terminal games')
        for row,episode,action_hash in zip(report['episode_results'],self.index['episodes'],self.action_hashes):
            if (row['evaluation_game']!=episode['episode_index']+1 or row['evaluation_seed']!=episode['environment_seed']
                    or row['native_seed']!=episode['native_seed'] or row['action_seed']!=episode['action_seed']
                    or row['evaluation_decisions']!=episode['rows'] or row['action_sha256']!=action_hash.hexdigest()
                    or row['outcome'] not in ('clear','gameover') or row.get('full_story_episode') is not True):
                raise ValueError('Native terminal seed/action/collected-row identity differs')
            episode.update(complete_episode=True,outcome=row['outcome'],native_frames=row['native_frames'],
                terminal_decision_row=episode['rows']-1,executed_student_action_sha256=action_hash.hexdigest(),action_sha256=action_hash.hexdigest())
        self.index['status']='completed_three_way_aligned_collection';self.save()
        write_json(self.folder/'COMMITTED.json',dict(schema=SCHEMA,index_sha256=file_sha(self.folder/'index.json'),
            student_checkpoint_sha256=self.index['student_checkpoint_sha256'],teacher_checkpoint_sha256=self.index['teacher_checkpoint_sha256'],
            rows=self.index['rows'],episodes=len(self.index['episodes']),complete_episodes=[r['episode_index'] for r in self.index['episodes']],teacher_actions=0,optimizer_updates=0))


class LabelOnlyDaggerPolicy(CenteredNativePolicy):
    def __init__(self, student, teacher, writer, *, episodes, brain_seed=613711, check_stop=lambda:None,
                 synthetic_fixture=False):
        super().__init__(student,episodes=episodes,brain_seed=brain_seed,synthetic_fixture=synthetic_fixture)
        if teacher.gru_hidden_size!=256:raise ValueError('D95 must retain its own GRU256')
        self.label_teacher=teacher;self.teacher_hidden=None;self.writer=writer;self.check_stop=check_stop
        self.teacher_forwards=0;self._phi=None
        self._phi_hook=student.sensory_adapter.register_forward_hook(self._capture_phi)

    def _capture_phi(self,module,args,output):self._phi=output.detach()

    def close(self):self._phi_hook.remove();self._phi=None

    @torch.no_grad()
    def forward_step(self,features,maps,cursor,*,objects):
        self.check_stop();new=cursor.item()==0;self._phi=None
        logits,_,following_cursor=super().forward_step(features,maps,cursor,objects=objects)
        if new:self.teacher_hidden=torch.zeros((1,1,256),dtype=torch.float32,device=features.device)
        # Only labels use this final risk-wrapped teacher. Its state never enters the student.
        final,_,teacher_next=self.label_teacher.forward_step(features,maps,self.teacher_hidden,objects=objects)
        self.teacher_forwards+=1
        if tuple(teacher_next.shape)!=(1,1,256) or not bool(torch.isfinite(teacher_next).all()):
            raise ValueError('D95 own continuous GRU state differs')
        self.teacher_hidden=teacher_next.detach()
        clean=policy_logits(final);p=torch.softmax(clean,-1);legal=torch.isfinite(clean)
        expected=torch.ones_like(legal);expected[:,18]=features[:,4]>0
        if (clean.shape!=(1,19) or not torch.equal(legal,expected) or self._phi is None
                or self._phi.shape!=(1,9721) or not bool(torch.isfinite(p).all())):
            raise ValueError('Same-observation fixed Phi/final19 inventory labels differ')
        action=int(action_probabilities(torch,logits,.25)[0].argmax().item())
        game=self.started_games-1;job=self.writer.index['episodes'][game]
        row=dict(dn_features=self.student.brain.readout_features(self.state)[0].cpu().numpy().copy(),
            phi=self._phi[0].cpu().numpy().copy(),teacher_final_logits=clean[0].cpu().numpy().copy(),
            teacher_probabilities=p[0].cpu().numpy().copy(),legal_action_mask=legal[0].cpu().numpy().copy(),
            student_action=action,episode_index=game,source_row=self.episode_records[-1]['decisions']-1,
            episode_starts=new,native_seed=job['native_seed'],environment_seed=job['environment_seed'],
            brain_rng_key=int(self.state.rng_keys.item()),noise_counter_after=int(self.state.noise_counter.item()))
        self.writer.append(row)
        return logits,None,following_cursor


def collect(student, teacher, config, output, *, episodes, seed, brain_seed=613711, provenance,
            check_stop=lambda:None,progress=None):
    from .thrl_training_mode import validate_training_mode,episode_protocol
    from .thrl_full_story_evaluation import evaluate
    import train_thrl_native as native_entry
    if not validate_training_mode(config) or config.get('headless') is not True or config.get('input_schema')!='semantic-separated-observed-laser-segments-v5':
        raise ValueError('Formal private V5/full-story configuration required')
    output=Path(output);jobs=train_jobs(seed,episodes,brain_seed)
    metadata=dict(provenance,native_split='train',seed=seed,brain_seed=brain_seed,
        student_action_selection='shared evaluator greedy/clamp/stock rule at T.25',teacher_label_temperature=1.,
        teacher_GRU_continuous_from_opening=True,student_full_own_brain_continuous_from_opening=True)
    writer=DaggerRows(output/'data',metadata,jobs)
    policy=LabelOnlyDaggerPolicy(student,teacher,writer,episodes=episodes,brain_seed=brain_seed,check_stop=check_stop).eval()
    local=copy.deepcopy(config);local['movement_guidance']=dict(local.get('movement_guidance',{}),online_teacher_coefficient=0.)
    kwargs=native_entry.make_env_kwargs(local,output/'games',1,'train')
    if (len(kwargs)!=1 or kwargs[0]['full_story_split']!='train' or not kwargs[0]['headless']
            or kwargs[0]['policy_action_count']!=19 or kwargs[0]['frame_skip']!=4
            or kwargs[0]['wall_time_limit_seconds'] is not None or 'online_teacher_revision' in kwargs[0]):
        raise ValueError('One train-split private native worker without teacher action bridge required')
    begun=time.monotonic()
    try:
        result=evaluate(torch,policy,kwargs,output/'games',episodes=episodes,seed=seed,check_stop=check_stop,
            summary=native_entry.development_summary,progress=progress,native_split='train',
            episode_protocol=dict(episode_protocol(local),neural_reset='fresh original episode only',
                teacher_role='D95 label only on student trajectories; independent continuous GRU'),
            action_selection='greedy',action_temperature=.25)
        writer.reconcile_complete(result)
        return dict(schema=SCHEMA,status='completed',rows=writer.index['rows'],evaluation=result,
            data_neural_forwards=policy.data_forwards,teacher_label_forwards=policy.teacher_forwards,
            teacher_actions=0,optimizer_updates=0,seconds=time.monotonic()-begun,
            data_index_sha256=file_sha(writer.folder/'index.json'),data_commit_sha256=file_sha(writer.folder/'COMMITTED.json'))
    except BaseException as error:
        writer.flush();writer.index.update(status='interrupted_or_failed',error=repr(error),
            student_data_forwards=policy.data_forwards,teacher_label_forwards=policy.teacher_forwards)
        writer.save();raise
    finally:policy.close()
