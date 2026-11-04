"""Thin 16-train/2-fit-held data adapter; metadata import is NoTorch/NoNumPy.

Core, supervision, typed recovery and held metrics remain in the six-lane
module. The only array extension preserves optional *real* 4/12/24 labels.
"""
from __future__ import annotations
import bisect
import copy
import hashlib
import json
from pathlib import Path
from .thrl_dodge_diagnostics import _shared_binary
from .continuous_rule_episode_pretraining import (
    RuleEpisodeReader, assemble_window, supervised_forward_window, scalar_terms,
    construct_source, critic_identity, evaluate_fit_held, VISUAL_SCHEMA, sha,
)

SCHEMA='continuous-complete-rule-train16-supervised-pretraining-v1'
MANIFEST_SCHEMA='rule-six-stage-16train2held-visual384-cache-locations-v1'
STATE_SCHEMA='continuous-sixteenlane-complete-history-resume-v1'
TRAIN_EPISODES=tuple(f'world{i:03}' for i in (*range(6),*range(9,19)))
HELD_EPISODES=('world007','world008')


def _json(path):return json.loads(Path(path).read_text('utf-8'))


def inspect_train16(manifest,*,expected_sha256):
    """Read only the manifest, indexes and commits; never array packets."""
    manifest=Path(manifest)
    with _shared_binary(manifest) as stream:raw=stream.read()
    digest=hashlib.sha256(raw).hexdigest();m=json.loads(raw)
    if digest!=expected_sha256 or m.get('schema')!=MANIFEST_SCHEMA:
        raise ValueError('Explicit train16 manifest schema/SHA differs')
    entries=m['episodes'];names=[e['episode'] for e in entries]
    if (names!=list(TRAIN_EPISODES+HELD_EPISODES)
            or m.get('train_episodes')!=list(TRAIN_EPISODES) or m.get('held_episodes')!=list(HELD_EPISODES)):
        raise ValueError('Exactly the fixed16 train and separate2 fit-held episodes required')
    specs=[];pending=[]
    for e in entries:
        episode=e['episode'];role='train' if episode in TRAIN_EPISODES else 'readout_fit_held'
        if e['role']!=role or type(e['rows']) is not int or e['rows']<=0:raise ValueError('Episode role/rows differs')
        horizons=e['retained_horizons_native_frames']
        required=[4,12,24] if 9<=int(episode[5:])<=18 else [4]
        if horizons!=required:raise ValueError('Actual original4f versus new full-horizon protocol differs')
        label=Path(e['label_cache']);visual=Path(e['visual_cache'])
        spec=dict(episode=episode,role=role,rows=e['rows'],native_seed=e['native_seed'],
            source_index_sha256=e['source_index_sha256'],label_root=str(label),visual_root=str(visual),
            retained_horizons_native_frames=list(horizons),stage_counts=e.get('stage_counts',{}),ready=False)
        if e.get('status')!='committed' or not (label/'COMMITTED.json').is_file():
            pending.append('label/visual:'+episode);specs.append(spec);continue
        available=e.get('original_rule_targets_preserved');weights=e.get('original_rule_weights_preserved')
        if type(available) is not bool or available!=weights:raise ValueError('Explicit original rule pair availability required')
        if episode=='world000' and available:raise ValueError('world000 has no original rule soft target/weight')
        li=_json(label/'index.json');lc=_json(label/'COMMITTED.json');label_sha=sha(label/'index.json')
        if (label_sha!=e['label_index_sha256'] or lc['index_sha256']!=label_sha
                or sha(label/'COMMITTED.json')!=e['label_COMMITTED_sha256']
                or li['source_index_sha256']!=e['source_index_sha256'] or li['rows']!=e['rows']
                or li.get('complete_episode') is not True or lc.get('complete_episode') is not True
                or li.get('retained_horizons_native_frames',[4])!=horizons):
            raise ValueError('Committed label/source/horizon identity differs:'+episode)
        spec.update(original_rule_available=available,label_index_sha256=label_sha,
            label_commit_sha256=e['label_COMMITTED_sha256'])
        if not (visual/'COMMITTED.json').is_file():
            pending.append('visual:'+episode);specs.append(spec);continue
        vi=_json(visual/'index.json');vc=_json(visual/'COMMITTED.json');identity=_json(visual/'identity.json')
        vi_sha=sha(visual/'index.json');identity_sha=sha(visual/'identity.json')
        if (vi.get('schema')!=VISUAL_SCHEMA or vi.get('status')!='completed_one_full_rule_visual_history'
                or e['visual_index_sha256']!=vi_sha or vc['index_sha256']!=vi_sha
                or e['visual_COMMITTED_sha256']!=sha(visual/'COMMITTED.json')
                or vi['identity_sha256']!=identity_sha or vc['identity_sha256']!=identity_sha
                or vc.get('complete_episode') is not True or vi.get('complete_episode') is not True
                or vi['rows']!=e['rows'] or vi['source_index_sha256']!=e['source_index_sha256']
                or vi['label_index_sha256']!=label_sha or identity['episode']!=episode):
            raise ValueError('Committed full visual/source/label identity differs:'+episode)
        cursor=0
        if len(vi['packets'])!=len(li['label_packets']):raise ValueError('Paired packet count differs')
        for vp,lp in zip(vi['packets'],li['label_packets']):
            if (vp['start']!=cursor or lp['start']!=cursor or vp['rows']!=lp['rows']
                    or vp['label_reference']['packet']!=lp
                    or Path(vp['label_reference']['root']).resolve()!=label.resolve()
                    or Path(vp['name']).name!=vp['name'] or Path(lp['name']).name!=lp['name']):
                raise ValueError('Complete packet order/reference differs:'+episode)
            cursor+=vp['rows']
        if cursor!=e['rows']:raise ValueError('Incomplete full episode:'+episode)
        spec.update(visual_index_sha256=vi_sha,visual_identity_sha256=identity_sha,
            visual_commit_sha256=e['visual_COMMITTED_sha256'],packet_count=len(vi['packets']),ready=True)
        specs.append(spec)
    commit_path=manifest.parent/'COMMITTED.json';top_ready=False
    if commit_path.is_file():
        commit=_json(commit_path)
        if commit.get('manifest_sha256')!=digest or commit.get('complete') is not True:
            raise ValueError('Top-level complete dataset commit differs')
        top_ready=m.get('status')=='completed_train16_data_extension' and m.get('complete') is True
    if not top_ready:pending.append('top_level_complete_commit')
    return dict(schema=SCHEMA,manifest_path=str(manifest),manifest_sha256=digest,
        dataset_commit_sha256=sha(commit_path) if top_ready else None,specs=specs,pending=pending,
        ready=top_ready and not pending,train_rows=sum(s['rows'] for s in specs if s['role']=='train'),
        held_rows=sum(s['rows'] for s in specs if s['role']=='readout_fit_held'),
        NumPy_imported=False,Torch_imported=False,array_packets_read=0)


class Train16RuleEpisodeReader(RuleEpisodeReader):
    """Reuse original paired streaming/clock checks; retain real future labels."""
    def fetch(self,start,limit):
        import numpy as np
        data=super().fetch(start,limit)
        if data is not None:
            n=len(data['actions']);margins=np.zeros((n,3,18),np.float32);trusted=np.zeros((n,3),bool)
            margins[:,0]=data['signed_margin4'];trusted[:,0]=data['trusted4']
            if self.spec['retained_horizons_native_frames']==[4,12,24]:
                row=start;offset=0
                while offset<n:
                    part=bisect.bisect_right(self.starts,row)-1;block=self._packet(part)
                    local=row-self.parts[part]['start'];take=min(n-offset,self.parts[part]['rows']-local)
                    full=block['margins'];trust=block['trusted_horizons'];rows=self.parts[part]['rows']
                    if (full.shape!=(rows,3,18) or full.dtype!=np.float32 or not np.isfinite(full).all()
                            or trust.shape!=(rows,3) or trust.dtype!=np.bool_
                            or not np.array_equal(full[:,0],block['signed_margin4'])
                            or not np.array_equal(trust[:,0],block['trusted4'])):
                        raise ValueError('Actual4/12/24 producer arrays differ')
                    margins[offset:offset+take]=full[local:local+take];trusted[offset:offset+take]=trust[local:local+take]
                    row+=take;offset+=take
            data.update(margins=margins,trusted_horizons=trusted)
        self.last_window=(start,data)
        return data


def assemble_train16_window(readers,cursors):
    """Original16-wide assembly, then preserve genuine optional future targets."""
    if len(readers)!=16 or len(cursors)!=16:raise ValueError('Exactly16 original train lanes required')
    data,used=assemble_window(readers,cursors)
    for lane,(reader,cursor,n) in enumerate(zip(readers,cursors,used)):
        if not n:continue
        start,block=reader.last_window
        if start!=cursor:raise ValueError('Own contiguous fetched window differs')
        # The legacy assembly already filled0;future trust stays False for old8.
        data['margins'][:n,lane].copy_(data['margins'].new_tensor(block['margins']))
        data['trusted'][:n,lane].copy_(data['trusted'].new_tensor(block['trusted_horizons']))
    return data,used


def data_binding(data):
    keys=('episode','rows','native_seed','source_index_sha256','label_index_sha256',
        'visual_index_sha256','visual_identity_sha256','retained_horizons_native_frames','original_rule_available')
    return dict(manifest_sha256=data['manifest_sha256'],dataset_commit_sha256=data['dataset_commit_sha256'],
        train=[{k:s[k] for k in keys} for s in data['specs'] if s['role']=='train'],
        held=[{k:s[k] for k in keys} for s in data['specs'] if s['role']=='readout_fit_held'])


def requested_data_state(source,binding,*,resume_data_state=False):
    """Default new epoch deliberately ignores any parent6/16-lane data state."""
    if not resume_data_state:return None
    state=source.get('offline_resume')
    if (not isinstance(state,dict) or state.get('schema')!=STATE_SCHEMA
            or state.get('data_binding')!=binding or state.get('epoch_complete') is not False):
        raise ValueError('Explicit resume requires the same unfinished16-lane epoch/data binding')
    cursors=state.get('cursors')
    if (not isinstance(cursors,list) or len(cursors)!=16 or
            any(type(c) is not int or not 0<=c<=s['rows'] for c,s in zip(cursors,binding['train']))):
        raise ValueError('Saved exact16 source cursors required')
    return state


def atomic_torch_save(path,payload):
    import os
    import torch
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('wb') as stream:torch.save(payload,stream);stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def checkpoint_train16(model,optimizer,*,algorithm,counters,provenance,binding,
        cursors,activity,costs,counter_base,epoch_complete):
    from .continuous_connectome_training import training_checkpoint_payload
    from .continuous_normalized_risk_coordinate import tag_normalized_payload
    payload=training_checkpoint_payload(model,optimizer,algorithm=algorithm,counters=counters,provenance=provenance)
    if provenance['normalized_risk']:tag_normalized_payload(payload,provenance['normalization_spec'])
    payload['offline_resume']=dict(schema=STATE_SCHEMA,data_binding=copy.deepcopy(binding),
        cursors=list(cursors),activity=activity.detach().cpu().clone(),costs=copy.deepcopy(costs),
        counter_base=copy.deepcopy(counter_base),epoch_complete=epoch_complete,
        history='own row0-origin state;TBPTT16 detached preceding-theta carry',native_mid_game_resume=False)
    payload['recurrent_activity_saved']=True;payload['offline_activity_only']=True
    return payload
