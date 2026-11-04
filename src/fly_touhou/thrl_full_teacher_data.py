"""Seed-grouped ragged native trajectories for one shared B1 student.

All original rows remain in the archive. Fixed BPTT chunks omit at most
length-1 terminal tail rows per trajectory from gradient batches, never turn
those rows into synthetic deaths, episodes, or observations.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .thrl_inputs import SEPARATED_LASER_INPUT, observation_shapes
from .thrl_study_io import read_shared_json as read
from .thrl_teacher_archive import TeacherTrajectoryReader


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FullTeacherDataset:
    def __init__(self, folder, *, reader_cache_bytes=32 * 1024**2):
        self.folder = Path(folder).resolve()
        self.manifest_path = self.folder/'demonstrations.json'
        self.manifest = read(self.manifest_path)
        m = self.manifest
        if (m.get('schema') != 'B1-current-V5-full-natural-teacher-demonstrations-v1'
                or m.get('status') != 'completed' or m.get('input_schema') != SEPARATED_LASER_INPUT
                or m.get('native_seed_split') != 'train' or m.get('native_frame_horizon', 'missing') is not None
                or m.get('first_hit_terminal') is not False or m.get('continues_allowed') is not False
                or m.get('policy_actions') != 19 or m.get('PPO_samples') != 0
                or m.get('optimizer_updates') != 0 or m.get('neural_model_test') is not False):
            raise ValueError('Sealed complete natural teacher data in the training seed partition required')
        self.records, used, offset = [], set(), 0
        for entry in m['episodes']:
            path = Path(entry['trajectory_index']).resolve()
            if not path.is_relative_to(self.folder/'trajectories') or sha(path) != entry['trajectory_index_sha256']:
                raise ValueError('Teacher trajectory belongs to another archive or changed index')
            reader = TeacherTrajectoryReader(path.parent, cache_bytes=reader_cache_bytes)
            meta = reader.index['metadata']
            seed = meta['native_seed']
            if (type(seed) is not int or not 0 <= seed <= 65535 or seed % 10 >= 6 or seed in used
                    or meta.get('native_seed_split') != 'train' or meta.get('input_schema') != SEPARATED_LASER_INPUT
                    or meta.get('use_for_training') is not True or meta.get('qualification_only')
                    or meta.get('teacher_revision') != m['teacher_revision']
                    or reader.rows < 1 or reader.rows != entry['recorded_samples'] or reader.rows != entry['policy_decisions']
                    or reader.index['native_episode_summary']['outcome'] != entry['outcome']):
                raise ValueError('Actual seed, episode, teacher or complete row count differs')
            required = {**{key:(np.dtype(np.float32), shape)
                          for key,shape in observation_shapes(SEPARATED_LASER_INPUT).items()},
                        'targets':(np.dtype(np.float32),(19,)), 'actions':(np.dtype(np.int64),()),
                        'dones':(np.dtype(np.bool_),()), 'episode_starts':(np.dtype(np.bool_),())}
            if any(reader.fields.get(key) != value for key,value in required.items()):
                raise ValueError('Original V5 observation/label dimensions or precision differ')
            self.records.append(dict(index=len(self.records), reader=reader, rows=reader.rows,
                offset=offset, native_seed=seed, native_episode=meta['native_episode'], entry=entry))
            used.add(seed);offset += reader.rows
        if (not self.records or len(self.records) != m['complete_native_episodes']
                or offset != m['samples'] or offset != m['acknowledged_native_actions']
                or used != set(m['actual_initial_native_seeds'])):
            raise ValueError('Teacher cohort sample/seed bookkeeping differs')
        self.rows, self.manifest_sha256 = offset, sha(self.manifest_path)

    def split(self, *, fraction=.2, seed=31):
        n = len(self.records)
        if n < 2 or not 0 < fraction < .5 or type(seed) is not int:
            raise ValueError('At least two complete trajectory groups and a heldout fraction below one half required')
        canonical = sorted(range(n),key=lambda i:self.records[i]['native_seed'])
        order = [canonical[i] for i in np.random.default_rng(seed).permutation(n).tolist()]
        count = max(1, min(n-1, round(n*fraction)))
        heldout, train = sorted(order[:count]), sorted(order[count:])
        return dict(schema='B1-native-training-seed-grouped-BC-validation-v1', seed=seed,
            train=train, heldout=heldout,
            train_native_seeds=[self.records[i]['native_seed'] for i in train],
            heldout_native_seeds=[self.records[i]['native_seed'] for i in heldout],
            separate_whole_trajectories=True, formal_original_test_seeds_used=False)

    def descriptors(self, indices, *, length=16):
        if type(length) is not int or length < 1:
            raise ValueError('Positive BPTT length required')
        return [(i,start) for i in indices for start in range(0,self.records[i]['rows']-length+1,length)]

    def iter_chunks(self, index, *, chunk_rows=32):
        record = self.records[index]
        previous = None
        for start in range(0,record['rows'],chunk_rows):
            stop = min(record['rows'],start+chunk_rows)
            data = record['reader'].fetch(start,stop)
            n = stop-start
            positions = np.arange(start,stop)
            if (not np.array_equal(data['episode_starts'],positions==0)
                    or not np.array_equal(data['dones'],positions==record['rows']-1)
                    or not np.all(data['native_seeds']==record['native_seed'])
                    or not np.all(data['native_episodes']==record['native_episode'])
                    or not np.all((0<=data['actions'])&(data['actions']<19))
                    or not np.isfinite(data['targets']).all() or np.any(data['targets']<0)
                    or not np.allclose(data['targets'].sum(-1,dtype=np.float64),1.,atol=2e-7,rtol=0)
                    or not np.array_equal(data['targets'].argmax(-1),data['actions'])
                    or np.any(data['after_episode_frames']<=data['before_episode_frames'])
                    or np.any((data['stage']<1)|(data['stage']>6))
                    or (start==0 and (data['before_episode_frames'][0]!=0 or data['stage'][0]!=1))):
                raise ValueError('Before-state labels, native clocks or episode boundaries differ')
            if n>1 and (not np.array_equal(data['before_frames'][1:],data['after_observation_frames'][:-1])
                    or not np.array_equal(data['before_episode_frames'][1:],data['after_episode_frames'][:-1])
                    or np.any(np.diff(data['stage'])<0)):
                raise ValueError('Native teacher chunk is not a contiguous chronological trajectory')
            if previous is not None and (data['before_frames'][0]!=previous[0]
                    or data['before_episode_frames'][0]!=previous[1] or data['stage'][0]<previous[2]):
                raise ValueError('Original trajectory discontinuity at a shard boundary')
            previous = (int(data['after_observation_frames'][-1]),int(data['after_episode_frames'][-1]),int(data['stage'][-1]))
            # An actual expert action remains an imitation label even when a
            # short-horizon geometry proxy abstained. Only native inventory
            # constrains legal actions; it never masks a dangerous movement.
            legal = (data['actions']!=18)|(data['features'][:,4]>0)
            yield start,stop,data,legal
        if previous[1] != record['entry']['native_frames']:
            raise ValueError('Last recorded original frame differs from the native endpoint')

    def close_caches(self):
        for record in self.records:
            record['reader'].cache.clear()
            record['reader'].cached_bytes=0
