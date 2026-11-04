"""Use real ResourceV10 demonstrations as partial-observation supervision.

Historical V5 archives do not contain the current complete native sensor.
Convert only information they actually stored, and never turn their simulated
geometry into a native collision label.  Teacher-controlled actions are for
recurrent burn-in and imitation, never for PPO.
"""
from __future__ import annotations

from bisect import bisect_right
import io
import json
from pathlib import Path
import zlib

import numpy as np

FIELDS = ('features', 'objects', 'targets', 'actions', 'episode_starts',
          'stage', 'stage_frame', 'before_frames', 'after_observation_frames',
          'teacher_guidance_weight')


class Archive:
    """Read only requested arrays; the original large maps remain untouched."""
    def __init__(self, folder):
        self.folder = Path(folder)
        self.index = json.loads((self.folder / 'index.json').read_text(encoding='utf-8'))
        if (self.index.get('schema') != 'B1-lossless-natural-teacher-trajectory-shards-v1'
                or not self.index.get('natural_episode_complete')
                or self.index['metadata'].get('teacher_revision') !=
                'resource-v10-immediate-safe-soft-route-support-v1'):
            raise ValueError('A complete ResourceV10 rule-teacher episode is required')
        self.shards = self.index['shards']
        self.starts = [s['start'] for s in self.shards]
        self.rows = self.index['rows']

    def fetch(self, first, stop, fields=FIELDS):
        if not 0 <= first < stop <= self.rows:
            raise ValueError('Requested teacher rows are outside the episode')
        result = {k: [] for k in fields}
        cursor = first
        while cursor < stop:
            part = bisect_right(self.starts, cursor) - 1
            s = self.shards[part]
            end = min(stop, s['start'] + s['rows'])
            raw = zlib.decompress((self.folder / s['name']).read_bytes())
            with np.load(io.BytesIO(raw), allow_pickle=False) as data:
                for key in fields:
                    result[key].append(data[key][cursor-s['start']:end-s['start']].copy())
            cursor = end
        return {key: np.concatenate(parts) for key, parts in result.items()}

    def first_stage_at_least(self, stage):
        """Completed full-story episodes proceed through stages in order."""
        low, high = 0, self.rows
        while low < high:
            mid = (low + high) // 2
            value = int(self.fetch(mid, mid+1, ('stage',))['stage'][0])
            if value < stage:
                low = mid + 1
            else:
                high = mid
        return low


def legacy_task_features(f, objects, stage):
    """Recover only visible enemy information, treating full items as unknown.

    The V5 objects encode up to96 selected entities, so an absent object is not
    proof of global absence.  In particular a complete item inventory, boss
    identity and clipped values cannot be recreated from this archive.
    """
    x = np.zeros(18, np.float32)
    x[0] = stage / 6
    # A false resources_observed value suppresses all item and POC features,
    # rather than claiming that an omitted set of items was empty.
    enemy = objects[(objects[:, 0] > .5) & (objects[:, 2] > .5)
                    & (objects[:, 15] > 0)]
    if len(enemy):
        # Mirrors the ordinary target tie-break where the retained data allow
        # it.  The later native primary-ECL identity was not archived.
        enemy = sorted(enemy, key=lambda row: (not (row[14] > .5),
                                               abs(row[6]), abs(row[7])))[0]
        hp = float(enemy[15]) * 30000.
        x[2:7] = (1., enemy[6]*384/368, enemy[7]*448/400,
                  hp/(hp+1000.), enemy[14] > .5)
    return np.clip(x, -1, 1)


def convert_window(raw):
    from .compact_spiking_sensation import encode_geometry
    inputs = []
    full = []
    for f, objects, stage in zip(raw['features'], raw['objects'], raw['stage']):
        base, info = encode_geometry(f, objects, complete=False)
        # Current release, boss and focus forecasts need native fields that do
        # not exist here.  Preserve the estimates as hints, not certified risk.
        base[72:75] = 0.
        inputs.append(np.concatenate((base, legacy_task_features(f, objects, stage))))
        full.append(int((objects[:, 0] > .5).sum()) >= 96)
    return dict(features=np.stack(inputs), targets=raw['targets'].copy(),
                actions=raw['actions'].copy(), episode_starts=raw['episode_starts'].copy(),
                teacher_guidance_weight=raw['teacher_guidance_weight'].copy(),
                stage=raw['stage'].copy(), stage_frame=raw['stage_frame'].copy(),
                before_frames=raw['before_frames'].copy(),
                after_observation_frames=raw['after_observation_frames'].copy(),
                object_slots_full=np.asarray(full, dtype=bool))


def near_threat_score(f, objects):
    """Rank observations for sampling, not a certified collision forecast."""
    present = objects[:, 0] > .5
    hazard = present & ((objects[:, 1] > .5) | (objects[:, 2] > .5)
                        | (objects[:, 3] > .5) | (objects[:, 4] > .5))
    o = objects[hazard]
    if not len(o):
        return 0.
    distance = np.linalg.norm(o[:, 6:8]*np.array((384., 448.)), axis=1)
    radius = np.linalg.norm(o[:, 10:12]*128., axis=1) + max(2., float(abs(f[13])*64))
    return float(np.maximum(0., 100.-(distance-radius)).max()/100. +
                 min(1., np.count_nonzero(distance < 100.)/20.))


def windows(folder, stages=(3, 4), *, count=2, length=48, burn_in=32, candidates=4):
    archive = Archive(folder)
    for stage in stages:
        first = archive.first_stage_at_least(stage)
        stop = archive.first_stage_at_least(stage + 1)
        if stop - first < length:
            continue
        # Sample several actual observations in each chronological bin.  A
        # local-nearby-threat score prioritizes informative windows without
        # turning that score into a training target or discarding diversity.
        edges = np.linspace(first, stop-length+1, count+1, dtype=np.int64)
        starts = []
        for left, right in zip(edges[:-1], edges[1:]):
            if right <= left:
                continue
            options = np.linspace(left, right-1, candidates, dtype=np.int64)
            scored = []
            for option in dict.fromkeys(int(x) for x in options):
                probe = archive.fetch(option+burn_in, option+burn_in+1, ('features', 'objects'))
                scored.append((near_threat_score(probe['features'][0], probe['objects'][0]), option))
            starts.append(max(scored)[1])
        for start in dict.fromkeys(starts):
            raw = archive.fetch(start, start + length)
            if not np.all(raw['stage'] == stage):
                raise ValueError('A selected window unexpectedly crosses stages')
            yield stage, start, raw
