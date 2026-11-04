"""Chronological raw V5 batches for full actor-path imitation.

Reuses the sealed FullTeacherDataset and its lossless reader. No Torch, model,
native process, optimizer, latent cache or recurrent state is constructed.
Every trajectory begins at source0; short terminal tails remain real rows.
The caller owns one recurrent state per episode index and detaches only at
its declared TBPTT boundaries. A stage transition or Miss is not a reset.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from numbers import Integral
from typing import Callable, Iterator, Mapping, Sequence

import numpy as np

from .thrl_full_teacher_data import FullTeacherDataset
from .thrl_inputs import SEPARATED_LASER_INPUT, observation_shapes

SHAPES = observation_shapes(SEPARATED_LASER_INPUT)
META_FIELDS = ("native_seeds", "native_episodes", "before_frames", "after_observation_frames",
               "before_episode_frames", "after_episode_frames", "stage", "stage_frame",
               "native_misses", "native_bombs_used")


@dataclass(frozen=True)
class RawTeacherBatch:
    episode_indices: tuple[int, ...]
    source_rows: np.ndarray
    features: np.ndarray
    maps: np.ndarray
    objects: np.ndarray
    actions: np.ndarray
    legal: np.ndarray
    dones: np.ndarray
    episode_starts: np.ndarray
    metadata: Mapping[str, np.ndarray]
    role: str

    @property
    def rows(self) -> int:
        return int(self.actions.size)

    @property
    def sequence_length(self) -> int:
        return int(self.actions.shape[1])


class SequentialRawTeacherBatches:
    """Finite, grouped streams; caller explicitly chooses train/heldout role.

    Groups contain at most batch_episodes complete natural trajectories. Each
    group's members advance in chronological blocks. Ragged blocks are grouped
    by their actual length, with no invented padding or repeated observations.
    Arrays are independently stacked copies; changing a delivered batch cannot
    change the source archive, its reader cache or another delivered batch.
    """

    def __init__(self, dataset: FullTeacherDataset, *, sequence_length=16,
                 batch_episodes=8, check_stop: Callable[[], None] | None = None):
        if (type(sequence_length) is not int or not 1 <= sequence_length <= 64
                or type(batch_episodes) is not int or not 1 <= batch_episodes <= 30):
            raise ValueError("Finite sequence1..64 and episode batch1..30 required")
        self.dataset = dataset
        self.length, self.batch_episodes = sequence_length, batch_episodes
        self.check_stop = check_stop or (lambda: None)
        self.delivered_rows = self.delivered_batches = 0
        self.finished_episodes = []
        self.complete_epoch = False
        self._active_epoch = False

    @staticmethod
    def _validate_piece(index, record, start, stop, data, legal):
        n = stop - start
        if (type(start) is not int or type(stop) is not int or not 0 <= start < stop <= record["rows"]
                or not 1 <= n <= 64
                or set(SHAPES) - set(data)):
            raise ValueError("Expected nonempty raw original observation block")
        for key, shape in SHAPES.items():
            value = data[key]
            if (value.dtype != np.float32 or value.shape != (n, *shape)
                    or not np.isfinite(value).all() or np.any(np.abs(value) > 1)):
                raise ValueError("Raw original FP32 V5 modality differs: " + key)
        for key in ("actions", *META_FIELDS):
            if data[key].dtype != np.int64 or data[key].shape != (n,):
                raise ValueError("Original scalar metadata differs: " + key)
        for key in ("dones", "episode_starts"):
            if data[key].dtype != np.bool_ or data[key].shape != (n,):
                raise ValueError("Original reset/terminal metadata differs")
        positions = np.arange(start, stop)
        expected_legal = (data["actions"] != 18) | (data["features"][:, 4] > 0)
        if (legal.dtype != np.bool_ or legal.shape != (n,)
                or not np.array_equal(legal, expected_legal)
                or not np.array_equal(data["episode_starts"], positions == 0)
                or not np.array_equal(data["dones"], positions == record["rows"] - 1)
                or np.any(data["native_seeds"] != record["native_seed"])
                or np.any(data["native_episodes"] != record["native_episode"])):
            raise ValueError("Whole-episode order/seed/legality differs")
        return index, positions, data, legal

    @staticmethod
    def _batch(pieces, role):
        indices = tuple(row[0] for row in pieces)
        if len(indices) != len(set(indices)):
            raise ValueError("A recurrent lane cannot contain two blocks of one episode")
        stack = lambda key: np.stack([row[2][key] for row in pieces])
        return RawTeacherBatch(indices, np.stack([row[1] for row in pieces]),
            *(stack(key) for key in ("features", "maps", "objects", "actions")),
            np.stack([row[3] for row in pieces]), stack("dones"), stack("episode_starts"),
            {key: stack(key) for key in META_FIELDS}, role)

    def epoch(self, indices: Sequence[int], *, role="train") -> Iterator[RawTeacherBatch]:
        if role not in ("train", "heldout", "inspect"):
            raise ValueError("Explicit train/heldout/inspect role required")
        indices = tuple(indices)
        if (not indices or any(isinstance(i, bool) or not isinstance(i, Integral)
                or not 0 <= i < len(self.dataset.records) for i in indices)
                or len(indices) != len(set(indices))):
            raise ValueError("Distinct existing complete episode indices required")
        if self._active_epoch:
            raise ValueError("Close the previous epoch before starting another on this stream")
        self._active_epoch = True
        self.delivered_rows = self.delivered_batches = 0
        self.finished_episodes = []
        self.complete_epoch = False
        try:
            for group_start in range(0, len(indices), self.batch_episodes):
                self.check_stop()
                self.dataset.close_caches()
                group = indices[group_start:group_start + self.batch_episodes]
                active = {i: iter(self.dataset.iter_chunks(i, chunk_rows=self.length)) for i in group}
                cursors = {i: 0 for i in group}
                while active:
                    pieces = defaultdict(list)
                    for i in tuple(active):
                        self.check_stop()
                        try:
                            start, stop, data, legal = next(active[i])
                        except StopIteration:
                            if cursors[i] != self.dataset.records[i]["rows"]:
                                raise ValueError("A natural source trajectory ended early")
                            self.finished_episodes.append(int(i))
                            del active[i]
                            continue
                        if start != cursors[i] or stop - start > self.length:
                            raise ValueError("Chronological source0 blocks cannot skip or repeat history")
                        piece = self._validate_piece(int(i), self.dataset.records[i], start, stop, data, legal)
                        cursors[i] = stop
                        pieces[stop - start].append(piece)
                    for length in sorted(pieces, reverse=True):
                        batch = self._batch(pieces[length], role)
                        self.delivered_rows += batch.rows
                        self.delivered_batches += 1
                        yield batch
            if self.delivered_rows != sum(self.dataset.records[i]["rows"] for i in indices):
                raise ValueError("A complete epoch lost original observations")
            self.complete_epoch = True
        finally:
            # Generator.close/STOP does not promote a prefix to a full epoch.
            self.dataset.close_caches()
            self._active_epoch = False
