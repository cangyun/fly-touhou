"""Batch-one cursor around the existing Phi -> delayed-LIF -> source19 API.

No reader import, file IO, teacher tensor, optimizer, sampling or environment
mutation occurs here. The caller obtains verified CachedSensoryRow objects
from the existing reader and keeps their teacher references in its loss layer.
"""
from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np
import torch

from .fly_distillation_student import (
    FlyDistillationStudent, PINNED_GRAPH_FINGERPRINT, PINNED_PORT_MANIFEST_SHA256,
)
from .whole_brain import WholeBrainState

if TYPE_CHECKING:
    from .fly_sensory_cache_reader import CachedSensoryRow

PINNED_D95_BANK = "1c6006dc7fbbbe5465e7afcef5f43afc8ffebfc4aba8e7833f4ad2b55eebb4f5"


class CachedWholeBrainBridge:
    """Own one complete neural state; never infer resets from stage or Miss.

    Production students must already have passed FlyDistillationStudent's
    full-graph fingerprint and verified-port constructor. This wrapper checks
    those identities/configuration, not a replacement graph/port loader.
    synthetic_fixture=True is explicit, CPU-only, and never a full-graph claim.
    """

    def __init__(self, student: FlyDistillationStudent, *, synthetic_fixture: bool = False):
        if not isinstance(student, FlyDistillationStudent) or type(synthetic_fixture) is not bool:
            raise ValueError("The existing FlyDistillationStudent and explicit fixture scope are required")
        brain = student.brain
        if (brain.substeps != 100 or brain.dt_ms != .2 or brain.n_actions != 19
                or brain.obs_dim != 9721 or tuple(student.sensory_adapter.map_pool_shape) != (12, 12)
                or brain.geometry_input_encoder is not None):
            raise ValueError("Only the existing V5 Phi9721/source19/100x0.2ms contract is accepted")
        if synthetic_fixture:
            if brain.require_full_graph or student.device.type != "cpu":
                raise ValueError("Synthetic fixtures must explicitly use a CPU non-full graph")
        elif (brain.require_full_graph is not True
                or student.graph_fingerprint != PINNED_GRAPH_FINGERPRINT
                or student.interface_provenance.get("manifest_sha256") != PINNED_PORT_MANIFEST_SHA256
                or brain.n_neurons != 138639 or brain.graph.n_edges != 15091983
                or len(brain.input_indices) != 1024 or len(brain.output_indices) != 1299):
            raise ValueError("Production bridge requires the constructor-verified complete graph and ports")
        self.student = student
        self.synthetic_fixture = synthetic_fixture
        self._state: WholeBrainState | None = None
        self._previous = None
        self._rng_key: int | None = None

    @property
    def state(self) -> WholeBrainState | None:
        return self._state

    @property
    def previous_metadata(self):
        return self._previous

    def detach_state(self) -> WholeBrainState | None:
        """Explicit TBPTT boundary only; no optimizer or automatic detachment."""
        if self._state is not None:
            self._state = self._state.detach()
        return self._state

    def step(self, row: CachedSensoryRow, *, episode_rng_key: int | None = None):
        """Return [1,19] raw logits and own complete differentiable brain state.

        An explicit key is required at row0 of each new original episode.
        A stage change, deathbomb/Miss or packet boundary is never a reset.
        New epochs/frozen replays use a fresh bridge and replay from row0.
        """
        # Do not import the CPU reader: its import sets CUDA_VISIBLE_DEVICES=-1.
        reader = sys.modules.get("fly_touhou.fly_sensory_cache_reader")
        if reader is None or not isinstance(row, reader.CachedSensoryRow):
            raise ValueError("Expected a real CachedSensoryRow from the already loaded verified reader")
        meta = row.metadata
        if not isinstance(meta, reader.SourceRowMetadata):
            raise ValueError("Expected original SourceRowMetadata")
        integer_fields = ("source_row", "before_frames", "after_observation_frames", "native_seeds",
                          "stage", "stage_frame", "native_misses", "native_bombs_used", "behaviour_actions")
        if (any(type(getattr(meta, key)) is not int for key in integer_fields)
                or type(meta.episode_starts) is not bool or type(meta.dones) is not bool
                or not isinstance(meta.episode, str) or not meta.episode
                or min(meta.source_row, meta.before_frames, meta.stage_frame, meta.native_misses, meta.native_bombs_used) < 0
                or meta.after_observation_frames < meta.before_frames
                or not 0 <= meta.native_seeds <= 65535 or not 1 <= meta.stage <= 6
                or not 0 <= meta.behaviour_actions < 19):
            raise ValueError("Invalid original row/clock/episode metadata")
        # Label contents are never opened or used; only the reader's bank/row identity is checked.
        reference = row.teacher
        if (not isinstance(reference, reader.TeacherLabelReference)
                or reference.episode != meta.episode or reference.source_row != meta.source_row
                or (not self.synthetic_fixture and reference.bank_identity_sha256 != PINNED_D95_BANK)):
            raise ValueError("Teacher reference identity differs from this cached source row/D95 bank")
        phi = row.phi
        if (not isinstance(phi, np.ndarray) or phi.dtype != np.float32 or phi.shape != (9721,)
                or phi.flags.writeable or not np.isfinite(phi).all() or np.any(np.abs(phi) > 1)):
            raise ValueError("Expected immutable finite normalized CPU FP32 Phi[9721]")
        previous = self._previous
        new_episode = previous is None or meta.episode != previous.episode
        if new_episode:
            if (meta.source_row != 0 or not meta.episode_starts or meta.stage != 1
                    or previous is not None and not previous.dones):
                raise ValueError("Start at source row0; only a true terminal permits a new original episode")
            if type(episode_rng_key) is not int or not 0 <= episode_rng_key < 2**31:
                raise ValueError("Each new original episode requires an explicit RNG key in [0,2**31)")
            state = self.student.initial_state(1, rng_seed=episode_rng_key)
            key = episode_rng_key
        else:
            if (previous.dones or meta.episode_starts or meta.source_row != previous.source_row + 1
                    or meta.before_frames != previous.after_observation_frames
                    or meta.native_seeds != previous.native_seeds or meta.stage < previous.stage):
                raise ValueError("Cached source rows/clocks/seed must be adjacent; no implicit reset or rewind")
            if episode_rng_key is not None and (type(episode_rng_key) is not int or episode_rng_key != self._rng_key):
                raise ValueError("Cannot change the episode RNG key mid-history")
            state, key = self._state, self._rng_key
        # Copy read-only CPU bytes, then call the original trainable input/LIF/DN path.
        sensory = torch.tensor(phi, dtype=torch.float32, device=self.student.device).unsqueeze(0)
        logits, next_state = self.student.brain(sensory, state)
        if logits.shape != (1, 19) or not isinstance(next_state, WholeBrainState) or not bool(torch.isfinite(logits).all()):
            raise ValueError("Existing brain must return finite raw source19 logits and WholeBrainState")
        self._state, self._previous, self._rng_key = next_state, meta, key
        return logits, next_state
