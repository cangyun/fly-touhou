"""Compact read-only DN activity and paired geometry labels after inference.

Geometry is recorded for offline analysis only. No recorded value is returned
to the policy, optimizer or game. This module starts nothing on import.
"""
import hashlib
import os
from pathlib import Path

import numpy as np
import torch


class DNDiagnosticCapture:
    def __init__(self, output, *, stride=8, maximum_rows=8192, chunk_rows=128):
        if any(type(v) is not int or v < 1 for v in (stride, maximum_rows, chunk_rows)):
            raise ValueError('Explicit finite recording schedule required')
        self.output = Path(output)
        self.stride, self.maximum_rows, self.chunk_rows = stride, maximum_rows, chunk_rows
        self.rows = self.parts = 0
        self.buffer, self.references, self.errors = [], [], []
        self.output.mkdir(parents=True, exist_ok=False)

    @torch.no_grad()
    def record(self, policy, context, features, maps, objects, logits, sensory):
        if self.rows >= self.maximum_rows or self.errors:
            return
        selected = [i for i, (worker, _) in enumerate(context)
            if (policy.workers[worker].decisions-1) % self.stride == 0]
        selected = selected[:self.maximum_rows-self.rows]
        if not selected:
            return
        try:
            from .thrl_dodge_supervision import prepare_dense_targets
            from .thrl_inputs import SEPARATED_LASER_INPUT
            index = torch.tensor(selected, device=features.device)
            f, o, m = (x.index_select(0, index).detach().cpu() for x in (features, objects, maps))
            n = len(selected)
            margins, trusted, _ = prepare_dense_targets(f[None], o[None], m[None],
                torch.full((1, n), -1, dtype=torch.long), torch.full((1, n), -1, dtype=torch.int8),
                input_schema=SEPARATED_LASER_INPUT, max_labels=n, horizon_specific=True)
            outputs = policy.student.brain.output_indices
            reference = policy.student.brain.parameters_source.reference_poisson_rate_hz
            for row, position in enumerate(selected):
                worker, game = context[position]
                live = policy.workers[worker]
                state = live.state
                values = dict(game=np.int64(game), worker=np.int16(worker),
                    decision=np.int64(live.decisions), brain_rng_key=np.int64(live.brain_rng_key),
                    noise_counter=state.noise_counter[0], features=f[row],
                    encoded_visual=sensory[position], margins=margins[0, row], trusted=trusted[0, row],
                    DN_rate=state.rate[0]/reference,
                    DN_membrane=state.membrane[0].index_select(0, outputs),
                    DN_synaptic=state.synaptic[0].index_select(0, outputs),
                    DN_refractory=state.refractory[0].index_select(0, outputs),
                    DN_window_spikes=state.window_spike_counts[0].index_select(0, outputs),
                    DN_recent_spikes=torch.stack([x[0].index_select(0, outputs) for x in state.delay_queue]).to(torch.uint8),
                    risk_logits=policy.risk_head(state)[0], policy_logits=logits[position])
                self.buffer.append({key: (value.detach().cpu().numpy().copy()
                    if torch.is_tensor(value) else np.asarray(value)) for key, value in values.items()})
                self.rows += 1
                if len(self.buffer) >= self.chunk_rows:
                    self.flush()
        except Exception as error:
            # Optional recording failure does not alter the already chosen
            # neural scores, rerun the brain, or discard a real game.
            self.errors.append(repr(error))

    def flush(self):
        if not self.buffer:
            return
        arrays = {key: np.stack([r[key] for r in self.buffer]) for key in self.buffer[0]}
        path = self.output/f'part-{self.parts:04}.npz'
        temporary = path.with_suffix('.npz.tmp')
        with temporary.open('wb') as stream:
            np.savez(stream, **arrays)
        os.replace(temporary, path)
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        self.references.append(dict(file=path.name, rows=len(self.buffer), sha256=digest))
        self.parts += 1
        self.buffer.clear()

    def close(self):
        from .thrl_durable_records import write_json
        try:
            self.flush()
        except Exception as error:
            self.errors.append(repr(error))
        report = dict(schema='fly-frozen-DN-risk-diagnostic-capture-v1', rows=self.rows,
            stride=self.stride, maximum_rows=self.maximum_rows, parts=self.references,
            complete=not self.errors, errors=self.errors,
            columns='same following DN rate,membrane,synaptic,refractory,window counts,recent delay spikes',
            geometry='paired current-motion4/12/24 labels;not native counterfactual truth',
            extra_brain_forwards=0, teacher_actor_used=False,
            geometry_used_for_actions=False, parameters_changed=False,
            recording_budget_limits_training_or_games=False)
        write_json(self.output/'index.json', report)
        return report
