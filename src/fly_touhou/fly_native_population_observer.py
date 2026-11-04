"""Bounded observation of the SAME native-policy forward;no control outputs.

Collects actual full-neuron window statistics before live state is committed.
All writes are separate diagnostic files. There is no neural/model/teacher
forward, optimizer, RNG draw, native step, or return action in this observer.
"""
from collections import Counter
import hashlib
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from .fly_population_path_trace import build_population_layout,_population_sum
from .thrl_durable_records import write_json

SCHEMA='fly-native-same-scene-population-observer-v1'


class NativePopulationObserver:
    def __init__(self,model,output,*,cadence=64,per_game=8,layout=None):
        self.model=model;self.output=Path(output)
        self.output.mkdir(parents=True,exist_ok=False)
        self.layout=build_population_layout(model.brain) if layout is None else layout
        if type(cadence) is not int or cadence<1 or type(per_game) is not int or not 1<=per_game<=32:
            raise ValueError('Finite diagnostic cadence and per-game sample count required')
        self.cadence=cadence;self.per_game=per_game;self.by_game=Counter();self.records=[]
        self.disabled=False;self.seconds=0.;self.record_batches=0
        b=model.brain
        self.window_ms=b.substeps*b.dt_ms
        self.weight=b.input_encoder.weight.detach().cpu().clone()
        self.bias=b.input_encoder.bias.detach().cpu().clone()
        self.max_rate=b.max_input_rate_hz
        write_json(self.output/'populations.json',dict(populations=self.layout.populations,
            population_sizes=self.layout.population_sizes.tolist(),sampled_cells=self.layout.sampled_cells,
            annotation_provenance=self.layout.provenance,layout_identity=self.layout.identity_sha256))
        self._index()

    def _index(self):
        write_json(self.output/'index.json',dict(schema=SCHEMA,status='disabled_after_diagnostic_error' if self.disabled else 'recording',
            records=self.records,recorded_rows=len(self.records),per_game=dict(self.by_game),
            cadence_decisions=self.cadence,maximum_rows_per_game=self.per_game,neural_window_ms=self.window_ms,
            scopes=dict(expected_external_Hz='encoder Poisson expectation;not realized injected events',
                input_neural_Hz='actual input-neuron output spikes during the same20ms',
                middle='engineering remaining role plus actual annotation partitions;not causal layers',
                DN_window_Hz='same20ms spike counts',DN_EMA_Hz='original filtered rate;different window'),
            sample_scope='first8 positive multiples of64 own decisions per game;not whole-story representative',
            elapsed_recording_seconds=self.seconds,extra_brain_forwards=0,extra_teacher_queries=0,
            extra_native_steps=0,optimizer_steps=0,causal_intervention=False,
            population_means_prove_information=False,actual_action_acknowledged=False))

    @torch.no_grad()
    def record_batch(self,*,model,context,selected,sensory,following,features,maps,objects,
                     raw,composed,cost,actual_logits):
        if self.disabled:return
        began=time.perf_counter()
        count_before=len(self.records)
        try:
            if model is not self.model:raise ValueError('Observer belongs to a different frozen model')
            def cpu(x):return x.detach().cpu().numpy().copy()
            for lane,((worker,game),before) in enumerate(zip(context,selected)):
                decision=before.decisions
                if not decision or decision%self.cadence or self.by_game[game]>=self.per_game:continue
                key=int(following.rng_keys[lane]);counter=int(following.noise_counter[lane])
                if key!=before.brain_rng_key or counter!=int(before.state.noise_counter[0])+model.brain.substeps:
                    raise ValueError('Native source and following-neural identity differ')
                counts=cpu(following.window_spike_counts[lane]).astype(np.float64)
                voltage=cpu(following.membrane[lane]).astype(np.float64)
                synaptic=cpu(following.synaptic[lane]).astype(np.float64)
                size=self.layout.population_sizes
                total=_population_sum(self.layout,counts)
                active=_population_sum(self.layout,(counts>0).astype(np.float64))
                z=sensory[lane].detach().cpu()
                expected=torch.sigmoid(F.linear(z,self.weight,self.bias))*self.max_rate
                logits=actual_logits[lane].detach().cpu();probabilities=logits.softmax(-1)
                payload=dict(encoded_visual=z.numpy().copy(),features=cpu(features[lane]),maps=cpu(maps[lane]),
                    objects=cpu(objects[lane]),raw_logits=cpu(raw[lane]),composed_logits=cpu(composed[lane]),
                    actual_legal_logits=logits.numpy(),action_probs=probabilities.numpy(),predicted_costs=cpu(cost[lane]),
                    input_expected_external_Hz=expected.numpy(),
                    input_neural_Hz=(counts[self.layout.input_indices]*1000/self.window_ms).astype(np.float32),
                    population_mean_neural_Hz=(total/size*1000/self.window_ms).astype(np.float32),
                    population_responding_fraction=(active/size).astype(np.float32),
                    population_mean_membrane_mv=(_population_sum(self.layout,voltage)/size).astype(np.float32),
                    population_mean_synaptic_mv=(_population_sum(self.layout,synaptic)/size).astype(np.float32),
                    sampled_cell_neural_Hz=(counts[self.layout.sampled_indices]*1000/self.window_ms).astype(np.float32),
                    DN_window_Hz=(counts[self.layout.output_indices]*1000/self.window_ms).astype(np.float32),
                    DN_EMA_Hz=cpu(following.rate[lane]))
                # Physical frame numbers are not inferred from decision count:
                # dialogue/event advancement may consume additional game ticks.
                source_hash=hashlib.sha256(b''.join(payload[k].tobytes() for k in ('features','maps','objects'))).hexdigest()
                name=f'game-{game:03}-decision-{decision:06}.npz';path=self.output/name
                with path.with_suffix('.npz.tmp').open('xb') as stream:np.savez_compressed(stream,**payload)
                path.with_suffix('.npz.tmp').replace(path)
                with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
                self.records.append(dict(file=name,sha256=digest,worker=worker,game=game,
                    own_decision_before=decision,brain_rng_key=key,noise_counter_before=counter-model.brain.substeps,
                    noise_counter_after=counter,original_observation_sha256=source_hash,
                    native_frame_number_not_queried=True,greedy_action_proposal=int(logits.argmax()),
                    actual_action_acknowledged=False))
                self.by_game[game]+=1
            self.record_batches+=1
        except (ValueError,OSError,RuntimeError) as error:
            self.disabled=True
            write_json(self.output/'diagnostic-error.json',dict(error=repr(error),
                evaluation_policy_unmodified=True,extra_brain_forwards=0))
        finally:
            self.seconds+=time.perf_counter()-began
            if len(self.records)!=count_before or self.disabled:self._index()

    def close(self):
        self._index()
