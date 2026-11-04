"""Typed .1-bounded postsynaptic-gain variant of the existing centered student.

Only the existing whole_brain knob is enabled. Migration copies three I/O
parameters and starts every new gain logit at zero; it is not numerical resume.
No graph, native process, training loop, GPU or file write runs on import.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import torch
from torch import nn

from . import fly_centered_distillation as centered
from .fly_centered_distillation import CenteredFlyDistillationStudent
from .fly_distillation_student import FlyDistillationStudent, PINNED_GRAPH_FINGERPRINT, connectome_fingerprint

PARAMETER_SCHEMA = 'fly-gain01-centered-Phi9721-source19-parameter-checkpoint-v1'
STUDENT_SCHEMA = 'fly-gain01-centered-Phi9721-source19-student-v1'
GAIN_NAME = 'brain.recurrent_gain_logits'
GAIN_LIMIT = .1
CENTERED_SOURCE_SHA = '9ecbb3c83d3d407095b1e1add1356f64300df9b5d43af2da3dc60604ee818807'
MIGRATION_SOURCE_SHA = '426968272608fd2ef217712f2c50166ef99ac2f6a102a702c86d11e746b2360c'


class GainCenteredFlyDistillationStudent(CenteredFlyDistillationStudent):
    def __init__(self, *args, recurrent_gain_limit=GAIN_LIMIT, **kwargs):
        if isinstance(recurrent_gain_limit, bool) or recurrent_gain_limit != GAIN_LIMIT:
            raise ValueError('This typed experiment fixes existing postsynaptic gain_limit=.1')
        # Reuse the parent's exact centering/physiology validation unchanged.
        # Before any state/forward/capture, enable the core's already registered
        # optional parameter. Its value/config are explicit in the new schema.
        super().__init__(*args, recurrent_gain_limit=0., **kwargs)
        if self.brain.substeps != 100 or self.brain.dt_ms != .2:
            raise ValueError('Keep the existing 100x.2ms neural decision')
        self.brain.recurrent_gain_limit = GAIN_LIMIT
        self.brain.recurrent_gain_logits = nn.Parameter(torch.zeros(self.brain.n_neurons, dtype=torch.float32, device=self.device))
        self.gain_migration = None

    @classmethod
    def from_centered_parameter_checkpoint(cls, *args, **kwargs):
        raise ValueError('Use from_centered_io_checkpoint for explicit gain migration, or the independent gain loader')

    @classmethod
    def from_initial_parameter_checkpoint(cls, *args, **kwargs):
        raise ValueError('This variant migrates a declared centered outer checkpoint, not the original initialization')

    def migrate_centered_io_payload(self, payload, *, source_identity=None):
        """Validate first, then copy only original I/O; do not rewrite source."""
        CenteredFlyDistillationStudent._outer(payload)
        base=payload['base_parameter_payload'];FlyDistillationStudent._check_checkpoint_header(base)
        source_config=base['brain_config'];target_config=self.brain.config()
        mean=payload.get('fixed_train_mean')
        if (not torch.is_tensor(mean) or mean.dtype!=torch.float32 or mean.shape!=(9721,)
                or not bool(torch.isfinite(mean).all())
                or hashlib.sha256(mean.detach().cpu().contiguous().numpy().tobytes()).hexdigest()!=self.statistics_provenance['mean_raw_sha256']):
            raise ValueError('Original fixed FP32 mean identity differs')
        expected_config=copy.deepcopy(source_config);expected_config['recurrent_gain_limit']=GAIN_LIMIT
        gain=self.brain.recurrent_gain_logits
        if (source_config.get('recurrent_gain_limit',0) != 0 or target_config != expected_config
                or gain is None or gain.shape != (self.brain.n_neurons,) or bool(torch.count_nonzero(gain))):
            raise ValueError('Only gain0 -> explicit gain.1 with all new logits zero is a migration')
        if (connectome_fingerprint(self.brain.graph) != self.graph_fingerprint
                or base['graph_fingerprint'] != self.graph_fingerprint
                or base['ports_manifest_sha256'] != self.interface_provenance.get('manifest_sha256')
                or base['sensory_adapter'] != self.sensory_adapter.specification()
                or payload['statistics_provenance'] != self.statistics_provenance
                or payload['input_gain_initialization'] != self.input_gain_initialization
                or not torch.equal(mean.cpu(), self.brain.input_encoder.fixed_train_mean.detach().cpu())):
            raise ValueError('Source graph/ports/Phi/fixed mean/initialization identity differs')
        source=base['trainable_parameters'];target=dict(self.named_parameters())
        if set(source)!=set(centered.INITIAL_IO_NAMES) or set(target)!=set(source)|{GAIN_NAME}:
            raise ValueError('Exactly the original three I/O tensors plus one per-neuron gain are allowed')
        for name in centered.INITIAL_IO_NAMES:
            value=source[name]
            if (not torch.is_tensor(value) or value.layout!=torch.strided or value.dtype!=torch.float32
                    or value.shape!=target[name].shape or not bool(torch.isfinite(value).all())):
                raise ValueError('Invalid original FP32 I/O tensor:'+name)
        with torch.no_grad():
            for name in centered.INITIAL_IO_NAMES:target[name].copy_(source[name].to(target[name].device))
        self.initial_parameter_source=copy.deepcopy(payload.get('initial_parameter_source'))
        self.gain_migration=dict(source_identity=copy.deepcopy(source_identity),source_schema=payload['schema'],
            source_brain_config=copy.deepcopy(source_config),target_brain_config=copy.deepcopy(target_config),
            copied_parameter_names=list(centered.INITIAL_IO_NAMES),new_parameter=GAIN_NAME,new_parameters=self.brain.n_neurons,
            new_gain_logits_all_zero=True,initial_effective_gain=1.,input_initialization_gain_reapplied=False,
            graph_and_fixed_mean_unchanged=True,optimizer_or_RNG_or_neural_state_restored=False)
        return copy.deepcopy(self.gain_migration)

    @classmethod
    def from_centered_io_checkpoint(cls, checkpoint, directory, *, expected_sha256=MIGRATION_SOURCE_SHA,
                                    device='cpu', port_manifest=None):
        if centered.file_sha(centered.__file__)!=CENTERED_SOURCE_SHA or centered.file_sha(checkpoint)!=expected_sha256:
            raise ValueError('Pinned centered source or explicitly declared migration checkpoint differs')
        payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
        CenteredFlyDistillationStudent._outer(payload)
        # Existing constructor path reuses all source physiology, timing,
        # backend, ports and mean. cls adds only the typed zero gain parameter.
        model=cls._from_base_configuration(payload['base_parameter_payload'],directory,payload['fixed_train_mean'],
            payload['statistics_provenance'],payload['input_gain_initialization'],device=device,port_manifest=port_manifest)
        model.migrate_centered_io_payload(payload,source_identity=dict(path=str(Path(checkpoint).resolve()),sha256=expected_sha256))
        return model

    @staticmethod
    def _outer(payload):
        if (not isinstance(payload,dict) or payload.get('schema')!=PARAMETER_SCHEMA
                or payload.get('centered_student_schema')!=STUDENT_SCHEMA
                or payload.get('gain_centered_source_sha256')!=centered.file_sha(__file__)
                or payload.get('centered_source_sha256')!=CENTERED_SOURCE_SHA
                or centered.file_sha(centered.__file__)!=CENTERED_SOURCE_SHA
                or payload.get('formula')!=centered.FORMULA or payload.get('base_payload_requires_outer_centering') is not True
                or payload.get('postsynaptic_gain_limit')!=GAIN_LIMIT
                or payload.get('optimizer_or_RNG_or_neural_state_restored') is not False):
            raise ValueError('Independent gain-centered parameter schema/source/configuration required')
        if payload.get('base_parameter_payload',{}).get('brain_config',{}).get('recurrent_gain_limit')!=GAIN_LIMIT:
            raise ValueError('Stored brain configuration must explicitly declare gain_limit=.1')

    def parameter_checkpoint_payload(self):
        payload=super().parameter_checkpoint_payload()
        payload.update(schema=PARAMETER_SCHEMA,centered_student_schema=STUDENT_SCHEMA,
            gain_centered_source_sha256=centered.file_sha(__file__),postsynaptic_gain_limit=GAIN_LIMIT,
            gain_migration=copy.deepcopy(self.gain_migration),optimizer_or_RNG_or_neural_state_restored=False)
        self._outer(payload)
        return payload

    def load_parameter_payload(self,payload):
        # Parent validates fixed mean/context and every base tensor before copy;
        # dynamic self._outer selects this independent header, never legacy.
        super().load_parameter_payload(payload)
        self.gain_migration=copy.deepcopy(payload.get('gain_migration'))

    @classmethod
    def from_gain_centered_parameter_checkpoint(cls,checkpoint,directory,*,device='cpu',port_manifest=None,expected_sha256=None):
        if expected_sha256 is not None and centered.file_sha(checkpoint)!=expected_sha256:
            raise ValueError('Explicit gain-centered checkpoint hash differs')
        payload=torch.load(checkpoint,map_location='cpu',weights_only=True);cls._outer(payload)
        base=payload['base_parameter_payload'];FlyDistillationStudent._check_checkpoint_header(base)
        config=base['brain_config'];adapter=base['sensory_adapter']
        if (config.get('require_full_graph') is not True or base.get('graph_fingerprint')!=PINNED_GRAPH_FINGERPRINT
                or config.get('obs_dim')!=9721 or config.get('n_actions')!=19 or config.get('substeps')!=100
                or config.get('dt_ms')!=.2 or config.get('geometry_prefix_dim',0)!=0 or adapter.get('map_pool_shape')!=[12,12]):
            raise ValueError('Full pinned Phi9721/source19/100x.2ms gain graph required')
        options={k:config[k] for k in ('substeps','dt_ms','readout_tau_ms','max_input_rate_hz','rng_seed',
            'encoder_init_scale','surrogate_scale','surrogate_slope_per_mv','parameters')}
        model=cls.from_full_connectome(directory,device=device,port_manifest=port_manifest,map_pool_shape=adapter['map_pool_shape'],
            fixed_train_mean=payload['fixed_train_mean'],statistics_provenance=payload['statistics_provenance'],
            input_gain=payload['input_gain_initialization'],initialize_input_gain=False,
            propagation_backend=config.get('propagation_backend','coo-mm'),**options)
        model.load_parameter_payload(payload)
        return model

    def specification(self):
        result=super().specification()
        result.update(schema=STUDENT_SCHEMA,centered_parent_student_schema=centered.STUDENT_SCHEMA,
            bounded_postsynaptic_gain=self.brain.recurrent_plasticity_specification(),
            added_trainable_parameters=self.brain.n_neurons,parameter_only_checkpoint_schema=PARAMETER_SCHEMA)
        return result
