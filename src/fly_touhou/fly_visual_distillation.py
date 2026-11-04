"""Independent frozen-D95-perception384 -> existing delayed-LIF student.

No Phi9721, ordinary recurrent policy, risk net or observation-to-action path.
This module defines a parameterization; importing it never loads a graph.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import fly_frozen_v5_perception as visual
from .connectome import SOURCE_FILES, SOURCE_REVISION
from .fly_distillation_student import (
    FlyDistillationStudent, PINNED_GRAPH_FINGERPRINT, DEFAULT_PORT_MANIFEST,
    SOURCE19_NATIVE36_ORDER, connectome_fingerprint, load_verified_interfaces,
)
from .whole_brain import WholeBrainStudent

STUDENT_SCHEMA = 'fly-frozen-V5-visual384-gain01-source19-student-v1'
PARAMETER_SCHEMA = 'fly-frozen-V5-visual384-gain01-source19-parameters-v1'
VISUAL_SOURCE_SHA256 = '6d73bbac0547ed3777cce2d51b3739eeb6c0e510129f9df58bb69917eae29350'
PERCEPTION_IDENTITY = '3458102328a9313b1db7c2fdfa8f8897e3ee8a685f3839cad589ee53ca7b4ca7'
STATISTICS_SHA256 = 'c7041498d7d5d4aaa3b68be36c4b96dbd4adf535a2146a421a3c4876d2f16627'
MEAN_SHA256 = 'a12978148fd9654095843ec3a7d492adff87346c0aaefd2657e87799f2344781'
STD_SHA256 = 'fd1063c327ecbac6d2802da14475036e5849d558243bbbed519dc1b2d81de258'
CACHE_INDEX_SHA256 = 'fa7ea1d5e46150b5e31f005474d44a1354a96054565bc1ba78e1cd157132f64d'
DEFAULT_STATISTICS = Path('./outputs/distillation/D95-frozen-visual384-prefix2048-v1/train-statistics.npz')
INPUT_INITIALIZATION_MULTIPLIER = 50.
GAIN_LIMIT = .1


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def perception_identity(configuration, state):
    value = dict(configuration=configuration, state_raw_sha256={k:tensor_sha(v) for k,v in state.items()})
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def load_visual_statistics(path=DEFAULT_STATISTICS):
    """Read only the already committed train000 mean/std, never held data."""
    path = Path(path)
    if file_sha(path) != STATISTICS_SHA256 or file_sha(path.parent/'index.json') != CACHE_INDEX_SHA256:
        raise ValueError('Require the committed visual384 train000 statistics/cache identity')
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {'mean','std','zero_variance'}:
            raise ValueError('Visual384 statistics keys differ')
        mean, std = (torch.from_numpy(archive[k].copy()) for k in ('mean','std'))
    if tensor_sha(mean) != MEAN_SHA256 or tensor_sha(std) != STD_SHA256:
        raise ValueError('Fixed train000 FP32 mean/std bytes differ')
    return mean, std, dict(statistics_sha256=STATISTICS_SHA256, cache_index_sha256=CACHE_INDEX_SHA256,
        population='world000 source rows0..2047 only', rows=2048, heldout_used=False,
        teacher_targets_used=False, mean_raw_sha256=MEAN_SHA256, std_raw_sha256=STD_SHA256,
        zero_variance_std_replacement=1., synthetic_fixture=False)


class FixedStandardizedVisual(nn.Module):
    """Stateless frozen perception followed by persistent fixed (z-mean)/std."""
    output_dim = 384

    def __init__(self, perception, mean, std, provenance):
        super().__init__()
        if not isinstance(perception, visual.FrozenV5Perception):
            raise ValueError('Only the explicit pure-perception module is accepted')
        for name,value in (('mean',mean),('std',std)):
            if (not torch.is_tensor(value) or value.dtype != torch.float32 or value.shape != (384,)
                    or not bool(torch.isfinite(value).all())):
                raise ValueError('Fixed FP32 visual384 statistic required: '+name)
        if not bool((std > 0).all()):
            raise ValueError('Fixed std must be positive; zero-variance columns use1')
        self.perception = perception.requires_grad_(False).eval()
        self.register_buffer('fixed_mean', mean.detach().clone())
        self.register_buffer('fixed_std', std.detach().clone())
        self.statistics_provenance = copy.deepcopy(provenance)

    def forward(self, features, maps, objects):
        return (self.perception(features,maps,objects)-self.fixed_mean)/self.fixed_std

    def specification(self):
        return dict(input='raw V5 features25/maps16x92x96/objects96x29',output_dim=384,
            formula='(FrozenV5Perception(rawV5)-fixed_train_mean)/fixed_train_std',
            Phi_or_raw_object_concatenation=False, frozen_perception=True,
            statistics=copy.deepcopy(self.statistics_provenance))


class VisualFlyDistillationStudent(FlyDistillationStudent):
    """One existing brain; only sensory projection, DN motor and .1 gain train.

    Full construction keeps the original exact graph/port guards. Artificial
    CPU fixtures explicitly set require_full_graph=False and cannot be deployed.
    """
    def __init__(self, graph, interfaces, *, perception, fixed_mean, fixed_std,
                 statistics_provenance, substeps=100, device='cpu', dt_ms=.2,
                 readout_tau_ms=20., max_input_rate_hz=300., rng_seed=613711,
                 encoder_init_scale=.02, surrogate_scale=.01, surrogate_slope_per_mv=.3,
                 parameters=None, recurrent_gain_limit=GAIN_LIMIT, require_full_graph=True,
                 propagation_backend='coo-mm', initialize_input_gain=True):
        nn.Module.__init__(self)
        if (substeps != 100 or dt_ms != .2 or recurrent_gain_limit != GAIN_LIMIT
                or encoder_init_scale != .02 or type(initialize_input_gain) is not bool):
            raise ValueError('Keep100x.2ms, gain_limit.1, core input init.02 and explicit load/init semantics')
        self.graph_fingerprint = connectome_fingerprint(graph)
        self.interface_provenance = dict(interfaces.provenance)
        self.sensory_adapter = FixedStandardizedVisual(perception, fixed_mean, fixed_std, statistics_provenance).to(device)
        if require_full_graph:
            if self.graph_fingerprint != PINNED_GRAPH_FINGERPRINT:
                raise ValueError('Require the unchanged complete source graph')
            default = Path(__file__).resolve().parents[2]/'data/connectome'/DEFAULT_PORT_MANIFEST
            audited = load_verified_interfaces(graph, interfaces.provenance.get('manifest_path',default))
            if (tuple(interfaces.input_indices)!=audited.input_indices or tuple(interfaces.output_indices)!=audited.output_indices):
                raise ValueError('Require the same1024 sensory/1299 descending ports')
            interfaces = audited
            self.interface_provenance = dict(audited.provenance)
            self._check_frozen_frontend(full=True)
        self.brain = WholeBrainStudent(384,19,graph,interfaces.input_indices,interfaces.output_indices,
            device=device,substeps=substeps,dt_ms=dt_ms,readout_tau_ms=readout_tau_ms,
            max_input_rate_hz=max_input_rate_hz,rng_seed=rng_seed,encoder_init_scale=encoder_init_scale,
            surrogate_scale=surrogate_scale,surrogate_slope_per_mv=surrogate_slope_per_mv,
            parameters=parameters,recurrent_gain_limit=GAIN_LIMIT,require_full_graph=require_full_graph,
            propagation_backend=propagation_backend)
        if initialize_input_gain:
            with torch.no_grad():
                self.brain.input_encoder.weight.mul_(INPUT_INITIALIZATION_MULTIPLIER)

    def _check_frozen_frontend(self, *, full):
        adapter = self.sensory_adapter
        if any(p.requires_grad for p in adapter.perception.parameters()):
            raise ValueError('D95 perception must remain frozen')
        if full:
            provenance = adapter.perception.source_provenance() or {}
            statistics = adapter.statistics_provenance
            if (file_sha(visual.__file__) != VISUAL_SOURCE_SHA256
                    or provenance.get('source_checkpoint_sha256') != visual.D95_SHA256
                    or perception_identity(adapter.perception.perception_configuration,adapter.perception.state_dict()) != PERCEPTION_IDENTITY
                    or statistics.get('statistics_sha256') != STATISTICS_SHA256
                    or statistics.get('cache_index_sha256') != CACHE_INDEX_SHA256
                    or statistics.get('population') != 'world000 source rows0..2047 only'
                    or statistics.get('rows') != 2048 or statistics.get('teacher_targets_used') is not False
                    or statistics.get('heldout_used') is not False
                    or statistics.get('synthetic_fixture') is not False
                    or tensor_sha(adapter.fixed_mean) != MEAN_SHA256 or tensor_sha(adapter.fixed_std) != STD_SHA256):
                raise ValueError('Pinned D95 perception or train000-only normalization differs')

    @classmethod
    def from_d95_full_connectome(cls,directory,*,checkpoint,statistics=DEFAULT_STATISTICS,**kwargs):
        perception = visual.FrozenV5Perception.from_d95_checkpoint(checkpoint)
        mean,std,provenance = load_visual_statistics(statistics)
        return cls.from_full_connectome(directory,substeps=100,perception=perception,
            fixed_mean=mean,fixed_std=std,statistics_provenance=provenance,**kwargs)

    @classmethod
    def from_parameter_checkpoint(cls,*args,**kwargs):
        raise ValueError('Use the independent visual384 checkpoint loader')

    def train(self,mode=True):
        super().train(mode)
        self.sensory_adapter.perception.requires_grad_(False).eval()
        return self

    def parameter_checkpoint_payload(self):
        self._check_frozen_frontend(full=self.brain.require_full_graph)
        adapter = self.sensory_adapter
        return dict(schema=PARAMETER_SCHEMA,student_schema=STUDENT_SCHEMA,
            visual_source_sha256=file_sha(__file__), frozen_visual_source_sha256=VISUAL_SOURCE_SHA256,
            graph_fingerprint=connectome_fingerprint(self.brain.graph),
            source_revision=SOURCE_REVISION, source_file_sha256={k:v['sha256'] for k,v in SOURCE_FILES.items()},
            source19_to_native36=list(SOURCE19_NATIVE36_ORDER),
            ports_manifest_sha256=self.interface_provenance.get('manifest_sha256'), brain_config=self.brain.config(),
            frozen_perception=dict(configuration=copy.deepcopy(adapter.perception.perception_configuration),
                provenance=adapter.perception.source_provenance(),
                state={k:v.detach().cpu().clone() for k,v in adapter.perception.state_dict().items()}),
            fixed_mean=adapter.fixed_mean.detach().cpu().clone(), fixed_std=adapter.fixed_std.detach().cpu().clone(),
            statistics_provenance=copy.deepcopy(adapter.statistics_provenance),
            input_initialization_multiplier=INPUT_INITIALIZATION_MULTIPLIER,
            initialization='fresh core .02/sqrt384 sensory W multiplied50 once;bias0;gain logits0;fresh core biasfree motor',
            brain_parameters={k:v.detach().cpu().clone() for k,v in self.brain.named_parameters()},
            optimizer_RNG_or_neural_state_inherited=False)

    @staticmethod
    def _header(payload):
        if (not isinstance(payload,dict) or payload.get('schema')!=PARAMETER_SCHEMA
                or payload.get('student_schema')!=STUDENT_SCHEMA or payload.get('visual_source_sha256')!=file_sha(__file__)
                or payload.get('frozen_visual_source_sha256')!=VISUAL_SOURCE_SHA256
                or payload.get('source_revision')!=SOURCE_REVISION
                or payload.get('source_file_sha256')!={k:v['sha256'] for k,v in SOURCE_FILES.items()}
                or payload.get('source19_to_native36')!=list(SOURCE19_NATIVE36_ORDER)
                or payload.get('input_initialization_multiplier')!=INPUT_INITIALIZATION_MULTIPLIER
                or payload.get('optimizer_RNG_or_neural_state_inherited') is not False
                or any(k in payload for k in ('optimizer','torch_rng','cuda_rng','neural_state','teacher_hidden'))):
            raise ValueError('Independent visual384 parameter-only schema/source required')

    def load_parameter_payload(self,payload):
        self._header(payload)
        adapter = self.sensory_adapter
        frozen = payload.get('frozen_perception',{})
        state = frozen.get('state',{})
        current = adapter.perception.state_dict()
        if (payload.get('graph_fingerprint')!=connectome_fingerprint(self.brain.graph)
                or payload.get('brain_config')!=self.brain.config()
                or payload.get('ports_manifest_sha256')!=self.interface_provenance.get('manifest_sha256')
                or frozen.get('configuration')!=adapter.perception.perception_configuration
                or frozen.get('provenance')!=adapter.perception.source_provenance()
                or payload.get('statistics_provenance')!=adapter.statistics_provenance or set(state)!=set(current)):
            raise ValueError('Visual384 graph/ports/physiology/perception/statistics context differs')
        for name,value in state.items():
            if (not torch.is_tensor(value) or value.layout!=torch.strided or value.dtype!=current[name].dtype
                    or value.shape!=current[name].shape or not torch.equal(value.cpu(),current[name].cpu())):
                raise ValueError('Frozen perception differs: '+name)
        for name,current_value in (('fixed_mean',adapter.fixed_mean),('fixed_std',adapter.fixed_std)):
            value=payload.get(name)
            if not torch.is_tensor(value) or value.dtype!=torch.float32 or not torch.equal(value.cpu(),current_value.cpu()):
                raise ValueError('Fixed normalization differs: '+name)
        stored=payload.get('brain_parameters',{});expected=dict(self.brain.named_parameters())
        if set(stored)!=set(expected):
            raise ValueError('Only exact sensory/motor/gain brain parameter names may load')
        for name,value in stored.items():
            if (not torch.is_tensor(value) or value.layout!=torch.strided or value.dtype!=torch.float32
                    or value.shape!=expected[name].shape or not bool(torch.isfinite(value).all())):
                raise ValueError('Invalid brain parameter: '+name)
        self._check_frozen_frontend(full=self.brain.require_full_graph)
        with torch.no_grad():
            for name,value in stored.items():expected[name].copy_(value.to(self.device))

    @classmethod
    def from_visual_parameter_checkpoint(cls,checkpoint,directory,*,expected_sha256,device='cpu',port_manifest=None):
        if file_sha(checkpoint)!=expected_sha256:
            raise ValueError('Explicit visual384 checkpoint SHA differs')
        payload=torch.load(checkpoint,map_location='cpu',weights_only=True);cls._header(payload)
        c=payload['brain_config'];frozen=payload['frozen_perception']
        if (c.get('require_full_graph') is not True or payload.get('graph_fingerprint')!=PINNED_GRAPH_FINGERPRINT
                or c.get('obs_dim')!=384 or c.get('n_actions')!=19 or c.get('geometry_prefix_dim',0)!=0
                or c.get('substeps')!=100 or c.get('dt_ms')!=.2 or c.get('recurrent_gain_limit')!=GAIN_LIMIT):
            raise ValueError('Full graph visual384/100x.2ms/gain.1/source19 checkpoint required')
        if (perception_identity(frozen['configuration'],frozen['state'])!=PERCEPTION_IDENTITY
                or tensor_sha(payload['fixed_mean'])!=MEAN_SHA256 or tensor_sha(payload['fixed_std'])!=STD_SHA256):
            raise ValueError('Frozen D95 perception/normalization identity differs before graph loading')
        perception=visual.FrozenV5Perception(frozen['configuration'])
        expected=perception.state_dict();stored=frozen['state']
        if set(stored)!=set(expected):raise ValueError('Frozen perception parameter/buffer names differ')
        for name,value in stored.items():
            if (not torch.is_tensor(value) or value.layout!=torch.strided or value.dtype!=expected[name].dtype
                    or value.shape!=expected[name].shape or not bool(torch.isfinite(value).all())):
                raise ValueError('Invalid frozen perception tensor: '+name)
        perception.load_state_dict(frozen['state'],strict=True)
        perception._source_provenance=copy.deepcopy(frozen['provenance'])
        options={k:c[k] for k in ('dt_ms','readout_tau_ms','max_input_rate_hz','rng_seed','encoder_init_scale',
            'surrogate_scale','surrogate_slope_per_mv','parameters')}
        model=cls.from_full_connectome(directory,substeps=100,device=device,port_manifest=port_manifest,
            perception=perception,fixed_mean=payload['fixed_mean'],fixed_std=payload['fixed_std'],
            statistics_provenance=payload['statistics_provenance'],initialize_input_gain=False,
            propagation_backend=c.get('propagation_backend','coo-mm'),**options)
        model.load_parameter_payload(payload)
        return model

    def specification(self):
        result=super().specification()
        result.update(schema=STUDENT_SCHEMA,parameter_only_checkpoint_schema=PARAMETER_SCHEMA,
            decision_path=['raw V5','frozen D95 perception384','fixed train000 standardization',
                'learned384-to-sensory ports','Poisson events','same signed delayed LIF graph','DN biasfree19'],
            frozen_perception_parameter_count=sum(p.numel() for p in self.sensory_adapter.perception.parameters()),
            input_initialization_multiplier=INPUT_INITIALIZATION_MULTIPLIER,
            input_init_nominal_logit_scale='order1 under standardized uncorrelated features;engineering initialization,not measured guarantee',
            teacher_hidden_actor_critic_risk_used=False)
        return result
