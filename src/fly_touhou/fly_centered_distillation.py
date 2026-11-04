"""Fixed train-only sensory centering around the unchanged brain/student/fit.

Importing this module does not import Torch. The explicit student class is
created lazily; inspect/provenance checks require NumPy only. No automatic run.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

STATISTICS_SCHEMA = 'fly-v5-sensory9721-train-prefix-statistics-v1'
CENTERING_SCHEMA = 'fly-fixed-train-sensory-centering-v1'
PARAMETER_SCHEMA = 'fly-centered-Phi9721-source19-parameter-checkpoint-v1'
STUDENT_SCHEMA = 'fly-centered-Phi9721-source19-student-v1'
FORMULA = 'F.linear(Phi-fixed_train_mean,weight,bias)'
DIMENSION = 9721
BANK_ID = '1c6006dc7fbbbe5465e7afcef5f43afc8ffebfc4aba8e7833f4ad2b55eebb4f5'
CACHE_ID = '1e21babfe49bf3b12344c7e8c5bf566f92436c866875a493f36c291c4548d120'
INITIAL_SHA = '6c42592ae4bd43f567f3eb3149f9d35190f12cc251870d266395172a30ab7617'
_CLASSES = None
INITIAL_IO_NAMES = ('brain.input_encoder.weight', 'brain.input_encoder.bias',
                    'brain.action_readout.weight')


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _is_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def validate_input_gain(value):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 < value <= 1000:
        raise ValueError('Explicit finite input initialization gain in (0,1000] required')
    return float(value)


def _population(value):
    required = dict(episode='world000', split='train', source_row_start=0,
                    source_row_count=2048, source_row_end_inclusive=2047,
                    complete_episode=False, heldout_used=False,
                    teacher_labels_used=False, teacher_hidden_or_values_used=False)
    if not isinstance(value, Mapping) or any(type(value.get(k)) is not type(v) or value.get(k) != v for k, v in required.items()):
        raise ValueError('Mean must use only train world000 original rows0..2047, never held/labels/hidden/values')


def _manifest_contract(manifest):
    if not isinstance(manifest, dict) or manifest.get('schema') != STATISTICS_SCHEMA:
        raise ValueError('Explicit versioned train-only sensory-statistics artifact required')
    _population(manifest.get('population'))
    sources = manifest.get('sources', {})
    if (sources.get('cache_identity_sha256') != CACHE_ID or sources.get('bank_identity_sha256') != BANK_ID
            or sources.get('initial_checkpoint_sha256') != INITIAL_SHA):
        raise ValueError('Statistics must bind the immutable D95 cache/bank and original untrained I/O')
    for key in ('cache_identity_file_sha256', 'bank_identity_file_sha256', 'cache_episode_index_sha256'):
        if not _is_sha(sources.get(key)):
            raise ValueError('Statistics source file identity missing: ' + key)
    original = sources.get('source_index', {})
    if not original.get('path') or not _is_sha(original.get('sha256')):
        raise ValueError('Original source episode index identity required')
    records = manifest.get('packet_records')
    if not isinstance(records, list) or len(records) != 64:
        raise ValueError('Exactly64 original32-row train prefix packets required')
    for number, record in enumerate(records):
        if record.get('packet') != number or record.get('source_row_start') != 32 * number or record.get('rows') != 32:
            raise ValueError('Statistics must cover consecutive original row0..2047 without padding')
        for kind in ('cache', 'source', 'label'):
            spec = record.get(kind, {})
            if (not isinstance(spec.get('name'), str) or Path(spec['name']).name != spec['name']
                    or not _is_sha(spec.get('sha256')) or type(spec.get('bytes')) is not int or spec['bytes'] <= 0):
                raise ValueError('Invalid packet provenance: ' + kind)
    for key, dtype, shape in (('phi_mean', 'float32', [9721]), ('initial_port_logit_std', 'float64', [1024])):
        spec = manifest.get('arrays', {}).get(key, {})
        if (spec.get('dtype') != dtype or spec.get('shape') != shape or spec.get('byte_order') != 'little'
                or not _is_sha(spec.get('raw_sha256'))):
            raise ValueError('Statistics dtype/shape/raw identity differs: ' + key)
    json.dumps(manifest, allow_nan=False)


@dataclass(frozen=True)
class SensoryStatistics:
    mean: Any
    initial_port_logit_std: Any
    provenance: dict

    def inspection(self):
        import numpy as np
        import sys
        return dict(schema=STATISTICS_SCHEMA, provenance=copy.deepcopy(self.provenance),
                    mean_elements=len(self.mean), mean_dtype=str(self.mean.dtype),
                    initial_port_logit_std_median=float(np.median(self.initial_port_logit_std)),
                    teacher_or_held_data_used=False, Torch_imported='torch' in sys.modules)


def load_sensory_statistics(path):
    """NPZ and JSON are a required pair; NPZ alone cannot prove train-only use."""
    import numpy as np
    path = Path(path).resolve()
    if path.suffix.lower() == '.npz':
        json_path = path.with_suffix('.json')
    elif path.suffix.lower() == '.json':
        json_path = path
    else:
        raise ValueError('--sensory-statistics must name the paired JSON or NPZ')
    manifest = json.loads(json_path.read_text(encoding='utf-8'))
    _manifest_contract(manifest)
    filename = manifest.get('npz_file')
    if not isinstance(filename, str) or Path(filename).name != filename or not filename.endswith('.npz'):
        raise ValueError('Statistics NPZ must be a same-directory relative filename')
    npz = json_path.parent / filename
    if (path.suffix.lower() == '.npz' and path != npz.resolve()) or not _is_sha(manifest.get('npz_sha256')) or file_sha(npz) != manifest['npz_sha256']:
        raise ValueError('Statistics NPZ path/SHA differs')
    with np.load(npz, allow_pickle=False) as archive:
        if set(archive.files) != {'phi_mean', 'initial_port_logit_std'}:
            raise ValueError('Statistics may contain only fixed mean and population logit std, never row/label/state caches')
        mean, std = archive['phi_mean'], archive['initial_port_logit_std']
    for key, value, dtype, shape in (('phi_mean', mean, np.float32, (9721,)), ('initial_port_logit_std', std, np.float64, (1024,))):
        if value.dtype != dtype or value.shape != shape or not np.isfinite(value).all() or not value.flags.c_contiguous:
            raise ValueError('Invalid fixed statistics array: ' + key)
        if hashlib.sha256(value.tobytes(order='C')).hexdigest() != manifest['arrays'][key]['raw_sha256']:
            raise ValueError('Statistics raw-array SHA differs: ' + key)
    if np.any(np.abs(mean) > 1) or np.any(std < 0) or not float(np.median(std)) > 0:
        raise ValueError('Expected normalized Phi mean and positive median initial drive variability')
    mean = np.frombuffer(mean.tobytes(), dtype=np.float32)
    std = np.frombuffer(std.tobytes(), dtype=np.float64)
    provenance = dict(schema=CENTERING_SCHEMA, statistics_schema=STATISTICS_SCHEMA,
        statistics_manifest_path=str(json_path), statistics_manifest_sha256=file_sha(json_path),
        statistics_npz_path=str(npz.resolve()), statistics_npz_sha256=manifest['npz_sha256'],
        mean_raw_sha256=manifest['arrays']['phi_mean']['raw_sha256'],
        statistics_manifest=manifest, synthetic_fixture=False)
    return SensoryStatistics(mean, std, provenance)


def validate_statistics_against_reader(statistics, reader):
    """Identity/index validation only; do not rescan Phi or teacher targets."""
    provenance = statistics.provenance
    manifest = provenance['statistics_manifest']
    sources = manifest['sources']
    from .fly_sensory_cache import json_sha
    identity = reader._formal_identity()
    folder, episode, index, commit_sha = reader._metadata('world000')
    if (Path(sources['cache_root']).resolve() != reader.root.resolve()
            or Path(sources['bank_root']).resolve() != reader.bank_root.resolve()
            or json_sha(identity) != CACHE_ID or reader.bank.identity_sha != BANK_ID
            or file_sha(reader.root/'identity.json') != sources['cache_identity_file_sha256']
            or reader.bank.identity_file_sha != sources['bank_identity_file_sha256']
            or file_sha(folder/'episode_index.json') != sources['cache_episode_index_sha256']
            or Path(sources['source_index']['path']).resolve() != Path(episode.source['index']).resolve()
            or sources['source_index']['sha256'] != episode.source['index_sha256']):
        raise ValueError('Mean statistics source does not match this immutable cache/episode/bank')
    for number, record in enumerate(manifest['packet_records']):
        for kind, expected in (('cache', index['cache_shards'][number]),
                               ('source', episode.source['shards'][number]),
                               ('label', episode.metadata['label_shards'][number])):
            if record[kind] != {k: expected[k] for k in ('name', 'sha256', 'bytes')}:
                raise ValueError('Mean provenance does not match immutable packet index: ' + kind)
    return dict(cache_COMMITTED_sha256=commit_sha, bank_COMMITTED_sha256=episode.commit_sha,
                verified_prefix_rows=2048, Phi_payloads_reloaded=False, teacher_targets_loaded=False)


def copy_initial_io_parameters(student, payload):
    """Explicit weights-only transfer,never rewrite a source config for a loader.

    All metadata and all three FP32 tensors validate before the first copy.
    The caller applies initialization gain afterwards,once. No source object,
    optimizer,RNG or neural state is changed/restored here.
    """
    import torch
    from .fly_distillation_student import FlyDistillationStudent, connectome_fingerprint
    FlyDistillationStudent._check_checkpoint_header(payload)
    source_config=payload.get('brain_config')
    target_config=student.brain.config()
    if not isinstance(source_config, Mapping):
        raise ValueError('Initial I/O source must retain its original model configuration')
    delta={key:dict(source_present=key in source_config,target_present=key in target_config,
                    source=copy.deepcopy(source_config.get(key)),target=copy.deepcopy(target_config.get(key)))
           for key in set(source_config)|set(target_config)
           if key not in source_config or key not in target_config or source_config[key]!=target_config[key]}
    if set(delta)-{'propagation_backend'}:
        raise ValueError('Initial I/O transfer permits only an explicit numerical backend configuration delta')
    source_backend=source_config.get('propagation_backend','coo-mm')
    target_backend=target_config.get('propagation_backend','coo-mm')
    if source_backend not in ('coo-mm','coo-cusparse-cached') or target_backend not in ('coo-mm','coo-cusparse-cached'):
        raise ValueError('Unsupported initial I/O numerical backend')
    fingerprint=connectome_fingerprint(student.brain.graph)
    if (fingerprint!=student.graph_fingerprint or payload.get('graph_fingerprint')!=fingerprint
            or payload.get('ports_manifest_sha256')!=student.interface_provenance.get('manifest_sha256')
            or payload.get('sensory_adapter')!=student.sensory_adapter.specification()):
        raise ValueError('Initial I/O graph,ports or sensory-adapter identity differs')
    stored=payload.get('trainable_parameters')
    expected=dict(student.named_parameters())
    if not isinstance(stored,Mapping) or set(stored)!=set(INITIAL_IO_NAMES) or set(expected)!=set(INITIAL_IO_NAMES):
        raise ValueError('Exactly the three original sensory weight/bias and motor tensors are required')
    def tensor_sha(value):
        return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    records={}
    for name in INITIAL_IO_NAMES:
        value,parameter=stored[name],expected[name]
        if (not torch.is_tensor(value) or value.layout!=torch.strided or value.dtype!=torch.float32
                or parameter.dtype!=torch.float32 or value.shape!=parameter.shape
                or not bool(torch.isfinite(value).all())):
            raise ValueError('Invalid FP32 initial I/O tensor: '+name)
        records[name]=dict(shape=list(value.shape),dtype='torch.float32',source_raw_sha256=tensor_sha(value))
    with torch.no_grad():
        for name in INITIAL_IO_NAMES:
            expected[name].copy_(stored[name].to(expected[name].device))
    for name in INITIAL_IO_NAMES:
        records[name]['copied_raw_sha256']=tensor_sha(expected[name])
        records[name]['copied_bitwise_exact']=records[name]['copied_raw_sha256']==records[name]['source_raw_sha256']
        if not records[name]['copied_bitwise_exact']:
            raise ValueError('Initial I/O copy was not bitwise exact: '+name)
    return dict(schema='fly-centered-original-IO-parameter-transfer-v1',
        source_configuration=copy.deepcopy(dict(source_config)),target_configuration=copy.deepcopy(target_config),
        configuration_delta=delta,source_backend=source_backend,target_backend=target_backend,
        tensors=records,source_payload_or_metadata_rewritten=False,
        optimizer_or_RNG_or_neural_state_restored=False)


def _student_classes():
    global _CLASSES
    if _CLASSES is not None:
        return _CLASSES
    import torch
    from torch import nn
    from torch.nn import functional as F
    from .fly_distillation_student import FlyDistillationStudent, PINNED_GRAPH_FINGERPRINT

    def mean_sha(mean):
        return hashlib.sha256(mean.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

    def check_context(provenance, mean, full):
        if (not isinstance(provenance, dict) or provenance.get('schema') != CENTERING_SCHEMA
                or provenance.get('statistics_schema') != STATISTICS_SCHEMA
                or not _is_sha(provenance.get('statistics_manifest_sha256'))
                or provenance.get('mean_raw_sha256') != mean_sha(mean)):
            raise ValueError('Fixed mean tensor must retain its explicit statistics/source identity')
        if provenance.get('synthetic_fixture') is True:
            if full:
                raise ValueError('Artificial centering provenance cannot construct a complete-graph student')
        else:
            _manifest_contract(provenance.get('statistics_manifest'))
            if provenance['statistics_manifest']['arrays']['phi_mean']['raw_sha256'] != provenance['mean_raw_sha256']:
                raise ValueError('Mean identity differs from source manifest')
        json.dumps(provenance, allow_nan=False)

    class FixedCenteredLinear(nn.Module):
        """Same weight/bias Parameters; a fixed brain-owned mean buffer."""
        def __init__(self, linear, mean):
            super().__init__()
            if not isinstance(linear, nn.Linear) or isinstance(linear, FixedCenteredLinear):
                raise ValueError('Replace an original Linear exactly once')
            if mean.dtype != torch.float32 or mean.shape != (linear.in_features,) or not bool(torch.isfinite(mean).all()):
                raise ValueError('Finite FP32 fixed mean must match the full original sensory input')
            self.in_features, self.out_features = linear.in_features, linear.out_features
            self.weight, self.bias = linear.weight, linear.bias
            self.register_buffer('fixed_train_mean', mean.detach().to(linear.weight.device).clone())

        def forward(self, value):
            return F.linear(value-self.fixed_train_mean, self.weight, self.bias)

    class CenteredFlyDistillationStudent(FlyDistillationStudent):
        def __init__(self, *args, fixed_train_mean, statistics_provenance, input_gain,
                     initialize_input_gain=True, **kwargs):
            gain = validate_input_gain(input_gain)
            mean = torch.as_tensor(fixed_train_mean)
            if mean.dtype != torch.float32 or mean.shape != (DIMENSION,) or not bool(torch.isfinite(mean).all()) or bool((mean.abs()>1).any()):
                raise ValueError('Fixed train mean must be finite normalized FP32 Phi9721')
            check_context(statistics_provenance, mean, kwargs.get('require_full_graph', True))
            super().__init__(*args, **kwargs)
            if self.brain.obs_dim != DIMENSION or self.brain.recurrent_gain_limit or self.brain.geometry_input_encoder is not None:
                raise ValueError('Centered interface keeps Phi9721 and fixed full-brain I/O-only path')
            self.statistics_provenance = copy.deepcopy(statistics_provenance)
            self.input_gain_initialization = gain
            self.brain.input_encoder = FixedCenteredLinear(self.brain.input_encoder, mean)
            self.initial_parameter_source = None
            if type(initialize_input_gain) is not bool:
                raise ValueError('Gain initialization requires an explicit bool')
            if initialize_input_gain:
                self._initialize_gain()

        def _initialize_gain(self):
            with torch.no_grad():
                self.brain.input_encoder.weight.mul_(self.input_gain_initialization)
            if not bool(torch.isfinite(self.brain.input_encoder.weight).all()):
                raise ValueError('Initialization gain produced nonfinite input parameters')

        @classmethod
        def from_parameter_checkpoint(cls, *args, **kwargs):
            raise ValueError('Use the explicit centered loader; a base checkpoint cannot retain fixed centering')

        @staticmethod
        def _outer(payload):
            if (not isinstance(payload, Mapping) or payload.get('schema') != PARAMETER_SCHEMA
                    or payload.get('centered_student_schema') != STUDENT_SCHEMA
                    or payload.get('centered_source_sha256') != file_sha(__file__)
                    or payload.get('formula') != FORMULA or payload.get('base_payload_requires_outer_centering') is not True):
                raise ValueError('Explicit centered checkpoint schema/source/formula required')

        def parameter_checkpoint_payload(self):
            mean = self.brain.input_encoder.fixed_train_mean
            check_context(self.statistics_provenance, mean, self.brain.require_full_graph)
            return dict(schema=PARAMETER_SCHEMA, centered_student_schema=STUDENT_SCHEMA,
                centered_source_sha256=file_sha(__file__), formula=FORMULA,
                fixed_train_mean=mean.detach().cpu().clone(),
                statistics_provenance=copy.deepcopy(self.statistics_provenance),
                input_gain_initialization=self.input_gain_initialization,
                gain_is_initialization_only=True, base_payload_requires_outer_centering=True,
                initial_parameter_source=copy.deepcopy(self.initial_parameter_source),
                base_parameter_payload=FlyDistillationStudent.parameter_checkpoint_payload(self))

        def load_parameter_payload(self, payload):
            self._outer(payload)
            mean = payload.get('fixed_train_mean')
            if (not torch.is_tensor(mean) or mean.dtype != torch.float32 or mean.shape != (DIMENSION,)
                    or not bool(torch.isfinite(mean).all()) or bool((mean.abs()>1).any())):
                raise ValueError('Centered checkpoint must retain its finite FP32 mean')
            provenance=payload.get('statistics_provenance')
            check_context(provenance, mean, self.brain.require_full_graph)
            if (provenance != self.statistics_provenance or validate_input_gain(payload.get('input_gain_initialization')) != self.input_gain_initialization
                    or payload.get('gain_is_initialization_only') is not True
                    or not torch.equal(mean.cpu(), self.brain.input_encoder.fixed_train_mean.detach().cpu())):
                raise ValueError('Centering identity/gain differs; no parameter was copied')
            # The original loader validates all parameter fields before copying.
            FlyDistillationStudent.load_parameter_payload(self, payload.get('base_parameter_payload'))
            self.initial_parameter_source=copy.deepcopy(payload.get('initial_parameter_source'))

        @classmethod
        def _from_base_configuration(cls, payload, directory, mean, provenance, gain, *, device='cpu',
                                     port_manifest=None, propagation_backend=None):
            FlyDistillationStudent._check_checkpoint_header(payload)
            if (not torch.is_tensor(mean) or mean.dtype != torch.float32 or mean.shape != (DIMENSION,)
                    or not bool(torch.isfinite(mean).all()) or bool((mean.abs()>1).any())):
                raise ValueError('Validate fixed mean before allocating the complete graph')
            check_context(provenance, mean, True)
            validate_input_gain(gain)
            config=payload.get('brain_config', {});adapter=payload.get('sensory_adapter', {})
            if (config.get('require_full_graph') is not True or payload.get('graph_fingerprint') != PINNED_GRAPH_FINGERPRINT
                    or config.get('n_actions') != 19 or config.get('obs_dim') != DIMENSION
                    or config.get('geometry_prefix_dim', 0) != 0 or config.get('recurrent_gain_limit', 0) != 0
                    or config.get('substeps') != 100 or config.get('dt_ms') != .2 or adapter.get('map_pool_shape') != [12,12]):
                raise ValueError('Only the existing complete Phi9721/source19/100x.2ms/gain0 configuration is accepted')
            options={k:config[k] for k in ('substeps','dt_ms','readout_tau_ms','max_input_rate_hz','rng_seed',
                'encoder_init_scale','surrogate_scale','surrogate_slope_per_mv','parameters')}
            options['propagation_backend']=propagation_backend or config.get('propagation_backend','coo-mm')
            model=cls.from_full_connectome(directory,device=device,port_manifest=port_manifest,
                map_pool_shape=adapter['map_pool_shape'],fixed_train_mean=mean,statistics_provenance=provenance,
                input_gain=gain,initialize_input_gain=False,**options)
            return model

        @classmethod
        def from_initial_parameter_checkpoint(cls, checkpoint, directory, *, statistics, input_gain,
                                              device='cpu', port_manifest=None, propagation_backend='coo-mm'):
            if file_sha(checkpoint) != INITIAL_SHA:
                raise ValueError('New pilot starts only from pinned original I/O, never failed512 parameters')
            payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
            model=cls._from_base_configuration(payload,directory,torch.from_numpy(statistics.mean.copy()),
                statistics.provenance,input_gain,device=device,port_manifest=port_manifest,
                propagation_backend=propagation_backend)
            transfer=copy_initial_io_parameters(model,payload)
            model._initialize_gain()
            for name in ('brain.input_encoder.bias','brain.action_readout.weight'):
                value=dict(model.named_parameters())[name]
                digest=hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                transfer['tensors'][name]['after_gain_raw_sha256']=digest
                if digest!=transfer['tensors'][name]['source_raw_sha256']:
                    raise ValueError('Sensory bias/motor changed during input-weight initialization')
            model.initial_parameter_source=dict(path=str(Path(checkpoint).resolve()),sha256=INITIAL_SHA,
                transfer=transfer,source_backend=transfer['source_backend'],target_backend=transfer['target_backend'],
                sensory_weight_scaled_once=True,sensory_bias_and_motor_exactly_inherited=True,
                input_weight_raw_sha256_after_gain=hashlib.sha256(model.brain.input_encoder.weight.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
                optimizer_or_RNG_or_neural_state_restored=False)
            return model

        @classmethod
        def from_centered_parameter_checkpoint(cls, checkpoint, directory, *, device='cpu',port_manifest=None):
            payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
            cls._outer(payload)
            mean=payload.get('fixed_train_mean');provenance=payload.get('statistics_provenance')
            model=cls._from_base_configuration(payload['base_parameter_payload'],directory,mean,provenance,
                payload['input_gain_initialization'],device=device,port_manifest=port_manifest)
            model.load_parameter_payload(payload)
            return model

        def specification(self):
            base=FlyDistillationStudent.specification(self)
            return dict(base, schema=STUDENT_SCHEMA, base_student_schema=base['schema'],
                sensory_centering=dict(formula=FORMULA,mean_trainable=False,
                    mean_raw_sha256=self.statistics_provenance['mean_raw_sha256'],
                    input_gain_initialization=self.input_gain_initialization,gain_is_initialization_only=True,
                    added_trainable_parameters=0,fixed_mean_buffer_bytes=4*DIMENSION,
                    statistics_provenance=copy.deepcopy(self.statistics_provenance)))

    FixedCenteredLinear.__module__=CenteredFlyDistillationStudent.__module__=__name__
    _CLASSES={'FixedCenteredLinear':FixedCenteredLinear,'CenteredFlyDistillationStudent':CenteredFlyDistillationStudent}
    return _CLASSES


def __getattr__(name):
    if name in ('FixedCenteredLinear','CenteredFlyDistillationStudent'):
        return _student_classes()[name]
    raise AttributeError(name)


def main(argv=None):
    """Fresh finite pilot; default inspect imports no Torch or full graph."""
    import argparse
    from dataclasses import asdict
    import os
    import sys
    import traceback
    from .fly_cached_distillation import (CachedFitConfig, FitGuard, atomic_json,
        import_reader_preserving_CVD, inspect_episodes, fit_cached_student, memory_plan, process_identity)
    root=Path(__file__).resolve().parents[2]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-root',type=Path,required=True)
    parser.add_argument('--bank',type=Path,required=True)
    parser.add_argument('--sensory-statistics',type=Path,required=True)
    parser.add_argument('--input-gain',type=float,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--execute-cuda',action='store_true')
    parser.add_argument('--initial-parameters',type=Path,default=Path('./outputs/distillation/D95-fly-Phi9721-continuous2048-v1/initial-parameters.pt'))
    parser.add_argument('--connectome',type=Path,default=root/'data/connectome')
    parser.add_argument('--split',type=Path,default=root/'artifacts/engineering/B1-autonomous-dodge-20261024/fly-migration-20261026/frozen-episode-split-v1.json')
    parser.add_argument('--propagation-backend',choices=('coo-mm','coo-cusparse-cached'),default='coo-mm')
    parser.add_argument('--capture-frozen-replays',action='store_true',
                        help='Only CACHED;capture no-grad frozen replays,never the training backward')
    parser.add_argument('--max-seconds',type=float,default=7200.)
    parser.add_argument('--gpu-fraction',type=float,default=.2)
    parser.add_argument('--minimum-ram-GiB',type=float,default=10.)
    parser.add_argument('--minimum-disk-GiB',type=float,default=4.)
    args=parser.parse_args(argv)
    if args.output.exists():
        raise ValueError('New output required; no resume or automatic restart')
    config=CachedFitConfig(max_rows=2048,validation_rows=2048,max_updates=512,
        max_model_decisions=10240,max_seconds=args.max_seconds,tbptt=4,learning_rate=1e-4,
        checkpoint_every_updates=100,gpu_fraction=args.gpu_fraction,
        minimum_ram_GiB=args.minimum_ram_GiB,minimum_disk_GiB=args.minimum_disk_GiB,rng_seed=613711).validate()
    if config.max_seconds>7200 or config.minimum_ram_GiB<10 or config.minimum_disk_GiB<4:
        raise ValueError('Keep the existing finite7200s/RAM10GiB/disk4GiB pilot bounds')
    guard=FitGuard(args.output,config)
    try:
        if 'torch' in sys.modules:
            raise ValueError('Fresh CLI must load the CPU reader before Torch')
        import psutil
        process=psutil.Process();process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        process.cpu_affinity(process.cpu_affinity()[:2])
        launch=dict(schema='fly-centered-sensory-finite-pilot-launch-v1',**process_identity(),
            utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
            mode='execute_cuda' if args.execute_cuda else 'NoTorch_inspect',native_worlds=0,
            source=__file__,source_sha256=file_sha(__file__))
        gain=validate_input_gain(args.input_gain)
        if args.capture_frozen_replays and args.propagation_backend!='coo-cusparse-cached':
            raise ValueError('--capture-frozen-replays requires explicit CACHED propagation')
        capture_setup=dict(requested=args.capture_frozen_replays,planned_neural_forwards=4 if args.capture_frozen_replays else 0,
            attempted=False,completed=False,completed_neural_forwards=0,
            failed_attempt_paid_forward_count='unknown0..4 if setup begins and fails',
            scope='three warmups plus one recording;separate from real data/history forwards')
        guard.check('before_statistics')
        statistics=load_sensory_statistics(args.sensory_statistics)
        module,CVD=import_reader_preserving_CVD()
        reader=module.FullEpisodeSensoryReader(args.cache_root,args.bank,guard)
        inspection=inspect_episodes(reader,['world000'],['world010'])
        mean_binding=validate_statistics_against_reader(statistics,reader)
        split=json.loads(args.split.read_text(encoding='utf-8'))
        if (split.get('bank_identity_sha256')!=BANK_ID or 'world000' not in split.get('train',[])
                or 'world010' not in split.get('heldout',[])):
            raise ValueError('Same frozen train000/held010 split required')
        if file_sha(args.initial_parameters)!=INITIAL_SHA:
            raise ValueError('Only the original initial6c425 I/O artifact may initialize this pilot')
        provenance=dict(schema='fly-centered-sensory-finite-pilot-v1',formula=FORMULA,
            parameter_checkpoint_schema=PARAMETER_SCHEMA,statistics=statistics.inspection(),
            mean_source_binding=mean_binding,input_gain_initialization=gain,
            input_gain_applied_to='sensory weight once at initialization;bias/motor inherited exactly',
            initial_parameter_checkpoint=str(args.initial_parameters.resolve()),initial_parameter_sha256=INITIAL_SHA,
            propagation_backend=args.propagation_backend,
            capture_frozen_replays=args.capture_frozen_replays,capture_setup=capture_setup,
            neural_forward_budget=dict(data_cap=config.max_model_decisions,
                capture_setup_cap=capture_setup['planned_neural_forwards'],
                total_cap=config.max_model_decisions+capture_setup['planned_neural_forwards']),
            cached_backend_GPU_qualification='required independently;not established by this preparation/CPU inspect',
            CVD_import_sequence=CVD,split_path=str(args.split.resolve()),split_SHA256=file_sha(args.split),
            config=asdict(config),inspection=inspection,
            memory_plan=dict(memory_plan(4),additional_fixed_mean_buffer_bytes=4*DIMENSION),launch=launch,
            train=['world000'],heldout=['world010'],Torch_imported=False,
            lifecycle='existing fit_cached_student;own state and finite-stop/atomic virtual checkpoint save unchanged')
        if not args.execute_cuda:
            args.output.mkdir(parents=True,exist_ok=False)
            atomic_json(args.output/'inspect.json',dict(provenance,status='NoTorch_inspect_only',GPU_or_optimizer_started=False))
            atomic_json(args.output/'launch.json',launch)
            return 0
        guard.check('before_Torch')
        if args.propagation_backend=='coo-cusparse-cached':
            if os.environ.get('CUBLAS_WORKSPACE_CONFIG',':4096:8')!=':4096:8':
                raise ValueError('Cached deterministic backend needs CUBLAS_WORKSPACE_CONFIG=:4096:8 in a fresh process')
            os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        import torch
        if torch.cuda.is_initialized():
            raise ValueError('Fresh CUDA context required')
        torch.set_num_threads(2);torch.manual_seed(config.rng_seed)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        if torch.is_autocast_enabled() or torch.is_autocast_enabled('cpu'):
            raise ValueError('Strict FP32 without autocast required')
        torch.cuda.set_per_process_memory_fraction(config.gpu_fraction,0)
        student=_student_classes()['CenteredFlyDistillationStudent'].from_initial_parameter_checkpoint(
            args.initial_parameters,args.connectome,statistics=statistics,input_gain=gain,
            propagation_backend=args.propagation_backend,device='cpu')
        guard.check('before_student_CUDA');student.to('cuda:0')
        if args.capture_frozen_replays:
            guard.check('before_frozen_capture_setup');capture_setup['attempted']=True
            capture_setup['completed_neural_forwards']=None
            student.brain.enable_cuda_capture(1)
            capture_setup.update(completed=True,completed_neural_forwards=4)
        report=fit_cached_student(student,reader,['world000'],['world010'],args.output,config,guard=guard)
        provenance.update(Torch_imported=True,CUDA=report['CUDA'],student=student.specification(),
            data_neural_forwards=report['total_model_decisions'],
            total_neural_forwards_including_capture_setup=report['total_model_decisions']+capture_setup['completed_neural_forwards'],
            training_uses_gradient_eager_path=True)
        atomic_json(args.output/'launch.json',launch)
        atomic_json(args.output/'provenance.json',provenance)
        return 0 if report['status']=='completed_finite_cached_distillation' else 2
    except BaseException as error:
        if not args.output.exists():
            args.output.mkdir(parents=True,exist_ok=False)
        atomic_json(args.output/'failure.json',dict(schema='fly-centered-sensory-finite-pilot-v1',
            status='failed_preparation',reason=str(error),traceback=traceback.format_exc(),
            automatic_restart=False,native_worlds_started=0,Torch_imported='torch' in sys.modules,
            capture_setup=locals().get('capture_setup')))
        if 'launch' in locals():
            atomic_json(args.output/'launch.json',launch)
        return 1
