"""Prepared typed FullSensory+pair-geometry family; never starts a job.

All prior sensory/body/head Parameters and named Adam states are retained.
Only the nine fresh pair-VALUE residual tensors are added. Old schemas contain
only their genuine old computation,with geometry saved in a separate envelope.
"""
from __future__ import annotations
import copy
import hashlib
import math
from pathlib import Path
import torch
from torch import nn

from .continuous_full_sensory_model import (
    FullTrainableVisual,FullSensoryActorCritic,FRONTEND_BATCH,FULL_SENSORY_GROUP,
    PARAMETER_SCHEMA as FULL_PARAMETER_SCHEMA,TRAINING_SCHEMA as FULL_TRAINING_SCHEMA,
)
from .continuous_pair_geometry_sensory import PairGeometryResidualObjects,SCHEMA as GEOMETRY_SCHEMA
from .continuous_action_query_learning_model import QUERY_GROUP,PARAMETER_SCHEMA as QUERY_PARAMETER_SCHEMA
from .continuous_object_learning_model import _cpu,_same,_validate_state,_DYNAMIC
from .continuous_policy_cost_routing import PolicyCostRoutingActorCritic

PARAMETER_SCHEMA='continuous-full-sensory-pair28value-DN-deep128x3-threevalue-parameters-v2'
TRAINING_SCHEMA='continuous-full-sensory-pair28value-DN-deep128x3-named-Adam-training-v2'
GEOMETRY_GROUP='sensory_pair_geometry'
ATTENTION_PREFIX='sensory_adapter.object_attention.'
INITIALIZATION_SCHEMA='pair-geometry-value-hidden-initialization-v1'
D95_SHA256='d95fe710068483fc81d8525d89bccf72e61d209c7e1bb442ea66d022b9c1d06c'
HIDDEN_SOURCE_KEYS=tuple(f'risk.network.{layer}.{part}' for layer in (0,2,4) for part in ('weight','bias'))
HIDDEN_TARGET_KEYS=tuple(f'pair_embedding.{layer}.{part}' for layer in (0,2,4) for part in ('weight','bias'))


def _random_initialization(seed):
    return dict(schema=INITIALIZATION_SCHEMA,mode='random',seed=seed,architecture=[28,128,128,128],activation='SiLU',
        copied_tensors=0,scalar_risk_head_copied=False,all_hidden_tensors_trainable=True,runtime_risk_scores_as_inputs=False)


def _validated_initialization(value,seed):
    if (not isinstance(value,dict) or value.get('schema')!=INITIALIZATION_SCHEMA
            or value.get('architecture')!=[28,128,128,128] or value.get('activation')!='SiLU'
            or value.get('scalar_risk_head_copied') is not False
            or value.get('all_hidden_tensors_trainable') is not True
            or value.get('runtime_risk_scores_as_inputs') is not False):
        raise ValueError('Explicit hidden-feature initialization provenance required')
    if value.get('mode')=='random':
        if value!=_random_initialization(seed):raise ValueError('Random initialization seed/provenance differs')
    elif value.get('mode')=='D95-point-hidden6-pretrained':
        from .thrl_pair_risk import QUERY_SCHEMA
        hashes=value.get('copied_source_tensor_sha256',{})
        if (value.get('source_checkpoint_sha256')!=D95_SHA256 or value.get('source_query_schema')!=QUERY_SCHEMA
                or value.get('copied_tensors')!=6 or value.get('copied_source_keys')!=list(HIDDEN_SOURCE_KEYS)
                or value.get('copied_target_keys')!=list(HIDDEN_TARGET_KEYS) or set(hashes)!=set(HIDDEN_SOURCE_KEYS)
                or any(not isinstance(v,str) or len(v)!=64 or any(c not in '0123456789abcdef' for c in v) for v in hashes.values())
                or value.get('pool_and_zero_projection_preserved') is not True):
            raise ValueError('Pinned D95 six-hidden-tensor initialization provenance differs')
    else:raise ValueError('Unknown geometry initialization mode')
    return copy.deepcopy(value)


def inherited_parameter_name(name):
    return ATTENTION_PREFIX+'base_action_query.'+name[len(ATTENTION_PREFIX):] if name.startswith(ATTENTION_PREFIX) else name


def initialize_geometry_from_D95_hidden(model,checkpoint,expected_sha):
    """Optional one-time pretrained FEATURE initialization,never risk answers.

    Copy only the original point MLP's three hidden Linear layers. The scalar
    margin layer,risk aggregation,teacher GRU/actions and optimizer are absent.
    Invoke on a fresh residual before training;all copied Parameters keep their
    identity/requires_grad and the caller's new Adam group remains untouched.
    """
    from .thrl_parameter_transfer import read_evaluation_source
    from .thrl_pair_risk import QUERY_SCHEMA
    if not isinstance(model,FullPairGeometryActorCritic) or expected_sha!=D95_SHA256:
        raise ValueError('Fresh typed FullPair model and explicitly pinned D95 SHA required')
    if model.sensory_adapter.geometry_initialization['mode']!='random':
        raise ValueError('Feature initialization is a one-time explicit migration,not a resume operation')
    geometry=model.geometry_attention
    if (any(p.grad is not None for p in model.geometry_parameters())
            or not all(p.requires_grad for p in model.sensory_adapter.parameters())
            or bool(torch.count_nonzero(geometry.final_projection.weight))
            or bool(torch.count_nonzero(geometry.final_projection.bias))):
        raise ValueError('Initialize before learning with all sensory trainable and an untouched zero projection')
    source=read_evaluation_source(torch,checkpoint,expected_sha)
    wrapper=source['model_config']['pair_risk_wrapper']
    if wrapper.get('query_schema')!=QUERY_SCHEMA:
        raise ValueError('D95 point feature schema does not match observation_queries28')
    parameters=dict(geometry.named_parameters());pairs=[];hashes={}
    for source_key,target_key in zip(HIDDEN_SOURCE_KEYS,HIDDEN_TARGET_KEYS):
        value=source['model'].get(source_key);target=parameters[target_key]
        if (not torch.is_tensor(value) or value.dtype!=torch.float32 or value.shape!=target.shape
                or not bool(torch.isfinite(value).all())):
            raise ValueError('Exact D95 point-hidden tensor required:'+source_key)
        pairs.append((target,value))
        hashes[source_key]=hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    pool=geometry.object_pool_query.detach().clone()
    projection={name:value.detach().clone() for name,value in geometry.final_projection.state_dict().items()}
    with torch.no_grad():
        for target,value in pairs:target.copy_(value.to(device=target.device))
    if (any(not torch.equal(target.detach().cpu(),value.detach().cpu()) for target,value in pairs)
            or not torch.equal(pool,geometry.object_pool_query.detach())
            or any(not torch.equal(value,geometry.final_projection.state_dict()[name]) for name,value in projection.items())):
        raise RuntimeError('Feature copy or preserved pool/zero projection differs')
    provenance=dict(schema=INITIALIZATION_SCHEMA,mode='D95-point-hidden6-pretrained',
        architecture=[28,128,128,128],activation='SiLU',source_checkpoint=str(Path(checkpoint).resolve()),
        source_checkpoint_sha256=expected_sha,source_query_schema=QUERY_SCHEMA,
        copied_tensors=6,copied_source_keys=list(HIDDEN_SOURCE_KEYS),copied_target_keys=list(HIDDEN_TARGET_KEYS),
        copied_source_tensor_sha256=hashes,excluded_scalar_risk_keys=['risk.network.6.weight','risk.network.6.bias'],
        scalar_risk_head_copied=False,all_hidden_tensors_trainable=True,runtime_risk_scores_as_inputs=False,
        pool_and_zero_projection_preserved=True,optimizer_read_or_modified=False,teacher_model_constructed=False)
    model.sensory_adapter.geometry_initialization=_validated_initialization(provenance,model._geometry_configuration['seed'])
    return copy.deepcopy(provenance)


class FullPairTrainableVisual(FullTrainableVisual):
    def __init__(self,source,*,geometry_seed=71967):
        nn.Module.__init__(self)
        from .action_query_residual_objects import ActionQueryResidualObjects
        if (not isinstance(source,FullTrainableVisual) or isinstance(source,FullPairTrainableVisual)
                or not isinstance(source.object_attention,ActionQueryResidualObjects)):
            raise TypeError('An unwrapped loaded FullSensory frontend is required')
        self.map_encoder,self.feature_proj=source.map_encoder,source.feature_proj
        self.relu=source.relu
        self.register_buffer('fixed_mean',source.fixed_mean);self.register_buffer('fixed_std',source.fixed_std)
        for name in ('perception_configuration','source_provenance','statistics_provenance'):
            setattr(self,name,copy.deepcopy(getattr(source,name)))
        self.source_static_cache_identity=source.source_static_cache_identity
        self.evidence_identity=source.evidence_identity;self.static_cache_identity=source.static_cache_identity
        self.object_attention=PairGeometryResidualObjects(source.object_attention,seed=geometry_seed).to(source.device)
        self.geometry_initialization=_random_initialization(geometry_seed)
        self.requires_grad_(True)
        self.train(source.training)  # inherited Full.train never freezes CNN/numeric

    def perception_parameter_groups(self):
        action=self.object_attention.base_action_query
        return dict(CNN=tuple(p for name,p in self.map_encoder.named_parameters() if not name.startswith('spatial_attention.')),
            spatial_attention=tuple(p for name,p in self.map_encoder.named_parameters() if name.startswith('spatial_attention.')),
            numeric=tuple(self.feature_proj.parameters()),object=tuple(action.base_object_attention.parameters()),
            query=tuple(p for name,p in action.named_parameters() if not name.startswith('base_object_attention.')),
            pair_geometry=tuple(p for name,p in self.object_attention.named_parameters() if not name.startswith('base_action_query.')))

    def specification(self):
        return self._specification_for_initialization(self.geometry_initialization)

    def _specification_for_initialization(self,initialization):
        spec=FullTrainableVisual.specification(self)
        initialized=initialization['mode']=='D95-point-hidden6-pretrained'
        spec.update(schema='V5-full-trainable-pair28value-object-query384-canonical30-v1',
            trainable=[*spec['trainable'],'pair-geometry value embedding/pooling/residual'],
            pair_geometry=dict(self.object_attention.specification(),current_model_optimizer_collector_runner_integrated=True,
                value_embedding=('D95 point-hidden6 pretrained28->128SiLU->128SiLU->128SiLU;all trainable;no final scalar risk layer'
                    if initialized else 'random28->128SiLU->128SiLU->128SiLU;all trainable;no D95 hidden weights or scalar risk layer'),
                hidden_initialization=copy.deepcopy(initialization),hidden_weights_frozen=False,
                integration='same raw/canonical30/frontend384/fullbrain/DN path;runner dispatch is external'))
        return spec


class FullPairGeometryActorCritic(FullSensoryActorCritic):
    source_schema=PARAMETER_SCHEMA

    def __init__(self,base,*,geometry_seed=71967):
        nn.Module.__init__(self)
        if not isinstance(base,FullSensoryActorCritic) or isinstance(base,FullPairGeometryActorCritic):
            raise TypeError('A loaded unwrapped FullSensoryActorCritic is required')
        self._source_full_configuration=copy.deepcopy(base.configuration())
        if (self._source_full_configuration['PairGeometry_integrated'] is not False
                or base.risk_head.beta!=8. or base.policy_cost_gradient!='supervised-only'):
            raise ValueError('Preserve prior Full graph/ports/beta8/threevalue/routing without a preexisting geometry branch')
        for name in ('_source_query_configuration','_source_object_configuration','_query_configuration'):
            setattr(self,name,copy.deepcopy(getattr(base,name)))
        self._source_static_cache_identity=base._source_static_cache_identity
        self._base_metadata=base._base_metadata
        self.core,self.actor,self.critic,self.risk_head=base.core,base.actor,base.critic,base.risk_head
        self.initialization_seed,self.policy_cost_gradient=base.initialization_seed,base.policy_cost_gradient
        self._geometry_configuration=dict(schema=GEOMETRY_SCHEMA,seed=geometry_seed)
        self.sensory_adapter=FullPairTrainableVisual(base.sensory_adapter,geometry_seed=geometry_seed)
        self.train(base.training)

    @property
    def geometry_attention(self):return self.sensory_adapter.object_attention

    @property
    def base_action_attention(self):return self.geometry_attention.base_action_query

    def geometry_parameters(self):
        return [p for name,p in self.geometry_attention.named_parameters() if not name.startswith('base_action_query.')]

    def geometry_state(self):
        return {name:value for name,value in self.geometry_attention.state_dict().items() if not name.startswith('base_action_query.')}

    def _current_query_state(self):
        return {name:value for name,value in self.base_action_attention.state_dict().items() if not name.startswith('base_object_attention.')}

    def _nested_query_payload(self):
        """Explicitly address bare ActionQuery;never pass pair keys to old schema."""
        body=_cpu(self._base_metadata)
        body.update(core_parameters=_cpu(dict(self.core.named_parameters())),actor=_cpu(self.actor.state_dict()),
            critic=_cpu(self.critic.state_dict()),risk_head=_cpu(self.risk_head.state_dict()))
        old=dict(schema=self._source_object_configuration['schema'],configuration=copy.deepcopy(self._source_object_configuration),
            base_parameters=body,object_attention=_cpu(self.base_action_attention.base_object_attention.state_dict()),
            static_schema='D95-frozen-numeric-map-static281-v1',static_cache_identity=self._source_static_cache_identity,
            contains_optimizer=False,contains_neural_state=False,contains_teacher_hidden=False)
        return dict(schema=QUERY_PARAMETER_SCHEMA,configuration=copy.deepcopy(self._source_query_configuration),
            object_model_payload=old,query_configuration=copy.deepcopy(self._query_configuration),query_state=_cpu(self._current_query_state()),
            static_schema=old['static_schema'],static_cache_identity=self._source_static_cache_identity,
            contains_optimizer=False,contains_neural_state=False,contains_teacher_hidden=False)

    def _nested_full_payload(self):
        """Genuine old Full payload including authoritative current CNN/numeric."""
        return dict(schema=FULL_PARAMETER_SCHEMA,configuration=copy.deepcopy(self._source_full_configuration),
            query_model_payload=self._nested_query_payload(),
            current_CNN_state=_cpu(self.sensory_adapter.map_encoder.state_dict()),
            current_numeric_state=_cpu(self.sensory_adapter.feature_proj.state_dict()),current_sensory_state_authoritative=True,
            nested_frozen_CNN_snapshot='initialization reference only;not the effective full-sensory frontend',
            contains_optimizer=False,contains_neural_state=False,contains_teacher_hidden=False)

    def configuration(self):
        return self._configuration_for_initialization(self.sensory_adapter.geometry_initialization)

    def _configuration_for_initialization(self,initialization):
        cfg=FullSensoryActorCritic.configuration(self)
        cfg['perception']=self.sensory_adapter._specification_for_initialization(initialization)
        cfg.update(schema=PARAMETER_SCHEMA,PairGeometry_integrated=True,
            source_full_parameter_schema=FULL_PARAMETER_SCHEMA,geometry_configuration=copy.deepcopy(self._geometry_configuration),
            geometry_initialization=copy.deepcopy(initialization),
            geometry_parameters=len(self.geometry_parameters()),
            geometry_source='pure observation_queries;optional hidden-weight initialization recorded;no runtime risk scores or chosen actions')
        return cfg

    def parameter_checkpoint_payload(self):
        return dict(schema=PARAMETER_SCHEMA,configuration=self.configuration(),full_sensory_model_payload=self._nested_full_payload(),
            geometry_configuration=copy.deepcopy(self._geometry_configuration),geometry_state=_cpu(self.geometry_state()),
            geometry_initialization=copy.deepcopy(self.sensory_adapter.geometry_initialization),
            contains_optimizer=False,contains_neural_state=False,contains_teacher_hidden=False)

    def load_parameter_payload(self,payload):
        if (payload.get('schema')!=PARAMETER_SCHEMA
                or not _same(payload.get('geometry_configuration'),self._geometry_configuration)):
            raise ValueError('Exact independent FullPair model/geometry configuration required')
        initialization=_validated_initialization(payload.get('geometry_initialization'),self._geometry_configuration['seed'])
        # Saved feature weights are authoritative at resume;do not reload D95
        # or pretend the learned values are fresh random initialization.
        if not _same(payload.get('configuration'),self._configuration_for_initialization(initialization)):
            raise ValueError('Effective FullPair configuration/initialization provenance differs')
        full=payload['full_sensory_model_payload'];q=full['query_model_payload'];old=q['object_model_payload'];body=old['base_parameters']
        if (full.get('schema')!=FULL_PARAMETER_SCHEMA or not _same(full.get('configuration'),self._source_full_configuration)
                or full.get('current_sensory_state_authoritative') is not True
                or q.get('schema')!=QUERY_PARAMETER_SCHEMA or not _same(q.get('configuration'),self._source_query_configuration)
                or not _same(old.get('configuration'),self._source_object_configuration)
                or not _same(q.get('query_configuration'),self._query_configuration)
                or q.get('static_cache_identity')!=self._source_static_cache_identity
                or not _same({k:v for k,v in body.items() if k not in _DYNAMIC},self._base_metadata)):
            raise ValueError('Genuine prior Full/Query graph/port/source context differs')
        _validate_state(dict(self.core.named_parameters()),body['core_parameters'],'core')
        for name,module in (('actor',self.actor),('critic',self.critic),('risk_head',self.risk_head)):
            _validate_state(module.state_dict(),body[name],name)
        for key in ('layers.DN_mean','layers.DN_scale'):
            if not _same(self.risk_head.state_dict()[key],body['risk_head'][key]):raise ValueError('Fixed DN risk normalization differs')
        _validate_state(self.base_action_attention.base_object_attention.state_dict(),old['object_attention'],'bare_object_attention')
        _validate_state(self._current_query_state(),q['query_state'],'bare_action_query')
        if not _same(self._current_query_state()['action_descriptors'],q['query_state']['action_descriptors']):
            raise ValueError('Source18 descriptors differ')
        _validate_state(self.sensory_adapter.map_encoder.state_dict(),full['current_CNN_state'],'current_CNN')
        _validate_state(self.sensory_adapter.feature_proj.state_dict(),full['current_numeric_state'],'current_numeric')
        _validate_state(self.geometry_state(),payload['geometry_state'],'pair_geometry')
        with torch.no_grad():
            for name,value in body['core_parameters'].items():dict(self.core.named_parameters())[name].copy_(value.to(self.device))
            for name,module in (('actor',self.actor),('critic',self.critic),('risk_head',self.risk_head)):
                module.load_state_dict(body[name],strict=True)
            self.base_action_attention.base_object_attention.load_state_dict(old['object_attention'],strict=True)
            query=self.base_action_attention.state_dict()
            for name,value in q['query_state'].items():query[name].copy_(value.to(query[name].device))
            self.sensory_adapter.map_encoder.load_state_dict(full['current_CNN_state'],strict=True)
            self.sensory_adapter.feature_proj.load_state_dict(full['current_numeric_state'],strict=True)
            geometry=self.geometry_attention.state_dict()
            for name,value in payload['geometry_state'].items():geometry[name].copy_(value.to(geometry[name].device))
        self.sensory_adapter.geometry_initialization=initialization
        return self

    def optimizer_groups(self,*,sensory_lr=3e-5,query_lr=3e-5,cnn_numeric_lr=3e-5,geometry_lr=3e-5,**body_rates):
        if any(not math.isfinite(lr) or lr<=0 for lr in (sensory_lr,query_lr,cnn_numeric_lr,geometry_lr)):
            raise ValueError('Positive explicit sensory rates required')
        groups=PolicyCostRoutingActorCritic.optimizer_groups(self,**body_rates)
        action=self.base_action_attention
        groups.extend((dict(name='sensory_object',params=list(action.base_object_attention.parameters()),lr=sensory_lr),
            dict(name=QUERY_GROUP,params=[p for name,p in action.named_parameters() if not name.startswith('base_object_attention.')],lr=query_lr),
            dict(name=FULL_SENSORY_GROUP,params=self.sensory_adapter.CNN_numeric_parameters(),lr=cnn_numeric_lr),
            dict(name=GEOMETRY_GROUP,params=self.geometry_parameters(),lr=geometry_lr)))
        return groups


def extend_loaded_full_sensory_model(model,optimizer,*,geometry_lr=3e-5,geometry_seed=71967):
    if not math.isfinite(geometry_lr) or geometry_lr<=0 or not isinstance(optimizer,torch.optim.Adam):
        raise ValueError('Source named Adam and positive geometry rate required')
    if any(g.get('name')==GEOMETRY_GROUP for g in optimizer.param_groups):raise ValueError('Geometry group already exists;use typed resume')
    names={id(p):name for name,p in model.named_parameters()}
    groups=[([id(p) for p in g['params']],copy.deepcopy({k:v for k,v in g.items() if k!='params'})) for g in optimizer.param_groups]
    states={id(p):value for p,value in optimizer.state.items()}
    full=FullPairGeometryActorCritic(model,geometry_seed=geometry_seed);current=dict(full.named_parameters())
    for g in optimizer.param_groups:
        for p in g['params']:
            if current.get(inherited_parameter_name(names[id(p)])) is not p:raise RuntimeError('Inherited Parameter/name changed')
    optimizer.add_param_group(dict(name=GEOMETRY_GROUP,params=full.geometry_parameters(),lr=geometry_lr))
    for g,(ids,settings) in zip(optimizer.param_groups[:-1],groups):
        if [id(p) for p in g['params']]!=ids or {k:v for k,v in g.items() if k!='params'}!=settings:
            raise RuntimeError('Inherited Adam group/settings changed')
    if any(optimizer.state[p] is not states[id(p)] for p in optimizer.state if id(p) in states):raise RuntimeError('Inherited moment changed')
    if any(p in optimizer.state for p in full.geometry_parameters()):raise RuntimeError('New geometry moments must be empty')
    return full,optimizer,dict(old_parameter_objects_and_Adam_preserved=True,all_old_sensory_remains_trainable=True,
        original_CNN_numeric_reinitialized=False,new_geometry_tensors=len(full.geometry_parameters()),
        new_geometry_parameters=sum(p.numel() for p in full.geometry_parameters()),new_group=GEOMETRY_GROUP,
        geometry_lr=geometry_lr,new_geometry_Adam_states=0,attention_name_translation=ATTENTION_PREFIX+'base_action_query.')


def migrate_full_sensory_source(source,connectome,*,geometry_lr=3e-5,device='cuda:0',restore_rng=True):
    from .continuous_full_sensory_model import restore_full_sensory_training_source
    from .continuous_object_learning_model import _restore_rng
    if source.get('schema')!=FULL_TRAINING_SCHEMA:raise ValueError('Complete prior FullSensory source required')
    model,optimizer,counters=restore_full_sensory_training_source(source,connectome,device=device,restore_rng=False)
    model,optimizer,migration=extend_loaded_full_sensory_model(model,optimizer,geometry_lr=geometry_lr)
    if restore_rng:_restore_rng(source['RNG'])
    migration.update(source_RNG_restored=bool(restore_rng),native_midgame_resume=False)
    return model,optimizer,counters,migration


def build_full_pair_model(payload,connectome,*,device='cuda:0'):
    from .continuous_full_sensory_model import build_full_sensory_model
    if payload.get('schema')!=PARAMETER_SCHEMA:raise ValueError('Independent FullPair parameter schema required')
    base=build_full_sensory_model(payload['full_sensory_model_payload'],connectome,device=device)
    return FullPairGeometryActorCritic(base,geometry_seed=payload['geometry_configuration']['seed']).load_parameter_payload(payload)


def full_pair_training_checkpoint_payload(model,optimizer,*,algorithm,counters,provenance=None):
    from .continuous_connectome_training import training_checkpoint_payload
    if not isinstance(model,FullPairGeometryActorCritic):raise TypeError('FullPair typed model required')
    result=training_checkpoint_payload(model,optimizer,algorithm=algorithm,counters=counters,provenance=provenance)
    result['schema']=TRAINING_SCHEMA
    result['sensory_replay']=dict(raw_inputs=['features25','maps16x92x96','objects96x29'],canonical_batch=FRONTEND_BATCH,
        detached281_evidence_only=True,all_CNN_numeric_object_query_trainable=True,geometry_schema=GEOMETRY_SCHEMA,
        geometry_group=GEOMETRY_GROUP,uses_teacher_risk_scores_or_selected_actions=False)
    return result


def restore_full_pair_training_checkpoint(model,optimizer,source,*,restore_rng=True):
    from .continuous_connectome_training import restore_training_checkpoint,SCHEMA as BASE_SCHEMA
    if source.get('schema')!=TRAINING_SCHEMA:raise ValueError('Independent FullPair training schema required')
    return restore_training_checkpoint(model,optimizer,dict(source,schema=BASE_SCHEMA),restore_rng=restore_rng)


def restore_full_pair_training_source(source,connectome,*,device='cuda:0',restore_rng=True):
    if source.get('schema')!=TRAINING_SCHEMA:raise ValueError('FullPair training source required;old Full is migration only')
    model=build_full_pair_model(source['model_payload'],connectome,device=device)
    names=dict(model.named_parameters());signature=source['optimizer_parameter_signature'];saved=source['optimizer']['param_groups']
    if len(signature)!=len(saved):raise ValueError('Complete named Adam signatures required')
    groups=[];seen=set()
    for specs,g in zip(signature,saved):
        parameters=[]
        for spec in specs:
            p=names[spec['name']]
            if spec['name'] in seen or list(p.shape)!=spec['shape'] or str(p.dtype)!=spec['dtype'] or not p.requires_grad:
                raise ValueError('Unique exact trainable Adam parameter required')
            seen.add(spec['name']);parameters.append(p)
        groups.append(dict({k:copy.deepcopy(v) for k,v in g.items() if k!='params'},params=parameters))
    if seen!={name for name,p in names.items() if p.requires_grad} or sum(g.get('name')==GEOMETRY_GROUP for g in groups)!=1:
        raise ValueError('Every trainable parameter and one geometry group must resume')
    optimizer=torch.optim.Adam(groups)
    counters=restore_full_pair_training_checkpoint(model,optimizer,source,restore_rng=restore_rng)
    return model,optimizer,counters
