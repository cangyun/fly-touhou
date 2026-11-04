"""Independent full-sensory continuous-connectome family; no job on import.

CNN/spatial attention, numeric projection, objects and action queries all
learn from raw observations. The complete graph/DN policy is unchanged.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import torch
from torch import nn

from .continuous_object_learning_model import ObjectLearningActorCritic, _cpu, _same, _validate_state, _DYNAMIC
from .continuous_action_query_learning_model import (
    ActionQueryLearningActorCritic, PARAMETER_SCHEMA as QUERY_PARAMETER_SCHEMA,
    TRAINING_SCHEMA as QUERY_TRAINING_SCHEMA, QUERY_GROUP,
)
from .continuous_policy_cost_routing import PolicyCostRoutingActorCritic

PARAMETER_SCHEMA = 'continuous-full-sensory-action-query-DN-deep128x3-threevalue-parameters-v1'
TRAINING_SCHEMA = 'continuous-full-sensory-action-query-DN-deep128x3-named-Adam-training-v1'
SENSORY_SCHEMA = 'V5-full-trainable-CNN-numeric-object-query384-canonical30-v1'
EVIDENCE_SCHEMA = 'V5-current-numeric-map281-detached-evidence-not-learning-cache-v1'
FULL_SENSORY_GROUP = 'sensory_CNN_numeric'
FRONTEND_BATCH = 30


class FullTrainableVisual(nn.Module):
    output_dim = 384
    static_dim = 281  # shape-compatible evidence only; never an input cache
    canonical_batch_size = FRONTEND_BATCH

    def __init__(self, source):
        super().__init__()
        from .continuous_object_sensory import ObjectTrainableVisual
        from .action_query_residual_objects import ActionQueryResidualObjects
        if not isinstance(source, ObjectTrainableVisual) or not isinstance(source.object_attention, ActionQueryResidualObjects):
            raise TypeError('Loaded query-family object frontend required')
        self.map_encoder, self.feature_proj = source.map_encoder, source.feature_proj
        self.object_attention, self.relu = source.object_attention, source.relu
        self.register_buffer('fixed_mean', source.fixed_mean)
        self.register_buffer('fixed_std', source.fixed_std)
        self.perception_configuration = copy.deepcopy(source.perception_configuration)
        self.source_provenance = copy.deepcopy(source.source_provenance)
        self.statistics_provenance = copy.deepcopy(source.statistics_provenance)
        self.source_static_cache_identity = source.static_cache_identity
        identity = dict(schema=EVIDENCE_SCHEMA, source_static_cache_identity=source.static_cache_identity,
            order=['raw_features25','current_numeric128','current_map128'], learning_cache=False)
        self.evidence_identity = hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        # Kept solely for legacy rollout evidence alignment,not frozen learning.
        self.static_cache_identity = self.evidence_identity
        self.requires_grad_(True)
        self.train(source.training)

    @property
    def device(self):
        return self.feature_proj.weight.device

    def train(self, mode=True):
        # The actual source has Conv/Linear/LN/MHA(dropout0),no BN/GroupNorm/
        # Dropout. Unlike the old object frontend this never re-freezes modules.
        return nn.Module.train(self,mode)

    def object_parameters(self):
        return self.object_attention.parameters()

    def CNN_numeric_parameters(self):
        return [*self.map_encoder.parameters(),*self.feature_proj.parameters()]

    def perception_parameter_groups(self):
        """Disjoint diagnostic branches; no forward or state mutation."""
        return dict(CNN=tuple(p for name,p in self.map_encoder.named_parameters() if not name.startswith('spatial_attention.')),
            spatial_attention=tuple(p for name,p in self.map_encoder.named_parameters() if name.startswith('spatial_attention.')),
            numeric=tuple(self.feature_proj.parameters()),
            object=tuple(self.object_attention.base_object_attention.parameters()),
            query=tuple(p for name,p in self.object_attention.named_parameters() if not name.startswith('base_object_attention.')))

    @staticmethod
    def frontend_cost(real_rows):
        batches = (real_rows + FRONTEND_BATCH - 1) // FRONTEND_BATCH
        return dict(real_rows=real_rows,canonical_batches=batches,
            forwarded_rows=batches*FRONTEND_BATCH,padding_rows=batches*FRONTEND_BATCH-real_rows)

    def _encode(self, features, maps, objects, *, evidence, geometry_attention=False):
        leading = features.shape[:-1]
        if (features.ndim < 2 or features.shape[-1]!=25 or min(leading)<1
                or maps.shape!=(*leading,16,92,96) or objects.shape!=(*leading,96,29)
                or any(value.dtype!=torch.float32 or value.device!=self.device for value in (features,maps,objects))):
            raise ValueError('Aligned raw FP32 V5 f25/maps16x92x96/objects96x29 on frontend device required')
        f=features.reshape(-1,25);m=maps.reshape(-1,16,92,96);o=objects.reshape(-1,96,29)
        encoded_rows, evidence_rows, attention_rows = [], [], []
        with torch.autocast(device_type=features.device.type,enabled=False):
            for first in range(0,len(f),FRONTEND_BATCH):
                n=min(FRONTEND_BATCH,len(f)-first)
                def padded(value):
                    real=value[first:first+n]
                    return torch.cat((real,real.new_zeros((FRONTEND_BATCH-n,*real.shape[1:]))),0) if n<FRONTEND_BATCH else real
                part_f,part_m,part_o = padded(f),padded(m),padded(o)
                numeric=self.relu(self.feature_proj(part_f))
                spatial=self.map_encoder(part_m)
                if geometry_attention:
                    context,details=self.object_attention.forward_with_geometry_details(part_f,part_o)
                    attention_rows.append(details['pooling_logits'][:n])
                else:context=self.object_attention(part_f,part_o)
                raw=torch.cat((numeric,context,spatial),-1)
                encoded_rows.append(((raw-self.fixed_mean)/self.fixed_std)[:n])
                if evidence:
                    evidence_rows.append(torch.cat((part_f,numeric,spatial),-1)[:n].detach())
        encoded=torch.cat(encoded_rows,0).reshape(*leading,384)
        if geometry_attention:
            return encoded,torch.cat(attention_rows,0).reshape(*leading,18,96)
        if evidence:
            return encoded,torch.cat(evidence_rows,0).reshape(*leading,281).detach()
        return encoded

    def encode_raw(self, features, maps, objects):
        return self._encode(features,maps,objects,evidence=False)

    def encode_raw_with_geometry_attention(self,features,maps,objects):
        """Same sensory forward, with differentiable training-only pool logits."""
        return self._encode(features,maps,objects,evidence=False,geometry_attention=True)

    def forward_with_evidence(self, features, maps, objects):
        return self._encode(features,maps,objects,evidence=True)

    def forward(self, features, maps, objects):
        return self.encode_raw(features,maps,objects)

    def encode_static(self, *args, **kwargs):
        raise ValueError('Full-sensory learning requires raw maps;static281 is detached evidence only')

    def encode_objects(self, *args, **kwargs):
        raise ValueError('Do not feed static281 or cached384 to full-sensory learning;use encode_raw')

    def specification(self):
        return dict(schema=SENSORY_SCHEMA,output_dim=384,columns=['numeric128','object_context128','map128'],
            trainable=['map CNN','spatial attention','numeric projection','object attention','action queries'],
            frozen_parameters=[],source_D95_weights='initialization/provenance only;current CNN/numeric saved separately',
            normalization='same fixed mean/std as reversible initial feature coordinates;upstream CNN is trainable',
            fixed_statistics_do_not_imply_frozen_CNN=True,source_checkpoint_sha256=self.source_provenance.get('source_checkpoint_sha256'),
            canonical_frontend_batch=FRONTEND_BATCH,tail='zero-pad to30 then discard padding outputs',
            raw_replay_inputs=['features25','maps16x92x96','objects96x29'],
            evidence_schema=EVIDENCE_SCHEMA,evidence_order=['features25','current_numeric128','current_map128'],
            static281_or_cached384_valid_for_learning=False,evidence_identity=self.evidence_identity,
            BatchNorm_modules=sum(isinstance(m,nn.modules.batchnorm._BatchNorm) for m in self.modules()),
            current_learner_integrated=True,complete_brain_and_DN_policy_required=True)


class FullSensoryActorCritic(ObjectLearningActorCritic):
    source_schema = PARAMETER_SCHEMA

    def __init__(self, base):
        nn.Module.__init__(self)
        if not isinstance(base,ActionQueryLearningActorCritic):
            raise TypeError('Loaded old Query family model required')
        cfg=base.core.config()
        if (cfg['n_neurons']!=138639 or cfg['n_edges']!=15091983 or cfg['input_ports']!=8053
                or cfg['output_ports']!=1299 or cfg['K']!=8 or not cfg['trainable_brain']
                or base.risk_head.beta!=8. or base.policy_cost_gradient!='supervised-only'):
            raise ValueError('Retain same complete K8 graph/ports/beta8/threevalue/routing')
        # Capture honest old-family metadata BEFORE replacing the frontend.
        # Its frozen CNN snapshot is initialization context,not the effective
        # CNN after full-sensory updates. Current CNN/numeric live outside it.
        self._source_query_configuration=copy.deepcopy(base.configuration())
        self._source_object_configuration=copy.deepcopy(ObjectLearningActorCritic.configuration(base))
        self._source_static_cache_identity=base.sensory_adapter.static_cache_identity
        self._query_configuration=copy.deepcopy(base._query_configuration)
        self._base_metadata=base._base_metadata
        self.core,self.actor,self.critic,self.risk_head=base.core,base.actor,base.critic,base.risk_head
        self.initialization_seed=base.initialization_seed
        self.policy_cost_gradient=base.policy_cost_gradient
        self.sensory_adapter=FullTrainableVisual(base.sensory_adapter)
        self.train(base.training)

    def train(self, mode=True):
        nn.Module.train(self,mode)
        self.sensory_adapter.train(mode)
        return self

    def _raw_tensors(self, features, maps, objects):
        return tuple(torch.as_tensor(value,dtype=torch.float32,device=self.device) for value in (features,maps,objects))

    def encode_raw(self, features, maps, *, objects):
        return self.sensory_adapter.encode_raw(*self._raw_tensors(features,maps,objects))

    def encode_raw_with_geometry_attention(self,features,maps,*,objects):
        return self.sensory_adapter.encode_raw_with_geometry_attention(*self._raw_tensors(features,maps,objects))

    def forward_with_evidence(self, features, maps, *, objects):
        return self.sensory_adapter.forward_with_evidence(*self._raw_tensors(features,maps,objects))

    def encode_visual(self, features, maps, *, objects):
        return self.encode_raw(features,maps,objects=objects)

    def encode_visual_with_static(self, features, maps, *, objects):
        """Legacy name returns evidence,not a reusable trainable-sensory cache."""
        return self.forward_with_evidence(features,maps,objects=objects)

    def encode_cached(self, *args, **kwargs):
        raise ValueError('Full-sensory update must recompute raw maps/f25/o29;281/384 are evidence only')

    def encode_static(self, *args, **kwargs):
        raise ValueError('Full-sensory static281 is not a learning cache;use forward_with_evidence')

    def configuration(self):
        return dict(schema=PARAMETER_SCHEMA,core=self.core.config(),risk_topology=copy.deepcopy(self._source_query_configuration['risk_topology']),
            risk_beta=self.risk_head.beta,policy_cost_gradient=self.policy_cost_gradient,
            perception=self.sensory_adapter.specification(),initialization_seed=self.initialization_seed,
            actor_critic_risk_inputs='following full-connectome DN only',
            all_three_value_heads_retained=True,original_parameter_objects_reused=True,
            source_query_parameter_schema=QUERY_PARAMETER_SCHEMA,current_sensory_state_separate_from_source_snapshot=True,
            action_query_configuration=copy.deepcopy(self._query_configuration),PairGeometry_integrated=False)

    def _current_query_state(self):
        return {name:value for name,value in self.sensory_adapter.object_attention.state_dict().items()
            if not name.startswith('base_object_attention.')}

    def _nested_query_payload(self):
        body=_cpu(self._base_metadata)
        body.update(core_parameters=_cpu(dict(self.core.named_parameters())),actor=_cpu(self.actor.state_dict()),
            critic=_cpu(self.critic.state_dict()),risk_head=_cpu(self.risk_head.state_dict()))
        old=dict(schema=self._source_object_configuration['schema'],configuration=copy.deepcopy(self._source_object_configuration),
            base_parameters=body,object_attention=_cpu(self.sensory_adapter.object_attention.base_object_attention.state_dict()),
            static_schema='D95-frozen-numeric-map-static281-v1',static_cache_identity=self._source_static_cache_identity,
            contains_optimizer=False,contains_neural_state=False,contains_teacher_hidden=False)
        return dict(schema=QUERY_PARAMETER_SCHEMA,configuration=copy.deepcopy(self._source_query_configuration),
            object_model_payload=old,query_configuration=copy.deepcopy(self._query_configuration),
            query_state=_cpu(self._current_query_state()),static_schema=old['static_schema'],
            static_cache_identity=self._source_static_cache_identity,
            contains_optimizer=False,contains_neural_state=False,contains_teacher_hidden=False)

    def parameter_checkpoint_payload(self):
        return dict(schema=PARAMETER_SCHEMA,configuration=self.configuration(),query_model_payload=self._nested_query_payload(),
            current_CNN_state=_cpu(self.sensory_adapter.map_encoder.state_dict()),
            current_numeric_state=_cpu(self.sensory_adapter.feature_proj.state_dict()),
            current_sensory_state_authoritative=True,
            nested_frozen_CNN_snapshot='initialization reference only;not the effective full-sensory frontend',
            contains_optimizer=False,contains_neural_state=False,contains_teacher_hidden=False)

    def load_parameter_payload(self, payload):
        if (payload.get('schema')!=PARAMETER_SCHEMA or payload.get('current_sensory_state_authoritative') is not True
                or not _same(payload.get('configuration'),self.configuration())):
            raise ValueError('Exact new full-sensory model/config required;old frozen loaders are not compatible')
        q=payload['query_model_payload'];old=q['object_model_payload'];body=old['base_parameters']
        if (q.get('schema')!=QUERY_PARAMETER_SCHEMA or not _same(q.get('configuration'),self._source_query_configuration)
                or not _same(old.get('configuration'),self._source_object_configuration)
                or q.get('static_cache_identity')!=self._source_static_cache_identity
                or not _same(q.get('query_configuration'),self._query_configuration)
                or not _same({k:v for k,v in body.items() if k not in _DYNAMIC},self._base_metadata)):
            raise ValueError('Original graph/source frozen snapshot/query initialization context differs')
        _validate_state(dict(self.core.named_parameters()),body['core_parameters'],'core')
        for name,module in (('actor',self.actor),('critic',self.critic),('risk_head',self.risk_head)):
            _validate_state(module.state_dict(),body[name],name)
        for key in ('layers.DN_mean','layers.DN_scale'):
            if not _same(self.risk_head.state_dict()[key],body['risk_head'][key]):
                raise ValueError('Existing fixed risk normalization differs')
        attention=self.sensory_adapter.object_attention
        _validate_state(attention.base_object_attention.state_dict(),old['object_attention'],'base_attention')
        _validate_state(self._current_query_state(),q['query_state'],'query')
        if not _same(self._current_query_state()['action_descriptors'],q['query_state']['action_descriptors']):
            raise ValueError('Immutable source18 command descriptors differ')
        _validate_state(self.sensory_adapter.map_encoder.state_dict(),payload['current_CNN_state'],'current_CNN')
        _validate_state(self.sensory_adapter.feature_proj.state_dict(),payload['current_numeric_state'],'current_numeric')
        with torch.no_grad():
            for name,value in body['core_parameters'].items():dict(self.core.named_parameters())[name].copy_(value.to(self.device))
            for name,module in (('actor',self.actor),('critic',self.critic),('risk_head',self.risk_head)):
                module.load_state_dict(body[name],strict=True)
            attention.base_object_attention.load_state_dict(old['object_attention'],strict=True)
            state=attention.state_dict()
            for name,value in q['query_state'].items():state[name].copy_(value.to(state[name].device))
            self.sensory_adapter.map_encoder.load_state_dict(payload['current_CNN_state'],strict=True)
            self.sensory_adapter.feature_proj.load_state_dict(payload['current_numeric_state'],strict=True)
        return self

    def optimizer_groups(self, *, sensory_lr=3e-5, query_lr=3e-5, cnn_numeric_lr=3e-5, **body_rates):
        if any(not math.isfinite(lr) or lr<=0 for lr in (sensory_lr,query_lr,cnn_numeric_lr)):
            raise ValueError('Positive explicit sensory learning rates required')
        groups=PolicyCostRoutingActorCritic.optimizer_groups(self,**body_rates)
        attention=self.sensory_adapter.object_attention
        groups.extend((dict(name='sensory_object',params=list(attention.base_object_attention.parameters()),lr=sensory_lr),
            dict(name=QUERY_GROUP,params=[p for name,p in attention.named_parameters() if not name.startswith('base_object_attention.')],lr=query_lr),
            dict(name=FULL_SENSORY_GROUP,params=self.sensory_adapter.CNN_numeric_parameters(),lr=cnn_numeric_lr)))
        return groups


def extend_loaded_query_model(model, optimizer, *, sensory_lr=3e-5):
    if not math.isfinite(sensory_lr) or sensory_lr<=0 or not isinstance(optimizer,torch.optim.Adam):
        raise ValueError('Named source Adam and positive CNN/numeric LR required')
    if any(g.get('name')==FULL_SENSORY_GROUP for g in optimizer.param_groups):
        raise ValueError('Source already has full-sensory Adam;use exact resume')
    names={id(p):name for name,p in model.named_parameters()}
    old_groups=[([id(p) for p in g['params']],copy.deepcopy({k:v for k,v in g.items() if k!='params'})) for g in optimizer.param_groups]
    old_states={id(p):state for p,state in optimizer.state.items()}
    new_parameters=[*model.sensory_adapter.map_encoder.parameters(),*model.sensory_adapter.feature_proj.parameters()]
    if any(p.requires_grad or id(p) in old_states for p in new_parameters):
        raise ValueError('CNN/numeric must be the original frozen32 tensors without inherited optimizer state')
    full=FullSensoryActorCritic(model);current=dict(full.named_parameters())
    if any(current.get(name) is not p for name,p in model.named_parameters()):
        raise RuntimeError('All original Parameter names/objects must be retained')
    optimizer.add_param_group(dict(name=FULL_SENSORY_GROUP,params=new_parameters,lr=sensory_lr))
    for group,(ids,settings) in zip(optimizer.param_groups[:-1],old_groups):
        if [id(p) for p in group['params']]!=ids or {k:v for k,v in group.items() if k!='params'}!=settings:
            raise RuntimeError('An inherited named Adam group changed')
    if any(optimizer.state[p] is not old_states[id(p)] for p in optimizer.state if id(p) in old_states):
        raise RuntimeError('An inherited Adam state object changed')
    if any(p in optimizer.state for p in new_parameters):raise RuntimeError('CNN/numeric Adam must start empty')
    return full,optimizer,dict(old_parameter_names_and_objects_preserved=True,old_named_Adam_and_group_settings_preserved=True,
        newly_trainable_parameters_initialized=False,new_CNN_numeric_tensors=len(new_parameters),
        new_CNN_numeric_parameters=sum(p.numel() for p in new_parameters),new_Adam_states=0,
        new_group=FULL_SENSORY_GROUP,new_group_learning_rate=sensory_lr,
        previously_frozen_CNN_numeric_names=[names[id(p)] for p in new_parameters],
        canonical_frontend_batch=FRONTEND_BATCH,PairGeometry_integrated=False)


def migrate_query_training_source(source, connectome, *, sensory_lr=3e-5, device='cuda:0', restore_rng=True):
    from .continuous_action_query_learning_model import restore_action_query_training_source
    from .continuous_object_learning_model import _restore_rng
    if source.get('schema')!=QUERY_TRAINING_SCHEMA:raise ValueError('Complete Query-family source/Adam required')
    model,optimizer,counters=restore_action_query_training_source(source,connectome,device=device,restore_rng=False)
    model,optimizer,migration=extend_loaded_query_model(model,optimizer,sensory_lr=sensory_lr)
    if restore_rng:_restore_rng(source['RNG'])
    migration.update(source_RNG_restored=bool(restore_rng),native_midgame_resume=False)
    return model,optimizer,counters,migration


def build_full_sensory_model(payload, connectome, *, device='cuda:0'):
    from .continuous_action_query_learning_model import build_action_query_model
    if payload.get('schema')!=PARAMETER_SCHEMA:raise ValueError('Explicit full-sensory parameter schema required')
    base=build_action_query_model(payload['query_model_payload'],connectome,device=device)
    return FullSensoryActorCritic(base).load_parameter_payload(payload)


def full_sensory_training_checkpoint_payload(model, optimizer, *, algorithm, counters, provenance=None):
    from .continuous_connectome_training import training_checkpoint_payload
    if not isinstance(model,FullSensoryActorCritic):raise TypeError('Full-sensory typed model required')
    result=training_checkpoint_payload(model,optimizer,algorithm=algorithm,counters=counters,provenance=provenance)
    result['schema']=TRAINING_SCHEMA
    result['sensory_replay']=dict(schema=SENSORY_SCHEMA,raw_inputs=['features25','maps16x92x96','objects96x29'],
        canonical_batch=FRONTEND_BATCH,detached281_evidence_only=True,uses_static281_or_stale384_for_gradient=False,
        trainable_CNN_numeric=True,group=FULL_SENSORY_GROUP)
    return result


def restore_full_sensory_training_checkpoint(model, optimizer, source, *, restore_rng=True):
    from .continuous_connectome_training import restore_training_checkpoint,SCHEMA as BASE_SCHEMA
    if source.get('schema')!=TRAINING_SCHEMA:raise ValueError('Complete full-sensory training schema required')
    return restore_training_checkpoint(model,optimizer,dict(source,schema=BASE_SCHEMA),restore_rng=restore_rng)


def restore_full_sensory_training_source(source, connectome, *, device='cuda:0', restore_rng=True):
    if source.get('schema')!=TRAINING_SCHEMA:raise ValueError('New full-sensory source required;not a frozen Query checkpoint')
    model=build_full_sensory_model(source['model_payload'],connectome,device=device)
    names=dict(model.named_parameters());signatures=source['optimizer_parameter_signature'];saved=source['optimizer']['param_groups']
    if len(signatures)!=len(saved):raise ValueError('Complete named Adam groups/signatures required')
    groups=[];seen=set()
    for specs,group in zip(signatures,saved):
        parameters=[]
        for spec in specs:
            p=names[spec['name']]
            if spec['name'] in seen or list(p.shape)!=spec['shape'] or str(p.dtype)!=spec['dtype'] or not p.requires_grad:
                raise ValueError('Unique exact trainable named Adam tensor required')
            seen.add(spec['name']);parameters.append(p)
        groups.append(dict({k:copy.deepcopy(v) for k,v in group.items() if k!='params'},params=parameters))
    if (seen!={name for name,p in names.items() if p.requires_grad}
            or sum(group.get('name')==FULL_SENSORY_GROUP for group in groups)!=1):
        raise ValueError('Every full-sensory parameter and one explicit CNN/numeric group required')
    optimizer=torch.optim.Adam(groups)
    counters=restore_full_sensory_training_checkpoint(model,optimizer,source,restore_rng=restore_rng)
    return model,optimizer,counters
