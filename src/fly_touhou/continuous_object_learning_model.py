"""New typed deep-risk continuous model with an object-trainable frontend.

The original full graph/DN actor/three critics/deep128x3 risk are retained.
Only the original object-attention frontend is newly trainable; frozen CNN,
numeric projection and fixed statistics keep their exact source identity.
No constructor, graph, optimizer or job runs on import.
"""
from __future__ import annotations
import copy
import math

import torch
from torch import nn

from .continuous_policy_cost_routing import PolicyCostRoutingActorCritic, MODES
from .continuous_deep_risk_readout import (
    DeepDNCostHead, PARAMETER_SCHEMA as DEEP_PARAMETER_SCHEMA, TOPOLOGY,
)
from .continuous_object_sensory import ObjectTrainableVisual, StaticVisualBatch, STATIC_SCHEMA

PARAMETER_SCHEMA = 'continuous-object-attention-DN-deep128x3-threevalue-parameters-v1'
TRAINING_SCHEMA = 'continuous-object-attention-DN-deep128x3-named-Adam-training-v1'
_DYNAMIC = ('core_parameters', 'actor', 'critic', 'risk_head')


def _cpu(value):
    from .continuous_connectome_training import _cpu_tree
    return _cpu_tree(value)


def _same(left, right):
    if torch.is_tensor(left) or torch.is_tensor(right):
        return (torch.is_tensor(left) and torch.is_tensor(right) and left.dtype == right.dtype
                and left.shape == right.shape and torch.equal(left.detach().cpu(), right.detach().cpu()))
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(_same(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(_same(a,b) for a,b in zip(left,right))
    return left == right


def _validate_state(current, stored, scope):
    if set(current) != set(stored):
        raise ValueError('Exact named tensor set required: ' + scope)
    for name, target in current.items():
        value = stored[name]
        if (not torch.is_tensor(value) or value.dtype != target.dtype or value.shape != target.shape
                or not bool(torch.isfinite(value).all())):
            raise ValueError('Finite exact tensor shape/dtype required: ' + scope + '.' + name)


class ObjectLearningActorCritic(PolicyCostRoutingActorCritic):
    """Explicit migration of a loaded deep model; no body/head initialization."""
    source_schema = PARAMETER_SCHEMA

    def __init__(self, base, base_parameters):
        nn.Module.__init__(self)
        if (not isinstance(base, PolicyCostRoutingActorCritic)
                or not isinstance(base.risk_head, DeepDNCostHead)
                or base_parameters.get('schema') != DEEP_PARAMETER_SCHEMA
                or base_parameters['configuration'].get('risk_topology') != TOPOLOGY
                or base.policy_cost_gradient not in MODES):
            raise ValueError('Loaded deep128x3 routed source and explicit typed parameters required')
        # Reuse the exact Parameter objects, so inherited Adam remains attached.
        self.core, self.actor, self.critic, self.risk_head = base.core, base.actor, base.critic, base.risk_head
        self.initialization_seed = base.initialization_seed
        self.policy_cost_gradient = base.policy_cost_gradient
        self.sensory_adapter = ObjectTrainableVisual(base.sensory_adapter).to(self.device)
        self._base_metadata = _cpu({k:v for k,v in base_parameters.items() if k not in _DYNAMIC})
        self.core.set_trainable_brain(True)
        for module in (self.core.input_encoder, self.actor, self.critic, self.risk_head):
            module.requires_grad_(True)
        self.train(base.training)

    def train(self, mode=True):
        # Do not call the old parent override that freezes the entire frontend.
        nn.Module.train(self, mode)
        self.sensory_adapter.train(mode)  # only map/numeric stay frozen
        return self

    def encode_static(self, features, maps):
        f,m=(torch.as_tensor(x,dtype=torch.float32,device=self.device) for x in (features,maps))
        return self.sensory_adapter.encode_static(f,m)

    def encode_cached(self, static281, objects):
        static281=torch.as_tensor(static281,dtype=torch.float32,device=self.device)
        objects=torch.as_tensor(objects,dtype=torch.float32,device=self.device)
        return self.sensory_adapter.encode_objects(StaticVisualBatch.from_packed(static281),objects)

    def encode_visual_with_static(self, features, maps, *, objects):
        static=self.encode_static(features,maps)
        objects=torch.as_tensor(objects,dtype=torch.float32,device=self.device)
        return self.sensory_adapter.encode_objects(static,objects),static.packed()

    def encode_visual(self, features, maps, *, objects):
        return self.encode_visual_with_static(features,maps,objects=objects)[0]

    def optimizer_groups(self, *, sensory_lr, **body_rates):
        if not math.isfinite(sensory_lr) or sensory_lr <= 0:
            raise ValueError('Sensory learning rate must be explicitly positive and finite')
        groups=super().optimizer_groups(**body_rates)
        return groups+[dict(name='sensory_object',params=list(self.sensory_adapter.object_parameters()),lr=sensory_lr)]

    def configuration(self):
        return dict(schema=PARAMETER_SCHEMA,core=self.core.config(),risk_topology=copy.deepcopy(TOPOLOGY),
            risk_beta=self.risk_head.beta,policy_cost_gradient=self.policy_cost_gradient,
            perception=self.sensory_adapter.specification(),
            actor_critic_risk_inputs='following full-connectome DN only',
            frozen=['CNN/spatial attention','numeric projection','fixed statistics'],
            source_body_and_heads_reinitialized=False,initialization_seed=self.initialization_seed)

    def parameter_checkpoint_payload(self):
        base=_cpu(self._base_metadata)
        base.update(core_parameters=_cpu(dict(self.core.named_parameters())),actor=_cpu(self.actor.state_dict()),
            critic=_cpu(self.critic.state_dict()),risk_head=_cpu(self.risk_head.state_dict()))
        # Nested deep payload explicitly retains immutable SOURCE perception.
        # Current adaptable object weights are separate, under this NEW schema.
        return dict(schema=PARAMETER_SCHEMA,configuration=self.configuration(),base_parameters=base,
            object_attention=_cpu(self.sensory_adapter.object_attention.state_dict()),
            static_schema=STATIC_SCHEMA,static_cache_identity=self.sensory_adapter.static_cache_identity,
            contains_optimizer=False,contains_neural_state=False,contains_teacher_hidden=False)

    def load_parameter_payload(self, payload):
        if (payload.get('schema') != PARAMETER_SCHEMA or payload.get('static_schema') != STATIC_SCHEMA
                or payload['static_cache_identity'] != self.sensory_adapter.static_cache_identity
                or payload['configuration']['policy_cost_gradient'] != self.policy_cost_gradient
                or payload['configuration']['risk_topology'] != TOPOLOGY):
            raise ValueError('Explicit object-learning/deep-risk/static-cache subtype required')
        base=payload['base_parameters']
        if not _same({k:v for k,v in base.items() if k not in _DYNAMIC},self._base_metadata):
            raise ValueError('Immutable graph/source perception/statistics/deep risk context differs')
        _validate_state(dict(self.core.named_parameters()),base['core_parameters'],'core')
        for name,module in (('actor',self.actor),('critic',self.critic),('risk_head',self.risk_head)):
            _validate_state(module.state_dict(),base[name],name)
        # Fixed train-only DN risk statistics must not silently become trainable.
        for key in ('layers.DN_mean','layers.DN_scale'):
            if not _same(self.risk_head.state_dict()[key],base['risk_head'][key]):
                raise ValueError('Fixed deep risk normalization differs')
        _validate_state(self.sensory_adapter.object_attention.state_dict(),payload['object_attention'],'object_attention')
        with torch.no_grad():
            for name,value in base['core_parameters'].items():dict(self.core.named_parameters())[name].copy_(value.to(self.device))
            for name,module in (('actor',self.actor),('critic',self.critic),('risk_head',self.risk_head)):
                module.load_state_dict(base[name],strict=True)
            self.sensory_adapter.object_attention.load_state_dict(payload['object_attention'],strict=True)
        return self


def build_object_model(payload, connectome, *, device='cuda:0'):
    """Explicit complete-graph builder; never invoked by import or CPU checks."""
    from .continuous_connectome import ContinuousConnectomeCore
    from .continuous_deep_risk_readout import load_deep_parameter_payload
    from .fly_frozen_v5_perception import FrozenV5Perception
    from .fly_visual_distillation import FixedStandardizedVisual
    if payload.get('schema') != PARAMETER_SCHEMA:
        raise ValueError('Independent object-learning parameter schema required')
    base_payload=payload['base_parameters'];c=base_payload['configuration'];cfg=c['core']
    if (cfg['K']!=8 or cfg['n_neurons']!=138639 or cfg['n_edges']!=15091983 or not cfg['trainable_brain']):
        raise ValueError('Same full trainable K8 connectome required')
    core=ContinuousConnectomeCore.from_full_connectome(connectome,K=8,trainable_brain=True,device='cpu',
        seed=cfg['seed'],epsilon=cfg['epsilon'],edge_chunk_size=cfg['sparse']['edge_chunk_size'])
    perception=FrozenV5Perception(base_payload['perception_configuration'])
    perception._source_provenance=copy.deepcopy(base_payload['perception_provenance'])
    adapter=FixedStandardizedVisual(perception,base_payload['fixed_mean'],base_payload['fixed_std'],base_payload['statistics_provenance'])
    base=PolicyCostRoutingActorCritic(core,adapter,seed=c['initialization_seed'],
        policy_cost_gradient=payload['configuration']['policy_cost_gradient']).to(device)
    load_deep_parameter_payload(base,base_payload)
    return ObjectLearningActorCritic(base,base_payload).load_parameter_payload(payload)


def migrate_deep_training_source(source, connectome, *, sensory_lr,
        policy_cost_gradient=None, source_policy_cost_gradient=None):
    """Keep body Adam/RNG; an explicit source->target routing change is recorded."""
    from .continuous_deep_policy_cost_routing import restore_deep_aware_routed_source
    if not math.isfinite(sensory_lr) or sensory_lr<=0:
        raise ValueError('No default sensory LR;explicit positive finite value required')
    if source.get('model_payload',{}).get('schema')!=DEEP_PARAMETER_SCHEMA:
        raise ValueError('Complete deep128x3 source required,not a parameter-only or old64 source')
    stored=source['algorithm'].get('policy_cost_gradient')
    if stored is not None and stored not in MODES:raise ValueError('Unknown stored routing mode')
    if stored is not None and source_policy_cost_gradient not in (None,stored):
        raise ValueError('Declared source routing conflicts with stored routing')
    source_mode=stored if stored is not None else source_policy_cost_gradient
    if source_mode not in MODES:
        raise ValueError('An unmarked source requires explicit source_policy_cost_gradient')
    mode=source_mode if policy_cost_gradient is None else policy_cost_gradient
    if mode not in MODES:raise ValueError('Explicit supported target routing required')
    base,optimizer,counters,_=restore_deep_aware_routed_source(source,connectome,policy_cost_gradient=mode)
    names_before={id(p):n for n,p in base.named_parameters()}
    groups_before=[([id(p) for p in g['params']],copy.deepcopy({k:v for k,v in g.items() if k!='params'}))
                   for g in optimizer.param_groups]
    states_before={id(p):state for p,state in optimizer.state.items()}
    model=ObjectLearningActorCritic(base,source['model_payload'])
    if any(dict(model.named_parameters()).get(names_before[id(p)]) is not p
           for g in optimizer.param_groups for p in g['params']):
        raise RuntimeError('Every inherited named body/head Parameter must be reused')
    optimizer.add_param_group(dict(name='sensory_object',params=list(model.sensory_adapter.object_parameters()),lr=sensory_lr))
    for g,(ids,settings) in zip(optimizer.param_groups[:-1],groups_before):
        if [id(p) for p in g['params']]!=ids or {k:v for k,v in g.items() if k!='params'}!=settings:
            raise RuntimeError('Inherited Adam group/settings changed')
    if any(optimizer.state[p] is not states_before[id(p)] for g in optimizer.param_groups[:-1]
           for p in g['params'] if id(p) in states_before):
        raise RuntimeError('Inherited named Adam state objects changed')
    # Constructor/deepcopy has no random initialization, but restore explicitly
    # at the end so the migration boundary retains every source global RNG.
    _restore_rng(source['RNG'])
    return model,optimizer,copy.deepcopy(counters),dict(
        source_body_Adam_and_group_settings_preserved=True,new_sensory_tensors=54,
        new_sensory_Adam_states=0,sensory_lr=sensory_lr,source_RNG_restored=True,
        source_policy_cost_gradient_preserved=mode==source_mode,
        routing_transition=dict(source=source_mode,target=mode,
            source_mode_recorded_in_checkpoint=stored is not None,
            source_mode_explicitly_declared=stored is None,
            mode_changed=source_mode!=mode,source_checkpoint_modified=False),
        native_midgame_resume=False)


def _restore_rng(state):
    import random
    import numpy as np
    random.setstate(state['python']);n=state['numpy']
    np.random.set_state((n['name'],np.asarray(n['keys'],np.uint32),n['pos'],n['has_gauss'],n['cached_gaussian']))
    torch.set_rng_state(state['torch_cpu'])
    if state['torch_cuda']:torch.cuda.set_rng_state_all(state['torch_cuda'])


def object_training_checkpoint_payload(model,optimizer,*,algorithm,counters,provenance=None):
    from .continuous_connectome_training import training_checkpoint_payload
    payload=training_checkpoint_payload(model,optimizer,algorithm=algorithm,counters=counters,provenance=provenance)
    payload['schema']=TRAINING_SCHEMA
    payload['sensory_replay']=dict(static_schema=STATIC_SCHEMA,
        static_cache_identity=model.sensory_adapter.static_cache_identity,
        uses_stale_encoded384_for_gradient=False,source_risk_topology=copy.deepcopy(TOPOLOGY))
    return payload


def restore_object_training_source(source,connectome,*,device='cuda:0',restore_rng=True):
    """Exact new-family resume of all named Adam groups,including sensory."""
    from .continuous_connectome_training import restore_training_checkpoint,SCHEMA as BASE_TRAINING_SCHEMA
    if source.get('schema')!=TRAINING_SCHEMA:
        raise ValueError('Full object-learning training checkpoint required')
    model=build_object_model(source['model_payload'],connectome,device=device)
    names=dict(model.named_parameters());groups=[];seen=set()
    signature=source['optimizer_parameter_signature'];saved=source['optimizer']['param_groups']
    if len(signature)!=len(saved):raise ValueError('Complete named Adam signature required')
    for specs,g in zip(signature,saved):
        parameters=[]
        for spec in specs:
            p=names[spec['name']]
            if (spec['name'] in seen or list(p.shape)!=spec['shape'] or str(p.dtype)!=spec['dtype'] or not p.requires_grad):
                raise ValueError('Exact unique trainable Adam tensor identity required')
            seen.add(spec['name']);parameters.append(p)
        groups.append(dict({k:copy.deepcopy(v) for k,v in g.items() if k!='params'},params=parameters))
    if seen!={n for n,p in names.items() if p.requires_grad} or sum(g.get('name')=='sensory_object' for g in groups)!=1:
        raise ValueError('Every trainable tensor and one explicit sensory group must be restored')
    optimizer=torch.optim.Adam(groups)
    counters=restore_training_checkpoint(model,optimizer,dict(source,schema=BASE_TRAINING_SCHEMA),restore_rng=restore_rng)
    return model,optimizer,counters
