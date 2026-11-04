"""Typed action-query sensory extension of the existing object-learning model.

Old complete-graph/body/head/attention Parameters and Adam states are reused.
Only ten new query tensors are initialized. No graph, optimizer, CUDA or native
job is constructed on import; full builders are explicitly caller-owned.
"""
from __future__ import annotations

import copy
import math
import torch
from torch import nn

from .action_query_residual_objects import ActionQueryResidualObjects, SCHEMA as QUERY_SCHEMA
from .continuous_object_learning_model import (
    ObjectLearningActorCritic, PARAMETER_SCHEMA as OBJECT_PARAMETER_SCHEMA,
    TRAINING_SCHEMA as OBJECT_TRAINING_SCHEMA, _cpu, _same, _validate_state, _DYNAMIC,
)
from .continuous_policy_cost_routing import PolicyCostRoutingActorCritic
from .continuous_object_sensory import STATIC_SCHEMA
from .continuous_deep_risk_readout import TOPOLOGY

PARAMETER_SCHEMA = 'continuous-action-query-object-DN-deep128x3-threevalue-parameters-v1'
TRAINING_SCHEMA = 'continuous-action-query-object-DN-deep128x3-named-Adam-training-v1'
QUERY_GROUP = 'sensory_action_query'
ATTENTION_PREFIX = 'sensory_adapter.object_attention.'


def inherited_parameter_name(name):
    """Only the original54 attention names acquire this explicit prefix."""
    if name.startswith(ATTENTION_PREFIX):
        return ATTENTION_PREFIX + 'base_object_attention.' + name[len(ATTENTION_PREFIX):]
    return name


class ActionQueryLearningActorCritic(ObjectLearningActorCritic):
    """Same complete actor/three critics/risk, with an isolated sensory residual."""
    source_schema = PARAMETER_SCHEMA

    def __init__(self, base, *, query_width=64, query_heads=4, query_seed=71953):
        nn.Module.__init__(self)
        if (not isinstance(base, ObjectLearningActorCritic)
                or isinstance(base, ActionQueryLearningActorCritic)
                or isinstance(base.sensory_adapter.object_attention, ActionQueryResidualObjects)):
            raise TypeError('An unwrapped, loaded object-learning model is required')
        cfg = base.core.config()
        if (cfg['n_neurons'] != 138639 or cfg['n_edges'] != 15091983 or cfg['K'] != 8
                or cfg['input_ports'] != 8053 or cfg['output_ports'] != 1299
                or not cfg['trainable_brain'] or base.policy_cost_gradient != 'supervised-only'
                or base.risk_head.beta != 8.):
            raise ValueError('Retain full K8/8053VP/1299DN,trainable brain,beta8 and supervised-only routing')
        old_ids = {id(p) for p in base.parameters()}
        self.core, self.actor, self.critic, self.risk_head = base.core, base.actor, base.critic, base.risk_head
        self.sensory_adapter = base.sensory_adapter
        self.initialization_seed = base.initialization_seed
        self.policy_cost_gradient = base.policy_cost_gradient
        self._base_metadata = base._base_metadata
        self._query_configuration = dict(schema=QUERY_SCHEMA, query_width=query_width,
            heads=query_heads, seed=query_seed)
        # The prepared residual initializes privately on CPU. torch.manual_seed
        # can also seed initialized CUDA generators, so preserve those here as
        # well when this explicit migration happens in a real CUDA process.
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
        with torch.random.fork_rng(devices=devices):
            self.sensory_adapter.object_attention = ActionQueryResidualObjects(
                self.sensory_adapter.object_attention, query_width=query_width,
                heads=query_heads, seed=query_seed).to(self.device)
        if not old_ids.issubset({id(p) for p in self.parameters()}):
            raise RuntimeError('All original Parameter objects must survive wrapping')
        self.train(base.training)

    @property
    def query_attention(self):
        return self.sensory_adapter.object_attention

    def query_parameters(self):
        return [p for name, p in self.query_attention.named_parameters()
            if not name.startswith('base_object_attention.')]

    def query_state(self):
        return {name: value for name, value in self.query_attention.state_dict().items()
            if not name.startswith('base_object_attention.')}

    def optimizer_groups(self, *, sensory_lr, query_lr=3e-5, **body_rates):
        for value in (sensory_lr, query_lr):
            if not math.isfinite(value) or value <= 0:
                raise ValueError('Explicit positive sensory/query learning rates required')
        # ObjectTrainableVisual.object_parameters() intentionally includes the
        # complete wrapped64 tensors. Split54/10 here so Adam has no duplicates.
        groups = PolicyCostRoutingActorCritic.optimizer_groups(self, **body_rates)
        groups.append(dict(name='sensory_object',
            params=list(self.query_attention.base_object_attention.parameters()), lr=sensory_lr))
        groups.append(dict(name=QUERY_GROUP, params=self.query_parameters(), lr=query_lr))
        return groups

    def configuration(self):
        cfg = ObjectLearningActorCritic.configuration(self)
        cfg['schema'] = PARAMETER_SCHEMA
        cfg['perception'] = dict(cfg['perception'],
            trainable='original54 object-attention tensors plus10 action-query tensors',
            current_learner_integrated=True, action_query_residual_integrated=True,
            runtime_input='same raw25/raw96x29;all18 fixed candidate descriptors;no selected-action/teacher input')
        cfg['action_query'] = dict(self.query_attention.specification(),
            current_model_collector_updater_integrated=True,
            integration='same existing object replay/384/fullbrain/DN path',
            initialized_zero_residual_at_migration=True,
            parameter_tensors=len(self.query_parameters()))
        cfg['action_query_configuration'] = copy.deepcopy(self._query_configuration)
        return cfg

    def _object_parameter_payload(self):
        """A genuine old-type nested payload: only bare base attention state."""
        body = _cpu(self._base_metadata)
        body.update(core_parameters=_cpu(dict(self.core.named_parameters())),
            actor=_cpu(self.actor.state_dict()), critic=_cpu(self.critic.state_dict()),
            risk_head=_cpu(self.risk_head.state_dict()))
        # This nested contract deliberately retains the old type. It does not
        # contain action-query names or pretend the residual is old attention.
        return dict(schema=OBJECT_PARAMETER_SCHEMA,
            configuration=ObjectLearningActorCritic.configuration(self), base_parameters=body,
            object_attention=_cpu(self.query_attention.base_object_attention.state_dict()),
            static_schema=STATIC_SCHEMA, static_cache_identity=self.sensory_adapter.static_cache_identity,
            contains_optimizer=False, contains_neural_state=False, contains_teacher_hidden=False)

    def parameter_checkpoint_payload(self):
        return dict(schema=PARAMETER_SCHEMA, configuration=self.configuration(),
            object_model_payload=self._object_parameter_payload(),
            query_configuration=copy.deepcopy(self._query_configuration), query_state=_cpu(self.query_state()),
            static_schema=STATIC_SCHEMA, static_cache_identity=self.sensory_adapter.static_cache_identity,
            contains_optimizer=False, contains_neural_state=False, contains_teacher_hidden=False)

    def load_parameter_payload(self, payload):
        if (payload.get('schema') != PARAMETER_SCHEMA
                or payload.get('static_schema') != STATIC_SCHEMA
                or payload.get('static_cache_identity') != self.sensory_adapter.static_cache_identity
                or not _same(payload.get('query_configuration'), self._query_configuration)
                or not _same(payload.get('configuration'), self.configuration())):
            raise ValueError('Exact typed query/body/static configuration required')
        old = payload['object_model_payload']
        if (old.get('schema') != OBJECT_PARAMETER_SCHEMA
                or old.get('static_schema') != STATIC_SCHEMA
                or old.get('static_cache_identity') != self.sensory_adapter.static_cache_identity
                or old['configuration']['policy_cost_gradient'] != 'supervised-only'
                or old['configuration']['risk_topology'] != TOPOLOGY):
            raise ValueError('Nested original object/deep subtype differs')
        body = old['base_parameters']
        if not _same({k:v for k,v in body.items() if k not in _DYNAMIC}, self._base_metadata):
            raise ValueError('Immutable graph/source perception/fixed normalization differs')
        _validate_state(dict(self.core.named_parameters()), body['core_parameters'], 'core')
        for name, module in (('actor',self.actor),('critic',self.critic),('risk_head',self.risk_head)):
            _validate_state(module.state_dict(), body[name], name)
        for key in ('layers.DN_mean','layers.DN_scale'):
            if not _same(self.risk_head.state_dict()[key], body['risk_head'][key]):
                raise ValueError('Fixed risk DN normalization differs')
        _validate_state(self.query_attention.base_object_attention.state_dict(), old['object_attention'], 'base_attention')
        _validate_state(self.query_state(), payload['query_state'], 'query')
        if not _same(self.query_state()['action_descriptors'], payload['query_state']['action_descriptors']):
            raise ValueError('Fixed source18 descriptor order/semantics differs')
        with torch.no_grad():
            for name,value in body['core_parameters'].items():
                dict(self.core.named_parameters())[name].copy_(value.to(self.device))
            for name,module in (('actor',self.actor),('critic',self.critic),('risk_head',self.risk_head)):
                module.load_state_dict(body[name], strict=True)
            self.query_attention.base_object_attention.load_state_dict(old['object_attention'], strict=True)
            own = self.query_attention.state_dict()
            for name,value in payload['query_state'].items():
                own[name].copy_(value.to(device=own[name].device))
        return self


def extend_loaded_object_model(model, optimizer, *, query_lr=3e-5, query_seed=71953):
    """Wrap an already loaded source; no old tensor/moment/group is replaced."""
    if not math.isfinite(query_lr) or query_lr <= 0:
        raise ValueError('Positive finite query learning rate required')
    if not isinstance(optimizer, torch.optim.Adam) or any(g.get('name') == QUERY_GROUP for g in optimizer.param_groups):
        raise ValueError('Original named Adam without a preexisting query group required')
    names = {id(p):name for name,p in model.named_parameters()}
    old_groups = [([id(p) for p in g['params']], copy.deepcopy({k:v for k,v in g.items() if k!='params'}))
        for g in optimizer.param_groups]
    old_states = {id(p):state for p,state in optimizer.state.items()}
    wrapped = ActionQueryLearningActorCritic(model, query_seed=query_seed)
    actual = dict(wrapped.named_parameters())
    for group in optimizer.param_groups:
        for p in group['params']:
            if actual.get(inherited_parameter_name(names[id(p)])) is not p:
                raise RuntimeError('An inherited Adam Parameter/name changed')
    optimizer.add_param_group(dict(name=QUERY_GROUP, params=wrapped.query_parameters(), lr=query_lr))
    for group,(ids,settings) in zip(optimizer.param_groups[:-1],old_groups):
        if [id(p) for p in group['params']] != ids or {k:v for k,v in group.items() if k!='params'} != settings:
            raise RuntimeError('Inherited optimizer settings/ordering changed')
    if any(optimizer.state[p] is not old_states[id(p)] for p in optimizer.state if id(p) in old_states):
        raise RuntimeError('An inherited Adam state object changed')
    if any(p in optimizer.state for p in wrapped.query_parameters()):
        raise RuntimeError('New query Adam moments must start empty')
    return wrapped, optimizer, dict(old_parameter_objects_preserved=True, old_Adam_states_preserved=True,
        old_group_settings_preserved=True, old_attention_name_translation=ATTENTION_PREFIX+'base_object_attention.',
        new_query_tensors=len(wrapped.query_parameters()), new_query_parameters=sum(p.numel() for p in wrapped.query_parameters()),
        query_group=QUERY_GROUP, query_lr=query_lr, new_query_Adam_states=0,
        source_body_heads_or_base_attention_reinitialized=False, source_policy_cost_gradient_preserved=True)


def migrate_object_training_source(source, connectome, *, query_lr=3e-5, device='cuda:0', restore_rng=True):
    from .continuous_object_learning_model import restore_object_training_source, _restore_rng
    if (source.get('schema') != OBJECT_TRAINING_SCHEMA
            or source['algorithm'].get('policy_cost_gradient') != 'supervised-only'):
        raise ValueError('Complete supervised-only object family source/Adam required')
    model, optimizer, counters = restore_object_training_source(source, connectome, device=device, restore_rng=False)
    model, optimizer, migration = extend_loaded_object_model(model, optimizer, query_lr=query_lr)
    if restore_rng:
        _restore_rng(source['RNG'])
    migration.update(source_RNG_restored=bool(restore_rng), native_midgame_resume=False)
    return model, optimizer, counters, migration


def build_action_query_model(payload, connectome, *, device='cuda:0'):
    from .continuous_object_learning_model import build_object_model
    if payload.get('schema') != PARAMETER_SCHEMA:
        raise ValueError('Explicit action-query parameter payload required')
    base = build_object_model(payload['object_model_payload'], connectome, device=device)
    q = payload['query_configuration']
    model = ActionQueryLearningActorCritic(base, query_width=q['query_width'], query_heads=q['heads'], query_seed=q['seed'])
    return model.load_parameter_payload(payload)


def action_query_training_checkpoint_payload(model, optimizer, *, algorithm, counters, provenance=None):
    from .continuous_connectome_training import training_checkpoint_payload
    if not isinstance(model, ActionQueryLearningActorCritic):
        raise TypeError('Typed action-query object model required')
    payload = training_checkpoint_payload(model, optimizer, algorithm=algorithm, counters=counters, provenance=provenance)
    payload['schema'] = TRAINING_SCHEMA
    payload['sensory_replay'] = dict(static_schema=STATIC_SCHEMA,
        static_cache_identity=model.sensory_adapter.static_cache_identity,
        uses_stale_encoded384_for_gradient=False, source_risk_topology=copy.deepcopy(TOPOLOGY),
        query_schema=QUERY_SCHEMA, query_group=QUERY_GROUP, action_queries_in_actual_forward=True)
    return payload


def restore_action_query_training_checkpoint(model, optimizer, source, *, restore_rng=True):
    from .continuous_connectome_training import restore_training_checkpoint, SCHEMA as BASE_TRAINING_SCHEMA
    if source.get('schema') != TRAINING_SCHEMA:
        raise ValueError('Complete action-query named-Adam training checkpoint required')
    return restore_training_checkpoint(model, optimizer, dict(source,schema=BASE_TRAINING_SCHEMA), restore_rng=restore_rng)


def restore_action_query_training_source(source, connectome, *, device='cuda:0', restore_rng=True):
    if source.get('schema') != TRAINING_SCHEMA:
        raise ValueError('New action-query training schema required;cannot masquerade as old object source')
    model = build_action_query_model(source['model_payload'], connectome, device=device)
    names = dict(model.named_parameters())
    signature, saved = source['optimizer_parameter_signature'], source['optimizer']['param_groups']
    if len(signature) != len(saved):
        raise ValueError('Complete named Adam signature/groups required')
    groups, seen = [], set()
    for specs, group in zip(signature,saved):
        parameters = []
        for spec in specs:
            p = names[spec['name']]
            if (spec['name'] in seen or list(p.shape) != spec['shape'] or str(p.dtype) != spec['dtype'] or not p.requires_grad):
                raise ValueError('Unique exact trainable named Adam tensor required')
            seen.add(spec['name'])
            parameters.append(p)
        groups.append(dict({k:copy.deepcopy(v) for k,v in group.items() if k!='params'},params=parameters))
    if (seen != {name for name,p in names.items() if p.requires_grad}
            or sum(g.get('name') == QUERY_GROUP for g in groups) != 1
            or sum(g.get('name') == 'sensory_object' for g in groups) != 1):
        raise ValueError('Every trainable tensor,one base-object and one query group required')
    optimizer = torch.optim.Adam(groups)
    counters = restore_action_query_training_checkpoint(model, optimizer, source, restore_rng=restore_rng)
    return model, optimizer, counters
