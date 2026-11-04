"""Offline-only single-factor object allocation candidate.

Not registered in make_encoder or any runtime configuration. All tokens,
normalization, maps, globals, ranking and reserve counts remain unchanged;
only filling the slots left after reserves changes. Do not combine with the
soft-age candidate or silently resume an existing checkpoint under this input.
"""
from __future__ import annotations

import numpy as np

from .thrl_geometry_observation import (OBJECT_COUNT, OBJECT_DIM,
    THRLGeometryObservationEncoder, shared_object_candidates)
from .thrl_inputs import SEPARATED_INPUT, SEPARATED_TIMERS_INPUT
from .thrl_timer_observation import THRLTimerObservationEncoder

CANDIDATE_SCHEMA='th14-object-selection-nonitem-fill-offline-v1'


def select_nonitem_fill_candidates(candidates):
    """Same ranking/reserves; unreserved items cannot evict non-item objects.

    If some non-item is omitted, at most the existing eight reserved items
    remain. With fewer non-items, spare slots may contain additional items.
    Return rows in the original semantic rank order, never selection order.
    """
    ranked=sorted(candidates,key=lambda row:(row[0],row[1]))
    selected=set()
    for kind,quota in (('boss',2),('item',8),('enemy',6),('laser',16)):
        selected.update([index for index,row in enumerate(ranked) if row[3]==kind][:quota])
    for kind_is_item in (False,True):
        for index,row in enumerate(ranked):
            if len(selected)>=OBJECT_COUNT:break
            if (row[3]=='item')==kind_is_item:selected.add(index)
    return [ranked[index] for index in sorted(selected)]


class OfflineObjectSelectionCandidateEncoder:
    """Explicit offline encoder; input-selection semantics are incompatible.

    require_events=False is only for parity on saved passive snapshots without
    event counters. It is not evidence that missing counters were actually0.
    """
    training_eligible=False
    checkpoint_compatible=False

    def __init__(self, *, input_schema=SEPARATED_TIMERS_INPUT, require_events=True):
        if input_schema not in (SEPARATED_INPUT,SEPARATED_TIMERS_INPUT):
            raise ValueError('Offline selection candidate requires separated object inputs')
        self.input_schema=input_schema
        self.base=(THRLTimerObservationEncoder(input_schema,require_events=require_events)
            if input_schema==SEPARATED_TIMERS_INPUT else
            THRLGeometryObservationEncoder(require_events=require_events))
        self.feature_shape=self.base.feature_shape
        self.object_shape=(OBJECT_COUNT,OBJECT_DIM)
        self.map_shape=self.base.map_shape
        self.feature_names=self.base.feature_names

    def specification(self):
        spec=self.base.specification()
        spec.update(schema=CANDIDATE_SCHEMA,source_input_schema=self.input_schema,
            offline_only=True,training_eligible=False,checkpoint_compatible=False,
            sole_changed_factor='fill unreserved object slots with non-items before extra items',
            combined_with_age_candidate=False)
        spec['objects']['selection']=('unchanged semantic ranking and reserves: up to2 bosses,8 items,'
            '6 enemies,16 laser samples; fill by original rank among non-items, then extra items only'
            ' if capacity remains; output original rank order')
        return spec

    def encode(self,snapshot):
        observation=self.base.encode(snapshot)
        prepared=shared_object_candidates(snapshot)
        objects=np.zeros(self.object_shape,np.float32)
        for index,row in enumerate(select_nonitem_fill_candidates(prepared['candidates'])):
            objects[index]=np.clip(row[2],-1,1)
        return {**observation,'objects':objects}
