"""Offline single-factor age encoding candidate; not a registered training input.

Changing normalization changes checkpoint semantics. This candidate is never
selected by make_encoder or current control-panel templates. It needs its own
versioned, from-common-initialization experiment before deployment.
"""
from __future__ import annotations

import numpy as np

from .thrl_geometry_observation import (OBJECT_FIELDS, shared_object_candidates,
                                      select_object_candidates)
from .thrl_inputs import SEPARATED_TIMERS_INPUT
from .thrl_timer_observation import THRLTimerObservationEncoder

CANDIDATE_SCHEMA='th14-separated-timers-soft-age-offline-candidate-v1'
AGE_COLUMN=OBJECT_FIELDS.index('age')


def bounded_age(age):
    """Signed t/(120+abs(t)); finite float32, not lossless integer transport."""
    values=np.asarray(age,dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError('Finite measured age required')
    return (values/(120.+np.abs(values))).astype(np.float32)


def candidate_object_tokens(snapshot):
    """Same geometry, identities, top-K, order, shape and 20 other columns."""
    prepared=shared_object_candidates(snapshot)
    selected=select_object_candidates(prepared['candidates'])
    result=np.zeros((96,len(OBJECT_FIELDS)),dtype=np.float32)
    for index,row in enumerate(selected):
        token=row[2]
        result[index]=np.clip(token,-1.,1.)
        # The shared constructor retains pre-clip age/120 at this column.
        result[index,AGE_COLUMN]=bounded_age(token[AGE_COLUMN]*120.)
    return result


class THRLSoftAgeCandidateEncoder(THRLTimerObservationEncoder):
    def __init__(self,*,require_events=True):
        super().__init__(SEPARATED_TIMERS_INPUT,require_events=require_events)

    def specification(self):
        result=super().specification()
        result.update(schema=CANDIDATE_SCHEMA,experimental_offline_only=True,
            source_input_schema=SEPARATED_TIMERS_INPUT,checkpoint_compatible=False,
            registered_for_training=False,
            changed_field='objects.age only: t/(120+abs(t)); all other fields unchanged',
            expected_learning_improvement='unmeasured')
        return result

    def encode(self,snapshot):
        result=super().encode(snapshot)
        return {**result,'objects':candidate_object_tokens(snapshot)}
