"""Versioned TH14 native protection/state scalars; original sensors stay frozen.

The current native invulnerability counter is not the entire remaining Bomb
duration: native effects can refresh it. No unverified Bomb-active/timer field
or inferred legal-action label is exposed as a measured observation.
"""
from __future__ import annotations

import math
import numpy as np

from .thrl_inputs import (ORIGINAL_TIMERS_INPUT, SEPARATED_TIMERS_INPUT,
                         SEPARATED_OBJECT_PRIORITY_INPUT,SEPARATED_CONTACT_INPUT,PRIORITY_INPUT_SCHEMAS,
                         TIMER_SCHEMAS, feature_dimension)

PROTECTION_TIME_SCALE_FRAMES = 60.0
DEATHBOMB_WINDOW_FRAMES = 8
STATE_FIELDS = tuple(f'player_native_state_{state}' for state in (1, 2, 3, 4))
TIMER_FIELDS = ('invulnerability_remaining_bounded', 'deathbomb_remaining_8frames', *STATE_FIELDS)


def timer_features(player):
    """Current native frames -> finite floats without an invented timer cap.

    t/(t+60) is monotonic and invertible in real arithmetic. Float32 is the
    ordinary observation precision; this is not a lossless integer transport.
    Zero/negative protection means no active counter. State0 has all-zero
    one-hot, whereas states1..4 keep their categorical identity.
    """
    frames = float(player.invulnerable_frames)
    remaining = float(player.deathbomb_timer)
    if (not math.isfinite(frames) or not math.isfinite(remaining)
            or frames != int(frames) or not 0 <= remaining <= DEATHBOMB_WINDOW_FRAMES
            or remaining != int(remaining) or player.state not in (0, 1, 2, 3, 4)):
        raise ValueError('Invalid measured TH14 player protection/state counters')
    frames = max(0., frames)
    # Reader only defines a deathbomb countdown in native state4.
    if player.state != 4 and remaining != 0:
        raise ValueError('Deathbomb countdown outside native state4')
    return np.asarray([frames/(frames+PROTECTION_TIME_SCALE_FRAMES),
        remaining/DEATHBOMB_WINDOW_FRAMES,
        *(float(player.state == state) for state in (1, 2, 3, 4))], dtype=np.float32)


class THRLTimerObservationEncoder:
    """Compose a frozen v1 sensor, changing only explicit v2 numeric features."""
    def __init__(self, schema, *, require_events=True):
        from .thrl_inputs import SEPARATED_LASER_INPUT
        if schema==SEPARATED_LASER_INPUT:
            raise ValueError('V5 laser segments require make_encoder, not a historical timer-only composer')
        if schema not in TIMER_SCHEMAS:
            raise ValueError('An explicit timers-v2 observation schema is required')
        self.schema = schema
        if schema == ORIGINAL_TIMERS_INPUT:
            from .thrl_observation import THRLObservationEncoder
            self.base = THRLObservationEncoder(require_events=require_events)
            self.feature_names = self.base.feature_names + TIMER_FIELDS
        else:
            from .thrl_geometry_observation import THRLGeometryObservationEncoder
            self.base = THRLGeometryObservationEncoder(require_events=require_events,
                nonitem_first=schema in PRIORITY_INPUT_SCHEMAS,
                hp_independent_contacts=schema==SEPARATED_CONTACT_INPUT)
            self.feature_names = (*self.base.feature_names[:19], TIMER_FIELDS[1],
                                  TIMER_FIELDS[0], *STATE_FIELDS)
            self.object_shape = self.base.object_shape
        self.feature_shape = (feature_dimension(schema),)
        self.map_shape = self.base.map_shape

    def specification(self):
        spec = self.base.specification()
        source_schema = spec['schema']
        spec.update(schema=self.schema, source_observation_schema=source_schema,
                    checkpoint_compatible_with_v1=False)
        if self.schema in PRIORITY_INPUT_SCHEMAS:
            spec.update(checkpoint_compatible_with_timers_v2=False,
                sole_change_from_timers_v2='non-item object allocation before surplus items',
                object_age_transform_unchanged=True)
        if self.schema==SEPARATED_CONTACT_INPUT:
            spec.pop('sole_change_from_timers_v2',None)
            spec.update(checkpoint_compatible_with_object_priority_v3=False,
                sole_change_from_object_priority_v3='retain measured contactable enemies independently of HP;consistent nonboss map footprints',
                numeric_feature_count_unchanged=True,object_fields_unchanged=True,
                object_selection_quotas_unchanged=True,legacy_v3_encoder_unchanged=True)
        spec['features'].update(shape=list(self.feature_shape), names=list(self.feature_names))
        if self.schema == ORIGINAL_TIMERS_INPUT:
            spec['features']['layout']['native_player_timers'] = [273, 279]
        spec['native_player_timing'] = {
            'source': 'Player from pinned TH14 NativeReader: zPlayer_iframes and state4 elapsed clock',
            'units': 'original simulation frames, independent of wall speed and action repeat',
            'invincible_boolean_retained': True,
            'invulnerability_transform': 'max(t,0)/(max(t,0)+60)',
            'invulnerability_reference_frames': PROTECTION_TIME_SCALE_FRAMES,
            'invulnerability_hard_maximum_assumed': None,
            'invulnerability_scope': 'current protection counter; effects may refresh it; not total remaining Bomb duration',
            'deathbomb_transform': 'native state4 remaining frames / 8; zero in all other states',
            'native_state_encoding': 'one-hot states1,2,3,4; state0 all-zero; unknown states rejected',
            'bomb_active_or_remaining_duration': 'not encoded: no verified snapshot field',
            'observation_only': True,
        }
        return spec

    def encode(self, snapshot):
        observation = self.base.encode(snapshot)
        values = timer_features(snapshot.player)
        if self.schema == ORIGINAL_TIMERS_INPUT:
            features = np.concatenate((observation['features'], values))
        else:
            base = observation['features'].copy()
            base[19] = values[1]
            features = np.concatenate((base, values[:1], values[2:]))
        if (features.shape != self.feature_shape or not np.isfinite(features).all()
                or np.abs(features).max() > 1):
            raise ValueError('Timer observation schema/finite/range invariant failed')
        return {**observation, 'features': features.astype(np.float32, copy=False)}
