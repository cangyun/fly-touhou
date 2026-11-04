"""Versioned observation shapes. Importable by workers without Torch/Frida."""
ORIGINAL_INPUT = 'original273-v1'
SEPARATED_INPUT = 'semantic-separated-v1'
ORIGINAL_TIMERS_INPUT = 'original273-timers-v2'
SEPARATED_TIMERS_INPUT = 'semantic-separated-timers-v2'
SEPARATED_OBJECT_PRIORITY_INPUT = 'semantic-separated-timers-object-priority-v3'
SEPARATED_CONTACT_INPUT = 'semantic-separated-timers-hp-independent-contact-v4'
SEPARATED_LASER_INPUT = 'semantic-separated-observed-laser-segments-v5'
COMPACT_SPIKING_INPUT = 'connectome-compact111-no-map-v1'
COMPACT_STORY_INPUT = 'connectome-compact129-full-story-v1'
PRIORITY_INPUT_SCHEMAS = (SEPARATED_OBJECT_PRIORITY_INPUT,SEPARATED_CONTACT_INPUT,SEPARATED_LASER_INPUT)
ORIGINAL_SCHEMAS = (ORIGINAL_INPUT, ORIGINAL_TIMERS_INPUT)
SEPARATED_SCHEMAS = (SEPARATED_INPUT, SEPARATED_TIMERS_INPUT, *PRIORITY_INPUT_SCHEMAS)
TIMER_SCHEMAS = (ORIGINAL_TIMERS_INPUT, SEPARATED_TIMERS_INPUT, *PRIORITY_INPUT_SCHEMAS)
SEPARATED_TIMER_SCHEMAS = (SEPARATED_TIMERS_INPUT, *PRIORITY_INPUT_SCHEMAS)
VERSIONED_SCHEMAS = ORIGINAL_SCHEMAS + SEPARATED_SCHEMAS + (COMPACT_SPIKING_INPUT,COMPACT_STORY_INPUT)


def is_separated_input(schema):
    return schema in SEPARATED_SCHEMAS


def is_original_input(schema):
    return schema in ORIGINAL_SCHEMAS


def feature_dimension(schema):
    if schema==COMPACT_STORY_INPUT:return 129
    if schema==COMPACT_SPIKING_INPUT:return 111
    dimensions = {ORIGINAL_INPUT: 273, SEPARATED_INPUT: 20,
                  ORIGINAL_TIMERS_INPUT: 279, SEPARATED_TIMERS_INPUT: 25,
                  SEPARATED_OBJECT_PRIORITY_INPUT: 25,SEPARATED_CONTACT_INPUT:25,SEPARATED_LASER_INPUT:25}
    if schema not in dimensions:
        raise ValueError('Unknown observation input schema: '+str(schema))
    return dimensions[schema]


def global_feature_dimension(schema):
    return (23 if schema == ORIGINAL_INPUT else 29) if is_original_input(schema) else feature_dimension(schema)


def policy_is_recurrent(config):
    model=config.get('model_config') or config.get('model_overrides') or {}
    core=model.get('temporal_core','gru')
    if core not in ('gru','mlp'):raise ValueError('Unknown policy temporal core')
    return core=='gru'


def observation_shapes(schema=SEPARATED_INPUT):
    if schema==COMPACT_STORY_INPUT:return {'features':(129,)}
    if schema==COMPACT_SPIKING_INPUT:return {'features':(111,)}
    if is_original_input(schema):
        return {'features':(feature_dimension(schema),), 'maps':(16,92,96)}
    if is_separated_input(schema):
        from .thrl_geometry_observation import OBJECT_COUNT, OBJECT_DIM
        return {'features':(feature_dimension(schema),), 'maps':(16,92,96), 'objects':(OBJECT_COUNT,object_dimension(schema))}
    raise ValueError('Unknown observation input schema: '+str(schema))


def object_dimension(schema):
    if not is_separated_input(schema):raise ValueError('Independent objects require a separated input')
    return 29 if schema==SEPARATED_LASER_INPUT else 21


def make_encoder(schema=SEPARATED_INPUT):
    if schema==COMPACT_STORY_INPUT:
        from .compact_story_sensation import CompactStoryEncoder
        return CompactStoryEncoder()
    if schema==COMPACT_SPIKING_INPUT:
        from .compact_spiking_sensation import CompactObservationEncoder
        return CompactObservationEncoder()
    if schema == SEPARATED_LASER_INPUT:
        from .thrl_laser_observation import THRLLaserObservationEncoder
        return THRLLaserObservationEncoder()
    if schema == ORIGINAL_INPUT:
        from .thrl_observation import THRLObservationEncoder
        return THRLObservationEncoder()
    if schema == SEPARATED_INPUT:
        from .thrl_geometry_observation import THRLGeometryObservationEncoder
        return THRLGeometryObservationEncoder()
    if schema in TIMER_SCHEMAS:
        from .thrl_timer_observation import THRLTimerObservationEncoder
        return THRLTimerObservationEncoder(schema)
    raise ValueError('Unknown observation input schema: '+str(schema))


def forward_observation(torch, model, observation, hidden):
    """Transfer modalities independently; never concatenate raw observations."""
    device=hidden.device
    extra=({'objects':torch.as_tensor(observation['objects'],device=device)} if 'objects' in observation else {})
    return model.forward_step(torch.as_tensor(observation['features'],device=device),
        torch.as_tensor(observation['maps'],device=device) if 'maps' in observation else None,hidden,**extra)
