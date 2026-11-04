"""Explicit, auditable upgrade of v3 weights to rescue supervision/full chains.

No input/action/reward changes. Historical results stay historical; new Adam
steps use a declared hybrid objective, not a claim of identical continuation.
"""
from copy import deepcopy

REVISION = 'deathbomb-supervision-full-chain-v1'
AUTHORIZATION = 'v3-to-deathbomb-full-chain-v1'
FIELDS = ('learning_revision', 'curriculum_chain_horizon',
          'deathbomb_auxiliary_coefficient', 'deathbomb_auxiliary_min_remaining_frames')
DEFAULTS = dict(learning_revision=None, curriculum_chain_horizon='lesson-boundary-v1',
                deathbomb_auxiliary_coefficient=0., deathbomb_auxiliary_min_remaining_frames=2)
TARGET = dict(learning_revision=REVISION, curriculum_chain_horizon='native-game-v1',
              deathbomb_auxiliary_coefficient=.05, deathbomb_auxiliary_min_remaining_frames=2)
FULL_STORY_REVISION = 'single-policy-full-story-v1'
FULL_STORY_AUTHORIZATION = 'curriculum-to-single-policy-full-story-v1'
FULL_STORY_MODE = 'full-story-v1'
FULL_STORY_VALUES = {**TARGET, 'learning_revision':FULL_STORY_REVISION,
                     'curriculum_chain_horizon':None}
RETIRED_START_FIELDS = ('start_bank','start_bank_sha256','segment_end','full_story_fraction')


def full_story_template(config):
    """Remove every course/reset field; retain the model and optimizer contract."""
    result={k:deepcopy(v) for k,v in config.items()
            if not k.startswith('curriculum_') and k not in RETIRED_START_FIELDS}
    result.update({k:v for k,v in FULL_STORY_VALUES.items() if not k.startswith('curriculum_')})
    result.update(training_mode=FULL_STORY_MODE, first_hit_terminal=False,
                  native_frame_horizon=None, finite_native_lives=True, continues_allowed=False,
                  authorized_resume_upgrade=FULL_STORY_AUTHORIZATION)
    return result


def revision_values(config):
    result={k:config.get(k,DEFAULTS[k]) for k in FIELDS}
    if config.get('training_mode')==FULL_STORY_MODE:
        result['curriculum_chain_horizon']=None
    return result


def upgrade_record(source, target):
    before,after=revision_values(source),revision_values(target)
    modes=(source.get('training_mode'),target.get('training_mode'))
    if before==after and modes[0]==modes[1]:
        from .thrl_movement_learning import upgrade_record as movement_upgrade
        return movement_upgrade(source,target)
    if modes[1]==FULL_STORY_MODE:
        from .thrl_training_mode import validate_training_mode
        validate_training_mode(target)
        protected=('input_schema','training_recipe','reward_schema','policy_action_count',
                   'model_overrides','attention','ppo_baseline','learning_rate','gamma',
                   'gae_lambda','parameters','entropy_decay_decisions','reward_gamma','ppo_search_space')
        for name in protected:
            if source.get(name)!=target.get(name):
                raise ValueError('Full-story upgrade cannot alter '+name)
        if (modes[0] is not None or before not in (DEFAULTS,TARGET)
                or after!=FULL_STORY_VALUES
                or target.get('authorized_resume_upgrade')!=FULL_STORY_AUTHORIZATION
                or source.get('training_recipe')!='th14-completion-v3'
                or source.get('curriculum_protocol') not in (
                    'th14-progressive-rehearsal-curriculum-v1',
                    'th14-progressive-rehearsal-curriculum-v2')):
            raise ValueError('Full-story revision requires an explicit supported migration')
        return dict(schema=FULL_STORY_AUTHORIZATION,before=before,after=after,
                    before_training_mode=modes[0],after_training_mode=modes[1],
                    weights_and_Adam_preserved=True,old_rollout_reused=False,
                    retired_curriculum={k:deepcopy(v) for k,v in source.items()
                        if k.startswith('curriculum_') or k in RETIRED_START_FIELDS},
                    source_teacher_protocol={k:source.get(k) for k in
                        ('training_recipe','reward_schema','input_schema','curriculum_protocol')},
                    objective='one policy, fresh story starts, native terminal; existing PPO and Bomb supervision',
                    curriculum_state_restored=False,historical_scores_are_pre_upgrade=True,
                    runtime_action_override=False)
    for name in ('input_schema','training_recipe','reward_schema','curriculum_protocol',
                 'policy_action_count','model_overrides','attention','ppo_baseline',
                 'learning_rate','gamma','gae_lambda','parameters','entropy_decay_decisions',
                 'reward_gamma','curriculum_options','curriculum_manifest_sha256','ppo_search_space'):
        if source.get(name)!=target.get(name):
            raise ValueError('Learning upgrade cannot alter '+name)
    if (source.get('training_recipe')!='th14-completion-v3'
            or source.get('policy_action_count')!=19
            or before!=DEFAULTS or after!=TARGET
            or target.get('authorized_resume_upgrade')!=AUTHORIZATION):
        raise ValueError('Learning revision differs without an explicit supported upgrade')
    return dict(schema=AUTHORIZATION, before=before, after=after,
                weights_and_Adam_preserved=True, old_rollout_reused=False,
                objective='PPO plus rare-state supervised Bomb target',
                runtime_action_override=False, historical_scores_are_pre_upgrade=True)


def auxiliary_parameters(config):
    values=revision_values(config)
    if values['learning_revision'] is None:
        if values['deathbomb_auxiliary_coefficient']!=0:
            raise ValueError('Auxiliary supervision requires a declared learning revision')
        return {}
    if values not in (TARGET,FULL_STORY_VALUES):raise ValueError('Unknown or inconsistent learning revision')
    from .thrl_movement_learning import parameters as movement_parameters
    return {**{k:values[k] for k in FIELDS if k.startswith('deathbomb_auxiliary_')},
            **movement_parameters(config)}


def verify_ppo_resume(prior,current,config):
    prior,current=deepcopy(prior),deepcopy(current)
    prior.pop('update_index',None);current.pop('update_index',None)
    if prior==current:return None
    migration=config.get('learning_migration') or {}
    from .thrl_movement_learning import AUTHORIZATION as MOVEMENT_AUTH,parameters as movement_parameters
    if migration.get('schema')==MOVEMENT_AUTH:
        from .thrl_movement_learning import contract as movement_contract,PARAMETER_DEFAULTS,REVISION as MOVEMENT_REVISION,TARGET as MOVEMENT_TARGET
        if (migration.get('before')!={'revision':None,**PARAMETER_DEFAULTS}
                or migration.get('after')!={'revision':MOVEMENT_REVISION,**MOVEMENT_TARGET}
                or movement_contract(config)!=migration['after']
                or config.get('authorized_movement_upgrade')!=MOVEMENT_AUTH):
            raise ValueError('Movement recovery requires its verified source/target migration')
        expected={**prior,**movement_parameters(config)}
        if expected!=current:raise ValueError('Movement resume may change only its declared auxiliary parameters')
        return {k:{'before':prior.get(k),'after':current[k]} for k in current if prior.get(k)!=current[k]}
    legacy=(migration.get('schema')==AUTHORIZATION and migration.get('before')==DEFAULTS
            and migration.get('after')==TARGET and revision_values(config)==TARGET)
    full=(migration.get('schema')==FULL_STORY_AUTHORIZATION and migration.get('before')==DEFAULTS
          and migration.get('after')==FULL_STORY_VALUES and revision_values(config)==FULL_STORY_VALUES)
    if not (legacy or full):
        raise ValueError('Recovery cannot change source PPO learning parameters')
    expected=deepcopy(prior)
    if expected.get('deathbomb_auxiliary_coefficient',0.)!=0:
        raise ValueError('Only a previously disabled auxiliary objective can be upgraded')
    expected.update(auxiliary_parameters(config))
    if expected!=current:
        raise ValueError('Learning upgrade may change only declared auxiliary PPO parameters')
    return {k:{'before':prior[k],'after':current[k]} for k in current if prior.get(k)!=current[k]}


def upgraded_course_state(state,config):
    if state is None:return None
    if config.get('training_mode')==FULL_STORY_MODE:
        if (config.get('learning_migration') or {}).get('schema')!=FULL_STORY_AUTHORIZATION:
            raise ValueError('Discarding legacy curriculum state requires an explicit migration')
        return None
    result=deepcopy(state)
    target=config.get('curriculum_chain_horizon',DEFAULTS['curriculum_chain_horizon'])
    old=result.get('curriculum_chain_horizon',DEFAULTS['curriculum_chain_horizon'])
    if old==target:return result
    migration=config.get('learning_migration') or {}
    if (old!=DEFAULTS['curriculum_chain_horizon'] or target!=TARGET['curriculum_chain_horizon']
            or migration.get('schema')!=AUTHORIZATION or revision_values(config)!=TARGET):
        raise ValueError('Curriculum horizon upgrade requires an explicit migration record')
    result['curriculum_chain_horizon']=target
    if result.get('gate'):
        result['gate']={**result['gate'],'inherited':True,
            'source_learning_revision':'completion-v3-base',
            'source_optimized_decisions':state.get('last_gate_decisions'),
            'upgrade_boundary_optimized_decisions':migration.get('boundary_optimized_decisions')}
    # Gates still measure identical target segments, so keep frontier/depth.
    # Keep paid decisions and RNG; new whole-game outcomes carry separate receipts.
    return result


def apply_learning_upgrade(source,target,migration=None):
    """Copy effective source settings, replacing only the supported revision."""
    verified=upgrade_record(source,target)
    if migration is not None and (verified is None or migration.get('schema')!=verified['schema']):
        raise ValueError('Learning migration does not match the requested source and target')
    result=deepcopy(source)
    if verified is None:return result
    from .thrl_movement_learning import AUTHORIZATION as MOVEMENT_AUTH
    if verified['schema']==MOVEMENT_AUTH:
        for k in ('movement_guidance_revision','movement_guidance','authorized_movement_upgrade'):
            result[k]=deepcopy(target[k])
        return result
    if verified['schema']==FULL_STORY_AUTHORIZATION:
        result=full_story_template(result)
        if 'full_story_burn_in' in target:result['full_story_burn_in']=target['full_story_burn_in']
    else:
        result.update({k:target[k] for k in FIELDS})
        result['authorized_resume_upgrade']=target['authorized_resume_upgrade']
    return result
