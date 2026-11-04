"""Thin native adapter; reuse the existing own-state cursor and full-story factory."""
from pathlib import Path

import torch
from torch import nn

from .fly_centered_native_evaluation import CenteredNativePolicy
from .fly_visual_distillation import VisualFlyDistillationStudent
from .thrl_inputs import SEPARATED_LASER_INPUT


class VisualNativePolicy(CenteredNativePolicy):
    """The inherited scalar cursor is not a GRU or the student's neural state."""
    def __init__(self,student,*,episodes=1,brain_seed=613711,synthetic_fixture=False):
        nn.Module.__init__(self)
        if (not isinstance(student,VisualFlyDistillationStudent) or type(episodes) is not int
                or not 1<=episodes<=20 or type(brain_seed) is not int or not 0<=brain_seed<2**31):
            raise ValueError('Typed visual384 student, finite1..20 games and explicit brain seed required')
        brain=student.brain
        if (brain.substeps!=100 or brain.dt_ms!=.2 or brain.obs_dim!=384 or brain.n_actions!=19
                or brain.recurrent_gain_limit!=.1 or brain.action_readout.bias is not None
                or brain.geometry_input_encoder is not None):
            raise ValueError('Require pure visual384/100-step/gain.1/DN biasfree19')
        if synthetic_fixture:
            if brain.require_full_graph or student.device.type!='cpu':
                raise ValueError('Artificial native adapter fixtures must be CPU/non-fullgraph')
        elif not brain.require_full_graph:
            raise ValueError('Native execution requires the verified complete graph')
        student._check_frozen_frontend(full=brain.require_full_graph)
        self.student=student
        self.episodes,self.brain_seed=episodes,brain_seed
        self.state=None
        self.started_games=self.data_forwards=0
        self.episode_records=[]


def evaluate_visual_native(student,config,output,*,episodes=1,seed=2027431401,
                           brain_seed=613711,check_stop=lambda:None,progress=None):
    """One private worker; existing capacity/GO/clear/cleanup path, no teacher."""
    from .thrl_training_mode import validate_training_mode,episode_protocol
    from .thrl_full_story_evaluation import evaluate
    from .thrl_durable_records import write_json
    import train_thrl_native as native_entry

    if (not validate_training_mode(config) or config.get('input_schema')!=SEPARATED_LASER_INPUT
            or config.get('headless') is not True or config.get('autonomous_bomb') is not True):
        raise ValueError('Existing natural full-story headless V5/source19 configuration required')
    policy=VisualNativePolicy(student,episodes=episodes,brain_seed=brain_seed).eval()
    kwargs=native_entry.make_env_kwargs(config,Path(output),1,'development')
    if (len(kwargs)!=1 or not kwargs[0]['headless'] or not kwargs[0]['hide_window']
            or kwargs[0]['policy_action_count']!=19 or kwargs[0]['frame_skip']!=4
            or kwargs[0]['wall_time_limit_seconds'] is not None or 'online_teacher_revision' in kwargs[0]):
        raise ValueError('One private natural source19 worker required')
    protocol=dict(episode_protocol(config),recurrent_refresh='none;frozen own delayed-LIF history from opening',
        neural_state='full own brain state;scalar evaluator cursor is not neural state',
        neural_reset='new original episode only;no stage/Miss reset',implicit_warmup=False)
    result=evaluate(torch,policy,kwargs,output,episodes=episodes,seed=seed,
        check_stop=check_stop,progress=progress,native_split='development',episode_protocol=protocol,
        summary=native_entry.development_summary,action_selection='greedy',action_temperature=.25)
    result['fruitfly_policy']=dict(student_schema=student.specification()['schema'],
        data_neural_forwards=policy.data_forwards,episode_records=policy.episode_records,
        action_path='rawV5->frozen D95 perception384->fixed train000 standardization->1024ports->same full delayed LIF->DN biasfree19',
        teacher_actor_hidden_risk_used=False,optimizer_updates=0,stage_or_Miss_resets=0,
        initial_state='zero own state;explicit per-game Poisson key;no warmup')
    write_json(Path(output)/'report.json',result)
    return result
