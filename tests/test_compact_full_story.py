from fly_touhou.compact_full_story import CompactFullStoryEnv
from fly_touhou.thrl_full_story_env import THRLFullStoryEnv


def test_miss_and_stage_transition_do_not_end_compact_full_game(monkeypatch):
    env=CompactFullStoryEnv.__new__(CompactFullStoryEnv);env._teacher=None;env._teacher_label=None
    env.encoder=None
    monkeypatch.setattr(THRLFullStoryEnv,'step',lambda self,a:({'features':None},0.,False,False,dict(misses=1,stage=2)))
    _,_,done,truncated,info=env.step(0)
    assert not done and not truncated and info['misses']==1 and info['stage']==2


def test_native_gameover_is_preserved(monkeypatch):
    env=CompactFullStoryEnv.__new__(CompactFullStoryEnv);env._teacher=None;env._teacher_label=None
    env.encoder=None
    monkeypatch.setattr(THRLFullStoryEnv,'step',lambda self,a:({},-1.,True,False,dict(episode_summary={'outcome':'gameover'})))
    _,_,done,truncated,info=env.step(0)
    assert done and not truncated and info['episode_summary']['outcome']=='gameover'


def test_respawn_sprite_outside_playfield_has_bounded_globals():
    import numpy as np
    from fly_touhou.compact_spiking_sensation import encode_geometry
    f=np.zeros(25,np.float32);f[0]=.5;f[1]=1.12;f[22]=1
    x,details=encode_geometry(f,np.zeros((0,29),np.float32),complete=True)
    assert x[1]==1 and not details['normal'] and np.abs(x).max()<=1


def test_formal_evaluation_routes_to_test_seed_partition_without_starting_a_game(monkeypatch,tmp_path):
    import json,pytest
    from types import SimpleNamespace
    from fly_touhou import compact_full_story as story,fly_dn_cost_native_policy as native,thrl_vector
    captured={}
    def kwargs(config,output,workers,role):captured['role']=role;return []
    class FinishedProbe(Exception):pass
    class Vector:
        def __init__(self,*args):raise FinishedProbe
    old=thrl_vector.worker_factory
    monkeypatch.setattr(story,'story_kwargs',kwargs)
    monkeypatch.setattr(native,'make_dn_cost_evaluation_vector',lambda *args:Vector)
    with pytest.raises(FinishedProbe):
        story.evaluate(SimpleNamespace(input_schema='connectome-compact129-full-story-v1'),{},tmp_path/'heldout',
            episodes=2,evaluation_split='test')
    assert captured['role']=='test' and thrl_vector.worker_factory is old
    plan=json.loads((tmp_path/'heldout/episode_plan.json').read_text())
    assert plan['evaluation_split']=='test' and plan['optimizer_updates']==plan['teacher_actions']==0
