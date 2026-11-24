from copy import deepcopy
from types import SimpleNamespace as NS
import numpy as np
import pytest

from fly_touhou.nbnm_event_progress import ATTACKS, EventProgress, event_unit
from fly_touhou.nbnm_objective import Objective


def scene(stage=1, name=None, *, observed=True, hp=100, frame=100):
    kind = 'midbosses' if name and name.startswith('MBoss') else 'main_bosses'
    roles = dict(observed=observed, main_bosses=[], midbosses=[])
    if name:
        roles[kind] = [dict(ecl_sub_name=name, hp=hp)]
    return NS(stage=stage, frame=frame, extra={'thrl_events': {'ecl_roles': roles}})


def test_six_stages_pay_exactly_one_bounded_global_budget_once():
    tracker = EventProgress(dict(mode='full_story'))
    tracker.begin(scene())
    total = 0.
    for stage, attacks in ATTACKS.items():
        if stage > 1:
            total += tracker.advance(scene(stage))[0]
        for attack in attacks:
            total += tracker.advance(scene(stage, attack))[0]
            assert tracker.advance(scene(stage, attack, hp=0, frame=200))[0] == 0
        if stage == 6:
            total += tracker.advance(scene(6), terminal=2, reason='native-six-stage-clear')[0]
    assert total == pytest.approx(1.)
    assert tracker.report()['counts'] == dict(road=12, nonspell=29, spell=25, stage_clear=6)
    assert len(tracker.events) == 72
    assert len(tracker.report()['attack_attempts']) == 54
    assert all(r['confirmed_complete'] for r in tracker.report()['attack_attempts'])
    assert tracker.advance(scene(6), terminal=2)[0] == 0
    objective = Objective(progress_revision='native-events-v1', progress_weight=.2)
    assert objective.weights == (1., 0., .2)
    assert float(objective.rewards(terminal=True, success=True, native_frames=40000,
        event_progress=total) @ objective.weights) == pytest.approx(1.2)


def test_hp_disappearance_helpers_and_unobserved_roles_are_not_events():
    tracker = EventProgress(dict(mode='stage', stage=1))
    tracker.begin(scene(1, 'BossCard1'))
    for snapshot in [scene(1, 'BossCard1', hp=0), scene(1), scene(1, 'BossHide'),
            scene(1, 'BossCard1Move'), scene(1, 'Boss2', observed=False)]:
        assert tracker.advance(snapshot) == (0., [])
    progress, rows = tracker.advance(scene(1, 'Boss2'))
    assert progress == event_unit(1) and rows[0]['kind'] == 'spell'
    assert tracker.advance(scene(1, 'BossCard1'))[0] == 0  # stale/backward phase
    assert tracker.advance(scene(1, 'Boss2'))[0] == 0


def test_first_contact_overrides_same_frame_phase_or_stage_completion():
    tracker = EventProgress(dict(mode='full_story'))
    tracker.begin(scene(1, 'BossCard2'))
    assert tracker.advance(scene(2), terminal=1, reason='first-native-contact') == (0., [])
    assert tracker.total == 0


def test_local_card_gets_only_its_event_and_never_prefix_or_stage_bonus():
    tracker = EventProgress(dict(mode='attack', stage=6, attack='BossCard6'))
    tracker.begin(scene(6, 'BossCard6'))
    progress, rows = tracker.advance(scene(6), terminal=2, reason='native-spell-capture')
    assert progress == event_unit(6)
    assert [r['key'] for r in rows] == ['s6/BossCard6']
    assert len(tracker.report()['attack_attempts'])==1
    assert tracker.advance(scene(6), terminal=2)[0] == 0


def test_same_attack_receives_same_global_unit_in_every_task_scope():
    tasks = [dict(mode='full_story'), dict(mode='stage', stage=2),
        dict(mode='chain', stage=2, attack='Boss1', end_stage=2, end_attack='BossCard2'),
        dict(mode='attack', stage=2, attack='BossCard1')]
    for task in tasks:
        tracker = EventProgress(task)
        tracker.begin(scene(2, 'BossCard1'))
        amount, rows = tracker.advance(scene(2, 'Boss2'), terminal=2 if task['mode']=='attack' else 0)
        assert amount == event_unit(2) and len(rows) == 1


def test_midboss_chain_earns_remaining_road_but_not_opening_road():
    tracker = EventProgress(dict(mode='chain', stage=3, attack='MBoss2', end_stage=3))
    tracker.begin(scene(3, 'MBoss2'))
    tracker.advance(scene(3, 'MBossCard1'))
    tracker.advance(scene(3))
    tracker.advance(scene(3, 'Boss1'))
    assert tracker.report()['counts'] == dict(road=1, nonspell=1, spell=1, stage_clear=0)
    assert 's3/road-before-midboss' not in tracker.credited


@pytest.mark.parametrize('prefix', ['Boss', 'BossA', 'BossB'])
def test_stage_four_route_variants_have_identical_event_slots(prefix):
    tracker = EventProgress(dict(mode='attack', stage=4, attack=prefix+'Card1'))
    tracker.begin(scene(4, prefix+'Card1'))
    amount, rows = tracker.advance(scene(4, prefix+'2'), terminal=2)
    assert amount == event_unit(4) and rows[0]['key'] == 's4/BossCard1'


def test_seija_mirror_helper_is_not_an_extra_card():
    tracker = EventProgress(dict(mode='stage', stage=5))
    tracker.begin(scene(5, 'BossCard3'))
    helper = scene(5)
    helper.extra['thrl_events']['ecl_roles']['main_bosses'] = [dict(ecl_sub_name='MBossCard3_at')]
    assert tracker.advance(helper) == (0., [])
    amount, rows = tracker.advance(scene(5, 'BossCard4'))
    assert amount == event_unit(5) and rows[0]['key'] == 's5/BossCard3'


def accounting_env(rewards, *, baseline=0.):
    return NS(objective=Objective(progress_revision='native-events-v1', progress_weight=.2),
        _task_returns=np.array(rewards, float), _episode_returns=np.zeros(3),
        _episode_raw_returns=np.zeros(3), _event_reward_activation_progress=baseline)


def test_old_checkpoint_replay_does_not_award_historical_events_and_second_resume_is_exact():
    from fly_touhou.nbnm_continuation import restore_reward_accounting
    old = dict(task_returns=np.array([0., 0., 7.2]))
    env = accounting_env([0., 0., .14])
    receipt = restore_reward_accounting(env, old)
    assert receipt['old_HP_return'] == 7.2 and not receipt['historical_events_rewarded']
    assert env._task_returns[2] == 0 and env._event_reward_activation_progress == .14
    saved = dict(task_returns=np.array([0., 0., .03]), reward_progress_revision='native-events-v1',
        event_reward_activation_progress=.14)
    second = accounting_env([0., 0., .17])
    assert restore_reward_accounting(second, saved) is None
    np.testing.assert_allclose(second._task_returns, saved['task_returns'], atol=1e-12)


def specifications():
    from fly_touhou.nbnm_resume import with_pending_objective
    spec = dict(tasks={'full_story': dict(mode='full_story')}, world_groups=['full_story'],
        segment_decisions=8192, nbnm=dict(objective=dict(phase='completion', speed_weight=.01),
        risk_mode='route'), task_entries={})
    patch = dict(schema='nbnm-next-resume-objective-v1', objective=dict(progress_revision='native-events-v1', progress_weight=.2))
    return spec, with_pending_objective(spec, patch), patch


def test_migration_defers_old_ppo_tape_and_preserves_completed_learning():
    from fly_touhou.nbnm_resume import objective_for_saved_rollout, validate_resume
    old, new, _ = specifications()
    saved = dict(spec=old, in_progress=dict(rollout='old-real-rollout.pt', logs=[{'update':1}]))
    active, deferred = objective_for_saved_rollout(new, saved)
    assert deferred and active['nbnm']['objective'] == old['nbnm']['objective']
    assert validate_resume(saved, active) == dict(route_prior_added=False)
    with pytest.raises(ValueError, match='committed segment'):
        validate_resume(saved, new)
    saved['in_progress'] = None
    active, deferred = objective_for_saved_rollout(new, saved)
    assert not deferred and validate_resume(saved, active)['event_progress_added']
    assert validate_resume(dict(spec=new, in_progress=None), new) == dict(route_prior_added=False)


def test_pending_patch_does_not_change_active_source_configuration_until_activation(tmp_path):
    import json
    from fly_touhou.nbnm_resume import pending_objective, with_pending_objective, record_objective_activation
    old, new, patch = specifications()
    path = tmp_path/'route.json'
    path.write_text(json.dumps(old), encoding='utf-8')
    path.with_suffix('.next-resume.json').write_text(json.dumps(patch), encoding='utf-8')
    loaded = with_pending_objective(old, pending_objective(path))
    assert loaded == new and json.loads(path.read_text()) == old
    (tmp_path/'run').mkdir()
    record_objective_activation(path, loaded, patch, tmp_path/'run', 212992, 832, {})
    assert json.loads(path.read_text()) == new


def test_only_third_critic_output_and_its_Adam_moments_are_reset():
    import torch
    from fly_touhou.nbnm_resume import reset_event_critic
    m = torch.nn.Module()
    m.critic = torch.nn.Sequential(torch.nn.Linear(5, 7), torch.nn.Tanh(), torch.nn.Linear(7, 3))
    m.policy = torch.nn.Linear(5, 18)
    optimizer = torch.optim.Adam(m.parameters(), lr=.001)
    x = torch.ones(4, 5)
    (m.critic(x).square().sum()+m.policy(x).square().sum()).backward();optimizer.step()
    weights = deepcopy(m.state_dict());moments = deepcopy(optimizer.state_dict())
    values = m.critic(x).detach();policy = m.policy(x).detach()
    reset_event_critic(m, optimizer)
    torch.testing.assert_close(m.critic(x)[:,:2], values[:,:2], rtol=0, atol=0)
    torch.testing.assert_close(m.policy(x), policy, rtol=0, atol=0)
    assert torch.count_nonzero(m.critic(x)[:,2]) == 0
    for key, value in m.state_dict().items():
        if key.startswith('critic.2.'):
            torch.testing.assert_close(value[:2], weights[key][:2], rtol=0, atol=0)
        else:
            torch.testing.assert_close(value, weights[key], rtol=0, atol=0)
    current = optimizer.state_dict()
    for index, state in current['state'].items():
        torch.testing.assert_close(state['step'], moments['state'][index]['step'], rtol=0, atol=0)


def test_production_three_independent_critics_keep_other_weights_and_optimizer_state():
    import torch
    from fly_touhou.fly_gru_aligned_model import ThreeDNValueCritics
    from fly_touhou.nbnm_resume import reset_event_critic
    model = torch.nn.Module()
    model.critic = ThreeDNValueCritics(5)
    model.policy = torch.nn.Linear(5, 18)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    x = torch.randn(4, 5)
    (model.critic(x).square().sum()+model.policy(x).square().sum()).backward()
    optimizer.step()
    weights = deepcopy(model.state_dict())
    moments = {name:deepcopy(optimizer.state[p]) for name,p in model.named_parameters()}
    values = model.critic(x).detach()
    report = reset_event_critic(model, optimizer)
    assert report['critic_layout'] == 'three-independent-DN-critics'
    torch.testing.assert_close(model.critic(x)[:,:2], values[:,:2], rtol=0, atol=0)
    assert torch.count_nonzero(model.critic(x)[:,2]) == 0
    for name,p in model.named_parameters():
        changed = name.startswith('critic.heads.2.')
        torch.testing.assert_close(p, torch.zeros_like(p) if changed else weights[name], rtol=0, atol=0)
        for key,value in optimizer.state[p].items():
            expected = torch.zeros_like(value) if changed and value.shape==p.shape else moments[name][key]
            torch.testing.assert_close(value, expected, rtol=0, atol=0)


def test_event_advantage_actually_prefers_further_progress_when_success_is_equal():
    import torch
    from dataclasses import replace
    from fly_touhou.compact_story_ready_data import prepare_ready_targets
    from fly_touhou.fly_gru_aligned_ppo import aligned_ppo_config
    cfg = replace(aligned_ppo_config(), gamma=1., objective_weights=(1.,0.,.2))
    data = dict(valid_mask=torch.ones(1,2,dtype=torch.bool), old_values=torch.zeros(1,2,3),
        reward_vectors=torch.tensor([[[-1.,0.,0.],[-1.,0.,.1]]]), bootstrap_values=torch.zeros(2,3),
        terminated=torch.ones(1,2,dtype=torch.bool), episode_starts=torch.ones(1,2,dtype=torch.bool),
        objective_weights=torch.tensor([[[1.,0.,.2],[1.,0.,.2]]]))
    _, _, advantages = prepare_ready_targets(data, cfg, 'cpu')
    assert advantages[0,0] < 0 < advantages[0,1]


def test_legacy_objective_is_unchanged_and_event_objective_ignores_hp():
    old = Objective()
    assert old.weights == (1.,0.,0.)
    assert old.rewards(terminal=False,success=False,native_frames=100,hp_progress=.3)[2] == pytest.approx(.3)
    new = Objective(progress_revision='native-events-v1', progress_weight=.2)
    a = new.rewards(terminal=False,success=False,native_frames=100,hp_progress=100.,event_progress=.01)
    b = new.rewards(terminal=False,success=False,native_frames=100,hp_progress=0.,event_progress=.01)
    np.testing.assert_array_equal(a, b)
