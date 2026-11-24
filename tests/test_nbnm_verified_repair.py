from train_nbnm_verified_repair import rehearsal_requests, replay_index


def test_verified_repair_does_not_starve_stage1_or_stage4_when_two_groups_exist():
    selected=[replay_index(14271,i,'correction')%2 for i in range(32)]
    assert selected.count(0)==selected.count(1)==16


def test_verified_repair_resume_uses_the_same_history_indices_as_uninterrupted_run():
    full=[rehearsal_requests(14271,i,3,True) for i in range(32)]
    resumed=[rehearsal_requests(14271,i,3,True) for i in range(13,32)]
    assert resumed==full[13:]
    assert full[12]==(('correction',replay_index(14271,12,'correction'),3),
                      ('early_retention',replay_index(14271,6,'early_retention'),1))
    assert full[13]==(('correction',replay_index(14271,13,'correction'),3),
                      ('retention',replay_index(14271,13,'retention'),1))


def test_reference_every_update_keeps_correction_order_and_uses_the_same_reference_windows():
    normal=[rehearsal_requests(14271,i,3,True) for i in range(20)]
    reference=[rehearsal_requests(14271,i,3,True,reference_every_update=True) for i in range(20)]
    assert [pair[0] for pair in reference]==[pair[0] for pair in normal]
    for i in range(0,20,2):
        assert reference[i][1]==reference[i+1][1]==normal[i][1]


def test_readout_repair_preserves_gradients_only_for_choice_and_exit_witness():
    import torch
    from train_nbnm_verified_repair import freeze_shared_representation

    model = torch.nn.Module()
    for name in ('core', 'risk', 'route', 'corridor_head', 'corridor_witness_head'):
        setattr(model, name, torch.nn.Linear(4, 4))
    freeze_shared_representation(model)
    trainable = {key.split('.')[0] for key,value in model.named_parameters() if value.requires_grad}
    assert trainable == {'route', 'corridor_witness_head'}
    freeze_shared_representation(model, train_witness=False)
    trainable = {key.split('.')[0] for key,value in model.named_parameters() if value.requires_grad}
    assert trainable == {'route'}


def test_reference_rehearsal_matches_score_differences(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    import numpy as np
    import torch
    from fly_touhou import compact_story_model
    from fly_touhou.nbnm_rehearsal import SCHEMA, SuccessRehearsal
    from fly_touhou.spiking_engineered_control import SpikingState

    reference = np.tile(np.arange(18, dtype=np.float32), (16, 1))
    np.savez(tmp_path / 'success.npz', features=np.zeros((16, 407), np.float32),
             actions=np.full(16, 17), reference_logits=reference)
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(dict(schema=SCHEMA, histories=[dict(
        file='success.npz', task=dict(mode='stage', stage=4), split='train',
        decisions=16, nbnm_success=True)])), encoding='utf-8')
    rehearsal = SuccessRehearsal(manifest, match_reference_scores=True)
    scores = torch.nn.Parameter(torch.from_numpy(reference) + 2.)

    def sequence(*args, **kwargs):
        return {'logits': torch.cat((scores, scores.new_full((16, 1), -torch.inf)), -1)[:, None]}

    monkeypatch.setattr(compact_story_model, 'sequence', sequence)
    state = SpikingState(*(torch.zeros((1, 2)) for _ in range(4)))
    model = SimpleNamespace(device=torch.device('cpu'), initial_state=lambda count: state)
    loss, info = rehearsal.loss(model, index=0)
    assert loss.item() == 0 and info['rehearsal_matches_reference_scores']

    with torch.no_grad():
        scores[:, 4] += 1.
    loss, _ = rehearsal.loss(model, index=0)
    loss.backward()
    assert loss.item() > 0 and torch.all(scores.grad[:, 4] > 0)
    torch.testing.assert_close(scores.grad.sum(-1), torch.zeros(16), rtol=0, atol=1e-7)


def test_observed_branch_gradient_is_local_to_the_recorded_pair(monkeypatch):
    from types import SimpleNamespace
    import numpy as np
    import torch
    from fly_touhou import compact_story_model
    from fly_touhou.nbnm_rehearsal import SuccessRehearsal

    scores=torch.zeros((16,1,19),requires_grad=True)
    monkeypatch.setattr(compact_story_model,'sequence',lambda *args,**kwargs:{'logits':scores})
    features=np.zeros((16,407),np.float32)
    actions=np.full(16,3,np.int64)
    row=dict(task=dict(mode='stage',stage=4),source_kind='native_verified_teacher_continuation',
             native_validation=dict(stage_clear=True,contacts=0,Bombs=0),
             native_preference=dict(history_index=2,preferred_action=3,comparison_action=2))
    model=SimpleNamespace(device=torch.device('cpu'))
    common=dict(length=16,native_short_risk_coefficient=0.,native_exit_coefficient=0.)
    args=(model,features,actions,row,0,None,None,None,{},[],False,True)
    ordinary,_=SuccessRehearsal._sequence_loss(SimpleNamespace(**common,prefer_observed_branches=False),*args)
    pair,details=SuccessRehearsal._sequence_loss(SimpleNamespace(**common,prefer_observed_branches=True),*args)
    delta=torch.autograd.grad(pair-ordinary,scores)[0]
    expected=torch.zeros_like(scores);expected[2,0,3]=-.5;expected[2,0,2]=.5
    assert details['rehearsal_observed_branch_rows']==1
    torch.testing.assert_close(delta,expected,rtol=0,atol=1e-7)


def test_native_student_road_and_validated_teacher_full_stage_share_the_stage_pool(tmp_path):
    import json
    from fly_touhou.nbnm_rehearsal import SCHEMA,SuccessRehearsal
    common=dict(split='train',decisions=32,nbnm_success=True,correction_windows=[[0,32]])
    stage=dict(common,file='teacher.npz',task=dict(mode='stage',stage=4),
        source_kind='native_verified_teacher_continuation',
        native_validation=dict(stage_clear=True,contacts=0,Bombs=0))
    road=dict(common,file='student.npz',task=dict(mode='route',stage=4,end_event='first-native-midboss'),
        source_kind='native_verified_student_continuation',native_validation=dict(stage_clear=False,
            contacts=0,Bombs=0,task_complete=True,goal='first-native-midboss',final_phase='s4/MBossB1'))
    path=tmp_path/'manifest.json'
    path.write_text(json.dumps(dict(schema=SCHEMA,histories=[stage,road])),encoding='utf8')
    pool=SuccessRehearsal(path)
    assert len(pool.groups)==2 and pool.correction_stages=={4:[0,1]}


def test_safe_multi_command_labels_require_the_exact_subsequent_greedy_actions_and_native_clock():
    import torch
    from fly_touhou.nbnm_teacher_outcomes import matched_native_futures
    actions=torch.tensor([4,2,5,3,7,0,1,2])
    clock=torch.arange(8)*4
    exposed=torch.ones(8,dtype=torch.bool)
    greedy=actions.clone();greedy[0]=9
    assert matched_native_futures(exposed,clock,actions,greedy,6).tolist()==[True,True,True,False,False,False,False,False]
    greedy[2]=1
    assert matched_native_futures(exposed,clock,actions,greedy,3).tolist()==[False,False,True,True,True,True,False,False]
    clock[4:]+=1
    assert not matched_native_futures(exposed,clock,actions,greedy,3)[2]
    exposed[6]=False
    assert not matched_native_futures(exposed,clock,actions,greedy,3)[5]


def test_native_exit_witness_uses_measured_suffix_beyond_the_training_window():
    import numpy as np
    from fly_touhou.nbnm_teacher_outcomes import verified_exit_mask

    f=np.zeros((40,407),np.float32)
    f[:,14]=1
    record=dict(native_frames=100+4*np.arange(34),native_action_frames=np.full(34,4),
                observed_contacts=np.zeros(34,bool))
    row=dict(training_start=6,split='train',nbnm_success=True,native_rule_geometry='native',
             native_validation=dict(task_complete=True,contacts=0,Bombs=0))
    result=verified_exit_mask(record,row,f,first=4,end=8,geometry='native')
    assert result.tolist()==[[False,False],[False,False],[True,True],[True,True]]

    record['native_frames'][20:]+=1
    result=verified_exit_mask(record,row,f,first=6,end=8,geometry='native')
    assert result.tolist()==[[True,False],[True,False]]
    record['observed_contacts'][5]=True
    assert not verified_exit_mask(record,row,f,6,8,'native').any()
    assert not verified_exit_mask(record,row,f,30,31,'native').any()


def test_follow_through_windows_preserve_the_anchor_and_sample_later_disjoint_intervals():
    import numpy as np
    from fly_touhou.nbnm_rehearsal import cacheable_window_prefixes, continuation_window_starts

    row={'correction_windows': [[8,24],[16,40]]}
    windows=continuation_window_starts(8,16,184,row,3,np.random.default_rng(11))
    assert windows[0]==8
    assert 40<=windows[1]<120<=windows[2]<=168
    assert min(np.diff(windows))>=16
    assert not cacheable_window_prefixes(windows)
    assert cacheable_window_prefixes(windows[1:])
    assert continuation_window_starts(8,16,40,row,3,np.random.default_rng(11))==[8]


def test_verified_hazard_window_cannot_be_skipped_by_uniform_route_sampling():
    import numpy as np
    from fly_touhou.nbnm_rehearsal import continuation_window_starts

    row=dict(training_start=12, correction_windows=[[12,28]], continuation_priority_starts=[42])
    for seed in range(12):
        selected=continuation_window_starts(12,16,2800,row,3,np.random.default_rng(seed))
        assert selected[0]==12 and 42 in selected and len(selected)==3
        assert all(b-a>=16 for a,b in zip(selected,selected[1:]))


def test_correction_sampler_balances_seeds_without_changing_stage_allocation():
    from types import SimpleNamespace

    from fly_touhou.nbnm_rehearsal import SuccessRehearsal

    sampler = SimpleNamespace(
        rows=[
            dict(native_seed=10, history='s2-a'),
            dict(native_seed=10, history='s2-b'),
            dict(native_seed=11, history='s2-c'),
            dict(native_seed=20, history='s4-a'),
        ],
        groups={},
        correction_stages={2: [0, 1, 2], 4: [3]},
        scope_weights={},
        phase_priorities={},
        balance_native_seeds=True,
    )

    def selected():
        return [SuccessRehearsal._select_row(sampler, i, None, 3)[0]['history']
                for i in range(8)]

    assert selected() == ['s2-a', 's4-a', 's2-c', 's4-a', 's2-b', 's4-a', 's2-c', 's4-a']
    sampler.balance_native_seeds = False
    assert selected() == ['s2-a', 's4-a', 's2-b', 's4-a', 's2-c', 's4-a', 's2-a', 's4-a']


def test_observed_branch_onset_is_selected_before_other_correction_windows():
    from types import SimpleNamespace

    import numpy as np
    from fly_touhou.nbnm_rehearsal import SuccessRehearsal

    features = np.zeros((128,407), np.float32)
    row = dict(training_start=20,correction_windows=[[70,98]],native_preference=dict(history_index=20))
    sampler = SimpleNamespace(length=16,prefer_observed_branches=True)
    rng = np.random.default_rng(41)
    assert SuccessRehearsal.window_start(sampler,features,row,rng) == (20,True)
    row['native_preference']['history_index'] = 127
    assert SuccessRehearsal.window_start(sampler,features,row,rng) == (112,True)
    sampler.prefer_observed_branches = False
    assert SuccessRehearsal.window_start(sampler,features,row,rng)[0] >= 70


def test_verified_native_frame_teaches_crossing_only_after_observed_arrival():
    from fly_touhou.nbnm_rehearsal import verified_correction

    task = dict(mode='route', stage=2, end_event='native-frame-arrival', end_native_frame=12600)
    proof = dict(contacts=0, Bombs=0, task_complete=True, native_seed=60006, task=task,
                 goal='native-frame-arrival', observed_stage=2, observed_frame=12600, stage_clear=False)
    row = dict(task=task, native_seed=60006, source_kind='native_verified_teacher_continuation',
               native_validation=proof)
    assert verified_correction(row)
    for change in (dict(observed_frame=12596), dict(observed_stage=3), dict(contacts=1)):
        assert not verified_correction(dict(row, native_validation=dict(proof, **change)))


def test_shared_prefix_saves_the_same_memory_as_independent_prefix_replays():
    import torch
    from fly_touhou.nbnm_prefix import burn_in, FixedPrefixCache, fixed_history_snapshots
    from fly_touhou.spiking_engineered_control import SpikingState

    class RecurrentFixture:
        device=torch.device('cpu')

        def initial_state(self, batch):
            return SpikingState(*(torch.full((batch,3),float(i)) for i in range(4)))

        def candidate_features(self, features):
            return features[:,None]+torch.arange(18)[None,:,None]

        def core(self, cue, state):
            return None, SpikingState(*(value*.95+cue for value in state.tensors()))

    model=RecurrentFixture()
    state=model.initial_state(1)
    features=torch.arange(48*3,dtype=torch.float32).reshape(48,3)/100.
    actions=torch.arange(48)%18
    snapshots={offset: None for offset in (0,12,31,48)}
    replay=lambda n: {'policy_burn_in': dict(features=features[:n],actions=actions[:n])}
    final=burn_in(model,replay(len(actions)),state,snapshots=snapshots)
    for offset, captured in snapshots.items():
        control=burn_in(model,replay(offset),state)
        for observed, expected in zip(captured.tensors(),control.tensors()):
            torch.testing.assert_close(observed,expected,rtol=0,atol=0)
    for captured, last in zip(snapshots[48].tensors(),final.tensors()):
        torch.testing.assert_close(captured,last,rtol=0,atol=0)

    histories={offset:replay(offset)['policy_burn_in'] for offset in snapshots}
    different=actions[:12].clone();different[0]=4
    histories['branch']=dict(features=features[:12],actions=different)
    cached,replayed=fixed_history_snapshots(model,histories)
    assert replayed == 48 + 12
    replay = FixedPrefixCache(model, histories)
    assert replay.planned_steps == replayed and replay.replayed_steps == 0
    for offset in (12,31,48):
        for observed, expected in zip(replay[offset].tensors(), cached[offset].tensors()):
            torch.testing.assert_close(observed,expected,rtol=0,atol=0)
        assert replay.replayed_steps == 48
    _ = replay['branch']
    assert replay.replayed_steps == replayed
    for key, history in histories.items():
        control=burn_in(model,dict(policy_burn_in=history),state)
        for observed, expected in zip(cached[key].tensors(),control.tensors()):
            torch.testing.assert_close(observed,expected,rtol=0,atol=0)


def test_prepared_prefix_roundtrip_and_changed_source_fallback(tmp_path):
    from types import SimpleNamespace

    import torch
    from fly_touhou.nbnm_fixed_readout_cache import identity, read_states
    from fly_touhou.spiking_engineered_control import SpikingState

    files=[tmp_path / name for name in ('source.pt','environment.json','tape.npz')]
    for path in files:
        path.write_bytes(b'original')
    checkpoint,environment,tape=files
    state=SpikingState(*(torch.full((1,3),float(i)) for i in range(4)))
    model=SimpleNamespace(device=torch.device('cpu'),nbnm={},initial_state=lambda batch:state)
    spec={'environment_config':environment}
    key=(str(tape),20)
    histories={key:{}}
    source=identity(model,spec,checkpoint,histories)
    archive=tmp_path / 'archive.pt'
    torch.save(dict(identity=source,states={key:state.tensors()}),archive)
    restored=read_states(archive,source,model)
    assert all(torch.equal(a,b) for a,b in zip(restored[key].tensors(),state.tensors(),strict=True))

    second_tape=tmp_path/'another-tape.npz'
    second_tape.write_bytes(b'another original replay')
    expanded={key:{},(str(second_tape),12):{}}
    new_source=identity(model,spec,checkpoint,expanded)
    assert read_states(archive,new_source,model) is None
    partial=read_states(archive,new_source,model,allow_subset=True)
    assert set(partial)=={key}
    assert all(torch.equal(a,b) for a,b in zip(partial[key].tensors(),state.tensors(),strict=True))

    wider_archive=tmp_path/'wider.pt'
    torch.save(dict(identity=new_source,states={window:state.tensors() for window in expanded}),wider_archive)
    subset=read_states(wider_archive,source,model,allow_subset=True)
    assert set(subset)==set(histories)
    assert all(torch.equal(a,b) for a,b in zip(subset[key].tensors(),state.tensors(),strict=True))

    tape.write_bytes(b'changed archive')
    assert read_states(archive,identity(model,spec,checkpoint,histories),model) is None
    assert read_states(archive,identity(model,spec,checkpoint,expanded),model,allow_subset=True)=={}
    checkpoint.write_bytes(b'changed model')
    assert read_states(archive,identity(model,spec,checkpoint,expanded),model,allow_subset=True) is None

def test_movement_rehearsal_reuses_brain_activity_with_current_readout_weights(tmp_path, monkeypatch):
    import json

    import numpy as np
    import torch
    from fly_touhou.nbnm_model import configure_model
    from fly_touhou.nbnm_rehearsal import SCHEMA, SuccessRehearsal
    from test_compact_story_ready import build
    from train_nbnm_verified_repair import freeze_shared_representation

    collector, model = build(tmp_path)
    configure_model(model, {'risk_mode': 'route'}, reset_critics=False)
    freeze_shared_representation(model, train_witness=False)
    model.train()
    features = np.pad(collector.obs['features'][0], (0, 152))[None].repeat(16, axis=0)
    features[:, 14] = 1.
    actions = np.arange(16, dtype=np.int32) % 18
    np.savez(tmp_path / 'measured.npz', features=features, actions=actions)
    manifest = tmp_path / 'movements.json'
    manifest.write_text(json.dumps(dict(schema=SCHEMA, histories=[dict(
        file='measured.npz', task=dict(mode='stage', stage=2), split='train',
        decisions=16, nbnm_success=True,
    )])), encoding='utf-8')
    pool = SuccessRehearsal(manifest)
    activity = {}
    _, created = next(pool.iter_losses(model, index=10, movement_activity=activity))
    assert not created['rehearsal_reused_movement_activity']
    assert len(activity) == 1

    with torch.no_grad():
        model.route[-1].weight.add_(.01)
    expected, _ = next(pool.iter_losses(model, index=10))
    expected.backward()
    gradients = [parameter.grad.detach().clone() for parameter in model.route.parameters()]
    model.zero_grad(set_to_none=True)

    def unexpected_replay(*args, **kwargs):
        raise AssertionError('Already cached movement windows must skip the full-brain step')

    monkeypatch.setattr(model, 'step', unexpected_replay)
    cached, details = next(pool.iter_losses(model, index=10, movement_activity=activity))
    cached.backward()
    assert details['rehearsal_reused_movement_activity']
    torch.testing.assert_close(cached, expected, rtol=0, atol=0)
    for before, parameter in zip(gradients, model.route.parameters(), strict=True):
        torch.testing.assert_close(before, parameter.grad, rtol=0, atol=0)
