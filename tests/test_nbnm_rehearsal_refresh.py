import json
from pathlib import Path

from fly_touhou.nbnm_rehearsal import SCHEMA,SuccessRehearsal


def test_refresh_admits_only_completed_training_successes_and_keeps_prefix_offset(tmp_path):
    task=dict(mode='stage',stage=1)
    row=dict(file='old.npz',task=task,split='train',nbnm_success=True,decisions=32)
    base=tmp_path/'base.json';source=tmp_path/'growing.json'
    base.write_text(json.dumps(dict(schema=SCHEMA,histories=[row])))
    replay=SuccessRehearsal(base,additional_manifests=[source])
    assert len(replay.rows)==1
    fresh=dict(row,file='new.npz',training_start=100)
    source.write_text(json.dumps(dict(schema=SCHEMA,histories=[fresh,
        dict(row,file='failed.npz',nbnm_success=False),dict(row,file='test.npz',split='test'),
        dict(row,file='short.npz',decisions=2)])))
    assert len(replay.rows)==1
    replay.refresh()
    assert len(replay.rows)==2 and replay.rows[1]['training_start']==100
    assert all(Path(r['file']).is_absolute() for r in replay.rows)


def test_challenging_rehearsal_never_trains_the_student_takeover_prefix(tmp_path):
    import numpy as np
    row=dict(file='history.npz',task=dict(mode='attack',spell_id=1),split='train',nbnm_success=True,decisions=100,training_start=80)
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(dict(schema=SCHEMA,histories=[row])))
    replay=SuccessRehearsal(manifest,challenging_fraction=1.)
    features=np.full((180,281),-1.,np.float32);features[:,72:75]=1.
    features[125,72:75]=0
    for seed in range(40):
        start,challenging=replay.window_start(features,row,np.random.default_rng(seed))
        assert challenging and start>=80 and start<=125<start+16


def test_phase_priority_windows_align_shorter_labels_without_learning_prefix(tmp_path):
    import numpy as np
    row=dict(file='history.npz',task=dict(mode='stage',stage=2),split='train',nbnm_success=True,decisions=100,training_start=100)
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(dict(schema=SCHEMA,histories=[row])))
    replay=SuccessRehearsal(manifest,phase_priorities={'s2/BossCard2':3.},priority_fraction=1.)
    features=np.zeros((200,281),np.float32)
    phases=np.array(['s2/road']*100,dtype='U30');phases[20:40]='s2/BossCard2'
    for seed in range(50):
        start,_=replay.window_start(features,row,np.random.default_rng(seed),phases)
        assert replay.priority_window and start>=100
        assert start<140 and start+16>120


def test_phase_priority_preserves_task_scope_and_spell_retention_budget(tmp_path):
    import numpy as np
    import pytest
    tasks=[dict(mode='attack',stage=2,attack='BossCard1'),dict(mode='attack',stage=2,attack='BossCard2'),
        dict(mode='attack',stage=2,attack='Boss1'),dict(mode='stage',stage=2)]
    rows=[dict(file=f'{i}.npz',task=t,split='train',nbnm_success=True,decisions=100) for i,t in enumerate(tasks)]
    path=tmp_path/'m.json';path.write_text(json.dumps(dict(schema=SCHEMA,histories=rows)))
    replay=SuccessRehearsal(path,scope_weights={'attack':.6,'stage':.4},attack_spell_fraction=.75,
        phase_priorities={'s2/BossCard2':3.})
    p=replay.group_probabilities(list(replay.groups.values()))
    assert p[3]==pytest.approx(.4) and p[2]==pytest.approx(.15)
    assert p[:2].sum()==pytest.approx(.45) and p[1]/p[0]==pytest.approx(3.)


def test_teacher_correction_admits_only_complete_success_and_samples_its_actual_escape(tmp_path):
    import numpy as np
    task=dict(mode='stage',stage=1)
    good=dict(file='verified.npz',task=task,split='train',nbnm_success=True,decisions=100,
        training_start=348,source_kind='native_verified_teacher_continuation',
        correction_windows=[[348,372]],native_validation=dict(stage_clear=True,contacts=0,Bombs=0))
    bad=dict(good,file='failed.npz',native_validation=dict(stage_clear=False,contacts=1,Bombs=0))
    manifest=tmp_path/'proof.json';manifest.write_text(json.dumps(dict(schema=SCHEMA,histories=[good,bad])))
    replay=SuccessRehearsal(manifest)
    assert len(replay.rows)==1
    for i in range(20):
        start,_=replay.window_start(np.zeros((448,281)),good,np.random.default_rng(i))
        assert 348<=start<=356 and start+16<=372


def test_verified_road_requires_observed_endpoint_and_keeps_full_prefix(tmp_path):
    import numpy as np
    from fly_touhou.nbnm_rehearsal import verified_correction
    row=dict(file='road.npz',task=dict(mode='route',stage=4,end_event='first-native-midboss'),
        split='train',nbnm_success=True,decisions=1267,training_start=0,
        source_kind='native_verified_teacher_continuation',correction_windows=[[180,500]],
        native_validation=dict(contacts=0,Bombs=0,stage_clear=False,task_complete=True,
            goal='first-native-midboss',final_phase='s4/MBossB1'))
    assert verified_correction(row)
    for change in (dict(task_complete=False),dict(final_phase='s4/road'),
                   dict(contacts=1),dict(goal='4000-frames')):
        assert not verified_correction(dict(row,native_validation=dict(row['native_validation'],**change)))
    path=tmp_path/'road.json';path.write_text(json.dumps(dict(schema=SCHEMA,histories=[row])))
    replay=SuccessRehearsal(path,native_correction=dict(manifest=str(path),multiplier=2.))
    for seed in range(20):
        first,_=replay.correction.window_start(np.zeros((1267,281)),row,np.random.default_rng(seed))
        assert 180<=first and first+16<=500
    assert replay.correction.rows[0]['training_start']==0


def test_verified_attack_entry_matches_the_declared_route():
    from fly_touhou.nbnm_rehearsal import verified_correction

    task = dict(mode='route', stage=4, end_event='first-native-attack-entry', end_attack='BossCard1')
    proof = dict(native_seed=60036, task=task, contacts=0, Bombs=0, stage_clear=False,
                 task_complete=True, goal=task['end_event'], final_phase='s4/BossCard1')
    row = dict(task=task, native_seed=60036, source_kind='native_verified_teacher_continuation',
               native_validation=proof)
    assert verified_correction(row)
    assert not verified_correction(dict(row, native_seed=60047))
    assert not verified_correction(dict(row, native_validation=dict(proof, final_phase='s4/Boss1')))
    assert not verified_correction(dict(row, native_validation=dict(proof, contacts=1)))
