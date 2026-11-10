"""Matched starts and extending real command tapes across variable loading."""
from pathlib import Path
import copy
import importlib.util


def script(name):
    path=Path(__file__).resolve().parents[1]/'scripts'/(name+'.py')
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_continued_prefix_rebases_loading_ticks_without_mutating_original():
    module=script('prepare_nbnm_start_bank')
    original=dict(transitions=[dict(before=dict(ticks=100),after=dict(ticks=120)),
                              dict(before=dict(ticks=300),after=dict(ticks=330))],
                  status='nbnm_warmup',ticks=500,raw_input_segments=[[1,500]])
    untouched=copy.deepcopy(original)
    receipt=dict(barriers=[dict(recorded_ticks=20,actual_ticks=24),
                           dict(recorded_ticks=30,actual_ticks=28)])
    result=module.continued_record(original,receipt,758)
    assert original==untouched
    assert result['transitions']==[dict(before=dict(ticks=100),after=dict(ticks=124)),
                                  dict(before=dict(ticks=304),after=dict(ticks=332))]
    assert result['ticks']==758 and result['status']=='recording'


def reports(before,after):
    def report(values):
        rows=[dict(native_seed=i,nbnm_success=success,task_native_frames=100+i,
                   end_reason='success' if success else 'native_contact') for i,success in enumerate(values)]
        return dict(task=dict(mode='attack',stage=3,attack='BossCard2'),frozen=True,
                    action_selection='joint_mode',opening_seeds=list(range(len(values))),
                    successes=sum(values),episode_results=rows)
    return report(before),report(after)


def test_paired_comparison_requires_matching_inputs_and_reports_regressions():
    compare=script('compare_nbnm_frozen').comparison
    a,b=reports([False]*8,[True]*8)
    entries={str(i):dict(features=[i],policy_burn_in=None) for i in range(8)}
    result=compare(a,b,entries,entries)
    assert result['improved_pairs']==8 and result['regressed_pairs']==0
    assert result['improvement_supported'] and result['exact_paired_success_p']<.05
    changed=copy.deepcopy(entries);changed['0']['features'][0]=.1
    assert not compare(a,b,entries,changed)['improvement_supported']
    a,b=reports([True,False],[False,True])
    result=compare(a,b,entries,entries)
    assert result['regressed_pairs']==result['improved_pairs']==1
    assert not result['improvement_supported']


def test_reference_stays_original_after_pilot_warmstart():
    choose=script('compare_nbnm_frozen').reference_checkpoint
    spec=dict(initial_checkpoint='pilot.pt',reference_checkpoint='original.pt')
    assert choose(spec)==Path('original.pt')
    assert choose(spec,'explicit.pt')==Path('explicit.pt')
    assert choose(dict(initial_checkpoint='pilot.pt'))==Path('pilot.pt')


def test_paired_report_labels_test_split():
    compare=script('compare_nbnm_frozen').comparison
    a,b=reports([False],[True])
    a['evaluation_split']=b['evaluation_split']='test'
    entries={'0':dict(features=[0],policy_burn_in=None)}
    result=compare(a,b,entries,entries)
    assert result['evaluation_split']=='test'
    assert 'Matched test starts' in result['scope']


def test_recording_appends_acknowledged_native_replay_commands():
    from types import SimpleNamespace as NS
    from fly_touhou.frida_controller import OriginalGameController
    posted=[]
    controller=OriginalGameController.__new__(OriginalGameController)
    controller._script=NS(post=posted.append);controller._checkpoint={}
    controller._hook_ready={'native_tape':{'schema':1}}
    controller._memory=NS();controller._record_input_tape=True
    controller._input_tape=[(1,4)]
    controller._wait=lambda name:dict(sequence=2)
    controller.read=lambda:'native snapshot'
    assert controller.advance_tape([(9,4),(17,8)])=='native snapshot'
    assert controller._input_tape==[(1,4),(9,4),(17,8)]
    assert posted[0]['payload']['tape']==[[9,4],[17,8]]


def test_frozen_evaluation_accepts_readonly_vector_snapshots(monkeypatch,tmp_path):
    import numpy as np
    from fly_touhou import nbnm_run,thrl_vector,compact_full_story
    from fly_touhou.nbnm_model import configure_model
    from test_nbnm import model_and_observations
    model,features=model_and_observations()
    configure_model(model,dict(objective={},risk_mode='route'),reset_critics=False)
    initial=np.concatenate((features[:1].numpy(),np.zeros((1,152),np.float32)),axis=1)
    initial.flags.writeable=False
    class Vector:
        _inflight=False
        def __init__(self,*args,**kwargs):pass
        def reset(self,**kwargs):return dict(features=initial),[dict(native_seed=6)]
        def submit_at(self,*args,**kwargs):self._inflight=True
        def receive_ready(self,**kwargs):
            self._inflight=False
            summary=dict(native_seed=6,nbnm_success=True,task_native_frames=4,end_reason='attack_complete')
            return [dict(index=0,operation='step',observation=dict(features=initial[0]),
                info=dict(episode_summary=summary),terminated=True,truncated=False)]
        def close(self):pass
    monkeypatch.setattr(thrl_vector,'THRLNativeVector',Vector)
    monkeypatch.setattr(compact_full_story,'story_kwargs',lambda *a,**k:[])
    monkeypatch.setattr(nbnm_run,'environment',lambda *a,**k:dict(nbnm={}))
    monkeypatch.setattr(nbnm_run,'evaluation_seeds',lambda *a,**k:[6])
    task=dict(mode='attack',stage=3,attack='BossCard1')
    report=nbnm_run.evaluate(model,dict(nbnm=dict(task=task)),tmp_path/'frozen')
    assert report['attempts']==report['successes']==1
    assert not initial.flags.writeable
