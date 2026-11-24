import importlib.util
import json
from pathlib import Path


def test_teacher_new_practice_seeds_do_not_require_nonexistent_prefixes(tmp_path):
    path=Path(__file__).resolve().parents[1]/'scripts/collect_nbnm_teacher_routes.py'
    module_spec=importlib.util.spec_from_file_location('teacher_collect',path)
    module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
    task=dict(mode='attack',stage=2,attack='Boss1')
    bank=tmp_path/'starts.json';bank.write_text(json.dumps(dict(entries=[
        dict(task=task,split='train',native_seed=s) for s in [50000,50001]])))
    spec=dict(tasks=dict(prefix=task,story=dict(mode='full_story')),nbnm=dict(entry_mode='story_prefix'),
        task_entries=dict(prefix=dict(entry_mode='native_stage_prefix',start_bank=str(bank))))
    jobs,plan=module.teacher_jobs(spec,['prefix','story'],[50002,50003])
    assert jobs==[('prefix',50000),('story',50002),('prefix',50001),('story',50003)]
    assert plan['prefix']['prefix_seeds']==[50000,50001]
