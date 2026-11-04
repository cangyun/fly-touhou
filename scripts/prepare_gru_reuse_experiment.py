"""Seal the four-pass GRU experiment without altering any live v1 source."""
from pathlib import Path
from copy import deepcopy
import json
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.experiment import file_sha256
from fly_touhou.native_diagnostic import write_json

BASE=Path('./outputs/legacy-build/source_snapshots/20261019-diagnostic-v1')
OLD_RUN=Path('./outputs/experiments/native-list-diagnostic-v1')
NEW_RUN=OLD_RUN.with_name('native-list-gru-reuse4-v1')
NEW_SOURCE=Path('./outputs/source_snapshots/20261019-gru-reuse4-v1')
PYTHON=ROOT/'.venv/Scripts/python.exe'


def replace_once(text,before,after):
    if text.count(before)!=1:
        raise ValueError(f'Frozen source edit anchor ambiguous: {before[:90]}')
    return text.replace(before,after,1)


def main():
    if NEW_SOURCE.exists() or NEW_RUN.exists():
        raise FileExistsError('Preserve every prepared experiment')
    base_path=ROOT/'artifacts/diagnostic_protocol/native-list-v1.json'
    assert file_sha256(base_path)=='f91d71d8507c2b484514336aa7b8100fa6c07a44fcad61db74396e05590aa61c'
    original=json.loads(base_path.read_text(encoding='utf-8'))
    source_manifest=json.loads((BASE/'snapshot_manifest.json').read_text(encoding='utf-8'))
    NEW_SOURCE.mkdir(parents=True)
    for item in source_manifest['files']:
        relative=Path(item['path'])
        assert file_sha256(BASE/relative)==item['sha256']
        destination=NEW_SOURCE/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(BASE/relative,destination)
    for relative in ('src/fly_touhou/diagnostic_gru_reuse.py',
                     'tests/test_diagnostic_gru.py','tests/test_diagnostic_gru_reuse.py',
                     'scripts/prepare_gru_reuse_experiment.py','scripts/run_gru_reuse_transition.py'):
        destination=NEW_SOURCE/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/relative,destination)
    runner=(NEW_SOURCE/'scripts/run_native_diagnostic.py').read_text(encoding='utf-8')
    runner=replace_once(runner,
        'from fly_touhou.diagnostic_gru import DiagnosticGRU, cpu_copy, fit_actor',
        'from fly_touhou.diagnostic_gru import DiagnosticGRU, cpu_copy\n'
        'from fly_touhou.diagnostic_gru_reuse import fit_actor_reuse as fit_actor')
    runner=runner.replace('ordinary-GRU-diagnostic-learner-v1','ordinary-GRU-diagnostic-learner-v2-reuse')
    runner=replace_once(runner,
        "protocol, encoder = load_protocol(args.protocol)",
        "protocol, encoder = load_protocol(args.protocol)\n"
        "    assert protocol['actor_proposals_per_batch'] == 4\n"
        "    assert protocol['learning_experiment'] == 'native-list-gru-reuse4-v1'\n"
        "    if args.mode == 'train' and args.checkpoint is not None:\n"
        "        raise ValueError('This controlled branch starts at its saved seed initialization')")
    runner=replace_once(runner,
        "'training_native_frames': 0, 'accepted_actor_updates': 0, 'critic_updates': 0}",
        "'training_native_frames': 0, 'accepted_actor_updates': 0, 'critic_updates': 0,\n"
        "        'actor_optimizer_steps': 0}")
    runner=replace_once(runner,
        "counters['accepted_actor_updates'] += int(metric['accepted'])",
        "counters['accepted_actor_updates'] += int(metric['accepted'])\n"
        "                counters['actor_optimizer_steps'] += metric['accepted_optimizer_steps']")
    runner=replace_once(runner,
        "'gru_code': SOURCE/'src/fly_touhou/diagnostic_gru.py',",
        "'gru_code': SOURCE/'src/fly_touhou/diagnostic_gru.py',\n"
        "        'gru_reuse_code': SOURCE/'src/fly_touhou/diagnostic_gru_reuse.py',")
    runner=replace_once(runner,
        "'complete_fly_actor': False, 'control_only': True,",
        "'complete_fly_actor': False, 'control_only': True,\n"
        "        'learning_experiment': 'native-list-gru-reuse4-v1',\n"
        "        'actor_optimizer_passes_per_rollout': 4,")
    (NEW_SOURCE/'scripts/run_native_diagnostic.py').write_text(runner,encoding='utf-8')

    audit=(ROOT/'scripts/audit_diagnostic_learning.py').read_text(encoding='utf-8')
    first=audit.index("        assert len(proposal['proposal_logp18'])")
    last=audit.index("        changed =",first)
    block='''        assert len(proposal['epochs']) == len(metric['epochs']) == metric['epochs_attempted']
        accepted_steps = 0
        previous_lp = data['old_logp18']
        for saved, epoch in zip(proposal['epochs'], metric['epochs']):
            cpu_lp, cpu_states = replay_history(model, data['observations'], data['episode_ids'])
            assert float((cpu_lp-previous_lp).abs().max()) < 2e-6
            assert float((cpu_states-saved['states_before_epoch']).abs().max()) < 2e-6
            before = recorded_policy_objective(previous_lp, *arguments)
            assert before == epoch['before']
            assert len(saved['proposal_logp18']) == len(epoch['trials'])
            assert epoch['coverage']['samples'] == len(rows)
            for lp, trial in zip(saved['proposal_logp18'], epoch['trials']):
                objective = recorded_policy_objective(lp, *arguments)
                divergence = (data['old_logp18'].exp()*(data['old_logp18']-lp)).sum(-1)
                assert objective == trial['objective']
                assert float(divergence.mean()) == trial['mean_KL_from_behavior']
                assert float(divergence.max()) == trial['maximum_KL_from_behavior']
                valid = (-1e-9 <= trial['mean_KL_from_behavior'] <= .01 and
                    objective['total'] < before['total']-1e-8*max(1,abs(before['total'])))
                assert trial['qualifies'] == valid
            if epoch['accepted']:
                assert epoch['trials'][-1]['qualifies']
                assert not any(t['qualifies'] for t in epoch['trials'][:-1])
                assert torch.equal(saved['accepted_logp18'],saved['proposal_logp18'][-1])
                accepted_steps += 1
            else:
                assert not any(t['qualifies'] for t in epoch['trials'])
                assert torch.equal(saved['accepted_logp18'],previous_lp)
                assert equal(saved['actor_after_epoch'],model.state_dict())
            model.load_state_dict(saved['actor_after_epoch'])
            final_lp, _ = replay_history(model, data['observations'], data['episode_ids'])
            assert float((final_lp-saved['accepted_logp18']).abs().max()) < 2e-6
            assert recorded_policy_objective(saved['accepted_logp18'],*arguments)==epoch['after']
            previous_lp=saved['accepted_logp18']
        assert accepted_steps==metric['accepted_optimizer_steps']
        assert metric['accepted']==(accepted_steps>0)
        assert equal(model.state_dict(),current['actor'])
        assert torch.equal(previous_lp,proposal['accepted_logp18'])
        assert recorded_policy_objective(previous_lp, *arguments) == metric['after']
'''
    audit=audit[:first]+block+audit[last:]
    audit=replace_once(audit,
        "assert all(int(v['step']) == c['accepted_actor_updates'] for v in current['actor_optimizer']['state'].values())",
        "assert c['actor_optimizer_steps'] == p['actor_optimizer_steps']+accepted_steps\n"
        "        assert all(int(v['step']) == c['actor_optimizer_steps'] for v in current['actor_optimizer']['state'].values())")
    audit=audit.replace('independent-GRU-diagnostic-learner-readback-v1',
                        'independent-GRU-diagnostic-reuse-learner-readback-v1')
    (NEW_SOURCE/'scripts/audit_diagnostic_learning.py').write_text(audit,encoding='utf-8')

    suite=(ROOT/'scripts/run_native_diagnostic_suite.py').read_text(encoding='utf-8')
    suite=suite.replace('native-list-gru-seed{seed}-v1','native-list-gru-seed{seed}-reuse4-v1')
    suite=replace_once(suite,
        "Path('./outputs/legacy-build/source_snapshots/20261019-diagnostic-learning-audit-v1/scripts/audit_diagnostic_learning.py')",
        "SOURCE/'scripts/audit_diagnostic_learning.py'")
    suite=suite.replace("'matched full-fly diagnostic' if qualifies", "'GRU gate passed;fly work remains deferred by current user priority' if qualifies")
    (NEW_SOURCE/'scripts/run_native_diagnostic_suite.py').write_text(suite,encoding='utf-8')

    protected=[]
    for item in original['protected_files']:
        relative=Path(item['path']).relative_to(BASE)
        assert file_sha256(NEW_SOURCE/relative)==item['sha256']
        protected.append(str(relative))
    files=[{'path':str(path.relative_to(NEW_SOURCE)), 'bytes':path.stat().st_size,
            'sha256':file_sha256(path)} for path in sorted(NEW_SOURCE.rglob('*')) if path.is_file()]
    write_json(NEW_SOURCE/'snapshot_manifest.json',{
        'schema':'fly-touhou-frozen-source-v1','origin':str(BASE),'files':files,
        'shared_physical_sources_unchanged':protected,'learning_change':'at most4passes instead of1',
        'new_fly_work':False},exclusive=True)
    protocol=deepcopy(original)
    protocol.update(source_snapshot=str(NEW_SOURCE),
        learning_experiment='native-list-gru-reuse4-v1',actor_proposals_per_batch=4,
        proposal_reuse='same behavior anchor,advantages,costs;fresh current hidden history every pass;total behaviorKL<=.01',
        base_protocol={'path':str(base_path),'sha256':file_sha256(base_path)},
        initialization='new isolated experiment,same three initial trainingseeds;preserve all prior learners',
        experiment_selected_without_v1_heldout_results=True,
        method_selection_evidence=str(OLD_RUN/'private_reuse_probe_v1/report.json'))
    protocol_path=ROOT/'artifacts/diagnostic_protocol/native-list-gru-reuse4-v1.json'
    write_json(protocol_path,protocol,exclusive=True)
    NEW_RUN.mkdir(parents=True)
    (NEW_RUN/'qualification').mkdir()
    shutil.copyfile(OLD_RUN/'qualification/independent_audit_v2.json',
                    NEW_RUN/'qualification/independent_audit_v2.json')
    specs=[]
    for seed in (11,29,47):
        spec={'schema':'fly-touhou-background-job-v1','python':str(PYTHON),
            'script':str(NEW_SOURCE/'scripts/run_native_diagnostic.py'),'cwd':str(ROOT),
            'control_directory':f'./outputs/legacy-build/background_jobs/native-list-gru-seed{seed}-reuse4-v1',
            'arguments':['--protocol',str(protocol_path),'--output',str(NEW_RUN/f'seed{seed}/train'),
                         '--mode','train','--seed',str(seed)],
            'environment':{'CUBLAS_WORKSPACE_CONFIG':':4096:8'}}
        path=ROOT/f'artifacts/runtime/native-list-gru-seed{seed}-reuse4-v1.json'
        write_json(path,spec,exclusive=True);specs.append(str(path))
    suite_spec={'schema':'fly-touhou-background-job-v1','python':str(PYTHON),
        'script':str(NEW_SOURCE/'scripts/run_native_diagnostic_suite.py'),'cwd':str(ROOT),
        'control_directory':'./outputs/legacy-build/background_jobs/native-list-gru-reuse4-suite-v1',
        'arguments':['--protocol',str(protocol_path),'--root',str(NEW_RUN),
                     '--output',str(NEW_RUN/'suite'),'--python',str(PYTHON)],
        'environment':{'CUBLAS_WORKSPACE_CONFIG':':4096:8'}}
    suite_spec_path=ROOT/'artifacts/runtime/native-list-gru-reuse4-suite-v1.json'
    write_json(suite_spec_path,suite_spec,exclusive=True)
    write_json(NEW_RUN/'preparation.json',{
        'schema':'sealed-GRU-reuse4-experiment-v1','status':'prepared-not-started',
        'source_manifest':{'path':str(NEW_SOURCE/'snapshot_manifest.json'),
            'sha256':file_sha256(NEW_SOURCE/'snapshot_manifest.json')},
        'protocol':{'path':str(protocol_path),'sha256':file_sha256(protocol_path)},
        'training_specs':specs,'suite_spec':str(suite_spec_path),
        'same_24_physical_sources':protected,'parent_fixed_course':str(OLD_RUN),
        'launch_only_after_parent_suite_terminal_success':True,
        'one_controlled_change':'one versus at most four actor passes per complete rollout',
        'heldout_data_used_for_this_method_selection':0,'new_fly_training':False,
        'G2_achieved':False,'final_acceptance':False},exclusive=True)
    print(json.dumps({'prepared':str(NEW_RUN),'protocol':str(protocol_path),
                      'snapshot':str(NEW_SOURCE),'files':len(files)},ensure_ascii=False))


if __name__=='__main__':main()
