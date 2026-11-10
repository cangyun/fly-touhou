"""Prepare the whole six-stage shared-policy course; never launch implicitly."""
from pathlib import Path
import argparse,copy,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]


def prepare(base,catalog,bank,*,bank_path,output,run_output,tensorboard,budget=65536):
    from fly_touhou.nbnm_task import attack_key
    from fly_touhou.nbnm_course import interleave_stages
    from fly_touhou.thrl_training_mode import seed_split
    from fly_touhou.nbnm_run import validate_spec
    spec=copy.deepcopy(base);tasks={};entries={};local=[];alternate=[];stage_tasks=[];bridges=[]
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    source_bank=str(bank_path);bank_path=output/'start-bank.json'
    # The source preparer may still finish extra development prefixes. A run
    # uses its own immutable manifest, never a changing pool of native starts.
    bank_path.write_text(json.dumps(dict(bank,source_bank=source_bank),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    development=[s for s in range(60000,61000) if seed_split(s)=='development'][:20]
    test=[s for s in range(61000,62000) if seed_split(s)=='test'][:20]
    evaluation_seeds=dict(development=development,test=test)
    for c in catalog['cards']:
        if not c.get('lunatic_available'):continue
        key=f"s{c['stage']}_{c['attack']}"
        tasks[key]=dict(mode='attack',**{k:c[k] for k in ('stage','attack','spell_index','spell_id')})
        entries[key]=dict(entry_mode='native_spell_practice',practice_evaluation_seeds=evaluation_seeds)
        (alternate if c['stage']==4 and c['attack'].startswith('BossB') else local).append(key)
    measured={}
    for entry in bank['entries']:
        t=entry['task']
        # Older capture named Seija's mirror helper as a new midboss card.
        # Keep its source record, but do not create a false training boundary.
        if t['stage']==5 and attack_key(t['attack'])==('MBoss',True,3):continue
        measured.setdefault((t['stage'],t['attack']),[]).append(entry)
    missing=[];coverage=[];transitions=[]
    for stage in range(1,7):
        phases=sorted([(attack,rows) for (s,attack),rows in measured.items() if s==stage],
            key=lambda pair:min(r['landing_stage_frame'] for r in pair[1]))
        if not phases:missing.append(f'stage{stage}: no actual attack entries');continue
        for attack,rows in phases:
            splits={r['split'] for r in rows}
            if 'train' not in splits:missing.append(f's{stage} {attack}: missing training entry')
            if not attack_key(attack)[1]:
                key=f's{stage}_{attack}'
                tasks[key]=dict(mode='attack',stage=stage,attack=attack);local.append(key)
                entries[key]=dict(entry_mode='native_stage_prefix',start_bank=str(bank_path),practice_evaluation_seeds=evaluation_seeds,
                    prepare_evaluation_prefixes=True)
        full=f's{stage}_whole_stage';tasks[full]=dict(mode='stage',stage=stage)
        entries[full]=dict(entry_mode='native_stage_practice',practice_evaluation_seeds=evaluation_seeds)
        stage_tasks.append(full)
        # One continuous Boss tail and one midboss-to-end chain per stage.
        # Intermediate attacks never reset memory or provide success rewards.
        for prefix in ('MBoss','Boss'):
            candidates=[a for a,_ in phases if attack_key(a)[0].startswith(prefix)]
            if not candidates:continue
            first=candidates[0];key=f's{stage}_{prefix}_through_end'
            tasks[key]=dict(mode='chain',stage=stage,attack=first,end_stage=stage)
            entries[key]=dict(entry_mode='native_stage_prefix',start_bank=str(bank_path),practice_evaluation_seeds=evaluation_seeds,
                prepare_evaluation_prefixes=True)
            bridges.append(key)
        for (start,_),(end,_) in zip(phases,phases[1:]):
            transitions.append(dict(stage=stage,start=start,end=end,
                task=dict(mode='chain',stage=stage,attack=start,end_stage=stage,end_attack=end),
                entry=dict(entry_mode='native_stage_prefix',start_bank=str(bank_path),practice_evaluation_seeds=evaluation_seeds,
                    prepare_evaluation_prefixes=True)))
        coverage.append(dict(stage=stage,spells=sum(tasks[k]['stage']==stage for k in local if tasks[k].get('spell_id') is not None),
            nonspells=sum(not attack_key(a)[1] for a,_ in phases),observed_attack_order=[a for a,_ in phases],
            entry_seeds={a:{split:sorted({r['native_seed'] for r in rows if r['split']==split})
                for split in ('train','development')} for a,rows in phases}))
    if missing:raise ValueError('Full-course entry preparation incomplete: '+'; '.join(missing))
    tasks['full_story']=dict(mode='full_story');entries['full_story']=dict(entry_mode='story_prefix')
    local=interleave_stages(local,tasks)
    # Start all late-stage paths early. Cycling then covers every stage and
    # Boss-tail chain, irrespective of first-contact failures in full runs.
    stage_pool=[f's{s}_whole_stage' for s in (3,4,5,6,1,2)]+interleave_stages(bridges,tasks)
    primary='s4_BossCard2'
    spec.update(tasks=tasks,task_entries=entries,workers=16,segment_decisions=8192,decision_budget=budget,
        world_groups=['local']*8+['stage_chain']*4+['full_story']*4,
        task_pools=dict(local=local,stage_chain=stage_pool,full_story=['full_story']),
        group_weights=dict(local=.5,stage_chain=.25,full_story=.25),
        output=str(run_output),tensorboard_directory=str(tensorboard),
        full_story_evaluation_seeds=development,full_story_heldout_seeds=test,evaluation_workers=4,
        evaluation_tasks=['s3_BossCard2','s4_BossCard2','s6_BossCard5','full_story'],
        evaluate_at_end=True,status_port=0,
        entry_cache_directory=str(output/'independent-entry-cache'),
        qualification=dict(minimum_attempts=20,minimum_successes=18,minimum_rate=.9,
            distinct_native_seeds=True,frozen_only=True,automatic_promotion=False),
        course=dict(single_shared_policy=True,rotation='at native episode terminal only',
            episode_preserved_across_updates=True,topology_changed=False,
            alternate_route_cards_prepared=alternate,alternate_route_cards_in_primary_training=False,
            task_catalog_complete=True))
    spec['nbnm'].update(task=tasks[primary],practice_evaluation_seeds=evaluation_seeds,
        teacher_mode='nbnm-route-v1',task_context='critic-only-v1',history_cuda_graph=True,
        rule_guidance_coefficient=.1,objective=dict(phase='completion',speed_weight=.01,time_scale_frames=3600))
    spec['experiment_notes']=dict(single_shared_policy=True,brain_topology_changed=False,preparation_only=True,
        teacher='all18 movement directions; native short horizons; conditional48/96 hints; no Bomb or teacher takeover',
        curriculum='all-stage local practice + forward stage chains + uninterrupted fresh-opening full Story')
    validation=validate_spec(spec)
    documents={'route.json':spec,'validation.json':validation,
        'coverage.json':dict(stages=coverage,primary_spells=len(local)-sum(r['nonspells'] for r in coverage),
            alternate_spells=len(alternate),native_menu_cards=len(catalog['cards']),
            primary_training_tasks=len(set(local+stage_pool+['full_story'])),prepared_tasks=len(tasks),
            complete_mechanics_preparation_is_not_a_claim_of_NBNM_mastery=True),
        'forward-course.json':dict(sampling_fractions=spec['group_weights'],tasks=tasks,task_entries=entries,
            adjacent_chains=transitions,global_connection='full Story from opening, same brain carried across every native transition',
            next_phase='Increase full_story fraction after separate frozen local/chain qualification, never swap per-card experts',
            qualifier=spec['qualification'])}
    for name,value in documents.items():(output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return spec


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('base-config','catalog','bank','output','run-output','tensorboard'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--decision-budget',type=int,default=65536);a=p.parse_args()
    from fly_touhou.nbnm_run import read
    prepare(read(a.base_config),read(a.catalog),read(a.bank),bank_path=a.bank,output=a.output,
        run_output=a.run_output,tensorboard=a.tensorboard,budget=a.decision_budget)
    print(a.output/'route.json')


if __name__=='__main__':main()
