"""Review completed rule-teacher data, never neural-model acceptance.

Reads episode metadata and archive indexes only. The CUDA fitter validates
every original shard and observation during its single cache-building pass.
"""
from pathlib import Path
import argparse
from collections import Counter
import hashlib
import json
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.atomic_json_io import atomic_write_json as write
from fly_touhou.thrl_study_io import read_shared_json as read,utc_now


def summarize(rows):
    if not rows:return None
    for row in rows:
        if row.get('outcome') not in ('clear','gameover') or row.get('full_story_episode') is not True or row.get('continues')!=0:
            raise ValueError('Only measured complete original-game teacher episodes')
        diag=row['reward_diagnostics']
        names=('native_misses','confirmed_deathbombs','ordinary_or_unconfirmed_bombs',
               'deathbomb_observed_window_count','deathbomb_decision_count')
        if any(type(diag.get(key)) is not int or diag[key]<0 for key in names):
            raise ValueError('Missing measured teacher resource counts')
        if (diag['native_misses']!=row['misses']
                or diag['confirmed_deathbombs']+diag['ordinary_or_unconfirmed_bombs']!=row['bombs_used']):
            raise ValueError('Native teacher resource counters disagree')
    collisions_lower=[r['misses']+r['reward_diagnostics']['confirmed_deathbombs'] for r in rows]
    return dict(episodes=len(rows),clears=sum(r['outcome']=='clear' for r in rows),
        mean_native_frames=statistics.mean(r['native_frames'] for r in rows),
        median_native_frames=statistics.median(r['native_frames'] for r in rows),
        mean_misses=statistics.mean(r['misses'] for r in rows),
        mean_Bombs=statistics.mean(r['bombs_used'] for r in rows),
        mean_confirmed_deathbombs=statistics.mean(r['reward_diagnostics']['confirmed_deathbombs'] for r in rows),
        mean_normal_or_unconfirmed_Bombs=statistics.mean(r['reward_diagnostics']['ordinary_or_unconfirmed_bombs'] for r in rows),
        known_Miss_plus_confirmed_save_collision_lower_bound=sum(collisions_lower),
        mean_known_collision_lower_bound=statistics.mean(collisions_lower),
        observed_contiguous_state4_windows=sum(r['reward_diagnostics']['deathbomb_observed_window_count'] for r in rows),
        decisions_made_in_observed_state4=sum(r['reward_diagnostics']['deathbomb_decision_count'] for r in rows),
        observed_windows_are_not_complete_native_collision_counts=True,
        no_Miss_games=sum(r['misses']==0 for r in rows),
        no_Miss_does_not_mean_no_collisions=True,
        highest_stage_histogram=dict(Counter(max(r['stage_history']) for r in rows)))


def run(args):
    from fly_touhou.thrl_full_teacher_data import FullTeacherDataset
    if args.output.exists():raise FileExistsError(args.output)
    dataset=FullTeacherDataset(args.folder)
    manifest=dataset.manifest;result=read(args.folder/'result.json')
    if (manifest.get('numerical_contract_fixture_only') or len(dataset.records)!=30
            or manifest!=result or result.get('complete_native_episodes')!=30
            or result.get('neural_model_test') is not False or result.get('optimizer_updates')!=0):
        raise ValueError('Completed current real thirty-world teacher cohort required')
    cleanup=read(args.folder/'native_cleanup.json')
    if cleanup.get('close_errors')!=[]:raise ValueError('Teacher worlds have unresolved cleanup errors')
    rows=result['episodes'];by_worker={r['worker']:r for r in rows}
    if set(by_worker)!=set(range(30)):raise ValueError('One natural teacher episode per world required')
    if sorted(r['native_seed'] for r in rows)!=sorted(result['actual_initial_native_seeds']):
        raise ValueError('Actual training seeds differ from the pinned collection plan')
    summary=summarize(rows);clear_summary=summarize([r for r in rows if r['outcome']=='clear'])
    report=dict(schema='B1-complete-natural-teacher-data-quality-review-v1',status='completed',utc=utc_now(),
        source_manifest=str(dataset.manifest_path),source_manifest_sha256=dataset.manifest_sha256,
        complete_native_teacher_games=30,original_teacher_rows=dataset.rows,
        teacher_revision=manifest['teacher_revision'],all_games=summary,clear_games=clear_summary,
        actual_native_training_seeds=result['actual_initial_native_seeds'],
        all_original_shards_validated_by_this_metadata_review=False,
        full_original_row_validation_assigned_to_fitter_cache_build=True,
        archive_scope='Thirty natural opening-to-native-end teacher trajectories;unequal lengths;no continue',
        original_neural_model_efficacy_not_proved=True,neural_model_test=False,
        teacher_quality_cannot_be_called_student_quality=True,
        parent_test_seeds_are_different=True,teacher_student_paired_comparison_not_claimed=True,
        native_worlds_started=0,optimizer_updates=0,source_data_or_policy_modified=False)
    args.output.mkdir(parents=True);write(args.output/'report.json',report)
    print(json.dumps(report))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--folder',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);run(p.parse_args())
