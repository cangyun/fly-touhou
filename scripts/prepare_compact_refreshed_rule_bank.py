"""Build training-only windows from a completed current-sensor replay.

Every window stays in the source game's training split. Frozen original games
provide evaluation; nearby windows from this same demonstration are not a test.
"""
from pathlib import Path
import argparse

from prepare_compact_native_rule_windows import make_bank
from analyze_native_teacher_windows import analyze
from fly_touhou.compact_native_rule_replay import NativeRuleTeacherReplay
from fly_touhou.thrl_study_io import read_shared_json
from fly_touhou.thrl_durable_records import write_json


def prepare(source, output, *, per_stage=128):
    source, output = Path(source), Path(output)
    index = read_shared_json(source / 'index.json')
    if index.get('status') != 'completed':
        raise ValueError('Finish the native replay before accepting this bank')
    if index['episode_summary']['native_seed_split'] != 'train':
        raise ValueError('A held-out game must not become training windows')
    if index.get('new_independent_clear') is not False or not index.get('recorded_source'):
        raise ValueError('Expected a refreshed recorded training demonstration')
    bank = make_bank(source, output, stages=tuple(range(1, 7)), count=per_stage,
        burn_in=32, learn_steps=16, heldout_fraction=0.,behavior_horizon=96,observed_route_share=.25,
        protected_route_windows=32)
    revision = dict(index['native_repair_revisions'])
    wall = index.get('native_wall_reactive_forecast')
    if wall is not None:
        revision['reactive_orbs'] = wall
    for window in bank['windows']:
        window['source_game'] = 0
    bank.update(games=[dict(split='train', source=str(source.resolve()),
            recorded_source=index['recorded_source'],
            native_seed=index['episode_summary']['full_story_native_seed'],
            new_independent_clear=False)],
        source_native_repair_revisions=[revision],
        source_native_wall_reactive_forecast=wall,
        refreshed_distribution_stages=index['refreshed_distribution_stages'],
        early_guidance=index['early_guidance'],
        split='Training source game only;no held-out windows or independent-clear claim.',
        evaluation='Separate frozen full-opening student games,never this teacher replay.')
    switch=source/'route-guidance-switch.json'
    if switch.exists():bank['route_guidance_switch']=read_shared_json(switch)
    write_json(output / 'manifest.json', bank)
    replay = NativeRuleTeacherReplay(output / 'manifest.json')
    if sorted(replay.stage_indices) != list(range(1, 7)):
        raise ValueError('The refreshed game must supply usable windows for all six stages')
    report = dict(loader=replay.summary(), labels=analyze(output / 'manifest.json'),
        teacher_controlled_data=True, PPO_samples=0, new_independent_clear=False,
        parameters_updated=False)
    write_json(output / 'report.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--per-stage', type=int, default=128)
    args = parser.parse_args()
    result = prepare(args.source, args.output, per_stage=args.per_stage)
    print(dict(windows=result['loader']['windows'], stages=result['loader']['stages'],
        PPO_samples=0, new_independent_clear=False))
