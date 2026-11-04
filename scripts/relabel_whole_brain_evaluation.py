"""Query a teacher on retained student states without changing native actions.

This is offline dataset aggregation. Reused evaluations become training
experience for subsequent models; a new independent game run remains required.
"""
from pathlib import Path
import argparse
import gzip
import json
import uuid

from fly_touhou.campaign import FULL_GAME_SHA256
from fly_touhou.experiment import ExperimentLogger, file_sha256, verify_events
from fly_touhou.learning import _context
from fly_touhou.resource_observation import ResourceObservationEncoder
from fly_touhou.resource_teacher import ResourceTeacher
from fly_touhou.state import Snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluations', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--motion-calibration', type=Path,
                        help='Required for raw hazard v3 observations; selects corrected native v4 teacher.')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError('Original observations and labels must never be overwritten')
    source_files = {'relabel_code': Path(__file__)}
    for module in ('resource_teacher', 'resource_observation', 'planner', 'state', 'experiment', 'learning'):
        source_files[module] = root/'src/fly_touhou'/f'{module}.py'
    sources, specification, executable = [], None, None
    for index, directory in enumerate(args.evaluations):
        directory = directory.resolve()
        manifest_path = directory/'manifest.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        events = verify_events(directory/'events.jsonl', run_id=manifest['run_id'])
        if (not events or events[0]['event'] != 'experiment_created'
                or events[0]['payload']['manifest_sha256'] != file_sha256(manifest_path)):
            raise ValueError('Parent manifest differs from its original creation event')
        config = manifest['config']
        if (config.get('actor') != 'complete-brain-descending-spikes'
                or config.get('game_exe_sha256') != FULL_GAME_SHA256
                or config.get('teacher_action_calls') != 0
                or config.get('safety_takeovers') != 0):
            raise ValueError('Only retained independent whole-brain original-game campaigns are accepted')
        if specification is not None and specification != config['encoder']:
            raise ValueError('Different observation specifications cannot be combined')
        specification = config['encoder']
        executable = Path(manifest['files']['game_executable']['path'])
        for name in ('manifest.json', 'events.jsonl', 'evaluation_summary.json'):
            source_files[f'parent_{index}_{name}'] = directory/name
        for path in sorted(directory.glob('campaign_*.jsonl.gz')):
            digest = file_sha256(path)
            source_files[f'campaign_{len(sources)}'] = path
            sources.append({'path': path, 'sha256': digest, 'parent_run': manifest['run_id']})
    if not sources or executable is None or file_sha256(executable) != FULL_GAME_SHA256:
        raise ValueError('No complete campaign or the formal executable identity differs')
    calibration=None
    if specification.get('schema') == 'native-resource-hazard-observation-v3':
        if args.motion_calibration is None:
            parser.error('Measured hazard observations require --motion-calibration and native v4 teacher')
        from fly_touhou.hazard_observation import HazardObservationEncoder
        from fly_touhou.motion_calibration import calibrated_planner
        from fly_touhou.native_hazard_planner import NativeHazardPlanner
        from fly_touhou.resource_teacher_v4 import ResourceTeacherV4
        encoder=HazardObservationEncoder.from_specification(specification)
        planner_config,calibration=calibrated_planner(encoder.planner.config,args.motion_calibration,
            exe_sha256=FULL_GAME_SHA256)
        teacher=ResourceTeacherV4(NativeHazardPlanner(planner_config))
        for module in ('hazard_observation','hazard_reader','motion_calibration','native_hazard_planner',
                       'dynamic_planner','resource_teacher_v2','resource_teacher_v4'):
            source_files[module]=root/'src/fly_touhou'/f'{module}.py'
        source_files['motion_calibration']=args.motion_calibration
    else:
        encoder = ResourceObservationEncoder.from_specification(specification)
        teacher = ResourceTeacher(encoder.planner)
    source_files['formal_executable'] = executable
    config = {'phase': 'offline-DAgger-student-state-relabel', 'curriculum': 'DAgger-student-states',
        'game_exe_sha256': FULL_GAME_SHA256, 'teacher_config': teacher.config(),
        'executed_actions': 'preserved whole-brain actions from immutable native evaluations',
        'new_native_game_runs': 0, 'teacher_gameplay_takeovers': 0,
        'reused_evaluations_are_training_experience': True,
        'future_acceptance_requires_new_independent_game_runs': True}
    if calibration is not None:
        config['motion_calibration']=calibration
    logger = ExperimentLogger(output.parent/(output.name+'.experiment'), config, files=source_files)
    output.parent.mkdir(parents=True, exist_ok=True)
    summaries = []
    with logger.phase('offline-resource-teacher-queries'), gzip.open(output, 'xt', encoding='utf-8') as target:
        def emit(row):
            target.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
        emit({'record_type':'metadata', 'schema':'whole-brain-campaign-demo-v1', 'n_actions':36,
              'encoder':specification, 'teacher_kind':type(teacher).__name__+' queried on whole-brain-controlled states',
              'config':config, 'trace_id':uuid.uuid4().hex})
        for source in sources:
            teacher.reset()
            episode_id = f'DAgger:{source["sha256"]}:0'
            count, frames, finished, final = 0, 0, False, None
            bomb_queries=correct_bomb_actions=deathbomb_queries=0
            recorded_bomb_mass=[]
            with gzip.open(source['path'], 'rt', encoding='utf-8') as stream:
                for line in stream:
                    row = json.loads(line)
                    if row['record_type'] != 'transition':
                        continue
                    if (finished or row['step'] != count or row.get('teacher_action_calls') != 0
                            or row.get('action_source') != 'whole-brain-descending-spikes'):
                        raise ValueError('Student trajectory is discontinuous or contains a different actor')
                    before = Snapshot.from_dict(row['snapshot'])
                    if before.extra.get('desktop_input') is not False:
                        raise ValueError('Original state did not certify process-local input')
                    encoder.encode(before)
                    if calibration is not None and (before.character,before.subshot,before.difficulty)!=(0,0,3):
                        raise ValueError('Native student identity differs from calibrated teacher')
                    probabilities = teacher.probabilities(before)
                    if probabilities.argmax()>=18:
                        bomb_queries+=1
                        correct_bomb_actions+=int(row['action']>=18)
                        saved=row.get('action_evidence',{}).get('probabilities')
                        if saved is not None:
                            recorded_bomb_mass.append(sum(saved[18:]))
                    if before.player and before.bombs>0 and (
                            before.player.state==4 or before.player.deathbomb_timer>0):
                        deathbomb_queries+=1
                    record = {key:row[key] for key in ('step','snapshot','action','reward','elapsed_frames',
                                                      'next_snapshot','terminated','truncated')}
                    record.update({'record_type':'transition', 'episode_id':episode_id,
                        'teacher_probabilities':probabilities.tolist(),
                        'teacher_diagnostic':teacher.diagnostics(),
                        'teacher_action':int(probabilities.argmax()),
                        'executed_action_source':'whole-brain-descending-spikes',
                        'label_source':'offline teacher query; no new native action',
                        'parent_campaign_sha256':source['sha256'], 'parent_run':source['parent_run']})
                    if count == 0:
                        record['native_context'] = _context(before, {'exe_sha256':FULL_GAME_SHA256})
                    emit(record)
                    count += 1
                    frames += row['elapsed_frames']
                    finished = row['terminated'] or row['truncated']
                    final = row['next_snapshot']
            if not finished or count == 0:
                raise ValueError('A complete terminated/truncated student episode is required')
            summary = {'record_type':'episode_summary', 'episode_id':episode_id,
                       'steps':count, 'native_frames':frames, 'teacher_queries':count,
                       'executed_actions_preserved':True, 'parent_campaign_sha256':source['sha256'],
                       'teacher_bomb_queries':bomb_queries,'actual_bomb_actions_on_these_queries':correct_bomb_actions,
                       'native_deathbomb_queries_with_bombs':deathbomb_queries,
                       'actual_policy_bomb_mass_on_teacher_bomb_queries':{
                           'observations':len(recorded_bomb_mass),
                           'mean':sum(recorded_bomb_mass)/len(recorded_bomb_mass) if recorded_bomb_mass else None,
                           'maximum':max(recorded_bomb_mass,default=None)},
                       'final_snapshot':final}
            emit(summary)
            summaries.append({k:v for k,v in summary.items() if k != 'final_snapshot'})
        emit({'record_type':'collection_summary', 'episodes':len(summaries),
              'native_frames':sum(s['native_frames'] for s in summaries), 'new_native_frames':0})
    report = {'config':config, 'episodes':summaries, 'output_sha256':file_sha256(output)}
    output.with_suffix('.report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    logger.event('offline-student-states-relabelled', report)
    print(json.dumps({k:v for k,v in report.items() if k != 'config'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
