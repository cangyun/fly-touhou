"""Record formal-game 36-action curricula, retaining original observations."""
from pathlib import Path
import argparse
import gzip
import json
import time
import uuid

from fly_touhou.campaign import CampaignTouhouEnv
from fly_touhou.campaign_teacher import CampaignTeacher
from fly_touhou.experiment import ExperimentLogger
from fly_touhou.observation import ObservationConfig
from fly_touhou.learning import _context
from fly_touhou.native import sha256_file

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--exe', type=Path, default=Path(
    r'games/th14/th14.exe'))
parser.add_argument('--profile', type=Path, default=ROOT/'profiles/th14_1.00b_original.json')
parser.add_argument('--difficulty', type=int, choices=range(4), required=True)
parser.add_argument('--episodes', type=int, default=4)
parser.add_argument('--frames', type=int, default=1800)
parser.add_argument('--teacher', choices=('legacy','resource'), default='resource')
parser.add_argument('--private-appdata', type=Path,
                    help='Explicit private save root for this independent worker (ASCII path).')
parser.add_argument('--dialogue-skip', action='store_true',
                    help='Deliver a native fresh-Z edge on dialogue entry; requires --private-appdata.')
parser.add_argument('--parallel-startup', action='store_true',
                    help='Use a unique owned startup mutex; requires --private-appdata.')
parser.add_argument('--window-title', default=None,
                    help='Optional label for this worker game window; requires --private-appdata.')
parser.add_argument('--output', type=Path, required=True)
options = parser.parse_args()
if (options.dialogue_skip or options.window_title or options.parallel_startup) and options.private_appdata is None:
    parser.error('--dialogue-skip, --parallel-startup and --window-title require an explicit --private-appdata directory')
isolated = options.private_appdata is not None
output = options.output.resolve()
output.parent.mkdir(parents=True, exist_ok=True)
if output.exists():
    raise SystemExit('An existing original-game trajectory will never be overwritten.')
run_id = uuid.uuid4().hex
config = {'phase':'teacher-curriculum-recording','role':'teacher, not final full-brain acceptance',
          'difficulty':options.difficulty,'episodes':options.episodes,'frame_cap':options.frames,
          'n_actions':36,'engineered_risk_features':False,'teacher':options.teacher,
          'game_rng_seed_applied':False, 'game_exe_sha256':sha256_file(options.exe)}
if options.teacher=='resource':
    from fly_touhou.resource_environment import ResourceCampaignEnv
    from fly_touhou.resource_teacher import ResourceTeacher
    env_type = ResourceCampaignEnv
    env_options = {}
    if isolated:
        from fly_touhou.isolated_campaign import IsolatedResourceCampaignEnv
        env_type = IsolatedResourceCampaignEnv
        if options.parallel_startup:
            from fly_touhou.parallel_campaign import ParallelResourceCampaignEnv
            env_type = ParallelResourceCampaignEnv
        env_options = {'private_appdata':options.private_appdata,
                       'dialogue_skip':options.dialogue_skip,'window_title':options.window_title}
    env = env_type(options.exe,options.profile,difficulty=options.difficulty,
                   max_episode_frames=options.frames,**env_options)
    teacher=ResourceTeacher(env.encoder.planner)
    config['teacher_config']=teacher.config()
    config['observation_schema']=env.encoder.specification()['schema']
else:
    env_type = CampaignTouhouEnv
    env_options = {}
    if isolated:
        from fly_touhou.isolated_campaign import IsolatedCampaignEnv
        env_type = IsolatedCampaignEnv
        if options.parallel_startup:
            from fly_touhou.parallel_campaign import ParallelCampaignEnv
            env_type = ParallelCampaignEnv
        env_options = {'private_appdata':options.private_appdata,
                       'dialogue_skip':options.dialogue_skip,'window_title':options.window_title}
    env = env_type(options.exe,options.profile,difficulty=options.difficulty,
        max_episode_frames=options.frames,observation_config=ObservationConfig(include_action_risk=False),
        **env_options)
    teacher=CampaignTeacher(env.encoder.planner)
if isolated:
    config.update({'private_appdata':str(env.private_appdata),
                   'dialogue_skip':options.dialogue_skip,'window_title':options.window_title,
                   'parallel_startup':options.parallel_startup,
                   'campaign_environment':type(env).__name__})
source_files = {'original_game':options.exe,'profile':options.profile,'collector':Path(__file__),
    'input_hook':ROOT/'src/fly_touhou/hooks/th14.js'}
for module in ('campaign_teacher', 'planner', 'observation', 'state', 'campaign',
               'background', 'native', 'frida_controller', 'experiment'):
    source_files[module] = ROOT/'src/fly_touhou'/f'{module}.py'
if options.teacher=='resource':
    for module in ('resource_teacher','resource_environment','resource_observation','resources'):
        source_files[module]=ROOT/'src/fly_touhou'/f'{module}.py'
if isolated:
    for module in ('isolated_campaign','dialogue_controller'):
        source_files[module]=ROOT/'src/fly_touhou'/f'{module}.py'
if options.parallel_startup:
    for module in ('parallel_controller','parallel_campaign'):
        source_files[module]=ROOT/'src/fly_touhou'/f'{module}.py'
    source_files['parallel_startup_hook']=ROOT/'src/fly_touhou/hooks/th14_parallel_startup.js'
logger = ExperimentLogger(output.parent/(output.name+'.experiment'),config,files=source_files)
reports=[]
started=time.monotonic()
try:
    with logger.phase('native-teacher-demonstrations'), gzip.open(output,'xt',encoding='utf-8') as stream:
        def emit(value):
            stream.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')
        emit({'record_type':'metadata','schema':'whole-brain-campaign-demo-v1','n_actions':36,
              'encoder':env.encoder.specification(),'teacher_kind':type(teacher).__name__,
              'config':config,'trace_id':run_id})
        for episode in range(options.episodes):
            teacher.reset()
            observation,info=env.reset(seed=episode)
            context=_context(env.snapshot,info)
            logger.record_native_context(env.snapshot,info)
            episode_id=f'{run_id}:{episode}'
            dialogue_cursor = 0
            if isolated:
                logger.event('native-worker-isolation',{'episode_id':episode_id,'pid':env.pid,
                    'private_appdata':str(env.private_appdata),'dialogue_skip':options.dialogue_skip,
                    'window_title':options.window_title,'window_label':info.get('window_label'),
                    'window_label_status':info.get('window_label_status'),
                    'desktop_input':False})
            reward_sum=0.0
            action_counts=[0]*36
            hazards=max_bullets=0
            modes={}
            initial_power=max_power=env.snapshot.power
            min_y=env.snapshot.player.y
            edge_decisions=poc_entries=peak_autocollecting=0
            print('EPISODE',episode,'difficulty',env.snapshot.difficulty,'native',env.pid,flush=True)
            for step in range(options.frames//env.frame_skip+1):
                before=env.snapshot
                probabilities=teacher.probabilities(before)
                diagnostic=teacher.diagnostics() if hasattr(teacher,'diagnostics') else None
                action=int(probabilities.argmax())
                observation,reward,terminated,truncated,info=env.step(action)
                after=env.snapshot
                reward_sum+=reward
                action_counts[action]+=1
                count=sum(b.active and not b.intangible for b in before.bullets)
                hazards+=int(count>0 or any(l.active for l in before.lasers))
                max_bullets=max(max_bullets,count)
                max_power=max(max_power,after.power)
                if after.player:
                    min_y=min(min_y,after.player.y)
                    edge_decisions+=abs(after.player.x)>=144
                    line=after.extra.get('resources',{}).get('poc_line_y',128)
                    poc_entries+=bool(before.player and before.player.y>line>=after.player.y)
                peak_autocollecting=max(peak_autocollecting,sum(
                    item.get('autocollecting',False) for item in after.extra.get('resources',{}).get('items',[])))
                if diagnostic:
                    mode=diagnostic['mode']
                    modes[mode]=modes.get(mode,0)+1
                record={'record_type':'transition','episode_id':episode_id,'step':step,
                        'snapshot':before.to_dict(),'teacher_probabilities':probabilities.tolist(),
                        'action':action,'reward':reward,'terminated':bool(terminated),'truncated':bool(truncated),
                        'elapsed_frames':info['elapsed_frames'],'next_snapshot':after.to_dict()}
                if diagnostic:
                    record['teacher_diagnostic']=diagnostic
                if step==0:
                    record['native_context']=context
                dialogue_events = getattr(env.controller,'dialogue_input_events',())
                new_dialogue_events = dialogue_events[dialogue_cursor:]
                if new_dialogue_events:
                    record['dialogue_input_events'] = list(new_dialogue_events)
                    for event in new_dialogue_events:
                        logger.event('native-dialogue-input',{'episode_id':episode_id,'pid':env.pid,**event})
                    dialogue_cursor = len(dialogue_events)
                emit(record)
                if step%100==0 or terminated or truncated:
                    stream.flush()
                    logger.event('native-curriculum-progress',{'episode_id':episode_id,'step':step,
                        'native_frames':env.episode_frames,'stage':after.stage,'bullets':len(after.bullets),
                        'lives':after.lives,'bombs':after.bombs,'terminated':terminated,'truncated':truncated})
                    print('PROGRESS',episode,step,after.stage_frame,len(after.bullets),after.lives,after.bombs,flush=True)
                if terminated or truncated:
                    report={'record_type':'episode_summary','episode_id':episode_id,'steps':step+1,
                        'native_frames':env.episode_frames,'reward':reward_sum,'actions':action_counts,
                        'hazard_decisions':hazards,'max_bullets':max_bullets,
                        'teacher_modes':modes,'initial_power':initial_power,'max_power':max_power,
                        'minimum_y':min_y,'edge_decisions':edge_decisions,'poc_entries':poc_entries,
                        'peak_native_autocollecting_items':peak_autocollecting,
                        'campaign':{k:v for k,v in info.items() if k not in ('snapshot',)},
                        'final_snapshot':after.to_dict()}
                    if isolated:
                        report.update({'private_appdata':str(env.private_appdata),
                            'dialogue_release_events':dialogue_cursor,
                            'window_title':options.window_title,
                            'window_label_status':getattr(env.controller,'window_label_status',None)})
                    emit(report)
                    reports.append(report)
                    logger.event('native-curriculum-episode-finished',{k:v for k,v in report.items()
                        if k not in ('final_snapshot','campaign')})
                    break
        emit({'record_type':'collection_summary','episodes':len(reports),
              'native_frames':sum(r['native_frames'] for r in reports)})
    summary={'path':str(output),'config':config,'episodes':reports,
             'elapsed_seconds':time.monotonic()-started}
    output.with_suffix('.report.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    logger.event('native-curriculum-file-sealed',{'file':str(output),'episodes':len(reports)})
finally:
    env.close()
