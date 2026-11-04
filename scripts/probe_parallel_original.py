"""Bounded second-original-process startup probe; no model performance claim."""
from pathlib import Path
import argparse
import json

from fly_touhou.campaign import FULL_GAME_SHA256
from fly_touhou.experiment import ExperimentLogger
from fly_touhou.parallel_controller import ParallelDialogueResourceGameController
from fly_touhou.dialogue_controller import label_owned_game_windows


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--game-exe', type=Path, default=Path(
        r'games/th14/th14.exe'))
    parser.add_argument('--profile', type=Path, default=root/'profiles/th14_1.00b_original.json')
    args = parser.parse_args()
    out = args.output.resolve()
    files = {'executable':args.game_exe, 'profile':args.profile, 'probe':Path(__file__)}
    for module in ('parallel_controller','dialogue_controller','frida_controller','resource_environment',
                   'resources','native','state','experiment'):
        files[module] = root/'src/fly_touhou'/f'{module}.py'
    for hook in ('th14.js','th14_parallel_startup.js'):
        files[hook] = root/'src/fly_touhou/hooks'/hook
    config = {'phase':'bounded-second-original-process-startup', 'game_exe_sha256':FULL_GAME_SHA256,
              'gameplay_acceptance_claim':False, 'policy_actor':'protocol constant SHOT',
              'native_frame_cap':256, 'private_appdata':str(out/'game_userdata'),
              'startup_adjustment':'exact native mutex caller/name receives a unique owned-process name',
              'resource_overrides':False, 'desktop_input':False}
    logger = ExperimentLogger(out,config,files=files)
    controller = ParallelDialogueResourceGameController(args.game_exe,args.profile,
        private_appdata=out/'game_userdata',worker_name='parallel-probe-b',timeout=25)
    snapshots = []
    try:
        with logger.phase('second-process-native-start-and-advance'):
            controller.start()
            before = controller.ensure_run(difficulty=3)
            logger.record_native_context(before, {'exe_sha256':FULL_GAME_SHA256})
            logger.event('owned-window-label-request', label_owned_game_windows(controller,'并行原作接口验证 B'))
            for step in range(64):
                after = controller.advance(1,4)
                snapshots.append({'step':step, 'snapshot':after.to_dict()})
            events = controller.parallel_startup_audit()
            replacements = [e for e in events['events'] if e['type']=='parallel-startup-mutex-replaced']
            results = [e for e in events['events'] if e['type']=='parallel-startup-mutex-result']
            if len(replacements)!=1 or len(results)!=1 or results[0]['native_last_error']==183:
                raise RuntimeError('The exact startup mutex was not independently created once')
            if after.frame-before.frame != 256 or after.difficulty!=3:
                raise RuntimeError('Second original process did not advance the bounded Lunatic frames')
            report = {'status':'passed-bounded-second-original-process-probe', 'pid':controller.pid,
                'native_frames':256, 'difficulty':after.difficulty, 'stage':after.stage,
                'native_lives':after.lives, 'startup':events, 'last_snapshot':after.to_dict(),
                'model_control_tested':False, 'desktop_input':False, 'resource_overrides':False}
            (out/'snapshots.json').write_text(json.dumps(snapshots,ensure_ascii=False),encoding='utf-8')
            (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
            logger.event('parallel-probe-completed',{k:v for k,v in report.items() if k!='last_snapshot'})
            logger.verify_inputs()
            print(json.dumps({k:report[k] for k in ('status','pid','native_frames','difficulty','stage')},ensure_ascii=False),flush=True)
    finally:
        controller.close()
        cleanup = controller.parallel_startup_audit()
        (out/'startup_cleanup.json').write_text(json.dumps(cleanup,ensure_ascii=False,indent=2),encoding='utf-8')
        logger.event('owned-second-process-closed',cleanup)


if __name__=='__main__':
    main()
