"""Bounded functional probe on a newly owned original TH14 process only."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--decisions', type=int, default=128)
    parser.add_argument('--rendered', action='store_true')
    parser.add_argument('--binary-reader', action='store_true')
    parser.add_argument('--verify-binary', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.decisions <= 8192:
        raise ValueError('Probe decisions must be in 1..8192')
    args.output.mkdir(parents=True, exist_ok=False)
    from fly_touhou.thrl_native_events import HiddenTHRLDialogueGeometryController
    controller = HiddenTHRLDialogueGeometryController(args.executable, args.profile,
        private_appdata=args.output/'userdata', worker_name='headless-functional-probe',
        timeout=25, headless=not args.rendered,binary_reader=args.binary_reader)
    report = {'schema': 'th14-headless-functional-probe-v1', 'status': 'started',
        'requested_headless': not args.rendered, 'desktop_input': False}
    try:
        started = time.perf_counter()
        controller.start()
        first = controller.ensure_run(3)
        if not args.rendered and not (first.extra.get('native_execution') or {}).get('device_created'):
            raise RuntimeError('No confirmed NULLREF device after entering live gameplay')
        report.update(pid=controller.pid, startup_seconds=time.perf_counter()-started,
                      execution=first.extra.get('native_execution'))
        trace = []
        binary_checks=0
        nonempty_binary_checks=0
        started = time.perf_counter()
        for index in range(args.decisions):
            state = controller.advance(1 | 8, 4)  # shot + slow, native finite lives
            if args.verify_binary and args.binary_reader:
                from verify_th14_execution_equivalence import canonical
                from fly_touhou.thrl_geometry_observation import THRLGeometryObservationEncoder
                import numpy as np
                sidecar=controller._binary_reader
                controller._binary_reader=None
                controller._memory.invalidate_prefetch()
                try:fresh=controller.read()
                finally:controller._binary_reader=sidecar
                if canonical(state)!=canonical(fresh):raise ValueError('Binary/direct semantic snapshot mismatch')
                e=THRLGeometryObservationEncoder()
                a,b=e.encode(state),e.encode(fresh)
                for key in a:np.testing.assert_array_equal(a[key],b[key])
                binary_checks+=1
                nonempty_binary_checks+=int(bool(state.bullets))
            trace.append({'index': index, 'frame': state.frame, 'stage_frame': state.stage_frame,
                'stage': state.stage, 'lives': state.lives, 'bullets': len(state.bullets),
                'player': asdict(state.player) if state.player else None,
                'rng': state.extra.get('rng')})
        elapsed = time.perf_counter()-started
        report.update(status='passed', decisions=len(trace), elapsed_seconds=elapsed,
            decisions_per_second=len(trace)/elapsed,
            native_frames_advanced=state.frame-first.frame,
            logical_frames_per_second=(state.frame-first.frame)/elapsed,
            frame_relation_exact=all(row['frame']==first.frame+4*(i+1) for i,row in enumerate(trace)),
            execution=state.extra.get('native_execution'),visibility=state.extra.get('native_visibility'))
        if not args.rendered and report['visibility'] and report['visibility']['on_screen_visible_checkpoints']:
            raise RuntimeError('Headless owned window appeared on screen')
        report['binary_reader']=state.extra.get('native_binary_reader')
        report['binary_direct_exact_checks']=binary_checks
        report['binary_direct_nonempty_bullet_checks']=nonempty_binary_checks
        (args.output/'trace.json').write_text(json.dumps(trace, ensure_ascii=False), encoding='utf-8')
        (args.output/'final_snapshot.json').write_text(json.dumps(state.to_dict(), ensure_ascii=False), encoding='utf-8')
    except BaseException as error:
        report.update(status='failed', error=repr(error), traceback=traceback.format_exc(),
                      execution=getattr(controller, '_execution_audit', None),
                      startup=controller.fine_start_audit())
        raise
    finally:
        controller.close()
        (args.output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
