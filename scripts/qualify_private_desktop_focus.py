"""Combine actual native advancement and independent input-desktop observation.

This script never launches a process or sends input. Failed startup observations
cannot qualify a runtime even if the foreground-event count happens to be zero.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text('utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_rendering_provenance(report,root=None):
    """New COM reports must identify the loaded DLL, not inherit NULLREF proof."""
    root=Path(root or ROOT)
    execution=report['final'].get('execution') or {}
    if execution.get('render_backend')!='fake-com-v1':
        if (execution.get('render_backend') not in (None,'nullref-v1')
                or execution.get('schema') not in (None,'th14-nullref-uncapped-v1')):
            raise ValueError('Unknown native rendering backend')
        if (not execution.get('device_created') or not execution.get('window_changes_disabled')
                or execution.get('behavior_flags',0)&0x800==0):
            raise ValueError('Historical NULLREF audit lacks its original device/flag evidence')
        return  # Historical NULLREF reports keep their original semantics.
    names=('native/th14_fake_d3d9.dll','native/th14_fake_d3d9.dll.sha256')
    hashes=report.get('source_hashes',{})
    if any(name not in hashes for name in names):
        raise ValueError('Software COM proof lacks DLL/checksum source identity')
    if (execution.get('schema')!='th14-fake-com-uncapped-v1'
            or execution.get('requested_device')!='host-memory-fake-com'
            or execution.get('fake_d3d9_loaded') is not True
            or execution.get('system_create_device_called') is not False
            or execution.get('unsupported_getters')!=0
            or not execution.get('device_created') or not execution.get('window_changes_disabled')
            or execution.get('fake_d3d9_sha256')!=hashes[names[0]]):
        raise ValueError('Software COM audit does not match the qualified DLL')
    if Path(execution.get('fake_d3d9_path','')).resolve()!=(Path(report['runtime'])/names[0]).resolve():
        raise ValueError('Loaded software COM DLL is outside the selected runtime')
    for name in names:
        if digest(root/name)!=hashes[name]:raise ValueError('Native DLL/checksum changed after qualification')


def qualified_source_hashes(root=None):
    root=Path(root or ROOT)
    paths=[p for p in (root/'src').rglob('*') if p.is_file()
        and p.suffix in ('.py','.js') and '__pycache__' not in p.parts]
    paths.extend(p for p in (root/'native').rglob('*') if p.is_file()
        and not {'target','.git','__pycache__'}.intersection(p.relative_to(root/'native').parts)
        and p.suffix not in ('.pyc','.pyo'))
    return {p.relative_to(root).as_posix():digest(p) for p in sorted(paths)}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--single', type=Path, required=True)
    parser.add_argument('--single-observer', type=Path, required=True)
    parser.add_argument('--concurrent', type=Path, nargs='+', required=True)
    parser.add_argument('--concurrent-observer', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve previous qualification')
    groups = []
    all_pids = set()
    render_backends=set()
    for label, paths, observer_path in (
        ('single', [args.single], args.single_observer),
        ('concurrent-shared-desktop', args.concurrent, args.concurrent_observer),
    ):
        observer = read(observer_path)
        if (observer.get('status') != 'observed' or observer.get('native_related_events')
                or observer.get('native_related_sample_transitions')
                or observer.get('observer_started_games') != 0
                or observer.get('observer_sent_input') is not False
                or observer.get('observer_activated_windows') is not False):
            raise ValueError('Independent input-desktop observation failed: '+str(observer_path))
        start = datetime.fromisoformat(observer['started_utc']).timestamp()
        end = datetime.fromisoformat(observer['ended_utc']).timestamp()
        seen = {row['pid']: row for row in observer['existing_native_processes']}
        rows = []
        desktops = set()
        for path in paths:
            report = read(path)
            identity = report['owned_identity']
            pid, created = identity['pid'], identity['create_time']
            if pid in all_pids:
                raise ValueError('Use separate observed native launches')
            all_pids.add(pid)
            if (report.get('status') != 'passed_runtime_checks'
                    or report.get('frames_advanced', 0) < 64
                    or not report.get('cleanup', {}).get('exact_owned_exit_confirmed')):
                raise ValueError('Actual native advancement/cleanup failed: '+str(path))
            validate_rendering_provenance(report)
            render_backends.add((report['final'].get('execution') or {}).get('render_backend') or 'nullref-v1')
            if not start < created < end or pid not in seen or abs(seen[pid]['create_time']-created) > .01:
                raise ValueError('Native launch was not covered by the independent observer')
            isolation = report['final']['native_desktop_isolation']
            name = isolation['desktop_name']
            if (isolation.get('main_thread_desktop') != name
                    or not isolation.get('main_thread_assignment_verified')
                    or isolation.get('input_desktop_after_checkpoint') == name
                    or isolation.get('desktop_switched')
                    or isolation.get('default_desktop_fallback')):
                raise ValueError('Actual private non-input desktop binding was not verified')
            for relative, expected in report['source_hashes'].items():
                if digest(ROOT/relative) != expected:
                    raise ValueError('Source changed after native qualification: '+relative)
            desktops.add(name)
            rows.append(dict(report=str(path.resolve()), sha256=digest(path),
                             owned_identity=identity, frames_advanced=report['frames_advanced']))
        if label.startswith('concurrent') and (len(paths) < 3 or len(desktops) != 1):
            raise ValueError('Require at least three actual starts on one shared private desktop')
        groups.append(dict(scope=label, launches=rows, desktops=sorted(desktops),
                           observer=str(observer_path.resolve()), observer_sha256=digest(observer_path),
                           native_foreground_or_focus_events=0))
    result = dict(status='passed', schema='th14-private-desktop-focus-qualification-v1',
        render_backend=(next(iter(render_backends)) if len(render_backends)==1 else 'mixed'),
        render_backends=sorted(render_backends),
        groups=groups, native_launches=len(all_pids), desktop_switched=False,
        source_sha256=qualified_source_hashes(),
        limitation='Finite original-game startup/advance/cleanup qualification; not 100-instance pressure or learning efficacy.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), 'utf-8')
    print(json.dumps(dict(status='passed', native_launches=len(all_pids), output=str(args.output))))


if __name__ == '__main__':
    main()
