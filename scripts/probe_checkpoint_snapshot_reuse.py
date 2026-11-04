"""Default validate-only plan for one bounded same-checkpoint read comparison.

No training, rendering/pacing patch, GPU learner, multi-world or soak mode.
Actual execution needs --execute, the reviewed manifest and an idle-project
guard. Never run concurrently with collection/training/development jobs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
FULL_SHA256 = 'f08e885e28d2e0fc8c579e88034dd7de8f2a648d8f08ce781cf08eadc9ffe353'
SOURCE_FILES = ('scripts/probe_checkpoint_snapshot_reuse.py','scripts/background_job.py',
    'src/fly_touhou/__init__.py','src/fly_touhou/background.py','src/fly_touhou/campaign.py',
    'src/fly_touhou/checkpoint_snapshot_reuse.py', 'src/fly_touhou/fixed_start_controller.py',
    'src/fly_touhou/geometry_environment.py', 'src/fly_touhou/hazard_environment.py',
    'src/fly_touhou/geometry_observation.py','src/fly_touhou/hazard_observation.py',
    'src/fly_touhou/isolated_campaign.py','src/fly_touhou/native_geometry.py',
    'src/fly_touhou/observation.py','src/fly_touhou/owned_window_visibility.py',
    'src/fly_touhou/parallel_campaign.py','src/fly_touhou/planner.py',
    'src/fly_touhou/resource_observation.py','src/fly_touhou/responsive_campaign.py',
    'src/fly_touhou/hazard_reader.py', 'src/fly_touhou/responsive_controller.py',
    'src/fly_touhou/parallel_controller.py', 'src/fly_touhou/dialogue_controller.py',
    'src/fly_touhou/resource_environment.py', 'src/fly_touhou/frida_controller.py',
    'src/fly_touhou/native_performance.py', 'src/fly_touhou/native.py',
    'src/fly_touhou/resources.py', 'src/fly_touhou/state.py',
    'src/fly_touhou/hooks/th14.js','src/fly_touhou/hooks/th14_responsive.js',
    'src/fly_touhou/hooks/th14_parallel_startup.js')


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def file_evidence(path):
    path = Path(path).resolve(strict=True)
    return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'bytes':path.stat().st_size}


def save_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(path)


class ProbeStopped(Exception):
    pass


class Budget:
    def __init__(self, seconds, stop_file=None, clock=time.monotonic):
        self.clock, self.started, self.total_seconds = clock,clock(),seconds
        self.work_deadline = self.started+seconds-30  # Reserve owned-child cleanup.
        self.stop_file, self.cancelled = stop_file,False

    def check(self):
        if self.cancelled or (self.stop_file is not None and self.stop_file.exists()):
            raise ProbeStopped('operator STOP')
        if self.clock() >= self.work_deadline:
            raise ProbeStopped('cooperative work deadline; cleanup reserve retained')


class HardDeadline:
    """Final deadline exits only this future probe host, closing its Job handles."""
    def __init__(self, seconds, marker, exit_process=os._exit):
        self.marker,self.exit_process,self.finished = marker,exit_process,False
        self.lock = threading.Lock()
        self.timer = threading.Timer(seconds,self.expire)
        self.timer.daemon = True

    def start(self):
        self.timer.start()

    def expire(self):
        with self.lock:
            if self.finished:
                return
            try:
                save_json(self.marker,{'status':'hard-wall-budget-exit','utc':utc_now(),
                    'only_probe_host_exited':True,'passed':False})
            finally:
                # The controller's existing KILL_ON_JOB_CLOSE owns its game;
                # no generic th14/Python process enumeration or killing occurs.
                self.exit_process(4)

    def finish(self):
        with self.lock:
            self.finished = True
            self.timer.cancel()


class CallCounters:
    """Instance-only instrumentation; preserve actual reader/memory identity."""
    def __init__(self):
        self.values = dict(raw_calls=0,memory_reads=0,memory_bytes=0)
        self.installed = []

    def attach(self, controller):
        controller.invalidate_snapshot_reuse('probe counter installation')
        for obj,kind in ((controller.reader,'raw_calls'),(controller._memory,'memory_reads')):
            if any(entry[0] is obj for entry in self.installed):
                continue
            original = obj.read
            present,prior = 'read' in vars(obj),vars(obj).get('read')
            def counted(*args,_original=original,_kind=kind,**kwargs):
                self.values[_kind] += 1
                if _kind == 'memory_reads':
                    self.values['memory_bytes'] += int(args[1] if len(args)>1 else kwargs['size'])
                return _original(*args,**kwargs)
            obj.read = counted
            self.installed.append((obj,present,prior,counted))

    def snapshot(self):
        return dict(self.values)

    def delta(self, before):
        return {key:self.values[key]-value for key,value in before.items()}

    def restore(self):
        errors=[]
        for obj,present,prior,wrapper in reversed(self.installed):
            if obj.read is not wrapper:
                errors.append('Probe instrumentation changed unexpectedly')
                continue
            if present:
                obj.read = prior
            else:
                del obj.read
        self.installed.clear()
        if errors:
            raise RuntimeError('; '.join(errors))


def first_difference(left, right, path='$'):
    """Strict field/type/value equality, without numeric tolerances or omissions."""
    if type(left) is not type(right):
        return {'path':path,'reason':'type','left':repr(type(left)),'right':repr(type(right))}
    if isinstance(left,dict):
        if left.keys() != right.keys():
            return {'path':path,'reason':'keys','left':sorted(left),'right':sorted(right)}
        for key in sorted(left):
            mismatch = first_difference(left[key],right[key],f'{path}.{key}')
            if mismatch:
                return mismatch
    elif isinstance(left,(list,tuple)):
        if len(left) != len(right):
            return {'path':path,'reason':'length','left':len(left),'right':len(right)}
        for index,(a,b) in enumerate(zip(left,right)):
            mismatch = first_difference(a,b,f'{path}[{index}]')
            if mismatch:
                return mismatch
    elif isinstance(left,float) and (not math.isfinite(left) or not math.isfinite(right)):
        return {'path':path,'reason':'nonfinite'}
    elif isinstance(left,float) and left==right==0 and math.copysign(1,left)!=math.copysign(1,right):
        return {'path':path,'reason':'signed-zero'}
    elif left != right:
        return {'path':path,'reason':'value','left':left,'right':right}
    return None


def checkpoint_identity(controller):
    event = controller._checkpoint
    if (controller._closed or not isinstance(event,dict) or event.get('type')!='checkpoint'
            or isinstance(event.get('sequence'),bool) or not isinstance(event.get('sequence'),int)
            or event['sequence']<1):
        raise RuntimeError('No valid parked IPC checkpoint')
    return (controller.pid,id(controller.reader),id(controller._memory),
            id(controller._script),id(event),event['sequence'])


def assert_sentinel(controller, identity, frame):
    if checkpoint_identity(controller) != identity or controller.reader.global_frame() != frame:
        raise RuntimeError('Parked checkpoint identity or native frame changed')


def checked_snapshot(snapshot):
    snapshot.validate()
    if (snapshot.stage!=1 or snapshot.difficulty!=3 or snapshot.character!=0 or snapshot.subshot!=0
            or any(snapshot.extra.get(k)!=0 for k in ('continues','run_mode','replay_playback'))):
        raise RuntimeError('Probe left original live Lunatic/ReimuA/no-continue first-stage conditions')


def uncached_read(controller):
    from fly_touhou.checkpoint_snapshot_reuse import CheckpointSnapshotReuseMixin
    controller.invalidate_snapshot_reuse('ordinary complete controller read reference')
    # Unlike read_fresh(), this is the actual pre-candidate controller read
    # chain, without the candidate's extra defensive copies or frame probes.
    return super(CheckpointSnapshotReuseMixin,controller).read()


def measured_read(controller, read, counters, budget):
    budget.check()
    before = counters.snapshot()
    started = time.perf_counter()
    snapshot = read(controller)
    elapsed = time.perf_counter()-started
    return snapshot,{'seconds':elapsed,**counters.delta(before)}


def compare_window(controller, counters, budget, index, reads, output=None):
    reference = uncached_read(controller)
    checked_snapshot(reference)
    expected,identity = reference.to_dict(),checkpoint_identity(controller)
    result = {'window':index,'frame':reference.frame,'stage_frame':reference.stage_frame,
        'checkpoint_sequence':identity[-1],'bullets':len(reference.bullets),
        'lasers':len(reference.lasers),'enemies':len(reference.enemies),
        'has_native_projectiles':bool(reference.bullets or reference.lasers),
        'near_player_active_bullets_within_96':sum(
            bool(reference.player and bullet.active and not bullet.intangible and
                 math.hypot(bullet.x-reference.player.x,bullet.y-reference.player.y)-bullet.radius<=96)
            for bullet in reference.bullets),
        'player_state':None if reference.player is None else reference.player.state,
        'lives':reference.lives,
        'order':['uncached','cached'] if index%2==0 else ['cached','uncached'],
        'conditions':{},'all_fields_compared':True,'sentinel_counts':{}}
    if output is not None:
        with gzip.open(output/f'window-{index}-reference.json.gz','wt',encoding='utf-8') as stream:
            json.dump(expected,stream,ensure_ascii=False,allow_nan=False)
    for condition in result['order']:
        budget.check()
        if condition=='cached':
            controller.invalidate_snapshot_reuse('explicit cold-cache warm-up')
            warm,warm_cost = measured_read(controller,lambda c:c.read(),counters,budget)
            mismatch=first_difference(expected,warm.to_dict())
            if mismatch:
                raise RuntimeError('Warm-up field mismatch: '+json.dumps(mismatch,ensure_ascii=False))
            result['cache_warmup']=warm_cost
        samples=[]
        for _ in range(reads):
            before_sentinel=counters.snapshot()
            assert_sentinel(controller,identity,reference.frame)
            row,metrics=measured_read(controller,uncached_read if condition=='uncached' else lambda c:c.read(),counters,budget)
            mismatch=first_difference(expected,row.to_dict())
            if mismatch:
                raise RuntimeError('Complete Snapshot mismatch: '+json.dumps(mismatch,ensure_ascii=False))
            assert_sentinel(controller,identity,reference.frame)
            all_cost=counters.delta(before_sentinel)
            for key in all_cost:
                result['sentinel_counts'][key]=result['sentinel_counts'].get(key,0)+all_cost[key]-metrics[key]
            if metrics['raw_calls'] != (1 if condition=='uncached' else 0):
                raise RuntimeError('Unexpected complete-reader invocation count')
            samples.append(metrics)
        result['conditions'][condition]={'samples':samples,
            'total_seconds':sum(v['seconds'] for v in samples),
            'mean_seconds':statistics.mean(v['seconds'] for v in samples),
            'raw_calls':sum(v['raw_calls'] for v in samples),
            'memory_reads':sum(v['memory_reads'] for v in samples),
            'memory_bytes':sum(v['memory_bytes'] for v in samples)}
    # A final real read also detects native same-frame field changes hidden by
    # repeated cache hits; no native/UI provenance field is silently removed.
    final=uncached_read(controller)
    mismatch=first_difference(expected,final.to_dict())
    if mismatch:
        raise RuntimeError('End-of-window fresh Snapshot mismatch: '+json.dumps(mismatch,ensure_ascii=False))
    assert_sentinel(controller,identity,reference.frame)
    return result


def active_conflicts(processes, own_pid, game_name):
    blockers=[]
    for item in processes:
        if item.get('pid')==own_pid:
            continue
        name=(item.get('name') or '').lower()
        reason=None
        if name==game_name.lower():
            reason='an original TH14 process already exists'
        elif name.startswith('python'):
            command=' '.join(item.get('cmdline') or []).lower().replace('\\','/')
            if 'fly-touhou' in command:
                reason='another project Python job exists'
        if reason:
            blockers.append({'pid':item['pid'],'name':name,'reason':reason})
    return blockers


def canonical_path(value):
    if not isinstance(value,(str,Path)) or not str(value):
        return None
    return str(Path(value).resolve()).replace('\\','/').casefold()


def command_script_tail(command, script):
    """Return exact arguments after this script, allowing Python -u/-X flags."""
    if not command or any(not isinstance(v,str) for v in command):
        return None
    expected=canonical_path(script)
    positions=[index for index,value in enumerate(command[1:],1)
               if not value.startswith('-') and canonical_path(value)==expected]
    return command[positions[0]+1:] if len(positions)==1 else None


def background_worker_tail(command):
    # A frozen probe may be started by the workspace's identical launcher.
    # File name alone is insufficient: verify the actual launcher bytes.
    if not command or any(not isinstance(v,str) for v in command):
        return None
    known=ROOT/'scripts/background_job.py'
    for index,value in enumerate(command[1:],1):
        if Path(value).name.casefold()=='background_job.py':
            try:
                if file_evidence(value)['sha256']==file_evidence(known)['sha256']:
                    return command[index+1:]
            except OSError:
                return None
    return None


def probe_owner_chain(processes, own_pid, probe_script, probe_arguments,
                      launcher_executable, base_executable):
    """Exempt only verifiable direct owners, never arbitrary Python ancestors.

    Windows venv python.exe can launch a base Python whose argv[0] differs.
    Exact image pair, full argument tail, parent PID and creation-time order
    must all agree. Unknown/missing evidence fails closed.
    """
    by_pid={item['pid']:item for item in processes}
    child=by_pid.get(own_pid)
    owned={own_pid}
    if child is None or command_script_tail(child.get('cmdline'),probe_script)!=probe_arguments:
        return owned
    launcher,base=canonical_path(launcher_executable),canonical_path(base_executable)
    while len(owned)<8:
        parent=by_pid.get(child.get('ppid'))
        if parent is None or parent['pid'] in owned:
            break
        times=(parent.get('create_time'),child.get('create_time'))
        if (any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0
                for v in times) or times[0]>times[1]):
            break  # A reused parent PID does not establish ownership.
        parent_cmd,child_cmd=parent.get('cmdline') or [],child.get('cmdline') or []
        venv_parent=(launcher and base and launcher!=base and parent_cmd and child_cmd
            and canonical_path(parent.get('exe'))==launcher
            and canonical_path(child.get('exe'))==base
            and canonical_path(parent_cmd[0])==launcher
            and canonical_path(child_cmd[0])==base
            and parent_cmd[1:]==child_cmd[1:])
        background_parent=False
        if not venv_parent and command_script_tail(child_cmd,probe_script)==probe_arguments:
            # This exact background_job worker must declare this exact probe
            # invocation. A trainer/spawner or a different probe spec is not
            # exempt merely because it is an ancestor or names the same file.
            worker_tail=background_worker_tail(parent_cmd)
            if worker_tail and len(worker_tail)==3 and '--worker' in worker_tail and '--spec' in worker_tail:
                try:
                    index=worker_tail.index('--spec')
                    spec_path=worker_tail[index+1]
                    if sorted(worker_tail[:index]+worker_tail[index+2:])!=['--worker']:
                        break
                    spec=json.loads(Path(spec_path).read_text(encoding='utf-8-sig'))
                    background_parent=(spec.get('schema')=='fly-touhou-background-job-v1'
                        and canonical_path(spec.get('script'))==canonical_path(probe_script)
                        and spec.get('arguments')==probe_arguments
                        and canonical_path(child.get('exe')) is not None
                        and canonical_path(child.get('exe')) in {launcher,base}
                        and canonical_path(spec.get('python'))==canonical_path(child.get('exe'))
                        and canonical_path(child_cmd[0])==canonical_path(child.get('exe'))
                        and canonical_path(parent.get('exe')) in {launcher,base}
                        and canonical_path(parent_cmd[0])==canonical_path(parent.get('exe')))
                except (OSError,ValueError,TypeError,KeyError,IndexError):
                    pass
        if not (venv_parent or background_parent):
            break
        owned.add(parent['pid'])
        child=parent
    return owned


def require_idle(game_name):
    import psutil
    processes=[]
    for process in psutil.process_iter(['pid','ppid','name','cmdline','exe','create_time'],ad_value=None):
        try:
            processes.append(process.info)
        except (psutil.NoSuchProcess,psutil.AccessDenied):
            continue
    owned=probe_owner_chain(processes,os.getpid(),Path(__file__),sys.argv[1:],
        sys.executable,getattr(sys,'_base_executable',sys.executable))
    blockers=active_conflicts([v for v in processes if v.get('pid') not in owned],
        os.getpid(),game_name)
    if blockers:
        raise ProbeStopped('refusing concurrent originals/project jobs: '+json.dumps(blockers))


def make_controller(protocol, output, budget, timeout):
    from fly_touhou.checkpoint_snapshot_reuse import CheckpointSnapshotReuseMixin
    from fly_touhou.fixed_start_controller import HiddenFineStartDialogueGeometryController

    class ProbeBudgetMixin:
        def __init__(self,*args,probe_budget=None,**kwargs):
            self.probe_budget=probe_budget or getattr(self,'probe_budget',None)
            super().__init__(*args,**kwargs)
        def read(self):
            self.probe_budget.check()
            return super().read()
        def advance(self,*args,**kwargs):
            self.probe_budget.check()
            return super().advance(*args,**kwargs)
        def _wait(self,expected):
            self.probe_budget.check()
            previous=self.timeout
            self.timeout=min(previous,max(.01,self.probe_budget.work_deadline-time.monotonic()))
            try:
                return super()._wait(expected)
            finally:
                self.timeout=previous

    class ProbeController(CheckpointSnapshotReuseMixin,ProbeBudgetMixin,
                          HiddenFineStartDialogueGeometryController):
        pass

    return ProbeController(protocol['game_exe'],protocol['profile'],
        private_appdata=output/'userdata',worker_name='isolated-cache-read-probe',
        timeout=timeout,probe_budget=budget,checkpoint_snapshot_reuse=True,
        checkpoint_wait_sleep_ms=10,checkpoint_timing=False)


def align_start(controller):
    snapshot=controller.read()
    if not 0<=snapshot.stage_frame<=12:
        raise RuntimeError('Fresh native start is already past declared stage12')
    if snapshot.stage_frame<12:
        snapshot=controller.advance(1,12-snapshot.stage_frame)
    checked_snapshot(snapshot)
    if snapshot.stage_frame!=12 or snapshot.player is None:
        raise RuntimeError('Probe did not align to initialized stage12')
    return snapshot


def prepare(args):
    protocol=json.loads(args.protocol.read_text(encoding='utf-8-sig'))
    game,profile=file_evidence(protocol['game_exe']),file_evidence(protocol['profile'])
    if game['sha256']!=FULL_SHA256 or protocol.get('game_exe_sha256')!=FULL_SHA256:
        raise ValueError('Expected exact pinned TH14 1.00b original executable')
    profile_data=json.loads(Path(profile['path']).read_text(encoding='utf-8-sig'))
    if game['sha256'] not in profile_data.get('sha256',[]) or profile_data.get('game_id')!=14:
        raise ValueError('Profile does not pin the selected original TH14 executable')
    if args.output is not None:
        private=args.output.resolve()/'userdata'
        if len(str(private))>210 or not str(private).isascii():
            raise ValueError('Probe private APPDATA requires an ASCII path shorter than 211 characters')
        if args.output.exists():
            raise FileExistsError('Preserve existing runtime output; choose a new output directory')
    source_manifest=[{'relative_path':name,**file_evidence(ROOT/name)} for name in SOURCE_FILES]
    return {'schema':'same-checkpoint-snapshot-reuse-probe-v1','status':'prepared-not-run',
        'created_utc':utc_now(),'source_root':str(ROOT),'source_manifest':source_manifest,
        'interpreter':{'launcher':file_evidence(sys.executable),
            'base':file_evidence(getattr(sys,'_base_executable',sys.executable)),
            'version':sys.version},
        'protocol':file_evidence(args.protocol),'game_exe':game,'profile':profile,
        'configuration':{'windows':args.windows,'reads_per_condition':args.reads_per_condition,
            'advance_decisions_between_windows':args.advance_decisions_between_windows,
            'max_wall_seconds':args.max_wall_seconds,'cleanup_reserve_seconds':30,
            'ipc_timeout_seconds':args.ipc_timeout_seconds,'worlds':1,'mask':1,'frame_skip':4,
            'start_stage_frame':12,'difficulty':3,'native_wait_sleep_ms':10,'native_timing':False,
            'output':None if args.output is None else str(args.output.resolve()),
            'stop_file':None if args.stop_file is None else str(args.stop_file.resolve())},
        'native_game_started':False,'GPU_learner_started':False,
        'requires_idle_project_before_execute':True,'end_to_end_mode_available':False,
        'rng_seed_applied':False,'cross_run_same_trajectory_claimed':False,
        'native_frame_pacing_modified':False,'long_stability_test':False,
        'limits':['Repeated-read speed is not end-to-end rollout/PPO speed.',
            'Counters add equal instance-only instrumentation to both conditions.',
            'Warm-up, sentinels, comparison, startup/restart/cleanup costs are separate from timed read spans.',
            'Lifecycle/source/STOP failure cannot be labeled a passed read comparison.'],
        'windows':[]}


def verify_manifest(plan, manifest_path):
    expected=json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    for field in ('protocol','game_exe','profile'):
        if plan[field]['sha256']!=expected[field]['sha256']:
            raise ValueError('Reviewed '+field+' hash differs')
    if plan['interpreter']!=expected['interpreter']:
        raise ValueError('Reviewed Python interpreter identity differs')
    current={v['relative_path']:v['sha256'] for v in plan['source_manifest']}
    frozen={v['relative_path']:v['sha256'] for v in expected['source_manifest']}
    if current!=frozen or plan['configuration']!=expected['configuration']:
        raise ValueError('Reviewed source manifest or configuration differs')


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--manifest',type=Path)
    parser.add_argument('--plan-out',type=Path)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--execute',action='store_true')
    mode.add_argument('--validate-only',action='store_true')
    parser.add_argument('--windows',type=int,default=3)
    parser.add_argument('--reads-per-condition',type=int,default=8)
    parser.add_argument('--advance-decisions-between-windows',type=int,default=64)
    parser.add_argument('--max-wall-seconds',type=float,default=300)
    parser.add_argument('--ipc-timeout-seconds',type=float,default=15)
    parser.add_argument('--stop-file',type=Path)
    args=parser.parse_args(argv)
    if args.stop_file is None and args.output is not None:
        args.stop_file=args.output.resolve()/'STOP'
    if not 1<=args.windows<=3 or not 1<=args.reads_per_condition<=16:
        parser.error('At most 3 checkpoint windows, 1..16 reads per condition')
    if not 1<=args.advance_decisions_between_windows<=64:
        parser.error('Use 1..64 fixed exact4 decisions between windows')
    if not 60<=args.max_wall_seconds<=540 or not 5<=args.ipc_timeout_seconds<=15:
        parser.error('Total wall limit 60..540s, IPC 5..15s')
    if args.execute and (args.output is None or args.manifest is None):
        parser.error('--execute requires a new --output and the reviewed --manifest')
    return args


def run(args, plan):
    import psutil
    from fly_touhou.native import NativeError
    require_idle(Path(plan['game_exe']['path']).name)
    output=args.output.resolve()
    if args.stop_file is not None and args.stop_file.exists():
        raise ProbeStopped('preserved STOP marker already exists')
    output.mkdir(parents=True,exist_ok=False)
    budget=Budget(args.max_wall_seconds,args.stop_file)
    watchdog=HardDeadline(args.max_wall_seconds,output/'hard-budget-exit.json')
    watchdog.start()
    controller=None
    counters=CallCounters()
    previous={signum:signal.signal(signum,lambda _s,_f:setattr(budget,'cancelled',True))
              for signum in (signal.SIGINT,signal.SIGTERM)}
    report=plan
    report.update(status='running',host_pid=os.getpid(),started_utc=utc_now(),owned_processes=[])
    save_json(output/'report.json',report)
    exit_code=0
    try:
        controller=make_controller({'game_exe':plan['game_exe']['path'],'profile':plan['profile']['path']},
            output,budget,args.ipc_timeout_seconds)
        controller.start()
        report['native_game_started']=True
        report['owned_processes'].append({'pid':controller.pid,
            'create_time':psutil.Process(controller.pid).create_time()})
        controller.ensure_run(3)
        align_start(controller)
        counters.attach(controller)
        for index in range(args.windows):
            if index:
                for _ in range(args.advance_decisions_between_windows):
                    budget.check()
                    before=controller.read()
                    after=controller.advance(1,4)
                    if after.frame-before.frame!=4:
                        raise RuntimeError('Exact4 advance violated')
                    checked_snapshot(after)
            report['windows'].append(compare_window(controller,counters,budget,index,
                args.reads_per_condition,output))
            save_json(output/'report.json',report)
        epoch=controller.snapshot_reuse_audit()['epoch']
        controller.restart()
        report['owned_processes'].append({'pid':controller.pid,
            'create_time':psutil.Process(controller.pid).create_time()})
        align_start(controller)
        counters.attach(controller)
        if controller.snapshot_reuse_audit()['epoch']<=epoch:
            raise RuntimeError('Restart did not invalidate snapshot lifecycle')
        # Restarted controller must first read natively, then hit its new cache.
        fresh,miss=measured_read(controller,lambda c:c.read(),counters,budget)
        same,hit=measured_read(controller,lambda c:c.read(),counters,budget)
        if miss['raw_calls']!=1 or hit['raw_calls']!=0 or first_difference(fresh.to_dict(),same.to_dict()):
            raise RuntimeError('Restart cache invalidation/parity failed')
        report['lifecycle_restart']={'passed':True,'first_read':miss,'second_read':hit}
        report['candidate_audit']=controller.snapshot_reuse_audit()
        report['status']='passed-bounded-same-checkpoint-read-probe'
    except ProbeStopped as error:
        report.update(status='stopped',stop_reason=str(error))
        exit_code=2
    except BaseException as error:
        report.update(status='failed',error=repr(error))
        exit_code=1
    finally:
        cleanup_errors=[]
        try:
            counters.restore()
        except BaseException as error:
            cleanup_errors.append(repr(error))
        try:
            if controller is not None:
                if controller.pid is not None:
                    # start() can spawn an owned original and then fail before
                    # returning. Never mislabel that partial start as no game.
                    report['native_game_started']=True
                    try:
                        identity={'pid':controller.pid,'create_time':psutil.Process(controller.pid).create_time()}
                        if identity not in report['owned_processes']:
                            report['owned_processes'].append(identity)
                    except psutil.NoSuchProcess:
                        pass
                controller.close()
                try:
                    controller.read()
                except NativeError:
                    report['closed_controller_rejected_cached_read']=True
                else:
                    raise RuntimeError('Closed controller returned a snapshot')
            alive=[]
            for item in report['owned_processes']:
                try:
                    process=psutil.Process(item['pid'])
                    if abs(process.create_time()-item['create_time'])<.001 and process.is_running():
                        alive.append(item)
                except psutil.NoSuchProcess:
                    pass
            report['owned_processes_still_alive']=alive
            if alive:
                raise RuntimeError('Owned probe process remains alive after close')
        except BaseException as error:
            cleanup_errors.append(repr(error))
        if cleanup_errors:
            report.update(status='failed-cleanup',cleanup_errors=cleanup_errors)
            exit_code=1
        report.update(finished_utc=utc_now(),total_wall_seconds=time.monotonic()-budget.started,
            instrumented_total_counts=counters.snapshot(),end_to_end_speedup_claimed=False)
        changed=[]
        for value in plan['source_manifest']:
            try:
                if file_evidence(ROOT/value['relative_path'])['sha256']!=value['sha256']:
                    changed.append(value['relative_path'])
            except OSError:
                changed.append(value['relative_path'])
        report['changed_sources']=changed
        if changed:
            report['status']='invalidated-source-changed'
            exit_code=1
        report['observed_projectile_windows']=sum(v['has_native_projectiles'] for v in report['windows'])
        report['coverage_status']=('native-projectiles-observed' if report['observed_projectile_windows']
            else 'empty-projectile-windows-only')
        report['dangerous_scene_coverage_claimed']=False
        if report['windows']:
            report['same_checkpoint_read_summary']={condition:{
                'checkpoint_blocks':len(report['windows']),
                'median_block_mean_seconds':statistics.median(v['conditions'][condition]['mean_seconds'] for v in report['windows']),
                'min_block_mean_seconds':min(v['conditions'][condition]['mean_seconds'] for v in report['windows']),
                'max_block_mean_seconds':max(v['conditions'][condition]['mean_seconds'] for v in report['windows']),
                'total_raw_calls':sum(v['conditions'][condition]['raw_calls'] for v in report['windows']),
                'total_memory_reads':sum(v['conditions'][condition]['memory_reads'] for v in report['windows'])}
                for condition in ('uncached','cached')}
        if report['status'].startswith('passed') and not report['observed_projectile_windows']:
            report['status']='passed-empty-scene-read-parity-only'
        save_json(output/'report.json',report)
        for signum,handler in previous.items():
            signal.signal(signum,handler)
        watchdog.finish()
    return exit_code,report


def main(argv=None):
    args=parse_args(argv)
    plan=prepare(args)
    if args.plan_out is not None:
        if args.plan_out.exists():
            raise FileExistsError('Preserve previous plan; select a new --plan-out')
        args.plan_out.parent.mkdir(parents=True,exist_ok=True)
        save_json(args.plan_out,plan)
    if not args.execute:
        print(json.dumps(plan,ensure_ascii=False,indent=2))
        return 0
    verify_manifest(plan,args.manifest)
    try:
        code,report=run(args,plan)
    except ProbeStopped as error:
        print(json.dumps({'status':'blocked-before-launch','reason':str(error),
            'native_game_started':False,'GPU_learner_started':False}))
        return 2
    print(json.dumps({'status':report['status'],'report':str(args.output/'report.json')},ensure_ascii=False))
    return code


if __name__=='__main__':
    raise SystemExit(main())
