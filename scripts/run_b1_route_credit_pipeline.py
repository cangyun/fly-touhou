"""Finite goal operation: closed greedy20 -> native read check -> one30-world PPO.

No recurring monitor or native retry. Every subprocess has an explicit control
directory. A failed predecessor stops this dispatch and preserves evidence.
"""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.thrl_study_io import read_shared_json as read,utc_now
from fly_touhou.atomic_json_io import atomic_write_json as write


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def choose_parent(candidates):
    """Provisional engineering choice; never independent acceptance evidence."""
    ranks={};cohort=None
    for c in candidates:
        r=c['report'];rows=r.get('episode_results',[])
        if (r.get('episodes')!=20 or len(rows)!=20 or r.get('original_game_rng_controlled') is not True
                or r.get('training_mode')!='full-story-v1' or r.get('PPO_updates')!=0
                or any(x.get('outcome') not in ('clear','gameover') for x in rows)):
            raise ValueError('Only complete frozen original20-game policies can select a training parent')
        keys=[(x['evaluation_game'],x['native_seed']) for x in rows]
        if cohort is None:cohort=keys
        if cohort!=keys:raise ValueError('Parent comparisons must match their initial native RNG cohort')
        stages=[max(x['stage_history']) for x in rows]
        if any(type(s) is not int or not 1<=s<=6 for s in stages):raise ValueError('Invalid native stage history')
        ranks[c['name']]=(sum(x['outcome']=='clear' for x in rows),sum(stages),sum(s>=6 for s in stages),sum(s>=5 for s in stages))
    # Candidate order starts with the prior initial policy. Equal coarse
    # progress does not promote a variant using Bomb time or a few extra frames.
    selected=max(candidates,key=lambda c:ranks[c['name']])
    return selected,dict(ranks=ranks,ranking='clears,whole-cohort sum maximum stages,reach6,reach5;stable ties',
        mean_game_time_not_selection_metric=True,diagnostic_cohort_not_independent_acceptance=True)


def run(a):
    import psutil
    from fly_touhou.thrl_control import alive
    p=read(a.plan);out=Path(p['output']);out.mkdir(parents=True,exist_ok=False)
    write(out/'launch.json',dict(pid=os.getpid(),create_time=psutil.Process().create_time(),utc=utc_now()))
    started=time.monotonic()
    def stop():
        if (out/'STOP').exists() or time.monotonic()-started>4*3600:raise RuntimeError('Finite goal dispatch stop or deadline')
    def publish(phase,**kw):write(out/'status.json',{**kw,'phase':phase,'utc':utc_now(),'goal_achieved':False})
    def idle():
        return not any('th14' in (x.info['name'] or '').lower() for x in psutil.process_iter(['name']))
    def await_exit(control,output,expected_identity=None):
        absent_since=None
        while True:
            stop();status_path=Path(control)/'status.json'
            status=read(status_path) if status_path.exists() else {}
            if status.get('status') in ('failed','launcher-error') or (Path(output)/'failure.json').exists():
                raise RuntimeError('Required finite phase failed: '+str(output))
            actual=expected_identity
            if actual is None and (Path(output)/'launch.json').exists():actual=read(Path(output)/'launch.json')
            live=actual is not None and alive(actual)
            if status.get('status')=='finished':
                if status.get('exit_code')!=0:raise RuntimeError('Required phase exit nonzero')
                if not live:return
            if live:absent_since=None
            elif absent_since is None:absent_since=time.monotonic()
            elif time.monotonic()-absent_since>60:raise RuntimeError('Phase handle absent without a successful terminal receipt;no restart')
            time.sleep(10)
    def execute(command,label):
        stop()
        with (out/(label+'.stdout.log')).open('xb') as stdout,(out/(label+'.stderr.log')).open('xb') as stderr:
            result=subprocess.run(command,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,cwd=ROOT,
                creationflags=subprocess.CREATE_NO_WINDOW,check=False)
        write(out/(label+'.result.json'),dict(command=command,exit_code=result.returncode))
        if result.returncode!=0:raise RuntimeError('Finite command failed: '+label)
    helper=Path(p['training_runtime'])/'scripts/background_job.py'
    try:
        if sha(ROOT/'source_manifest.json')!=p['workflow_manifest_sha256']:raise ValueError('Pinned finite workflow source changed')
        if sha(helper)!=p['background_helper_sha256']:raise ValueError('Pinned background helper changed')
        publish('waiting_for_greedy20_and_cleanup',native_worlds_started=0)
        await_exit(p['after_control'],p['after_output'],p['after_identity'])
        if read(Path(p['after_output'])/'greedy/native_cleanup.json').get('close_errors')!=[] or not idle():
            raise RuntimeError('Greedy native pool did not close;no new phase started')
        candidates=[]
        for c in p['candidates']:
            row={**c,'report':read(c['report'])}
            if sha(c['checkpoint'])!=c['checkpoint_sha256']:raise ValueError('Candidate checkpoint changed')
            if row['report'].get('full_game_trace_capture',{}).get('model_checkpoint_sha256')!=c['checkpoint_sha256']:
                raise ValueError('Candidate report and checkpoint identity differ')
            candidates.append(row)
        selected,selection=choose_parent(candidates)
        write(out/'parent_selection.json',dict(utc=utc_now(),selected={k:v for k,v in selected.items() if k!='report'},
            **selection,evidence_reports=[dict(path=c['report'],sha256=sha(c['report'])) for c in p['candidates']]))
        publish('native_item_read_qualification',selected_parent=selected['name'])
        qcontrol=Path(p['native_control']);qout=Path(p['native_output'])
        if (qcontrol/'launch.json').exists():
            write(out/'native_qualification_existing_dispatch.json',dict(control=str(qcontrol),new_launches=0))
        else:execute([p['python'],str(helper),'--spec',str(qcontrol/'background_spec.json')],'dispatch_native_read_qualification')
        await_exit(qcontrol,qout)
        if not idle():raise RuntimeError('Qualification world not closed')
        publish('prepare_selected30_worker_training',selected_parent=selected['name'])
        runtime=Path(p['training_runtime'])
        if sha(runtime/'source_manifest.json')!=p['training_manifest_sha256']:raise ValueError('Prepared training runtime changed')
        command=[p['python'],str(runtime/'scripts/prepare_b1_route_credit_ppo.py'),
            '--parent',selected['checkpoint'],'--parent-sha256',selected['checkpoint_sha256'],
            '--base-config',p['base_config'],'--runtime',str(runtime),'--output-control',p['training_control'],
            '--training-output',p['training_output'],'--native-read-report',str(qout/'report.json'),
            '--panel-id',p['panel_job_id'],'--panel-root',p['panel_root']]
        execute(command,'prepare_selected_training')
        tc=Path(p['training_control']);config=read(tc/'config.json')
        # This gate checks only the just-modified sensor read implementation;
        # it never filters a rollout or refuses an individual PPO batch.
        verify_code=("import sys;from pathlib import Path;sys.path.insert(0,str(Path(sys.argv[2])/'src'));"
            "from fly_touhou.thrl_study_io import read_shared_json;"
            "from fly_touhou.thrl_native_read_qualification import verify;"
            "from fly_touhou.atomic_json_io import atomic_write_json;"
            "atomic_write_json(Path(sys.argv[3]),verify(read_shared_json(sys.argv[1]),Path(sys.argv[2])))")
        execute([p['python'],'-c',verify_code,str(tc/'config.json'),str(runtime),
            str(out/'native_reader_qualification_used.json')],'verify_native_read_gate')
        prior=read(p['scratch_qualification'])
        if (prior.get('status')!='qualified-scratch-only' or prior.get('scratch_Adam_steps')!=2
                or prior.get('both_frozen_risk_heads_exact') is not True or prior.get('scratch_weights_discarded') is not True):
            raise ValueError('Prior CUDA transaction is not a qualified discarded update')
        if prior.get('parent_sha256')==selected['checkpoint_sha256'] and prior.get('source_manifest_sha256')==config['source_manifest_sha256']:
            write(out/'scratch_qualification_reused.json',dict(report=p['scratch_qualification'],sha256=sha(p['scratch_qualification'])))
        else:
            publish('selected_parent_discarded_CUDA_qualification',selected_parent=selected['name'])
            execute([p['python'],str(runtime/'scripts/qualify_b1_route_credit_cuda.py'),'--config',str(tc/'config.json'),
                '--output',p['selected_scratch_output']],'selected_CUDA_qualification')
        if not idle():raise RuntimeError('Other original worlds appeared before30-worker launch')
        if (tc/'launch.json').exists():raise RuntimeError('Training already dispatched;no duplicate launch')
        publish('dispatch30_worker_CUDA_training',selected_parent=selected['name'])
        execute([p['python'],str(helper),'--spec',str(tc/'background_spec.json')],'dispatch_training')
        tout=Path(p['training_output']);deadline=time.monotonic()+90;actual=None
        while time.monotonic()<deadline:
            stop()
            if (tout/'failure.json').exists():raise RuntimeError('Selected training startup failed')
            if (tout/'launch.json').exists():
                actual=read(tout/'launch.json')
                if alive(actual):break
            time.sleep(2)
        if actual is None or not alive(actual):raise RuntimeError('Training dispatch has no verified live handle')
        receipt=dict(status='training-dispatched',utc=utc_now(),selected_parent=selected['name'],training_output=str(tout),
            training_control=str(tc),identity=actual,workers=30,device='cuda:0',
            initial_phase='fresh20 no-Bomb baseline,then30 native training worlds',goal_achieved=False,
            new_native_training_optimized_decisions_at_dispatch=0,finite_dispatch_finished_not_goal_complete=True)
        write(out/'report.json',receipt);publish('training_dispatched',**{k:v for k,v in receipt.items() if k!='goal_achieved'})
    except BaseException:
        write(out/'failure.json',dict(utc=utc_now(),traceback=traceback.format_exc(),no_automatic_retry=True,goal_achieved=False));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--plan',type=Path,required=True);run(parser.parse_args())
