"""Read all six native Spell Practice menus and confirm each Lunatic ECL entry.

Uses the user's unlocked score copy and private desktop. No policy training,
game-state warps, resource edits, or user input. Stages are independent jobs.
"""
from pathlib import Path
import argparse,json,os,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]


def discover(job):
    stage,cfg,score,folder=job;folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    os.environ['FLY_TOUHOU_DESKTOP_NAME']='fly-touhou-job-nbnm-catalog-'+str(os.getpid())
    from fly_touhou.thrl_native_events import HiddenTHRLDialogueGeometryController
    from fly_touhou.thrl_job_paths import configure_native_temp,cleanup_native_temp
    from fly_touhou.runtime_cache_paths import configure_generated_caches
    from fly_touhou.nbnm_spell_practice import copy_score,bind_controller
    from fly_touhou.nbnm_practice_access import unlock,restore
    from fly_touhou.nbnm_sensation import current_attack
    from fly_touhou.thrl_durable_records import write_json
    configure_generated_caches();tmp=configure_native_temp(folder)
    def controller(name):
        app=folder/name/'appdata';copy_score(score,app)
        return HiddenTHRLDialogueGeometryController(cfg['game_exe'],cfg['profile'],timeout=25,
            private_appdata=app,worker_name=name,headless=True,native_seed=50000,binary_reader=True)
    c=controller('menu');rows=[];pending_row=None;next_row=False;done=False;events=[];last=None;row_tries={}
    try:
        s=c.start();unlock(c)
        for _ in range(800):
            e=s.extra;page=e.get('menu_page');sub=e.get('menu_substate');sel=e.get('menu_selection')
            if (page,sub,sel)!=last:
                last=(page,sub,sel);events.append(last)
            mask=0
            if sub==2:
                if page==1:mask=1 if sel==3 else 0x20
                elif page==17:mask=1 if sel==stage-1 else 0x20
                elif page==18:
                    if next_row:
                        s=c.tap(0x20,4);next_row=False;continue
                    if any(r['spell_index']==sel for r in rows):done=True;break
                    row_tries[sel]=row_tries.get(sel,0)+1
                    if row_tries[sel]>3:
                        rows.append(dict(stage=stage,spell_index=sel,difficulty_spell_ids=[-1]*4,
                            menu_row_locked=True,lunatic_available=False))
                        next_row=True;continue
                    pending_row=sel;mask=1
                elif page==19:
                    mem=c._memory;menu=mem.integer(mem.base+c.hooks['menu_pointer_rva'])
                    ids=[mem.integer(menu+0x5d70+4*i,signed=True) for i in range(4)]
                    rows.append(dict(stage=stage,spell_index=pending_row,difficulty_spell_ids=ids))
                    write_json(folder/'menu-rows.json',rows);mask=2;next_row=True
                else:mask=1
            elif page in (None,0):mask=1
            s=c.tap(mask,4) if mask else c.advance(0,8)
        if not done:
            write_json(folder/'failure-menu.json',dict(rows=rows,events=events,screen=s.screen,page=s.extra.get('menu_page'),substate=s.extra.get('menu_substate'),selection=s.extra.get('menu_selection')))
            raise RuntimeError('Native spell menu did not wrap to its first row: stage '+str(stage))
    finally:c.close()
    try:
        for row in rows:
            spell_id=row['difficulty_spell_ids'][3]
            if spell_id<0:row['lunatic_available']=False;continue
            c=controller('row-'+str(row['spell_index']))
            try:
                # Use the measured ID to select the row. The attack name is
                # discovered here, so reuse the menu walker with its expected
                # attack check disabled by a tiny proxy only in this probe.
                s=c.start();unlock(c)
                for _ in range(320):
                    if s.screen==7 and s.player is not None and s.player.state==1:break
                    e=s.extra;page=e.get('menu_page');sub=e.get('menu_substate');sel=e.get('menu_selection')
                    target={1:3,17:stage-1,18:row['spell_index'],19:3}.get(page)
                    mask=(1 if target is None or sel==target else 0x20) if sub==2 else (1 if page in (None,0) else 0)
                    s=c.tap(mask,4) if mask else c.advance(0,8)
                from fly_touhou.nbnm_task import attack_key
                from fly_touhou.nbnm_spell_practice import spell_state
                for _ in range(600):
                    key=attack_key(current_attack(s))
                    if key and key[1] and (spell_state(c) or {}).get('id')==spell_id:break
                    s=c.advance(0,1)
                if not key or not key[1] or (spell_state(c) or {}).get('id')!=spell_id:
                    write_json(folder/'entry-error.json',dict(row=row,attack=current_attack(s),spell=spell_state(c),
                        stage=s.stage,frame=s.stage_frame,screen=s.screen,extra=s.extra))
                    raise RuntimeError('Selected native spell did not match its menu ID: '+str((stage,row['spell_index'],spell_id,current_attack(s))))
                row.update(lunatic_available=True,attack=current_attack(s),spell_id=spell_id,
                    entry_stage_frame=s.stage_frame,run_mode=s.extra['run_mode'],
                    character=s.character,subshot=s.subshot)
                write_json(folder/'cards.json',rows)
            finally:c.close()
        return rows
    finally:cleanup_native_temp(tmp)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=6);a=p.parse_args()
    from fly_touhou.nbnm_run import read
    from fly_touhou.thrl_durable_records import write_json
    spec=read(a.config);cfg=read(spec['environment_config']);a.output.mkdir(parents=True,exist_ok=True)
    from concurrent.futures import ProcessPoolExecutor,as_completed
    rows=[];begun=time.monotonic()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        jobs=[(s,cfg,spec['nbnm']['score_template'],str(a.output/f'stage{s}')) for s in range(1,7)]
        futures=[pool.submit(discover,j) for j in jobs]
        for future in as_completed(futures):
            rows.extend(future.result());write_json(a.output/'progress.json',dict(rows=len(rows),stages=sorted({r['stage'] for r in rows}),network_training=False))
    rows.sort(key=lambda r:(r['stage'],r['spell_index']))
    write_json(a.output/'catalog.json',dict(schema='th14-measured-Lunatic-spell-menu-v1',cards=rows,
        character='ReimuA',all_six_stage_menus=True,network_training=False,elapsed_seconds=time.monotonic()-begun))
    print(json.dumps(rows,ensure_ascii=False))


if __name__=='__main__':main()
