"""Inspect native Spell Practice through process-local menu input, without a policy."""
from pathlib import Path
import argparse,json,os,shutil,sys,time,traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--score',type=Path)
    parser.add_argument('--seed',type=int,default=4747)
    parser.add_argument('--stage',type=int,default=1)
    parser.add_argument('--spell-index',type=int,default=0,help='Native spell row within the selected stage')
    parser.add_argument('--play-frames',type=int,default=0,help='Optional movement-teacher probe after entry; no network training')
    parser.add_argument('--teacher',choices=('geometry','resource'),default='geometry')
    a=parser.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        FLY_TOUHOU_DESKTOP_NAME='fly-touhou-job-spell-menu-'+str(os.getpid()))
    from fly_touhou.thrl_job_paths import configure_native_temp,cleanup_native_temp
    from fly_touhou.runtime_cache_paths import configure_generated_caches
    from fly_touhou.thrl_durable_records import write_json
    from fly_touhou.thrl_native_events import HiddenTHRLDialogueGeometryController
    configure_generated_caches();temporary=configure_native_temp(a.output)
    cfg=json.loads(a.config.read_text(encoding='utf-8-sig'))
    appdata=a.output/'appdata'
    if a.score:
        target=appdata/'ShanghaiAlice/th14/scoreth14.dat';target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(a.score,target)
    controller=HiddenTHRLDialogueGeometryController(cfg['game_exe'],cfg['profile'],timeout=25,
        private_appdata=appdata,worker_name='spell-menu',headless=True,native_seed=a.seed,
        binary_reader=True)
    rows=[];last=None;begun=time.monotonic()
    try:
        s=controller.start()
        for step in range(300):
            e=s.extra
            key=(s.screen,e.get('menu_page'),e.get('menu_substate'),e.get('menu_selection'))
            if key!=last:
                row=dict(step=step,screen=s.screen,page=key[1],substate=key[2],selection=key[3],
                    run_mode=e.get('run_mode'),difficulty=s.difficulty,stage=s.stage,stage_frame=s.stage_frame)
                # The native selection-to-spell table is documented by thprac
                # thprac/src/thprac_th14.cpp, THGuiSP::State (0x4db6a4+0x5d70).
                if key[1]==19:
                    memory=controller._memory;menu=memory.integer(memory.base+controller.hooks['menu_pointer_rva'])
                    row['difficulty_spell_ids']=[memory.integer(menu+0x5d70+4*i,signed=True) for i in range(4)]
                rows.append(row);last=key
                write_json(a.output/'menu-events.json',rows)
            if s.screen==7 and s.player is not None and s.player.state==1:
                from fly_touhou.nbnm_sensation import current_attack
                from fly_touhou.nbnm_task import attack_key
                for tick in range(400):
                    key_attack=attack_key(current_attack(s))
                    if key_attack and key_attack[1]:break
                    s=controller.advance(0,1)
                e=s.extra
                write_json(a.output/'entry.json',dict(run_mode=e.get('run_mode'),stage=s.stage,
                    stage_frame=s.stage_frame,difficulty=s.difficulty,character=s.character,subshot=s.subshot,
                    player=dict(x=s.player.x,y=s.player.y,state=s.player.state,protection=s.player.invulnerable_frames),
                    power=s.power,lives=s.lives,bombs=s.bombs,attack=current_attack(s),extra=e,network_training=False,
                    elapsed_seconds=time.monotonic()-begun))
                if a.play_frames:
                    from fly_touhou.compact_full_story import factory,story_kwargs
                    from fly_touhou.nbnm_teacher import label_current
                    from fly_touhou.background import action_mask
                    from fly_touhou.thrl_native_env import SOURCE19_TO_NATIVE36
                    encenv=factory(story_kwargs(cfg,a.output/'encoder',1,'train',teacher=a.teacher=='resource')[0])
                    trace=[];last_phase=None
                    try:
                        for decision in range(a.play_frames//4):
                            offsets=controller.profile.offsets['spell_card'];memory=controller._memory
                            pointer=memory.integer(memory.base+offsets['spell_card_pointer'])
                            spell={k:memory.integer(pointer+offsets['zSpellCard_'+k],signed=True) for k in ('indicator','id','bonus')} if pointer else None
                            phase=(current_attack(s),s.player.state if s.player else None,s.pause,
                                s.extra.get('pause_menu_state'),s.extra.get('pause_menu_result'),str(spell))
                            if phase!=last_phase:
                                trace.append(dict(decision=decision,stage_frame=s.stage_frame,attack=phase[0],
                                    player_state=phase[1],pause=phase[2],menu_state=phase[3],menu_result=phase[4],
                                    spell=spell,misses=s.extra.get('thrl_events',{}).get('misses'),
                                    power=s.power,score=s.score,enemies=[dict(hp=e.hp,x=e.x,y=e.y) for e in s.enemies if e.boss]))
                                last_phase=phase
                            if s.pause!=2 or (s.player is not None and s.player.state==4):break
                            encenv.snapshot=s;encenv.episode_id=1
                            encenv._last_measurable_observation=encenv.encoder.encode(s)
                            encenv._last_measurable_frame=s.frame
                            if a.teacher=='resource':label=encenv._label_current({})['teacher_label']
                            else:label=label_current(encenv.encoder.last_result,s,1)
                            if label['action']==18:
                                label['action']=max(range(18),key=lambda k:label['probabilities'][k])
                            s=controller.advance(action_mask(SOURCE19_TO_NATIVE36[label['action']]),4)
                        write_json(a.output/'practice-playout.json',dict(trace=trace,teacher_actions=decision,network_training=False))
                    finally:encenv.close()
                break
            page,substate,selection=key[1:]
            if page==1 and substate==2:
                mask=1 if selection==3 else 0x20
            elif page==17 and substate==2:
                mask=1 if selection==a.stage-1 else 0x20
            elif page==18 and substate==2:
                mask=1 if selection==a.spell_index else 0x20
            elif page in (5,19) and substate==2:
                mask=1 if selection==3 else 0x20
            elif substate==2:mask=1
            elif page in (None,0):mask=1
            else:mask=0
            s=controller.tap(mask,4) if mask else controller.advance(0,8)
        write_json(a.output/'result.json',dict(native_entry=(a.output/'entry.json').exists(),
            last_menu=rows[-1] if rows else None,network_training=False,elapsed_seconds=time.monotonic()-begun))
    except BaseException:
        write_json(a.output/'failure.json',dict(error=traceback.format_exc()));raise
    finally:controller.close();cleanup_native_temp(temporary)


if __name__=='__main__':main()
