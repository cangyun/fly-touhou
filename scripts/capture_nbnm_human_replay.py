"""Capture a human .rpy through TH14's own Replay menu on a private desktop.

No network, no gameplay/RNG edit, no PPO rows. Original replay actions decide
the game. Features use the same configured sensor as the student's old281.
"""
from pathlib import Path
import argparse,json,sys,time,shutil,traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]


def enter_replay(controller, *, first_frame=6):
    controller._fine_begin('native-human-replay')
    s=controller.read()
    def playing(row):
        return (row.screen==7 and row.player is not None and row.player.state==1
            and row.stage_frame>=first_frame and row.extra.get('replay_playback')==1)
    for _ in range(600):
        if playing(s):return s
        e=s.extra;page=e.get('menu_page');sub=e.get('menu_substate');sel=e.get('menu_selection')
        if page==11 and sub==4:mask=1
        elif sub==2:mask=0x20 if page==1 and sel!=4 else 1
        else:mask=1 if page in (None,0) else 0
        for key,ticks in ((mask,4),(0,8)):
            s,ready=controller._fine_segment(key,ticks,playing)
            if ready:return s
    raise RuntimeError('TH14 Replay menu did not enter the imported first-stage replay')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--decoder-libs',type=Path)
    p.add_argument('--stop-after-stage',type=int,choices=range(1,7))
    p.add_argument('--routes-only',action='store_true',help='Capture route positions/context without recomputing neural risk features')
    p.add_argument('--stride',type=int,default=4,choices=(4,12))
    p.add_argument('--item-collection-input', action='store_true',
                   help='Record native collection observations with the movement labels')
    a=p.parse_args()
    if a.decoder_libs:sys.path.insert(0,str(a.decoder_libs))
    from fly_touhou.th14_human_replay import read_replay,movement_distribution
    from fly_touhou.nbnm_run import read,runtime,environment
    from fly_touhou.thrl_durable_records import write_json
    from fly_touhou.thrl_job_paths import cleanup_native_temp
    from fly_touhou.compact_full_story import factory,story_kwargs
    from fly_touhou.nbnm_route_atlas import phase_progress
    from fly_touhou.nbnm_demonstration_routes import scene_signature
    meta,stages=read_replay(a.replay)
    if (meta['stage_count']!=6 or meta['difficulty']!=3 or meta['shot']!=0 or meta['subshot']!=0
            or meta['spell_id']!=0xffffffff or any(s.metadata['continues'] or s.metadata['Bomb_pressed_frames'] for s in stages)):
        raise ValueError('Require a six-stage Lunatic ReimuA replay without Bomb input or continues')
    a.output.mkdir(parents=True,exist_ok=False);temporary=runtime(a.output)
    spec=read(a.config);cfg=environment(spec,evaluation=True)
    cfg['nbnm']=dict(spec['nbnm'],task=dict(mode='full_story'),entry_mode='story_prefix')
    if a.item_collection_input:
        cfg['nbnm']['item_collection_input'] = True
    env=factory(story_kwargs(cfg,a.output/'native',1,'train',teacher=False)[0])
    feature_kind = 'route-matching-only' if a.routes_only else env.encoder.schema
    write_json(a.output/'source.json',dict(source=str(a.replay.resolve()),metadata=meta,
        input_source='original TH14 Replay playback',network_training=False,PPO_rows=0,
        feature_kind=feature_kind, movement_target_order='policy18',native_sample_stride=a.stride))
    env.native_seed=None;env.private_appdata=a.output/'userdata'
    dest=env.private_appdata/'ShanghaiAlice/th14/replay/th14_01.rpy'
    dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(a.replay,dest)
    c=env._create_controller();c._checkpoint_performance_options['wait_sleep_ms']=1
    env.controller=c
    rows=[];phase_names=[];stage_reports=[];started=time.monotonic();stage_start=0
    features=[];targets=[];weights=[];input_checks=0;input_disagreements=0
    def save_stage(number):
        nonlocal stage_start
        end=len(rows)
        if end==stage_start:return
        path=a.output/f'stage{number:02d}.npz'
        part=rows[stage_start:end]
        np.savez_compressed(path,features=np.asarray(features[stage_start:end],np.float32),
            movement_targets=np.asarray(targets[stage_start:end],np.float32),
            supervision_weights=np.asarray(weights[stage_start:end],np.float32),
            position=np.asarray([r['position'] for r in part],np.float32),
            frame=np.asarray([r['frame'] for r in part],np.int32),
            stage_frame=np.asarray([r['stage_frame'] for r in part],np.int32),
            phase=np.asarray([r['phase'] for r in part],np.int16),
            phase_clock=np.asarray([r['phase_clock'] for r in part],np.int32),
            masks=np.asarray([r['masks'] for r in part],np.uint16))
        if a.routes_only:
            # Context is only for matching a demonstration to a live scene;
            # no new columns are appended to the student's observation.
            with np.load(path,allow_pickle=False) as z:payload={k:z[k].copy() for k in z.files}
            payload['scene_signature']=np.asarray([r['signature'] for r in part],np.float32)
            np.savez_compressed(path,**payload)
        stage_reports.append(dict(stage=number,file=path.name,rows=end-stage_start,
            source_native_seed=stages[number-1].metadata['native_seed'],
            initial_replay_seed=stages[0].metadata['native_seed'],native_Misses=events['misses'],
            native_Bombs=events['bombs_used'],phase_names=list(phase_names)))
        stage_start=end
        write_json(a.output/'stages.json',stage_reports)
    try:
        c.start();s=enter_replay(c);env.encoder.reset();env.encoder.set_native_episode(1)
        current_stage=s.stage;history=[current_stage];events=s.extra['thrl_events']
        write_json(a.output/'entry.json',dict(stage=s.stage,stage_frame=s.stage_frame,
            replay_playback=s.extra.get('replay_playback'),power=s.power,lives=s.lives,bombs=s.bombs))
        full_clear=False
        while len(rows)<40000:
            events=s.extra['thrl_events']
            if events['misses'] or events['bombs_used'] or (s.player is not None and s.player.state==4):
                raise RuntimeError('Original replay desynchronized or contains a native contact/Bomb: '+str(dict(stage=s.stage,stage_frame=s.stage_frame,misses=events['misses'],bombs=events['bombs_used'])))
            if s.stage!=current_stage and s.stage>0:
                save_stage(current_stage);current_stage=s.stage;history.append(current_stage)
                if a.stop_after_stage and current_stage>a.stop_after_stage:break
            full_clear=bool(s.stage==6 and (s.extra.get('game_thread_flags',0)&0x4000 or
                any(x.get('stage')==6 and x.get('ending') for x in s.extra.get('native_milestones',()))))
            if full_clear:save_stage(current_stage);break
            if s.extra.get('replay_playback')!=1:
                raise RuntimeError('Replay ended before an observed native six-stage clear')
            observed=(s.player is not None and s.extra.get('resources',{}).get('observed') is True
                and s.extra['thrl_events'].get('ecl_roles',{}).get('observed') is True)
            replay_stage=stages[s.stage-1]
            clock=int(s.stage_frame)
            if not observed or clock<6 or clock+a.stride>replay_stage.metadata['frame_count']:
                s=c.advance(0,1);continue
            f=(np.asarray([(s.player.x+184)/368,(s.player.y-32)/400,s.power/400],np.float32)
                if a.routes_only else env.encoder.encode(s)['features'])
            signature=scene_signature(s) if a.routes_only else None
            phase,phase_clock=phase_progress(s)
            if phase not in phase_names:phase_names.append(phase)
            masks=replay_stage.inputs[clock:clock+a.stride,0].copy()
            distribution=movement_distribution(masks)
            weight=float(s.player.state==1 and s.player.invulnerable_frames==0)
            # Human fire release changes future kill timing. Keep its route,
            # but reduce movement imitation weight instead of pretending that
            # this is our always-shoot four-frame action.
            weight*=.25+.75*float(np.mean((masks&1)!=0))
            before=s;s=c.advance(0,a.stride)
            compatible=(s.stage==before.stage and s.stage_frame==before.stage_frame+a.stride)
            if not compatible:weight=0.
            if compatible:
                native_mask=c._memory.integer(c._memory.base+c.profile.offsets['statics']['input'])&255
                input_checks+=1
                if native_mask!=(int(masks[-1])&255):input_disagreements+=1;weight=0.
            rows.append(dict(position=[before.player.x,before.player.y],frame=before.frame,
                stage_frame=before.stage_frame,phase=phase_names.index(phase),
                phase_clock=phase_clock if phase_clock is not None else -1,masks=masks.tolist(),signature=signature))
            features.append(f);targets.append(distribution);weights.append(weight)
            if len(rows)%64==0:
                write_json(a.output/'progress.json',dict(phase='native-replay-capture',rows=len(rows),
                    stage=s.stage,stage_frame=s.stage_frame,elapsed_seconds=time.monotonic()-started,
                    Misses=events['misses'],Bombs=events['bombs_used'],input_checks=input_checks,
                    input_disagreements=input_disagreements,network_training=False))
        report=dict(schema='TH14-human-replay-native-capture-v1',source=str(a.replay.resolve()),
            full_game_NBNM=bool(full_clear and history==[1,2,3,4,5,6] and not events['misses'] and not events['bombs_used']),
            stages=stage_reports,stage_history=history,rows=len(rows),phase_names=phase_names,
            Misses=events['misses'],Bombs=events['bombs_used'],input_checks=input_checks,
            input_disagreements=input_disagreements,elapsed_seconds=time.monotonic()-started,
            network_training=False,PPO_rows=0,source_replay_playback=True,
            feature_kind=feature_kind,
            movement_target_order='policy18',native_sample_stride=a.stride,
            notes='Human replay evidence, not student performance. Soft movement labels preserve within-decision human key changes.')
        write_json(a.output/'report.json',report)
        write_json(a.output/'progress.json',dict(phase='complete',**{k:v for k,v in report.items() if k!='stages'}))
    except BaseException:
        write_json(a.output/'failure.json',dict(error=traceback.format_exc(),rows=len(rows),network_training=False))
        raise
    finally:
        c.close();cleanup_native_temp(temporary)


if __name__=='__main__':main()
