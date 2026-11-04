"""Render actual recorded inputs in TH14, retaining each original game frame."""
from pathlib import Path
import os,sys,json,time,subprocess,ctypes,traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]


class WindowFrames:
    def __init__(self,presenter):
        from ctypes import wintypes as w
        self.p=presenter;self.u=presenter.user;self.g=ctypes.WinDLL('gdi32',use_last_error=True)
        self.u.GetDC.argtypes=[w.HWND];self.u.GetDC.restype=w.HDC
        self.u.ReleaseDC.argtypes=[w.HWND,w.HDC];self.u.ReleaseDC.restype=ctypes.c_int
        self.u.PrintWindow.argtypes=[w.HWND,w.HDC,w.UINT];self.u.PrintWindow.restype=w.BOOL
        for name,args,ret in [('CreateCompatibleDC',[w.HDC],w.HDC),('CreateCompatibleBitmap',[w.HDC,ctypes.c_int,ctypes.c_int],w.HBITMAP),
            ('SelectObject',[w.HDC,w.HGDIOBJ],w.HGDIOBJ),('GetBitmapBits',[w.HBITMAP,w.LONG,ctypes.c_void_p],w.LONG),
            ('DeleteObject',[w.HGDIOBJ],w.BOOL),('DeleteDC',[w.HDC],w.BOOL)]:
            f=getattr(self.g,name);f.argtypes=args;f.restype=ret
        rect=w.RECT();self.u.GetClientRect(self.p.hwnd,ctypes.byref(rect));self.width=rect.right;self.height=rect.bottom
        self.dc=self.u.GetDC(self.p.hwnd);self.memory=self.g.CreateCompatibleDC(self.dc)
        self.bitmap=self.g.CreateCompatibleBitmap(self.dc,self.width,self.height);self.old=self.g.SelectObject(self.memory,self.bitmap)
        self.buffer=ctypes.create_string_buffer(self.width*self.height*4)
    def bytes(self):
        if not self.u.PrintWindow(self.p.hwnd,self.memory,3):raise RuntimeError('Original window capture failed')
        if self.g.GetBitmapBits(self.bitmap,len(self.buffer),self.buffer)!=len(self.buffer):raise RuntimeError('Incomplete renderer image')
        return self.buffer.raw
    def close(self):
        self.g.SelectObject(self.memory,self.old);self.g.DeleteObject(self.bitmap);self.g.DeleteDC(self.memory);self.u.ReleaseDC(self.p.hwnd,self.dc)


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--recording',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-frames',type=int,default=0);p.add_argument('--until-stage',type=int,default=0)
    p.add_argument('--from-stage',type=int,default=1)
    p.add_argument('--audio-only',action='store_true')
    p.add_argument('--music-table',type=Path,default=Path('./outputs/demonstration/audio-assets/thbgm.fmt'))
    p.add_argument('--ffmpeg',default='ffmpeg');args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    os.environ.pop('FLY_TOUHOU_DESKTOP_NAME',None)
    from fly_touhou.compact_full_story import CompactFullStoryEnv,story_kwargs
    from fly_touhou.thrl_native_events import THRLDialogueGeometryController
    from fly_touhou.thrl_visible_demo import OwnedWindowPresenter
    from fly_touhou.thrl_control import process_identity
    from fly_touhou.thrl_study_io import read_shared_json
    from fly_touhou.thrl_durable_records import write_json
    from fly_touhou.thrl_job_paths import configure_native_temp,cleanup_native_temp
    from fly_touhou.native_replay import semantic_state
    from fly_touhou.snn_demo_audio import MusicObserverMixin,timed_music_events,render_music,mix_original_effects
    import numpy as np
    # An already-running full-recording supervisor can adopt the incremental
    # stage pipeline when it next launches this renderer, without restarting
    # collection or re-rendering the completed stages.
    publication=args.recording/'stage-publication.json'
    if publication.exists() and args.from_stage==1 and not (args.until_stage or args.max_frames or args.audio_only):
        index=read_shared_json(publication)
        while True:
            progress=read_shared_json(index['status'])
            if progress['phase']=='failed':raise RuntimeError('Stage publication failed: '+progress.get('error',''))
            if progress['phase']=='completed':break
            write_json(args.output/'progress.json',dict(status='waiting_for_stage_publication',stage=progress.get('stage'),
                published_stages=progress.get('published_stages',0)))
            time.sleep(3)
        media=read_shared_json(progress['media'])
        os.link(media['video'],args.output/'game.mp4')
        os.link(media['frame_map'],args.output/'video-source-frames.npy')
        media.update(status='completed',reused_stage_publication=True,newly_inferred_actions=0,optimizer_updates=0)
        write_json(args.output/'result.json',media)
        write_json(args.output/'cleanup.json',dict(removed=0,errors=[],owned_native_process=False))
        print('reused completed stage videos',media['frames'],flush=True)
        return
    class MusicController(MusicObserverMixin,THRLDialogueGeometryController):
        pass
    class RenderEnv(CompactFullStoryEnv):
        def _create_controller(self):
            return MusicController(self.executable,self.profile_path,timeout=self.timeout,
                private_appdata=self.private_appdata,worker_name='Fly SNN video renderer',checkpoint_wait_sleep_ms=1,
                checkpoint_timing=False,headless=args.audio_only,presentation_guard=not args.audio_only,
                binary_reader=self.binary_reader,native_seed=self.native_seed)
    cfg=read_shared_json(args.recording/'environment-config.json');cfg['compact_capture_stage6_prefix']=False
    cfg.pop('compact_frozen_prefix_replay',None)
    rows=[]
    for line in (args.recording/'commands.jsonl').read_text(encoding='utf-8').splitlines(keepends=True):
        if not line.endswith('\n'):break
        if line.strip():rows.append(json.loads(line))
    temp=configure_native_temp(args.output);env=None;capture=None;encoder=None;mapping=[];milestones=[]
    capture_active=args.from_stage<=1;capture_offset=0
    try:
        options=story_kwargs(cfg,args.output,1,'development')[0];options.pop('online_teacher_revision',None)
        env=RenderEnv(**dict(options,headless=args.audio_only,hide_window=False))
        env.reset(options={'evaluation_seed':3574003864});ctl=env.controller
        music_start_tick=ctl.demo_music.state()['ticks']
        before=semantic_state(env.snapshot);source=read_shared_json(args.recording/'native-start.json')['state']
        for s in [before,source]:
            for key in ['visual_seed','visual_counter']:s['rng'].pop(key,None)
        if json.loads(json.dumps(before))!=source:
            fields=[k for k in source if json.loads(json.dumps(before.get(k)))!=source[k]]
            raise RuntimeError('Rendered original opening differs: '+','.join(fields))
        width,height=640,480
        if not args.audio_only:
            presenter=OwnedWindowPresenter(process_identity(ctl.pid));presenter.show();capture=WindowFrames(presenter)
            width,height=capture.width,capture.height
            encoder=subprocess.Popen([args.ffmpeg,'-hide_banner','-loglevel','error','-y','-f','rawvideo','-pixel_format','bgr0',
                '-video_size',f'{width}x{height}','-framerate','60','-i','pipe:0','-an','-c:v','h264_nvenc','-preset','p5',
                '-cq','16','-pix_fmt','yuv420p',str(args.output/'game.mp4')],stdin=subprocess.PIPE,creationflags=subprocess.CREATE_NO_WINDOW)
        def frame(source_frame):
            if encoder is not None and capture_active:encoder.stdin.write(capture.bytes())
            mapping.append(int(source_frame))
        def tick(mask,source_frame):
            ctl._checkpoint=None
            if hasattr(ctl._memory,'invalidate_prefetch'):ctl._memory.invalidate_prefetch()
            ctl._script.post({'type':'control','payload':{'mask':int(mask),'ticks':1}})
            ctl._checkpoint=ctl._wait('checkpoint');frame(source_frame)
        frame(rows[0]['frame']);stopped=False
        for row in rows:
            total=sum(t for _,t in row['segments']);used=0
            initial=4 if total>4 else total
            for mask,ticks in row['segments']:
                if not capture_active or args.audio_only:
                    count=min(ticks,initial-used)
                    if capture_active and args.max_frames:count=min(count,args.max_frames-(len(mapping)-capture_offset))
                    if count:
                        ctl._checkpoint=None
                        if hasattr(ctl._memory,'invalidate_prefetch'):ctl._memory.invalidate_prefetch()
                        ctl._script.post({'type':'control','payload':{'mask':int(mask),'ticks':int(count)}})
                        ctl._checkpoint=ctl._wait('checkpoint')
                        mapping.extend(row['frame']+used+i for i in range(1,count+1));used+=count
                    if capture_active and args.max_frames and len(mapping)-capture_offset>=args.max_frames:
                        stopped=True;break
                    if used>=initial:break
                    continue
                for _ in range(min(ticks,initial-used)):
                    if args.max_frames and len(mapping)-capture_offset>=args.max_frames:stopped=True;break
                    used+=1;tick(mask,row['frame']+used)
                if stopped or used>=initial:break
            if stopped:break
            if total>4:
                target=row['after'];s=ctl.read();bridge=0
                while (s.stage,s.stage_frame)!=(target['stage'],target['stage_frame']):
                    if capture_active and args.max_frames and len(mapping)-capture_offset>=args.max_frames:stopped=True;break
                    if bridge>=3600:raise RuntimeError('Loading interval did not reach original landing')
                    tick(1,row['frame']+4);bridge+=1;s=ctl.read()
                if not stopped:mapping[-1]=row['frame']+total
            if stopped:break
            if not capture_active and row['after']['stage']>=args.from_stage:
                capture_active=True;capture_offset=len(mapping)
            reached_stage=bool(args.until_stage and row['after']['stage']>=args.until_stage)
            if row['index']%128==0 or total>4 or reached_stage:
                s=ctl.read();expected=row['after']
                observed=dict(stage=s.stage,stage_frame=s.stage_frame,lives=s.lives,bombs=s.bombs,power=s.power,
                    position=[s.player.x,s.player.y] if s.player else None)
                if observed!=expected:raise RuntimeError(f'Native video diverged at decision {row["index"]}: {observed} != {expected}')
                milestones.append(dict(decision=row['index'],video_frame=len(mapping)-1,matched=True))
                write_json(args.output/'progress.json',dict(status='rendering',frames=len(mapping)-capture_offset if capture_active else 0,
                    decision=row['index'],matched=True,prefix_replay=not capture_active,
                    stage=s.stage,stage_frame=s.stage_frame))
            if reached_stage:break
        if encoder is not None:
            encoder.stdin.close()
            if encoder.wait()!=0:raise RuntimeError('Native video encoder failed')
            encoder=None
        mapping=mapping[capture_offset:]
        np.save(args.output/'video-source-frames.npy',np.array(mapping,np.int64))
        observed=ctl.demo_music.state();events=timed_music_events(observed,music_start_tick)
        for event in events:event['frame']-=capture_offset
        write_json(args.output/'music-events.json',dict(schema='th14-observed-bgm-v1',start_tick=music_start_tick,
            end_tick=observed['ticks'],frames=len(mapping),events=events))
        music=render_music(Path(cfg['game_exe']).parent/'thbgm.dat',args.music_table,events,len(mapping),args.output/'music.wav')
        effects=[dict(event,frame=int(event['tick'])-int(music_start_tick)+1-capture_offset) for event in observed.get('effects',[])]
        write_json(args.output/'sound-effects.json',dict(schema='th14-native-effects-v1',frames=len(mapping),events=effects))
        assets=args.music_table.parent
        if effects:
            music.update(mix_original_effects(args.output/'music.wav',effects,assets.parent/'audio-analysis/sfx-id-map.json',
                assets/'sfx-44100',args.output/'soundtrack.wav'))
        write_json(args.output/'result.json',dict(status='completed',frames=len(mapping),fps=60,dimensions=[width,height],
            source_recording=str(args.recording),source='Original TH14 Direct3D frames',milestones=milestones,
            newly_inferred_actions=0,optimizer_updates=0,excerpt=bool(args.max_frames or args.until_stage),until_stage=args.until_stage,
            audio_only=args.audio_only,from_stage=args.from_stage,video_frame_offset=capture_offset,**music))
        print('rendered',len(mapping),width,height,flush=True)
    except BaseException as exc:
        write_json(args.output/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()));raise
    finally:
        if encoder is not None:
            try:encoder.stdin.close()
            except (BrokenPipeError,OSError):pass
            encoder.wait()
        if capture is not None:capture.close()
        if env is not None:env.close()
        write_json(args.output/'cleanup.json',cleanup_native_temp(temp))


if __name__=='__main__':main()
