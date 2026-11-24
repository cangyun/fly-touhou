"""Publish each finished stage, then assemble prefixes without replaying inference."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time,traceback,wave

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from fly_touhou.thrl_study_io import read_shared_json
from fly_touhou.thrl_durable_records import write_json
from fly_touhou.snn_demo_media import normalize_video_track,validate_timeline
import numpy as np
import psutil


def concatenate(chapters, output, ffmpeg):
    output.mkdir(parents=True,exist_ok=True)
    frames=sum(c['frames'] for c in chapters)
    audio=output/'music.wav'
    with wave.open(str(audio),'wb') as target:
        target.setnchannels(2);target.setsampwidth(2);target.setframerate(44100)
        for chapter in chapters:
            with wave.open(chapter['audio'],'rb') as source:
                if source.getnframes()!=chapter['frames']*735:
                    raise RuntimeError('Stage audio and video lengths differ')
                while block:=source.readframes(65536):target.writeframesraw(block)
    np.save(output/'video-source-frames.npy',np.concatenate([np.load(c['frame_map']) for c in chapters]))
    for key,name in [('video','game.mp4'),('presentation_video','presentation-1080p60.mp4')]:
        listing=output/(key+'-concat.txt')
        # concat demuxer requires matching stream time bases. Merely choosing
        # the output track time scale cannot fix mismatched input packet PTS.
        tracks=[normalize_video_track(c[key],c['frames'],ffmpeg) for c in chapters]
        listing.write_text(''.join("file '"+track.as_posix().replace("'", "'\\''")+"'\nduration "+format(c['frames']/60,'.9f')+'\n'
            for c,track in zip(chapters,tracks)),encoding='utf-8')
        command=[ffmpeg,'-hide_banner','-loglevel','error','-y','-f','concat','-safe','0','-i',str(listing)]
        if key=='presentation_video':command+=['-i',str(audio),'-map','0:v:0','-map','1:a:0','-c:a','aac','-b:a','192k']
        else:command+=['-map','0:v:0','-an']
        command+=['-c:v','copy','-t',str(frames/60),'-video_track_timescale','60000','-movflags','+faststart',str(output/name)]
        subprocess.run(command,check=True,creationflags=subprocess.CREATE_NO_WINDOW)
        verification=validate_timeline(output/name,frames,audio=key=='presentation_video',ffmpeg=ffmpeg)
        write_json(output/(key+'-timeline-validation.json'),verification)
    offsets=[];offset=0
    for chapter in chapters:
        offsets.append(dict(stage=chapter['stage'],start_frame=offset,frames=chapter['frames'],video=chapter['presentation_video']))
        offset+=chapter['frames']
    media=dict(status='completed',frames=frames,fps=60,video_frame_offset=0,
        video=str(output/'game.mp4'),frame_map=str(output/'video-source-frames.npy'),audio=str(audio),
        presentation_video=str(output/'presentation-1080p60.mp4'),chapters=offsets,
        scope='complete-recorded-clear' if chapters[-1]['stage']==6 else f'stages-1-to-{chapters[-1]["stage"]}-preview',
        source='Original TH14 frames and recorded SNN activity',sound_effects=all(c.get('sound_effects',False) for c in chapters))
    write_json(output/'media.json',media)
    return media


def main():
    p=argparse.ArgumentParser();p.add_argument('--recording',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--presentation',type=Path,required=True)
    p.add_argument('--ui-python',type=Path,required=True);p.add_argument('--ffmpeg',default='ffmpeg')
    p.add_argument('--parallelism',type=int,default=1)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    def status(phase,**values):write_json(a.output/'status.json',dict(phase=phase,pid=os.getpid(),**values))
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONIOENCODING='utf-8')
    try:
        index=a.output/'chapters.json'
        if index.exists():chapters=read_shared_json(index)['chapters']
        else:
            current=read_shared_json(a.recording/'presentation-media.json')
            if current.get('frames')!=14842:raise RuntimeError('Expected the already-published complete first stage')
            preview=read_shared_json(a.presentation)['preview']
            chapters=[dict(current,stage=1,presentation_video=preview)]
            write_json(index,dict(chapters=chapters))
        write_json(a.recording/'stage-publication.json',dict(status=str(a.output/'status.json'),output=str(a.output),pid=os.getpid()))
        selected=read_shared_json(a.presentation);selected['stage_publication_status']=str(a.output/'status.json');write_json(a.presentation,selected)
        if a.parallelism>1:
            from concurrent.futures import ThreadPoolExecutor
            from prepare_snn_stage import prepare_stage
            pending=list(range(len(chapters)+1,7))
            jobs=[dict(stage=stage,status=str(a.output/f'stage-{stage:02d}'/'prepare-status.json')) for stage in pending]
            with ThreadPoolExecutor(max_workers=min(3,a.parallelism)) as pool:
                futures={stage:pool.submit(prepare_stage,a.recording,a.output,stage,a.ui_python,env,a.ffmpeg) for stage in pending}
                for stage in pending:
                    status('parallel_preparing',stage=stage,published_stages=len(chapters),parallel_jobs=jobs,parallelism=min(3,a.parallelism))
                    media=futures[stage].result()
                    if media['video_frame_offset']!=sum(c['frames'] for c in chapters):
                        raise RuntimeError(f'Stage {stage} does not join the preceding frame boundary')
                    chapters.append(media);write_json(index,dict(chapters=chapters))
                    status('publishing',stage=stage,published_stages=stage-1,parallel_jobs=jobs)
                    combined=a.output/f'through-stage-{stage}';published=concatenate(chapters,combined,a.ffmpeg)
                    write_json(a.recording/'presentation-media.json',published)
                    selected=read_shared_json(a.presentation)
                    selected.update(recording=str(a.recording),preview=published['presentation_video'],stage_preview=published['presentation_video'],
                        available_preview=dict(frames=published['frames'],seconds=published['frames']/60,scope=published['scope']))
                    write_json(a.presentation,selected)
                    status('stage_published',stage=stage,published_stages=stage,media=str(combined/'media.json'),video=published['presentation_video'],parallel_jobs=jobs)
            status('completed',published_stages=6,media=str(a.output/'through-stage-6/media.json'));return
        for stage in range(len(chapters)+1,7):
            status('waiting_for_stage',stage=stage,published_stages=len(chapters))
            while True:
                state=read_shared_json(a.recording/'status.json')
                if int(state.get('game',{}).get('stage',0))>stage:break
                if state['status']=='completed':
                    result=read_shared_json(a.recording/'result.json')
                    # Final collector status drops the live `game` object.
                    # A completed six-stage clear includes every earlier stage.
                    if result['episode']['outcome']=='clear':break
                    raise RuntimeError(f'Stage {stage} is not complete in the source recording')
                if state['status'] in ('failed','stopped'):raise RuntimeError('Source collection stopped before the next stage completed')
                time.sleep(3)
            folder=a.output/f'stage-{stage:02d}';folder.mkdir(parents=True,exist_ok=True)
            native=folder/'native-video';receipt=folder/'native-dispatch.json'
            status('rendering',stage=stage,published_stages=len(chapters),progress=str(native/'progress.json'))
            if not (native/'result.json').exists():
                if receipt.exists():
                    dispatch=read_shared_json(receipt)
                    while True:
                        try:alive=abs(psutil.Process(dispatch['pid']).create_time()-dispatch['create_time'])<.1
                        except psutil.NoSuchProcess:alive=False
                        if not alive:break
                        time.sleep(3)
                else:
                    command=[sys.executable,'-B',str(ROOT/'scripts/render_snn_native_video.py'),'--recording',str(a.recording),
                        '--output',str(native),'--from-stage',str(stage)]
                    if stage<6:command+=['--until-stage',str(stage+1)]
                    with (folder/'native.stdout.log').open('wb') as stdout,(folder/'native.stderr.log').open('wb') as stderr:
                        child=subprocess.Popen(command,stdout=stdout,stderr=stderr,env=env,creationflags=subprocess.CREATE_NO_WINDOW)
                        write_json(receipt,dict(pid=child.pid,create_time=psutil.Process(child.pid).create_time(),command=command))
                        if child.wait()!=0:raise RuntimeError(f'Stage {stage} native render failed; see {folder}/native.stderr.log')
            media=read_shared_json(native/'result.json')
            if media['video_frame_offset']!=sum(c['frames'] for c in chapters):
                raise RuntimeError(f'Stage {stage} does not join the preceding recorded frame boundary')
            # Renderers already running when sound support was installed keep
            # their original code. Recover their audio from the same input tape.
            if not media.get('sound_effects'):
                recovery=folder/'audio-correction'
                if not (recovery/'result.json').exists():
                    status('recovering_audio',stage=stage,published_stages=len(chapters))
                    command=[sys.executable,'-B',str(ROOT/'scripts/render_snn_native_video.py'),'--recording',str(a.recording),
                        '--output',str(recovery),'--from-stage',str(stage),'--audio-only']
                    if stage<6:command+=['--until-stage',str(stage+1)]
                    with (folder/'audio.stdout.log').open('wb') as stdout,(folder/'audio.stderr.log').open('wb') as stderr:
                        subprocess.run(command,env=env,stdout=stdout,stderr=stderr,check=True,creationflags=subprocess.CREATE_NO_WINDOW)
                corrected=read_shared_json(recovery/'result.json')
                if not np.array_equal(np.load(native/'video-source-frames.npy'),np.load(recovery/'video-source-frames.npy')):
                    raise RuntimeError('Recovered audio does not match the recorded video timeline')
                media.update({k:v for k,v in corrected.items() if (k.startswith('audio') and k!='audio_only') or k.startswith('sound_effect')})
                write_json(native/'result.json',media)
            media.update(video=str(native/'game.mp4'),frame_map=str(native/'video-source-frames.npy'),stage=stage,scope=f'stage-{stage}')
            write_json(folder/'media.json',media)
            final=folder/f'stage-{stage}-1080p60-bgm.mp4'
            status('composing',stage=stage,published_stages=len(chapters),progress=str(folder/'compose-progress.json'))
            if not final.exists():
                with (folder/'compose.stdout.log').open('wb') as stdout,(folder/'compose.stderr.log').open('wb') as stderr:
                    subprocess.run([str(a.ui_python),'-B',str(ROOT/'scripts/open_snn_demonstration.py'),'--recording',str(a.recording),
                        '--media',str(folder/'media.json'),'--export',str(final),'--progress',str(folder/'compose-progress.json')],
                        env=env,stdout=stdout,stderr=stderr,check=True,creationflags=subprocess.CREATE_NO_WINDOW)
            media['presentation_video']=str(final);write_json(folder/'media.json',media)
            chapters.append(media);write_json(index,dict(chapters=chapters))
            status('publishing',stage=stage,published_stages=stage-1)
            combined=a.output/f'through-stage-{stage}';published=concatenate(chapters,combined,a.ffmpeg)
            write_json(a.recording/'presentation-media.json',published)
            selected=read_shared_json(a.presentation)
            selected.update(recording=str(a.recording),preview=published['presentation_video'],stage_preview=published['presentation_video'],
                available_preview=dict(frames=published['frames'],seconds=published['frames']/60,scope=published['scope']))
            write_json(a.presentation,selected)
            status('stage_published',stage=stage,published_stages=stage,media=str(combined/'media.json'),video=published['presentation_video'])
        status('completed',published_stages=6,media=str(a.output/'through-stage-6/media.json'))
    except BaseException as error:
        status('failed',error=repr(error),traceback=traceback.format_exc());raise


if __name__=='__main__':main()
