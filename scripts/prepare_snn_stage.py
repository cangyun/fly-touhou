"""An independent stage job: original frames, soundtrack and presentation."""
from pathlib import Path
import os,subprocess,sys,time,traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import psutil
from fly_touhou.thrl_study_io import read_shared_json
from fly_touhou.thrl_durable_records import write_json
from fly_touhou.snn_demo_media import validate_timeline


def prepare_stage(recording,output,stage,ui_python,env,ffmpeg='ffmpeg'):
    recording=Path(recording);folder=Path(output)/f'stage-{stage:02d}';folder.mkdir(parents=True,exist_ok=True)
    def status(phase,**details):write_json(folder/'prepare-status.json',dict(phase=phase,stage=stage,pid=os.getpid(),**details))
    try:
        if (folder/'prepared.json').exists():
            media=read_shared_json(folder/'prepared.json');status('completed',video=media['presentation_video']);return media
        status('waiting_for_collection')
        while True:
            state=read_shared_json(recording/'status.json')
            if int(state.get('game',{}).get('stage',0))>stage:break
            if state['status']=='completed' and read_shared_json(recording/'result.json')['episode']['outcome']=='clear':break
            if state['status'] in ('failed','stopped','completed'):raise RuntimeError(f'Stage {stage} was not fully collected')
            time.sleep(3)
        native=folder/'native-video';receipt=folder/'native-dispatch.json'
        status('rendering',progress=str(native/'progress.json'))
        if not (native/'result.json').exists():
            if receipt.exists():
                dispatch=read_shared_json(receipt)
                while True:
                    try:alive=abs(psutil.Process(dispatch['pid']).create_time()-dispatch['create_time'])<.1
                    except psutil.NoSuchProcess:alive=False
                    if not alive:break
                    time.sleep(3)
                if not (native/'result.json').exists():raise RuntimeError(f'Stage {stage} native renderer stopped without a completed result')
            else:
                command=[sys.executable,'-B',str(ROOT/'scripts/render_snn_native_video.py'),'--recording',str(recording),
                    '--output',str(native),'--from-stage',str(stage)]
                if stage<6:command+=['--until-stage',str(stage+1)]
                with (folder/'native.stdout.log').open('wb') as stdout,(folder/'native.stderr.log').open('wb') as stderr:
                    child=subprocess.Popen(command,stdout=stdout,stderr=stderr,env=env,creationflags=subprocess.CREATE_NO_WINDOW)
                    write_json(receipt,dict(pid=child.pid,create_time=psutil.Process(child.pid).create_time(),command=command))
                    if child.wait()!=0:raise RuntimeError(f'Stage {stage} native rendering failed')
        media=read_shared_json(native/'result.json')
        if not media.get('sound_effects'):
            recovery=folder/'audio-correction';status('recovering_audio')
            if not (recovery/'result.json').exists():
                command=[sys.executable,'-B',str(ROOT/'scripts/render_snn_native_video.py'),'--recording',str(recording),
                    '--output',str(recovery),'--from-stage',str(stage),'--audio-only']
                if stage<6:command+=['--until-stage',str(stage+1)]
                with (folder/'audio.stdout.log').open('wb') as stdout,(folder/'audio.stderr.log').open('wb') as stderr:
                    subprocess.run(command,env=env,stdout=stdout,stderr=stderr,check=True,creationflags=subprocess.CREATE_NO_WINDOW)
            if not np.array_equal(np.load(native/'video-source-frames.npy'),np.load(recovery/'video-source-frames.npy')):
                raise RuntimeError('Recovered audio does not match original video frames')
            corrected=read_shared_json(recovery/'result.json')
            media.update({k:v for k,v in corrected.items() if (k.startswith('audio') and k!='audio_only') or k.startswith('sound_effect')})
        media.update(video=str(native/'game.mp4'),frame_map=str(native/'video-source-frames.npy'),stage=stage,scope=f'stage-{stage}')
        write_json(folder/'media.json',media)
        final=folder/f'stage-{stage}-1080p60-full-audio.mp4'
        pending=folder/f'stage-{stage}-1080p60.rendering.mp4'
        status('composing',progress=str(folder/'compose-progress.json'))
        with (folder/'compose.stdout.log').open('wb') as stdout,(folder/'compose.stderr.log').open('wb') as stderr:
            subprocess.run([str(ui_python),'-B',str(ROOT/'scripts/open_snn_demonstration.py'),'--recording',str(recording),
                '--media',str(folder/'media.json'),'--export',str(pending),'--progress',str(folder/'compose-progress.json'),
                '--ffmpeg',ffmpeg],env=env,stdout=stdout,stderr=stderr,check=True,creationflags=subprocess.CREATE_NO_WINDOW)
        validation=validate_timeline(pending,media['frames'],audio=True,ffmpeg=ffmpeg)
        os.replace(pending,final)
        media['presentation_video']=str(final)
        write_json(folder/'media.json',media);write_json(folder/'prepared.json',media)
        validation['path']=str(final);write_json(folder/'timeline-validation.json',validation)
        status('completed',video=str(final),frames=media['frames']);return media
    except BaseException as error:
        status('failed',error=repr(error),traceback=traceback.format_exc());raise
