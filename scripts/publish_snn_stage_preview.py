"""Publish a completed stage while full neural acquisition continues."""
from pathlib import Path
import sys,os,subprocess,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--recording',type=Path,required=True)
    p.add_argument('--preview-recording',type=Path,required=True);p.add_argument('--presentation',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--ui-python',type=Path,required=True)
    p.add_argument('--until-stage',type=int,default=2);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    from fly_touhou.thrl_study_io import read_shared_json
    from fly_touhou.thrl_durable_records import write_json
    def status(phase,**values):write_json(a.output/'status.json',dict(phase=phase,pid=os.getpid(),**values))
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONIOENCODING='utf-8')
    try:
        status('rendering_stage_preview');video=a.output/'native-video'
        subprocess.run([sys.executable,'-B',str(ROOT/'scripts/render_snn_native_video.py'),'--recording',str(a.recording),
            '--output',str(video),'--until-stage',str(a.until_stage)],env=env,check=True,creationflags=subprocess.CREATE_NO_WINDOW)
        media=read_shared_json(video/'result.json');media.update(video=str(video/'game.mp4'),
            frame_map=str(video/'video-source-frames.npy'),scope=f'stages-1-to-{a.until_stage-1}-preview')
        current_media=a.recording/'presentation-media.json'
        if current_media.exists() and read_shared_json(current_media).get('scope')=='complete-recorded-clear':
            status('full_recording_already_available');return
        if not (a.recording/'presentation-options.json').exists():
            write_json(a.recording/'presentation-options.json',read_shared_json(a.preview_recording/'presentation-options.json'))
        write_json(current_media,media)
        selected=read_shared_json(a.presentation);selected['recording']=str(a.recording)
        selected['available_preview']=dict(frames=media['frames'],seconds=media['frames']/60,scope=media['scope'])
        write_json(a.presentation,selected)
        final=a.output/f'stage-{a.until_stage-1}-1080p60.mp4';status('available_in_viewer_and_composing',frames=media['frames'])
        subprocess.run([str(a.ui_python),'-B',str(ROOT/'scripts/open_snn_demonstration.py'),'--recording',str(a.recording),
            '--export',str(final),'--frames',str(media['frames'])],env=env,check=True,creationflags=subprocess.CREATE_NO_WINDOW)
        selected=read_shared_json(a.presentation);selected['stage_preview']=str(final);selected['preview']=str(final);write_json(a.presentation,selected)
        status('completed',video=str(final),frames=media['frames'],fps=60)
    except BaseException as e:
        status('failed',error=repr(e),traceback=traceback.format_exc());raise


if __name__=='__main__':main()
