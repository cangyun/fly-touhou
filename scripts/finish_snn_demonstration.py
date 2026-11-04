"""Finish the authorized recording and publish the full60FPS presentation."""
from pathlib import Path
import sys,os,time,subprocess,traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--recording',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--ui-python',type=Path,required=True)
    p.add_argument('--collect-bundle',type=Path);p.add_argument('--connectome',type=Path)
    p.add_argument('--presentation',type=Path);p.add_argument('--preview-recording',type=Path)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    from fly_touhou.thrl_study_io import read_shared_json
    from fly_touhou.thrl_durable_records import write_json
    def status(phase,**details):write_json(args.output/'status.json',dict(phase=phase,pid=os.getpid(),**details))
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONIOENCODING='utf-8')
    try:
        if args.collect_bundle:
            if args.connectome is None:raise ValueError('--connectome is required for collection')
            status('collecting_real_neural_recording')
            subprocess.run([sys.executable,'-B',str(ROOT/'scripts/collect_snn_demonstration.py'),
                '--bundle',str(args.collect_bundle),'--connectome',str(args.connectome),'--output',str(args.recording)],
                env=env,check=True,creationflags=subprocess.CREATE_NO_WINDOW)
        status('waiting_for_real_neural_recording')
        while True:
            if (args.output/'STOP').exists():status('stopped');return
            state=read_shared_json(args.recording/'status.json')
            if state['status'] in ('failed','stopped'):
                status('source_recording_incomplete',source=state);return
            if state['status']=='completed' and (args.recording/'cleanup.json').exists():break
            time.sleep(5)
        result=read_shared_json(args.recording/'result.json')
        if result['episode']['outcome']!='clear':
            status('recording_did_not_clear',source=result);return
        if args.preview_recording:
            write_json(args.recording/'presentation-options.json',read_shared_json(args.preview_recording/'presentation-options.json'))
        video=args.output/'native-video'
        status('rendering_original_game_at_native_frame_resolution')
        subprocess.run([sys.executable,'-B',str(ROOT/'scripts/render_snn_native_video.py'),
            '--recording',str(args.recording),'--output',str(video)],env=env,check=True,
            creationflags=subprocess.CREATE_NO_WINDOW)
        media=read_shared_json(video/'result.json');media.update(video=str(video/'game.mp4'),
            frame_map=str(video/'video-source-frames.npy'),scope='complete-recorded-clear')
        write_json(args.recording/'presentation-media.json',media)
        if args.presentation:
            selected=read_shared_json(args.presentation);selected['recording']=str(args.recording)
            write_json(args.presentation,selected)
        final=args.output/'果蝇辉针城-Lunatic-1080p60.mp4'
        status('composing_1080p60',native_media=media)
        subprocess.run([str(args.ui_python),'-B',str(ROOT/'scripts/open_snn_demonstration.py'),
            '--recording',str(args.recording),'--export',str(final)],env=env,check=True,
            creationflags=subprocess.CREATE_NO_WINDOW)
        status('completed',video=str(final),frames=media['frames'],fps=60,audio=bool(media.get('audio')))
    except BaseException as exc:
        status('failed',error=repr(exc),traceback=traceback.format_exc());raise


if __name__=='__main__':main()
