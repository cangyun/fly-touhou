"""Replace published BGM-only soundtracks while retaining the recorded pictures."""
from pathlib import Path
import argparse,sys,subprocess,wave
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
from fly_touhou.thrl_study_io import read_shared_json
from fly_touhou.thrl_durable_records import write_json
from publish_snn_stages import concatenate


def main():
    p=argparse.ArgumentParser();p.add_argument('--publication',type=Path,required=True)
    p.add_argument('--audio-recording',type=Path,required=True);p.add_argument('--recording',type=Path,required=True)
    p.add_argument('--presentation',type=Path,required=True);p.add_argument('--ffmpeg',default='ffmpeg');a=p.parse_args()
    chapters=read_shared_json(a.publication/'chapters.json')['chapters']
    audio=read_shared_json(a.audio_recording/'result.json')
    expected=np.concatenate([np.load(c['frame_map']) for c in chapters])
    if not np.array_equal(expected,np.load(a.audio_recording/'video-source-frames.npy')):
        raise RuntimeError('Audio backfill does not match the published original frames')
    with wave.open(audio['audio'],'rb') as source:
        for chapter in chapters:
            stage=chapter['stage'];folder=a.publication/f'stage-{stage:02d}';folder.mkdir(exist_ok=True)
            soundtrack=folder/'soundtrack-full.wav'
            with wave.open(str(soundtrack),'wb') as target:
                target.setnchannels(2);target.setsampwidth(2);target.setframerate(44100)
                remaining=chapter['frames']*735
                while remaining:
                    count=min(65536,remaining);block=source.readframes(count)
                    if len(block)!=count*4:raise RuntimeError('Recovered soundtrack is incomplete')
                    target.writeframesraw(block);remaining-=count
            final=folder/f'stage-{stage}-1080p60-full-audio.mp4'
            subprocess.run([a.ffmpeg,'-hide_banner','-loglevel','error','-y','-i',chapter['presentation_video'],
                '-i',str(soundtrack),'-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','aac','-b:a','192k',
                '-t',str(chapter['frames']/60),'-video_track_timescale','60000','-movflags','+faststart',str(final)],
                check=True,creationflags=subprocess.CREATE_NO_WINDOW)
            chapter.update(audio_bgm=chapter['audio'],audio=str(soundtrack),sound_effects=True,
                presentation_video=str(final),audio_mix_gain=audio['audio_mix_gain'],audio_backfill=str(a.audio_recording))
            write_json(folder/'media.json',chapter)
    write_json(a.publication/'chapters.json',dict(chapters=chapters))
    combined=a.publication/f'through-stage-{len(chapters)}-full-audio'
    media=concatenate(chapters,combined,a.ffmpeg)
    write_json(a.recording/'presentation-media.json',media)
    selected=read_shared_json(a.presentation)
    selected.update(preview=media['presentation_video'],stage_preview=media['presentation_video'],
        available_preview=dict(frames=media['frames'],seconds=media['frames']/60,scope=media['scope']))
    write_json(a.presentation,selected)
    write_json(a.publication/'audio-backfill-result.json',dict(status='completed',stages=len(chapters),frames=media['frames'],
        video=media['presentation_video'],sound_effects=True,frame_map_matched=True))
    print('Published music and original effects for',len(chapters),'stages',flush=True)


if __name__=='__main__':main()
