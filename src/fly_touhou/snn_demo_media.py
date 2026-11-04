"""Media joins use one video time base and validate the packet time line."""
from pathlib import Path
from fractions import Fraction
import json,subprocess

import numpy as np


def probe_tool(ffmpeg):
    path=Path(ffmpeg)
    return str(path.with_name('ffprobe'+path.suffix)) if path.parent!=Path('.') else 'ffprobe'


def probe(path,ffmpeg='ffmpeg'):
    command=[probe_tool(ffmpeg),'-v','error','-show_entries',
        'stream=index,codec_type,time_base,start_time,duration,nb_frames,r_frame_rate','-of','json',str(path)]
    return json.loads(subprocess.run(command,capture_output=True,check=True,
        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)).stdout)['streams']


def normalize_video_track(path,frames,ffmpeg='ffmpeg'):
    """Stream copy only: strip audio and put all video timestamps in 1/60000 s."""
    path=Path(path);output=path.with_name(path.stem+'.video60000.mp4')
    if output.exists():
        streams=probe(output,ffmpeg)
        if len(streams)==1 and streams[0].get('time_base')=='1/60000' and int(streams[0].get('nb_frames',0))==frames:
            return output
    subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-y','-i',str(path),'-map','0:v:0',
        '-c:v','copy','-an','-video_track_timescale','60000','-movflags','+faststart',str(output)],
        check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    return output


def validate_timeline(path,frames,*,audio,ffmpeg='ffmpeg'):
    streams=probe(path,ffmpeg);video=next(s for s in streams if s['codec_type']=='video')
    command=[probe_tool(ffmpeg),'-v','error','-select_streams','v:0','-show_packets',
        '-show_entries','packet=pts,dts','-of','json',str(path)]
    packets=json.loads(subprocess.run(command,capture_output=True,check=True,
        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)).stdout)['packets']
    pts=np.sort(np.array([int(p['pts']) for p in packets],dtype=np.int64))
    dts=np.array([int(p['dts']) for p in packets],dtype=np.int64)
    unit=float(Fraction(video['time_base']));expected=np.arange(frames)/60
    if len(pts)!=frames:raise RuntimeError(f'Video contains {len(pts)} frames, expected {frames}')
    error=float(np.max(np.abs(pts*unit-expected))) if frames else 0.
    if error>1/60000+.000001 or np.any(np.diff(dts)<=0):
        raise RuntimeError(f'Invalid 60 FPS video timeline: max timestamp error={error:.6f}s')
    sound=next((s for s in streams if s['codec_type']=='audio'),None)
    if audio and (sound is None or abs(float(sound.get('start_time',0)))>.002
                  or abs(float(sound['duration'])-frames/60)>.025):
        raise RuntimeError('Audio does not cover the same video timeline')
    return dict(path=str(path),frames=frames,seconds=frames/60,time_base=video['time_base'],
        max_timestamp_error_ms=error*1000,video_start=float(video.get('start_time',0)),
        audio_start=float(sound.get('start_time',0)) if sound else None,
        audio_seconds=float(sound['duration']) if sound else None,passed=True)
