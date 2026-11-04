"""Original TH14 music, timed by observed game commands rather than wall time."""
from pathlib import Path
import json
import struct
import wave

import numpy as np


class MusicObserver:
    """Read-only hooks in the renderer's owned process; no sound/state writes."""
    def __init__(self, controller):
        config = dict(module=controller.executable.name, input=controller.hooks['input']['rva'])
        self.script = controller._session.create_script('const C = '+json.dumps(config)+';\n'+r'''
const base = Process.getModuleByName(C.module).base;
let ticks = 0;
const events = [];
const effects = [];
Interceptor.attach(base.add(C.input), {onEnter() {ticks++;}});
// TH14 1.00b BGM manager command queue, __thiscall(cmd, value, filename).
Interceptor.attach(base.add(0x6ef90), {onEnter(args) {
    const command = args[0].toInt32(), value = args[1].toInt32();
    let name = null;
    if (!args[2].isNull()) name = args[2].readCString();
    events.push({tick:ticks,command,value,name});
}});
// Accepted sound-buffer playback, after native same-frame request merging.
Interceptor.attach(base.add(0x6eb30), {onEnter(args) {
    const entry = this.context.ecx;
    const id = entry.sub(base.add(0xf95dc)).toInt32()/0x18;
    if (Number.isInteger(id) && id>=0 && id<76)
        effects.push({tick:ticks,command:'play',id,pan:args[0].toInt32(),buffer_available:!entry.readPointer().isNull()});
}});
Interceptor.attach(base.add(0x6e550), {onEnter(args) {
    if(args[0].toInt32()<0) effects.push({tick:ticks,command:'stop',id:-1});
}});
Interceptor.attach(base.add(0x6ea7a), {onEnter() {
    effects.push({tick:ticks,command:'stop',id:this.context.ebx.toInt32()});
}});
Interceptor.attach(base.add(0x373b0), {onEnter() {effects.push({tick:ticks,command:'pause',id:-1});}});
for (const rva of [0x4904b,0x4c070])
    Interceptor.attach(base.add(rva), {onEnter() {effects.push({tick:ticks,command:'resume',id:-1});}});
Interceptor.attach(base.add(0x524da), {onEnter() {
    effects.push({tick:ticks,command:'pan',id:20,pan:this.context.eax.toInt32()});
}});
rpc.exports = {state() {return {ticks,events,effects};}};
''')
        self.script.load()

    def state(self):
        return dict(self.script.exports_sync.state())


class MusicObserverMixin:
    def _wait(self, kind):
        result = super()._wait(kind)
        if kind == 'hook-ready':
            # The controller has verified/installed its input hook, but has
            # not resumed the owned process. Observe after that verification.
            self.demo_music = MusicObserver(self)
        return result


def read_music_table(path):
    data = Path(path).read_bytes()
    result = {}
    for offset in range(0, len(data)-51, 52):
        name, pos, preload, intro, size, tag, channels, rate, avg, align, bits, extra, pad = struct.unpack_from('<16sIIiiHHIIHHHH', data, offset)
        name = name.split(b'\0', 1)[0].decode('ascii')
        if not name:
            break
        if (tag, channels, rate, bits, align) != (1, 2, 44100, 16, 4):
            raise ValueError(f'Unsupported original music format: {name}')
        result[name] = dict(name=name, offset=pos, intro=intro, size=size, sample_rate=rate,
                            channels=channels, bits=bits, block_align=align)
    return result


def timed_music_events(observed, start_tick):
    """An event after an input poll appears in the next captured native frame."""
    return [dict(event, frame=int(event['tick'])-int(start_tick)+1)
            for event in observed['events']]


def render_music(dat_path, table_path, events, frames, output, *, gain=0.8):
    """Reconstruct looping PCM using original slots, play/stop and fade events.

    Only BGM is reconstructed. Native game effects and system audio are excluded.
    44.1 kHz / 60 FPS gives exactly 735 stereo samples per video frame.
    """
    tracks = read_music_table(table_path)
    slots = {}
    pcm_cache = {}
    current = None
    cursor = 0
    paused = False
    fade = None
    switches = []
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(enumerate(events), key=lambda item:(int(item[1]['frame']), item[0]))
    first = min(0, int(ordered[0][1]['frame'])) if ordered else 0
    pointer = 0
    energy = 0.
    samples_written = 0
    def load(name):
        name = str(name).replace('\\', '/').split('/')[-1]
        if name not in tracks:
            raise ValueError(f'Unknown music selected by original game: {name}')
        if name not in pcm_cache:
            spec = tracks[name]
            with Path(dat_path).open('rb') as source:
                source.seek(spec['offset']);raw = source.read(spec['size'])
            if len(raw) != spec['size']:
                raise ValueError('Original music stream is truncated')
            pcm_cache[name] = np.frombuffer(raw, dtype='<i2').reshape(-1, 2)
        return name
    with wave.open(str(output), 'wb') as stream:
        stream.setnchannels(2);stream.setsampwidth(2);stream.setframerate(44100)
        for frame in range(first, int(frames)):
            while pointer < len(ordered) and int(ordered[pointer][1]['frame']) <= frame:
                event = ordered[pointer][1];pointer += 1
                command, value, name = event['command'], event['value'], event.get('name')
                if command == 1 and name:
                    slots[value] = name
                elif command == 2:
                    selected = name if value == -1 else slots.get(value)
                    if not selected:
                        raise ValueError(f'Original BGM slot {value} was not observed preloading')
                    current = load(selected);cursor = 0;fade = None;paused = False
                    switches.append(dict(frame=frame, seconds=frame/60, track=current))
                elif command in (3, 4):
                    current = None;fade = None
                elif command == 5:
                    fade = (frame, max(1, value*60))
                elif command == 9:
                    raise ValueError('Music format/loop update needs explicit handling before soundtrack export')
                elif command == 6:
                    paused = True
                elif command == 7:
                    paused = False
            signal = np.zeros((735, 2), dtype=np.float32)
            if current and not paused:
                pcm = pcm_cache[current];intro = tracks[current]['intro']//4
                positions = np.arange(cursor, cursor+735)
                positions = np.where(positions < len(pcm), positions, intro+(positions-intro)%(len(pcm)-intro))
                signal[:] = pcm[positions]
                cursor += 735
                if fade is not None:
                    progress = min(1., (frame-fade[0]+1)/fade[1])
                    # Original DirectSound attenuation: 0 to -5000 hundredths
                    # of a decibel over N frames, then stop the stream.
                    attenuation = int((1.-progress)*5000)-5000
                    signal *= 10.**(attenuation/2000.)
                    if progress >= 1.:
                        current = None
            if frame >= 0:
                signal *= float(gain)
                energy += float(np.square(signal/32768.).sum())
                samples_written += len(signal)
                stream.writeframesraw(np.clip(np.rint(signal), -32768, 32767).astype('<i2').tobytes())
    return dict(audio=str(output.resolve()), audio_source='Original local TH14 thbgm.dat',
                audio_events='Observed native BGM command queue', audio_sample_rate=44100,
                audio_channels=2, audio_frames=int(frames), audio_gain=float(gain),
                audio_rms=float(np.sqrt(energy/max(1, samples_written*2))),
                music_changes=switches, sound_effects=False)


def mix_original_effects(bgm, events, metadata_path, pcm_directory, output):
    """Mix independent original sound IDs; retriggering an ID cuts its old tail."""
    from collections import Counter
    table={int(item['id']):item for item in json.loads(Path(metadata_path).read_text(encoding='utf-8'))['effects']}
    with wave.open(str(bgm),'rb') as source:
        samples=source.getnframes()
        signal=np.frombuffer(source.readframes(samples),dtype='<i2').reshape(-1,2).astype(np.float32)/32768.
    cache={};voices={};paused={};counts=Counter()
    def finish(effect_id, end_sample):
        voice=voices.pop(effect_id,None)
        if voice is None:return
        item=table[effect_id];name=item['filename']
        if name not in cache:
            cache[name]=np.fromfile(Path(pcm_directory)/(Path(name).stem+'.f32'),dtype='<f4').reshape(-1,2)
        pcm=cache[name]
        start=int(voice['frame'])*735
        end=min(samples,int(end_sample))
        offset=int(voice.get('offset',0))
        if not item['loop']:end=min(end,start+len(pcm)-offset)
        left=max(0,start)
        if end<=left:return
        pan=int(voice.get('pan',0))
        gains=np.array([10.**(-max(0,pan)/2000.),10.**(min(0,pan)/2000.)],np.float32)
        gains*=float(item['unmuted_100_percent_gain'])
        for pos in range(left,end,262144):
            stop=min(end,pos+262144)
            indices=np.arange(pos-start+offset,stop-start+offset)
            if item['loop']:indices%=len(pcm)
            signal[pos:stop]+=pcm[indices]*gains
    for event in sorted(events,key=lambda item:int(item['frame'])):
        effect_id=int(event['id']);sample=int(event['frame'])*735
        if event['command']=='play':
            finish(effect_id,sample);paused.pop(effect_id,None);voices[effect_id]=event;counts[effect_id]+=1
        elif event['command']=='stop':
            if effect_id<0:
                for active in list(voices):finish(active,sample)
                paused.clear()
            else:finish(effect_id,sample);paused.pop(effect_id,None)
        elif event['command'] in ('pause','pan'):
            targets=list(voices) if event['command']=='pause' else [effect_id]
            for active in targets:
                previous=voices.get(active)
                if previous is None:continue
                updated=dict(previous,frame=event['frame'],offset=int(previous.get('offset',0))+sample-int(previous['frame'])*735)
                finish(active,sample)
                if event['command']=='pause':paused[active]=updated
                else:voices[active]=dict(updated,pan=event['pan'])
        elif event['command']=='resume':
            for active,voice in paused.items():voices[active]=dict(voice,frame=event['frame'])
            paused.clear()
    for effect_id in list(voices):finish(effect_id,samples)
    peak=float(np.max(np.abs(signal))) if samples else 0.
    gain=min(1.,.98/max(peak,1e-8))
    energy=0.
    with wave.open(str(output),'wb') as target:
        target.setnchannels(2);target.setsampwidth(2);target.setframerate(44100)
        for pos in range(0,samples,262144):
            block=signal[pos:pos+262144]*gain
            energy+=float(np.square(block).sum())
            target.writeframesraw(np.rint(block*32767).astype('<i2').tobytes())
    return dict(audio=str(Path(output).resolve()),audio_bgm=str(Path(bgm).resolve()),sound_effects=True,
        sound_effect_events=len(events),sound_effect_play_counts={str(k):v for k,v in sorted(counts.items())},
        audio_peak_before_normalization=peak,audio_mix_gain=gain,audio_rms=float(np.sqrt(energy/max(1,samples*2))),
        audio_effect_source='Original local TH14 effects with observed native sound IDs, pan and per-ID attenuation')
