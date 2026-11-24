"""Verify an imported six-stage replay using native Miss/Bomb and ending events."""
from pathlib import Path
import argparse,sys,time,shutil,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--executable',type=Path,required=True);p.add_argument('--decoder-libs',type=Path)
    a=p.parse_args()
    if a.decoder_libs:sys.path.insert(0,str(a.decoder_libs))
    from fly_touhou.th14_human_replay import read_replay
    from fly_touhou.nbnm_run import runtime
    from fly_touhou.thrl_durable_records import write_json
    from fly_touhou.thrl_native_events import HiddenTHRLDialogueGeometryController
    from fly_touhou.thrl_job_paths import cleanup_native_temp
    from capture_nbnm_human_replay import enter_replay
    meta,stages=read_replay(a.replay)
    if (meta['stage_count']!=6 or meta['difficulty']!=3 or (meta['shot'],meta['subshot'])!=(0,0)
            or any(s.metadata['Bomb_pressed_frames'] or s.metadata['continues'] for s in stages)):
        raise ValueError('Expected full Lunatic ReimuA no-Bomb input')
    a.output.mkdir(parents=True,exist_ok=False);temporary=runtime(a.output)
    dest=a.output/'userdata/ShanghaiAlice/th14/replay/th14_01.rpy'
    dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(a.replay,dest)
    c=HiddenTHRLDialogueGeometryController(a.executable,ROOT/'configs/th14-profile.json',
        private_appdata=a.output/'userdata',worker_name='verify-human-rpy',headless=True,
        binary_reader=True,checkpoint_wait_sleep_ms=1,timeout=25)
    started=time.monotonic();history=[];ticks=0
    try:
        c.start();s=enter_replay(c)
        while ticks<180000:
            e=s.extra['thrl_events']
            if s.stage>0 and (not history or s.stage!=history[-1]):history.append(s.stage)
            if e['misses'] or e['bombs_used'] or (s.player is not None and s.player.state==4):
                raise RuntimeError(f'Native replay contact/Bomb at stage{s.stage} frame{s.stage_frame}; {e["misses"]} Miss, {e["bombs_used"]} Bomb')
            ending=bool(s.stage==6 and (s.extra.get('game_thread_flags',0)&0x4000 or
                any(x.get('stage')==6 and x.get('ending') for x in s.extra.get('native_milestones',()))))
            if ending:
                meta['native_revalidated']=True
                report=dict(schema='TH14-original-replay-verification-v1',source=str(a.replay.resolve()),
                    metadata=meta,full_game_NBNM=history==[1,2,3,4,5,6],stage_history=history,
                    Misses=e['misses'],Bombs=e['bombs_used'],native_six_stage_ending=True,
                    end_stage_frame=s.stage_frame,elapsed_seconds=time.monotonic()-started,
                    controller='original Replay playback; human demonstration',network_training=False,
                    student_performance=False,gameplay_modified=False)
                write_json(a.output/'report.json',report);write_json(a.output/'progress.json',dict(phase='complete',**report));return
            if s.extra.get('replay_playback')!=1:raise RuntimeError('Native replay ended before six-stage ending')
            s=c.advance(0,60);ticks+=60
            if ticks%1800==0:
                write_json(a.output/'progress.json',dict(stage=s.stage,stage_frame=s.stage_frame,input_ticks=ticks,
                    stage_history=history,Misses=e['misses'],Bombs=e['bombs_used'],network_training=False))
        raise RuntimeError('Replay did not finish within its verification frame budget')
    except BaseException:
        write_json(a.output/'failure.json',dict(error=traceback.format_exc(),stage_history=history,input_ticks=ticks));raise
    finally:c.close();cleanup_native_temp(temporary)


if __name__=='__main__':main()
