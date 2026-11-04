"""One-shot export of a sealed development20 result; never a new monitor."""
from pathlib import Path
import argparse
import hashlib
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.thrl_study_io import read_shared_json as read,utc_now
from fly_touhou.thrl_durable_records import write_json as write


def export(source,logdir,receipt):
    source=Path(source);report=read(source/'report.json')
    if (report.get('schema')!='B1-BC-only-frozen-full-story-development-v1'
            or report.get('status')!='completed' or report.get('episodes')!=20
            or report.get('native_games')!=20 or report.get('formal_test_seeds_used') is not False
            or report.get('new_optimizer_updates')!=0 or report.get('native_seed_split')!='development'
            or len(report.get('episode_results',[]))!=20):
        raise ValueError('Only a sealed, frozen neural BC development20 may be exported')
    sha=hashlib.sha256((source/'report.json').read_bytes()).hexdigest()
    if receipt.exists():
        if read(receipt)['report_sha256']!=sha:raise ValueError('Previously exported development report changed')
        return read(receipt)
    from tensorboard.summary.writer.event_file_writer import EventFileWriter
    from tensorboard.compat.proto.event_pb2 import Event
    from tensorboard.compat.proto.summary_pb2 import Summary,SummaryMetadata
    from tensorboard.compat.proto.tensor_pb2 import TensorProto
    from tensorboard.compat.proto.tensor_shape_pb2 import TensorShapeProto
    run=Path(logdir)/source.name;writer=EventFileWriter(str(run))
    try:
        for key in ('episodes','clears','clear_rate','mean_native_frames','mean_max_stage','mean_misses','mean_bombs_used'):
            writer.add_event(Event(wall_time=time.time(),step=20,
                summary=Summary(value=[Summary.Value(tag='BC_full_story_development/final/'+key,simple_value=float(report[key]))])))
        text=('冻结BC神经网络；20局完整自然开局；Lunatic；允许自主Bomb；禁止续关。\n'
            '使用开发种子，非教师成绩、非独立测试，不与其他种子集合直接排名。\n'
            '本次没有PPO或其他优化更新。来源：'+str(source/'report.json'))
        writer.add_event(Event(wall_time=time.time(),step=20,summary=Summary(value=[Summary.Value(
            tag='about/BC_full_story_development',tensor=TensorProto(dtype=7,string_val=[text.encode('utf-8')],
            tensor_shape=TensorShapeProto(dim=[TensorShapeProto.Dim(size=1)])),
            metadata=SummaryMetadata(plugin_data=SummaryMetadata.PluginData(plugin_name='text')))])))
        writer.flush()
    finally:writer.close()
    result=dict(utc=utc_now(),source=str(source),report_sha256=sha,log_directory=str(run),
        development=True,formal_test=False,new_optimizer_updates=0,new_monitors=0)
    write(receipt,result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--logdir',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True)
    a=p.parse_args();print(export(a.source,a.logdir,a.receipt))
