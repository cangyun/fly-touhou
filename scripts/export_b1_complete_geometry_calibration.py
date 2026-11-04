"""One-shot export of completed geometric calibration, separate from native scores."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fly_touhou.thrl_study_io import read_shared_json as read, utc_now
from fly_touhou.thrl_dodge_diagnostics import _shared_binary
from fly_touhou.thrl_durable_records import write_json as write


def export(source, logdir, receipt):
    source, logdir, receipt = Path(source), Path(logdir), Path(receipt)
    with _shared_binary(source / 'result.json') as stream:
        result_bytes = stream.read()
    with _shared_binary(source / 'risk_fit_history.jsonl') as stream:
        history_bytes = stream.read()
    result = json.loads(result_bytes)
    history = [json.loads(line) for line in history_bytes.splitlines() if line.strip()]
    if (result.get('schema') != 'B1-full-geometry-point-risk-calibration-result-v1'
            or result.get('status') != 'completed'
            or result.get('actual_native_worlds_started') != 0
            or result.get('PPO_optimizer_updates') != 0
            or result.get('actor_GRU_critic_laser_and_rewards_unchanged') is not True
            or len(history) != result.get('epochs_executed')
            or [row['epoch'] for row in history] != list(range(1, len(history) + 1))
            or history[-1]['risk_fit_Adam_steps'] != result.get('paid_risk_fit_Adam_steps')):
        raise ValueError('Only a completed offline geometry calibration may be exported')
    selected = history[result['selected_epoch'] - 1]
    if (selected['groups'] != result['selected']['groups']
            or selected['risk_fit_Adam_steps'] != result['selected_risk_fit_Adam_steps']):
        raise ValueError('Selected checkpoint metrics do not match their immutable history')
    hashes = dict(result_sha256=hashlib.sha256(result_bytes).hexdigest(),
                  history_sha256=hashlib.sha256(history_bytes).hexdigest())
    if receipt.exists():
        previous = read(receipt)
        if any(previous[key] != value for key, value in hashes.items()):
            raise ValueError('Previously exported calibration inputs changed')
        return previous

    from tensorboard.summary.writer.event_file_writer import EventFileWriter
    from tensorboard.compat.proto.event_pb2 import Event
    from tensorboard.compat.proto.summary_pb2 import Summary, SummaryMetadata
    from tensorboard.compat.proto.tensor_pb2 import TensorProto
    from tensorboard.compat.proto.tensor_shape_pb2 import TensorShapeProto
    from tensorboard.plugins.custom_scalar import layout_pb2, metadata

    run = logdir / 'offline-point-risk-full-geometry-calibration-v1'
    if run.exists() and any(run.iterdir()):
        raise ValueError('Unreceipted event directory exists; inspect rather than append duplicates')
    writer = EventFileWriter(str(run))
    scalar_count = 0
    try:
        charts = [
            ('危险对召回：正常与放大人物（几何预测）', 'dangerous_pair_recall'),
            ('边界误差：正常与放大人物（几何预测）', 'boundary_MAE'),
            ('安全对误报：正常与放大人物（几何预测）', 'safe_false_positive_rate'),
        ]
        layout = layout_pb2.Layout(category=[layout_pb2.Category(
            title='离线尺寸校准：几何代理指标，非原作通关率',
            chart=[layout_pb2.Chart(title=title,
                multiline=layout_pb2.MultilineChartContent(tag=[
                    f'risk_geometry/{group}/{metric}' for group in ('normal', 'enlarged')]))
                for title, metric in charts])])
        writer.add_event(Event(wall_time=time.time(), step=0, summary=Summary(value=[
            Summary.Value(tag=metadata.CONFIG_SUMMARY_TAG,
                metadata=metadata.create_summary_metadata(),
                tensor=TensorProto(dtype=7, string_val=[layout.SerializeToString()]))])))
        initial = dict(epoch=0, risk_fit_Adam_steps=0, **result['initial'])
        for row in [initial, *history]:
            values = {}
            for group, group_values in row['groups'].items():
                for key, value in group_values.items():
                    values[f'risk_geometry/{group}/{key}'] = value
            values['risk_geometry/selection_score'] = row['selection_score']
            values['risk_fit/Adam_steps'] = row['risk_fit_Adam_steps']
            if 'learning_rate' in row:
                values['risk_fit/learning_rate'] = row['learning_rate']
            summaries = []
            for tag, value in values.items():
                if isinstance(value, (int, float)) and math.isfinite(value):
                    summaries.append(Summary.Value(tag=tag, simple_value=float(value)))
            scalar_count += len(summaries)
            writer.add_event(Event(wall_time=time.time(), step=row['epoch'],
                summary=Summary(value=summaries)))
        scope = (
            '离线点弹幕风险分支校准；横轴为校准epoch，0为校准前。\n'
            '目标为当前速度与碰撞尺寸的几何预测，包含单独标记的尺寸增广；'
            '不是原作反事实动作结果，也不是通关率或实机避弹召回。\n'
            f'执行{result["epochs_executed"]} epoch / {result["paid_risk_fit_Adam_steps"]}次Adam；'
            f'按保留几何集选定epoch {result["selected_epoch"]}。\n'
            '仅8个点风险权重张量发生变化；演员、GRU、价值、激光分支、奖励均未改变。\n'
            '原作效果由另行冻结的baseline/candidate配对开发局评判；本导出不新增训练或游戏实例。\n'
            '来源：' + str(source / 'result.json'))
        writer.add_event(Event(wall_time=time.time(), step=result['selected_epoch'],
            summary=Summary(value=[Summary.Value(tag='about/offline_geometry_calibration',
                tensor=TensorProto(dtype=7, string_val=[scope.encode('utf-8')],
                    tensor_shape=TensorShapeProto(dim=[TensorShapeProto.Dim(size=1)])),
                metadata=SummaryMetadata(plugin_data=SummaryMetadata.PluginData(plugin_name='text')))])))
        writer.flush()
    finally:
        writer.close()
    exported = dict(schema='B1-completed-geometry-calibration-TensorBoard-export-v1',
        utc=utc_now(), source=str(source), log_directory=str(run), **hashes,
        epochs=len(history), selected_epoch=result['selected_epoch'],
        event_scalar_values=scalar_count, native_outcomes=False,
        new_optimizer_updates=0, new_original_worlds=0, new_monitors=0)
    write(receipt, exported)
    return exported


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--logdir', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    arguments = parser.parse_args()
    print(export(arguments.source, arguments.logdir, arguments.receipt))
