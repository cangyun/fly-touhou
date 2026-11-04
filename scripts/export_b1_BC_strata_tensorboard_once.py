"""Append a sealed BC diagnostic once without restarting the live dashboard."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from fly_touhou.thrl_study_io import read_shared_json as read,utc_now
from fly_touhou.atomic_json_io import atomic_write_json as write

def validated_snapshots(report,result,split,config):
    if report.get('status')!='completed':return []
    if (report.get('schema')!='B1-pre-PPO-BC-six-whole-heldout-trajectory-stratification-v1'
            or result.get('status')!='completed'
            or report.get('actual_BC_sha256')!=result['policy_sha256']
            or report.get('parent_sha256')!=config['parent_sha256']
            or report.get('teacher_manifest_sha256')!=config['teacher_dataset_sha256']
            or report.get('original_heldout_indices')!=split['heldout']
            or report.get('original_heldout_native_seeds')!=split['heldout_native_seeds']
            or len(split['heldout'])!=6 or report.get('independent_test_seeds_used') is not False
            or report.get('optimizer_updates')!=0 or report.get('native_worlds_started')!=0
            or report.get('full_original_prefix_context_recomputed_for_each_policy') is not True):
        raise ValueError('Completed fixed BC diagnostic provenance differs')
    epoch=result['selected_epoch'];labels=result['selected_heldout']['legal_labels']
    if type(epoch) is not int or epoch<=0:raise ValueError('Actual selected BC epoch required')
    if report.get('heldout_scored_labels')!=labels:raise ValueError('Original heldout label count changed')
    rows=[]
    for step,label in [(0,'parent_before_BC'),(epoch,'selected_BC')]:
        check=report['aggregate_reproduction'][label]
        if (check.get('labels_equal') is not True or abs(check['CE_difference'])>3e-5
                or abs(check['agreement_difference'])>2/labels+1e-10):
            raise ValueError('Original aggregate validation must be reproduced')
        groups=report['comparisons'][label]
        if groups['all']['samples']!=labels:raise ValueError('Diagnostic scored another population')
        for group,values in groups.items():
            if type(values['samples']) is not int or values['samples']<0:raise ValueError('Invalid group count')
            for metric in ('action_agreement','hard_teacher_CE','macro_action_recall','largest_label_class_share','samples'):
                value=values[metric]
                if value is None:continue
                if not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
                    raise ValueError('Invalid completed diagnostic metric')
                if metric not in ('samples','hard_teacher_CE') and value>1:raise ValueError('Invalid proportion')
                rows.append(('bc/strata/'+group+'/'+metric,value,step))
    return rows

def publish(a):
    report=read(a.report);result=read(a.fit/'result.json');split=read(a.fit/'split.json');config=read(a.fit/'config.json')
    rows=validated_snapshots(report,result,split,config)
    if not rows:raise ValueError('Incomplete diagnostics do not become dashboard scores')
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    identity=dict(report_sha256=sha(a.report),BC_sha256=result['policy_sha256'])
    if a.receipt.exists():
        if read(a.receipt)['identity']!=identity:raise ValueError('Existing one-time export belongs to another result')
        print(json.dumps(dict(status='already_published',receipt=str(a.receipt))));return
    if sha(result['policy'])!=result['policy_sha256']:raise ValueError('Selected actual BC policy changed')
    from tensorboard.summary.writer.event_file_writer import EventFileWriter
    from tensorboard_live import RunExport
    cfg=read(a.config);logdir=Path(cfg['log_directory'])/a.fit.name
    # A separate event file adds new tags. The live bridge's cursors, writer,
    # service, current training source and scalar histories remain untouched.
    export=object.__new__(RunExport)
    export.writer=EventFileWriter(str(logdir),max_queue_size=100,flush_secs=1)
    export.state={'last_steps':{}};export.new_scalars=0
    try:
        export.custom_layout()
        for tag,value,step in rows:export.scalar(tag,value,step)
        export.text('bc/strata/scope',
            '六条原始整局留出轨迹；仅两个模型快照。横轴0为BC前已经训练过PPO的祖先模型，'
            +str(result['selected_epoch'])+'为选中的BC遍数；中间遍数未测量。'
            '\n\n这里是与教师完整动作（含快慢速）的一致率。原地动作也可能用于避弹；'
            '风险分组来自输入/固定风险读出代理。这些比例都不是原作避弹成功率或通关率。'
            '\n\n此结果属于当前PPO开始前的初始化模型，不代表正在变化的PPO策略。',result['selected_epoch'])
        export.writer.flush()
    finally:export.writer.close()
    receipt=dict(schema='B1-static-BC-stratification-TensorBoard-export-v1',utc=utc_now(),identity=identity,
        report=str(a.report),run=a.fit.name,log_directory=str(logdir),scalar_points=len(rows),
        measured_BC_epochs=[0,result['selected_epoch']],live_bridge_restarted=False,
        live_cursors_modified=False,training_files_modified=False,native_worlds_started=0,optimizer_updates=0)
    write(a.receipt,receipt);print(json.dumps(receipt,ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for name in ('report','fit','config','receipt'):p.add_argument('--'+name,type=Path,required=True)
    publish(p.parse_args())
