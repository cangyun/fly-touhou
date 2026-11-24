"""Frozen NBNM checkpoint evaluation with live progress and cached pairing."""
from pathlib import Path
import argparse
import json
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--resume-bookmark',type=Path,help='Resume a saved frozen Story stage boundary')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--task',help='Override the configured primary task, e.g. full_story')
    parser.add_argument('--cache-pair-report',type=Path)
    parser.add_argument('--split',choices=('development','test'),
                        help='Use the configured original-game seeds for this split')
    parser.add_argument('--opening-stage',type=int,choices=range(1,7),
                        help='Development diagnostic ending at the first observed native midboss')
    parser.add_argument('--through-stage',type=int,choices=range(1,6),
                        help='Play from original Story opening through this stage or first contact')
    args=parser.parse_args()
    args.resume_entry = None
    if args.resume_bookmark is not None:
        if any((args.config,args.checkpoint,args.task,args.cache_pair_report,args.opening_stage,args.through_stage,args.split)):
            parser.error('A bookmark supplies its saved parameters, configuration, seed, and split')
        from fly_touhou.nbnm_story_bookmark import load
        args.resume_entry = load(args.resume_bookmark)
        args.config = Path(args.resume_entry['configuration'])
        args.checkpoint = Path(args.resume_entry['checkpoint'])
        args.task, args.split = 'full_story', args.resume_entry['split']
    else:
        if args.config is None or args.checkpoint is None:
            parser.error('Supply --config and --checkpoint, or --resume-bookmark')
        args.split = args.split or 'development'
    if args.split!='development' and (args.opening_stage is not None or args.through_stage is not None):
        parser.error('Development endpoints are separate from held-out full-task evaluation')
    if args.opening_stage is not None and args.cache_pair_report is not None:
        parser.error('A midboss diagnostic cannot populate a complete task comparison')
    if args.through_stage is not None and (args.opening_stage is not None or args.cache_pair_report is not None):
        parser.error('The stage-clear comparison is a separate development endpoint')
    from fly_touhou.nbnm_job_phase import gpu_phase
    with gpu_phase('evaluation'):
        return run(args, parser)


def run(args, parser):
    from fly_touhou.nbnm_run import read,runtime,load_model,evaluate
    from fly_touhou.thrl_durable_records import write_json
    from fly_touhou.thrl_job_paths import cleanup_native_temp
    import torch,psutil
    args.output.mkdir(parents=True,exist_ok=False)
    spec=read(args.config)
    if args.task is not None and args.task not in spec.get('tasks',{}):
        parser.error('Unknown registered task '+args.task)
    task=spec['tasks'][args.task] if args.task is not None else spec['nbnm']['task']
    temporary=runtime(args.output);started=time.monotonic();proc=psutil.Process()
    proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    def progress(row):
        write_json(args.output/'progress.json',dict(row,PID=proc.pid,create_time=proc.create_time(),
            elapsed_seconds=time.monotonic()-started,network_training=False,checkpoint=str(args.checkpoint)))
    class StopRequested(Exception):
        pass
    def guard():
        if (args.output/'STOP').exists():
            raise StopRequested()
    try:
        torch.manual_seed(spec['seed']);torch.cuda.manual_seed_all(spec['seed'])
        model,_=load_model(spec,args.checkpoint)
        report=evaluate(model,spec,args.output/'native-result',task=task,split=args.split,progress=progress,
            opening_stage=args.opening_stage,through_stage=args.through_stage,checkpoint=args.checkpoint,
            resume_bookmark=args.resume_entry,guard=guard)
        if task.get('mode')=='full_story' and args.split=='test':
            from fly_touhou.nbnm_run import full_story_qualification
            report['full_story_qualification']=full_story_qualification(report,spec)
        write_json(args.output/'report.json',report)
        if args.cache_pair_report is not None and not args.cache_pair_report.exists():
            args.cache_pair_report.mkdir(parents=True,exist_ok=False)
            # Only completed genuine native evaluations may populate a cache.
            write_json(args.cache_pair_report/'entry-observations.json',read(args.output/'native-result/entry-observations.json'))
            write_json(args.cache_pair_report/'report.json',dict(report,native_evaluation_source_dir=str(args.output)))
        count='stage_passes' if args.through_stage is not None else 'opening_passes' if args.opening_stage is not None else 'successes'
        progress(dict(phase='complete',attempts=report['attempts'],**{count:report[count]}))
    except StopRequested:
        write_json(args.output/'report.json',dict(operator_requested_stop=True,attempts=0))
        progress(dict(phase='stopped'))
    except BaseException:
        write_json(args.output/'failure.json',dict(error=traceback.format_exc()));raise
    finally:
        cleanup_native_temp(temporary)


if __name__=='__main__':main()
