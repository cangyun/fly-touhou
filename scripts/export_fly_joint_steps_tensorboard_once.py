"""Append completed first-Adam diagnostics to the existing TensorBoard run.

One-shot JSON consumer only: no Torch, trainer edits, process control or service.
The cursor belongs to this supplement, not the existing live exporter.
"""
from pathlib import Path
import argparse
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('run','control','configuration'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--label',required=True)
    a=parser.parse_args()
    from fly_touhou.thrl_study_io import read_shared_json
    from fly_touhou.thrl_durable_records import write_json
    from tensorboard.summary.writer.event_file_writer import EventFileWriter
    from tensorboard.compat.proto.event_pb2 import Event
    from tensorboard.compat.proto.summary_pb2 import Summary
    config=read_shared_json(a.configuration)
    logdir=Path(config['log_directory'])/Path(config['fly_algorithm_source_directory']).name
    cursor=a.control/'first-step-tensorboard-export.json'
    state=read_shared_json(cursor) if cursor.exists() else dict(source=str(a.run.resolve()),last_rollout=0,scalar_events=0)
    if state['source']!=str(a.run.resolve()):raise ValueError('Supplement cursor belongs to another job')
    writer=None;new_events=0
    try:
        for path in sorted((a.run/'traces').glob('update-*.json')):
            number=int(path.stem.rsplit('-',1)[-1])
            if number<=state['last_rollout']:continue
            row=read_shared_json(path);learning=row['learning']
            gradients=learning.get('first_update_gradient_groups',[])
            movement=learning.get('first_update_parameter_movement',[])
            if not gradients or not movement:continue
            step=int(row['costs']['student_actions_executed'])
            scalars=[]
            for entry in gradients:
                for key in ('preclip_L2','preclip_RMS','all_finite'):
                    tag=f"fly_iterations/{a.label}/first_Adam/{entry['group']}/{key}"
                    scalars.append(Summary.Value(tag=tag,simple_value=float(entry[key])))
            for entry in movement:
                for key in ('parameter_delta_L2','parameter_delta_RMS','changed_tensors'):
                    tag=f"fly_iterations/{a.label}/first_Adam/{entry['group']}/{key}"
                    scalars.append(Summary.Value(tag=tag,simple_value=float(entry[key])))
            if writer is None:writer=EventFileWriter(str(logdir))
            writer.add_event(Event(wall_time=time.time(),step=step,summary=Summary(value=scalars)))
            writer.flush();new_events+=len(scalars)
            state.update(last_rollout=number,scalar_events=state['scalar_events']+len(scalars),
                log_directory=str(logdir),axis='real collected student decisions;first Adam of each rollout only',
                new_services=0,Torch_imported='torch' in sys.modules)
            write_json(cursor,state)
    finally:
        if writer is not None:writer.close()
    print(dict(last_rollout=state['last_rollout'],new_scalar_events=new_events,Torch_imported='torch' in sys.modules,new_services=0))


if __name__=='__main__':main()
