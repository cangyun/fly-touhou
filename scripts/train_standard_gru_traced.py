"""Unchanged PPO entry plus safe JSON writer and first-eight read-only observer."""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.atomic_json_io import atomic_write_json
from fly_touhou.ppo_rollout_trace import RolloutTraceObserver
import fly_touhou.ppo_rollout_trace as trace_module

ENTRY_SHA='e8e90087a7a2583beb511d6c2c83d847b2535b6c492f908f74344f31d4af48ad'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_forwarded(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--receipt',type=Path,required=True)
    options,forwarded=parser.parse_known_args(argv)
    return options,forwarded[1:] if forwarded[:1]==['--'] else forwarded


def install_observer(entry,observers):
    original=entry.Progress
    class TracedProgress(original):
        def __init__(self,output,*args,**kwargs):
            super().__init__(output,*args,**kwargs)
            self.diagnostic_observer=RolloutTraceObserver(output,entry.save_training_checkpoint,
                entry.save_training_checkpoint.__globals__['capture_rng_state'],entry.optimizer_steps)
            observers.append(self.diagnostic_observer)
        def _on_training_start(self):
            super()._on_training_start()
            self.diagnostic_observer.initial_decisions=int(self.model.num_timesteps)
        def _on_rollout_end(self):
            # Original callback exceptions are never converted to optional IO failures.
            super()._on_rollout_end()
            self.diagnostic_observer.capture(self.model)
    entry.write_json=atomic_write_json
    entry.Progress=TracedProgress


def main(argv=None):
    options,forwarded=parse_forwarded(argv)
    path=ROOT/'scripts/train_standard_gru.py'
    if digest(path)!=ENTRY_SHA:
        raise ValueError('Original PPO entry differs from pinned unchanged source')
    receipt=options.receipt.resolve()
    if '--output' in forwarded:
        output=Path(forwarded[forwarded.index('--output')+1]).resolve()
        if receipt.is_relative_to(output):
            raise ValueError('Wrapper receipt must be outside exclusive training output')
    record={'schema':'unchanged-PPO-with-bounded-pre-update-observer-v1','status':'starting',
        'started_utc':datetime.now(timezone.utc).isoformat(),'entry':str(path),'entry_sha256':ENTRY_SHA,
        'wrapper':str(Path(__file__).resolve()),'wrapper_sha256':digest(__file__),
        'trace_source':str(Path(trace_module.__file__).resolve()),'trace_source_sha256':digest(trace_module.__file__),
        'forwarded_arguments':forwarded,'only_original_writer_and_Progress_subclass_replaced':True,
        'algorithm_or_hyperparameters_modified':False,'trace_hard_cap':8,'trace_phase':'pre_update_diagnostic',
        'observer_failure':'IO-only fail-open and disable; read-only violations and original exceptions propagate',
        'diagnostic_is_regular_training_resume':False,'auto_restart':False}
    receipt.parent.mkdir(parents=True,exist_ok=True)
    with receipt.open('x',encoding='utf-8') as stream:
        json.dump(record,stream,ensure_ascii=False,indent=2,allow_nan=False)
    observers=[]
    try:
        spec=importlib.util.spec_from_file_location('reviewed_traced_training_entry',path)
        entry=importlib.util.module_from_spec(spec)
        sys.modules[spec.name]=entry
        spec.loader.exec_module(entry)
        install_observer(entry,observers)
        sys.argv=[str(path),*forwarded]
        entry.main()
        record['status']='finished'
    except SystemExit as error:
        record.update(status='finished' if error.code in (None,0) else 'failed',exit_code=error.code)
        raise
    except BaseException as error:
        record.update(status='failed',exception=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        record.update(finished_utc=datetime.now(timezone.utc).isoformat(),observer_summaries=[value.summary() for value in observers])
        atomic_write_json(receipt,record)


if __name__=='__main__':
    main()
