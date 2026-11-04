"""One serial CPU encoder controller; never starts CUDA/native/neural training.

Each child must complete one natural source episode. STOP_NEXT, if explicitly
created by the operator, is checked only between children; no active episode is
interrupted. Failed prefixes are preserved and never retried automatically.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'

ROOT = Path(__file__).resolve().parents[1]
M = ROOT / 'artifacts/engineering/B1-autonomous-dodge-20261024/fly-migration-20261026'
NAMES = ['world000', 'world001', 'world002', 'world003', 'world004', 'world005', 'world007', 'world008']
SCHEMA = 'fixed8-rule-full-visual384-serial-CPU-encoding-v1'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def atomic_json(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    os.replace(temp, path)


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=M / 'teacher-rule-fixed8-visual384-preparation-v1/visual-cache-manifest.json')
    parser.add_argument('--output', type=Path, default=M / 'teacher-rule-fixed8-visual384-CPU-v1')
    parser.add_argument('--execute-cpu', action='store_true')
    a = parser.parse_args(argv)
    import psutil
    process = psutil.Process()
    process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    if 18 not in process.cpu_affinity():
        raise ValueError('CPU18 must be caller-allowed')
    process.cpu_affinity([18])
    source = read(a.manifest)
    entries = source['episodes']
    if [e['episode'] for e in entries] != NAMES:
        raise ValueError('Only the locked original fixed8 sequence is allowed')
    if sha(Path(source['encoder_entry'])) != source['encoder_entry_sha256']:
        raise ValueError('Root location manifest must match current encoder entry')
    for e in entries:
        args = e['encoding_argv_cpu']
        if ('--execute-cpu' not in args or '--execute-cuda' in args
                or args[args.index('--episode') + 1] != e['episode']
                or Path(args[args.index('--output') + 1]).resolve() != Path(e['visual_cache']).resolve()):
            raise ValueError('Only CPU argv with root-owned exact episode/output pointers is allowed')
    a.output.mkdir(parents=True, exist_ok=True)
    snapshot = a.output / 'CPU-location-manifest-snapshot.json'
    if snapshot.exists():
        raise FileExistsError('One fresh controller run; preserve prior run and prefixes')
    atomic_json(snapshot, source)
    status = dict(schema=SCHEMA, status='prepared_not_executed',
        location_manifest=str(a.manifest), location_manifest_sha256=sha(a.manifest),
        encoder_entry_sha256=source['encoder_entry_sha256'],
        process=dict(pid=process.pid, create_time=process.create_time()),
        backend='CPU', threads=1, affinity=[18], priority='BelowNormal', CUDA_VISIBLE_DEVICES='-1',
        current_episode=None, children=[], complete=False, CUDA_calls=0, brain_forwards=0,
        neural_teacher_queries=0, native_actions=0, optimizer_updates=0,
        stop_between_episodes=str(a.output / 'STOP_NEXT'))
    atomic_json(a.output / 'controller-progress.json', status)
    if not a.execute_cpu:
        print(json.dumps(status), flush=True)
        return
    started = time.monotonic()
    try:
        for e in entries:
            if (a.output / 'STOP_NEXT').exists():
                status.update(status='stopped_between_episodes_by_operator', current_episode=None,
                    elapsed_seconds=time.monotonic()-started)
                atomic_json(a.output / 'controller-progress.json', status)
                return
            if sha(Path(source['encoder_entry'])) != source['encoder_entry_sha256']:
                raise ValueError('Encoder changed after root manifest was frozen')
            if Path(e['visual_cache']).exists():
                raise FileExistsError('Preserve an existing output, including unknown/incomplete prefixes')
            status.update(status='encoding', current_episode=e['episode'])
            args = e['encoding_argv_cpu']
            logfile = a.output / (e['episode'] + '.log')
            child = subprocess.Popen(args, cwd=ROOT, env=os.environ.copy(), stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding='utf8', errors='replace',
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            child_process = psutil.Process(child.pid)
            child_record = dict(episode=e['episode'], argv=args, pid=child.pid,
                create_time=child_process.create_time(), log=str(logfile), status='running')
            status['children'].append(child_record)
            atomic_json(a.output / 'controller-progress.json', status)
            print(json.dumps(dict(child_launched=child_record)), flush=True)
            with logfile.open('x', encoding='utf8') as log:
                for line in child.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end='', flush=True)
            code = child.wait()
            child_record['exit_code'] = code
            if code:
                child_record['status'] = 'failed_preserved_prefix_no_retry'
                raise RuntimeError('CPU child failed; no retry: ' + e['episode'])
            root = Path(e['visual_cache'])
            commit = read(root / 'COMMITTED.json')
            index = read(root / 'index.json')
            if (sha(root / 'index.json') != commit['index_sha256'] or index['rows'] != e['rows']
                    or not commit['complete_episode'] or index.get('backend') != 'CPU'
                    or index['source_index_sha256'] != e['source_index_sha256']
                    or index['label_index_sha256'] != e['label_index_sha256']):
                raise ValueError('Completed CPU child cache does not match root source pointers')
            child_record.update(status='completed', rows=e['rows'], COMMITTED_sha256=sha(root / 'COMMITTED.json'),
                index_sha256=commit['index_sha256'], elapsed_seconds=index['elapsed_seconds'], timings=index['timings'])
            status['elapsed_seconds'] = time.monotonic()-started
            atomic_json(a.output / 'controller-progress.json', status)
            print(json.dumps(dict(child_completed=child_record)), flush=True)
        status.update(status='completed_fixed8', current_episode=None, complete=True,
            elapsed_seconds=time.monotonic()-started, controller_sha256=sha(Path(__file__)))
        atomic_json(a.output / 'controller-progress.json', status)
        atomic_json(a.output / 'COMMITTED.json', dict(schema=SCHEMA,
            controller_result_sha256=sha(a.output / 'controller-progress.json'), episodes=8,
            rows=sum(e['rows'] for e in entries), backend='CPU'))
    except Exception as error:
        status.update(status='failed_preserved_prefix_no_retry', error=repr(error),
            elapsed_seconds=time.monotonic()-started)
        atomic_json(a.output / 'controller-progress.json', status)
        atomic_json(a.output / 'controller-failure.json', status)
        raise


if __name__ == '__main__':
    main()
