"""One bounded sequential CPU label preparation on seven newly selected games.

world000 is reused unchanged. No other21 source episodes are opened here.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '2'

ROOT = Path(__file__).resolve().parents[1]
M = ROOT / 'artifacts/engineering/B1-autonomous-dodge-20261024/fly-migration-20261026'
INVENTORY = M / 'teacher-rule-world000-visual384-preparation-v1/episode-index-inventory.json'
NAMES = ('world000', 'world001', 'world002', 'world003', 'world004', 'world005', 'world007', 'world008')
SCHEMA = 'fixed-eight-original-rule-six-stage-signed4f-cache-manifest-v1'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def atomic_json(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    os.replace(temp, path)


def completed_record(entry):
    root = Path(entry['label_cache'])
    commit = read(root / 'COMMITTED.json')
    index = read(root / 'index.json')
    if (sha(root / 'index.json') != commit['index_sha256']
            or sha(root / 'identity.json') != commit['identity_sha256']
            or index['source_index_sha256'] != entry['source_index_sha256']
            or index['rows'] != entry['rows'] or index['complete_episode'] is not True
            or commit['complete_episode'] is not True):
        raise ValueError('Complete exact source-bound label cache required')
    return dict(status='committed', COMMITTED_sha256=sha(root / 'COMMITTED.json'),
        index_sha256=commit['index_sha256'], identity_sha256=commit['identity_sha256'],
        counts=index['counts'], stage_counts=index['stage_counts'], timings=index['timings'],
        original_rule_targets_preserved=index['counts'].get('source_rule_targets_rows', 0) == entry['rows'],
        original_rule_weights_preserved=index['counts'].get('source_teacher_guidance_weight_rows', 0) == entry['rows'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-cpu', action='store_true')
    a = parser.parse_args(argv)
    import psutil
    p = psutil.Process()
    p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    p.cpu_affinity(p.cpu_affinity()[-2:])
    output = M / 'teacher-rule-fixed8-signed4f-preparation-v1'
    output.mkdir(parents=True, exist_ok=True)
    inventory = read(INVENTORY)
    candidates = {r['episode']: r for r in inventory['all_records']}
    episodes = []
    for name in NAMES:
        r = candidates[name]
        if not r['natural_episode_complete'] or not r['reached_all_six_stages'] or r['outcome'] != 'clear':
            raise ValueError('Locked six-stage clear source required')
        episodes.append(dict(episode=name, role='train' if name in NAMES[:6] else 'readout_fit_held',
            source_index=r['index'], source_index_sha256=r['index_sha256'], native_seed=r['native_seed'],
            rows=r['rows'], source_packets=r['packets'],
            label_cache=f'./outputs/distillation/teacher-rule-{name}-signed4f-cache-v1',
            status='prepared'))
    episodes[0].update(completed_record(episodes[0]), reused_without_recalculation=True,
        original_rule_labels_note='Legacy world000 labels omit original targets/weight; source still exists, but no retrofit/re-read/recompute in this job.')
    manifest = dict(schema=SCHEMA, status='prepared', inventory_sha256=sha(INVENTORY),
        process=dict(pid=p.pid, create_time=p.create_time()), episodes=episodes,
        train_episodes=list(NAMES[:6]), held_episodes=list(NAMES[6:]),
        total_rows=sum(e['rows'] for e in episodes), newly_prepared_rows=sum(e['rows'] for e in episodes[1:]),
        heldout_scope='Only new brain/readout supervised-fit trajectory holdout; perception/teacher may have seen them. Not independent end-to-end native evaluation.',
        source_behaviour='resource-v10 rule policy; not D95-operated trajectories',
        CPU=dict(threads=2, affinity=p.cpu_affinity(), below_normal=True, CUDA_VISIBLE_DEVICES='-1'),
        neural_forwards=0, CUDA_calls=0, native_actions=0, optimizer_updates=0,
        source_maps_objects_replicated=False, other21_episode_payloads_opened=False)
    manifest_path = output / 'fixed8-manifest.json'
    atomic_json(manifest_path, manifest)
    if not a.execute_cpu:
        print(json.dumps(dict(status='prepared_not_executed', manifest=str(manifest_path), newly_prepared_rows=manifest['newly_prepared_rows'])))
        return
    spec = importlib.util.spec_from_file_location('original_geometry_label_job', ROOT / 'scripts/prepare_rule_episode_signed4f_cpu.py')
    job = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(job)
    started = time.monotonic()
    try:
        for entry in episodes[1:]:
            if Path(entry['label_cache']).exists():
                raise FileExistsError('Preserve unexpected existing new-episode root; no automatic retry/overwrite')
            entry['status'] = 'running'
            manifest['status'] = 'running'
            manifest['current_episode'] = entry['episode']
            atomic_json(manifest_path, manifest)
            job.main(['--episode', entry['episode'], '--source-index-sha256', entry['source_index_sha256'],
                '--output', entry['label_cache'], '--receipt', str(output / entry['episode'])])
            entry.update(completed_record(entry), reused_without_recalculation=False)
            manifest['elapsed_seconds'] = time.monotonic() - started
            atomic_json(manifest_path, manifest)
            print(json.dumps(dict(fixed8_episode_completed=entry['episode'], rows=entry['rows'],
                elapsed_seconds=manifest['elapsed_seconds'])), flush=True)
        manifest.update(status='completed_fixed8', complete=True, current_episode=None,
            elapsed_seconds=time.monotonic() - started,
            controller_sha256=sha(Path(__file__)), producer_entry_sha256=sha(ROOT / 'scripts/prepare_rule_episode_signed4f_cpu.py'))
        atomic_json(manifest_path, manifest)
        atomic_json(output / 'COMMITTED.json', dict(schema=SCHEMA, fixed8_manifest_sha256=sha(manifest_path),
            episodes=8, reused_episodes=1, new_episodes=7, rows=manifest['total_rows']))
        print(json.dumps(dict(status=manifest['status'], manifest=str(manifest_path),
            fixed8_manifest_sha256=sha(manifest_path), rows=manifest['total_rows'], elapsed_seconds=manifest['elapsed_seconds'])), flush=True)
    except Exception as error:
        manifest.update(status='failed_preserved_partial_no_retry', error=repr(error), elapsed_seconds=time.monotonic()-started)
        atomic_json(manifest_path, manifest)
        atomic_json(output / 'failure.json', dict(schema=SCHEMA, error=repr(error), elapsed_seconds=manifest['elapsed_seconds']))
        raise


if __name__ == '__main__':
    main()
