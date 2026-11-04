"""One serial CPU data extension: new world009..018 labels, then visual384.

The eight existing caches are referenced unchanged. No model training or native
process is created; the only neural computation is frozen CPU perception.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_name] = '1'

ROOT = Path(__file__).resolve().parents[1]
M = ROOT / 'artifacts/engineering/B1-autonomous-dodge-20261024/fly-migration-20261026'
OUTPUT = M / 'teacher-rule-train16-data-extension-v1'
INVENTORY = M / 'teacher-rule-world000-visual384-preparation-v1/episode-index-inventory.json'
BASE_LABELS = M / 'teacher-rule-fixed8-signed4f-preparation-v1/fixed8-manifest.json'
BASE_VISUAL = M / 'teacher-rule-fixed8-visual384-CPU-v1/completed-cache-manifest.json'
TRAIN = [f'world{i:03d}' for i in range(6)] + [f'world{i:03d}' for i in range(9, 19)]
HELD = ['world007', 'world008']
NEW = [f'world{i:03d}' for i in range(9, 19)]
SCHEMA = 'rule-six-stage-16train2held-visual384-cache-locations-v1'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    os.replace(tmp, path)


def complete_label(entry):
    root = Path(entry['label_cache'])
    commit, index = read(root / 'COMMITTED.json'), read(root / 'index.json')
    if (sha(root / 'index.json') != commit['index_sha256']
            or sha(root / 'identity.json') != commit['identity_sha256']
            or index['source_index_sha256'] != entry['source_index_sha256']
            or index['rows'] != entry['rows'] or not index['complete_episode']
            or not commit['complete_episode'] or index['retained_horizons_native_frames'] != [4, 12, 24]):
        raise ValueError('Complete exact-source three-horizon label cache required')
    entry.update(status='label_committed', label_index_sha256=commit['index_sha256'],
        label_identity_sha256=commit['identity_sha256'], label_COMMITTED_sha256=sha(root / 'COMMITTED.json'),
        original_rule_targets_preserved=index['counts'].get('source_rule_targets_rows', 0) == entry['rows'],
        original_rule_weights_preserved=index['counts'].get('source_teacher_guidance_weight_rows', 0) == entry['rows'],
        label_counts=index['counts'], label_stage_counts=index['stage_counts'],
        label_process=index['process'], label_seconds=index['timings']['total_CPU_wall_seconds'])


def complete_visual(entry):
    root = Path(entry['visual_cache'])
    commit, index = read(root / 'COMMITTED.json'), read(root / 'index.json')
    if (sha(root / 'index.json') != commit['index_sha256']
            or sha(root / 'identity.json') != commit['identity_sha256']
            or index['source_index_sha256'] != entry['source_index_sha256']
            or index['label_index_sha256'] != entry['label_index_sha256']
            or index['rows'] != entry['rows'] or not index['complete_episode']
            or not commit['complete_episode'] or index['backend'] != 'CPU'):
        raise ValueError('Complete exact source/label-bound CPU visual history required')
    entry.update(status='committed', visual_index_sha256=commit['index_sha256'],
        visual_identity_sha256=commit['identity_sha256'], visual_COMMITTED_sha256=sha(root / 'COMMITTED.json'),
        visual_process=index['process'], visual_seconds=index['elapsed_seconds'],
        visual_npz_bytes=sum(p['bytes'] for p in index['packets']))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-cpu', action='store_true')
    a = parser.parse_args(argv)
    import psutil
    process = psutil.Process()
    process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    if 18 not in process.cpu_affinity():
        raise ValueError('CPU18 must be caller-allowed')
    process.cpu_affinity([18])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if (OUTPUT / 'controller-state.json').exists():
        raise FileExistsError('Preserve prior controller state; no automatic resume or retry')
    inventory = {e['episode']: e for e in read(INVENTORY)['all_records']}
    base_labels = {e['episode']: e for e in read(BASE_LABELS)['episodes']}
    base_visual = {e['episode']: e for e in read(BASE_VISUAL)['episodes']}
    python = str(ROOT / '.venv/Scripts/python.exe')
    label_script = ROOT / 'scripts/prepare_rule_episode_signed4f_cpu.py'
    visual_script = ROOT / 'scripts/encode_rule_episode_visual384.py'
    entries = []
    for name in TRAIN + HELD:
        src = inventory[name]
        if not src['natural_episode_complete'] or not src['reached_all_six_stages'] or src['outcome'] != 'clear':
            raise ValueError('Locked complete six-stage clear metadata required')
        entry = dict(episode=name, role='train' if name in TRAIN else 'readout_fit_held',
            source_index=src['index'], source_index_sha256=src['index_sha256'],
            native_seed=src['native_seed'], rows=src['rows'], source_packets=src['packets'])
        if name in base_labels:
            lab, vis = base_labels[name], base_visual[name]
            entry.update(status='committed', reused_existing=True,
                label_cache=lab['label_cache'], label_index_sha256=lab['index_sha256'],
                label_identity_sha256=lab['identity_sha256'], label_COMMITTED_sha256=lab['COMMITTED_sha256'],
                original_rule_targets_preserved=lab['original_rule_targets_preserved'],
                original_rule_weights_preserved=lab['original_rule_weights_preserved'],
                label_counts=lab['counts'], label_stage_counts=lab['stage_counts'],
                retained_horizons_native_frames=[4],
                visual_cache=vis['visual_cache'], visual_index_sha256=vis['index_sha256'],
                visual_identity_sha256=vis['identity_sha256'], visual_COMMITTED_sha256=vis['COMMITTED_sha256'])
        else:
            label_root = f'./outputs/distillation/teacher-rule-{name}-signed4f-cache-v1'
            visual_root = f'./outputs/distillation/teacher-rule-train16-visual384-v1/{name}'
            entry.update(status='pending', reused_existing=False, label_cache=label_root, visual_cache=visual_root,
                retained_horizons_native_frames=[4, 12, 24], original_rule_targets_preserved=None,
                original_rule_weights_preserved=None,
                geometry_argv=[python, '-B', str(label_script), '--episode', name,
                    '--source-index-sha256', src['index_sha256'], '--output', label_root,
                    '--receipt', str(OUTPUT / name / 'geometry'), '--cpu-threads', '1', '--cpu-id', '18',
                    '--preserve-all-horizons'],
                encoding_argv_cpu=[python, '-B', str(visual_script), '--episode', name, '--label-cache', label_root,
                    '--output', visual_root, '--execute-cpu', '--max-seconds', '21600',
                    '--preparation', str(OUTPUT / name / 'visual')])
        entries.append(entry)
    manifest = dict(schema=SCHEMA, status='prepared_pending_extension', complete=False,
        train_episodes=TRAIN, held_episodes=HELD, episodes=entries,
        total_rows=sum(e['rows'] for e in entries), new_rows=sum(inventory[n]['rows'] for n in NEW),
        train_rows=sum(inventory[n]['rows'] for n in TRAIN), held_rows=sum(inventory[n]['rows'] for n in HELD),
        base_labels_manifest=str(BASE_LABELS), base_labels_manifest_sha256=sha(BASE_LABELS),
        base_visual_manifest=str(BASE_VISUAL), base_visual_manifest_sha256=sha(BASE_VISUAL),
        original_inventory_sha256=sha(INVENTORY), label_entry_sha256=sha(label_script),
        encoder_entry_sha256=sha(visual_script), controller_entry_sha256=sha(Path(__file__)),
        source_behaviour='resource-v10 rule policy; not D95-operated trajectories',
        heldout_scope='New brain/readout fit trajectory holdout only; perception/teacher may have seen these. Not independent native testing.',
        current_epoch_or_live_training_modified=False, existing8_recomputed_or_modified=False,
        native_counterfactual_labels=False, factual_outcomes='unknown -1; real clocks preserved, no frame interpolation',
        CPU=dict(threads=1, affinity=[18], priority='BelowNormal', CUDA_VISIBLE_DEVICES='-1'),
        new_GPU_calls=0, brain_forwards=0, neural_teacher_queries=0, native_actions=0, optimizer_updates=0)
    path = OUTPUT / 'visual-cache-manifest.json'
    atomic_json(path, manifest)
    state = dict(schema=SCHEMA, status='prepared', process=dict(pid=process.pid, create_time=process.create_time()),
        current_episode=None, current_phase=None, children=[], complete=False)
    atomic_json(OUTPUT / 'controller-state.json', state)
    if not a.execute_cpu:
        print(json.dumps(dict(status='prepared_not_executed', manifest=str(path))), flush=True)
        return
    started = time.monotonic()

    def child_job(entry, phase, args):
        state.update(status='running', current_episode=entry['episode'], current_phase=phase)
        manifest.update(status='running_data_extension', current_episode=entry['episode'], current_phase=phase)
        atomic_json(path, manifest)
        logpath = OUTPUT / entry['episode'] / (phase + '.log')
        logpath.parent.mkdir(parents=True, exist_ok=True)
        child = subprocess.Popen(args, cwd=ROOT, env=os.environ.copy(), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding='utf8', errors='replace',
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        record = dict(episode=entry['episode'], phase=phase, argv=args, pid=child.pid,
            create_time=psutil.Process(child.pid).create_time(), log=str(logpath), status='running')
        state['children'].append(record)
        atomic_json(OUTPUT / 'controller-state.json', state)
        print(json.dumps(dict(child_launched=record)), flush=True)
        with logpath.open('x', encoding='utf8') as log:
            for line in child.stdout:
                log.write(line); log.flush(); print(line, end='', flush=True)
        code = child.wait()
        record.update(exit_code=code, status='completed' if code == 0 else 'failed_preserved_prefix_no_retry')
        atomic_json(OUTPUT / 'controller-state.json', state)
        if code:
            raise RuntimeError('Data child failed; no retry: ' + entry['episode'] + '/' + phase)

    try:
        for name in NEW:
            entry = next(e for e in entries if e['episode'] == name)
            if (OUTPUT / 'STOP_NEXT').exists():
                state.update(status='stopped_between_episodes_by_operator', current_episode=None, current_phase=None)
                atomic_json(OUTPUT / 'controller-state.json', state)
                return
            if Path(entry['label_cache']).exists() or Path(entry['visual_cache']).exists():
                raise FileExistsError('Preserve unexpected existing new source prefix; no overwrite')
            if sha(label_script) != manifest['label_entry_sha256'] or sha(visual_script) != manifest['encoder_entry_sha256']:
                raise ValueError('Production entry changed during the fixed CPU workflow')
            child_job(entry, 'geometry', entry['geometry_argv'])
            complete_label(entry)
            atomic_json(path, manifest)
            child_job(entry, 'visual', entry['encoding_argv_cpu'])
            complete_visual(entry)
            manifest['elapsed_seconds'] = time.monotonic()-started
            atomic_json(path, manifest)
            print(json.dumps(dict(episode_completed=name, rows=entry['rows'], elapsed_seconds=manifest['elapsed_seconds'])), flush=True)
        manifest.update(status='completed_train16_data_extension', complete=True,
            current_episode=None, current_phase=None, elapsed_seconds=time.monotonic()-started)
        atomic_json(path, manifest)
        state.update(status=manifest['status'], complete=True, current_episode=None,
            current_phase=None, elapsed_seconds=manifest['elapsed_seconds'])
        atomic_json(OUTPUT / 'controller-state.json', state)
        atomic_json(OUTPUT / 'COMMITTED.json', dict(schema=SCHEMA, manifest_sha256=sha(path), complete=True,
            episodes=18, train_episodes=16, held_episodes=2, new_episodes=10, reused_episodes=8, rows=manifest['total_rows']))
        print(json.dumps(dict(status=manifest['status'], manifest=str(path), manifest_sha256=sha(path),
            rows=manifest['total_rows'], elapsed_seconds=manifest['elapsed_seconds'])), flush=True)
    except Exception as error:
        state.update(status='failed_preserved_prefix_no_retry', error=repr(error), elapsed_seconds=time.monotonic()-started)
        manifest.update(status=state['status'], error=repr(error), elapsed_seconds=state['elapsed_seconds'])
        atomic_json(OUTPUT / 'controller-state.json', state)
        atomic_json(path, manifest)
        atomic_json(OUTPUT / 'failure.json', state)
        raise


if __name__ == '__main__':
    main()
