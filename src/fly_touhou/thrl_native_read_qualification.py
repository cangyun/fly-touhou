"""Prelaunch evidence for the optional reusable native item-read buffer."""
from pathlib import Path
import hashlib
from .thrl_study_io import read_shared_json as read


def verify(config,runtime):
    if not config.get('native_item_read_qualification'):return None
    path=Path(config['native_item_read_qualification']);r=read(path)
    if (r.get('schema')!='B1-native-same-checkpoint-item-storage-qualification-v1'
            or r.get('status')!='qualified-native-reading-only' or r.get('exact_field_results') is not True
            or r.get('checkpoint_frame_and_RNG_unchanged') is not True
            or r.get('nonempty_checkpoints',0)<1 or len(r.get('samples',[]))<32
            or not all(s.get('item_results_exact') is True and s.get('checkpoint_frame_and_RNG_unchanged') is True
                       for s in r.get('samples',[]))
            or r.get('native_worlds')!=1 or r.get('PPO_updates')!=0):
        raise ValueError('Completed same-checkpoint native item-reader qualification required')
    cleanup=read(path.parent/'cleanup.json')
    if cleanup.get('closed') is not True or cleanup.get('close_errors')!=[]:
        raise ValueError('Item-reader qualification world did not close')
    source=Path(r['plan']['source_runtime']);manifest_path=source/'source_manifest.json'
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if sha(manifest_path)!=r['source_manifest_sha256']:raise ValueError('Qualified source manifest changed')
    declared={x['path'].replace('\\','/'):x['sha256'] for x in read(manifest_path)['files']}
    for name in ['src/fly_touhou/native.py','src/fly_touhou/resources.py']:
        if sha(source/name)!=declared.get(name) or sha(Path(runtime)/name)!=declared.get(name):
            raise ValueError('Training reader differs from qualified native reader: '+name)
    return dict(report=str(path),report_sha256=sha(path),source_manifest_sha256=r['source_manifest_sha256'],
                exact_native_fields=True,complete_game_speedup_not_claimed=True)
