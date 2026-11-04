"""Create an exclusive, checked source copy for a long background experiment."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    output=args.output.resolve()
    if output.exists():
        raise FileExistsError('Frozen source copies are never overwritten')
    files=[]
    for folder in ('src','scripts','profiles','native'):
        files.extend(p for p in (root/folder).rglob('*') if p.is_file()
            and '__pycache__' not in p.parts and (folder=='native' or p.suffix in ('.py','.js','.cjs','.json')))
    # WholeBrainStudent validates this default source relative to its module.
    # The large graph is passed explicitly and is not copied.
    files.append(root/'data/connectome/model.py')
    files.extend(p for p in (root/'COPYING',root/'third_party/rl-reference-20261020/thrl/LICENSE') if p.is_file())
    records=[]
    for path in sorted(files):
        content=path.read_bytes()
        if path.suffix=='.py':
            if b'\0' in content:
                raise ValueError(f'Incomplete source bytes: {path}')
            compile(content,str(path),'exec')
        relative=path.relative_to(root)
        destination=output/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes(content)
        records.append({'path':str(relative),'bytes':len(content),
            'sha256':hashlib.sha256(content).hexdigest()})
    manifest=json.dumps({'schema':'fly-touhou-frozen-source-v1','origin':str(root),'files':records},indent=2)
    (output/'snapshot_manifest.json').write_text(manifest,encoding='utf-8')
    (output/'source_manifest.json').write_text(manifest,encoding='utf-8')
    print(json.dumps({'snapshot':str(output),'files':len(records)}))


if __name__=='__main__':
    main()
