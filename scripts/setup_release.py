"""Prepare pinned graph data and verify the published runtime assets."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def download(url, destination, expected_sha256=None):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if expected_sha256 and hashlib.sha256(destination.read_bytes()).hexdigest() != expected_sha256:
            raise ValueError(f'Existing data differs from the pinned version: {destination}')
        print('Exists:', destination, flush=True)
        return
    temporary = destination.with_name(destination.name+'.part')
    print('Downloading:', url, flush=True)
    request = urllib.request.Request(url, headers={'User-Agent': 'fly-touhou-release-setup'})
    digest = hashlib.sha256()
    with urllib.request.urlopen(request, timeout=60) as response, temporary.open('wb') as output:
        while chunk := response.read(1024*1024):
            output.write(chunk)
            digest.update(chunk)
    if expected_sha256 and digest.hexdigest() != expected_sha256:
        raise ValueError(f'Downloaded data differs from the pinned version: {url}')
    temporary.replace(destination)


def install_native(archive, root=ROOT):
    root = Path(root).resolve()
    allowed = {'native/th14_fake_d3d9.dll', 'native/th14_fake_d3d9.dll.sha256',
               'native/fake_d3d9/LICENSE-MIT', 'native/fake_d3d9/LICENSE-APACHE',
               'native/fake_d3d9/NOTICE.md'}
    with zipfile.ZipFile(archive) as source:
        if set(source.namelist()) != allowed:
            raise ValueError('Unexpected native archive contents')
        for name in source.namelist():
            target = root/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read(name))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets-directory', type=Path, help='Import compatible assets from a local historical archive')
    parser.add_argument('--data-source', type=Path, help='Reuse an existing complete data/connectome directory')
    parser.add_argument('--connectome', type=Path, default=ROOT/'data/connectome')
    parser.add_argument('--skip-data', action='store_true', help='Only check the packaged weights and native DLLs')
    args = parser.parse_args(argv)
    release = json.loads((ROOT/'configs/release.json').read_text(encoding='utf-8'))
    if args.assets_directory:
        target = ROOT/release['checkpoint_path']
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.assets_directory/release['checkpoint'], target)
        install_native(args.assets_directory/release['native_asset'])
    weights = ROOT/'weights/published'
    manifest = json.loads((weights/'manifest.json').read_text(encoding='utf-8'))
    for entry in manifest:
        weight = weights/entry['file']
        if not weight.is_file() or weight.stat().st_size != entry['bytes']:
            parser.error(f'Missing {weight.name}. Run git lfs pull.')
        with weight.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != entry['sha256']:
                parser.error(f'Published weight differs from its manifest: {weight.name}')
    for backend in (ROOT/'native/th14_fake_d3d9.dll', ROOT/'native/th08/d3d8.dll'):
        if not backend.is_file():
            parser.error(f'Native dependency is missing: {backend}')
    if not args.skip_data:
        directory = args.connectome.resolve()
        directory.mkdir(parents=True, exist_ok=True)
        sources = json.loads((ROOT/'release-data/annotation-sources.json').read_text(encoding='utf-8'))
        if args.data_source:
            pairs = [(name, name) for name in ('Completeness_783.csv', 'Connectivity_783.parquet', 'LICENSE', 'model.py')]
            pairs += [('annotations/'+f['name'], 'annotations/'+f['name']) for f in sources['files']]
            for source, target in pairs:
                target = directory/target
                target.parent.mkdir(parents=True, exist_ok=True)
                if (args.data_source/source).resolve() != target.resolve():
                    shutil.copy2(args.data_source/source, target)
        else:
            subprocess.run([sys.executable, '-B', str(ROOT/'scripts/fetch_connectome.py'),
                            '--directory', str(directory), '--full', '--skip-prepare',
                            '--manifest', str(ROOT/'outputs/connectome-manifest.json')], cwd=ROOT, check=True)
            for item in sources['files']:
                download(item['url'], directory/'annotations'/item['name'], item['sha256'])
        port_target = directory/'annotations/selected_ports_1024_v783.json'
        port_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/'release-data/selected_ports_1024_v783.json', port_target)
    print('Setup complete. No game or training process was started.')


if __name__ == '__main__':
    main()
