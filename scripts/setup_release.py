"""Download this branch's weights, headless backend and pinned upstream data."""
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
    parser.add_argument('--assets-directory', type=Path, help='Use already downloaded Release attachments')
    parser.add_argument('--data-source', type=Path, help='Reuse an existing complete data/connectome directory')
    parser.add_argument('--connectome', type=Path, default=ROOT/'data/connectome')
    parser.add_argument('--skip-data', action='store_true', help='Install only the weights and native DLL')
    args = parser.parse_args(argv)
    release = json.loads((ROOT/'configs/release.json').read_text(encoding='utf-8'))
    base = f"https://github.com/{release['repository']}/releases/download/{release['tag']}"
    assets = ROOT/'outputs/downloads'
    assets.mkdir(parents=True, exist_ok=True)
    for name, target in [(release['checkpoint'], ROOT/release['checkpoint_path']),
                         (release['native_asset'], assets/release['native_asset'])]:
        target.parent.mkdir(parents=True, exist_ok=True)
        if args.assets_directory:
            shutil.copy2(args.assets_directory/name, target)
        else:
            download(base+'/'+name, target)
    install_native(assets/release['native_asset'])
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
