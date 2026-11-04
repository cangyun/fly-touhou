"""Build the separately licensed, process-local x86 software D3D9 implementation.

Uses an explicit portable toolchain without modifying machine/user PATH.
Compilation and caches stay on the supplied output volume, outside C: caches.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--toolchain', type=Path, default=Path('./outputs/toolchains/rust-portable'))
    parser.add_argument('--target-dir', type=Path, default=Path('./outputs/toolchains/fake-d3d9-target'))
    parser.add_argument('--msvc', type=Path, default=Path('C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.41.34120'))
    parser.add_argument('--sdk', type=Path, default=Path('C:/Program Files (x86)/Windows Kits/10'))
    parser.add_argument('--sdk-version', default='10.0.22621.0')
    parser.add_argument('--log', type=Path, required=True)
    args = parser.parse_args()
    if args.log.exists():
        raise FileExistsError('Preserve prior build logs')
    env = dict(os.environ)
    env.update(RUSTUP_HOME=str(args.toolchain/'rustup'), CARGO_HOME=str(args.toolchain/'cargo'),
        CARGO_TARGET_DIR=str(args.target_dir),
        CARGO_TARGET_I686_PC_WINDOWS_MSVC_LINKER=str(args.msvc/'bin/Hostx64/x86/link.exe'),
        LIB=';'.join(str(p) for p in (args.msvc/'lib/x86',
            args.sdk/'Lib'/args.sdk_version/'um/x86', args.sdk/'Lib'/args.sdk_version/'ucrt/x86')))
    cargo = args.toolchain/'cargo/bin/cargo.exe'
    rustc = args.toolchain/'cargo/bin/rustc.exe'
    crate = ROOT/'native/fake_d3d9'
    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    args.log.parent.mkdir(parents=True, exist_ok=True)
    commands = []
    if not (crate/'Cargo.lock').exists():
        commands.append([str(cargo), 'generate-lockfile', '--manifest-path', str(crate/'Cargo.toml')])
    commands.append([str(cargo), 'build', '--locked', '--manifest-path', str(crate/'Cargo.toml'),
                     '--target', 'i686-pc-windows-msvc', '--release'])
    with args.log.open('w', encoding='utf-8') as log:
        for command in commands:
            completed = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                stderr=subprocess.STDOUT, creationflags=flags)
            if completed.returncode:
                raise SystemExit('Build failed; see '+str(args.log))
    source = args.target_dir/'i686-pc-windows-msvc/release/th14_fake_d3d9.dll'
    target = ROOT/'native/th14_fake_d3d9.dll'
    shutil.copyfile(source, target)
    version = subprocess.check_output([str(rustc), '--version'], env=env,
        creationflags=flags, text=True).strip()
    inputs = {p.relative_to(ROOT).as_posix(): sha(p)
              for p in sorted(crate.rglob('*')) if p.is_file() and 'target' not in p.parts}
    metadata = dict(schema='th14-fake-d3d9-build-v1', dll_sha256=sha(target),
        architecture='i686', target='i686-pc-windows-msvc', rustc=version,
        created_utc=datetime.now(timezone.utc).isoformat(), source_sha256=inputs,
        upstream='https://github.com/astral4/nokozero', license='MIT OR Apache-2.0',
        build_log=str(args.log.resolve()), real_game_validation=False)
    (ROOT/'native/th14_fake_d3d9.build.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), 'utf-8')
    (ROOT/'native/th14_fake_d3d9.dll.sha256').write_text(metadata['dll_sha256']+'\n', 'ascii')
    print(json.dumps(dict(status='built', dll=str(target), sha256=metadata['dll_sha256'], rustc=version)))


if __name__ == '__main__':
    main()
