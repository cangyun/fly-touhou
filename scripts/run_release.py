"""Run the branch's frozen policy against a user-provided TH14 executable."""
from pathlib import Path
import argparse
from datetime import datetime
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def make_plan(args, root=ROOT):
    root = Path(root).resolve()
    release = json.loads((root/'configs/release.json').read_text(encoding='utf-8'))
    cfg = json.loads((root/release['environment']).read_text(encoding='utf-8'))
    cfg['game_exe'] = str(args.game_exe.expanduser().resolve())
    cfg['profile'] = str((root/cfg['profile']).resolve())
    checkpoint = (args.checkpoint or root/release['checkpoint_path']).expanduser().resolve()
    connectome = (args.connectome or root/'data/connectome').expanduser().resolve()
    output = (args.output or root/'outputs'/('evaluation-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f'))).resolve()
    seeds = args.opening_seeds if args.opening_seeds is not None else release['opening_seeds']
    if not seeds or any(not 0 <= seed < 2**32 for seed in seeds):
        raise ValueError('Opening seeds must be uint32 values.')
    config_path = output.parent/(output.name+'-environment.json')
    cache = (args.cache or root/'outputs/cache').resolve()
    command = [sys.executable, '-B', str(root/'scripts/evaluate_compact_repaired_checkpoint.py'),
               '--checkpoint', str(checkpoint), '--connectome', str(connectome),
               '--environment-config', str(config_path), '--kernel-cache', str(cache/'native-kernels'),
               '--output', str(output), '--episodes', str(len(seeds)),
               '--evaluation-seed', str(release['evaluation_seed']),
               '--policy-seed', str(release['policy_seed']),
               '--action-selection', release['action_selection'],
               '--opening-seeds', *map(str, seeds)]
    env = dict(os.environ, FLY_TOUHOU_CACHE_ROOT=str(cache),
               FLY_TOUHOU_TEMP_ROOT=str(cache/'native-temporary'),
               PYTHONDONTWRITEBYTECODE='1')
    required = [Path(cfg['game_exe']), Path(cfg['profile']), checkpoint,
                root/'native/th14_fake_d3d9.dll', root/'native/th14_fake_d3d9.dll.sha256',
                connectome/'Completeness_783.csv', connectome/'Connectivity_783.parquet',
                connectome/'annotations/flywire_annotations_v2.1.0.tsv',
                connectome/'annotations/selected_ports_1024_v783.json']
    return dict(release=release['tag'], command=command, environment_config=cfg,
                config_path=str(config_path), output=str(output), required=[str(p) for p in required],
                cache=str(cache)), env


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--game-exe', type=Path, required=True, help='Original Japanese TH14 1.00b executable')
    parser.add_argument('--checkpoint', type=Path, help='Override this branch\'s released parameters')
    parser.add_argument('--connectome', type=Path)
    parser.add_argument('--output', type=Path, help='New directory; must not already exist')
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--opening-seeds', type=int, nargs='+', help='One new game for each specified opening seed')
    parser.add_argument('--dry-run', action='store_true', help='Print the exact plan; do not launch or create files')
    args = parser.parse_args(argv)
    try:
        plan, env = make_plan(args)
    except ValueError as error:
        parser.error(str(error))
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    if os.name != 'nt':
        parser.error('The native TH14 controller currently requires Windows.')
    missing = [p for p in plan['required'] if not Path(p).is_file()]
    if missing:
        parser.error('Missing files; run scripts/setup_release.py first or supply your game: '+', '.join(missing))
    output = Path(plan['output'])
    if output.exists():
        parser.error('Output already exists; choose a new --output directory.')
    output.parent.mkdir(parents=True, exist_ok=True)
    Path(plan['config_path']).write_text(json.dumps(plan['environment_config'], ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print('Release:', plan['release'], '\nOutput:', output, flush=True)
    print('Frozen evaluation only; no training, no teacher takeover.', flush=True)
    return subprocess.call(plan['command'], cwd=ROOT, env=env)


if __name__ == '__main__':
    raise SystemExit(main())
