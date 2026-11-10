"""Public NBNM configuration works without the author's disks or game files."""
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def test_portable_nbnm_preparation_uses_explicit_paths_and_no_game(tmp_path):
    output=tmp_path/'control'
    bank=tmp_path/'bank/manifest.json'
    checkpoint=tmp_path/'shared models/cycle087-parameters.pt'
    connectome=tmp_path/'shared connectome'
    game=tmp_path/'game/th14.exe'
    run_root=tmp_path/'training'
    subprocess.run([sys.executable,'-B',str(ROOT/'scripts/prepare_nbnm_experiment.py'),
        '--output',str(output),'--start-bank',str(bank),'--game-exe',str(game),
        '--checkpoint',str(checkpoint),'--connectome',str(connectome),
        '--run-root',str(run_root),'--tensorboard-root',str(tmp_path/'tensorboard')],check=True)
    route=json.loads((output/'route.json').read_text())
    baseline=json.loads((output/'baseline.json').read_text())
    environment=json.loads((output/'environment.json').read_text())
    assert route['initial_checkpoint']==str(checkpoint.resolve())
    assert route['connectome']==str(connectome.resolve())
    assert route['nbnm']['start_bank']==str(bank.resolve())
    assert environment['game_exe']==str(game.resolve())
    assert Path(environment['profile']).is_absolute()
    assert route['world_groups'].count('card')==12 and route['world_groups'].count('full')==4
    assert route['decision_budget']==32768 and route['segment_decisions']==8192
    assert route['nbnm']['risk_mode']=='route' and baseline['nbnm']['risk_mode']=='baseline'
    for key in ('output','tensorboard_directory'):
        route.pop(key);baseline.pop(key)
    baseline['nbnm']['risk_mode']='route'
    assert route==baseline
    assert not game.exists() and not run_root.exists()


def test_public_warm_start_is_cycle087_and_not_nbnm_clear():
    release=json.loads((ROOT/'configs/release.json').read_text())
    assert release['branch']=='experimental/nbnm'
    assert release['checkpoint']=='cycle087-parameters.pt'
    assert release['tag']=='v0.1.0-clear'
    assert 'does not claim full-game NBNM' in release['result']
