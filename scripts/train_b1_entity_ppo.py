"""Continue the proven B1 policy with a frozen neural entity-risk readout."""
from pathlib import Path
import argparse,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))

if __name__=='__main__':
    from train_b1_avoidance_ppo import run
    from fly_touhou.thrl_pair_policy import model_from_config
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--config',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    run(p.parse_args(),model_factory=model_from_config)
