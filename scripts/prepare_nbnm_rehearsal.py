"""Extract successful training-split histories; performs no network updates."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rollout',type=Path,required=True)
    p.add_argument('--episodes',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    import torch
    from fly_touhou.nbnm_rehearsal import export_successes
    rows=[json.loads(line) for line in a.episodes.read_text(encoding='utf-8').splitlines() if line.strip()]
    result=export_successes(torch.load(a.rollout,map_location='cpu',weights_only=False),rows,a.output,source=a.rollout)
    print(json.dumps(dict(histories=len(result['histories']),PPO_rows=0,network_training=False)))


if __name__=='__main__':main()
