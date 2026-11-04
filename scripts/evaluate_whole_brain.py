from pathlib import Path
import argparse
import torch
from fly_touhou.connectome import load_connectome
from fly_touhou.whole_brain_learning import load_whole_brain_checkpoint
from fly_touhou.whole_brain_evaluation import evaluate_whole_brain

root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description='Independent whole-brain formal original-game campaign')
parser.add_argument('--checkpoint',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--game-exe',type=Path,default=Path(
    r'games/th14/th14.exe'))
parser.add_argument('--profile',type=Path,default=root/'profiles/th14_1.00b_original.json')
parser.add_argument('--data',type=Path,default=root/'data/connectome')
parser.add_argument('--ports',type=Path,default=root/'data/connectome/annotations/selected_ports_1024_v783.json')
parser.add_argument('--episodes',type=int,default=1)
parser.add_argument('--difficulty',type=int,choices=range(4),default=3)
parser.add_argument('--frame-skip',type=int,default=4,help='Requested native ticks per brain decision; >=2, default 4')
parser.add_argument('--episode-frames',type=int,default=120000)
parser.add_argument('--device',default='cuda')
parser.add_argument('--capture-inference',action='store_true',help='Exact no-grad CUDA replay for explicit cached-runtime checkpoints')
parser.add_argument('--no-dialogue-skip',action='store_true')
parser.add_argument('--window-title',default='果蝇全脑学生 · Lunatic')
parser.add_argument('--parallel-startup',action='store_true',help='Isolate the exact startup mutex for an independent original-game process')
parser.add_argument('--stochastic',action='store_true',help='Sample the brain readout distribution instead of taking its largest logit')
parser.add_argument('--action-seed',type=int,default=42)
args=parser.parse_args()
if args.frame_skip < 2:
    parser.error('--frame-skip must be >=2 to deliver and release a native Bomb edge')
torch.set_num_threads(4)
graph=load_connectome(args.data,max_neurons=None)
policy=load_whole_brain_checkpoint(args.checkpoint,graph,device=args.device)
if args.capture_inference:
    policy.student.enable_cuda_capture()
evaluate_whole_brain(policy,args.game_exe,args.profile,args.checkpoint,args.output,
    source_dir=args.data,ports_manifest=args.ports,episodes=args.episodes,
    difficulty=args.difficulty,max_episode_frames=args.episode_frames,
    deterministic=not args.stochastic,
    dialogue_skip=not args.no_dialogue_skip,window_title=args.window_title,
    parallel_startup=args.parallel_startup,action_seed=args.action_seed,frame_skip=args.frame_skip)
