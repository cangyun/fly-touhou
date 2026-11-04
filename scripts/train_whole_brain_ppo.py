"""Continue a distilled complete-brain policy using original TH14 rewards.

This starts a hidden original-game campaign only after all hashes/full-graph
checks pass. It does not claim an independent six-stage campaign success.
"""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
from fly_touhou.connectome import load_connectome  # noqa: E402
from fly_touhou.whole_brain_learning import load_whole_brain_checkpoint  # noqa: E402
from fly_touhou.whole_brain_ppo import campaign_environment_for_policy, train_whole_brain_ppo  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--game-exe", type=Path, required=True)
    parser.add_argument("--profile", type=Path, default=ROOT / "profiles" / "th14_1.00b_original.json")
    parser.add_argument("--source-dir", type=Path, default=ROOT / "data" / "connectome")
    parser.add_argument("--ports", type=Path,
        default=ROOT / "data" / "connectome" / "annotations" / "selected_ports_1024_v783.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--difficulty", type=int, choices=range(4), default=3)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--max-episode-frames", type=int, default=120000)
    parser.add_argument("--private-appdata", type=Path,
                        help="Explicit independent native save directory for this worker")
    parser.add_argument("--dialogue-skip", action="store_true",
                        help="Use verified process-local dialogue input edges; requires private APPDATA")
    parser.add_argument("--parallel-startup", action="store_true",
                        help="Isolate the owned child's startup mutex; requires private APPDATA")
    parser.add_argument("--responsive-wait", action="store_true",
                        help="Pump window messages during CUDA checkpoint waits; requires parallel startup and private APPDATA")
    parser.add_argument("--window-title", help="Label only this worker's owned game window; requires private APPDATA")
    parser.add_argument('--native-worlds', type=int, choices=(1, 2, 3, 4), default=1,
                        help='Independent original games sharing one B=1 CUDA actor and one PPO optimizer')
    parser.add_argument('--hide-windows', action='store_true',
                        help='Hide only owned original windows; preserves native rendering and game timing')
    parser.add_argument('--reward-profile', choices=('native-legacy', 'native-life-avoidance-v1', 'native-life-emergency-bomb-v1', 'native-life-emergency-geometry-v2'), default='native-legacy',
                        help='Explicit original-transition training feedback; no action or native resource override')
    parser.add_argument('--change-reward', action='store_true',
                        help='Explicit supported reward-profile transition; preserves the complete PPO learner state')
    parser.add_argument('--action-feedback',choices=('native-geometry-action-set-v2',),
                        help='Explicit sealed before-state training targets; requires qualified learner, no inference action mask')
    parser.add_argument('--teacher-feedback',type=Path,
                        help='Closed V6 validation report for explicit training-only expert guidance; requires qualified learner')
    parser.add_argument('--critical-full-rollout-first-step',action='store_true',
                        help='Qualified current critical rollouts use one full sample-weighted first step; no native action override')
    parser.add_argument("--total-timesteps", type=int, default=4096)
    parser.add_argument("--rollout-steps", type=int, default=32)
    parser.add_argument("--chunk-decisions", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch-chunks", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--value-clip", type=float, default=0.2)
    parser.add_argument("--gradient-clip", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    restoration = parser.add_mutually_exclusive_group()
    restoration.add_argument("--resume", type=Path, help="Strict resume of the same immutable run/configuration")
    restoration.add_argument("--continue-training", action="store_true",
        help="New linked run from a PPO initial checkpoint; preserve actor, critic, Adam, RNGs and cumulative counters")
    parser.add_argument("--progress-interval", type=int, default=4,
                        help="Write optimizer progress every this many steps (default: 4)")
    parser.add_argument('--rollout-compression', choices=('none','gzip'), default='none',
                        help='Lossless compression of retained full-brain replay tensors')
    parser.add_argument('--capture-inference',action='store_true',
                        help='Use verified whole-brain no-grad CUDA replay; requires coo-cusparse-cached checkpoint')
    parser.add_argument('--capture-backward',action='store_true',
                        help='Use exact full-brain PPO loss/backward replay; ordinary FP64 clipping and continuing Adam remain unchanged')
    args = parser.parse_args()
    if args.threads < 1 or args.progress_interval < 1:
        parser.error("--threads and --progress-interval must be positive")
    if (args.parallel_startup or args.dialogue_skip or args.window_title or args.responsive_wait) and args.private_appdata is None:
        parser.error("--parallel-startup/--dialogue-skip/--window-title require --private-appdata")
    if args.responsive_wait and not args.parallel_startup:
        parser.error("--responsive-wait requires --parallel-startup and --private-appdata")
    if (args.native_worlds > 1 or args.hide_windows) and not (args.private_appdata and args.parallel_startup and args.responsive_wait):
        parser.error('Parallel or hidden worlds require private APPDATA, parallel startup and responsive wait')
    if args.change_reward and (not args.continue_training or args.reward_profile == 'native-legacy'):
        parser.error('--change-reward requires --continue-training and a nonlegacy explicit reward profile')
    if args.critical_full_rollout_first_step and (not args.teacher_feedback or not args.action_feedback or args.minibatch_chunks!=1):
        parser.error('--critical-full-rollout-first-step requires V6/action feedback and single-chunk microbatches')
    torch.set_num_threads(args.threads)
    graph = load_connectome(args.source_dir, max_neurons=None)
    policy = load_whole_brain_checkpoint(args.resume or args.initial_checkpoint, graph, args.device)
    if args.capture_inference:
        policy.student.enable_cuda_capture()
    environments = []
    from fly_touhou.campaign_rewards import attach_training_reward
    for index in range(args.native_worlds):
        private_path = (args.private_appdata / f'world_{index}' if args.native_worlds > 1 else args.private_appdata)
        title = f'{args.window_title} · 采样实例{index + 1}' if args.window_title and args.native_worlds > 1 else args.window_title
        world = campaign_environment_for_policy(policy, args.game_exe, args.profile, difficulty=args.difficulty,
            frame_skip=args.frame_skip, max_episode_frames=args.max_episode_frames,
            private_appdata=private_path, dialogue_skip=args.dialogue_skip,
            parallel_startup=args.parallel_startup, window_title=title, responsive_wait=args.responsive_wait,
            hide_window=args.hide_windows)
        environments.append(attach_training_reward(world, args.reward_profile, gamma=args.gamma))
    env = environments[0]
    try:
        train_whole_brain_ppo(env, policy, args.output, initial_checkpoint=args.initial_checkpoint,
            game_executable=args.game_exe, profile_path=args.profile, source_dir=args.source_dir,
            ports_manifest=args.ports, total_timesteps=args.total_timesteps, rollout_steps=args.rollout_steps,
            chunk_decisions=args.chunk_decisions, epochs=args.epochs, minibatch_chunks=args.minibatch_chunks,
            learning_rate=args.learning_rate, gamma=args.gamma, gae_lambda=args.gae_lambda,
            clip_range=args.clip_range, entropy_coefficient=args.entropy_coefficient,
            value_coefficient=args.value_coefficient, value_clip=args.value_clip, gradient_clip=args.gradient_clip,
            target_kl=args.target_kl, seed=args.seed, resume_checkpoint=args.resume,
            continue_training=args.continue_training, progress_interval=args.progress_interval,
            rollout_compression=args.rollout_compression, capture_backward=args.capture_backward,
            additional_envs=tuple(environments[1:]), change_reward=args.change_reward,
            action_feedback=args.action_feedback,teacher_feedback_reference=args.teacher_feedback,
            critical_full_rollout_first_step=args.critical_full_rollout_first_step)
    finally:
        for world in environments:
            world.close()


if __name__ == "__main__":
    main()
