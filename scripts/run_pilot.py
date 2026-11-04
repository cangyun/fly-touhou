"""Small real-original-game integration run, with explicit limited scope."""
from pathlib import Path
import argparse
import json
import time

import torch
from fly_touhou.background import BackgroundTouhouEnv
from fly_touhou.connectome import load_connectome
from fly_touhou.learning import (collect_demonstrations, distill_student,
    evaluate_policy, load_student_checkpoint, train_ppo_teacher)
from fly_touhou.student_ppo import train_student_ppo

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, default=ROOT/'artifacts/pilot')
options = parser.parse_args()
OUTPUT = options.output.resolve()
OUTPUT.mkdir(parents=True, exist_ok=True)
if (OUTPUT/'planner_demonstrations.jsonl').exists():
    raise SystemExit('Existing native trace is preserved. Select a new --output directory.')

def write(name, value):
    (OUTPUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')

def environment(frames=1800):
    return BackgroundTouhouEnv(ROOT/'games/th14_trial/extracted/app/th14.exe',
                              ROOT/'profiles/th14_trial_0.01b.json', max_episode_frames=frames)

started = time.time()
torch.set_num_threads(4)
print('PHASE collect original-game planner demonstrations', flush=True)
with environment() as env:
    report = collect_demonstrations(env, OUTPUT/'planner_demonstrations.jsonl',
        episodes=4, planner=env.encoder.planner, max_steps=450)
    write('collect.json', report)
    print('COLLECT', {k:v for k,v in report.items() if k not in ('episodes','samples_per_action')}, flush=True)
    print('PHASE stay baseline original-game evaluation', flush=True)
    baseline = evaluate_policy(env, lambda observation: 0, episodes=2, max_steps=450)
    write('stay_evaluation.json', baseline)
    print('STAY', baseline['mean_native_frames'], baseline['hit_rate'], flush=True)

print('PHASE connectome policy distillation', flush=True)
graph = load_connectome(ROOT/'data/connectome', max_neurons=2048)
distillation = distill_student(OUTPUT/'planner_demonstrations.jsonl', graph,
    OUTPUT/'student_distilled.pt', epochs=4, chunk_size=32, device='cuda', seed=0)
write('distillation.json', distillation)
print('DISTILL', distillation['validation_metrics'], flush=True)
policy = load_student_checkpoint(OUTPUT/'student_distilled.pt', graph, device='cuda')
with environment() as env:
    print('PHASE student independent original-game evaluation', flush=True)
    evaluation = evaluate_policy(env, policy, episodes=2, max_steps=450)
    write('student_evaluation.json', evaluation)
    print('STUDENT', evaluation['mean_native_frames'], evaluation['hit_rate'], flush=True)
    print('PHASE student recurrent PPO integration', flush=True)
    reinforcement = train_student_ppo(env, policy, OUTPUT/'student_ppo.pt',
        total_timesteps=128, rollout_steps=64, sequence_length=16, n_epochs=2,
        minibatch_chunks=2, seed=1)
    write('student_ppo.json', reinforcement)

print('PHASE ordinary PPO teacher integration', flush=True)
with environment() as env:
    model, teacher = train_ppo_teacher(env, OUTPUT/'ppo_teacher.zip',
        total_timesteps=512, n_steps=128, batch_size=64, seed=2, device='cpu')
    write('ppo_teacher.json', teacher)

summary = {'scope': 'original TH14 trial v0.01b Normal Reimu A; initial integration pilot',
    'elapsed_seconds': time.time()-started, 'demonstration_steps': report['total_steps'],
    'demonstration_frames': report['total_native_frames'],
    'hazard_decisions': report['decisions_with_hazards'],
    'max_collidable_bullets': report['max_collidable_bullets'],
    'neurons': graph.n_neurons, 'distillation_validation': distillation['validation_metrics'],
    'stay_mean_survival_frames': baseline['mean_native_frames'],
    'student_mean_survival_frames': evaluation['mean_native_frames'],
    'student_ppo_decisions': reinforcement['native_steps'],
    'teacher_ppo_decisions': teacher['native_steps'],
    'game_rng_controlled': False,
    'performance_claim': 'pilot only; no claim of solved stage, full-game competence, or biological advantage'}
write('summary.json', summary)
print('COMPLETE', summary, flush=True)
