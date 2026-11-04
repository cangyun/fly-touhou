"""Run baseline, actual CUDA lesson, then fresh native first-hit evaluation."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--ports', type=Path, required=True)
    parser.add_argument('--game-exe', type=Path, required=True)
    parser.add_argument('--updates', type=int, default=16)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output/'pipeline.json').exists():
        raise FileExistsError('Do not overwrite an existing lesson pipeline')
    started = time.time()
    common = ['--data', str(args.data), '--ports', str(args.ports), '--game-exe', str(args.game_exe)]
    script = Path(__file__).with_name('train_movement_curriculum.py')
    stages = {}
    try:
        for phase in ('baseline', 'train', 'evaluation'):
            checkpoint = args.checkpoint
            if phase == 'evaluation':
                checkpoint = Path(json.loads((output/'train/latest-checkpoint.json').read_text(encoding='utf-8'))['path'])
            arguments = [sys.executable, str(script), '--mode', 'train' if phase == 'train' else 'evaluate',
                '--checkpoint', str(checkpoint), '--output', str(output/phase), *common,
                '--updates', str(args.updates), '--episodes', '4', '--difficulty', '3',
                '--rollout-steps', '64', '--passes', '2', '--learning-rate', '.0003']
            write_json(output/'pipeline.json', {'status': 'running', 'phase': phase,
                'started_unix': started, 'stages': stages, 'command': arguments, 'desktop_input': False})
            with (output/f'{phase}.stdout.log').open('xb') as stdout, (output/f'{phase}.stderr.log').open('xb') as stderr:
                result = subprocess.run(arguments, stdout=stdout, stderr=stderr, stdin=subprocess.DEVNULL,
                    env={**os.environ, 'PYTHONUNBUFFERED': '1'}, check=False)
            stages[phase] = {'exit_code': result.returncode, 'report': str(output/phase/'report.json')}
            if result.returncode:
                raise RuntimeError(f'Movement {phase} failed; retained its diagnostics and previous stage evidence')
        baseline = json.loads((output/'baseline/report.json').read_text(encoding='utf-8'))
        candidate = json.loads((output/'evaluation/report.json').read_text(encoding='utf-8'))
        comparison = {'status': 'finished', 'started_unix': started, 'finished_unix': time.time(),
            'stages': stages, 'baseline_frames': [v['native_frames_to_first_hit_or_limit'] for v in baseline['episodes']],
            'candidate_frames': [v['native_frames_to_first_hit_or_limit'] for v in candidate['episodes']],
            'baseline_median': baseline['median_frames_to_first_hit_or_limit'],
            'candidate_median': candidate['median_frames_to_first_hit_or_limit'],
            'scope': 'four fresh native Lunatic first-hit ordinary18 movement lessons per policy; game RNG not controlled; not G8',
            'teacher_action_calls': 0, 'safety_takeovers': 0, 'final_acceptance': False}
        write_json(output/'comparison.json', comparison)
        write_json(output/'pipeline.json', comparison)
        print(json.dumps(comparison, ensure_ascii=False), flush=True)
    except BaseException as error:
        write_json(output/'pipeline.json', {'status': 'failed', 'error': repr(error),
            'started_unix': started, 'finished_unix': time.time(), 'stages': stages})
        raise


if __name__ == '__main__':
    main()
