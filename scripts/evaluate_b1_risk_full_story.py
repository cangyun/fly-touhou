"""Complete original games for fixed risk/parent policies; no teacher execution."""
from pathlib import Path
import argparse, sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'))
from fly_touhou.thrl_study_io import read_shared_json as read


def run(a):
    from evaluate_b1_risk_policy import reserved_worker_bound
    plan = read(a.plan)
    reserved = reserved_worker_bound(Path(plan['parallel_teacher_control']))
    own = min(int(plan['workers']), int(plan['episodes']))
    if not 1 <= own <= 30 or own + reserved > 30:
        raise ValueError('Combined native budget exceeds30')
    # No Torch import at module load: native children remain model-free.
    from fly_touhou.thrl_risk_policy import model_from_config
    from evaluate_b1_full_story_comparison import run as compare
    compare(a, model_factory=model_from_config)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--plan', type=Path, required=True)
    run(p.parse_args())
