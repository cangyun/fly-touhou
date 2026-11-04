"""Small CPU data-consumption entry; no model or training execution."""
import os
from pathlib import Path
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "2"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fly_touhou.fly_sensory_full_episode_reader import main

if __name__ == "__main__":
    raise SystemExit(main())
