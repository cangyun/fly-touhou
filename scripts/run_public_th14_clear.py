"""Run the archived TH14 finite-life implementations with the shared job lease."""

from pathlib import Path
import runpy
import sys

from fly_touhou.nbnm_job_phase import gpu_phase


def main():
    files = {"train": ("training", "train_compact_story_ppo.py"),
             "evaluate": ("evaluation", "evaluate_compact_repaired_checkpoint.py")}
    phase, name = files[sys.argv.pop(1)]
    script = Path(__file__).with_name(name)
    sys.argv[0] = str(script)
    with gpu_phase(phase):
        runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
