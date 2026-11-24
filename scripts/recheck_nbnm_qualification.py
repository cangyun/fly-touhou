"""Apply current qualification thresholds to a completed native frozen report."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fly_touhou.nbnm_run import full_story_qualification, read
from fly_touhou.thrl_durable_records import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report_path = args.report.resolve()
    config_path = args.config.resolve()
    output_path = args.output.resolve()
    if output_path in (report_path, config_path):
        parser.error("The qualification result needs its own output file")

    report = read(report_path)
    spec = read(config_path)
    result = full_story_qualification(report, spec)
    write_json(output_path, dict(
        source_report=str(report_path),
        qualification_config=str(config_path),
        previous_qualification=report.get("full_story_qualification"),
        full_story_qualification=result,
    ))


if __name__ == "__main__":
    main()
