"""Read measured displacement/focus after native four-tick input delivery."""
from pathlib import Path
import argparse
import gzip
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from fly_touhou.experiment import file_sha256
from fly_touhou.native_diagnostic import write_json
from fly_touhou.state import decode_action


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trajectory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    count = 0
    with gzip.open(args.trajectory,'rt',encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            before,after = row['snapshot']['player'],row['next_snapshot']['player']
            if row['hit'] or not before or not after or before['state'] != 1 or after['state'] != 1:
                continue
            dx,dy,focused = decode_action(row['action'])
            vx,vy = after['x']-before['x'],after['y']-before['y']
            assert bool(after['focused']) == focused
            for requested,measured in ((dx,vx),(dy,vy)):
                assert abs(measured) <= 1e-4 if requested == 0 else measured*requested >= -1e-4
            count += 1
    assert count > 0
    result = {'schema':'native-measured-input-direction-focus-readback-v1','status':'passed',
        'trajectory':str(args.trajectory),'trajectory_sha256':file_sha256(args.trajectory),
        'auditor':str(Path(__file__)),'auditor_sha256':file_sha256(Path(__file__)),
        'nonhit_controllable_transitions_checked':count,'focus_mismatches':0,'direction_mismatches':0,
        'scope':'measured focus and displacement sign only;does not prove exact continuous path or subframe firstcollision time',
        'final_acceptance':False}
    write_json(args.output,result,exclusive=True)
    print(json.dumps(result,ensure_ascii=False))


if __name__ == '__main__':
    main()
