"""Build a teacher-only route library from a native-verified complete replay."""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fly_touhou.nbnm_demonstration_routes import build_bank

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(build_bank(a.capture,a.output),ensure_ascii=False,indent=2))
