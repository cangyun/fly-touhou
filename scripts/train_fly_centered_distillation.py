"""Inspect by default; --execute-cuda explicitly enters the original finite fit."""
from pathlib import Path
import sys

if __name__=='__main__':
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
    from fly_touhou.fly_centered_distillation import main
    raise SystemExit(main())
