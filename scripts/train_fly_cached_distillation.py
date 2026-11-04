"""Default NoTorch inspect; explicit finite CUDA learning uses the fixed D95 bank."""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fly_touhou.fly_cached_distillation import main

if __name__=='__main__':
    raise SystemExit(main())
