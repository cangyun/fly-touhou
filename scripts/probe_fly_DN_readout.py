"""CPU diagnostic only;does not extract features,load a model or export a policy."""
from pathlib import Path
import sys
if __name__=='__main__':
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
    from fly_touhou.fly_dn_readout_probe import main
    raise SystemExit(main())
