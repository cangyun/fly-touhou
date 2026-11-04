"""Portable locations for regenerable CUDA and framework caches."""
from pathlib import Path
import os
import sys

def configure_generated_caches(root=None):
    if root is None:
        root = Path(os.environ.get('FLY_TOUHOU_CACHE_ROOT', str(Path(__file__).resolve().parents[2]/'outputs/cache')))
    root = Path(root).resolve()
    locations = {'CUDA_CACHE_PATH': root/'cuda', 'TRITON_CACHE_DIR': root/'triton',
                 'TORCHINDUCTOR_CACHE_DIR': root/'torchinductor', 'MPLCONFIGDIR': root/'matplotlib'}
    for name, path in locations.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.dont_write_bytecode = True
    return {name: str(path) for name, path in locations.items()}

def generated_kernel_cache_base(requested):
    return Path(requested)
