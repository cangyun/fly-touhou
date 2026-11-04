"""Per-job native extraction temp, outside the immutable result directory."""
from pathlib import Path
import os
import re
import shutil
import tempfile


def configure_native_temp(output):
    output=Path(output).resolve()
    configured=os.environ.get('FLY_TOUHOU_TEMP_ROOT')
    if configured:
        temporary=Path(configured)/(output.name+'-'+str(os.getpid()))
    else:
        temporary=output.parent/'native-temporary'/(output.name+'-'+str(os.getpid()))
    temporary.mkdir(parents=True,exist_ok=True)
    os.environ['TEMP']=str(temporary);os.environ['TMP']=str(temporary)
    os.environ['TMPDIR']=str(temporary);tempfile.tempdir=str(temporary)
    return temporary


def cleanup_native_temp(temporary):
    """Remove this finished job's extracted Frida binaries, retaining logs.

    Call only after the job's native environments have closed. A busy file or
    an unfamiliar directory is retained for a later cleanup, not treated as a
    training failure.
    """
    root=Path(temporary)
    removed=0
    errors=[]
    if not root.is_dir():return dict(removed=removed,errors=errors)
    for child in root.iterdir():
        if not re.fullmatch(r'frida-[0-9a-f]{32}',child.name) or not child.is_dir():
            continue
        try:
            if child.is_symlink() or getattr(child.lstat(),'st_file_attributes',0)&0x400:
                continue
            contents=list(child.rglob('*'))
            if any(item.is_symlink() or getattr(item.lstat(),'st_file_attributes',0)&0x400
                   or (item.is_file() and item.suffix.lower() not in ('.exe','.dll'))
                   for item in contents):
                continue
            shutil.rmtree(child)
            removed+=1
        except OSError as error:
            errors.append(dict(path=str(child),error=str(error)))
    return dict(removed=removed,errors=errors)
