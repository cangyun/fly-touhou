"""Training-only physical attention labels beside immutable raw replay."""
from __future__ import annotations

import os
from pathlib import Path
import numpy as np

from .continuous_guided_focused_replay import FocusedSequenceReplay
from .continuous_guided_dagger import BURN_IN,SEQUENCE_LENGTH
from .geometry_attention_teaching import per_object_first4,SCHEMA
from .thrl_movement_teacher import MovementSafetyTeacher


class AttentionSequenceReplay(FocusedSequenceReplay):
    def __init__(self,root):
        super().__init__(root)
        self.attention_root=self.root/'attention-labels'
        self.attention_root.mkdir(exist_ok=True)
        self.teacher=MovementSafetyTeacher()

    def load_record(self,record):
        data=super().load_record(record)
        path=self.attention_root/(record['sha256']+'-'+SCHEMA+'.npz')
        if path.exists():
            with np.load(path,allow_pickle=False) as saved:
                data.update({k:saved[k] for k in saved.files})
            return data
        margins=np.full((SEQUENCE_LENGTH,18,96),64.,np.float32)
        valid=np.zeros((SEQUENCE_LENGTH,96),bool)
        # Burn-in supplies history only. Unknown, protected and laser scenes
        # receive no point-attention label, while their main losses are retained.
        for t in range(BURN_IN,SEQUENCE_LENGTH):
            labels,_=per_object_first4(data['features'][t],data['objects'][t],data['maps'][t],teacher=self.teacher)
            if labels is not None:margins[t],valid[t]=labels
        targets=dict(attention_margins=margins,attention_valid_objects=valid)
        temp=path.with_suffix('.npz.tmp')
        with temp.open('wb') as stream:
            np.savez_compressed(stream,**targets);stream.flush();os.fsync(stream.fileno())
        os.replace(temp,path)
        data.update(targets)
        return data
