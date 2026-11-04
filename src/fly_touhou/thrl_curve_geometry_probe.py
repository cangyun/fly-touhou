"""Bounded raw native curve observations for an explicit format qualification.

No actions or native state are changed. This is never a training factory.
"""
from pathlib import Path
import gzip
import hashlib
import json

import numpy as np

from .thrl_full_story_evaluation import FullStoryEvaluationEnv
from .atomic_json_io import atomic_write_json as write


class CurveGeometryProbeEnv(FullStoryEvaluationEnv):
    def __init__(self,*args,curve_probe_directory,**kwargs):
        self._curve_directory=Path(curve_probe_directory)
        self._curve_receipts=[];self._single_node_ids={};self._saved_curve_kinds=set()
        super().__init__(*args,**kwargs)

    def _encode(self,snapshot):
        kind=None;selected=[]
        for laser in snapshot.lasers:
            if len(laser.nodes)==1:
                self._single_node_ids[laser.id]=self.episode_id
                if 'single_node' not in self._saved_curve_kinds:
                    kind='single_node';selected.append(laser)
            elif (len(laser.nodes)>1 and self._single_node_ids.get(laser.id)==self.episode_id
                    and 'grown_after_single' not in self._saved_curve_kinds):
                kind='grown_after_single';selected.append(laser)
        if kind is None:return super()._encode(snapshot)
        self._curve_directory.mkdir(parents=True,exist_ok=True)
        path=self._curve_directory/f'episode{self.episode_id:04d}_frame{snapshot.frame}_{kind}.json.gz'
        raw=json.dumps(dict(snapshot=snapshot.to_dict(),native_episode=self.episode_id,
                            purpose='format qualification only;not policy skill'),ensure_ascii=False,allow_nan=True).encode('utf-8')
        compressed=gzip.compress(raw,compresslevel=1)
        with path.open('xb') as stream:stream.write(compressed)
        receipt=dict(kind=kind,raw_snapshot=str(path),sha256=hashlib.sha256(compressed).hexdigest(),
            uncompressed_sha256=hashlib.sha256(raw).hexdigest(),stage=snapshot.stage,
            stage_frame=snapshot.stage_frame,native_frame=snapshot.frame,native_episode=self.episode_id,
            curves=[dict(id=laser.id,node_count=len(laser.nodes),width=repr(laser.width),
                length=repr(laser.length),active=laser.active,
                nodes_finite=bool(np.isfinite(np.asarray(laser.nodes,float)).all())) for laser in selected],
            encoder_accepted=False,actions_changed=False,native_counterfactual=False)
        receipt_path=path.with_suffix('.receipt.json');write(receipt_path,receipt)
        try:
            observation=super()._encode(snapshot)
        except BaseException as error:
            receipt['encoder_error']=repr(error);write(receipt_path,receipt)
            raise
        receipt.update(encoder_accepted=True,observation_shapes={k:list(v.shape) for k,v in observation.items()},
                       observation_finite=all(np.isfinite(v).all() for v in observation.values()))
        write(receipt_path,receipt);self._curve_receipts.append(str(receipt_path));self._saved_curve_kinds.add(kind)
        return observation

    def _info(self,*args,**kwargs):
        result=super()._info(*args,**kwargs)
        result['curve_probe_receipts']=list(self._curve_receipts)
        return result


def curve_geometry_probe_factory(options):
    return CurveGeometryProbeEnv(**options)
