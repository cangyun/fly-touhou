"""Frame-indexed presentation data. This module never chooses game actions."""
from pathlib import Path
import csv
import json
from concurrent.futures import ThreadPoolExecutor
import numpy as np


def probabilities(logits):
    x = np.asarray(logits, dtype=np.float64)
    p = np.exp(x-np.max(x));p /= p.sum()
    movement = float(p[:18].sum())
    conditional = p[:18]/movement if movement > 0 else np.zeros(18)
    return dict(joint=p, directions=conditional[:9]+conditional[9:],
                slow=float(conditional[9:].sum()), bomb=float(p[18]))


def write_atlas(connectome, node_ids, inputs, outputs, destination):
    """Use published v783 anchors, including the anisotropic voxel conversion."""
    lookup = {}
    with (Path(connectome)/'annotations/flywire_annotations_v2.1.0.tsv').open(encoding='utf-8', newline='') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            lookup[r['root_id']] = r
    xyz = np.full((len(node_ids),3), np.nan, np.float32)
    groups = np.zeros(len(node_ids),np.uint8)
    groups[np.asarray(inputs,dtype=int)] = 1
    groups[np.asarray(outputs,dtype=int)] = 2
    for i, root in enumerate(node_ids):
        row = lookup.get(str(root))
        if row is not None:
            try:xyz[i] = [float(row[k]) for k in ('pos_x','pos_y','pos_z')]
            except ValueError:pass
    xyz *= np.array([.004,.004,.04],np.float32)
    np.savez_compressed(destination,xyz_um=xyz,groups=groups,root_ids=np.asarray(node_ids),
        input_indices=inputs,output_indices=outputs)
    return dict(nodes=len(node_ids),positioned=int(np.isfinite(xyz).all(1).sum()),
        coordinate_kind='v783 neuron anchor;not a neuron skeleton',units='micrometres',
        source_voxel_nm=[4,4,40],input_ports=len(inputs),descending_outputs=len(outputs))


class TraceWriter:
    """Persist chosen recurrent activity, not a blend of the18 candidates."""
    def __init__(self, output, groups, *, chunk_size=64):
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
        self.groups=np.asarray(groups);self.chunk_size=chunk_size
        self.rows=[];self.rates=[];self.spikes=[];self.count=0;self.chunk=0
        self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='demo-compression')
        self.pending=[]
        self.timeline=(self.output/'timeline.jsonl').open('w',encoding='utf-8')
        self.commands=(self.output/'commands.jsonl').open('w',encoding='utf-8')

    def record(self, info, out, state, native_segments, after):
        rate=state.rate[0].detach().cpu().numpy()
        spike=state.spikes[0].detach().cpu().numpy()
        membrane=state.membrane[0].detach().cpu().numpy()
        logits=out['logits'][0].detach().cpu().numpy()
        p=probabilities(logits)
        row=dict(index=self.count,frame=int(info['episode_frames']),
            stage=int(info['stage']),stage_frame=int(info['stage_frame']),
            lives=int(info['lives']),bombs=int(info['bombs']),
            misses=int(info['misses']),bombs_used=int(info['bombs_used']),
            action=int(out['actions'][0]),probabilities=p['joint'].tolist(),
            native_focus_after=bool(after.player.focused) if after.player else False,
            population_rate=[float(rate[self.groups==i].mean()) for i in range(3)],
            population_membrane=[float(membrane[self.groups==i].mean()) for i in range(3)])
        self.rows.append(row)
        self.rates.append(np.rint(np.clip(rate,0,1)*255).astype(np.uint8))
        self.spikes.append(np.packbits(spike>0,bitorder='little'))
        self.timeline.write(json.dumps(row,separators=(',',':'))+'\n')
        self.commands.write(json.dumps(dict(index=self.count,frame=row['frame'],
            segments=native_segments,after=dict(stage=after.stage,stage_frame=after.stage_frame,
                lives=after.lives,bombs=after.bombs,power=after.power,
                position=[after.player.x,after.player.y] if after.player else None)),separators=(',',':'))+'\n')
        self.count+=1
        if len(self.rows)>=self.chunk_size:self.flush()

    def flush(self):
        if not self.rows:return
        data=dict(frame=np.array([r['frame'] for r in self.rows],np.int64),
            rate_u8=np.stack(self.rates),spikes_packed=np.stack(self.spikes))
        destination=self.output/f'brain-{self.chunk:06d}.npz'
        self.pending.append(self.executor.submit(self._write_chunk,destination,data))
        self.chunk+=1;self.rows=[];self.rates=[];self.spikes=[]
        self.timeline.flush();self.commands.flush()
        if len(self.pending)>2:self.pending.pop(0).result()

    @staticmethod
    def _write_chunk(destination,data):
        pending=destination.with_suffix('.pending')
        with pending.open('wb') as f:np.savez_compressed(f,**data)
        pending.replace(destination)

    def close(self):
        self.flush()
        for job in self.pending:job.result()
        self.executor.shutdown();self.timeline.close();self.commands.close()


class TraceReader:
    def __init__(self,root):
        self.root=Path(root);self.rows=[];self.frames=np.zeros(0,np.int64)
        self.cached_chunk=None;self.data=None;self.cached_row=None;self.cached_rate=None;self.refresh()
    def refresh(self):
        path=self.root/'timeline.jsonl'
        if not path.exists():return
        rows=[]
        with path.open(encoding='utf-8') as f:
            for line in f:
                try:rows.append(json.loads(line))
                except json.JSONDecodeError:break
        self.rows=rows;self.frames=np.array([r['frame'] for r in rows],np.int64)
    def at(self,frame):
        if not self.rows:return None,None
        i=int(np.clip(np.searchsorted(self.frames,frame,side='right')-1,0,len(self.rows)-1))
        chunk=i//64;path=self.root/f'brain-{chunk:06d}.npz'
        if not path.exists():return self.rows[i],None
        if chunk!=self.cached_chunk:
            if self.data is not None:self.data.close()
            self.data=np.load(path);self.cached_chunk=chunk
            self.rate=self.data['rate_u8']
        if self.cached_row!=i:
            self.cached_row=i;self.cached_rate=self.rate[i%64].astype(np.float32)/255
        return self.rows[i],self.cached_rate
