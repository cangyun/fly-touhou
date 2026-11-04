"""Read exact original EXE and a bounded completed episode in a live trace.

This diagnostic never controls a game or admits the unsealed file as training
data.  The selected complete episode is identified by its raw JSON line hashes.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import struct

import capstone
import pefile

SHA = 'f08e885e28d2e0fc8c579e88034dd7de8f2a648d8f08ce781cf08eadc9ffe353'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe', type=Path, required=True)
    p.add_argument('--trace', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--episode-index', type=int, default=0)
    opt = p.parse_args()
    opt.output.mkdir(parents=True, exist_ok=False)
    raw = opt.exe.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SHA
    pe = pefile.PE(data=raw)
    base = pe.OPTIONAL_HEADER.ImageBase
    md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_32)
    blocks = {}
    for name, start, end in (
        ('bullet_collision_shape_and_scale_gate', 0x416d70, 0x416dff),
        ('player_rectangle_collision', 0x44ee80, 0x44ef99),
        ('player_circle_collision_and_large_player_multiplier', 0x44efa0, 0x44f0a6),
        ('player_scaled_collision_bounds', 0x44e227, 0x44e315),
    ):
        data = pe.get_data(start-base, end-start)
        blocks[name] = {'va': hex(start), 'rva': hex(start-base), 'bytes': data.hex(),
            'instructions': [{'va':hex(i.address), 'bytes':i.bytes.hex(),
                              'asm':i.mnemonic+' '+i.op_str} for i in md.disasm(data,start)]}
    constants = {hex(a): {'bytes': pe.get_data(a-base,4).hex(),
                          'value': struct.unpack('<f',pe.get_data(a-base,4))[0]}
                 for a in (0x4c19b0,0x4c1ab8)}
    stages = {}
    samples = []
    last_scale = None
    maximum_scale = 0
    largest_player_sample = None
    selected_episode = None
    summary = None
    decoded_digest = hashlib.sha256()
    decoded_bytes = 0
    transitions = 0
    with gzip.open(opt.trace,'rb') as f:
        for line_number,line in enumerate(f,1):
            decoded_digest.update(line)
            decoded_bytes += len(line)
            row = json.loads(line)
            kind = row.get('record_type')
            eid = row.get('episode_id','')
            if kind == 'metadata':
                (opt.output/'trace_metadata.json').write_bytes(line)
            if kind == 'episode_summary' and eid.rsplit(':',1)[-1] == str(opt.episode_index):
                selected_episode = eid
                summary = row
                break
            if kind != 'transition' or eid.rsplit(':',1)[-1] != str(opt.episode_index):
                continue
            transitions += 1
            s = row['snapshot']
            stage = str(s['stage'])
            d = stages.setdefault(stage, {'decisions':0,'max_player_scale':0,
                'scaled_bullet_records':0,'scaled_collision_bullet_records':0,
                'max_bullet_scale':0,'circle_flag_counts':Counter(),
                'large_player_flag_decisions':0,'misses':[]})
            d['decisions'] += 1
            x = s.get('extra',{})
            scale = x.get('player_scale',1.)
            flags = x.get('resources',{}).get('player_flags',0)
            d['large_player_flag_decisions'] += bool(flags & 0x10)
            d['max_player_scale'] = max(d['max_player_scale'],scale)
            hazard = x.get('hazard_metadata',{})
            for b in hazard.get('bullets',[]):
                bs,bf = b['native_scale'],b['native_flags']
                d['circle_flag_counts'][str(bool(bf & 0x10))] += 1
                d['max_bullet_scale'] = max(d['max_bullet_scale'],bs)
                d['scaled_bullet_records'] += bs != 1
                d['scaled_collision_bullet_records'] += bool(bf & 0x40 and bs != 1)
            sample = {'line':line_number,'raw_line_sha256':hashlib.sha256(line).hexdigest(),
                'episode_id':eid,'step':row['step'],'stage':s['stage'],'stage_frame':s['stage_frame'],
                'player_scale':scale,'player_flags':flags,'legacy_player_radius':(s.get('player') or {}).get('radius'),
                'base_player_radius':x.get('base_player_radius'),'snapshot':s,
                'action':row['action'],'next_snapshot':row['next_snapshot']}
            if s['stage']==6 and scale != last_scale:
                if len(samples)<20 or scale in (1.,2.,4.,8.):
                    samples.append(sample)
                last_scale = scale
            if scale > maximum_scale:
                maximum_scale = scale
                largest_player_sample = sample
            if row['next_snapshot']['lives'] < s['lives']:
                d['misses'].append({k:sample[k] for k in ('line','raw_line_sha256','step',
                    'stage_frame','player_scale','player_flags','legacy_player_radius','base_player_radius')})
                if s['stage']==6:
                    samples.append(sample)
    if summary is None:
        raise RuntimeError('Requested episode has no complete summary; do not label it complete')
    for d in stages.values():
        d['circle_flag_counts']=dict(d['circle_flag_counts'])
    report = {'schema':'original-th14-stage6-geometry-diagnostic-v1',
        'exe':str(opt.exe),'exe_sha256':SHA,'trace':str(opt.trace),
        'trace_file_is_unsealed_live_prefix':True,
        'selected_episode':selected_episode,'transitions':transitions,
        'decoded_prefix_sha256':decoded_digest.hexdigest(),'decoded_prefix_bytes':decoded_bytes,
        'whole_file_hash_claimed':False,'static_tool':'local capstone/pefile; REA IDA provider unavailable',
        'blocks':blocks,'constants':constants,'stages':stages,
        'episode_summary':summary,
        'findings':[
            'flags bit0x40 gates collision scaling; native visual scale alone does not imply collision scale',
            'flags bit0x10 selects circle. Otherwise +be0/+be4 are full rectangle widths',
            'circle helper tests distance squared against bullet_radius squared plus player_component squared',
            'large-player flag0x10 multiplies definition radius by scale*3.6 inside circle helper',
            'rectangle player bounds use native scaled base x/y half-widths',
            'this report diagnoses an existing native trace; it is not corrected-teacher runtime acceptance']}
    (opt.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with gzip.open(opt.output/'selected_samples.jsonl.gz','xt',encoding='utf-8') as f:
        for sample in samples+[largest_player_sample]:
            if sample:f.write(json.dumps(sample,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps({'episode':selected_episode,'transitions':transitions,'stages':stages},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
