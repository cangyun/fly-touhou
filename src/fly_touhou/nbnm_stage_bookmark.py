"""Save a frozen student's Stage 6 entry and the game's own TH14 replay-stage record."""
import os
from pathlib import Path
import shutil
import struct

from .thrl_durable_records import write_json


def story_bookmark_index(stages, stage_frames):
    """First decision 32 ticks into the initialized sixth stage, after its loading clock resets."""
    import numpy as np
    ids=np.flatnonzero(np.asarray(stages)==6)
    if not len(ids):return None
    clocks=np.asarray(stage_frames)[ids]
    # At a Story transition TH14 briefly reports stage 6 using the departing
    # player's clock. The initialized stage has its own counter and input log.
    resets=np.flatnonzero(np.diff(clocks[:80])<0)
    start=int(resets[0])+1 if len(resets) else 0
    eligible=np.flatnonzero(clocks[start:]>=32)
    return int(ids[start+eligible[0]]) if len(eligible) else None


def restore_entry_encoder(encoder, snapshot, features, attack_age=None):
    """Rebase the observed attack timer on the current GameThread frame counter."""
    encoder.reset()
    encoder.set_native_episode(1)
    encoder.encode(snapshot)
    if attack_age is None:
        encoded=float(features[276])
        attack_age=round(600.*encoded/(1.-encoded))
    encoder._attack_start=snapshot.frame-attack_age
    encoder._nbnm_cache=None
    return encoder.encode(snapshot)


def replay_stage_record(memory, stage=6):
    """Read TH14 1.00b's recorder, populated by TH14 even during live play."""
    if memory.exe.name.lower()!='th14.exe' or stage not in range(1,7):
        raise ValueError('Expected an original TH14 main-game stage')
    recorder=memory.integer(memory.base+0xdb688)
    if not recorder or memory.integer(recorder+0x10)!=0:
        return None
    pointer=memory.integer(recorder+0x20+4*stage)
    if not pointer:
        return None
    header=memory.read(pointer,220)
    number,seed=struct.unpack_from('<HH',header)
    if number!=stage:
        return None
    outer=memory.integer(recorder+0x18)
    inner=memory.integer(recorder+0x1c)
    if not outer or not inner:
        return None
    x,y=struct.unpack_from('<ii',header,12)
    get=lambda off:struct.unpack_from('<i',header,off)[0]
    link=memory.integer(recorder+(stage+4)*16+4)
    first=memory.integer(link) if link else 0
    count=(memory.integer(first+0x1518)-first)//6 if first else 0
    opening=memory.read(first,6*min(count,128)) if 0<count<=900 else b''
    if not opening or get(32)!=3:
        return None  # The stage number is set before its entry resources and inputs exist.
    return dict(schema='th14-1.00b-native-replay-stage-record-v1',stage=stage,
        stage_rng_seed=seed,position_subpixels=[x,y],shot=get(20),subshot=get(24),
        difficulty=get(32),continues=get(36),power=get(68),lives=get(80),bombs=get(92),
        initial_key_frames=opening.hex(),stage_header=header.hex(),replay_header=memory.read(inner,148).hex(),
        container_header=memory.read(outer,36).hex())


def save_checkpoint(source, target, model=None):
    from .nbnm_run import atomic_torch_save
    if target.exists():return
    if source is None:
        atomic_torch_save(model.parameter_payload(dict(stage6_frozen_entry=True)),target)
        return
    try:
        os.link(source,target)
    except OSError:
        temporary=target.with_suffix('.copying')
        shutil.copyfile(source,temporary)
        os.replace(temporary,target)


def save_live_brain(output,model,state,index,features,info,spec,checkpoint=None):
    from .nbnm_run import atomic_torch_save
    root=Path(output)/'stage6-bookmarks'
    dest=root/f'seed-{info["native_seed"]}-episode-{info["native_episode"]}-world-{index}'
    dest.mkdir(parents=True,exist_ok=True)
    save_checkpoint(checkpoint,root/'frozen-parameters.pt',model)
    write_json(dest/'configuration.json',spec)
    from .nbnm_run import read
    from .th14_stage_replay import write_stage_replay
    native=read(Path(info['native_stage_record']))
    write_json(dest/'native-stage.json',native)
    write_stage_replay(dest/'stage6-start.rpy',native)
    atomic_torch_save(dict(schema='frozen-SNN-stage-entry-brain-v2',opening_seed=info['native_seed'],
        features=features.copy(),state=[v[index:index+1].detach().cpu().clone() for v in state.tensors()],
        stage_frame=info['stage_frame'],context=dict(stage=6,stage_frame=info['stage_frame']),
        optimizer_updates=0),dest/'brain.pt')
    write_json(dest/'bookmark.json',dict(stage=6,seed=info['native_seed'],stage_frame=info['stage_frame'],
        native_input_prefix=info['stage6_prefix_capture'],native_stage_record='native-stage.json',
        brain='brain.pt',checkpoint='../frozen-parameters.pt',configuration='configuration.json',
        attack_age_frames=info.get('stage6_attack_age'),
        brain_timing='immediately before the bookmarked stage 6 decision',source='frozen full Story'))


def rebuild_brain(model,features,actions,cut,seed,frame,output):
    """Recover an already running frozen student's boundary from its observed inputs and moves."""
    import numpy as np
    import torch
    from .frozen_binary_messages import FrozenBinaryMessages
    from .nbnm_prefix import burn_in
    from .nbnm_run import atomic_torch_save
    from .spiking_engineered_control import SpikingState
    output=Path(output)
    partial=output.with_suffix('.partial.pt')
    state=model.initial_state(1)
    start=0
    if partial.is_file():
        saved=torch.load(partial,map_location='cpu',weights_only=False)
        if (saved['opening_seed'],saved['cut'])!=(seed,cut):
            raise ValueError('Partial brain belongs to another entry')
        state=SpikingState(*(t.to(model.device) for t in saved['state']))
        start=saved['completed']
    with torch.no_grad(),FrozenBinaryMessages(model.core).capture():
        for a in range(start,cut,256):
            end=min(cut,a+256)
            history=dict(features=features[a:end],actions=actions[a:end])
            state=burn_in(model,{'policy_burn_in':history},state,use_graph=True)
            if (end%1024==0 or end==cut):
                atomic_torch_save(dict(state=[v.detach().cpu() for v in state.tensors()],
                    opening_seed=seed,cut=cut,completed=end),partial)
                write_json(output.with_suffix('.progress.json'),dict(completed=end,cut=cut))
    atomic_torch_save(dict(schema='frozen-SNN-stage-entry-brain-v2',opening_seed=seed,
        state=[v.detach().cpu() for v in state.tensors()],features=np.array(features[cut],copy=True),
        stage_frame=frame,source='recorded frozen candidate inputs and committed actions',optimizer_updates=0),
        output)
    partial.unlink(missing_ok=True)
    return state
