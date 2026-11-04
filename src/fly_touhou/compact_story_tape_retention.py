"""Retain live supervision independently of expendable long-batch tapes."""
import os
import re
from pathlib import Path
import torch


def load_replays(path,revisions):
    """Keep observed deathbomb facts when a geometry repair invalidates labels.

    Historical route/forecast labels must not survive a sensor revision. A
    recorded stocked native state4 window remains a valid Bomb target, though,
    and retains its original recurrent state and movement history.
    """
    archive=torch.load(path,map_location='cpu',weights_only=False)
    if archive['native_repair_revisions']==revisions:return archive
    critical=archive.get('critical_replay');removed=0;rows=0;sequences=0
    if critical is not None:
        for entry in critical.sequences:
            data=entry['data'];removed+=int((data['rule_weights']>0).sum())
            data['rule_weights'].zero_()
            rows+=int(data['deathbomb_mask'].sum())
        sequences=len(critical.sequences)
        critical.normal_sequences=[]
        critical.native_repair_revisions=dict(revisions)
        critical.outdated_geometry_rule_rows+=removed
    migration=dict(previous_revision=archive['native_repair_revisions'],
        current_revision=dict(revisions),retained_native_deathbomb_rows=rows,
        retained_critical_sequences=sequences,obsolete_route_rows_excluded=removed,
        obsolete_late_risk_replay_excluded=archive.get('late_replay') is not None,
        original_recurrent_states_preserved=True,PPO_samples_added=0)
    return dict(archive,native_repair_revisions=dict(revisions),late_replay=None,
                geometry_revision_migration=migration)


def merge_native_deathbombs(current,restored):
    """Merge sanitized archived facts into an already-running replay bank."""
    incoming=restored.get('critical_replay')
    if incoming is None:return dict(sequences_added=0,deathbomb_rows_added=0)
    if incoming.native_repair_revisions!=current.native_repair_revisions:
        raise ValueError('Load the archived facts for the current sensor revision first')
    def key(entry):
        return (str(entry['source']).replace('\\','/').lower(),
                entry['source_time'],entry['source_world'])
    present={key(entry) for entry in current.sequences};added=rows=0
    for entry in incoming.sequences:
        identity=key(entry)
        if identity in present:continue
        current.sequences.append(entry);present.add(identity);added+=1
        rows+=int(entry['data']['deathbomb_mask'].sum())
    current.observed_rows+=rows
    current.sources.extend(source for source in incoming.sources if source not in current.sources)
    return dict(sequences_added=added,deathbomb_rows_added=rows,
                obsolete_route_labels_restored=False,PPO_samples_added=0)


def save_replays(path,critical,late,revisions):
    path=Path(path);temp=path.with_suffix('.tmp')
    payload=dict(schema='compact-student-supervision-archive-v1',native_repair_revisions=revisions,
        critical_replay=critical,late_replay=late.payload() if late is not None else None,
        teacher_data_is_PPO=False)
    with temp.open('wb') as stream:
        torch.save(payload,stream);stream.flush();os.fsync(stream.fileno())
    os.replace(temp,path)


def prune_tapes(directory,keep):
    directory=Path(directory).resolve(strict=True)
    if keep<1:raise ValueError('Keep at least one completed tape')
    candidates=[]
    for path in directory.glob('rollout-*.pt'):
        match=re.fullmatch(r'rollout-(\d+)\.pt',path.name)
        if match:
            if path.resolve().parent!=directory:raise ValueError('Tape escapes the active output directory')
            candidates.append((int(match[1]),path))
    deleted=[]
    for cycle,path in sorted(candidates)[:-keep]:
        size=path.stat().st_size;path.unlink()
        deleted.append(dict(cycle=cycle,path=str(path),bytes=size))
    return dict(keep_full_tapes=keep,deleted=deleted,deleted_bytes=sum(p['bytes'] for p in deleted),
        scope='Only numbered full rollout caches in this job;logs,metrics,checkpoints,and supervision archives retained')
