"""Explicit OS scheduling of a verified learner's own worker descendants.

No work on import; no model/native hooks or input actions. The learner itself
is untouched. Receipts retain prior masks/priorities for reversible scheduling.
"""
from __future__ import annotations

import math

ALLOW_NAMES={'python.exe','pythonw.exe','th14.exe','frida-helper-x86.exe','frida-helper-x86_64.exe'}


def worker_cpu_pool(available):
    cpus=sorted(set(available))
    if len(cpus)>6:return cpus[4:-2]
    return cpus[2:] if len(cpus)>2 else cpus


def spread_owned_workers(pid,expected_create_time,cpus):
    """Only already-owned descendants, no launches, termination, or parent change."""
    import psutil
    if (type(pid) is not int or pid<=0 or not math.isfinite(expected_create_time)
            or not cpus or any(type(i) is not int or i<0 for i in cpus)):
        raise ValueError('Explicit parent identity and CPU list required')
    parent=psutil.Process(pid)
    if not parent.is_running() or abs(parent.create_time()-expected_create_time)>.01:
        raise RuntimeError('Learner PID/creation time does not match')
    before_parent=parent.cpu_affinity()
    rows=[]
    for child in parent.children(recursive=True):
        try:
            if child.name().lower() not in ALLOW_NAMES:continue
            created=child.create_time()
            if created<expected_create_time or not child.is_running():continue
            if not any(p.pid==pid and abs(p.create_time()-expected_create_time)<.01 for p in child.parents()):
                continue
            before=child.cpu_affinity();priority=child.nice()
            row=dict(PID=child.pid,create_time=created,name=child.name(),old_affinity=before,
                     old_priority=priority,requested_affinity=list(cpus))
            try:
                child.cpu_affinity(list(cpus))
                child.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
                row.update(applied_affinity=child.cpu_affinity(),applied_priority=child.nice(),status='applied')
            except psutil.Error as error:
                row.update(status='unavailable',error=repr(error))
            rows.append(row)
        except psutil.Error:
            # A natural game/helper can exit between enumeration and lookup.
            continue
    if parent.cpu_affinity()!=before_parent:raise RuntimeError('Learner affinity unexpectedly changed')
    return dict(parent_PID=pid,parent_create_time=expected_create_time,
        parent_affinity_before=before_parent,parent_affinity_after=parent.cpu_affinity(),
        worker_cpu_pool=list(cpus),workers=rows,game_actions_sent=0,new_processes_started=0,
        processes_terminated=0,model_or_reward_or_frame_step_changed=False)
