"""Fair forward task rotation for one shared NBNM policy.

Rotation happens only after an actual episode terminal. PPO update boundaries
never reset a game. A pool is interleaved by stage so late stages get exposure
from the first rollout, rather than waiting for a successful opening run.
"""
from copy import deepcopy


def interleave_stages(keys,tasks):
    buckets={s:[] for s in range(1,7)}
    for key in keys:buckets[tasks[key].get('stage',1)].append(key)
    return [rows[i] for i in range(max(map(len,buckets.values()),default=0))
            for rows in buckets.values() if i<len(rows)]


def worker_settings(spec):
    from .nbnm_run import task_settings
    pools=spec.get('task_pools')
    if not pools:return [(spec['tasks'][g],spec.get('task_entries',{}).get(g,{})) for g in spec['world_groups']]
    from collections import Counter
    sizes=Counter(spec['world_groups']);offsets={};result=[]
    for group in spec['world_groups']:
        index=offsets.get(group,0);offsets[group]=index+1
        keys=pools[group];first=keys[index%len(keys)]
        pool=[dict(key=k,settings=task_settings(spec,spec['tasks'][k])) for k in keys]
        entry=deepcopy(spec.get('task_entries',{}).get(first,{}))
        # Stagger each group's workers across the whole catalog. Advancing
        # every worker by one would repeat the first few cards for many rounds.
        entry.update(task_pool=pool,task_pool_offset=index,task_pool_stride=sizes[group],task_key=first)
        result.append((spec['tasks'][first],entry))
    return result


class TaskRotation:
    def __init__(self,rows,offset=0,stride=1):
        if not rows:raise ValueError('A training task pool cannot be empty')
        self.rows=deepcopy(rows);self.cursor=int(offset);self.stride=max(1,int(stride));self.last_cursor=None

    def next(self):
        self.last_cursor=self.cursor
        row=deepcopy(self.rows[self.cursor%len(self.rows)]);self.cursor+=self.stride
        return row
