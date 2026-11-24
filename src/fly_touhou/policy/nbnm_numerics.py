"""Validate CPU data already copied for NBNM sampling and diagnostics."""
import numpy as np

REVISION='nbnm-observed-forecast-and-neural-finiteness-v1'


def validate_network_arrays(network,*,features=None,selected_DN=None,log_probs=None):
    """Never let argmax or a later mask hide a broken neural output.

    No extra CUDA synchronization: callers supply their existing CPU copies.
    Negative infinity is permitted only as a logit mask, with legal support.
    """
    arrays=dict(network)
    for name,value in (('features',features),('selected_DN',selected_DN),('log_probs',log_probs)):
        if value is not None:arrays[name]=value
    for name,value in arrays.items():
        x=np.asarray(value)
        if name=='logits':
            valid=(np.isfinite(x)|np.isneginf(x)).all() and np.isfinite(x).any(axis=-1).all()
        else:valid=np.isfinite(x).all()
        if not valid:raise FloatingPointError('Non-finite NBNM inference: '+name)


def verified_movement_actions(scores, actions):
    """Verify the chosen commands against all 18 neural movement scores."""
    scores=np.asarray(scores)
    actions=np.asarray(actions)
    if (scores.shape!=(len(actions),18) or not np.isfinite(scores).all()
            or not np.array_equal(scores.argmax(axis=-1),actions)):
        raise ValueError('Frozen original movement must choose the highest of 18 finite neural scores')
    return len(actions)


def validate_saved_brain_state(state,*,world,decision):
    # These state blocks have already been copied to CPU for recurrent PPO.
    for name in ('membrane','synaptic','spikes','rate'):
        x=getattr(state,name).numpy()
        if not np.isfinite(x).all():
            raise FloatingPointError(f'Non-finite NBNM brain {name}, world {world}, decision {decision}')
