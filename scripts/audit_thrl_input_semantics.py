"""Inventory every original scalar and pin which branch consumes each meaning."""
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def ast_fingerprint(path,names):
    tree=ast.parse(Path(path).read_text('utf-8'))
    nodes={node.name:node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef))}
    return {name:hashlib.sha256(ast.dump(nodes[name],include_attributes=False).encode()).hexdigest() for name in names}


def audit():
    from fly_touhou.thrl_inputs import ORIGINAL_INPUT,SEPARATED_INPUT,make_encoder
    from fly_touhou.thrl_observation import THRLObservationEncoder
    from fly_touhou.thrl_geometry_observation import GLOBAL_FIELDS,OBJECT_FIELDS
    global_indices={0:0,1:1,8:2,9:3,10:4,11:5,12:6,13:7,16:8,32:9,33:10,34:11,35:12}
    rows=[]
    for i,name in enumerate(THRLObservationEncoder.feature_names):
        if i in global_indices:
            action='keep_global_once';target=GLOBAL_FIELDS[global_indices[i]]
        elif i in (2,3,14,15):
            action='remove_unavailable_constant_zero';target=None
        elif i in (4,5,6,7,31,36):
            action='remove_deterministic_duplicate';target='player coordinates / stage / difficulty already retained'
        elif 17<=i<31:
            field=(i-17)%7
            action='object_only' if field!=6 else 'remove_derived_distance'
            target=('enemy/boss token: present,hp,dx,dy,vx,vy' if field!=6 else 'derivable from dx,dy')
        elif 37<=i<149:
            field=(i-37)%7
            action=('object_only' if field in (0,1,5) else
                    'remove_derived_distance' if field==6 else 'remove_constant_or_ambiguous_type')
            target='current laser samples; measured width added; velocities explicitly unavailable (zero)'
        elif 149<=i<261:
            field=(i-149)%7
            action='object_only' if field<5 else 'remove_derived_speed_or_distance'
            target='current bullet dx,dy,vx,vy,kind; add measured collision shape and gating'
        else:
            action='object_only';target='item token dx,dy,kind; additionally current measured velocity'
        rows.append({'index':i,'original_field':name,'original_group':'retained exactly in273 (routed by meaning)',
                     'separated_group_action':action,'target_or_reason':target})
    import torch
    from fly_touhou.thrl_model import MOActorCritic
    torch.set_num_threads(2)
    models={schema:MOActorCritic(feature_dim=273 if schema==ORIGINAL_INPUT else 20,action_dim=19,
        attention='both',input_schema=schema,object_embed_dim=256,object_attention_heads=8,
        object_attention_layers=4,spatial_attention_layers=2) for schema in (ORIGINAL_INPUT,SEPARATED_INPUT)}
    old=Path('./outputs/source_snapshots/20261021-thrl-parallel-abc-v3/src/fly_touhou')
    current=ROOT/'src/fly_touhou'
    functions=['PPOConfig','compute_gae','entropy_coeff','combine_advantages','ppo_loss_terms','make_optimizer']
    before=ast_fingerprint(old/'thrl_ppo.py',functions);after=ast_fingerprint(current/'thrl_ppo.py',functions)
    if before!=after:raise ValueError('Scientific PPO functions changed')
    reward_sha=hashlib.sha256((current/'thrl_rewards.py').read_bytes()).hexdigest()
    if reward_sha!=hashlib.sha256((old/'thrl_rewards.py').read_bytes()).hexdigest():
        raise ValueError('Reward file changed')
    return {'schema':'thrl-input-semantics-review-v1','all_original_scalars_reviewed':len(rows),
        'scalar_decisions':rows,'new_global_fields':list(GLOBAL_FIELDS),'new_object_fields':list(OBJECT_FIELDS),
        'input_specifications':{key:make_encoder(key).specification() for key in models},
        'models':{key:{'configuration':model.model_config(),'parameters':model.parameter_counts(),
                       'attention':model.attention_specification()} for key,model in models.items()},
        'common_training':{'actions':19,'frame_skip':4,'finite_lives':True,'autonomous_bomb':True,
            'continues':False,'episode_native_frame_cap':None,'PPO_functions_AST_unchanged':after,
            'PPO_changes':'optional independent objects tensor plumbing in rollout/update only',
            'reward_file_unchanged_sha256':reward_sha},
        'intentional_complementarity':['Full-field map retains every source object, while tokens retain at most96 precise entities.',
            'Object positions and map occupancy describe the same scene at different spatial scales.',
            'Map distance/type channels are retained in this first comparison to preserve the agreed16-channel map; ablation remains untested.'],
        'comparison_limitations':['This tests two complete input designs (K/geometry/global pruning), not one isolated feature.',
            'Original273 item padding ambiguity and constant placeholders are retained only in the reference.',
            'Input projection sizes differ slightly; shared trunk/attention widths are matched.',
            'Four groups do not substitute for independent training seeds or formal held-out games.']}


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    report=audit();a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',encoding='utf-8') as stream:json.dump(report,stream,ensure_ascii=False,indent=2)
    print(json.dumps({'reviewed':len(report['scalar_decisions']),
        'parameters':{key:value['parameters']['total'] for key,value in report['models'].items()}}))
