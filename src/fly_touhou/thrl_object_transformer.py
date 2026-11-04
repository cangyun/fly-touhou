"""Masked multi-layer entity attention inspired by nokozero's token encoder.

PyTorch/PPO adaptation, not its JAX DQN objective. Coordinates carry spatial
information; no rank/slot embeddings make object attention order-dependent.
"""
import torch
from torch import nn
from .thrl_tensor_constants import constant_like

from .thrl_geometry_observation import FEATURE_DIM, OBJECT_COUNT, OBJECT_DIM
from .thrl_inputs import (ORIGINAL_INPUT, SEPARATED_INPUT, VERSIONED_SCHEMAS,
                         is_original_input, feature_dimension, global_feature_dimension,object_dimension)


def kinematic_object_features(features,objects):
    """Eight near-range descriptors derived from the same measured fields.

    No chosen action, future script, teacher direction or native memory enters.
    The transformer still decides which entities matter and the policy acts.
    """
    if features.shape[-1]!=25 or objects.shape[-1] not in (20,28):
        raise ValueError('Kinematic embedding requires separated25 and20 object fields')
    original=objects;objects=objects[...,:20]
    r=objects[...,5:7]*constant_like(objects,[384.,448.]);v=objects[...,7:9]*12
    if original.shape[-1]==28:
        laser=(objects[...,2]>.5)|(objects[...,3]>.5)
        def decode(x,scale):return x*scale/(1-x.abs()).clamp_min(1e-6)
        r=torch.where(laser[...,None],decode(objects[...,5:7],constant_like(objects,[384.,448.])),r)
        v=torch.where(laser[...,None],decode(objects[...,7:9],12.),v)
        half=decode(original[...,20:22],constant_like(objects,[384.,448.]))
        span=2*half;first=r-half
        fraction=(-(first*span).sum(-1)/span.square().sum(-1).clamp_min(1e-9)).clamp(0,1)
        r=torch.where(laser[...,None],first+fraction[...,None]*span,r)
        delta=decode(original[...,22:24],12.)
        v=torch.where(laser[...,None],v+(2*fraction-1)[...,None]*delta,v)
    distance=(r.square().sum(-1)+1e-6).sqrt();speed=(v.square().sum(-1)+1e-6).sqrt()
    dot=(r*v).sum(-1);tau=(-dot/(v.square().sum(-1)+1e-4)).clamp(-24,48)
    radius=(objects[...,9].square()*128**2+features[:,None,13].square()*64**2+1e-6).sqrt()
    if original.shape[-1]==28:
        laser_radius=decode(objects[...,9],128.)+features[:,None,13]*64
        radius=torch.where(laser,laser_radius,radius)
    extents=objects[...,9:11]*128+features[:,None,14:16]*64
    offsets=features[:,None,16:18]*64
    def gap(position):
        circle=(position.square().sum(-1)+1e-6).sqrt()-radius
        rectangle=((-position+offsets).abs()-extents).amax(-1)
        return torch.where(objects[...,11]>.5,circle,rectangle)
    zoom=torch.asinh(r/8)/4.852091
    derived=torch.cat((zoom,gap(r)[...,None].div(8).tanh(),
        gap(r+v*tau.clamp_min(0)[...,None])[...,None].div(8).tanh(),
        (tau/48)[...,None],(-dot/(distance*speed+1e-4)).clamp(-1,1)[...,None],
        (speed/8).tanh()[...,None],(radius/16).tanh()[...,None]),-1)
    # Preserve all legacy first20 +8 derived input columns. New laser fields
    # append after them, permitting an explicit zero-column weight expansion.
    return torch.cat((objects,derived,original[...,20:]),-1)


class EntityBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.norm1=nn.LayerNorm(width)
        self.attention=nn.MultiheadAttention(width,heads,bias=False,dropout=0.,batch_first=True)
        self.norm2=nn.LayerNorm(width)
        self.ffn=nn.Sequential(nn.Linear(width,2*width),nn.GELU(),nn.Linear(2*width,width))

    def forward(self,tokens,padding):
        norm=self.norm1(tokens)
        tokens=tokens+self.attention(norm,norm,norm,key_padding_mask=padding,need_weights=False)[0]
        return tokens+self.ffn(self.norm2(tokens))


class GeometryObjectAttention(nn.Module):
    def __init__(self,output_dim=128,embed_dim=256,heads=8,layers=4,initialization='standard',
                 input_schema=SEPARATED_INPUT,geometry_embedding=False):
        super().__init__()
        if layers<1 or heads<1 or embed_dim%heads:
            raise ValueError('Positive layer count and embedding divisible by attention heads required')
        self.embed_dim,self.heads,self.layers,self.output_dim=embed_dim,heads,layers,output_dim
        if input_schema not in VERSIONED_SCHEMAS:
            raise ValueError('Explicit original or separated input required')
        self.input_schema=input_schema
        self.geometry_embedding=geometry_embedding
        if geometry_embedding and (is_original_input(input_schema) or feature_dimension(input_schema)!=25):
            raise ValueError('Derived geometry requires separated25 native input')
        self.global_dim=global_feature_dimension(input_schema)
        self.object_count=38 if is_original_input(input_schema) else OBJECT_COUNT
        self.object_dim=None if is_original_input(input_schema) else object_dimension(input_schema)
        raw_dim=7 if is_original_input(input_schema) else self.object_dim-1
        self.object_encoder=nn.Sequential(nn.Linear(raw_dim+(8 if geometry_embedding else 0),embed_dim),
                                         nn.GELU(),nn.Linear(embed_dim,embed_dim))
        self.type_embedding=nn.Embedding(4,embed_dim) if is_original_input(input_schema) else None
        if self.type_embedding is not None:
            self.register_buffer('object_types',torch.tensor([0]*2+[1]*16+[2]*16+[3]*4))
        self.blocks=nn.ModuleList([EntityBlock(embed_dim,heads) for _ in range(layers)])
        self.pool_query=nn.Parameter(torch.zeros(1,1,embed_dim))
        self.global_query=nn.Linear(self.global_dim,embed_dim)
        self.pool_norm=nn.LayerNorm(embed_dim)
        self.pool=nn.MultiheadAttention(embed_dim,heads,bias=False,dropout=0.,batch_first=True)
        self.context_projection=nn.Linear(embed_dim,output_dim)
        self.residual_gate=nn.Parameter(torch.tensor(0. if initialization=='identity' else 1.))

    def split_features(self,features,objects=None):
        expected_dim=feature_dimension(self.input_schema)
        if features.ndim!=2 or features.shape[-1]!=expected_dim:
            raise ValueError(f'Expected versioned features[B,{expected_dim}]')
        if is_original_input(self.input_schema):
            if objects is not None:
                raise ValueError('Original273 input must not receive new object measurements')
            from .thrl_model import ObjectCrossAttention
            globals,tokens,valid=ObjectCrossAttention.split_features(features[:,:273])
            if expected_dim>273:
                globals=torch.cat((globals,features[:,273:]),dim=-1)
            return globals,tokens,valid
        if objects is None:
            raise ValueError(f'Separated input requires globals[B,{expected_dim}] AND objects[B,96,{self.object_dim}]')
        if objects.shape!=(features.shape[0],OBJECT_COUNT,self.object_dim):
            raise ValueError('Wrong separate object tensor shape')
        return features,objects[...,1:],objects[...,0]>.5

    def forward(self,features,objects=None):
        globals,objects,valid=self.split_features(features,objects)
        objects=torch.where(valid[...,None],objects,torch.zeros_like(objects))
        if self.geometry_embedding:objects=kinematic_object_features(features,objects)
        tokens=self.object_encoder(objects)
        if self.type_embedding is not None:
            tokens=tokens+self.type_embedding(self.object_types)[None]
        empty=~valid.any(dim=1)
        safe=valid.clone()
        safe[:,0]|=empty
        tokens=torch.where(valid[...,None],tokens,torch.zeros_like(tokens))
        for block in self.blocks:
            tokens=block(tokens,~safe)
            tokens=torch.where(valid[...,None],tokens,torch.zeros_like(tokens))
        query=self.pool_query+self.global_query(globals)[:,None]
        norm=self.pool_norm(tokens)
        context=self.pool(query,norm,norm,key_padding_mask=~safe,need_weights=False)[0][:,0]
        result=self.context_projection(context)
        return self.residual_gate*torch.where(empty[:,None],torch.zeros_like(result),result)
