"""Minimal standalone MLX SigLIP2 Base vision backend.

Weights are loaded from the pinned Hugging Face revision on first use.  The
cache pipeline creates this backend only on its single inference thread.
"""

from __future__ import annotations

import math
from typing import Any


def _load_dependencies():
    try:
        import mlx.core as mx
        import mlx.nn as nn
        import numpy as np
        import torch
        from transformers import AutoModel
    except ImportError as exc:  # pragma: no cover - target Mac dependency
        raise RuntimeError("MLX SigLIP2 extraction requires mlx, numpy, torch, and transformers") from exc
    return mx, nn, np, torch, AutoModel


class _MLXModelFactory:
    def __init__(self, model_id: str, revision: str):
        mx, nn, np, torch, AutoModel = _load_dependencies()
        self.mx, self.nn, self.np = mx, nn, np
        hf = AutoModel.from_pretrained(model_id, revision=revision)
        cfg = getattr(hf.config, "vision_config", hf.config)
        self.image_size = int(getattr(cfg, "image_size", 224))
        patch = int(getattr(cfg, "patch_size", 16))
        dim = int(getattr(cfg, "hidden_size", 768))
        intermediate = int(getattr(cfg, "intermediate_size", 3072))
        layers = int(getattr(cfg, "num_hidden_layers", 12))
        heads = int(getattr(cfg, "num_attention_heads", 12))
        self.model = _Vision(nn, dim, intermediate, layers, heads, self.image_size, patch)
        sd = hf.state_dict()
        def arr(name): return mx.array(sd[name].detach().cpu().numpy(), dtype=mx.float16)
        m = self.model
        m.emb.patch.weight = mx.array(sd["vision_model.embeddings.patch_embedding.weight"].detach().cpu().numpy().transpose(0,2,3,1), dtype=mx.float16)
        m.emb.patch.bias = arr("vision_model.embeddings.patch_embedding.bias")
        m.emb.pos.weight = arr("vision_model.embeddings.position_embedding.weight")
        for i, layer in enumerate(m.layers):
            prefix = f"vision_model.encoder.layers.{i}."
            for target, key in ((layer.n1, "layer_norm1"), (layer.n2, "layer_norm2")):
                target.weight = arr(prefix + key + ".weight"); target.bias = arr(prefix + key + ".bias")
            for target, key in ((layer.attn.q,"q_proj"),(layer.attn.k,"k_proj"),(layer.attn.v,"v_proj"),(layer.attn.o,"out_proj")):
                target.weight = arr(prefix + "self_attn." + key + ".weight"); target.bias = arr(prefix + "self_attn." + key + ".bias")
            for target, key in ((layer.mlp.fc1,"fc1"),(layer.mlp.fc2,"fc2")):
                target.weight = arr(prefix + "mlp." + key + ".weight"); target.bias = arr(prefix + "mlp." + key + ".bias")
        m.post.weight = arr("vision_model.post_layernorm.weight"); m.post.bias = arr("vision_model.post_layernorm.bias")
        m.head.probe = arr("vision_model.head.probe")
        m.head.ipw = arr("vision_model.head.attention.in_proj_weight"); m.head.ipb = arr("vision_model.head.attention.in_proj_bias")
        m.head.o.weight = arr("vision_model.head.attention.out_proj.weight"); m.head.o.bias = arr("vision_model.head.attention.out_proj.bias")
        m.head.ln.weight = arr("vision_model.head.layernorm.weight"); m.head.ln.bias = arr("vision_model.head.layernorm.bias")
        m.head.mlp.fc1.weight = arr("vision_model.head.mlp.fc1.weight"); m.head.mlp.fc1.bias = arr("vision_model.head.mlp.fc1.bias")
        m.head.mlp.fc2.weight = arr("vision_model.head.mlp.fc2.weight"); m.head.mlp.fc2.bias = arr("vision_model.head.mlp.fc2.bias")
        self.forward = lambda pixels: m(pixels)
        warmup = mx.zeros((1, self.image_size, self.image_size, 3), dtype=mx.float16)
        mx.eval(self.forward(warmup))

    def __call__(self, images: Any):
        mx, np = self.mx, self.np
        pixels = mx.array(np.asarray(images, dtype=np.float32) / 127.5 - 1.0, dtype=mx.float16)
        output = self.forward(pixels); mx.eval(output)
        output = np.asarray(output, dtype=np.float32)
        return output / np.maximum(np.linalg.norm(output, axis=1, keepdims=True), 1e-12)


def _linear(nn, dim, out): return nn.Linear(dim, out, bias=True)


class _Emb:
    def __init__(self, nn, dim, image_size, patch):
        self.patch = nn.Conv2d(3, dim, kernel_size=patch, stride=patch, bias=True)
        self.pos = nn.Embedding((image_size // patch) ** 2, dim)
    def __call__(self, x):
        y = self.patch(x); b,h,w,c = y.shape
        return y.reshape(b, h*w, c) + self.pos.weight


class _Attn:
    def __init__(self, nn, dim, heads):
        self.dim, self.heads, self.hdim = dim, heads, dim // heads
        self.q, self.k, self.v, self.o = (_linear(nn, dim, dim) for _ in range(4))
    def __call__(self, x):
        mx = __import__("mlx.core", fromlist=["core"]); b,l,_ = x.shape
        def split(y): return y.reshape(b,l,self.heads,self.hdim).transpose(0,2,1,3)
        q,k,v = split(self.q(x)),split(self.k(x)),split(self.v(x))
        y = mx.softmax((q @ k.transpose(0,1,3,2)) / math.sqrt(self.hdim), axis=-1) @ v
        return self.o(y.transpose(0,2,1,3).reshape(b,l,self.dim))


class _MLP:
    def __init__(self, nn, dim, intermediate): self.fc1,self.fc2 = _linear(nn,dim,intermediate),_linear(nn,intermediate,dim)
    def __call__(self,x): return self.fc2(nn_gelu(x, self.fc1(x)))


def nn_gelu(x, y):
    return __import__("mlx.nn", fromlist=["gelu_approx"]).gelu_approx(y)


class _Layer:
    def __init__(self, nn, dim, intermediate, heads): self.n1,self.attn,self.n2,self.mlp = nn.LayerNorm(dim,eps=1e-6),_Attn(nn,dim,heads),nn.LayerNorm(dim,eps=1e-6),_MLP(nn,dim,intermediate)
    def __call__(self,x):
        x = x + self.attn(self.n1(x)); return x + self.mlp(self.n2(x))


class _Head:
    def __init__(self, nn, dim, heads):
        self.dim,self.heads,self.hdim=dim,heads,dim//heads; self.probe=__import__("mlx.core",fromlist=["zeros"]).zeros((1,1,dim),dtype=__import__("mlx.core",fromlist=["float16"]).float16)
        mx=__import__("mlx.core",fromlist=["zeros"]); self.ipw=mx.zeros((3*dim,dim),dtype=mx.float16); self.ipb=mx.zeros((3*dim,),dtype=mx.float16)
        self.o=nn.Linear(dim,dim); self.ln=nn.LayerNorm(dim,eps=1e-6); self.mlp=_MLP(nn,dim,3072)
    def __call__(self,x):
        mx=__import__("mlx.core",fromlist=["softmax"]); b,l,d=x.shape; query=__import__("mlx.core",fromlist=["broadcast_to"]).broadcast_to(self.probe,(b,1,d))
        def proj(w,z,bias): return z @ w.T + bias
        q,k,v=proj(self.ipw[:d],query,self.ipb[:d]),proj(self.ipw[d:2*d],x,self.ipb[d:2*d]),proj(self.ipw[2*d:],x,self.ipb[2*d:])
        q=q.reshape(b,1,self.heads,self.hdim).transpose(0,2,1,3); k=k.reshape(b,l,self.heads,self.hdim).transpose(0,2,1,3); v=v.reshape(b,l,self.heads,self.hdim).transpose(0,2,1,3)
        y=(mx.softmax((q@k.transpose(0,1,3,2))/math.sqrt(self.hdim),axis=-1)@v).transpose(0,2,1,3).reshape(b,1,d)
        hidden=query+self.o(y); residual=hidden; hidden=self.ln(hidden); y=(residual+self.mlp(hidden)).squeeze(1)
        return y / mx.linalg.norm(y,axis=-1,keepdims=True)


class _Vision:
    def __init__(self, nn, dim, intermediate, layers, heads, image_size, patch): self.emb=_Emb(nn,dim,image_size,patch); self.layers=[_Layer(nn,dim,intermediate,heads) for _ in range(layers)]; self.post=nn.LayerNorm(dim,eps=1e-6); self.head=_Head(nn,dim,heads)
    def __call__(self,x):
        x=self.emb(x)
        for layer in self.layers: x=layer(x)
        return self.head(self.post(x))


def load_backend(model_id: str, revision: str):
    return _MLXModelFactory(model_id, revision)
