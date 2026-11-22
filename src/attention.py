"""Causal attention on [batch, time, heads, features] tensors."""
import math

import jittor as jt
from jittor import nn


def feature(x):
    return nn.elu(x) + 1


def normalize(q, s, z):
    return (q.unsqueeze(-1) * s).sum(-2) / ((q * z).sum(-1, keepdims=True) + 1e-6)


def attention(kind, q, k, v, dropout=None):
    if kind == "causal-linear":
        keys = feature(k)
        products = keys.unsqueeze(-1) * v.unsqueeze(-2)
        return normalize(feature(q), products.cumsum(1), keys.cumsum(1))
    if kind != "full":
        raise ValueError(kind)
    q, k, v = (x.transpose(1, 2) for x in (q, k, v))
    scores = jt.matmul(q / math.sqrt(q.shape[-1]), k.transpose(-1, -2))
    positions = jt.arange(k.shape[-2])
    mask = positions.unsqueeze(0) <= positions.unsqueeze(1)
    weights = nn.softmax(jt.where(mask, scores, -float("inf")), dim=-1)
    if dropout is not None:
        weights = dropout(weights)
    return jt.matmul(weights, v).transpose(1, 2)


def step(kind, q, k, v, state=None):
    """Consume the current token, return its output and per-head state."""
    if kind == "causal-linear":
        z = feature(k)
        s = z.unsqueeze(-1) * v.unsqueeze(-2)
        if state is not None:
            s = state[0] + s
            z = state[1] + z
        return normalize(feature(q), s, z), (s, z)
    if kind != "full":
        raise ValueError(kind)
    keys, values = k.unsqueeze(1), v.unsqueeze(1)
    if state is not None:
        keys = jt.concat([state[0], keys], dim=1)
        values = jt.concat([state[1], values], dim=1)
    scores = (q.unsqueeze(1) * keys).sum(-1) / math.sqrt(q.shape[-1])
    weights = nn.softmax(scores, dim=1)
    return (weights.unsqueeze(-1) * values).sum(1), (keys, values)
