"""Jittor port of the pixel generator in the authors' MNIST notebook."""
import math

import jittor as jt
from jittor import nn
import numpy as np

from .attention import attention, step


class Position(nn.Module):
    def __init__(self, length, width):
        positions = np.arange(length, dtype=np.float32)[:, None]
        frequencies = np.exp(np.arange(0, width, 2) * (-math.log(10000) / width))
        table = np.zeros((1, length, width), dtype=np.float32)
        table[0, :, 0::2] = np.sin(positions * frequencies)
        table[0, :, 1::2] = np.cos(positions * frequencies)
        self.register_buffer("pe", jt.array(table).stop_grad())

    def execute(self, embedded, offset):
        batch, length, width = embedded.shape
        if offset < 0 or offset + length > self.pe.shape[1]:
            raise ValueError("Sequence exceeds positional embedding range")
        pos = self.pe[:, offset:offset + length].expand((batch, length, width))
        return jt.concat([embedded, pos], dim=-1)


class MultiHead(nn.Module):
    def __init__(self, width, heads, kind, attention_dropout):
        self.kind = kind
        self.heads = heads
        self.query_projection = nn.Linear(width, width)
        self.key_projection = nn.Linear(width, width)
        self.value_projection = nn.Linear(width, width)
        self.out_projection = nn.Linear(width, width)
        self.dropout = nn.Dropout(attention_dropout)

    def execute(self, x, incremental, state):
        batch, length, width = x.shape
        shape = (batch, length, self.heads, width // self.heads)
        q, k, v = (layer(x).reshape(shape) for layer in
                   (self.query_projection, self.key_projection, self.value_projection))
        if incremental:
            y, state = step(self.kind, q[:, 0], k[:, 0], v[:, 0], state)
        else:
            y = attention(self.kind, q, k, v, self.dropout)
        return self.out_projection(y.reshape(x.shape)), state


class Layer(nn.Module):
    def __init__(self, width, heads, kind, dropout, attention_dropout):
        self.attention = MultiHead(width, heads, kind, attention_dropout)
        self.linear1 = nn.Linear(width, 4 * width)
        self.linear2 = nn.Linear(4 * width, width)
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)

    def execute(self, x, incremental, state):
        attended, state = self.attention(x, incremental, state)
        x = self.norm1(x + self.dropout(attended))
        hidden = self.dropout(nn.relu(self.linear1(x)))
        return self.norm2(x + self.dropout(self.linear2(hidden))), state


class Encoder(nn.Module):
    def __init__(self, layers, width, heads, kind, dropout, attention_dropout):
        self.layers = nn.ModuleList([Layer(width, heads, kind, dropout, attention_dropout)
                                     for _ in range(layers)])
        self.norm = nn.LayerNorm(width)

    def execute(self, x, incremental, states):
        if states is None:
            states = [None] * len(self.layers)
        if len(states) != len(self.layers):
            raise ValueError("Expected one state per encoder layer")
        new_states = []
        for layer, state in zip(self.layers, states):
            x, state = layer(x, incremental, state)
            new_states.append(state)
        return self.norm(x), new_states


class Generator(nn.Module):
    def __init__(self, attention_type, d_model=256, sequence_length=783, mixtures=10,
                 n_layers=8, n_heads=8, d_query=32, dropout=0.1, attention_dropout=0.1):
        if d_model != n_heads * d_query or d_model % 4:
            raise ValueError("Expected width = heads * key dimension and a multiple of four")
        self.pos_embedding = Position(sequence_length, d_model // 2)
        self.value_embedding = nn.Embedding(256, d_model // 2)
        self.transformer = Encoder(n_layers, d_model, n_heads, attention_type,
                                   dropout, attention_dropout)
        self.predictor = nn.Linear(d_model, 3 * mixtures)

    def execute(self, tokens):
        x = self.pos_embedding(self.value_embedding(tokens), 0)
        return self.predictor(self.transformer(x, False, None)[0])

    def next_logits(self, token, position, states=None):
        x = self.pos_embedding(self.value_embedding(token.reshape((-1, 1))), position)
        x, states = self.transformer(x, True, states)
        return self.predictor(x[:, 0]), states

    def import_weights(self, arrays):
        current = self.state_dict()
        if set(current) != set(arrays):
            raise ValueError(f"Checkpoint keys differ: {sorted(set(current) ^ set(arrays))}")
        for name, tensor in current.items():
            if tuple(tensor.shape) != tuple(arrays[name].shape):
                raise ValueError(f"Checkpoint shape mismatch for {name}")
        self.load_parameters(arrays)


def mol_nll(logits, targets):
    """Per-target negative log discretized logistic mixture mass, in nats."""
    weights, means, raw = jt.chunk(logits, 3, dim=-1)
    scales = nn.elu(raw) + 1.0001
    labels = targets.unsqueeze(-1)
    centers = labels.float32() / 127.5 - 1
    upper = (centers + 1 / 255 - means) / scales
    lower = (centers - 1 / 255 - means) / scales
    # log(1 - exp(-gap)) via a small-gap expansion to avoid cancellation.
    gap = 2 / (255 * scales)
    small = jt.minimum(gap, 0.01)
    correction = jt.log(small) + jt.log(1 - small / 2 + small.sqr() / 6)
    log_gap = jt.where(gap < 0.01, correction,
                       jt.log(1 - jt.exp(-jt.maximum(gap, 0.01))))
    left, right = -nn.softplus(-upper), -nn.softplus(lower)
    mass = jt.where(labels == 0, left, jt.where(labels == 255, right, left + right + log_gap))
    terms = nn.log_softmax(weights, dim=-1) + mass
    peak = terms.max(-1, keepdims=True).detach()
    return -(peak.squeeze(-1) + (terms - peak).exp().sum(-1).log())


def sample_pixel(logits, rng):
    weights, means, raw = np.split(logits, 3, axis=-1)
    probs = np.exp(weights - weights.max(axis=-1, keepdims=True))
    probs /= probs.sum(axis=-1, keepdims=True)
    components = [rng.choice(len(row), p=row) for row in probs]
    idx = (np.arange(len(components)), components)
    scale = np.maximum(raw[idx], 0) + np.expm1(np.minimum(raw[idx], 0)) + 1.0001
    values = means[idx] + scale * rng.logistic(size=len(components))
    return np.rint((np.clip(values, -1, 1) + 1) * 127.5).astype(np.int32)
