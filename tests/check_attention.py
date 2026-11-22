"""Independent dense PyTorch oracle for the Jittor attention operators."""
import argparse

import jittor as jt
import numpy as np
import torch
from torch.nn import functional as F

from src.attention import attention, step
from src.runtime import configure, write_result


def reference(kind, q, k, v):
    if kind == "causal-linear":
        scores = torch.einsum("blhd,bshd->bhls", F.elu(q) + 1, F.elu(k) + 1)
        weights = scores.tril()
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-6)
    else:
        scores = torch.einsum("blhd,bshd->bhls", q, k) / q.shape[-1] ** 0.5
        mask = torch.ones(scores.shape[-2:], dtype=torch.bool).tril()
        weights = scores.masked_fill(~mask, -torch.inf).softmax(-1)
    return torch.einsum("bhls,bshv->blhv", weights, v)


def compare(actual, expected):
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-4)
    return float(np.max(np.abs(actual - expected)))


def check(kind, length, rng, dump):
    q, k, v, upstream = [rng.normal(size=(2, length, 2, d)).astype("float32")
                         for d in (4, 4, 3, 3)]
    inputs = [torch.tensor(a, requires_grad=True) for a in (q, k, v)]
    ref = reference(kind, *inputs)
    ref.backward(torch.from_numpy(upstream))
    x = [jt.array(a) for a in (q, k, v)]
    output = attention(kind, *x)
    gradients = jt.grad((output * jt.array(upstream)).sum(), x)
    expected = ref.detach().numpy()
    actual = output.numpy()
    errors = {"output": compare(actual, expected)}
    for name, gradient, tensor in zip(("dQ", "dK", "dV"), gradients, inputs):
        errors[name] = compare(gradient.numpy(), tensor.grad.numpy())
    outputs, state = [], None
    for t in range(length):
        y, state = step(kind, *(a[:, t] for a in x), state)
        outputs.append(y.numpy())
        expected_shapes = ((2, 2, 4, 3), (2, 2, 4)) if kind == "causal-linear" else (
            (2, t+1, 2, 4), (2, t+1, 2, 3))
        assert tuple(tuple(a.shape) for a in state) == expected_shapes
    errors["recurrent"] = compare(np.stack(outputs, axis=1), actual)
    if length > 1:
        changed = [a.copy() for a in (q, k, v)]
        for a in changed:
            a[:, length // 2:] += 5
        new = attention(kind, *(jt.array(a) for a in changed)).numpy()
        np.testing.assert_array_equal(new[:, :length // 2], actual[:, :length // 2])
        errors["future_leak"] = 0.0
    if dump:
        dump.mkdir(parents=True, exist_ok=True)
        np.savez(dump / f"{kind}-{length}.npz", q=q, k=k, v=v, upstream=upstream,
                 output=expected, **{n: t.grad.numpy() for n, t in zip(("dq","dk","dv"), inputs)})
    return {"kind": kind, "length": length, "max_abs": errors}


if __name__ == "__main__":
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--fixtures", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    env = configure(args.device)
    torch.set_num_threads(1)
    rows = [check(kind, length, np.random.default_rng(1229 + length), args.fixtures)
            for kind in ("full", "causal-linear") for length in (1, 17)]
    write_result(args.out, dict(environment=env, torch=torch.__version__, seed=1229,
                                batch=2, heads=2, key_dim=4, value_dim=3, cases=rows))
