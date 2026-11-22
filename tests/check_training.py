"""Real-image training, held-out evaluation, generation and checkpoint replay."""
import argparse
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import jittor as jt
import numpy as np
import torch

from src.data import decode, encode, read_images, shifted
from src.model import Generator, mol_nll, sample_pixel
from src.runtime import configure, write_result
from src.training import RAdam


def check_optimizer():
    # Eight updates exercise both the initial and variance-rectified RAdam branches.
    layer = jt.nn.Linear(3, 1, bias=False)
    initial = np.array([[0.25, -0.5, 1.]], dtype=np.float32)
    layer.weight.update(jt.array(initial))
    param = torch.nn.Parameter(torch.from_numpy(initial.copy()))
    other = torch.optim.RAdam([param], lr=1e-3)
    opt = RAdam(layer, lr=1e-3)
    rng = np.random.default_rng(1229)
    errors = []
    for _ in range(8):
        grad = rng.normal(size=initial.shape).astype(np.float32)
        param.grad = torch.from_numpy(grad)
        other.step()
        opt.step((layer.weight * jt.array(grad)).sum())
        actual, expected = layer.weight.numpy(), param.detach().numpy()
        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=1e-5)
        errors.append(float(np.abs(actual - expected).max()))
    return max(errors)


def check_loss():
    rng = np.random.default_rng(1229)
    logits = rng.normal(size=(1, 6, 30)).astype(np.float32)
    labels = np.array([[0, 1, 93, 127, 254, 255]], dtype=np.int32)
    a = torch.tensor(logits, dtype=torch.float64, requires_grad=True)
    weights, means, raw = a.chunk(3, dim=-1)
    scale = torch.nn.functional.elu(raw) + 1.0001
    centers = torch.tensor(labels, dtype=torch.float64).unsqueeze(-1) / 127.5 - 1
    left = torch.sigmoid((centers - 1/255 - means) / scale)
    right = torch.sigmoid((centers + 1/255 - means) / scale)
    mass = torch.where(centers == -1, right, torch.where(centers == 1, 1-left, right-left))
    ref = -(weights.softmax(-1) * mass).sum(-1).log()
    ref.sum().backward()
    b = jt.array(logits)
    loss = mol_nll(b, jt.array(labels))
    checks = {"nll": (loss.numpy(), ref.detach().numpy()),
              "gradient": (jt.grad(loss.sum(), b).numpy(), a.grad.numpy())}
    errors = {}
    for name, (actual, expected) in checks.items():
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-4)
        errors[name] = float(np.abs(actual - expected).max())
    return errors


def tensors(batch, length):
    return tuple(jt.array(a.astype("int32")) for a in shifted(batch, length))


def check_data(dataset, path):
    patch = np.array([[[[1, 11, 21], [2, 12, 22]], [[3, 13, 23], [4, 14, 24]]]], np.uint8)
    np.testing.assert_array_equal(encode(patch)[0], [1,11,21,2,12,22,3,13,23,4,14,24])
    splits = []
    for split in ("train", "test"):
        images = read_images(dataset, path, split, 2)
        tokens = encode(images)
        assert tokens.shape == (2, 784 if dataset == "mnist" else 3072)
        if dataset == "cifar10":
            expected_rgb = [59, 62, 63] if split == "train" else [158, 112, 49]
            np.testing.assert_array_equal(tokens[0, :3], expected_rgb)
        np.testing.assert_array_equal(decode(tokens, images.shape[1:]), images)
        x, y = shifted(tokens)
        np.testing.assert_array_equal(x[:, 1:], y[:, :-1])
        splits.append(tokens)
    return splits


def check(dataset, kind, train, test):
    jt.set_global_seed(1229)
    length = 192
    config = dict(attention_type=kind, d_model=32, n_heads=2, d_query=16, n_layers=1,
                  sequence_length=train.shape[1] - 1, dropout=0., attention_dropout=0.)
    x, y = tensors(train, length)
    tx, ty = tensors(test, length)
    model = Generator(**config)
    opt = RAdam(model, lr=1e-4)
    before = model.predictor.weight.numpy().copy()

    def update(net, optimizer):
        loss = mol_nll(net(x), y).mean()
        value = float(loss.item())
        if not np.isfinite(value):
            raise AssertionError("Nonfinite training loss")
        optimizer.step(loss)
        return value

    first = update(model, opt)
    delta = float(np.max(np.abs(model.predictor.weight.numpy() - before)))
    assert delta > 0
    with TemporaryDirectory() as tmp:
        saved = Path(tmp) / "step.npz"
        opt.save(saved)
        second = update(model, opt)
        replay = Generator(**config)
        other = RAdam(replay)
        other.restore(saved)
        second_again = update(replay, other)
        assert opt.steps == other.steps == 2
        diffs = []
        for name, param in model.state_dict().items():
            a, b = param.numpy(), replay.state_dict()[name].numpy()
            np.testing.assert_allclose(a, b, atol=2e-6, rtol=1e-5)
            diffs.append(float(np.abs(a - b).max()))
        np.testing.assert_allclose(second, second_again, atol=2e-6, rtol=1e-5)

    model.eval()
    rng = np.random.default_rng(1229)
    with jt.no_grad():
        nll = mol_nll(model(tx), ty).numpy()
        bpd = float(nll.sum(dtype=np.float64) / (nll.size * math.log(2)))
        assert nll.shape == (2, length) and np.isfinite(bpd)
        token = test[:1, 0].astype("int32")
        generated, states = [], None
        for t in range(8):
            logits, states = model.next_logits(jt.array(token), t, states)
            token = sample_pixel(logits.numpy(), rng)
            generated.append(int(token[0]))
        assert all(0 <= value <= 255 for value in generated)
    return dict(dataset=dataset, kind=kind, tokens_per_image=train.shape[1],
                train_images=len(train), test_images=len(test), targets_per_image=length,
                eval_targets=int(nll.size), updates=opt.steps, train_nll=[first, second],
                predictor_weight_delta_after_one=delta, test_prefix_bpd=bpd,
                replay_max_abs_parameter_error=max(diffs), generated=generated)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mnist", type=Path, required=True)
    p.add_argument("--cifar10", type=Path, required=True)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    env = configure(args.device)
    rows = []
    for dataset, path in (("mnist", args.mnist), ("cifar10", args.cifar10)):
        train, test = check_data(dataset, path)
        for kind in ("full", "causal-linear"):
            rows.append(check(dataset, kind, train, test))
    write_result(args.out, dict(environment=env, seed=1229, cases=rows,
                                radam_8_step_max_abs=check_optimizer(), mol_max_abs=check_loss()))
