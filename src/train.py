"""Train and evaluate a Jittor autoregressive pixel model. Run from the repo root."""
import argparse
import json
import math
from pathlib import Path

import jittor as jt
import numpy as np

from .data import encode, read_images, shifted
from .model import Generator, mol_nll
from .runtime import configure, write_result
from .training import RAdam


def pixel_nll(model, tokens, targets):
    x, y = shifted(tokens, targets)
    return mol_nll(model(jt.array(x.astype("int32"))), jt.array(y.astype("int32")))


def epoch_pass(model, data, order, batch, length, optimizer=None, seed=0):
    total, count = 0., 0
    for start in range(0, len(order), batch):
        rows = data[order[start:start + batch]]
        if optimizer is not None:
            # Key stochastic layers to the step so epoch-boundary checkpoints can replay.
            jt.set_global_seed(seed + optimizer.steps)
        nll = pixel_nll(model, rows, length)
        value = float(nll.mean().item())
        if not math.isfinite(value):
            raise FloatingPointError("Nonfinite pixel loss")
        size = len(rows) * length
        total += value * size
        count += size
        if optimizer is not None:
            optimizer.step(nll.mean())
    return {"images": len(order), "targets": count, "bpd": total / (count * math.log(2))}


def main(args):
    env = configure(args.device, args.seed)
    train, test = [encode(read_images(args.dataset, args.data, split, limit)) for split, limit
                   in (("train", args.train_images), ("test", args.test_images))]
    length = args.targets or train.shape[1] - 1
    if not 1 <= length < train.shape[1]:
        raise ValueError("Prefix length exceeds dataset")
    config = dict(attention_type=args.attention, sequence_length=train.shape[1] - 1,
                  n_layers=8 if args.dataset == "mnist" else 16)
    if args.tiny:
        config.update(d_model=32, n_heads=2, d_query=16, n_layers=1,
                      dropout=0., attention_dropout=0.)
    model = Generator(**config)
    opt = RAdam(model, lr=args.lr)
    spec = dict(config=config, dataset=args.dataset, train_images=len(train), test_images=len(test),
                batch=args.batch, length=length, seed=args.seed, lr=args.lr, device=args.device)
    ckpt = args.run_dir / "last.npz"
    history = []
    if args.resume:
        with np.load(ckpt, allow_pickle=False) as pack:
            meta = json.loads(str(pack["meta"]))
        if meta["spec"] != spec:
            raise ValueError("Resume configuration differs from checkpoint")
        opt.restore(ckpt)
        history = meta["history"]
    elif ckpt.exists():
        raise FileExistsError(f"Use --resume or choose another run directory: {ckpt}")
    if len(history) > args.epochs:
        raise ValueError("Target epoch precedes checkpoint")
    for epoch in range(len(history), args.epochs):
        model.train()
        order = np.random.default_rng(args.seed + epoch).permutation(len(train))
        trained = epoch_pass(model, train, order, args.batch, length, opt, args.seed)
        model.eval()
        with jt.no_grad():
            evaluated = epoch_pass(model, test, np.arange(len(test)), args.batch, length)
        model.train()
        history.append(dict(epoch=epoch + 1, steps=opt.steps, train=trained, test=evaluated))
        opt.save(ckpt, {"spec": spec, "history": history})
        print(json.dumps(history[-1]), flush=True)
    write_result(args.run_dir / "metrics.json", dict(environment=env, **spec, history=history))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["mnist", "cifar10"], required=True)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--attention", choices=["full", "causal-linear"], required=True)
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--epochs", type=int, default=1, help="Total target epochs including resumed epochs")
    p.add_argument("--targets", type=int, help="Number of predicted tokens per image (default: all)")
    p.add_argument("--train-images", type=int)
    p.add_argument("--test-images", type=int)
    p.add_argument("--seed", type=int, default=1229)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--tiny", action="store_true", help="One layer, width 32, dropout off")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if min(args.batch, args.epochs) < 1 or not (0 < args.lr < float("inf")) or args.targets == 0:
        p.error("Batch, epochs, learning rate and targets must be positive")
    main(args)
