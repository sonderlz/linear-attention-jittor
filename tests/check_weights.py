"""Import published MNIST weights and compare against the pinned author model."""
import argparse
from pathlib import Path
import sys

import jittor as jt
import numpy as np
import torch

from src.data import encode, read_images, shifted
from src.model import Generator, mol_nll
from src.runtime import configure, write_result


def check(kind, weights, batch, baseline):
    source = weights / ("full_author.pt" if kind == "full" else "linear_author.pt")
    state, config, _ = baseline.load_checkpoint(source, kind)
    original = baseline.build(config)
    original.load_state_dict(state, strict=True)
    original.eval()
    model = Generator(**config)
    model.import_weights({k: v.numpy() for k, v in state.items()})
    model.eval()
    x, y = shifted(batch)
    logits_error, loss_error, shared_error, torch_losses, jt_losses = [], [], [], [], []
    with torch.no_grad(), jt.no_grad():
        for inputs, targets in zip(x, y):
            ref = original(torch.tensor(inputs[None], dtype=torch.long))
            nll = baseline.mol_nll(ref, torch.tensor(targets[None])).numpy()
            pred = model(jt.array(inputs[None].astype("int32")))
            actual = pred.numpy()
            actual_nll = mol_nll(pred, jt.array(targets[None].astype("int32"))).numpy()
            shared_nll = mol_nll(jt.array(ref.numpy()), jt.array(targets[None].astype("int32"))).numpy()
            np.testing.assert_allclose(actual, ref.numpy(), atol=1e-3, rtol=1e-4)
            # Narrow logistic bins amplify logit differences. Test each stage and the image mean.
            np.testing.assert_allclose(shared_nll, nll, atol=2e-4, rtol=1e-4)
            np.testing.assert_allclose(actual_nll, nll, atol=1e-3, rtol=1e-4)
            np.testing.assert_allclose(actual_nll.mean(), nll.mean(), atol=1e-5, rtol=1e-5)
            logits_error.append(float(np.abs(actual - ref.numpy()).max()))
            loss_error.append(float(np.abs(actual_nll - nll).max()))
            shared_error.append(float(np.abs(shared_nll - nll).max()))
            torch_losses.extend(nll.ravel())
            jt_losses.extend(actual_nll.ravel())
        # Compare a short fresh prefix that includes ink as well as background.
        start = int(np.flatnonzero(x[0])[0])
        window = x[:1, start:start + 17]
        ref = original(torch.tensor(window, dtype=torch.long))
        states, errors = None, []
        for t in range(17):
            out, states = model.next_logits(jt.array(window[:, t].astype("int32")), t, states)
            one = out.numpy()
            np.testing.assert_allclose(one, ref[:, t].numpy(), atol=1e-3, rtol=1e-4)
            errors.append(float(np.abs(one - ref[:, t].numpy()).max()))
    return dict(kind=kind, sha256=baseline.sha256(source), state_entries=len(state),
                logit_max_abs=max(logits_error), recurrent_window_start=start,
                recurrent_logit_max_abs=max(errors), nll_max_abs=max(loss_error),
                nll_same_logits_max_abs=max(shared_error),
                torch_bpd=float(np.mean(torch_losses, dtype=np.float64) / np.log(2)),
                jittor_bpd=float(np.mean(jt_losses, dtype=np.float64) / np.log(2)))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mnist", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--baseline-dir", type=Path, default=Path("baselines/pytorch"))
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    env = configure(args.device)
    sys.path.insert(0, str(args.baseline_dir.resolve()))
    import experiment as baseline

    baseline.verify_sources()
    baseline.configure("cpu", threads=1)
    pixels = encode(read_images("mnist", args.mnist, "test", 4))
    rows = [check(kind, args.weights, pixels, baseline) for kind in ("full", "causal-linear")]
    write_result(args.out, dict(environment=env, torch=torch.__version__, images="MNIST test[0:4]",
                                targets_per_image=783, targets=4*783, recurrent_positions=17, cases=rows))
