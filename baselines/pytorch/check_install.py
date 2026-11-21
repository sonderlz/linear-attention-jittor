"""Exercise the author's compiled forward/backward and recurrent model on target device."""
import argparse
import json
import torch
from pathlib import Path
from experiment import build, configure, environment, mol_nll, sync, verify_sources, write_json
from fast_transformers.causal_product import causal_dot_product


def check(device):
    device = configure(device, 1229, 2 if device == "cpu" else None)
    result = {"sources": verify_sources(), "environment": environment(device), "models": {}}
    # Check the compiled kernel's values AND gradients against explicit causal attention.
    # The reference is test-only; training always uses the author's compiled kernel.
    base = [torch.rand(2, 2, 17, 32, device=device) / 4 for _ in range(3)]
    def execute(fn):
        q, k, v = [x.clone().requires_grad_() for x in base]
        y = fn(q, k, v)
        return y.detach(), torch.autograd.grad(y.square().sum(), (q, k, v))
    def reference(q, k, v):
        scores = torch.einsum("nhid,nhjd->nhij", q, k)
        scores = scores * torch.ones(17, 17, device=device).tril()
        return torch.einsum("nhij,nhje->nhie", scores, v)
    actual, gradients = execute(causal_dot_product)
    expected, expected_gradients = execute(reference)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
    for a, b in zip(gradients, expected_gradients):
        torch.testing.assert_close(a, b, atol=2e-4, rtol=5e-4)
    result["compiled_kernel"] = {"forward_max_abs_error": float((actual - expected).abs().max()),
                                 "gradient_max_abs_errors": [float((a-b).abs().max()) for a,b in zip(gradients, expected_gradients)]}
    for kind in ("causal-linear", "full"):
        config = dict(d_model=32, sequence_length=7, mixtures=3, attention_type=kind,
                      n_layers=2, n_heads=4, d_query=8, dropout=0.0, attention_dropout=0.0)
        net = build(config).to(device).eval()
        recurrent = build(config, recurrent=True).to(device).eval()
        recurrent.load_state_dict(net.state_dict(), strict=True)
        pixels = torch.randint(0, 256, (2, 8), device=device)
        parallel = net(pixels[:, :-1])
        with torch.no_grad():
            memory, outputs = None, []
            for position in range(7):
                y, memory = recurrent(pixels[:, position], i=position, memory=memory)
                outputs.append(y)
            stepwise = torch.stack(outputs, dim=1)
        error = float((parallel - stepwise).abs().max())
        torch.testing.assert_close(parallel, stepwise, atol=2e-5, rtol=2e-5)
        loss = mol_nll(parallel, pixels[:, 1:]).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite loss")
        loss.backward()
        if not all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None):
            raise RuntimeError("Nonfinite gradients")
        sync(device)
        result["models"][kind] = {"parallel_recurrent_max_abs_error": error, "nll_finite": bool(torch.isfinite(loss)),
                                  "forward_backward_passed": True}
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--out")
    args = p.parse_args()
    if args.out and Path(args.out).exists():
        raise FileExistsError("Use a new --out")
    result = check(args.device)
    if args.out:
        write_json(args.out, result)
