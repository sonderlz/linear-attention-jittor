"""Figure-1-style causal attention forward/backward benchmark, using author ops."""
import argparse
import gc
import statistics
import time
from pathlib import Path
import torch
from experiment import configure, environment, sync, verify_sources, write_json
from source_check import sha256
from fast_transformers.attention import CausalLinearAttention, FullAttention
from fast_transformers.masking import LengthMask, TriangularCausalMask


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--attention", choices=["causal-linear", "full"], required=True)
    p.add_argument("--length", type=int, required=True)
    p.add_argument("--batch", type=int, required=True)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--memory-fraction", type=float, default=0.8)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    if Path(args.out).exists():
        raise FileExistsError("Use a new output file")
    if min(args.length, args.batch, args.warmup, args.repeats) < 1 or not 0 < args.memory_fraction <= 1:
        raise ValueError("Invalid benchmark dimensions")
    device = configure("cuda", 1229, 2)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, torch.cuda.current_device())
    result = {"arguments": vars(args), "environment": environment(device), "sources": verify_sources(),
              "script_sha256": sha256(__file__), "status": "running",
              "protocol": {"heads": 8, "head_dim": 32, "dtype": "float32", "mask": "causal",
                           "includes": "attention forward and Q/K/V gradients; inputs, output-gradient and masks included in peak allocated memory",
                           "excludes": "QKV projections, FFN, optimizer, input initialization, correctness checks",
                           "batch_rule": "max(1, 8192 // sequence_length), same for both methods",
                           "comparison": "Figure-1-style causal variant; A30 and specified dimensions/batches, not an exact hardware replication"}}
    trials = []
    try:
        shape = (args.batch, args.length, 8, 32)
        q, k, v = [torch.randn(shape, device=device, requires_grad=True) for _ in range(3)]
        gradient = torch.randn(shape, device=device)
        lengths = LengthMask(torch.full((args.batch,), args.length, device=device, dtype=torch.long), max_len=args.length)
        mask = TriangularCausalMask(args.length, device=device)
        net = CausalLinearAttention(32) if args.attention == "causal-linear" else FullAttention(attention_dropout=0.0)
        net = net.to(device).eval()
        def step():
            y = net(q, k, v, mask, lengths, lengths)
            y.backward(gradient)
            return y
        for _ in range(args.warmup):
            q.grad = k.grad = v.grad = None
            output = step()
            sync(device)
            del output
        for repeat in range(args.repeats):
            q.grad = k.grad = v.grad = None
            sync(device)
            torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            output = step()
            sync(device)
            seconds = time.perf_counter() - start
            trials.append({"repeat": repeat, "seconds": seconds, "ms_per_sequence": seconds * 1000 / args.batch,
                           "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                           "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)})
            if repeat == 0:
                if not torch.isfinite(output).all().item() or not all(torch.isfinite(x.grad).all().item() for x in (q, k, v)):
                    raise RuntimeError("Nonfinite outputs or gradients")
            del output
        result.update(status="ok", trials=trials,
                      median_ms_per_sequence=statistics.median(t["ms_per_sequence"] for t in trials),
                      peak_allocated_bytes=max(t["peak_allocated_bytes"] for t in trials),
                      peak_reserved_bytes=max(t["peak_reserved_bytes"] for t in trials),
                      outputs_and_gradients_finite=True)
    except torch.OutOfMemoryError as error:
        result.update(status="oom", trials=trials, error=str(error),
                      oom_interpretation=f"Does not fit this process's declared {args.memory_fraction:.0%} A30 allocator budget; not a claim about all possible implementations")
    write_json(args.out, result)
    print({k: v for k, v in result.items() if k in ("status", "median_ms_per_sequence", "peak_allocated_bytes", "error")}, flush=True)


if __name__ == "__main__":
    main()
