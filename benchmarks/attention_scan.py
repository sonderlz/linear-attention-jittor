"""Time one synchronized forward + Q/K/V backward pass per trial."""
import argparse
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys
from tempfile import TemporaryDirectory
import time


def measure(args):
    import jittor as jt
    import numpy as np
    from src.attention import attention
    from src.runtime import configure, write_result

    env = configure(args.device)
    length = args.lengths[0]
    shape = (args.batch, length, args.heads, args.dim)
    rng = np.random.default_rng(1229)
    times = []
    for _ in range(1 + args.warmup + args.repeats):
        # Fresh leaves avoid reusing an already evaluated expression across trials.
        q, k, v, upstream = [jt.array(rng.standard_normal(shape, dtype=np.float32)) for _ in range(4)]
        jt.sync_all(True)
        start = time.perf_counter()
        out = attention(args.kind, q, k, v)
        grads = jt.grad((out * upstream).sum(), [q, k, v])
        jt.sync_all(True)
        times.append((time.perf_counter() - start) * 1000)
        if not all(np.isfinite(t.numpy()).all() for t in [out, *grads]):
            raise FloatingPointError("Nonfinite output or gradient")
        del q, k, v, upstream, out, grads
    measured = times[1 + args.warmup:]
    write_result(args.out, dict(kind=args.kind, length=length, status="ok", environment=env,
                                first_call_ms=times[0], trials_ms=measured,
                                median_ms_per_batch=statistics.median(measured),
                                gpu_peak_bytes=None))


def scan(args):
    from src.runtime import write_result

    rows = []
    with TemporaryDirectory() as tmp:
        out = Path(tmp) / "case.json"
        for length in args.lengths:
            for kind in ("full", "causal-linear"):
                cmd = [sys.executable, "-m", "benchmarks.attention_scan", "--kind", kind,
                       "--device", args.device, "--lengths", str(length), "--out", str(out)]
                for key in ("batch", "heads", "dim", "warmup", "repeats"):
                    cmd.extend(["--" + key, str(getattr(args, key))])
                run = subprocess.run(cmd, capture_output=True, text=True)
                if run.returncode == 0:
                    rows.append(json.loads(out.read_text()))
                    continue
                # Each length runs in a fresh process so CUDA allocation failure is isolated.
                msg = re.search(r"[^\n]*(?:CUDA out of memory|cudaErrorMemoryAllocation|"
                                r"Unable to alloc cuda device memory|CUDA error[^\n]*out of memory)[^\n]*",
                                run.stderr, re.I)
                if args.device == "cuda" and msg:
                    rows.append(dict(kind=kind, length=length, status="oom", error=msg[0].strip(),
                                     gpu_peak_bytes=None))
                else:
                    raise RuntimeError(f"Worker failed ({kind}, L={length})\n{run.stderr[-4000:]}")
    write_result(args.out, dict(batch=args.batch, heads=args.heads, dim=args.dim, seed=1229,
                                warmup_after_first=args.warmup, repeats=args.repeats, cases=rows))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--lengths", nargs="+", type=int, default=[16, 32, 64])
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--dim", type=int, default=8)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--kind", choices=["full", "causal-linear"], help=argparse.SUPPRESS)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    if min(*args.lengths, args.batch, args.heads, args.dim, args.repeats) < 1 or args.warmup < 0:
        p.error("Sizes and repeats must be positive, warmup must be nonnegative")
    if args.kind:
        measure(args)
    else:
        scan(args)
