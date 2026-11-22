"""Experiment setup and portable JSON output."""
import json
import os
from pathlib import Path
import platform

import jittor as jt


def configure(device, seed=1229):
    if device == "cuda" and not jt.has_cuda:
        raise RuntimeError("Jittor CUDA backend unavailable")
    jt.flags.use_cuda = int(device == "cuda")
    jt.set_global_seed(seed)
    cpu = platform.machine()
    if Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[-1].strip()
                break
    return {"device": device, "dtype": "float32", "jittor": jt.__version__,
            "python": platform.python_version(), "cpu": cpu, "os": platform.system(),
            "omp_threads": os.getenv("OMP_NUM_THREADS"), "use_mkl": os.getenv("use_mkl")}


def write_result(path, data):
    report = json.dumps(data, allow_nan=False, indent=2, ensure_ascii=False) + "\n"
    if path is None:
        print(report)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report, encoding="utf-8")
