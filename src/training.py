"""RAdam with named moments and a portable deterministic-step checkpoint."""
import json
import math
import os
import tempfile
from pathlib import Path

import jittor as jt
import numpy as np


class RAdam:
    """Default PyTorch RAdam equations, zero weight decay."""
    def __init__(self, model, lr=1e-3):
        self.model = model
        self.lr = lr
        self.steps = 0
        self.params = {k: p for k, p in model.state_dict().items() if not p.is_stop_grad()}
        self.first = {k: jt.zeros_like(p).stop_grad() for k, p in self.params.items()}
        self.second = {k: jt.zeros_like(p).stop_grad() for k, p in self.params.items()}

    def step(self, loss):
        grads = jt.grad(loss, list(self.params.values()))
        self.steps += 1
        t = self.steps
        b1, b2 = 0.9, 0.999
        rho_inf = 2 / (1 - b2) - 1
        rho = rho_inf - 2 * t * b2**t / (1 - b2**t)
        for (k, p), g in zip(self.params.items(), grads):
            m = b1 * self.first[k] + (1 - b1) * g
            v = b2 * self.second[k] + (1 - b2) * g.sqr()
            delta = m / (1 - b1**t)
            if rho > 5:
                rect = math.sqrt((rho-4)*(rho-2)*rho_inf / ((rho_inf-4)*(rho_inf-2)*rho))
                delta = delta * (rect * math.sqrt(1 - b2**t)) / (jt.sqrt(v) + 1e-8)
            self.first[k].update(m)
            self.second[k].update(v)
            p.update(p - self.lr * delta)
        jt.sync_all()

    def save(self, path, metadata=None):
        arrays = {"steps": self.steps, "lr": self.lr, "meta": json.dumps(metadata)}
        arrays.update({"weight/" + k: v for k, v in self.model.state_dict(to="numpy").items()})
        for group, moments in (("m/", self.first), ("v/", self.second)):
            arrays.update({group + k: v.numpy() for k, v in moments.items()})
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as f:
            temp = Path(f.name)
        try:
            np.savez(temp, **arrays)
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)

    def restore(self, path):
        with np.load(path, allow_pickle=False) as saved:
            self.model.import_weights({k[7:]: saved[k] for k in saved if k.startswith("weight/")})
            for group, moments in (("m/", self.first), ("v/", self.second)):
                for k, v in moments.items():
                    a = saved[group + k]
                    if tuple(v.shape) != a.shape:
                        raise ValueError(f"Invalid optimizer moment for {k}")
                    v.update(jt.array(a))
            self.steps, self.lr = int(saved["steps"]), float(saved["lr"])
