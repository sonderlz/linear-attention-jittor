"""Team experiment harness. The model and attention implementations are upstream."""
import json
import math
import platform
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from source_check import ROOT, UPSTREAM, sha256, verify_sources

# Force the pinned author library. No local replacement kernels.
sys.path.insert(0, str(UPSTREAM))
try:
    import fast_transformers
    from fast_transformers.causal_product import CausalDotProduct
    from author_model import Generator, RecurrentGenerator, sample_mol, predict_with_recurrent
except ImportError as error:
    raise ImportError("Build the original author extensions first: see the build commands in baselines/pytorch/README.md. " + str(error)) from error
if not Path(fast_transformers.__file__).resolve().is_relative_to(UPSTREAM.resolve()):
    raise RuntimeError("A different fast_transformers installation was imported")

DEFAULT_CONFIG = {"d_model": 256, "sequence_length": 783, "mixtures": 10,
                  "n_layers": 8, "n_heads": 8, "d_query": 32,
                  "dropout": 0.1, "attention_dropout": 0.1}


def configure(device, seed=1229, threads=None):
    if threads:
        torch.set_num_threads(threads)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    result = torch.device(device)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but PyTorch cannot use it")
    if CausalDotProduct.dot.get(result.type) is None:
        raise RuntimeError("Author causal extension was not built for " + result.type)
    return result


def build(config, recurrent=False):
    args = dict(config)
    if recurrent and args["attention_type"] == "causal-linear":
        args["attention_type"] = "linear"
    return (RecurrentGenerator if recurrent else Generator)(**args)


def mol_nll(logits, targets):
    """Discretized logistic bin mass matched to the author's sample_mol.

    Added by the team: the public Colab supplies sampling, not a training loss.
    Uses the SAME unbounded means, ELU(raw_scale)+1.0001, and [-1,1] bins.
    """
    mixes = logits.shape[-1] // 3
    weights, means, raw_scales = logits.split(mixes, dim=-1)
    scales = F.elu(raw_scales) + 1.0001
    centers = (targets.to(logits.dtype) / 127.5 - 1).unsqueeze(-1)
    upper = (centers + 1.0 / 255 - means) / scales
    lower = (centers - 1.0 / 255 - means) / scales
    # Stable log(sigmoid(upper)-sigmoid(lower)); no small-mass floor.
    interior = F.logsigmoid(upper) + F.logsigmoid(-lower) + torch.log(-torch.expm1(-2.0 / (255 * scales)))
    log_mass = torch.where(targets.unsqueeze(-1) == 0, F.logsigmoid(upper),
                          torch.where(targets.unsqueeze(-1) == 255, F.logsigmoid(-lower), interior))
    return -torch.logsumexp(F.log_softmax(weights, dim=-1) + log_mass, dim=-1)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def environment(device):
    result = {"python": platform.python_version(), "torch": str(torch.__version__),
              "platform": platform.platform(), "cuda_runtime": torch.version.cuda,
              "device": str(device), "threads": torch.get_num_threads(), "dtype": "float32", "tf32": False,
              "causal_backend": "author compiled " + device.type}
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(device)
        result.update(gpu=p.name, gpu_total_bytes=p.total_memory, capability=list(torch.cuda.get_device_capability(device)))
    return result


def write_json(path, data):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(target)


def load_checkpoint(path, attention=None):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, dict) and payload.get("format") == "author-mnist-v1":
        config = payload["config"]
        if attention and attention != config["attention_type"]:
            raise ValueError("--attention conflicts with checkpoint")
        return payload["state_dict"], config, payload
    if isinstance(payload, dict) and "value_embedding.weight" in payload and "pos_embedding.pe" in payload:
        if attention is None:
            raise ValueError("Author raw weights require --attention full|causal-linear")
        from fetch_author_weights import WEIGHTS
        name = "linear_author.pt" if attention == "causal-linear" else "full_author.pt"
        if sha256(path) != WEIGHTS[name][1]:
            raise ValueError("Raw checkpoint does not match the published author weights for " + attention)
        config = dict(DEFAULT_CONFIG, attention_type=attention)
        return payload, config, {"format": "author_raw_weights", "epoch": None}
    raise ValueError("Unsupported checkpoint format for the author MNIST model.")


def load_model(path, attention, device, recurrent=False):
    state, config, info = load_checkpoint(path, attention)
    net = build(config, recurrent=recurrent).to(device)
    net.load_state_dict(state, strict=True)
    return net.eval(), config, info


def load_mnist(path):
    from verify_mnist import verify
    identity = verify(path)
    arrays = []
    for prefix, count in (("train", 60000), ("t10k", 10000)):
        raw = (Path(path) / (prefix + "-images-idx3-ubyte")).read_bytes()
        arrays.append(torch.from_numpy(np.frombuffer(raw, dtype=np.uint8, offset=16).reshape(count, 784).copy()))
    return *arrays, identity


@torch.no_grad()
def evaluate(net, data, device, batch_size, max_batches=0):
    net.eval()
    total, n = 0.0, 0
    for step, start in enumerate(range(0, len(data), batch_size)):
        if max_batches and step >= max_batches:
            break
        pixels = data[start:start + batch_size].to(device).long()
        total += float(mol_nll(net(pixels[:, :-1]), pixels[:, 1:]).sum())
        n += len(pixels)
    return {"images": n, "complete_test_set": n == len(data), "predicted_pixels": n * 783,
            "bpd_predicted_783": total / (n * 783 * math.log(2)),
            "bpd_image_784_first_pixel_conditioned": total / (n * 784 * math.log(2)),
            "definition": "p(x[1:784] | x[0]); separate 783-target and 784-image normalizations"}


@torch.no_grad()
def generate(net, batch, device, mode, sequence_length=783):
    net.eval()
    if mode == "recurrent":
        # This is the author's helper, unmodified; includes the first fixed zero.
        return predict_with_recurrent(net, torch.zeros(batch, sequence_length, device=device, dtype=torch.long), 1).squeeze(-1)
    # Minimal correction outside the preserved notebook: recompute the final
    # prediction too. The author's demo predict() reused the previous logits.
    output = torch.zeros(batch, sequence_length + 1, device=device, dtype=torch.long)
    for position in range(1, sequence_length + 1):
        output[:, position:position + 1] = sample_mol(net(output[:, :position])[:, -1], 256)
    return output


def save_grid(pixels, path):
    from PIL import Image
    rows = pixels.detach().cpu().numpy().astype(np.uint8)
    side = math.isqrt(rows.shape[1])
    if side * side != rows.shape[1]:
        raise ValueError("Image grid requires square sequences")
    cols = min(8, len(rows))
    image = Image.new("L", (cols * side, math.ceil(len(rows) / cols) * side))
    for i, row in enumerate(rows):
        image.paste(Image.fromarray(row.reshape(side, side)), ((i % cols) * side, (i // cols) * side))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def checkpoint_meta(path, config, info):
    return {"path": str(Path(path).resolve()), "sha256": sha256(path), "format": info["format"],
            "epoch": info.get("epoch"), "config": config, "training": info.get("training"),
            "data": info.get("data")}
