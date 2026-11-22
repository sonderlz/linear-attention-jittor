"""Verified raw MNIST / CIFAR-10 to row-major image tokens."""
import hashlib
import pickle
from pathlib import Path

import numpy as np

from baselines.pytorch.verify_mnist import verify


# Per-file checksums from torchvision.datasets.CIFAR10 (Python archive).
CIFAR_MD5 = {
    "data_batch_1": "c99cafc152244af753f735de768cd75f",
    "data_batch_2": "d4bba439e000b95fd0a9bffe97cbabec",
    "data_batch_3": "54ebc095f3ab1f0389bbae665268c751",
    "data_batch_4": "634d18415352ddfa80567beed471001a",
    "data_batch_5": "482c414d41f54cd18b22e5b47cb7c3b",
    "test_batch": "40351d587109b95175f43aff81a1287e",
}


def encode(images):
    """Batch of gray [N,H,W] or interleaved RGB [N,H,W,3] uint8 images."""
    if images.dtype != np.uint8 or images.ndim not in (3, 4):
        raise ValueError("Expected batched uint8 images")
    if images.ndim == 4 and images.shape[-1] != 3:
        raise ValueError("RGB input must have channels last")
    return np.ascontiguousarray(images.reshape(len(images), -1))


def decode(tokens, image_shape):
    return tokens.reshape((len(tokens), *image_shape))


def shifted(tokens, targets=None):
    length = tokens.shape[1] - 1 if targets is None else targets
    if length < 1 or length >= tokens.shape[1]:
        raise ValueError("Need one more token than the number of predictions")
    return tokens[:, :length], tokens[:, 1:length + 1]


def read_images(dataset, directory, split, limit=None):
    if split not in ("train", "test") or (limit is not None and limit < 1):
        raise ValueError("Invalid split or image count")
    root = Path(directory)
    if dataset == "mnist":
        verify(root)
        prefix, count = ("train", 60000) if split == "train" else ("t10k", 10000)
        images = np.memmap(root / (prefix + "-images-idx3-ubyte"), mode="r",
                           dtype=np.uint8, offset=16, shape=(count, 28, 28))
        return images[:limit]
    if dataset != "cifar10":
        raise ValueError(dataset)
    names = list(CIFAR_MD5)[:-1] if split == "train" else ["test_batch"]
    images = []
    remaining = limit
    for name in names:
        blob = (root / name).read_bytes()
        if hashlib.md5(blob).hexdigest() != CIFAR_MD5[name]:
            raise ValueError(f"Invalid canonical CIFAR-10 batch: {name}")
        planar = pickle.loads(blob, encoding="bytes")[b"data"]
        if planar.dtype != np.uint8 or planar.shape != (10000, 3072):
            raise ValueError("Invalid CIFAR-10 image array")
        part = planar[:remaining].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        images.append(part)
        if remaining is not None:
            remaining -= len(part)
            if remaining == 0:
                break
    return np.concatenate(images)
