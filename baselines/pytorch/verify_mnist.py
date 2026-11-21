"""Canonical MNIST identity, IDX validation, download and a labelled preview."""
import argparse
import gzip
import hashlib
import json
import struct
import urllib.request
from pathlib import Path

MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
SOURCE = "https://github.com/pytorch/vision/blob/main/torchvision/datasets/mnist.py"
# Raw SHA256 derived from downloads that passed torchvision's gzip MD5 checks.
FILES = {
    "train-images-idx3-ubyte": ("f68b3c2dcbeaaa9fbdd348bbdeb94873", "ba891046e6505d7aadcbbe25680a0738ad16aec93bde7f9b65e87a2fc25776db", (60000, 28, 28)),
    "train-labels-idx1-ubyte": ("d53e105ee54ea40749a09fcbcd1e9432", "65a50cbbf4e906d70832878ad85ccda5333a97f0f4c3dd2ef09a8a9eef7101c5", (60000,)),
    "t10k-images-idx3-ubyte": ("9fb629c4189551a2d022fa330f9573f3", "0fa7898d509279e482958e8ce81c8e77db3f2f8254e26661ceb7762c4d494ce7", (10000, 28, 28)),
    "t10k-labels-idx1-ubyte": ("ec29112dd5afa0611ce80d1b7f02629c", "ff7bcfd416de33731a308c3f266cc351222c34898ecbeaf847f06e48f7ec33f2", (10000,)),
}


def check_idx(blob, expected_shape):
    dims = len(expected_shape)
    size = 4 + 4 * dims
    if len(blob) < size:
        raise ValueError("Truncated IDX header")
    magic, *shape = struct.unpack(">" + "I" * (1 + dims), blob[:size])
    if magic != 0x800 + dims or tuple(shape) != tuple(expected_shape):
        raise ValueError("Invalid IDX magic or shape: %s %s" % (magic, shape))
    count = 1
    for n in shape:
        count *= n
    if len(blob) != size + count:
        raise ValueError("IDX byte count mismatch")
    if dims == 1 and max(blob[size:]) > 9:
        raise ValueError("Labels are outside 0..9")


def verify(data_dir):
    root = Path(data_dir)
    records = {}
    for name, (gz_md5, raw_sha, shape) in FILES.items():
        blob = (root / name).read_bytes()
        check_idx(blob, shape)
        actual = hashlib.sha256(blob).hexdigest()
        if actual != raw_sha:
            raise ValueError("%s is NOT canonical MNIST (SHA256 mismatch); use a fresh verified directory" % name)
        records[name] = {"sha256": actual, "bytes": len(blob), "shape": list(shape)}
    return {"dataset": "MNIST", "verified": True, "files": records,
            "data_dir": str(root.resolve()), "checksum_source": SOURCE}


def download(data_dir):
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    for name, (gz_md5, raw_sha, shape) in FILES.items():
        compressed = root / (name + ".gz")
        raw = root / name
        if raw.exists() and hashlib.sha256(raw.read_bytes()).hexdigest() != raw_sha:
            raise ValueError("Refusing to overwrite non-MNIST file: %s. Choose a fresh directory." % raw)
        if not compressed.exists():
            print("Downloading", name + ".gz", flush=True)
            with urllib.request.urlopen(MIRROR + name + ".gz", timeout=60) as response:
                blob = response.read()
            if hashlib.md5(blob).hexdigest() != gz_md5:
                raise ValueError("Downloaded gzip checksum mismatch: " + name)
            compressed.write_bytes(blob)
        blob = compressed.read_bytes()
        if hashlib.md5(blob).hexdigest() != gz_md5:
            raise ValueError("Existing gzip checksum mismatch: " + str(compressed))
        expanded = gzip.decompress(blob)
        check_idx(expanded, shape)
        if hashlib.sha256(expanded).hexdigest() != raw_sha:
            raise ValueError("Decompressed checksum mismatch: " + name)
        if not raw.exists():
            raw.write_bytes(expanded)
    return verify(root)


def preview(data_dir, output):
    import random
    from PIL import Image, ImageDraw
    root = Path(data_dir)
    pixels = (root / "train-images-idx3-ubyte").read_bytes()[16:]
    labels = (root / "train-labels-idx1-ubyte").read_bytes()[8:]
    grid = Image.new("RGB", (8 * 64, 8 * 78), "white")
    draw = ImageDraw.Draw(grid)
    for cell, idx in enumerate(random.Random(1229).sample(range(60000), 64)):
        tile = Image.frombytes("L", (28, 28), pixels[idx * 784:(idx + 1) * 784])
        x, y = (cell % 8) * 64, (cell // 8) * 78
        grid.paste(tile.resize((56, 56)), (x + 4, y))
        draw.text((x + 4, y + 58), str(labels[idx]), fill="black")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    grid.save(output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["download", "verify"])
    p.add_argument("--data-dir", required=True, help="Directory containing the four uncompressed IDX files")
    p.add_argument("--out", help="Optional verification JSON path")
    p.add_argument("--preview", help="Optional labelled PNG path")
    args = p.parse_args()
    report = download(args.data_dir) if args.action == "download" else verify(args.data_dir)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    if args.preview:
        preview(args.data_dir, args.preview)
    print(text)


if __name__ == "__main__":
    main()
