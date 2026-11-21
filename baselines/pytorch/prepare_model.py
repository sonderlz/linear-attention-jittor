"""Extract the pinned MNIST model from the authors' public Colab.

Pass a downloaded notebook with --notebook to prepare the model offline.
"""
import argparse
import hashlib
import json
import tempfile
import urllib.request
from pathlib import Path

from source_check import ROOT, sha256

URL = "https://drive.google.com/uc?export=download&id=1BV4OaWRAHYGeimO0cqHD86GfnfgUaKD4"
HEADER = """# Extracted verbatim from the official Colab linked at https://linear-transformers.com/
# Cells 6, 8, 10; only these module imports/header are added.
import torch
from fast_transformers.builders import TransformerEncoderBuilder
from fast_transformers.masking import TriangularCausalMask

"""


def save_verified(path, data, expected):
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"Unexpected SHA-256 for {path.name}. See the notebook link in README.md.")
    if path.exists():
        if sha256(path) != expected:
            raise FileExistsError(f"Existing {path} differs from the pinned version.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(data)
    Path(handle.name).replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path)
    args = parser.parse_args()
    lock = json.loads((ROOT / "AUTHOR_SOURCES.json").read_text(encoding="utf-8"))
    target = ROOT / "upstream/author_mnist.ipynb"
    if args.notebook or target.exists():
        raw = (args.notebook or target).read_bytes()
    else:
        with urllib.request.urlopen(URL, timeout=90) as response:
            raw = response.read()
    save_verified(target, raw, lock["notebook_sha256"])
    cells = json.loads(raw)["cells"]
    body = "\n\n".join("".join(cells[i]["source"]) for i in lock["extracted_cells"])
    # The module used by the A30 runs has CRLF bytes on all platforms.
    module = (HEADER + body + "\n").replace("\n", "\r\n").encode("utf-8")
    save_verified(ROOT / "author_model.py", module, lock["extracted_module_sha256"])
    print("Verified the original MNIST notebook and model.")


if __name__ == "__main__":
    main()
