"""Verify the exact author sources without importing PyTorch or compiled code."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT / "upstream" / "fast-transformers"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_sources():
    lock = json.loads((ROOT / "AUTHOR_SOURCES.json").read_text(encoding="utf-8"))
    for name, expected in lock["source_files"].items():
        if sha256(UPSTREAM / name) != expected:
            raise ValueError("Author source changed: " + name)
    if sha256(ROOT / "upstream/author_mnist.ipynb") != lock["notebook_sha256"]:
        raise ValueError("Original Colab changed")
    if sha256(ROOT / "author_model.py") != lock["extracted_module_sha256"]:
        raise ValueError("Extracted author model changed")
    return {"commit": lock["commit"], "verified_source_files": len(lock["source_files"]),
            "notebook_sha256": lock["notebook_sha256"], "core_changes": []}


if __name__ == "__main__":
    print(json.dumps(verify_sources(), indent=2))
