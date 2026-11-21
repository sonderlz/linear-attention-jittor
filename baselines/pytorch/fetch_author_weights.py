"""Download the two pretrained checkpoints linked by the author's Colab."""
import argparse
from pathlib import Path
import urllib.request
from source_check import sha256

WEIGHTS = {
    "linear_author.pt": ("17fc94TzytTdAwNMVCE7qOg75-CWLGi_p", "a2e915d5200e4d51fc636210a9d9a5cfa8d069d3d9f0a4af5da3cb2ea83a1d1b"),
    "full_author.pt": ("1L47Ode6GxCMQbVMK33_ANjCu2iA4rf8l", "17c7944afb0b818a4375779274459eb5078cfaff5ab5b4dd2b5b40acd0a2c12e"),
}


def fetch(directory):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    for name, (file_id, expected) in WEIGHTS.items():
        target = root / name
        if not target.exists():
            temp = target.with_suffix(".download")
            with urllib.request.urlopen("https://drive.google.com/uc?export=download&id=" + file_id, timeout=90) as response:
                temp.write_bytes(response.read())
            if sha256(temp) != expected:
                raise ValueError("Download does not match author checkpoint (possibly an HTML quota page): " + str(temp))
            temp.replace(target)
        if sha256(target) != expected:
            raise ValueError("Existing checkpoint mismatch; not overwriting: " + str(target))
        print(name, "SHA256 verified", expected, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default="weights")
    fetch(p.parse_args().out_dir)
