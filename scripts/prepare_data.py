"""Download CIFAR-10 from the HuggingFace mirror (uoft-cs/cifar10) and write
data/cifar_train.npz and data/cifar_test.npz  (x: N,32,32,3 uint8 ; y: N int64).

The dataset is NOT stored in the git repository; run this once after cloning:
    python scripts/prepare_data.py
"""
import argparse
import io
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from PIL import Image

REPO = "uoft-cs/cifar10"
FILES = {
    "train": ["plain_text/train-00000-of-00001.parquet"],
    "test": ["plain_text/test-00000-of-00001.parquet"],
}


def to_arrays(parquet_path):
    df = pd.read_parquet(parquet_path)
    img_col = "img" if "img" in df.columns else "image"
    x = np.stack([np.array(Image.open(io.BytesIO(d["bytes"])).convert("RGB")) for d in df[img_col]])
    y = df["label"].to_numpy().astype(np.int64)
    return x.astype(np.uint8), y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "data"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for split, files in FILES.items():
        xs, ys = [], []
        for f in files:
            p = hf_hub_download(REPO, f, repo_type="dataset")
            x, y = to_arrays(p)
            xs.append(x)
            ys.append(y)
        x, y = np.concatenate(xs), np.concatenate(ys)
        np.savez(out / f"cifar_{split}.npz", x=x, y=y)
        print(f"{split}: x{x.shape} y{y.shape} classes={np.bincount(y).tolist()} -> {out / f'cifar_{split}.npz'}")


if __name__ == "__main__":
    main()
