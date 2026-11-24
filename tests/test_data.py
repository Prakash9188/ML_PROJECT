"""Smoke tests for the data pipeline. Run:  python tests/test_data.py   (needs data/*.npz)"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rb import data as D  # noqa: E402


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = D.load(dev)
    assert d["xtr"].shape == (45000, 3, 32, 32) and d["xva"].shape == (5000, 3, 32, 32)
    assert d["xte"].shape == (10000, 3, 32, 32) and d["xtr"].dtype == torch.uint8
    assert set(d["ytr"].unique().tolist()) == set(range(10))
    x = D.augment(D.to_float(d["xtr"][:64]))
    assert x.shape == (64, 3, 32, 32) and 0 <= x.min() and x.max() <= 1
    # split is deterministic
    d2 = D.load(dev)
    assert torch.equal(d["yva"], d2["yva"])
    print("data tests OK")


if __name__ == "__main__":
    main()
