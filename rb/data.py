"""Data loading and GPU-side augmentation.

Everything is kept as uint8 tensors on the device and converted to float in [0,1] per batch;
models take [0,1] images (normalisation lives inside the model) so corruptions/attacks work in pixel space.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load(dev, val_n=5000, seed=0):
    """Return dict of uint8 image tensors (N,3,32,32) and int64 labels on `dev`.

    The 50k CIFAR-10 training set is split once (fixed seed) into 45k train / `val_n` validation.
    Validation is used for model selection signals, temperature scaling, the abstention threshold and
    the adaptive scheduler; the official 10k test set is never touched during training.
    """
    tr, te = np.load(DATA_DIR / "cifar_train.npz"), np.load(DATA_DIR / "cifar_test.npz")
    perm = np.random.RandomState(seed).permutation(len(tr["y"]))
    vi, ti = perm[:val_n], perm[val_n:]
    f = lambda a: torch.from_numpy(a).permute(0, 3, 1, 2).contiguous().to(dev)
    g = lambda a: torch.from_numpy(a).to(dev)
    return dict(xtr=f(tr["x"][ti]), ytr=g(tr["y"][ti]), xva=f(tr["x"][vi]), yva=g(tr["y"][vi]),
                xte=f(te["x"]), yte=g(te["y"]))


def to_float(xu8):
    return xu8.float().div_(255)


def augment(x):
    """Random crop (reflect-pad 4) + horizontal flip, per sample, entirely on the GPU."""
    B, C, H, W = x.shape
    p = F.pad(x, (4, 4, 4, 4), mode="reflect")
    i = torch.randint(0, 9, (B,), device=x.device)
    j = torch.randint(0, 9, (B,), device=x.device)
    ar = torch.arange(H, device=x.device)
    iy = (i[:, None] + ar)[:, :, None]
    ix = (j[:, None] + ar)[:, None, :]
    out = p[torch.arange(B, device=x.device)[:, None, None], :, iy, ix].permute(0, 3, 1, 2)
    flip = torch.rand(B, device=x.device) < 0.5
    out = torch.where(flip.view(B, 1, 1, 1), out.flip(3), out)
    return out.contiguous()
