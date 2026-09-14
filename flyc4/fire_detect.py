"""The fly brain detects forest fire from aerial/satellite imagery.

Pattern after jerryjliu/fly_ocr (MIT): the MaleCNS v1.0 connectome runs frozen
as a feature extractor — each image drives the retina (grayscale -> R1-R6
receptor rows, red/thermal channel -> R8 rows), the fixed 25.6M-synapse wiring
integrates 48 ticks, and spike counts from 1,024 downstream neurons become the
features of a tiny trained decoder. The connectome is never modified.

Dataset: FLAME aerial fire imagery (Shamsoshoara et al. 2021) via
alarmod/forest_fire (GPL-3): 3 balanced classes from filename
(image_<id>_<class>_<crop>.jpg — fire / smoke / non-fire).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .connectome import load_graph
from .sim import FlySim

IMAGES_DIR = Path("/library/datasets/fire_sat/flame")


def class_of(path: Path) -> int:
    """image_<id>_<class>_<crop>.jpg -> class in {0,1,2}."""
    return int(path.stem.split("_")[2])


def load_image(path: Path) -> np.ndarray:
    """99-dim retinal sample: 54 brightness patches -> R1-R6 rows,
    45 red-dominance patches -> R8 rows. The fly's two-channel eye."""
    img = Image.open(path).convert("RGB")
    a = np.asarray(img, dtype=np.float32) / 255.0
    def pool(ch, rows, cols):
        im = Image.fromarray((ch * 255).astype(np.uint8)).resize((cols * 8, rows * 8))
        b = np.asarray(im, dtype=np.float32) / 255.0
        bh, bw = b.shape[0] // rows, b.shape[1] // cols
        return b[: rows * bh, : cols * bw].reshape(rows, bh, cols, bw).mean(axis=(1, 3))
    bright = pool(np.asarray(img.convert("L"), dtype=np.float32) / 255.0, 6, 9)   # (6,9)
    red = np.clip(a[:, :, 0] - (a[:, :, 1] + a[:, :, 2]) / 2, 0, 1)
    redd = pool(red, 5, 9)                                                        # (5,9)
    return np.concatenate([bright.reshape(-1), redd.reshape(-1)]).astype(np.float32)  # 99


def build_drive(sim: FlySim, feats: np.ndarray) -> torch.Tensor:
    """(B, 198) -> [N, B] drive: first 99 rows grayscale, next 99 red-dominance."""
    B = feats.shape[0]
    d = torch.zeros(sim.N, B, device=sim.device, dtype=sim.dtype)
    srows = torch.from_numpy(np.concatenate([sim.own_rows.flatten().cpu().numpy(),
                                             sim.opp_rows.flatten().cpu().numpy()])).to(sim.device)
    vals = torch.as_tensor(feats, dtype=sim.dtype, device=sim.device)
    d[srows] = vals.T * 2.4
    return d


@torch.no_grad()
def pick_downstream(sim: FlySim, drive_sample: torch.Tensor, steps: int, n: int = 1024):
    """The n busiest non-sensory neurons by spike count on a sample of drives."""
    tot = torch.zeros(sim.N, device=sim.device)
    all_idx = torch.arange(sim.N, device=sim.device)
    for i in range(0, drive_sample.shape[1], 128):
        out = sim.run(drive_sample[:, i:i + 128], steps, count_idx=all_idx)
        tot += out["counts"].sum(1)
    tot[sim.own_rows.flatten()] = 0
    tot[sim.opp_rows.flatten()] = 0
    return torch.topk(tot, n).indices


@torch.no_grad()
def run_features(sim: FlySim, drive: torch.Tensor, steps: int = 48,
                 downstream: torch.Tensor | None = None, chunk: int = 256):
    if downstream is None:
        raise RuntimeError("precompute downstream with pick_downstream first")
    """log1p spike counts of `downstream` neurons for every drive column."""
    counts = torch.zeros(downstream.numel(), drive.shape[1], device=sim.device)
    for i in range(0, drive.shape[1], chunk):
        out = sim.run(drive[:, i:i + chunk], steps, count_idx=downstream)
        counts[:, i:i + chunk] = out["counts"]
    return torch.log1p(counts.T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, default=IMAGES_DIR / "train/train/images")
    ap.add_argument("--val-images", type=Path, default=IMAGES_DIR / "val/val/images")
    ap.add_argument("--limit", type=int, default=3000)
    ap.add_argument("--val-limit", type=int, default=1200)
    ap.add_argument("--steps", type=int, default=48)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--out", type=Path, default=Path("/library/datasets/flyc4/live/fire"))
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    meta = load_graph()
    sim = FlySim(meta, dev)
    print(json.dumps({"evt": "circuit", "device": dev, "neurons": meta["n_nodes"],
                      "edges": meta["n_edges"]}), flush=True)

    train_files = sorted(args.images.glob("*.jpg"))[: args.limit]
    val_files = sorted(args.val_images.glob("*.jpg"))[: args.val_limit]
    print(json.dumps({"evt": "data", "train": len(train_files), "val": len(val_files)}), flush=True)

    def feats(files):
        arr = np.stack([load_image(p) for p in files])
        return torch.as_tensor(arr, device=dev)

    t0 = time.time()
    ftr = feats(train_files)
    dtr = build_drive(sim, ftr)
    downstream = pick_downstream(sim, dtr[:, :512], args.steps, 1024)
    Xtr = run_features(sim, dtr, args.steps, downstream=downstream)
    fva = feats(val_files)
    Xva = run_features(sim, build_drive(sim, fva), args.steps, downstream=downstream)
    ytr = torch.as_tensor([class_of(p) for p in train_files], device=dev)
    yva = torch.as_tensor([class_of(p) for p in val_files], device=dev)
    print(json.dumps({"evt": "features", "seconds": round(time.time() - t0, 1),
                      "shape": list(Xtr.shape)}), flush=True)

    # tiny decoder: 64-unit MLP (fly_ocr letters-v2 style)
    torch.manual_seed(0)
    mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-6)
    Xtr_n = (Xtr - mu) / sd
    Xva_n = (Xva - mu) / sd
    dec = torch.nn.Sequential(torch.nn.Linear(Xtr.shape[1], 64), torch.nn.ReLU(),
                              torch.nn.Linear(64, 3)).to(dev)
    w_class = torch.tensor([len(ytr[ytr == c]) for c in range(3)], dtype=torch.float32, device=dev)
    w_class = w_class.sum() / w_class
    opt = torch.optim.Adam(dec.parameters(), lr=3e-3)
    for ep in range(args.epochs):
        perm = torch.randperm(Xtr.shape[0], device=dev)
        for i in range(0, Xtr.shape[0], 256):
            idx = perm[i:i + 256]
            loss = torch.nn.functional.cross_entropy(dec(Xtr_n[idx]), ytr[idx], weight=w_class)
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pred = dec(Xva_n).argmax(1)
        acc = float((pred == yva).float().mean())
        cm = torch.zeros(3, 3, dtype=torch.int64)
        for p, y in zip(pred.tolist(), yva.tolist()):
            cm[y, p] += 1
    report = {
        "task": "forest fire / smoke / non-fire classification from aerial imagery",
        "circuit": "MaleCNS v1.0 frozen (166,700 neurons, 25.6M synapses); "
                   "48-tick response; 1,024 downstream-neuron spike features",
        "decoder": "64-unit MLP, 3 classes",
        "val_accuracy": round(acc, 4),
        "confusion_rows_true": cm.tolist(),
        "train_images": len(train_files), "val_images": len(val_files),
        "dataset": "FLAME aerial fire imagery (Shamsoshoara et al. 2021) via alarmod/forest_fire, GPL-3",
        "recipe_after": "jerryjliu/fly_ocr (MIT): frozen circuit + tiny decoder",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez(args.out / "readout.npz", W1=dec[0].weight.detach().cpu().numpy(),
             b1=dec[0].bias.detach().cpu().numpy(), W2=dec[2].weight.detach().cpu().numpy(),
             b2=dec[2].bias.detach().cpu().numpy(),
             downstream=downstream.cpu().numpy())
    print(json.dumps({"evt": "done", **{k: report[k] for k in ("val_accuracy",)}}), flush=True)


if __name__ == "__main__":
    main()
