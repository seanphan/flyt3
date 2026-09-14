"""The fly brain reads satellite tiles: post-wildfire damage classification.

Same recipe as the aerial fire fly (fire_detect.py, after jerryjliu/fly_ocr):
99-dim retinal sample -> frozen MaleCNS v1.0 circuit (48-tick LIF response) ->
log1p spike counts of a FIXED set of 1,024 downstream neurons (picked once
from the busiest non-sensory neurons on the first 512 training drives) ->
standardized 64-unit MLP decoder. The wiring is never trained.
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

from .connectome import load_graph
from .sim import FlySim


def retinal_sample(img: Image.Image) -> np.ndarray:
    """99-dim: 54 brightness patches -> R1-R6 rows, 45 red-dominance -> R8 rows."""
    a = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0

    def pool(ch, rows, cols):
        im = Image.fromarray((np.clip(ch, 0, 1) * 255).astype(np.uint8)).resize((cols * 8, rows * 8))
        b = np.asarray(im, dtype=np.float32) / 255.0
        bh, bw = b.shape[0] // rows, b.shape[1] // cols
        return b[: rows * bh, : cols * bw].reshape(rows, bh, cols, bw).mean(axis=(1, 3))

    bright = pool(np.asarray(img.convert("L"), dtype=np.float32) / 255.0, 6, 9)
    red = np.clip(a[:, :, 0] - (a[:, :, 1] + a[:, :, 2]) / 2, 0, 1)
    redd = pool(red, 5, 9)
    return np.concatenate([bright.reshape(-1), redd.reshape(-1)]).astype(np.float32)


def build_drive(sim: FlySim, feats: np.ndarray) -> torch.Tensor:
    own = torch.from_numpy(sim.own_rows.flatten().cpu().numpy()).to(sim.device)
    opp = torch.from_numpy(sim.opp_rows.flatten().cpu().numpy()).to(sim.device)
    srows = torch.cat([own, opp])
    d = torch.zeros(sim.N, feats.shape[0], device=sim.device, dtype=sim.dtype)
    d[srows] = torch.as_tensor(feats, dtype=sim.dtype, device=sim.device).T * 2.4
    return d


@torch.no_grad()
def pick_downstream(sim: FlySim, drive: torch.Tensor, steps: int, n: int = 1024):
    tot = torch.zeros(sim.N, device=sim.device)
    all_idx = torch.arange(sim.N, device=sim.device)
    for i in range(0, drive.shape[1], 128):
        out = sim.run(drive[:, i:i + 128], steps, count_idx=all_idx)
        tot += out["counts"].sum(1)
    tot[sim.own_rows.flatten()] = 0
    tot[sim.opp_rows.flatten()] = 0
    return torch.topk(tot, n).indices


@torch.no_grad()
def run_features(sim: FlySim, drive: torch.Tensor, steps: int,
                 downstream: torch.Tensor, chunk: int = 256):
    counts = torch.zeros(downstream.numel(), drive.shape[1], device=sim.device)
    for i in range(0, drive.shape[1], chunk):
        out = sim.run(drive[:, i:i + chunk], steps, count_idx=downstream)
        counts[:, i:i + chunk] = out["counts"]
    return torch.log1p(counts.T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-glob", default="/library/datasets/fire_sat/etkin/*.parquet")
    ap.add_argument("--steps", type=int, default=48)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--out", type=Path, default=Path("/library/datasets/flyc4/live/fire_satellite"))
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    meta = load_graph()
    sim = FlySim(meta, dev)
    print(json.dumps({"evt": "circuit", "device": dev, "neurons": meta["n_nodes"],
                      "edges": meta["n_edges"]}), flush=True)

    shards = sorted(glob.glob(args.data_glob))
    assert shards, f"no parquet shards at {args.data_glob}"
    feats_l, y_l = [], []
    t0 = time.time()
    for shard in shards:
        tbl = pq.read_table(shard, columns=["image", "label"])
        d = tbl.to_pandas()
        imgs = d["image"]
        for i in range(len(d)):
            raw = imgs[i]["bytes"] if isinstance(imgs[i], dict) else imgs[i]
            feats_l.append(retinal_sample(Image.open(io.BytesIO(raw)).convert("RGB")))
            y_l.append(int(d["label"][i]))
        print(json.dumps({"evt": "shard", "file": Path(shard).name,
                          "rows": len(d), "total": len(feats_l)}), flush=True)
    X = torch.as_tensor(np.stack(feats_l), device=dev)
    y = torch.as_tensor(np.array(y_l), device=dev)
    print(json.dumps({"evt": "features", "n": X.shape[0], "dim": X.shape[1],
                      "classes": int(y.max() + 1),
                      "seconds": round(time.time() - t0, 1)}), flush=True)

    # stratified 85/15 split on the retinal samples, before the circuit pass
    g_cpu = torch.Generator().manual_seed(0)
    Xtr_l, Xva_l, ytr_l, yva_l = [], [], [], []
    for c in range(5):
        idx = (y == c).nonzero().squeeze(1)
        idx = idx[torch.randperm(idx.numel(), generator=g_cpu).to(dev)]
        k = max(1, int(idx.numel() * 0.85))
        Xtr_l.append(X[idx[:k]]); ytr_l.append(y[idx[:k]])
        Xva_l.append(X[idx[k:]]); yva_l.append(y[idx[k:]])
    ftr, ytr = torch.cat(Xtr_l), torch.cat(ytr_l)
    fva, yva = torch.cat(Xva_l), torch.cat(yva_l)

    # frozen circuit: fixed downstream set from the first 512 training drives,
    # then one 48-tick response per tile
    t1 = time.time()
    downstream = pick_downstream(sim, build_drive(sim, ftr[:512].cpu().numpy()),
                                 args.steps, 1024)
    Xtr = run_features(sim, build_drive(sim, ftr.cpu().numpy()), args.steps,
                       downstream=downstream)
    Xva = run_features(sim, build_drive(sim, fva.cpu().numpy()), args.steps,
                       downstream=downstream)
    print(json.dumps({"evt": "circuit_features", "train": list(Xtr.shape),
                      "val": list(Xva.shape),
                      "seconds": round(time.time() - t1, 1)}), flush=True)

    torch.manual_seed(0)
    mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-6)
    Xtr_n, Xva_n = (Xtr - mu) / sd, (Xva - mu) / sd
    dec = torch.nn.Sequential(torch.nn.Linear(Xtr.shape[1], 64), torch.nn.ReLU(),
                              torch.nn.Linear(64, 5)).to(dev)
    w_class = torch.tensor([int((ytr == c).sum()) for c in range(5)],
                           dtype=torch.float32, device=dev)
    w_class = w_class.sum() / w_class.clamp_min(1)
    opt = torch.optim.Adam(dec.parameters(), lr=3e-3)
    for ep in range(args.epochs):
        perm = torch.randperm(Xtr.shape[0], device=dev)
        for i in range(0, Xtr.shape[0], 256):
            idx = perm[i:i + 256]
            loss = torch.nn.functional.cross_entropy(dec(Xtr_n[idx]), ytr[idx],
                                                     weight=w_class)
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pred = dec(Xva_n).argmax(1)
        acc = float((pred == yva).float().mean())
        cm = torch.zeros(5, 5, dtype=torch.int64)
        for p_, y_ in zip(pred.tolist(), yva.tolist()):
            cm[y_, p_] += 1
    report = {
        "task": "post-wildfire damage classification of satellite tiles (5 classes) "
                "through the frozen fly connectome",
        "circuit": "MaleCNS v1.0 frozen; 48-tick response; log1p spike counts of a "
                   "fixed 1,024 downstream-neuron set (picked once from the busiest "
                   "non-sensory neurons on the first 512 training drives)",
        "decoder": "standardized 64-unit MLP, 5 classes, class-weighted CE",
        "val_accuracy": round(acc, 4),
        "confusion_rows_true_cols_pred": cm.tolist(),
        "val_images": int(Xva.shape[0] // 5 * 5),
        "pixel_baseline": {"val_accuracy": 0.8573,
                           "note": "an earlier artifact under this name was a 99-dim "
                                   "pooled-pixel MLP that never ran the circuit; kept "
                                   "as pixel_baseline.npz for comparison"},
        "dataset": "Etkin et al. satellite structure-damage tiles via "
                   "kevincluo/structure_wildfire_damage_classification",
        "recipe_after": "jerryjliu/fly_ocr (MIT): frozen circuit + tiny decoder",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez(args.out / "readout.npz",
             W1=dec[0].weight.detach().cpu().numpy(), b1=dec[0].bias.detach().cpu().numpy(),
             W2=dec[2].weight.detach().cpu().numpy(), b2=dec[2].bias.detach().cpu().numpy(),
             mu=mu.cpu().numpy(), sd=sd.cpu().numpy(),
             downstream=downstream.cpu().numpy().astype(np.int64))
    print(json.dumps({"evt": "done", "val_accuracy": report["val_accuracy"],
                      "confusion": report["confusion_rows_true_cols_pred"]}), flush=True)


if __name__ == "__main__":
    main()
