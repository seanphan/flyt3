"""Extract a small labeled sample-tile set from the Etkin shards for the web.

Writes `samples/` under the fire_satellite live dir: one JPEG per tile plus
manifest.json ({id, file, true_label}) and the class-name list, so the serve
pod can show and classify real dataset tiles without the 5 GB parquet mount.
Deterministic (seed 0); shards are scanned in order until every class has its
quota, so the common classes come from shard 0 and rare ones from later ones.
"""
from __future__ import annotations

import argparse
import glob
import io
import json
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

CLASSES = ["No damage", "Affected (1-9%)", "Minor (10-25%)", "Major (26-50%)",
           "Destroyed (>50%)"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-glob", default="/library/datasets/fire_sat/etkin/*.parquet")
    ap.add_argument("--per-class", type=int, default=4)
    ap.add_argument("--out", type=Path,
                    default=Path("/library/datasets/flyc4/live/fire_satellite/samples"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import random
    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    need = {c: args.per_class for c in range(len(CLASSES))}
    manifest = []
    for shard in sorted(glob.glob(args.data_glob)):
        if not any(need.values()):
            break
        tbl = pq.read_table(shard, columns=["image", "label"]).to_pandas()
        # deterministic per-shard row order before the quota scan
        rows = list(range(len(tbl)))
        rng.shuffle(rows)
        for i in rows:
            y = int(tbl["label"][i])
            if need.get(y, 0) <= 0:
                continue
            raw = tbl["image"][i]
            raw = raw["bytes"] if isinstance(raw, dict) else raw
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            tid = f"c{y}_{len([m for m in manifest if m['true_label'] == y]):03d}"
            img.save(args.out / f"{tid}.jpg", quality=88)
            manifest.append({"id": tid, "file": f"{tid}.jpg", "true_label": y})
            need[y] -= 1
        print(json.dumps({"evt": "shard", "file": Path(shard).name,
                          "have": len(manifest), "still_need": need}), flush=True)
    missing = {CLASSES[c]: n for c, n in need.items() if n > 0}
    assert not missing, f"not enough tiles for: {missing}"
    (args.out / "manifest.json").write_text(json.dumps(
        {"classes": CLASSES, "tiles": manifest}, indent=1) + "\n")
    print(json.dumps({"evt": "done", "tiles": len(manifest), "out": str(args.out)}),
          flush=True)


if __name__ == "__main__":
    main()
