# FLYT3 — wrapped. Fire-fly ticket handoff.

## What exists (all live, all pushed to seanphan/flyt3)

- **Tic-tac-toe vs the fly** — playable at http://192.168.2.36:30180
  (NodePort on k3s-w6; web pod `flyc4/fly-web` on w5, CPU-mode ~2s/move).
  Two completed 1M-game brains:
  - REINFORCE readout: 1,000,448 games, 9h (report.json)
  - Mushroom-body dopamine readout: 1,000,000 games, ~5h (mb_weights.npz,
    mb_training.csv) — final eval 74% win vs random
  - **Satellite Etkin: 60.8% 5-class through the circuit**
    (live/fire_satellite/report.json; the earlier 85.7% was a pooled-pixel
    MLP that never ran the circuit — kept as pixel_baseline.npz)
  - Fire watch mode is live in the web (mode switch / `?mode=fire`):
    tile in → 5-class verdict + connectome wave; lesion buttons degrade it
    live. 20 labeled sample tiles: live/fire_satellite/samples/.
- **Stack**: MaleCNS v1.0 LIF sim (166,700 neurons, 25.6M synapses, signed
  GABA, homeostatic thresholds), region silencing (lesion buttons), BFS
  hop-depth wave, three.js board + visible neon fruit fly + connectome cloud,
  animal-name leaderboard, `?flat=1` 2D fallback.

## Artifacts (on /library)

`/library/datasets/flyc4/live/` — readout.npz, ckpt.pt, mb_weights.npz,
mb_training.csv, metrics.jsonl, report.json, leaderboard.json, fire/,
fire_satellite/.
`/library/datasets/fire_sat/flame/` — FLAME aerial images (GPL-3, images only).
`/library/datasets/fire_sat/etkin/` — 9 Etkin satellite parquet shards
(5.08 GB, apache-2.0).

## Fresh ticket: the fire fly

1. ~~Serve the fire fly in the web~~ — done (fire watch mode).
2. **Satellite data v2** — the Bruhtian HF set (34.5k frames) turned out
   unlabeled; find a labeled satellite fire set or weak-label from FIRMS
   thermal points, then retrain the satellite decoder beyond 60.8%.
   (Pixel baseline sits at 85.7%, so there is headroom in features too.)
3. **MB dopamine trainer for the fire task** — the tic-tac-toe MB learner
   (flyc4/mushroom.py + train_mb.py, dopamine RPE, no gradients) adapts to
   (tile, class) pairs; race it against the supervised decoder.
4. **Lesion demos** — wired: silence a region, re-run a tile. Scripted
   exhibit still possible.

## Gotchas
- /library web mount must be RW (leaderboard writes).
- Deploy cycle: build → `docker save | gzip` → chimera → w6 ctr import
  (VM215 /tmp is tight; gz is ~2.5 GB).
- Headless QA: use `?flat=1` (this box has no WebGL); real browsers get 3D.
- k3s containerd on w5 AND w6 both hold `flyc4:latest` — import to BOTH
  after every image change or the pod lands on stale image.
