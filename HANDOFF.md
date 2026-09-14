# FLYT3 — wrapped. Fire-fly ticket handoff.

## What exists (all live, all pushed to seanphan/flyt3)

- **Tic-tac-toe vs the fly** — playable at http://192.168.2.36:30180
  (NodePort on k3s-w6; web pod `flyc4/fly-web` on w5, CPU-mode ~2s/move).
  Two completed 1M-game brains:
  - REINFORCE readout: 1,000,448 games, 9h (report.json)
  - Mushroom-body dopamine readout: 1,000,000 games, ~5h (mb_weights.npz,
    mb_training.csv) — final eval 74% win vs random
- **Fire flies** (frozen connectome + tiny decoder, fly_ocr recipe):
  - Aerial FLAME: 55.2% 3-class → live/fire/report.json
  - **Satellite Etkin: 85.7% 5-class** → live/fire_satellite/report.json
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

Suggested next steps, in order:
1. **Serve the fire fly in the web** — add a mode switch (fly tic-tac-toe is
   default; fire mode shows a satellite tile + the decoder's per-class scores
   + the connectome wave). Server code: reuse fire_satellite.run_features.
2. **Satellite data v2** — the Bruhtian HF set (34.5k frames) turned out
   unlabeled; find a labeled satellite fire set or weak-label from FIRMS
   thermal points, then retrain the satellite decoder beyond 85.7%.
3. **MB dopamine trainer for the fire task** — the tic-tac-toe MB learner
   (flyc4/mushroom.py + train_mb.py, dopamine RPE, no gradients) adapts to
   (tile, class) pairs; race it against the supervised decoder.
4. **Lesion demos** — lesion switches already work in the sim; wire them to
   the fire decoder for "the fly goes blind" exhibits.

## Gotchas
- /library web mount must be RW (leaderboard writes).
- Deploy cycle: build → `docker save | gzip` → chimera → w6 ctr import
  (VM215 /tmp is tight; gz is ~2.5 GB).
- Headless QA: use `?flat=1` (this box has no WebGL); real browsers get 3D.
- k3s containerd on w5 AND w6 both hold `flyc4:latest` — import to BOTH
  after every image change or the pod lands on stale image.
