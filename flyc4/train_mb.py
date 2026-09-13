"""The mushroom body learns tic-tac-toe by dopamine — the real insect algorithm.

Self-play where every move is valued by the MaleCNS mushroom body: KC codes of
(board, candidate move) pairs through real PN->KC synapses, values as
approach-minus-avoid MBON drive through real KC->MBON synapses, eps-greedy
choice, and terminal reward-prediction error carried back as dopamine that
depresses the active KC->MBON synapses of every move in the game (Handler 2019,
Bennett 2021). No gradients anywhere.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .mushroom import MushroomQ, extract_mb_circuit

WIN_LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]


def features_from_cells(cells: list[int]) -> np.ndarray:
    """99-dim retinal features: own marks -> R1-R6 patch slots, opponent -> R8."""
    f = np.zeros(99, dtype=np.float32)
    for k in range(9):
        if cells[k] == 1:
            f[k*6:k*6+6] = 1.0
        elif cells[k] == 2:
            f[54+k*5:54+k*5+5] = 1.0
    return f


def winner(cells: list[int]):
    for a, b, c in WIN_LINES:
        if cells[a] and cells[a] == cells[b] == cells[c]:
            return cells[a]
    return 0 if all(cells) else None


def legal(cells):
    return [k for k in range(9) if cells[k] == 0]


class MBSelfPlay:
    """Both sides of every game are played and learned by one shared MB."""

    def __init__(self, mb: MushroomQ, eps: float = 0.1):
        self.mb = mb
        self.eps = eps

    def choose(self, cells, player):
        feats = torch.as_tensor(np.stack([
            features_from_cells(cells[:k] + [player] + cells[k+1:]) for k in legal(cells)
        ]), device=self.mb.device)
        codes = torch.eye(9, device=self.mb.device)[torch.tensor(legal(cells), device=self.mb.device)]
        kc = self.mb.kc_code(feats, codes)
        vals = self.mb.value(kc)
        if np.random.random() < self.eps:
            pick = int(np.random.randint(0, len(legal(cells))))
        else:
            pick = int(torch.argmax(vals))
        return legal(cells)[pick], kc[pick], float(vals[pick])

    def play_game(self):
        cells = [0]*9
        p = 1
        decisions = []      # (player, kc_row, value) per move
        while True:
            k, kc, v = self.choose(cells, p)
            decisions.append((p, kc, v))
            cells[k] = p
            w = winner(cells)
            if w or all(cells):
                z = (1 if w == 1 else -1 if w == 2 else 0)
                return cells, decisions, z
            p = 3 - p

    def learn(self, cells, decisions, z):
        mb = self.mb
        rpes, kcs = [], []
        for (p, kc_row, v) in decisions:
            z_mover = z if p == 1 else -z
            rpes.append(z_mover - v)
            kcs.append(kc_row)
        kc = torch.as_tensor(np.stack(kcs), device=mb.device)
        rpe = torch.as_tensor(np.array(rpes, dtype=np.float32), device=mb.device)
        return mb.dopamine_update(kc, rpe)


def evaluate_random(mb, n=200, eps=0.05):
    """Win/draw/loss of the MB (playing eps-greedy, both colors) vs random."""
    rng = np.random.default_rng(123)
    w = d = l = 0
    chooser = MBSelfPlay(mb, eps=eps)
    for _ in range(n):
        cells = [0]*9
        p = 1
        mb_side = 1 if rng.random() < 0.5 else 2
        while True:
            legal_cells = legal(cells)
            if not legal_cells:
                d += 1; break
            if p == mb_side:
                k, _, _ = chooser.choose(cells, p)
            else:
                k = int(rng.integers(0, len(legal_cells)))
            cells[k] = p
            win = winner(cells)
            if win or all(cells):
                if win == mb_side: w += 1
                elif win: l += 1
                else: d += 1
                break
            p = 3 - p
    return {"win": w/n, "draw": d/n, "loss": l/n, "n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=1_000_000)
    ap.add_argument("--eps-start", type=float, default=0.2)
    ap.add_argument("--eps-end", type=float, default=0.02)
    ap.add_argument("--eval-every", type=int, default=25_000)
    ap.add_argument("--out", type=Path, default=Path("/library/datasets/flyc4/live"))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--kc-active", type=int, default=200)
    args = ap.parse_args()
    np.random.seed(args.seed)

    circuit = extract_mb_circuit()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mb = MushroomQ(circuit, n_features=99, device=dev, seed=args.seed,
                   kc_active=args.kc_active)
    rng = np.random.default_rng(args.seed)
    cal_feats = torch.as_tensor(np.stack([
        features_from_cells([(rng.integers(0, 3) if rng.random() < 0.6 else 0)
                             for _ in range(9)]) for _ in range(128)
    ]), device=dev)
    cal = mb.calibrate(cal_feats)
    print(json.dumps({"evt": "calibrated", **cal}), flush=True)

    sp = MBSelfPlay(mb, eps=args.eps_start)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    csv = out / "mb_training.csv"
    if not csv.exists():
        csv.write_text("games,eps,mean_rpe,dw,win,draw,loss,vs_random_win,sat_floor,sat_ceil\n")
    t0 = time.time()
    games = 0
    while games < args.games:
        wins = draws = losses = 0
        dw_sum = rpe_sum = 0.0
        n = min(5000, args.games - games)
        for _ in range(n):
            cells, decisions, z = sp.play_game()
            dw = sp.learn(cells, decisions, z)
            dw_sum += dw; rpe_sum += abs(z - decisions[0][2])
            wins += z == 1; draws += z == 0; losses += z == -1
        games += n
        sp.eps = args.eps_end + (args.eps_start - args.eps_end) * max(0.0, 1 - games/ (args.games*0.5))
        ev = evaluate_random(mb, n=200, eps=0.02)
        floor, ceil = mb.saturation()
        with csv.open("a") as fh:
            fh.write(f"{games},{sp.eps:.3f},{rpe_sum/n:.3f},{dw_sum/n:.4f},"
                     f"{ev['win']:.3f},{ev['draw']:.3f},{ev['loss']:.3f},"
                     f"{ev['win']:.3f},{floor:.3f},{ceil:.3f}\n")
        np.savez(out / "mb_weights.npz", **mb.state_dict_np())
        print(json.dumps({"evt": "progress", "games": games,
                          "selfplay_wdl": [round(wins/n,3), round(draws/n,3), round(losses/n,3)],
                          "vs_random": {k: round(v,3) for k,v in ev.items()},
                          "ent_like_eps": round(sp.eps,3), "dw": round(dw_sum/n,4),
                          "elapsed_s": round(time.time()-t0)}), flush=True)
    print(json.dumps({"evt": "done", "games": games}), flush=True)


if __name__ == "__main__":
    main()
