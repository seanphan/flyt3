"""Opponents and batched evaluation: random and depth-2 minimax."""
from __future__ import annotations

import torch

from .env import NCOLS, NROWS, apply, init, legal_mask
from .policy import Readout

COL_W = (3., 2., 1., 0., 1., 2., 3.)


def mm_move(mine: int, opp: int, hts: list[int], depth: int = 2) -> int:
    """Negamax column choice for one game (python, eval-time only)."""

    def drop(m: int, h: list[int], c: int):
        bit = 1 << (c * 7 + h[c])
        m2 = m | bit
        won = any(m2 & (m2 >> s) & (m2 >> (2 * s)) & (m2 >> (3 * s)) for s in (1, 7, 6, 8))
        return m2, won

    def score(m: int) -> float:
        return sum(COL_W[c] for c in range(NCOLS) for r in range(NROWS) if m >> (c * 7 + r) & 1)

    def negmax(m: int, o: int, h: list[int], d: int) -> float:
        legal = [c for c in range(NCOLS) if h[c] < NROWS]
        if not legal:
            return 0.0
        best = -1e9
        for c in legal:
            m2, won = drop(m, h, c)
            if won:
                return 1000.0
            if d == 0:
                v = score(m2) - score(o)
            else:
                h2 = h[:]
                h2[c] += 1
                v = -negmax(o, m2, h2, d - 1)
            best = max(best, v)
        return best

    legal = [c for c in range(NCOLS) if hts[c] < NROWS]
    best_c, best_v = legal[0], -1e18
    for c in legal:
        m2, won = drop(mine, hts, c)
        if won:
            return c
        h2 = hts[:]
        h2[c] += 1
        v = -negmax(opp, m2, h2, depth - 1)
        if v > best_v:
            best_c, best_v = c, v
    return best_c


@torch.no_grad()
def evaluate(sim, pol: Readout, opponent: str, n_games: int, steps: int, device,
             tau: float = 0.1) -> dict:
    """Batched games; fly acts near-greedily; opponent 'random' or 'minimax2'."""
    st = init(n_games, device)
    agent_is = torch.rand(n_games, device=device) < 0.5  # fly is side 0 in half
    z = torch.zeros(n_games, device=device)
    while not bool(st["done"].all()):
        ply_even = (st["ply"] % 2) == 0
        fm = (~st["done"]) & (agent_is == ply_even)
        if fm.any():
            idx = fm.nonzero().squeeze(1)
            drive = sim.drive_from_board(st["mine"][idx], st["opp"][idx], int(fm.sum()))
            out = sim.run(drive, steps)
            rates = out["dn_rates"].t()
            legal = legal_mask(st)[idx]
            cols, _, _ = pol.act(rates, legal, tau=tau)
            apply(st, cols, idx)
            fin = idx[st["done"][idx]]
            z[fin] = (st["res"][fin] == 1).to(z.dtype)
        om = (~st["done"]) & (agent_is != ply_even)
        if om.any():
            idx = om.nonzero().squeeze(1)
            legal = legal_mask(st)[idx]
            if opponent == "random":
                probs = legal.float() / legal.float().sum(1, keepdim=True)
                cols = torch.multinomial(probs, 1).squeeze(1)
            else:
                cols = torch.tensor(
                    [mm_move(int(st["mine"][g]), int(st["opp"][g]),
                             st["hts"][g].tolist(), depth=2) for g in idx.tolist()],
                    dtype=torch.int64, device=device)
            apply(st, cols, idx)
            fin = idx[st["done"][idx]]
            z[fin] = -(st["res"][fin] == 1).to(z.dtype)
    return {
        "win": float((z > 0).float().mean()),
        "draw": float((z == 0).float().mean()),
        "loss": float((z < 0).float().mean()),
        "n": n_games,
    }
