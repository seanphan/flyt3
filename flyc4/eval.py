"""Opponents and batched evaluation: random and perfect (full-minimax)."""
from __future__ import annotations

import torch

from .env import LINES, NCELLS, apply, init, legal_mask
from .policy import Readout


def mm_move(mine: int, opp: int) -> int:
    """Perfect tic-tac-toe move for one position (depth-aware negamax)."""
    lines = [int(l) for l in LINES.tolist()]

    def won(m: int) -> bool:
        return any(m & l == l for l in lines)

    def negamax(m: int, o: int, free: list[int], depth: int) -> int:
        if won(o):
            return -10 + depth      # opponent won with their last move
        if not free:
            return 0                # draw
        best = -100
        for c in free:
            v = -negamax(o, m | (1 << c), [f for f in free if f != c], depth - 1)
            if v > best:
                best = v
        return best

    legal = [c for c in range(NCELLS) if not (mine | opp) >> c & 1]
    best_c, best_v = legal[0], -1000
    for c in legal:
        if won(mine | (1 << c)):
            return c
        v = -negamax(opp, mine | (1 << c), [f for f in legal if f != c], 9)
        if v > best_v:
            best_c, best_v = c, v
    return best_c


@torch.no_grad()
def evaluate(sim, pol: Readout, opponent: str, n_games: int, steps: int, device,
             tau: float = 0.1) -> dict:
    """Batched games; fly acts near-greedily; opponent 'random' or 'perfect'."""
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
                    [mm_move(int(st["mine"][g]), int(st["opp"][g])) for g in idx.tolist()],
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
