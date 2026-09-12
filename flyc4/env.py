"""Batched Connect-4 on int64 bitboards (7 bits per column, bit 6 is a guard).

Board tensor is a dict of torch tensors living on one device:
  mine [B] int64 bitmasks for the side to move
  opp  [B] bitmask for the other side
  hts  [B, 7] int8 column heights
  done [B] bool, res [B] int8 (from the mover's perspective at game end)
  ply  [B] int16
The side to move is implicit: apply() flips mine/opp after each drop.
"""
from __future__ import annotations

import torch

NCOLS, NROWS = 7, 6
WIN_SHIFTS = (1, 7, 6, 8)  # vertical, horizontal, diag/, diag\


def init(batch: int, device) -> dict:
    z = torch.zeros(batch, dtype=torch.int64, device=device)
    return {
        "mine": z.clone(), "opp": z.clone(),
        "hts": torch.zeros(batch, NCOLS, dtype=torch.int8, device=device),
        "done": torch.zeros(batch, dtype=torch.bool, device=device),
        "res": torch.zeros(batch, dtype=torch.int8, device=device),
        "ply": torch.zeros(batch, dtype=torch.int16, device=device),
    }


def legal_mask(st: dict) -> torch.Tensor:
    return st["hts"] < NROWS


def _four(m: torch.Tensor, s: int) -> torch.Tensor:
    return m & (m >> s) & (m >> (2 * s)) & (m >> (3 * s))


def apply(st: dict, cols: torch.Tensor, idx: torch.Tensor | None = None) -> dict:
    """Drop for the side to move on the given games (all games if idx is None)."""
    dev = st["mine"].device
    b = idx if idx is not None else torch.arange(st["mine"].shape[0], device=dev)
    cols = cols.to(torch.int64)
    b = b.to(torch.int64)
    mine, opp = st["mine"][b], st["opp"][b]
    hts = st["hts"][b]
    act = ~st["done"][b]
    h = hts.gather(1, cols.unsqueeze(1)).squeeze(1).to(torch.int64)
    bit = torch.ones_like(mine) << (cols.to(torch.int64) * 7 + h)
    m = mine | (bit * act)
    won = torch.zeros_like(act)
    for s in WIN_SHIFTS:
        won |= _four(m, s).ne(0)
    won &= act
    draw = act & ~won & (st["ply"][b] + 1 == NCOLS * NROWS)
    newh = hts.clone()
    rows = torch.arange(b.shape[0], device=dev)
    newh[rows, cols] = torch.where(act, (h + 1).to(torch.int8), newh[rows, cols])
    st["hts"][b] = newh
    st["mine"][b], st["opp"][b] = opp, m
    st["res"][b] = torch.where(won, torch.ones_like(st["res"][b]), st["res"][b])
    st["res"][b] = torch.where(draw, torch.zeros_like(st["res"][b]), st["res"][b])
    st["done"][b] = st["done"][b] | won | draw
    st["ply"][b] = st["ply"][b] + act.to(torch.int16)
    return st



def bits_to_cells(bm: torch.Tensor) -> torch.Tensor:
    """[B] bitboard -> [B, 42] occupancy in column-major c*6+r cell order."""
    k = torch.arange(42, device=bm.device, dtype=torch.int64)
    shifts = (k // 6) * 7 + (k % 6)  # skip the guard bit of each column
    return ((bm.unsqueeze(1) >> shifts) & 1).ne(0)
