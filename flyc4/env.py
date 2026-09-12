"""Batched tic-tac-toe on int64 bitboards (9 cells, cell = r*3 + c).

Board tensor is a dict of torch tensors living on one device:
  mine [B] int64 bitmask for the side to move
  opp  [B] bitmask for the other side
  occ  [B] bitmask of all occupied cells
  done [B] bool, res [B] int8 (from the mover's perspective at game end)
  ply  [B] int16
The side to move is implicit: apply() flips mine/opp after each drop.
"""
from __future__ import annotations

import torch

NCELLS = 9
FULL = (1 << NCELLS) - 1

LINES = torch.tensor([
    0b000000111, 0b000111000, 0b111000000,   # rows
    0b100100100, 0b010010010, 0b001001001,   # columns
    0b100010001, 0b001010100,                # diagonals
], dtype=torch.int64)


def init(batch: int, device) -> dict:
    z = torch.zeros(batch, dtype=torch.int64, device=device)
    return {
        "mine": z.clone(), "opp": z.clone(), "occ": z.clone(),
        "done": torch.zeros(batch, dtype=torch.bool, device=device),
        "res": torch.zeros(batch, dtype=torch.int8, device=device),
        "ply": torch.zeros(batch, dtype=torch.int16, device=device),
    }


def legal_mask(st: dict) -> torch.Tensor:
    shifts = torch.arange(NCELLS, device=st["occ"].device, dtype=torch.int64)
    free = (~st["occ"].unsqueeze(1)) & FULL
    return ((free >> shifts) & 1).ne(0)
def apply(st: dict, cols: torch.Tensor, idx: torch.Tensor | None = None) -> dict:
    """Move for the side to move on the given games (all games if idx is None)."""
    dev = st["mine"].device
    b = idx if idx is not None else torch.arange(st["mine"].shape[0], device=dev)
    cols = cols.to(torch.int64)
    b = b.to(torch.int64)
    mine, opp = st["mine"][b], st["opp"][b]
    occ = st["occ"][b]
    act = ~st["done"][b]
    bit = (torch.ones_like(mine) << cols) * act
    m = mine | bit
    o = occ | bit
    won = torch.zeros_like(act)
    for L in LINES.to(dev):
        won |= (m & L) == L
    won &= act
    draw = act & ~won & (st["ply"][b] + 1 == NCELLS)
    st["mine"][b], st["opp"][b], st["occ"][b] = opp, m, o
    st["res"][b] = torch.where(won, torch.ones_like(st["res"][b]), st["res"][b])
    st["res"][b] = torch.where(draw, torch.zeros_like(st["res"][b]), st["res"][b])
    st["done"][b] = st["done"][b] | won | draw
    st["ply"][b] = st["ply"][b] + act.to(torch.int16)
    return st


def bits_to_cells(bm: torch.Tensor) -> torch.Tensor:
    """[B] bitboard -> [B, 9] occupancy, cell = r*3 + c."""
    shifts = torch.arange(NCELLS, device=bm.device, dtype=torch.int64)
    return ((bm.unsqueeze(1) >> shifts) & 1).ne(0)
