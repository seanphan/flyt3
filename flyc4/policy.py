"""The only trained component: a linear readout of descending-neuron activity.

9-way move policy + value baseline, REINFORCE with terminal reward. Wiring
stays fixed; these ~10k numbers are everything the fly learns.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Readout(nn.Module):
    def __init__(self, n_motor: int, nmoves: int = 9):
        super().__init__()
        self.nmoves = nmoves
        self.head = nn.Linear(n_motor, nmoves)   # motor rates -> move logits
        self.value = nn.Linear(n_motor, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        nn.init.normal_(self.value.weight, std=1e-3)

    def masked_logits(self, rates: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
        return self.head(rates).masked_fill(~legal, float("-inf"))

    @torch.no_grad()
    def act(self, rates: torch.Tensor, legal: torch.Tensor, tau: float = 1.0):
        probs = F.softmax(self.masked_logits(rates, legal) / tau, dim=-1)
        cols = torch.multinomial(probs, 1).squeeze(1)
        logp = torch.log(probs.gather(1, cols.unsqueeze(1)).squeeze(1) + 1e-12)
        ent = -(probs * torch.log(probs + 1e-12)).sum(-1)
        return cols, logp, ent

    def value_of(self, rates: torch.Tensor) -> torch.Tensor:
        return self.value(rates).squeeze(-1)
