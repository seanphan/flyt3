"""The only trained component: a linear readout of descending-neuron activity.

7-way column policy + value baseline, REINFORCE with terminal reward. Wiring
stays fixed; these ~2k numbers are everything the fly learns.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Readout(nn.Module):
    def __init__(self, n_motor: int, ncols: int = 7):
        super().__init__()
        self.cols = ncols
        self.head = nn.Linear(n_motor, ncols)   # motor rates -> column logits
        self.value = nn.Linear(n_motor, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        nn.init.normal_(self.value.weight, std=1e-3)

    def logits(self, rates: torch.Tensor) -> torch.Tensor:
        return self.head(rates)

    def masked_logits(self, rates: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
        z = self.head(rates)
        return z.masked_fill(~legal, float("-inf"))

    @torch.no_grad()
    def act(self, rates: torch.Tensor, legal: torch.Tensor, tau: float = 1.0):
        probs = F.softmax(self.masked_logits(rates, legal) / tau, dim=-1)
        cols = torch.multinomial(probs, 1).squeeze(1)
        logp = torch.log(probs.gather(1, cols.unsqueeze(1)).squeeze(1) + 1e-12)
        ent = -(probs * torch.log(probs + 1e-12)).sum(-1)
        return cols, logp, ent

    def value_of(self, rates: torch.Tensor) -> torch.Tensor:
        return self.value(rates).squeeze(-1)
