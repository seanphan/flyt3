"""Batched LIF simulation over the fixed MaleCNS v1.0 wiring.

Approximate neural dynamics (engineering choice, doomfly-style): synapses are
signed contact-count strengths (GABA edges negative, from the dataset's
consensus neurotransmitter predictions); state is a leaky integrate-and-fire
membrane per neuron under per-neuron homeostatic threshold control, which
keeps responses graded and prevents runaway recruitment. Board cells drive
retinal patches: own pieces -> 6-cell R1-R6 patches, opponent pieces ->
5-cell R8 patches. The wiring is never modified; nothing about Connect-4
is stored in it.
"""
from __future__ import annotations

import numpy as np
import torch

from .env import bits_to_cells

N_OWN = 42 * 6    # own-piece channel: R1-R6 patches
N_OPP = 42 * 5    # opponent channel: R8 patches


class FlySim:
    def __init__(self, graph: dict, device, decay: float = 0.85, gain: float = 0.15,
                 noise: float = 0.01, homeo_k: float = 12.0, target_rate: float = 0.03,
                 dtype=torch.float32):
        z = graph["arrays"]
        self.device = device
        self.N = graph["n_nodes"]
        n = self.N
        crow = torch.from_numpy(z["indptr"].astype(np.int32))
        col = torch.from_numpy(z["indices"].astype(np.int32))
        val = torch.from_numpy(z["values"].astype(np.float32))
        self.W = torch.sparse_csr_tensor(
            crow, col, val, size=(n, n), dtype=dtype, device=device)
        theta = gain * torch.sqrt(torch.from_numpy(z["theta_scale"].astype(np.float32)) + 1.0)
        self.theta = theta.to(device)
        sidx = z["sensory_idx"]
        self.own_rows = torch.from_numpy(sidx[:N_OWN].astype(np.int64)).view(42, 6).to(device)
        self.opp_rows = torch.from_numpy(sidx[N_OWN:].astype(np.int64)).view(42, 5).to(device)
        self.motor = torch.from_numpy(z["motor_idx"]).to(device)
        self.ppl = torch.from_numpy(z["ppl_idx"]).to(device)
        self.decay, self.noise, self.dtype = decay, noise, dtype
        self.homeo_k, self.target_rate = homeo_k, target_rate

    def drive_from_board(self, mine: torch.Tensor, opp: torch.Tensor,
                         batch: int, own_drive: float = 2.4, opp_drive: float = 1.2):
        """[N, B] drive: own cell -> patch of 6 R cells, opponent -> patch of 5 R8."""
        d = torch.zeros(self.N, batch, device=self.device, dtype=self.dtype)
        if batch == 0:
            return d
        cells_own = bits_to_cells(mine)      # [B, 42] bool
        cells_opp = bits_to_cells(opp)
        d[self.own_rows] += (own_drive / self.own_rows.shape[1]) \
            * cells_own.t().to(self.dtype).unsqueeze(1)
        d[self.opp_rows] += (opp_drive / self.opp_rows.shape[1]) \
            * cells_opp.t().to(self.dtype).unsqueeze(1)
        return d

    @torch.no_grad()
    def run(self, drive: torch.Tensor, steps: int, want_raster: bool = False):
        """Simulate `steps` ticks for a [N, B] drive; returns decision signals."""
        batch = drive.shape[1]
        v = torch.rand(self.N, batch, device=self.device, dtype=self.dtype) * 0.05
        s_prev = torch.zeros(self.N, batch, device=self.device, dtype=self.dtype)
        dn_counts = torch.zeros(self.motor.numel(), batch, device=self.device, dtype=self.dtype)
        total_spikes = torch.zeros(steps, device=self.device)
        raster = torch.zeros(self.motor.numel(), steps, device=self.device) if want_raster else None
        noise = self.noise * torch.randn(self.N, 1, device=self.device, dtype=self.dtype)
        rate = torch.full((self.N, 1), self.target_rate, device=self.device, dtype=self.dtype)
        theta_base = self.theta.unsqueeze(1)
        for t in range(steps):
            cur = torch.sparse.mm(self.W, s_prev)
            v = self.decay * v + cur + drive + noise
            s = (v > theta_base * torch.exp(self.homeo_k * (rate - self.target_rate))).to(self.dtype)
            v = v.masked_fill(s.bool(), 0.0)
            rate = 0.97 * rate + 0.03 * s.mean(1, keepdim=True)
            s_prev = s
            dn_counts += s[self.motor]
            total_spikes[t] = s.sum()
            if want_raster:
                raster[:, t] = s[self.motor].sum(1)
        return {
            "dn_rates": dn_counts / steps,
            "total_spikes": total_spikes,
            "motor_raster": raster,
        }

    @torch.no_grad()
    def punish(self, drive: torch.Tensor, steps: int = 32, amp: float = -3.0):
        """Aversive dopamine pulse into the PPL101 pair (doomfly-style)."""
        d = drive.clone()
        d[self.ppl] += amp
        return self.run(d, steps)
