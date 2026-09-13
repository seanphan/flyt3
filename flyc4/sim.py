"""Batched LIF simulation over the fixed MaleCNS v1.0 wiring.

Approximate neural dynamics (engineering choice, doomfly-style): synapses are
signed contact-count strengths (GABA edges negative, from the dataset's
consensus neurotransmitter predictions); state is a leaky integrate-and-fire
membrane per neuron under per-neuron homeostatic threshold control. Board
cells drive retinal patches: own pieces -> 6-cell R1-R6 patches, opponent
pieces -> 5-cell R8 patches. Per-tick spike counts are tracked per brain
region (optic / central / VNC) so the web view can play the real wave, and a
region can be lesioned live (its spikes suppressed before propagation).
The wiring is never modified; nothing about tic-tac-toe is stored in it.
"""
from __future__ import annotations

import numpy as np
import torch

from .env import NCELLS, bits_to_cells

N_OWN = NCELLS * 6   # own-piece channel: R1-R6 patches
N_OPP = NCELLS * 5   # opponent channel: R8 patches


class FlySim:
    def __init__(self, graph: dict, device, decay: float = 0.85, gain: float = 0.15,
                 noise: float = 0.01, homeo_k: float = 12.0, target_rate: float = 0.03,
                 dtype=torch.float32):
        z = graph["arrays"]
        self.device = device
        self.dtype = dtype
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
        self.own_rows = torch.from_numpy(sidx[:N_OWN].astype(np.int64)).view(NCELLS, 6).to(device)
        self.opp_rows = torch.from_numpy(sidx[N_OWN:].astype(np.int64)).view(NCELLS, 5).to(device)
        self.motor = torch.from_numpy(z["motor_idx"]).to(device)
        self.ppl = torch.from_numpy(z["ppl_idx"]).to(device)
        region = torch.from_numpy(z["region"].astype(np.int64)).to(device)
        self.region = region
        self._region_masks = [(region == c).to(dtype) for c in (0, 1, 2)]
        self._alive = torch.ones(n, 1, device=device, dtype=dtype)
        self.decay, self.noise = decay, noise
        self.homeo_k, self.target_rate = homeo_k, target_rate
        self.silenced = None   # region code to lesion (0 optic, 1 central, 2 VNC), or None

    def set_silence(self, region):
        """Lesion a brain region live: its spikes are suppressed before propagation."""
        self.silenced = region
        if region is None:
            self._alive.fill_(1.0)
        else:
            self._alive = 1.0 - self._region_masks[region].unsqueeze(1)

    def drive_from_board(self, mine: torch.Tensor, opp: torch.Tensor,
                         batch: int, own_drive: float = 2.4, opp_drive: float = 1.2):
        """[N, B] drive: own cell -> patch of 6 R cells, opponent -> patch of 5 R8."""
        d = torch.zeros(self.N, batch, device=self.device, dtype=self.dtype)
        if batch == 0:
            return d
        cells_own = bits_to_cells(mine)      # [B, 9] bool
        cells_opp = bits_to_cells(opp)
        d[self.own_rows] += (own_drive / self.own_rows.shape[1]) \
            * cells_own.t().to(self.dtype).unsqueeze(1)
        d[self.opp_rows] += (opp_drive / self.opp_rows.shape[1]) \
            * cells_opp.t().to(self.dtype).unsqueeze(1)
        return d

    @torch.no_grad()
    def run(self, drive: torch.Tensor, steps: int, want_raster: bool = False,
            count_idx: torch.Tensor | None = None):
        """Simulate `steps` ticks for a [N, B] drive; returns decision signals."""
        batch = drive.shape[1]
        v = torch.rand(self.N, batch, device=self.device, dtype=self.dtype) * 0.05
        s_prev = torch.zeros(self.N, batch, device=self.device, dtype=self.dtype)
        dn_counts = torch.zeros(self.motor.numel(), batch, device=self.device, dtype=self.dtype)
        region_wave = torch.zeros(3, steps, device=self.device, dtype=self.dtype)
        total_spikes = torch.zeros(steps, device=self.device)
        counts = (torch.zeros(count_idx.numel(), batch, device=self.device, dtype=self.dtype)
                  if count_idx is not None else None)
        raster = torch.zeros(self.motor.numel(), steps, device=self.device) if want_raster else None
        noise = self.noise * torch.randn(self.N, 1, device=self.device, dtype=self.dtype)
        rate = torch.full((self.N, 1), self.target_rate, device=self.device, dtype=self.dtype)
        theta_base = self.theta.unsqueeze(1)
        alive = self._alive
        for t in range(steps):
            cur = torch.sparse.mm(self.W, s_prev)
            v = self.decay * v + cur + drive + noise
            s = (v > theta_base * torch.exp(self.homeo_k * (rate - self.target_rate))).to(self.dtype)
            if self.silenced is not None:
                s = s * alive
            v = v.masked_fill(s.bool(), 0.0)
            rate = 0.97 * rate + 0.03 * s.mean(1, keepdim=True)
            s_prev = s
            dn_counts += s[self.motor]
            for ri, rmask in enumerate(self._region_masks):
                region_wave[ri, t] = (s * rmask.unsqueeze(1)).sum()
            if counts is not None:
                counts += s[count_idx]
            total_spikes[t] = s.sum()
            if want_raster:
                raster[:, t] = s[self.motor].sum(1)
        return {
            "dn_rates": dn_counts / steps,
            "total_spikes": total_spikes,
            "motor_raster": raster,
            "counts": counts,
            "region_wave": region_wave,
        }

    @torch.no_grad()
    def punish(self, drive: torch.Tensor, steps: int = 32, amp: float = -3.0):
        """Aversive dopamine pulse into the PPL101 pair (doomfly-style)."""
        d = drive.clone()
        d[self.ppl] += amp
        return self.run(d, steps)
