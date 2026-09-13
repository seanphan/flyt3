"""Mushroom body circuit from the MaleCNS v1.0 connectome + dopamine Q-learner.

Adapted for FLYC4's tic-tac-toe from adonis-singh/TMNF-C (MIT), python/tmnf_fly/
mushroom_body.py, which itself implements Handler et al. 2019 / Bennett et al.
2021 dopamine-gated plasticity.

Circuit extracted from the same MaleCNS feathers the rest of FLYC4 uses:
  ALPN -> KC synapse counts (the fixed expansion, never learned),
  KC -> MBON synapse counts (the only plastic synapses),
  MBON compartment group from DAN connectivity: PAM-innervated MBONs are the
  "avoid" group (positive RPE depresses them), PPL1-innervated MBONs the
  "approach" group (negative RPE depresses them) — Aso et al. 2014.

The learner, per candidate move (board, action k):
  PN features -> real PN->KC matrix -> top-k KC code (APL winner-take-all)
  -> KC->MBON drives -> value = approach drive - avoid drive.
Action selection: eps-greedy over the nine values.
Learning: dw = -eta * KC * signed_dopamine, bounded [0, w0]. No gradients.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

DATA = Path("/library/datasets/malecns_v1")
CACHE = DATA / "processed-flyc4"
AVOID, APPROACH = 0, 1


def extract_mb_circuit(cache_dir: Path = CACHE, data_dir: Path = DATA) -> dict:
    """Pull the PN/KC/MBON/DAN subgraph out of the MaleCNS feathers and save
    the two connectome matrices plus the MBON valence groups."""
    import pandas as pd
    import pyarrow.feather as feather

    F_ANN = "body-annotations-male-cns-v1.0-minconf-0.5.feather"
    F_EDGES = "connectome-weights-male-cns-v1.0-minconf-0.5.feather"

    cache_dir = Path(cache_dir)
    out_path = cache_dir / "mushroom_body.npz"
    if out_path.exists():
        return dict(np.load(out_path, allow_pickle=True))

    ann = feather.read_table(data_dir / F_ANN).to_pandas()
    edges = feather.read_table(data_dir / F_EDGES)
    types = ann.type.astype(str)

    def ids_of(pattern):
        return np.array(sorted(ann.bodyId[types.str.contains(pattern, regex=True, na=False)]),
                        dtype=np.uint64)

    # antennal lobe projection neurons: type names end in PN (DA1_lPN, VM5d_adPN, ...)
    kc_u = ids_of(r"KC")            # Kenyon cells: KCab-, KCg-s, KCc- ...
    mb_u = ids_of(r"^MBON")
    pam_u = ids_of(r"^PAM")
    ppl1_u = ids_of(r"^PPL1")
    others = np.concatenate([kc_u, mb_u, pam_u, ppl1_u])
    pn_u = np.setdiff1d(ids_of(r"PN"), others)

    pre = edges.column("body_pre").to_numpy(zero_copy_only=False)
    post = edges.column("body_post").to_numpy(zero_copy_only=False)
    w = edges.column("weight").to_numpy(zero_copy_only=False).astype(np.float64)

    def mat(pre_ids, post_ids):
        kp = np.searchsorted(pre_ids, pre)
        jp = np.searchsorted(post_ids, post)
        keep = (kp < len(pre_ids)) & (jp < len(post_ids))
        keep &= pre_ids[np.minimum(kp, len(pre_ids) - 1)] == pre
        keep &= post_ids[np.minimum(jp, len(post_ids) - 1)] == post
        M = np.zeros((len(pre_ids), len(post_ids)))
        np.add.at(M, (kp[keep], jp[keep]), w[keep])
        return M

    pn_kc = mat(pn_u, kc_u)
    kc_mbon = mat(kc_u, mb_u)

    # MBON valence from DAN connectivity: PAM input -> avoid group, PPL1 -> approach
    dan_u = np.concatenate([pam_u, ppl1_u])
    dan_mb = mat(dan_u, mb_u)
    is_pam = np.isin(dan_u, pam_u)
    pam_in = dan_mb[is_pam].sum(axis=0)
    ppl1_in = dan_mb[~is_pam].sum(axis=0)
    group = np.full(len(mb_u), -1, dtype=np.int64)
    group[(pam_in > 0) & (ppl1_in == 0)] = AVOID
    group[(ppl1_in > 0) & (pam_in == 0)] = APPROACH
    # mixed-input MBONs: resolve by compartment token in the instance name
    # (Aso et al. 2014: a/b lobes = PPL1 punishment, a'/g lobes = PAM reward)
    import re as _re
    instances = ann.set_index(ann.bodyId.astype(np.uint64)).instance.astype(str)
    PPL1_LOBES = ("a1", "a2", "a3", "b1", "b2")
    PAM_LOBES = ("a'1", "a'2", "a'3", "b'1", "b'2", "g1", "g2", "g3", "g4", "g5")
    for j in np.nonzero(group < 0)[0]:
        toks = set(_re.findall(r"a'?\d|b'?\d|g\d", instances.get(int(mb_u[j]), "")))
        if any(t in PPL1_LOBES for t in toks) and not any(t in PAM_LOBES for t in toks):
            group[j] = APPROACH
        elif any(t in PAM_LOBES for t in toks) and not any(t in PPL1_LOBES for t in toks):
            group[j] = AVOID

    keep_mbon = group >= 0
    kc_mbon = kc_mbon[:, keep_mbon]
    mb_u = mb_u[keep_mbon]
    group = group[keep_mbon]
    keep_kc = kc_mbon.sum(axis=1) > 0
    kc_mbon = kc_mbon[keep_kc]
    kc_u = kc_u[keep_kc]
    pn_kc = pn_kc[:, keep_kc]
    keep_pn = pn_kc.sum(axis=1) > 0
    pn_u = pn_u[keep_pn]
    pn_kc = pn_kc[keep_pn]

    np.savez_compressed(out_path, pn_u=pn_u, kc_u=kc_u, mb_u=mb_u,
                        pn_kc=pn_kc.astype(np.float32), kc_mbon=kc_mbon.astype(np.float32),
                        group=group.astype(np.int64))
    return dict(np.load(out_path, allow_pickle=True))


F_ANN_ = "body-annotations-male-cns-v1.0-minconf-0.5.feather"
F_EDGES_ = "connectome-weights-male-cns-v1.0-minconf-0.5.feather"


class MushroomQ:
    """Dopamine-gated Q learner over the real KC->MBON connectome.

    value(kc) = approach drive - avoid drive; the only learned change is
    dw = -eta * KC * signed_dopamine at existing KC->MBON synapses.
    """

    REWARD, PUNISHMENT = 0, 1   # PAM compartment, PPL1 compartment

    def __init__(self, circuit: dict, *, n_features: int, device="cpu", seed: int = 7,
                 kc_active: int = 200, pn_state_inputs: int = 8,
                 pn_action_fraction: float = 0.8, pn_quantile: float = 0.5,
                 gain: float = 40.0, alpha: float = 0.02):
        self.device = torch.device(device)
        self.gen = torch.Generator(device="cpu").manual_seed(seed)
        self.n_features = n_features
        self.kc_active = kc_active
        self.gain = gain
        self.alpha = alpha
        self.pn_state_inputs = pn_state_inputs
        self.pn_action_fraction = pn_action_fraction
        self.pn_quantile = pn_quantile

        pn_kc = torch.as_tensor(np.asarray(circuit["pn_kc"]), dtype=torch.float32, device=self.device)
        kc_mbon = torch.as_tensor(np.asarray(circuit["kc_mbon"]), dtype=torch.float32, device=self.device)
        group = torch.as_tensor(np.asarray(circuit["group"]), dtype=torch.int64, device=self.device)
        self.n_pn, self.n_kc = pn_kc.shape
        self.n_mbon = kc_mbon.shape[1]
        has_in = pn_kc.sum(dim=0) > 0
        self.pn_kc = pn_kc / pn_kc.sum(dim=0).clamp_min(1.0)   # weighted mean per KC
        self.kc_bias = torch.where(has_in, torch.zeros_like(has_in, dtype=torch.float32),
                                   torch.full_like(has_in, -float("inf")).to(torch.float32))
        self.avoid = group == self.REWARD       # PAM -> avoid group
        self.approach = group == self.PUNISHMENT
        self.w0 = kc_mbon.clone()
        self.w = kc_mbon.clone()
        # fixed MBON->value gains: approach +, avoid -
        self.readout = torch.zeros(self.n_mbon, device=self.device)
        self.eta = 0.0

        # fixed random sparse PN features; a fraction also reads the action code
        g = torch.Generator().manual_seed(seed)
        u = torch.zeros(self.n_pn, n_features)
        for i in range(self.n_pn):
            cols = torch.randperm(n_features, generator=g)[:pn_state_inputs]
            u[i, cols] = torch.randn(pn_state_inputs, generator=g)
        reads_action = torch.rand(self.n_pn, generator=g) < pn_action_fraction
        act_col = torch.randint(0, 9, (self.n_pn,), generator=g)   # 9 moves
        act_sign = torch.sign(torch.randn(self.n_pn, generator=g))
        self.pn_u = u.to(self.device)
        mask = torch.zeros(self.n_pn, 9, device=self.device)
        idx = torch.arange(self.n_pn, device=self.device)[reads_action]
        mask[idx, act_col[reads_action]] = act_sign[reads_action].to(self.device)
        self.pn_action_mask = mask
        self.action_gain = 1.0
        self.feat_mean = torch.zeros(n_features, device=self.device)
        self.feat_scale = torch.ones(n_features, device=self.device)
        self.pn_theta = torch.zeros(self.n_pn, device=self.device)
        self.pn_scale = torch.ones(self.n_pn, device=self.device)

    # ------------------------------------------------------------------ codes
    def _pn_pre(self, features: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        x = (features - self.feat_mean) / self.feat_scale
        return x @ self.pn_u.T + self.action_gain * (codes @ self.pn_action_mask.T)

    def _pn(self, features: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        return torch.relu(self._pn_pre(features, codes) - self.pn_theta) * self.pn_scale

    def kc_code(self, features: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        h = self._pn(features, codes) @ self.pn_kc + self.kc_bias
        idx = torch.topk(h, self.kc_active, dim=1).indices
        kc = torch.zeros(h.shape[0], self.n_kc, dtype=torch.bool, device=self.device)
        kc.scatter_(1, idx, True)
        return kc

    def kc_all_actions(self, features: torch.Tensor) -> torch.Tensor:
        """(N, F) -> (N, 9, n_kc) bool codes for every candidate move."""
        n = features.shape[0]
        codes = torch.eye(9, device=features.device)
        f = features.unsqueeze(1).expand(n, 9, self.n_features).reshape(-1, self.n_features)
        c = codes.unsqueeze(0).expand(n, -1, -1).reshape(-1, 9)
        return self.kc_code(f, c).view(n, 9, self.n_kc)

    def drives(self, kc: torch.Tensor) -> torch.Tensor:
        return (kc.to(torch.float32) @ self.w) / self.kc_active

    def value(self, kc: torch.Tensor) -> torch.Tensor:
        d = self.drives(kc.reshape(-1, self.n_kc))
        return (d @ self.readout).view(kc.shape[:-1])

    # -------------------------------------------------------------- calibrate
    @torch.no_grad()
    def calibrate(self, features: torch.Tensor, target_overlap: float = 0.5) -> dict:
        m = features.shape[0]
        self.feat_mean = features.mean(dim=0)
        self.feat_scale = features.std(dim=0).clamp_min(0.05)
        codes = torch.eye(9, device=features.device)[
            torch.randint(0, 9, (m,), generator=self.gen).to(features.device)]

        def set_gain(gv):
            self.action_gain = gv
            self.pn_theta = torch.zeros(self.n_pn, device=self.device)
            self.pn_scale = torch.ones(self.n_pn, device=self.device)
            z = self._pn_pre(features, codes)
            self.pn_theta = torch.quantile(z, self.pn_quantile, dim=0)
            p = torch.relu(z - self.pn_theta)
            self.pn_scale = 1.0 / p.mean(dim=0).clamp_min(1e-6)

        def overlap():
            kc = self.kc_all_actions(features[:1024])
            a, brest = kc[:, 0], kc[:, 1:]
            inter = (a.unsqueeze(1) & brest).sum(dim=2).float()
            return float((inter / self.kc_active).mean())

        lo, hi = 0.0, 64.0
        for _ in range(17):
            mid = 0.5 * (lo + hi)
            set_gain(mid)
            if overlap() > target_overlap:
                lo = mid
            else:
                hi = mid
        set_gain(0.5 * (lo + hi))
        overlap_v = overlap()
        kc = self.kc_all_actions(features[:1024]).reshape(-1, self.n_kc)
        drive0 = self.drives(kc).mean(dim=0)
        connected = drive0 > 0
        scale = torch.where(connected, 1.0 / drive0.clamp_min(1e-9), torch.zeros_like(drive0))
        self.w0 = self.w0 * scale
        self.w = self.w0.clone()
        self.readout = torch.zeros(self.n_mbon, device=self.device)
        n_app = int((self.approach & connected).sum())
        n_avd = int((self.avoid & connected).sum())
        self.readout[self.approach & connected] = self.gain / max(n_app, 1)
        self.readout[self.avoid & connected] = -self.gain / max(n_avd, 1)
        conn = (self.w0 > 0).to(torch.float32)
        f = (kc.to(torch.float32) @ conn).mean(dim=0) / self.kc_active
        eff_app = float((f * self.readout.clamp_min(0)).sum())
        eff_avd = float((f * (-self.readout).clamp_min(0)).sum())
        self.eta = self.alpha / max(eff_app + eff_avd, 1e-9)
        return {"action_gain": self.action_gain, "action_overlap": overlap_v,
                "eta": self.eta, "mbons_connected": int(connected.sum()),
                "approach_mbons": n_app, "avoid_mbons": n_avd,
                "kc_total": self.n_kc}

    # --------------------------------------------------------------- learning
    @torch.no_grad()
    def dopamine_update(self, kc: torch.Tensor, rpe: torch.Tensor) -> float:
        """dw = -eta * KC * signed_dopamine for a batch of (KC code, RPE) pairs.

        PAM (reward) compartments carry +RPE and act on avoid MBONs; PPL1
        compartments carry -RPE and act on approach MBONs. Above-baseline
        dopamine with an active KC depresses the synapse; below baseline it
        recovers. Weights stay in [0, w0]."""
        kcf = kc.to(torch.float32)
        coincidence = kcf.T @ rpe
        before = self.w
        signed = torch.where(self.avoid.unsqueeze(0), coincidence.unsqueeze(1),
                             torch.zeros_like(self.w))
        signed = torch.where(self.approach.unsqueeze(0), -coincidence.unsqueeze(1), signed)
        self.w = torch.minimum((self.w - self.eta * signed).clamp_min(0.0), self.w0)
        exist = self.w0 > 0
        return float((self.w - before).abs()[exist].mean())

    def saturation(self):
        exist = self.w0 > 0
        return (float((self.w[exist] <= 0).float().mean()),
                float((self.w[exist] >= self.w0[exist]).float().mean()))

    def state_dict_np(self) -> dict:
        return {"w": self.w.cpu().numpy(), "w0": self.w0.cpu().numpy(),
                "readout": self.readout.cpu().numpy(), "group": self.region.cpu().numpy(),
                "pn_u": self.pn_u.cpu().numpy(), "pn_action_mask": self.pn_action_mask.cpu().numpy(),
                "pn_theta": self.pn_theta.cpu().numpy(), "pn_scale": self.pn_scale.cpu().numpy(),
                "feat_mean": self.feat_mean.cpu().numpy(), "feat_scale": self.feat_scale.cpu().numpy(),
                "action_gain": np.array(self.action_gain), "eta": np.array(self.eta),
                "gain": np.array(self.gain), "kc_active": np.array(self.kc_active)}

    def load_state_np(self, state: dict):
        self.w = torch.as_tensor(state["w"]).to(self.device)
        self.w0 = torch.as_tensor(state["w0"]).to(self.device)
        self.readout = torch.as_tensor(state["readout"]).to(self.device)
        self.mbon_group = torch.as_tensor(state["group"]).to(self.device)
        self.avoid = self.mbon_group == self.REWARD
        self.approach = self.mbon_group == self.PUNISHMENT
        self.pn_u = torch.as_tensor(state["pn_u"]).to(self.device)
        self.pn_action_mask = torch.as_tensor(state["pn_action_mask"]).to(self.device)
        self.pn_theta = torch.as_tensor(state["pn_theta"]).to(self.device)
        self.pn_scale = torch.as_tensor(state["pn_scale"]).to(self.device)
        self.feat_mean = torch.as_tensor(state["feat_mean"]).to(self.device)
        self.feat_scale = torch.as_tensor(state["feat_scale"]).to(self.device)
        self.action_gain = float(state["action_gain"]); self.eta = float(state["eta"])
        self.gain = float(state["gain"]); self.kc_active = int(state["kc_active"])
