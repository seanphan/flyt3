"""Self-play REINFORCE training of the fly's linear readout.

Every batch: `--games` Connect-4 games in lockstep on the GPU connectome.
The fly's moves are sampled from its descending-neuron rates through the
readout; opponent is the previous snapshot or uniform-random. Terminal result
is the reward; losses also schedule an aversive PPL101 dopamine pulse (the
gradient itself is REINFORCE + value baseline + entropy bonus).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .connectome import build, load_graph
from .env import apply, init, legal_mask
from .eval import evaluate
from .policy import Readout
from .sim import FlySim


def fly_drive_move(sim, pol, st, idx, steps, tau, grad: bool):
    """One fly decision round for games `idx`; returns (cols, logp, ent, v, rates)."""
    drive = sim.drive_from_board(st["mine"][idx], st["opp"][idx], int(idx.numel()))
    out = sim.run(drive, steps)
    rates = out["dn_rates"].t()
    legal = legal_mask(st)[idx]
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        logits = pol.masked_logits(rates, legal)
        probs = F.softmax(logits / tau, dim=-1)
        cols = torch.multinomial(probs, 1).squeeze(1)
        logp = torch.log(probs.gather(1, cols.unsqueeze(1)).squeeze(1) + 1e-12)
        ent = -(probs * torch.log(probs + 1e-12)).sum(-1)
        v = pol.value_of(rates)
    return cols, logp, ent, v, out


def play_batch(sim, pol, snap, B, steps, tau, device):
    st = init(B, device)
    agent_is = torch.rand(B, device=device) < 0.5        # fly moves first in half
    opp_kind = torch.rand(B, device=device) < 0.5        # snapshot vs random
    gid, logps, ents = [], [], []
    v_last = torch.zeros(B, device=device)
    z = torch.full((B,), float("nan"), device=device)
    n_pulses = 0
    while not bool(st["done"].all()):
        ply_even = (st["ply"] % 2) == 0
        fm = (~st["done"]) & (agent_is == ply_even)
        if fm.any():
            idx = fm.nonzero().squeeze(1)
            cols, lp, ent, v, _ = fly_drive_move(sim, pol, st, idx, steps, tau, grad=True)
            apply(st, cols, idx)
            v_last[idx] = v
            fin = idx[st["done"][idx]]
            if fin.numel():
                zf = (st["res"][fin] == 1).to(torch.float32)
                z[fin] = zf
                lost = fin[zf == 0]
                if lost.numel():
                    n_pulses += int(lost.numel())
                    d = sim.drive_from_board(st["mine"][lost], st["opp"][lost], int(lost.numel()))
                    sim.punish(d)  # aversive pulse; documented, gradient comes from REINFORCE
            gid.append(idx)
            logps.append(lp)
            ents.append(ent)
        om = (~st["done"]) & (agent_is != ply_even)
        if om.any():
            idx = om.nonzero().squeeze(1)
            legal = legal_mask(st)[idx]
            use_snap = opp_kind[idx]
            cols = torch.empty(idx.numel(), dtype=torch.int64, device=device)
            si = idx[use_snap]
            if si.numel():
                d = sim.drive_from_board(st["mine"][si], st["opp"][si], int(si.numel()))
                out = sim.run(d, steps)
                sc, _, _ = snap.act(out["dn_rates"].t(), legal_mask(st)[si], tau=0.5)
                cols[use_snap] = sc
            ri = idx[~use_snap]
            if ri.numel():
                lg = legal[~use_snap]
                probs = lg.float() / lg.float().sum(1, keepdim=True)
                cols[~use_snap] = torch.multinomial(probs, 1).squeeze(1)
            apply(st, cols, idx)
            fin = idx[st["done"][idx]]
            if fin.numel():
                zf = -(st["res"][fin] == 1).to(torch.float32)
                z[fin] = zf
                lost = fin[zf < 0]
                if lost.numel():
                    n_pulses += int(lost.numel())
                    d = sim.drive_from_board(st["mine"][lost], st["opp"][lost], int(lost.numel()))
                    sim.punish(d)  # aversive PPL101 pulse on every fly loss
    gid = torch.cat(gid)
    logps = torch.cat(logps)
    ents = torch.cat(ents)
    sum_logp = torch.zeros(B, device=device).index_add_(0, gid, logps)
    sum_ent = torch.zeros(B, device=device).index_add_(0, gid, ents)
    n_dec = torch.zeros(B, device=device).index_add_(0, gid, torch.ones_like(logps))
    return {"z": z, "sum_logp": sum_logp, "sum_ent": sum_ent, "n_dec": n_dec,
            "v_last": v_last, "opp_snap": opp_kind, "n_pulses": n_pulses}


def update(pol, opt, batch, ent_coef, vf_coef=0.5):
    z = batch["z"]
    adv = (z - batch["v_last"]).detach()
    n = float(z.numel())
    loss_pi = -(adv * batch["sum_logp"]).sum() / n
    loss_v = ((batch["v_last"] - z) ** 2).mean()
    loss_ent = -(batch["sum_ent"] / batch["n_dec"].clamp(min=1)).mean()
    loss = loss_pi + vf_coef * loss_v + ent_coef * loss_ent
    return {"loss": float(loss.detach()), "pi": float(loss_pi.detach()), "v": float(loss_v.detach()),
            "ent": float(-loss_ent.detach()), "mean_abs_adv": float(adv.abs().mean())}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=200)
    ap.add_argument("--games", type=int, default=768)
    ap.add_argument("--steps", type=int, default=96)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--snapshot-every", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=Path("/out"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    meta = load_graph() if (Path("/library/datasets/malecns_v1/processed-flyc4/meta.json")).exists() else build()
    sim = FlySim(meta, args.device)
    pol = Readout(sim.motor.numel()).to(args.device)
    snap = Readout(sim.motor.numel()).to(args.device)
    opt = torch.optim.AdamW(pol.parameters(), lr=args.lr)

    start_batch, m_off = 0, 0
    ckpt = args.out / "ckpt.pt"
    if args.resume and ckpt.exists():
        s = torch.load(ckpt, map_location=args.device, weights_only=False)
        pol.load_state_dict(s["pol"]); snap.load_state_dict(s["snap"])
        opt.load_state_dict(s["opt"])
        start_batch, m_off = s["batch"] + 1, s.get("metrics_lines", 0)
        print(json.dumps({"evt": "resumed", "batch": start_batch}), flush=True)

    metrics_path = args.out / "metrics.jsonl"
    t0 = time.time()
    for b in range(start_batch, args.batches):
        tb = time.time()
        batch = play_batch(sim, pol, snap, args.games, args.steps, args.tau, args.device)
        stats = update(pol, opt, batch, args.ent_coef)
        z = batch["z"]
        row = {
            "batch": b, "games": args.games,
            "win_vs_all": float((z > 0).float().mean()), "loss_vs_all": float((z < 0).float().mean()),
            "win_vs_snapshot": float((z[batch["opp_snap"]] > 0).float().mean()),
            "seconds": round(time.time() - tb, 2),
            "dopamine_pulses": batch["n_pulses"],
            **{k: round(v, 4) for k, v in stats.items()},
        }
        print(json.dumps({"evt": "batch", **row}), flush=True)
        if b == start_batch or (b + 1) % args.snapshot_every == 0:
            snap.load_state_dict(pol.state_dict())
        if (b + 1) % args.eval_every == 0 or b == args.batches - 1:
            ev_r = evaluate(sim, pol, "random", 256, args.steps, args.device)
            ev_m = evaluate(sim, pol, "minimax2", 96, args.steps, args.device)
            with metrics_path.open("a") as fh:
                fh.write(json.dumps({"batch": b, "vs_random": ev_r, "vs_minimax2": ev_m}) + "\n")
            print(json.dumps({"evt": "eval", "batch": b, "vs_random": ev_r, "vs_minimax2": ev_m}), flush=True)
        if (b + 1) % args.save_every == 0 or b == args.batches - 1:
            torch.save({"pol": pol.state_dict(), "snap": snap.state_dict(),
                        "opt": opt.state_dict(), "batch": b, "metrics_lines": m_off}, ckpt)

    # final export
    W = pol.head.weight.detach().cpu().numpy().astype(np.float32)
    np.savez(args.out / "readout.npz", Wp=W, head_bias=pol.head.bias.detach().cpu().numpy(),
             value_weight=pol.value.weight.detach().cpu().numpy(),
             value_bias=pol.value.bias.detach().cpu().numpy(),
             motor_types=np.array(meta["motor_types"]))
    total_games = (args.batches - start_batch) * args.games
    report = {
        "algorithm": "REINFORCE + value baseline + entropy bonus on a linear readout of "
                     "descending-neuron spike rates; connectome wiring fixed throughout",
        "batches": args.batches - start_batch, "games": int(total_games),
        "dopamine_pulses_total": "see metrics", "wall_seconds": round(time.time() - t0, 1),
        "gpu": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else "cpu",
        "connectome": {k: meta[k] for k in ("n_nodes", "n_edges", "interface")},
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"evt": "done", **{k: report[k] for k in ("batches", "games", "wall_seconds")}}), flush=True)


if __name__ == "__main__":
    main()
