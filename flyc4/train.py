"""Self-play REINFORCE training of the fly's linear readout.

Every batch: `--games` tic-tac-toe games in lockstep on the GPU connectome.
The fly's moves are sampled from its descending + motor-neuron rates through
the readout. Half the games are pure self-play (both sides played and learned
by the same policy, credited from their own perspective); the other half are
against a uniform-random opponent (only the fly's side learns). Terminal
result is the reward; losses also schedule an aversive PPL101 dopamine pulse
-- the gradient itself is REINFORCE + value baseline + entropy bonus.
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


def fly_decision(sim, pol, st, idx, steps, tau, grad: bool):
    """One fly decision round for games `idx`; returns (cols, logp, ent, v)."""
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
    return cols, logp, ent, v


def play_batch(sim, pol, B, steps, tau, device):
    st = init(B, device)
    agent_is = torch.rand(B, device=device) < 0.5        # fly is side 0 in half
    self_play = torch.rand(B, device=device) < 0.5       # both sides learned
    gid, logps, ents = [], [], []
    v_last = torch.zeros(B, device=device)
    z = torch.full((B,), float("nan"), device=device)
    n_pulses = 0
    while not bool(st["done"].all()):
        ply_even = (st["ply"] % 2) == 0
        for side0 in (True, False):
            mv = (~st["done"]) & (ply_even == side0)
            if not mv.any():
                continue
            idx = mv.nonzero().squeeze(1)
            mover_is_fly = (agent_is[idx] == side0)
            learns = mover_is_fly | self_play[idx]       # credit only policy moves
            li = idx[learns]
            cols = torch.empty(idx.numel(), dtype=torch.int64, device=device)
            if li.numel():
                lc, lp, ent, v = fly_decision(sim, pol, st, li, steps, tau, grad=True)
                cols[learns] = lc
                gid.append(li)
                logps.append(lp * torch.where(mover_is_fly[learns], 1.0, -1.0))
                ents.append(ent)
                v_last[li[mover_is_fly[learns]]] = v[mover_is_fly[learns]]
            ri = idx[~learns]
            if ri.numel():
                lg = legal_mask(st)[ri]
                probs = lg.float() / lg.float().sum(1, keepdim=True)
                cols[~learns] = torch.multinomial(probs, 1).squeeze(1)
            apply(st, cols, idx)
            fin = idx[st["done"][idx]]
            if fin.numel():
                mover_won = st["res"][fin] == 1
                mover_is_fly = ((st["ply"][fin] - 1) % 2 == 0) == agent_is[fin]
                zf = torch.where(mover_won == mover_is_fly, 1.0, -1.0)
                zf = torch.where(st["res"][fin] == 0, torch.zeros_like(zf), zf)
                z[fin] = zf
                lost = fin[zf < 0]
                if lost.numel():
                    n_pulses += int(lost.numel())
                    d = sim.drive_from_board(st["mine"][lost], st["opp"][lost], int(lost.numel()))
                    sim.punish(d)  # aversive pulse; gradient comes from REINFORCE
    gid = torch.cat(gid)
    logps = torch.cat(logps)
    ents = torch.cat(ents)
    sum_logp = torch.zeros(B, device=device).index_add_(0, gid, logps)
    sum_ent = torch.zeros(B, device=device).index_add_(0, gid, ents)
    n_dec = torch.zeros(B, device=device).index_add_(0, gid, torch.ones_like(logps))
    return {"z": z, "sum_logp": sum_logp, "sum_ent": sum_ent, "n_dec": n_dec,
            "v_last": v_last, "n_pulses": n_pulses}


def update(pol, opt, batch, ent_coef, vf_coef=0.5):
    z = batch["z"]
    adv = (z - batch["v_last"]).detach()
    adv = (adv - adv.mean()) / (adv.std() + 1e-6)
    n = float(z.numel())
    loss_pi = -(adv * batch["sum_logp"]).sum() / n
    loss_v = ((batch["v_last"] - z) ** 2).mean()
    loss_ent = -(batch["sum_ent"] / batch["n_dec"].clamp(min=1)).mean()
    loss = loss_pi + vf_coef * loss_v + ent_coef * loss_ent
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    return {"loss": float(loss.detach()), "pi": float(loss_pi.detach()), "v": float(loss_v.detach()),
            "ent": float(-loss_ent.detach()), "mean_abs_adv": float(adv.abs().mean())}


def export_readout(pol, path, meta):
    W = pol.head.weight.detach().cpu().numpy().astype(np.float32)
    np.savez(path, Wp=W, head_bias=pol.head.bias.detach().cpu().numpy(),
             value_weight=pol.value.weight.detach().cpu().numpy(),
             value_bias=pol.value.bias.detach().cpu().numpy(),
             motor_types=np.array(meta["motor_types"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=200)
    ap.add_argument("--games", type=int, default=512)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--lr", type=float, default=3e-2)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--save-every", type=int, default=5)
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
    opt = torch.optim.AdamW(pol.parameters(), lr=args.lr)

    start_batch, m_off = 0, 0
    ckpt = args.out / "ckpt.pt"
    if args.resume and ckpt.exists():
        try:
            s = torch.load(ckpt, map_location=args.device, weights_only=False)
            pol.load_state_dict(s["pol"])
            opt.load_state_dict(s["opt"])
            opt.param_groups[0]["lr"] = args.lr  # CLI lr wins over the checkpointed one
            start_batch, m_off = s["batch"] + 1, s.get("metrics_lines", 0)
            print(json.dumps({"evt": "resumed", "batch": start_batch}), flush=True)
        except RuntimeError as e:
            print(json.dumps({"evt": "fresh-start", "reason": str(e)[:120]}), flush=True)
            start_batch, m_off = 0, 0

    metrics_path = args.out / "metrics.jsonl"
    t0 = time.time()
    for b in range(start_batch, args.batches):
        tb = time.time()
        batch = play_batch(sim, pol, B=args.games, steps=args.steps, tau=args.tau, device=args.device)
        stats = update(pol, opt, batch, args.ent_coef)
        z = batch["z"]
        row = {
            "batch": b, "games": args.games,
            "win_vs_all": float((z > 0).float().mean()), "loss_vs_all": float((z < 0).float().mean()),
            "seconds": round(time.time() - tb, 2),
            "dopamine_pulses": batch["n_pulses"],
            **{k: round(v, 4) for k, v in stats.items()},
        }
        print(json.dumps({"evt": "batch", **row}), flush=True)
        if (b + 1) % args.eval_every == 0 or b == args.batches - 1:
            ev_r = evaluate(sim, pol, "random", 256, args.steps, args.device)
            ev_p = evaluate(sim, pol, "perfect", 96, args.steps, args.device)
            with metrics_path.open("a") as fh:
                fh.write(json.dumps({"batch": b, "vs_random": ev_r, "vs_perfect": ev_p}) + "\n")
            print(json.dumps({"evt": "eval", "batch": b, "vs_random": ev_r, "vs_perfect": ev_p}), flush=True)
        if (b + 1) % args.save_every == 0 or b == args.batches - 1:
            torch.save({"pol": pol.state_dict(), "opt": opt.state_dict(),
                        "batch": b, "metrics_lines": m_off}, ckpt)
            export_readout(pol, args.out / "readout.npz", meta)  # web serves the latest brain live
        if args.batches >= 4 and (b + 1) in (args.batches // 2, 3 * args.batches // 4):
            opt.param_groups[0]["lr"] *= 0.5
            print(json.dumps({"evt": "lr", "lr": opt.param_groups[0]["lr"]}), flush=True)

    # final export
    export_readout(pol, args.out / "readout.npz", meta)
    total_games = (args.batches - start_batch) * args.games
    report = {
        "algorithm": "symmetric self-play REINFORCE + value baseline + entropy bonus on a "
                     "linear readout of descending + motor-neuron spike rates; "
                     "connectome wiring fixed throughout",
        "batches": args.batches - start_batch, "games": int(total_games),
        "wall_seconds": round(time.time() - t0, 1),
        "gpu": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else "cpu",
        "connectome": {k: meta[k] for k in ("n_nodes", "n_edges", "interface")},
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"evt": "done", **{k: report[k] for k in ("batches", "games", "wall_seconds")}}), flush=True)


if __name__ == "__main__":
    main()
