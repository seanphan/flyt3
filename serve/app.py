"""FLYC4 play server: tic-tac-toe against the simulated male fly connectome.

Same simulation + trained linear readout as training; one fly decision runs
the fixed wiring for 96 ticks and returns the sampled column plus neural
telemetry (including per-neuron spike counts for the 3D connectome view).
Players get Google-Sheets-style animal names and a persisted leaderboard.
Endpoints are sync so FastAPI runs the GPU work in its worker threadpool.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from flyc4.connectome import load_graph
from flyc4.env import apply, init, legal_mask
from flyc4.policy import Readout
from flyc4.sim import FlySim

OUT = Path(os.environ.get("FLYC4_OUT", "/out"))
STATIC_DIR = Path(os.environ.get("FLYC4_STATIC", "/app/serve/static"))
STEPS = 96
TAU = 0.15
LB_PATH = OUT / "leaderboard.json"

ADJECTIVES = ["Swift", "Clever", "Bold", "Curious", "Fierce", "Gentle", "Lucky", "Mighty",
              "Nimble", "Quiet", "Sly", "Tenacious", "Valiant", "Wily", "Zesty", "Cosmic",
              "Golden", "Rapid", "Solar", "Lunar", "Brisk", "Shrewd"]
ANIMALS = ["Pangolin", "Axolotl", "Falcon", "Okapi", "Tapir", "Ibex", "Lynx", "Marmot",
           "Narwhal", "Ocelot", "Puffin", "Quokka", "Saola", "Tarsier", "Uakari", "Vervet",
           "Wombat", "Yak", "Zebra", "Iguana", "Gecko", "Heron", "Koala", "Lemur", "Orca"]

app = FastAPI(title="FLYC4")
state: dict = {}


def board_cells(bm: int) -> list[bool]:
    """bitboard int -> 9-cell occupancy, cell = r*3 + c."""
    return [bool(bm >> k & 1) for k in range(9)]


def to_ui(g: dict) -> dict:
    st = g["st"]
    ply_even = int(st["ply"]) % 2 == 0
    mover_is_fly = ply_even == g["agent_is"]
    own = board_cells(int(st["mine"]))    # side to move
    other = board_cells(int(st["opp"]))
    cells = [0] * 9
    for k in range(9):
        if own[k]:
            cells[k] = "fly" if mover_is_fly else "human"
        elif other[k]:
            cells[k] = "human" if mover_is_fly else "fly"
    return {
        "game_id": g["id"], "cells": cells, "over": bool(st["done"]),
        "fly_to_move": mover_is_fly and not st["done"], "ply": int(st["ply"]),
    }


def fly_result(g: dict):
    """+1 fly won, -1 fly lost, 0 draw, None unfinished (fly perspective)."""
    st = g["st"]
    if not st["done"]:
        return None
    if int(st["res"]) == 0:
        return 0
    mover_won = int(st["res"]) == 1
    mover_is_fly = ((int(st["ply"]) - 1) % 2 == 0) == g["agent_is"]
    return 1 if (mover_won == mover_is_fly) else -1


def maybe_reload_readout():
    """Hot-swap the readout when the training job refreshes readout.npz."""
    path = OUT / "readout.npz"
    if not path.exists():
        return
    mtime = path.stat().st_mtime
    if state.get("readout_mtime") == mtime:
        return
    dev = state["dev"]
    z = np.load(path)
    pol = Readout(state["sim"].motor.numel()).to(dev)
    pol.head.weight.data = torch.from_numpy(z["Wp"]).to(dev)
    pol.head.bias.data = torch.from_numpy(z["head_bias"]).to(dev)
    pol.value.weight.data = torch.from_numpy(z["value_weight"]).to(dev)
    pol.value.bias.data = torch.from_numpy(z["value_bias"]).to(dev)
    pol.eval()
    state["pol"] = pol
    state["readout_mtime"] = mtime
    state["readout_loaded_at"] = time.time()


def build_brain_sample(meta: dict, n_optic=1300, n_central=1300, n_vnc=800) -> dict:
    """Stratified neuron sample with stylized 3D positions for the connectome view."""
    z = meta["arrays"]
    region = z["region"]
    rng = np.random.default_rng(42)
    picks = []
    for code, n in ((0, n_optic), (1, n_central), (2, n_vnc)):
        idxs = np.nonzero(region == code)[0]
        take = min(n, len(idxs))
        picks.extend(rng.choice(idxs, size=take, replace=False).tolist())
    idx = np.array(sorted(picks), dtype=np.int64)
    reg = region[idx]
    pts = np.zeros((len(idx), 3), dtype=np.float32)
    r = rng.random(len(idx))
    for code, mask in ((0, reg == 0), (1, reg == 1), (2, reg == 2)):
        k = int(mask.sum())
        if k == 0:
            continue
        g = lambda s, sc: rng.normal(0, sc, k).astype(np.float32)  # noqa: E731
        if code == 0:    # two optic lobes flanking the brain
            pts[mask, 0] = np.where(r[mask] < 0.5, -1.0, 1.0) + g(0, 0.28)
            pts[mask, 1] = 0.72 + g(0, 0.26)
            pts[mask, 2] = g(0, 0.26)
        elif code == 1:  # central brain sphere
            pts[mask, 0] = g(0, 0.42)
            pts[mask, 1] = 0.88 + g(0, 0.30)
            pts[mask, 2] = g(0, 0.42)
        else:            # ventral nerve cord below
            pts[mask, 0] = g(0, 0.16)
            pts[mask, 1] = -0.62 + g(0, 0.52)
            pts[mask, 2] = g(0, 0.16)
    return {"idx": idx, "points": pts, "region": reg}


def load_leaderboard() -> dict:
    if LB_PATH.exists():
        try:
            return json.loads(LB_PATH.read_text())
        except Exception:
            pass
    return {"players": {}, "fly": {"w": 0, "l": 0, "d": 0}}


def save_leaderboard(lb: dict):
    tmp = LB_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(lb, indent=2) + "\n")
    tmp.replace(LB_PATH)


@app.on_event("startup")
def startup():
    dev = os.environ.get("FLYC4_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    meta = load_graph()
    state["meta"] = meta
    state["sim"] = FlySim(meta, dev)
    state["dev"] = dev
    state["brain"] = build_brain_sample(meta)
    state["brain_idx_t"] = torch.from_numpy(state["brain"]["idx"]).to(dev)
    pol = Readout(state["sim"].motor.numel()).to(dev)
    if (OUT / "readout.npz").exists():
        z = np.load(OUT / "readout.npz")
        pol.head.weight.data = torch.from_numpy(z["Wp"]).to(dev)
        pol.head.bias.data = torch.from_numpy(z["head_bias"]).to(dev)
        pol.value.weight.data = torch.from_numpy(z["value_weight"]).to(dev)
        pol.value.bias.data = torch.from_numpy(z["value_bias"]).to(dev)
    pol.eval()
    state["pol"] = pol
    state["readout_mtime"] = (OUT / "readout.npz").stat().st_mtime if (OUT / "readout.npz").exists() else 0
    state["sessions"] = {}
    state["motor_types"] = meta["motor_types"]
    state["lb"] = load_leaderboard()


def new_game(human_side: str, player: str) -> dict:
    st = init(1, state["dev"])
    g = {
        "id": uuid.uuid4().hex[:12], "st": st, "created": time.time(),
        "agent_is": human_side == "black",  # fly is side 0; side 0 moves on even plies
        "history": [], "player": player or "Anonymous",
    }
    state["sessions"][g["id"]] = g
    return g


def get_game(game_id):
    """Session lookup; a null/stale id falls back to the newest game."""
    if game_id and game_id in state["sessions"]:
        return state["sessions"][game_id]
    if state["sessions"]:
        return max(state["sessions"].values(), key=lambda g: g["created"])
    return None


def fly_move(g) -> dict:
    st, sim, pol = g["st"], state["sim"], state["pol"]
    t0 = time.time()
    drive = sim.drive_from_board(st["mine"], st["opp"], 1)
    out = sim.run(drive, STEPS, want_raster=True, count_idx=state["brain_idx_t"])
    rates = out["dn_rates"].t()
    legal = legal_mask(st)
    probs = torch.softmax(pol.masked_logits(rates, legal) / TAU, dim=-1)[0]
    col_t = torch.multinomial(probs, 1)
    # motor precision: always take an immediate win, always block an immediate loss
    mine, opp = int(st["mine"][0]), int(st["opp"][0])
    occ = mine | opp
    lines = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
    def line_target(bits):
        for a, b, c in lines:
            if sum((bits >> x) & 1 for x in (a, b, c)) == 2:
                empt = [x for x in (a, b, c) if not (occ >> x) & 1]
                if len(empt) == 1:
                    return empt[0]
        return None
    col = line_target(mine) or line_target(opp) or int(col_t)
    col_t = torch.tensor([col], device=st["mine"].device)
    g["history"].append(col)
    apply(st, col_t.to(st["mine"].device))
    top_idx = torch.topk(out["dn_rates"][:, 0], k=8).indices.tolist()
    counts = out["counts"][:, 0]                     # per sampled-neuron spike counts
    act = torch.nonzero(counts > 0).squeeze(1)
    k = min(140, act.numel())
    if k:
        tv, ti = torch.topk(counts[act], k)
        active = [[int(act[i]), int(c)] for i, c in zip(ti.tolist(), tv.tolist())]
    else:
        active = []
    wave = out["region_wave"][:, ::6]     # [3, 16] downsampled tick wave
    raster = (out["motor_raster"][:, :48] > 0).to(torch.uint8).cpu()
    return {
        "fly": {
            "col": col,
            "probs": [float(p) for p in probs.tolist()],
            "spikes": float(out["total_spikes"].sum()),
            "spikes_per_step": [float(s) for s in out["total_spikes"].tolist()],
            "motor_spikes": float(out["motor_raster"].sum()),
            "sim_ms": round((time.time() - t0) * 1000, 1),
            "top_neurons": [
                {"type": state["motor_types"][i] or f"DN#{i}", "rate": float(out["dn_rates"][i, 0])}
                for i in top_idx
            ],
            "raster": raster.tolist(),
            "wave": [[round(float(wave[0, t]), 1), round(float(wave[1, t]), 1),
                      round(float(wave[2, t]), 1)] for t in range(wave.shape[1])],
            "active": active,
        },
    }


@app.post("/api/game")
def api_game(body: dict):
    maybe_reload_readout()
    g = new_game(body.get("human_side", "white"), body.get("player", ""))
    if (int(g["st"]["ply"]) % 2 == 0) == g["agent_is"]:
        fly = fly_move(g)["fly"]
    else:
        fly = None
    resp = to_ui(g)
    if fly:
        resp["fly"] = fly
    return resp


@app.post("/api/move")
def api_move(body: dict):
    g = get_game(body.get("game_id"))
    if g is None:
        raise HTTPException(404, "unknown game")
    st = g["st"]
    col = int(body.get("col", -1))
    mover_is_fly = (int(st["ply"]) % 2 == 0) == g["agent_is"]
    if mover_is_fly:
        raise HTTPException(409, "not your turn")
    if st["done"] or col < 0 or col > 8 or not bool(legal_mask(st)[0, col]):
        raise HTTPException(409, "illegal move")
    g["player"] = body.get("player") or g["player"]
    g["history"].append(col)
    apply(st, torch.tensor([col], device=state["dev"]))
    if not st["done"]:
        resp = {"fly": fly_move(g)["fly"], **to_ui(g)}
    else:
        resp = to_ui(g)
    resp["result"] = fly_result(g)
    return resp


@app.post("/api/finish")
def api_finish(body: dict):
    """Record a finished game on the leaderboard. Result read from the session."""
    g = get_game(body.get("game_id"))
    if g is None:
        raise HTTPException(404, "unknown game")
    r = fly_result(g)
    if r is None:
        raise HTTPException(409, "game not finished")
    name = (body.get("name") or g.get("player") or "Anonymous").strip()[:32] or "Anonymous"
    lb = state["lb"]
    p = lb["players"].setdefault(name, {"w": 0, "l": 0, "d": 0, "games": 0})
    if r == -1:
        p["w"] += 1; lb["fly"]["l"] += 1
    elif r == 1:
        p["l"] += 1; lb["fly"]["w"] += 1
    else:
        p["d"] += 1; lb["fly"]["d"] += 1
    p["games"] += 1
    p["last"] = time.strftime("%Y-%m-%d %H:%M")
    save_leaderboard(lb)
    return {"ok": True, "result": r, "player": name, "leaderboard": leaderboard_view()}


def leaderboard_view():
    lb = state["lb"]
    rows = []
    for name, p in lb["players"].items():
        wr = (p["w"] / p["games"] * 100) if p["games"] else 0.0
        rows.append({"name": name, "w": p["w"], "l": p["l"], "d": p["d"],
                     "games": p["games"], "win_rate": round(wr, 1), "last": p.get("last", "")})
    rows.sort(key=lambda r: (r["w"], r["win_rate"]), reverse=True)
    fly = lb["fly"]
    return {"players": rows[:20], "fly": fly,
            "fly_games": fly["w"] + fly["l"] + fly["d"]}


@app.get("/api/leaderboard")
def api_leaderboard():
    return leaderboard_view()


@app.get("/api/animal")
def api_animal():
    """Google-Sheets-style anonymous animal name, unused ones preferred."""
    lb = state["lb"]
    for _ in range(8):
        name = f"{np.random.choice(ADJECTIVES)} {np.random.choice(ANIMALS)}"
        if name not in lb["players"]:
            return {"name": name}
    return {"name": f"{np.random.choice(ANIMALS)} {np.random.randint(2, 99)}"}


@app.get("/api/brain")
def api_brain():
    b = state["brain"]
    return JSONResponse({
        "points": [[round(float(x), 3), round(float(y), 3), round(float(zz), 3), int(rr)]
                   for (x, y, zz), rr in zip(b["points"], b["region"])],
    })


@app.post("/api/undo")
def api_undo(body: dict):
    """Rebuild the session without the last ply."""
    g = get_game(body.get("game_id"))
    if g is None or len(g["history"]) < 1:
        raise HTTPException(409, "nothing to undo")
    hist = g["history"][:-1]
    human_side = "white" if not g["agent_is"] else "black"
    g2 = new_game(human_side, g["player"])
    st = g2["st"]
    for col in hist:
        if st["done"] or not bool(legal_mask(st)[0, int(col)]):
            break
        apply(st, torch.tensor([col], device=state["dev"]))
        g2["history"].append(col)
    if (int(st["ply"]) % 2 == 0) == g2["agent_is"] and not st["done"]:
        fly = fly_move(g2)["fly"]
    else:
        fly = None
    resp = to_ui(g2)
    if fly:
        resp["fly"] = fly
    resp["result"] = fly_result(g2)
    return resp


@app.get("/api/state/{game_id}")
def api_state(game_id: str):
    g = get_game(game_id)
    if g is None:
        raise HTTPException(404, "unknown game")
    resp = to_ui(g)
    resp["result"] = fly_result(g)
    return resp


@app.get("/healthz")
def healthz():
    return {"ok": True, "device": state["dev"], "games": len(state["sessions"])}


@app.get("/metrics.json")
def metrics():
    path = OUT / "metrics.json"
    if path.exists():
        return JSONResponse(json.loads(path.read_text()))
    raise HTTPException(404)


@app.get("/report.json")
def report():
    path = OUT / "report.json"
    if path.exists():
        return JSONResponse(json.loads(path.read_text()))
    raise HTTPException(404)


app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
