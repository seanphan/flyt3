"""FLYC4 play server: Connect-4 against the simulated male fly connectome.

Same simulation + trained linear readout as training; one fly decision runs
the fixed wiring for 96 ticks on GPU and returns the sampled column plus
neural telemetry for the spectator UI. Sessions live in process memory.
"""
from __future__ import annotations

import json
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

OUT = Path("/out")
STEPS = 96
TAU = 0.35

app = FastAPI(title="FLYC4")
state: dict = {}


def board_cells(bm: int) -> list[bool]:
    cells = [False] * 42
    for k in range(42):
        c, r = k // 6, k % 6
        if bm >> (c * 7 + r) & 1:
            cells[k] = True
    return cells


def to_ui(g: dict) -> dict:
    st = g["st"]
    ply_even = int(st["ply"]) % 2 == 0
    mover_is_fly = ply_even == g["agent_is"]
    own = board_cells(int(st["mine"]))    # side to move
    other = board_cells(int(st["opp"]))
    cells = [0] * 42
    for k in range(42):
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


@app.on_event("startup")
def startup():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    meta = load_graph()
    state["meta"] = meta
    state["sim"] = FlySim(meta, dev)
    state["dev"] = dev
    pol = Readout(state["sim"].motor.numel()).to(dev)
    z = np.load(OUT / "readout.npz")
    pol.head.weight.data = torch.from_numpy(z["Wp"]).to(dev)
    pol.head.bias.data = torch.from_numpy(z["head_bias"]).to(dev)
    pol.value.weight.data = torch.from_numpy(z["value_weight"]).to(dev)
    pol.value.bias.data = torch.from_numpy(z["value_bias"]).to(dev)
    pol.eval()
    state["pol"] = pol
    state["sessions"] = {}
    state["motor_types"] = meta["motor_types"]


def new_game(human_side: str) -> dict:
    st = init(1, state["dev"])
    g = {
        "id": uuid.uuid4().hex[:12], "st": st,
        "agent_is": human_side == "black",  # fly is side 0; side 0 moves on even plies
        "history": [],
    }
    state["sessions"][g["id"]] = g
    return g


@app.post("/api/game")
def api_game(body: dict):
    g = new_game(body.get("human_side", "white"))
    resp = to_ui(g)
    if (int(g["st"]["ply"]) % 2 == 0) == g["agent_is"]:
        resp.update(fly_move(g))
    return resp


@app.post("/api/move")
def api_move(body: dict):
    g = state["sessions"].get(body.get("game_id"))
    if g is None:
        raise HTTPException(404, "unknown game")
    st = g["st"]
    col = int(body.get("col", -1))
    mover_is_fly = (int(st["ply"]) % 2 == 0) == g["agent_is"]
    if mover_is_fly:
        raise HTTPException(409, "not your turn")
    if st["done"] or col < 0 or col > 6 or not bool(legal_mask(st)[0, col]):
        raise HTTPException(409, "illegal move")
    g["history"].append(col)
    apply(st, torch.tensor([col], device=state["dev"]))
    resp = to_ui(g)
    if not st["done"]:
        resp.update(fly_move(g))
    resp["result"] = fly_result(g)
    return resp


def fly_move(g) -> dict:
    st, sim, pol = g["st"], state["sim"], state["pol"]
    t0 = time.time()
    drive = sim.drive_from_board(st["mine"], st["opp"], 1)
    out = sim.run(drive, STEPS, want_raster=True)
    rates = out["dn_rates"].t()
    legal = legal_mask(st)
    probs = torch.softmax(pol.masked_logits(rates, legal) / TAU, dim=-1)[0]
    cols = torch.multinomial(probs, 1).squeeze(1)
    col = int(cols)
    g["history"].append(col)
    apply(st, cols.to(st["mine"].device))
    if state["dev"] == "cuda":
        torch.cuda.synchronize()
    top_idx = torch.topk(out["dn_rates"][:, 0], k=8).indices.tolist()
    raster = (out["motor_raster"][:, :48] > 0).to(torch.uint8).cpu()  # first 48 ticks
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
        },
    }


@app.post("/api/undo")
def api_undo(body: dict):
    """Rebuild the session without the last human+fly ply pair."""
    g = state["sessions"].get(body.get("game_id"))
    if g is None or len(g["history"]) < 1:
        raise HTTPException(409, "nothing to undo")
    hist = g["history"][:-1]  # drop the trailing ply
    human_side = "white" if not g["agent_is"] else "black"
    g2 = new_game(human_side)
    st = g2["st"]
    for col in hist:
        if st["done"] or not bool(legal_mask(st)[0, int(col)]):
            break
        apply(st, torch.tensor([col], device=state["dev"]))
        g2["history"].append(col)
    resp = to_ui(g2)
    if (int(st["ply"]) % 2 == 0) == g2["agent_is"] and not st["done"]:
        resp.update(fly_move(g2))
    resp["result"] = fly_result(g2)
    return resp


@app.get("/api/state/{game_id}")
def api_state(game_id: str):
    g = state["sessions"].get(game_id)
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


app.mount("/", StaticFiles(directory="/app/serve/static", html=True), name="static")
