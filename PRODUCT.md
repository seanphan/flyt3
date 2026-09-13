# Product

## Name
FLYC4 — Tic-Tac-Toe vs. a Fruit Fly Brain

## Users
Anyone on the team or the internet who opens the link and wants to test themselves
against a simulated fruit-fly brain (MaleCNS v1.0 connectome, 166,700 neurons).
Casual players; neuroscience-curious visitors; teammates demoing the k3s cluster.
Players get auto-assigned animal names (Google-Sheets style) and compete on a
public leaderboard. (Assumption, from the brief: single-game sessions, no accounts.)

## Product Purpose
Let a human play tic-tac-toe against a real simulated fly connectome — 166,700
neurons and 25.6M fixed synapses rendered live, where only a linear readout of
the fly's motor neurons was trained by 1M+ games of self-play RL. The product's
honesty is the hook: the wiring is never modified, nothing about the game is
stored in the brain, and every loss the fly suffers fires a dopamine pulse in
training — doomfly-style.

## Job
The visitor plays a game, watches the brain spike in 3D, saves their result to
the leaderboard, and comes back to beat their animal name's record.

## Principles
The brain is the spectacle; the game is the doorway. Scientific honesty over
hype — label what is model vs. biology. Motion means something: spikes are real
data, never decoration.

## Constraints
Runs from a single static page + small API on a k3s cluster; must render on
laptop GPUs; three.js vendored (no CDN dependency); the fly thinks server-side
(~0.1-3s).
