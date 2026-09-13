---
version: 1
slug: "serve-static-index-html"
primary_target: "serve/static/index.html"
related_targets: []
---

Scope: serve/static/index.html (the whole game surface). Visitor mode: Experience.

THESIS: The game IS a neon sign being fabricated and lit, one tube at a time —
refusing the flat-web tic-tac-toe grid: every stroke is glass tube, every move an
ignition, the win a continuous run of gas discharge.

OWN-WORLD: Near-black workshop (#06070a), cobalt gas-discharge blue (#5ac8ff) for
the fly's tubes, warm electrode-red (#ff4a3d) for the player's, faint glass gray
for unlit tube; sealed-electrode warmth at tube ends; printed timer digits and
grease-pencil labels as the only chrome. No gradients on glass — light comes
from the tube wall itself.

STORY: You challenged a fly brain to tic-tac-toe. Its neurons (25.6M real
synapses) visibly process your move — sight, integration, motor burst — then a
tube ignites where it decided. You will probably lose; the leaderboard remembers.

FIRST VIEWPORT: Full-bleed dark scene. Center-left: the # board on a dark shelf,
unlit glass, camera slightly off-axis. Upper right, floating behind: the cobalt
plasma point-cloud of the connectome. Top-left wordmark in tube lettering. Right
rail, thin: timer digits, test-strip move probabilities, your animal name and
the leaderboard.

FORM: Single three.js scene, orbit+dolly camera, raycast picking, drop-in piece
ignition with transformer flicker while the fly thinks; the wave of spikes
crosses the brain (optic → central → motor) from real per-tick data.

RISK: Additive glow can blow out to white mush; density and alpha tuned per
region, dark held everywhere else.
