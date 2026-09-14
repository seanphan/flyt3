# FLYT3 — trained fly-brain artifacts

Tic-tac-toe brains trained inside a simulated *Drosophila melanogaster* CNS
(MaleCNS v1.0: 166,700 neurons, 25,582,938 fixed synapses, Janelia + Google,
CC-BY). The wiring is never modified; only a linear readout of descending +
motor-neuron spike rates (REINFORCE) or KC→MBON synapses (dopamine RPE) was
trained. Play: the web app in the GitHub repo (seanphan/flyt3).

## Artifacts

| File | Brain | Algorithm | Games | Skill |
|---|---|---|---|---|
| `readout.npz` | Tic-tac-toe (serving) | Symmetric self-play REINFORCE + value baseline + entropy | 1,000,448 | 62% win vs random |
| `mb_weights.npz` | Tic-tac-toe (mushroom body) | Dopamine-gated KC→MBON depression, no gradients (Handler 2019, Bennett 2021) | 1,000,000 | 74% win vs random |
| `fire/readout.npz` | Aerial fire (FLAME) | Frozen circuit + 64-unit decoder, 3-class | — | 55.2% (chance 33%) |
| `fire_satellite/readout.npz` | Etkin satellite tiles | Frozen circuit + standardized 64-unit decoder, 5-class | — | **85.7%** (chance 20%) |

## Training data
- Tic-tac-toe: pure self-play (both flies)
- Aerial: FLAME drone imagery via alarmod/forest_fire (GPL-3)
- Satellite: Etkin et al. structure-damage tiles via
  kevincluo/structure_wildfire_damage_classification (apache-2.0)

## Honest limits
Neural dynamics, retinal mapping and dopamine gating are engineering choices,
not validated biology. The fire decoder reads coarse 99-patch retinal samples;
a 3° tilt broke the fly_ocr original and our tiles are no more forgiving.
The tic-tac-toe task is solved — 74% vs random is not superhuman.

## Credits
MaleCNS v1.0: HHMI Janelia + Google (CC-BY) · dopamine rule: Handler 2019,
Bennett 2021 via adonis-singh/TMNF-C (MIT) · recipe: jerryjliu/fly_ocr (MIT),
nftechie/doomfly (MIT) · listed in cobanov/awesome-fly (pending PR)
