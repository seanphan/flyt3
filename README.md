# FLYC4 — Connect Four vs. a Fruit Fly Brain

A simulated *Drosophila melanogaster* CNS (MaleCNS v1.0, 166,700 neurons /
25.6M retained connections) plays Connect-4. The wiring is never modified;
the only trained component is a linear readout of descending + VNC motor
neuron spike rates, learned with self-play REINFORCE on a k3s GPU worker.
A web app lets a human play against the trained fly brain, with live
neural telemetry.

Method follows Alex Wormuth's [DOOMFLY](https://github.com/nftechie/doomfly)
(the fly connectome playing Doom, driven by photoreceptor input, motor-neuron
readout, PPL101 dopamine aversion) and
[flychess-hq.vercel.app](https://flychess-hq.vercel.app/) (the same recipe for
chess). This project reproduces the recipe for Connect-4 with real RL on the
readout.

## Layout

| Path | Contents |
| --- | --- |
| `flyc4/connectome.py` | MaleCNS v1.0 feather import -> CSR graph + task interface |
| `flyc4/env.py` | batched Connect-4 (int64 bitboards, 7 bits/column) |
| `flyc4/sim.py` | batched GPU LIF simulation (signed GABA synapses, homeostatic thresholds) |
| `flyc4/policy.py` | the trained linear readout + value baseline |
| `flyc4/train.py` | self-play REINFORCE loop (Job entrypoint) |
| `flyc4/eval.py` | batched evaluation vs random / minimax-depth-2 |
| `serve/app.py` | FastAPI play server (graph + readout on GPU) |
| `serve/static/` | the playable UI |
| `deploy/` | Dockerfile + k8s manifests (namespace, train Job, web Deployment) |

## Task interface (deterministic, documented)

- **Sensory in**: own pieces drive 6-cell R1-R6 photoreceptor patches (42x6);
  opponent pieces drive 5-cell R8 patches (42x5). R cells are spread evenly
  across the retina (sorted-bodyId linspace). Position -> current, not spikes.
- **Motor out**: top 512 descending neurons + top 512 VNC motor neurons by
  outgoing synapses. Spike counts over a 96-tick decision window -> linear
  head -> masked softmax over the 7 columns.
- **Dopamine**: terminal losses schedule an aversive current into the PPL101
  pair (doomfly-style). The gradient itself is REINFORCE + value baseline +
  entropy bonus.
- **Dynamics**: LIF with leak 0.85, signed synapse strengths
  (log1p(contact count) / sqrt(fan-out), GABA edges negative per the dataset's
  consensus neurotransmitter predictions) and per-neuron homeostatic
  threshold control (target rate 3%, gain 12). Engineering choices, not
  validated biology.

## Data

MaleCNS v1.0 flat connectome feathers (HHMI Janelia + Google, CC-BY) from
`storage.googleapis.com/flyem-male-cns/v1.0/...` — downloaded to
`/library/datasets/malecns_v1/` (canonical dataset storage). The processed
CSR cache lives beside it in `processed-flyc4/`.

## Run

Training (k3s Job, pinned to `k3s-w6` by user direction — normally this lane
selects `hardware=rtx3090-x1` labels instead):

```sh
kubectl apply -f deploy/namespace.yaml
kubectl apply -f deploy/train-smoke.yaml   # 3-batch pipeline check
kubectl logs -n flyc4 -f job/fly-smoke
kubectl apply -f deploy/train-job.yaml     # full 200x768-game run
```

Serve after training:

```sh
kubectl apply -f deploy/web.yaml           # NodePort 30180
```
