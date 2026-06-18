# PennyLane Lab

An interactive, browser-based quantum-circuit lab built on
[PennyLane](https://pennylane.ai/). You build a circuit by dragging gates onto a
grid, and a [FastAPI](https://fastapi.tiangolo.com/) backend runs the simulation
with PennyLane and streams the results back: measurement probabilities, per-qubit
Bloch vectors, ⟨Z⟩ expectation values, and a live readout of how much **memory**
the quantum state costs. There is also a Variational Quantum Eigensolver (VQE)
demo tab.

The whole thing is one Python server and one `index.html` page — no build step,
no JavaScript framework, no database.

## What it demonstrates

Beyond being a circuit playground, the lab is built to make a few real ideas in
quantum simulation tangible:

- **State-vector simulation costs memory exponentially.** A simulator stores
  `2ⁿ` complex amplitudes, so memory doubles with every qubit. The UI shows this
  live as you change the qubit count.
- **Sparsity ≠ compressibility.** The "Memory & representation" panel compares
  three ways to store the same state — dense, sparse (non-zero amplitudes only),
  and an MPS / tensor network (sized by *entanglement*, via exact SVD bond
  dimensions). It shows, for example, that a product state is dense in
  amplitudes yet compresses to almost nothing under MPS, while a highly
  entangling state compresses under neither.

## Features

- Drag a gate from the palette onto any qubit at any time step (or click a gate, then click a cell). Right-click a placed gate to remove it.
- 1-qubit gates: H, X, Y, Z, S, T, RX, RY, RZ (rotation angles are editable — click a placed RX/RY/RZ).
- 2-qubit gates: CNOT, CZ, SWAP — place on the control qubit, then click any target qubit in the same column (fully flexible control/target).
- Choose which qubits to measure; results marginalize to just those qubits.
- Live results: measurement probabilities, Bloch vectors, and ⟨Z⟩ expectation values.
- **Live memory readout**: a per-layer statevector-size row under the grid, plus a Memory badge that updates on every edit.
- **Memory & representation panel**: dense vs sparse vs MPS comparison computed from the real statevector on each run.
- Optional finite-shot sampling, and a Variational Quantum Eigensolver (VQE) demo tab.
- A standalone study script, [`mps_vs_dense.py`](./mps_vs_dense.py), that prints the dense/sparse/MPS memory of several circuits and how each scales with qubit count.

## Run it locally

The app is a standard FastAPI server. You need **Python 3.9 or newer**.

```bash
git clone https://github.com/Oussema-t/pennylane-lab-cloud.git
cd pennylane-lab-cloud

# create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# install pinned dependencies
pip install -r requirements.txt

# start the server
uvicorn main:app --reload --port 8000
```

Then open <http://localhost:8000> in your browser.

Optional: run the memory study script with `python3 mps_vs_dense.py`.

### Troubleshooting

- **`command not found: python` / `pip`** — use `python3` and `python3 -m pip`. On macOS you may need to install Python first (e.g. `brew install python`, or from <https://www.python.org/downloads/>).
- **`uvicorn: command not found`** — run it as a module: `python3 -m uvicorn main:app --reload --port 8000`.
- **`AttributeError: module 'autoray.autoray' has no attribute 'NumpyMimic'`** — you have a too-new `autoray`. The pinned `requirements.txt` fixes this; reinstall with `pip install -r requirements.txt`, or `pip install "autoray==0.6.12"`.
- **Port 8000 busy** — pick another, e.g. `--port 8080`, and open that.
- **Already cloned?** `git pull` to get the latest, then reinstall requirements.

## Make it public (so anyone can use it)

The app reads a `PORT` environment variable and binds `0.0.0.0`, so it deploys to
standard Python hosts with no code changes. The simulator is capped at 6 qubits,
so it stays light enough for free tiers.

### Option A — A temporary public link (quickest)

With the server already running on `localhost:8000`, expose it with a
[Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/):

```bash
# macOS: brew install cloudflared
cloudflared tunnel --url http://localhost:8000
```

It prints a public `https://….trycloudflare.com` URL that works while your
machine and the server stay running. Great for a quick demo; not a permanent
host. ([ngrok](https://ngrok.com/) does the same thing.)

### Option B — A permanent free deploy (Render)

1. Make sure this repo is pushed to GitHub (it is).
2. Go to <https://render.com>, sign in with GitHub, and choose **New → Web Service**, then pick this repository.
3. Set:
   - **Runtime:** Python 3
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Create the service. Render builds it and gives you a public `https://<name>.onrender.com` URL anyone can open.

Free Render services sleep after inactivity and take a few seconds to wake on the
first request — fine for a demo. [Railway](https://railway.app/) and
[Fly.io](https://fly.io/) work the same way with the same start command, and
[Hugging Face Spaces](https://huggingface.co/spaces) can host it as a Docker/FastAPI space.

## API

- `POST /api/simulate` — body `{ "n_qubits", "operations": [{ "gate", "wires": [...], "param" }], "device", "shots?" }`. Returns amplitudes, Bloch vectors, ⟨Z⟩ values, and a `memory` object (dense/sparse/MPS sizes and bond dimensions).
- `GET  /api/info` — PennyLane version, devices, supported gates, max qubits.
- `POST /api/vqe` — body `{ "steps", "learning_rate" }`.
- `GET  /api/health` — health check.

## Project layout

| File | Purpose |
| --- | --- |
| `main.py` | FastAPI backend: builds the circuit, runs PennyLane, returns results + memory analysis. |
| `index.html` | Single-page frontend: circuit builder, results, and memory panels. |
| `mps_vs_dense.py` | Standalone script comparing dense/sparse/MPS memory across circuits. |
| `requirements.txt` | Pinned dependencies. |

## Notes

CodeSandbox can be used to view the project, but its preview depends on the
CodeSandbox API/microVM being available. If the sandbox fails to start
("Unable to start the microVM" / "Initial connect to Pitcher"), check
<https://codesandboxstatus.statuspage.io/> and run the project locally with the
steps above in the meantime.
