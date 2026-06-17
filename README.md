# pennylane-lab-cloud

An interactive quantum-circuit lab built on [PennyLane](https://pennylane.ai/).
A FastAPI backend runs the simulations and serves a single-page `index.html`
frontend featuring a drag-and-drop circuit builder, flexible control/target
selection for two-qubit gates, live measurement probabilities, Bloch vectors,
expectation values, and a VQE demo.

## Features

- Drag a gate from the palette onto any qubit at any time step (or click a gate, then click a cell).
- 1-qubit gates: H, X, Y, Z, S, T, RX, RY, RZ (rotation angles are editable — click a placed RX/RY/RZ).
- 2-qubit gates: CNOT, CZ, SWAP — place on the control qubit, then click any target qubit in the same column (fully flexible control/target).
- Live results: measurement probabilities, Bloch vectors, and ⟨Z⟩ expectation values.
- Optional sampling (shots) and a Variational Quantum Eigensolver (VQE) demo tab.

## Run locally

The app is a standard FastAPI server, so you can run it on your own machine
without CodeSandbox:

```bash
git clone https://github.com/Oussema-t/pennylane-lab-cloud.git
cd pennylane-lab-cloud

# create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# install dependencies
pip install -r requirements.txt

# start the server
uvicorn main:app --reload --port 8000
```

Then open <http://localhost:8000> in your browser.

Tips:

- If `uvicorn` is not found, run `python -m uvicorn main:app --reload --port 8000`.
- If port 8000 is busy, use another port (e.g. `--port 8080`) and open that.
- Already cloned? Just `git pull` to get the latest `index.html`.

## API

- `POST /api/simulate` — body `{ "n_qubits", "operations": [{ "gate", "wires": [...], "param" }], "device", "shots?" }`
- `GET  /api/info` — PennyLane version, devices, supported gates
- `POST /api/vqe` — body `{ "steps", "learning_rate" }`
- `GET  /api/health` — health check

## Notes

CodeSandbox can be used to view the project, but its preview depends on the
CodeSandbox API/microVM being available. If the sandbox fails to start
("Unable to start the microVM" / "Initial connect to Pitcher"), check
<https://codesandboxstatus.statuspage.io/> — and run the project locally with
the steps above in the meantime.
# pennylane-lab-cloud
