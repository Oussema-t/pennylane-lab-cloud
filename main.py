"""
PennyLane Lab — backend
=======================

A small, version-agnostic FastAPI server that runs quantum circuits with
PennyLane and returns everything the frontend needs to visualize a quantum
state: probabilities, the raw statevector, per-qubit Bloch vectors, and
expectation values.

Design notes
------------
* The circuit is described declaratively by the frontend (a list of gate
  operations). The server builds a QNode from that spec at request time, so
  adding new gates only means extending GATE_TABLE below.
* Bloch vectors and probabilities are computed directly from the statevector
  with NumPy (partial trace), rather than relying on newer PennyLane helpers
  like qml.density_matrix. This keeps the server working across a very wide
  range of PennyLane versions — which is what you want for an experimentation
  environment.

Run it
------
    pip install -r requirements.txt
    uvicorn app:app --reload --port 8000

Then open ../frontend/index.html in Chrome.
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="PennyLane Lab", version="1.0.0")

# The frontend is opened as a local file (file://) or from a different port,
# so we allow all origins in this development setup.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Gate table — extend this to add capabilities
# ---------------------------------------------------------------------------
# Each entry says how many wires the gate uses and whether it takes an angle.
GATE_TABLE = {
    "H": dict(wires=1, param=False, op=qml.Hadamard),
    "X": dict(wires=1, param=False, op=qml.PauliX),
    "Y": dict(wires=1, param=False, op=qml.PauliY),
    "Z": dict(wires=1, param=False, op=qml.PauliZ),
    "S": dict(wires=1, param=False, op=qml.S),
    "T": dict(wires=1, param=False, op=qml.T),
    "RX": dict(wires=1, param=True, op=qml.RX),
    "RY": dict(wires=1, param=True, op=qml.RY),
    "RZ": dict(wires=1, param=True, op=qml.RZ),
    "CNOT": dict(wires=2, param=False, op=qml.CNOT),
    "CZ": dict(wires=2, param=False, op=qml.CZ),
    "SWAP": dict(wires=2, param=False, op=qml.SWAP),
}


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------
class Operation(BaseModel):
    gate: str
    wires: List[int]
    param: Optional[float] = None  # radians, for rotation gates


class CircuitSpec(BaseModel):
    n_qubits: int = Field(ge=1, le=6)
    operations: List[Operation] = []
    device: str = "default.qubit"
    shots: Optional[int] = None  # None => exact statevector


class VQERequest(BaseModel):
    steps: int = Field(default=60, ge=1, le=400)
    learning_rate: float = Field(default=0.2, gt=0, le=2.0)


# ---------------------------------------------------------------------------
# Core simulation — importable + unit-testable on its own
# ---------------------------------------------------------------------------
def _bloch_vectors_from_state(state: np.ndarray, n: int):
    """Per-qubit Bloch vector (<X>, <Y>, <Z>) via partial trace of the state."""
    psi = np.asarray(state, dtype=complex).reshape([2] * n)
    vectors = []
    for q in range(n):
        # Bring qubit q to the front, flatten the rest, build 2x2 reduced rho.
        a = np.moveaxis(psi, q, 0).reshape(2, -1)
        rho = a @ a.conj().T  # 2 x 2 reduced density matrix
        bx = 2.0 * np.real(rho[0, 1])
        by = 2.0 * np.imag(rho[1, 0])
        bz = np.real(rho[0, 0] - rho[1, 1])
        purity = float(np.real(np.trace(rho @ rho)))
        vectors.append({"x": float(bx), "y": float(by), "z": float(bz),
                        "purity": purity})
    return vectors


def _memory_analysis(state: np.ndarray, n: int):
    """
    Compare how much memory THIS exact quantum state needs, three ways:

      * dense  — the full 2**n complex128 amplitudes (what default.qubit stores)
      * sparse — only the non-zero amplitudes (amplitude + integer basis index)
      * MPS    — a Matrix Product State, sized by ENTANGLEMENT (bond dimension),
                 not by amplitude sparsity. We compute the EXACT minimal bond
                 dimensions with a left-to-right SVD sweep (Schmidt ranks).

    This is what powers the "Memory & representation" panel in the UI and shows
    why a product state (dense in amplitudes, zero entanglement) compresses to
    almost nothing under MPS, while a highly entangling state does not.
    """
    complex_bytes = 16          # complex128 = 8 B real + 8 B imaginary
    sparse_entry_bytes = 24     # one amplitude (16 B) + its int64 index (8 B)
    zero_tol = 1e-12
    svd_tol = 1e-10

    vec = np.asarray(state, dtype=complex).reshape(-1)
    total = int(vec.size)       # 2**n
    dense_bytes = complex_bytes * total

    nonzero = int(np.count_nonzero(np.abs(vec) > zero_tol))
    sparse_bytes = nonzero * sparse_entry_bytes

    bonds: List[int] = []
    elements = 0
    if n <= 1:
        elements = total
    else:
        chi_left = 1
        residual = vec.reshape(1, -1)
        for _ in range(n - 1):
            mat = residual.reshape(chi_left * 2, -1)
            _, s, vh = np.linalg.svd(mat, full_matrices=False)
            cutoff = svd_tol * (s[0] if s.size else 1.0)
            chi = int(np.count_nonzero(s > cutoff)) or 1
            bonds.append(chi)
            elements += chi_left * 2 * chi          # site tensor (chi_left, 2, chi)
            residual = np.diag(s[:chi]) @ vh[:chi, :]
            chi_left = chi
        elements += chi_left * 2                    # final site tensor (chi_left, 2, 1)

    mps_bytes = elements * complex_bytes
    max_bond = max(bonds) if bonds else 1

    options = {"dense": dense_bytes, "sparse": sparse_bytes, "mps": mps_bytes}
    best = min(options, key=options.get)

    return {
        "complex_bytes": complex_bytes,
        "total_amplitudes": total,
        "nonzero_amplitudes": nonzero,
        "dense_bytes": dense_bytes,
        "sparse_bytes": sparse_bytes,
        "sparse_entry_bytes": sparse_entry_bytes,
        "mps_bytes": mps_bytes,
        "mps_max_bond": max_bond,
        "mps_bonds": bonds,
        "best": best,
    }


def run_circuit(spec: CircuitSpec):
    n = spec.n_qubits

    # Validate operations early with friendly errors.
    for op in spec.operations:
        if op.gate not in GATE_TABLE:
            raise ValueError(f"Unknown gate '{op.gate}'.")
        info = GATE_TABLE[op.gate]
        if len(op.wires) != info["wires"]:
            raise ValueError(
                f"Gate {op.gate} needs {info['wires']} wire(s), got {len(op.wires)}."
            )
        for w in op.wires:
            if not (0 <= w < n):
                raise ValueError(f"Wire {w} out of range for {n} qubit(s).")

    dev = qml.device(spec.device, wires=n)

    def apply_ops():
        for op in spec.operations:
            info = GATE_TABLE[op.gate]
            if info["param"]:
                info["op"](op.param or 0.0, wires=op.wires[0])
            elif info["wires"] == 1:
                info["op"](wires=op.wires[0])
            else:
                info["op"](wires=op.wires)

    # Exact statevector for visualization.
    @qml.qnode(dev)
    def state_node():
        apply_ops()
        return qml.state()

    state = np.asarray(state_node(), dtype=complex)
    probs = np.abs(state) ** 2

    amplitudes = [
        {"re": float(state[i].real),
         "im": float(state[i].imag),
         "prob": float(probs[i]),
         "phase": float(np.angle(state[i]))}
        for i in range(len(state))
    ]

    bloch = _bloch_vectors_from_state(state, n)

    result = {
        "n_qubits": n,
        "amplitudes": amplitudes,
        "bloch": bloch,
        "expvals_z": [b["z"] for b in bloch],
        "memory": _memory_analysis(state, n),
    }

    # Optional finite-shot sampling, to show measurement statistics.
    if spec.shots:
        def _probs():
            apply_ops()
            return qml.probs(wires=range(n))

        if hasattr(qml, "set_shots"):
            # Newer PennyLane: the transform is the recommended way.
            counts_node = qml.set_shots(qml.QNode(_probs, qml.device(spec.device, wires=n)),
                                        shots=spec.shots)
        else:
            # Older PennyLane: shots live on the device.
            counts_node = qml.QNode(_probs, qml.device(spec.device, wires=n, shots=spec.shots))

        sampled = np.asarray(counts_node(), dtype=float)
        result["sampled_probs"] = [float(p) for p in sampled]
        result["shots"] = spec.shots

    return result


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def frontend():
    """Serve the single-page frontend so the whole app is one server, one URL."""
    return FileResponse(os.path.join(BASE_DIR, "index.html"))


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "PennyLane Lab"}


@app.get("/api/info")
def info():
    """Tell the frontend what this server can do and which versions are live."""
    gates = {
        name: {"wires": g["wires"], "param": g["param"]}
        for name, g in GATE_TABLE.items()
    }
    return {
        "pennylane_version": qml.version(),
        "numpy_version": np.__version__,
        "gates": gates,
        "devices": ["default.qubit"],
        "max_qubits": 6,
    }


@app.post("/api/simulate")
def simulate(spec: CircuitSpec):
    try:
        return run_circuit(spec)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # pragma: no cover - surfaced to the UI
        raise HTTPException(status_code=500, detail=f"Simulation failed: {e}")


@app.post("/api/vqe")
def vqe(req: VQERequest):
    """
    A small variational eigensolver demo.

    Hamiltonian: H = -Z0 Z1 - 0.5 (X0 + X1) (a 2-qubit transverse-field Ising
    cell). We optimize a shallow ansatz to find its ground-state energy and
    return the convergence curve so the frontend can plot the descent.
    """
    n = 2
    coeffs = [-1.0, -0.5, -0.5]
    obs = [
        qml.PauliZ(0) @ qml.PauliZ(1),
        qml.PauliX(0),
        qml.PauliX(1),
    ]
    H = qml.Hamiltonian(coeffs, obs)

    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev)
    def cost(params):
        qml.RY(params[0], wires=0)
        qml.RY(params[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RY(params[2], wires=0)
        qml.RY(params[3], wires=1)
        return qml.expval(H)

    params = pnp.array([0.1, -0.1, 0.2, -0.2], requires_grad=True)
    opt = qml.GradientDescentOptimizer(stepsize=req.learning_rate)

    energies = [float(cost(params))]
    for _ in range(req.steps):
        params = opt.step(cost, params)
        energies.append(float(cost(params)))

    return {
        "hamiltonian": "H = -Z0 Z1 - 0.5 (X0 + X1)",
        "energies": energies,
        "final_energy": energies[-1],
        "final_params": [float(p) for p in params],
        "exact_ground_energy": float(_exact_ground_energy(coeffs, obs, n)),
    }


def _exact_ground_energy(coeffs, obs, n):
    """Diagonalize the Hamiltonian matrix for a reference ground-state energy."""
    matrix = qml.matrix(qml.Hamiltonian(coeffs, obs), wire_order=range(n))
    eigvals = np.linalg.eigvalsh(np.asarray(matrix))
    return float(np.min(eigvals))


if __name__ == "__main__":
    import uvicorn
    # 0.0.0.0 so cloud editors (CodeSandbox, etc.) can detect and proxy the port.
    # PORT env var is respected if the host sets one; otherwise default to 8000.
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
