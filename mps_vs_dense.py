"""
mps_vs_dense.py
===============
Compare three ways to represent the SAME quantum state, and how much memory each
needs:

  1. Dense state vector   -> what PennyLane's `default.qubit` actually uses
  2. Sparse state vector  -> store only the NON-ZERO amplitudes
  3. Matrix Product State -> a tensor network that exploits LOW ENTANGLEMENT

The lesson this script makes concrete
--------------------------------------
"Sparsity" (few non-zero amplitudes) and "entanglement" (how correlated the
qubits are) are DIFFERENT things, and they call for different tricks:

  * Sparse storage only helps when most amplitudes are zero. But most quantum
    circuits quickly fill in the zeros (superposition + entanglement), so it
    rarely helps for real circuits.

  * MPS memory depends on ENTANGLEMENT (the "bond dimension"), not on how many
    amplitudes are non-zero. A product state like "H on every qubit" is fully
    DENSE in amplitudes (all 2^n non-zero) yet has bond dimension 1, so an MPS
    stores it in O(n) -- essentially nothing. That is why real "many-qubit"
    simulators use tensor networks, not zero-pruning.

  * A highly entangling circuit pushes the bond dimension toward 2^(n/2), so the
    MPS becomes as big as the dense vector: no method beats the worst case.

This runs with only `pennylane`, `numpy`, and `scipy`, which are already in your
venv -- no extra packages to install. The MPS sizes here are computed EXACTLY
(via SVD), so they are the true minimal bond dimensions for each state.

Run it:
    python3 mps_vs_dense.py
"""

import numpy as np
import pennylane as qml

COMPLEX_BYTES = 16        # complex128 = 8 bytes real + 8 bytes imaginary
SPARSE_ENTRY_BYTES = 24   # one amplitude (16 B) + its integer basis index (8 B);
                          # a real dict/hash-map is heavier still, so this is optimistic.
ZERO_TOL = 1e-12          # |amplitude| below this counts as zero
SVD_TOL = 1e-10           # singular values below SVD_TOL * largest count as zero


# --------------------------------------------------------------------------- #
#  Circuits to compare (each takes the qubit count n)                         #
# --------------------------------------------------------------------------- #
def product_layer(n):
    """H on every qubit -> uniform superposition. DENSE amplitudes, ZERO entanglement."""
    for w in range(n):
        qml.Hadamard(w)


def ghz(n):
    """GHZ state -> only 2 non-zero amplitudes, bond dimension 2 (mildly entangled)."""
    qml.Hadamard(0)
    for w in range(n - 1):
        qml.CNOT([w, w + 1])


def your_layer(n):
    """The H + RY/RZ(pi/2) layer from your app's circuit, generalised to n qubits."""
    for w in range(n):
        qml.Hadamard(w)
    for w in range(n):
        (qml.RY if w % 2 == 0 else qml.RZ)(np.pi / 2, w)


def highly_entangling(n):
    """Brickwork of random RY rotations + CNOTs -> near-maximal entanglement."""
    rng = np.random.default_rng(0)
    for d in range(n):                       # depth = n layers
        for w in range(n):
            qml.RY(float(rng.uniform(0, np.pi)), w)
        for w in range(d % 2, n - 1, 2):     # alternating CNOT pattern
            qml.CNOT([w, w + 1])


CIRCUITS = {
    "product (H on all)":  product_layer,
    "GHZ":                 ghz,
    "your H+RY/RZ layer":  your_layer,
    "highly entangling":   highly_entangling,
}


# --------------------------------------------------------------------------- #
#  Helpers                                                                    #
# --------------------------------------------------------------------------- #
def fmt_bytes(b):
    b = float(b)
    if b < 1024:
        return f"{int(b)} B"
    for unit in ("KB", "MB", "GB", "TB", "PB"):
        b /= 1024.0
        if b < 1024:
            return f"{b:.2f} {unit}" if b < 10 else f"{b:.1f} {unit}"
    return f"{b:.1f} EB"


def statevector(circuit, n):
    """Exact statevector from PennyLane's dense default.qubit device."""
    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev)
    def qnode():
        circuit(n)
        return qml.state()

    return np.asarray(qnode(), dtype=complex)


def mps_analysis(state, n):
    """
    Build the exact MPS of `state` by a left-to-right SVD sweep and return:
      bonds        : list of bond dimensions (length n-1)
      mps_elements : total complex numbers stored across all site tensors
    The bond dimension at each cut is the Schmidt rank = number of non-negligible
    singular values, i.e. the TRUE minimal MPS size for this state.
    """
    if n == 1:
        return [], state.size

    bonds = []
    elements = 0
    chi_left = 1
    residual = state.reshape(1, -1)          # shape: (chi_left, 2^n)

    for _ in range(n - 1):
        mat = residual.reshape(chi_left * 2, -1)
        u, s, vh = np.linalg.svd(mat, full_matrices=False)
        cutoff = SVD_TOL * (s[0] if s.size else 1.0)
        chi = int(np.count_nonzero(s > cutoff)) or 1
        bonds.append(chi)
        elements += chi_left * 2 * chi       # site tensor shape (chi_left, 2, chi)
        residual = np.diag(s[:chi]) @ vh[:chi, :]
        chi_left = chi

    elements += chi_left * 2 * 1             # final site tensor (chi_left, 2, 1)
    return bonds, elements


def analyse(circuit, n):
    state = statevector(circuit, n)
    dense_bytes = COMPLEX_BYTES * (2 ** n)

    nonzero = int(np.count_nonzero(np.abs(state) > ZERO_TOL))
    sparse_bytes = nonzero * SPARSE_ENTRY_BYTES

    bonds, mps_elems = mps_analysis(state, n)
    mps_bytes = mps_elems * COMPLEX_BYTES
    max_bond = max(bonds) if bonds else 1

    return {
        "n": n,
        "dense_bytes": dense_bytes,
        "nonzero": nonzero,
        "total_amps": 2 ** n,
        "sparse_bytes": sparse_bytes,
        "max_bond": max_bond,
        "mps_bytes": mps_bytes,
    }


def winner(r):
    best = min(r["dense_bytes"], r["sparse_bytes"], r["mps_bytes"])
    if best == r["mps_bytes"]:
        return "MPS"
    if best == r["sparse_bytes"]:
        return "sparse"
    return "dense"


# --------------------------------------------------------------------------- #
#  Report                                                                      #
# --------------------------------------------------------------------------- #
def main():
    n = 10  # qubit count for the main comparison (statevector is 2^10 = 1024 amps)
    print()
    print("=" * 92)
    print(f"  MEMORY OF ONE QUANTUM STATE, THREE WAYS   (n = {n} qubits, "
          f"dense statevector = {fmt_bytes(COMPLEX_BYTES * 2 ** n)})")
    print("=" * 92)
    header = (f"  {'circuit':<22}{'dense':>12}{'non-zero':>12}{'sparse':>12}"
              f"{'max bond':>10}{'MPS':>12}{'best':>9}")
    print(header)
    print("  " + "-" * 88)
    for name, circ in CIRCUITS.items():
        r = analyse(circ, n)
        print(f"  {name:<22}{fmt_bytes(r['dense_bytes']):>12}"
              f"{str(r['nonzero']) + '/' + str(r['total_amps']):>12}"
              f"{fmt_bytes(r['sparse_bytes']):>12}{r['max_bond']:>10}"
              f"{fmt_bytes(r['mps_bytes']):>12}{winner(r):>9}")
    print()
    print("  Read it like this:")
    print("   - 'product (H on all)' is DENSE (all amplitudes non-zero) so sparse storage")
    print("     does NOTHING, yet its bond dimension is 1 -> MPS crushes it. Sparsity and")
    print("     entanglement are different properties.")
    print("   - 'GHZ' is sparse (2 amplitudes) AND low-entanglement (bond 2): both tricks win.")
    print("   - 'highly entangling' fills the amplitudes and drives the bond dimension up, so")
    print("     MPS approaches the dense size: no method beats a genuinely hard circuit.")
    print()

    # Scaling: watch dense blow up exponentially while a low-entanglement state stays cheap.
    print("=" * 92)
    print("  SCALING WITH QUBIT COUNT  -  dense vs MPS")
    print("=" * 92)
    print(f"  {'n':>4}{'dense statevector':>22}{'product MPS':>16}"
          f"{'GHZ MPS':>14}{'entangling MPS':>18}")
    print("  " + "-" * 88)
    for nn in (4, 6, 8, 10, 12):
        dense = COMPLEX_BYTES * 2 ** nn
        prod = analyse(product_layer, nn)["mps_bytes"]
        gz = analyse(ghz, nn)["mps_bytes"]
        ent = analyse(highly_entangling, nn)["mps_bytes"]
        print(f"  {nn:>4}{fmt_bytes(dense):>22}{fmt_bytes(prod):>16}"
              f"{fmt_bytes(gz):>14}{fmt_bytes(ent):>18}")
    print()
    print("  The dense column doubles every qubit (16 x 2^n). The product/GHZ MPS columns")
    print("  grow only linearly -- that is the win that lets tensor-network simulators reach")
    print("  far more qubits, but ONLY while entanglement stays low.")
    print()
    print("  Production note: PennyLane ships a real MPS device, `default.tensor` (needs the")
    print("  `quimb` package: pip install quimb). Swap qml.device('default.qubit', ...) for")
    print("  qml.device('default.tensor', ...) to run circuits that never form the full")
    print("  statevector at all. This script computes exact MPS sizes with SVD so it stays")
    print("  dependency-free and shows you the underlying truth.")
    print()


if __name__ == "__main__":
    main()
