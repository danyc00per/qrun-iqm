# QRUN — IQM Resonance service (separate from the IBM transpiler on purpose:
# qiskit-iqm has its own dependency stack, and we never want an IQM dependency
# bump to break the live IBM transpiler).
#
# What it does: receives an OpenQASM 3 circuit, transpiles it for an IQM
# backend (Garnet by default), submits it to IQM Resonance, and returns the
# measurement counts.
#
# ── Environment variables (set these on the Render service) ──────────────────
#   IQM_RESONANCE_TOKEN   your IQM Resonance API token (SECRET — Render only)
#   QRUN_IQM_KEY          optional shared key; if set, every call must send
#                         header  X-QRUN-KEY: <that value>  (401 otherwise).
#
# ── Safe testing ─────────────────────────────────────────────────────────────
#   Use device = "garnet:mock"  → compiles + runs the FULL pipeline WITHOUT
#   spending credits (random results, not a simulator). Switch to "garnet"
#   only once the whole chain is validated end-to-end.
#
# ── Backends (IQM Resonance) ─────────────────────────────────────────────────
#   garnet         real 20-qubit Crystal QPU (pay-as-you-go, spends credits)
#   garnet:mock    syntax checker / pipeline test (FREE, random results)
#   emerald        real 54-qubit QPU (if enabled on your account)
#   emerald:mock   free pipeline test for emerald

import os
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

app = FastAPI()

COCOS = "https://cocos.resonance.meetiqm.com"   # IQM Resonance job server
DEFAULT_DEVICE = "garnet:mock"                   # safe default — never spends credits
MAX_QASM_CHARS = 20_000
MAX_QUBITS = 20                                  # Garnet = 20 qubits
MAX_SHOTS = 20_000
ALLOWED_DEVICES = {
    "garnet", "garnet:mock", "garnet:timeslot",
    "emerald", "emerald:mock", "emerald:timeslot",
}


class IQMRunRequest(BaseModel):
    qasm: str = Field(max_length=MAX_QASM_CHARS)
    shots: int = 1024
    device: str = DEFAULT_DEVICE


@app.get("/")
def health():
    have_token = bool(os.environ.get("IQM_RESONANCE_TOKEN", "").strip())
    return {"ok": True, "service": "qrun-iqm", "status": "alive", "token_configured": have_token}


@app.post("/iqm/run")
def iqm_run(req: IQMRunRequest, x_qrun_key: str | None = Header(default=None)):
    # Optional shared-key gate (same pattern as the transpiler).
    expected = os.environ.get("QRUN_IQM_KEY", "").strip()
    if expected and (x_qrun_key or "").strip() != expected:
        raise HTTPException(status_code=401, detail="unauthorized")

    token = os.environ.get("IQM_RESONANCE_TOKEN", "").strip()
    if not token:
        return {"ok": False, "error": "IQM_RESONANCE_TOKEN not configured on the service"}

    device = req.device if req.device in ALLOWED_DEVICES else DEFAULT_DEVICE
    shots = max(1, min(int(req.shots or 1024), MAX_SHOTS))

    try:
        # Imported inside the handler so a health check never fails if the heavy
        # quantum stack has an import hiccup at boot.
        from qiskit import transpile
        from qiskit.qasm3 import loads
        from iqm.qiskit_iqm import IQMProvider

        circuit = loads(req.qasm)              # parse OpenQASM 3
        if circuit.num_qubits > MAX_QUBITS:
            return {"ok": False, "error": f"circuit too large ({circuit.num_qubits} qubits > {MAX_QUBITS})"}

        url = f"{COCOS}/{device}"
        backend = IQMProvider(url, token=token).get_backend()

        # Standard Qiskit transpile — the IQM plugin auto-adapts to native gates
        # (r, cz) and inserts MOVE gates for resonator architectures if needed.
        transpiled = transpile(circuit, backend=backend, optimization_level=1)

        job = backend.run(transpiled, shots=shots)   # submit to IQM
        result = job.result()                        # blocks until finished
        counts = result.get_counts()

        return {
            "ok": True,
            "device": device,
            "shots": shots,
            "num_qubits": circuit.num_qubits,
            "counts": counts,                        # { "00000": 512, "11111": 488, ... }
        }
    except HTTPException:
        raise
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
