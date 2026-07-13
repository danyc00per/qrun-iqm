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
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI()

# Allow browser calls (the QRUN app + local test pages). The real protection is
# the QRUN_IQM_KEY shared-key header, not the browser origin, so "*" is fine here.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

RESONANCE = "https://resonance.meetiqm.com"      # IQM Resonance server (iqm-client API)
DEFAULT_DEVICE = "garnet:mock"                   # safe default — never spends credits
MAX_QASM_CHARS = 20_000
MAX_QUBITS = 54                                  # Emerald = 54 (max across machines)
MAX_SHOTS = 20_000
ALLOWED_DEVICES = {
    "garnet", "garnet:mock", "garnet:timeslot",
    "emerald", "emerald:mock", "emerald:timeslot",
    # Sirius (Star topology + MOVE gates) intentionally NOT enabled — Crystal
    # machines (Garnet/Emerald) cover our needs with simpler, more reliable
    # transpilation. Add sirius here + in _lib/iqm.js IQM_DEVICES to re-enable.
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
        from qiskit.qasm3 import loads
        from iqm.qiskit_iqm import IQMProvider, transpile_to_IQM

        circuit = loads(req.qasm)              # parse OpenQASM 3
        if circuit.num_qubits > MAX_QUBITS:
            return {"ok": False, "error": f"circuit too large ({circuit.num_qubits} qubits > {MAX_QUBITS})"}

        # New iqm-client API: single Resonance URL + quantum_computer selects the
        # device (e.g. "garnet:mock"). transpile_to_IQM adapts to IQM native gates
        # (r, cz) and inserts MOVE gates for resonator architectures as needed.
        provider = IQMProvider(RESONANCE, quantum_computer=device, token=token)
        backend = provider.get_backend()
        transpiled = transpile_to_IQM(circuit, backend)

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


# ── Async path (submit → poll) — needed for the real Garnet queue ────────────
# /iqm/run above blocks until done: fine for mock and quick jobs, but the real
# device has a queue that can exceed HTTP timeouts. These two endpoints let the
# caller submit, get a job_id back immediately, then poll for the result.

class IQMSubmitRequest(BaseModel):
    qasm: str = Field(max_length=MAX_QASM_CHARS)
    shots: int = 1024
    device: str = DEFAULT_DEVICE


class IQMStatusRequest(BaseModel):
    job_id: str
    device: str = DEFAULT_DEVICE


def _check_key(x_qrun_key):
    expected = os.environ.get("QRUN_IQM_KEY", "").strip()
    if expected and (x_qrun_key or "").strip() != expected:
        raise HTTPException(status_code=401, detail="unauthorized")


def _backend(device, token):
    from iqm.qiskit_iqm import IQMProvider
    return IQMProvider(RESONANCE, quantum_computer=device, token=token).get_backend()


@app.post("/iqm/submit")
def iqm_submit(req: IQMSubmitRequest, x_qrun_key: str | None = Header(default=None)):
    _check_key(x_qrun_key)
    token = os.environ.get("IQM_RESONANCE_TOKEN", "").strip()
    if not token:
        return {"ok": False, "error": "IQM_RESONANCE_TOKEN not configured on the service"}
    device = req.device if req.device in ALLOWED_DEVICES else DEFAULT_DEVICE
    shots = max(1, min(int(req.shots or 1024), MAX_SHOTS))
    try:
        from qiskit.qasm3 import loads
        from iqm.qiskit_iqm import transpile_to_IQM
        circuit = loads(req.qasm)
        if circuit.num_qubits > MAX_QUBITS:
            return {"ok": False, "error": f"circuit too large ({circuit.num_qubits} qubits > {MAX_QUBITS})"}
        backend = _backend(device, token)
        transpiled = transpile_to_IQM(circuit, backend)
        job = backend.run(transpiled, shots=shots)   # submits, returns immediately (no blocking)
        return {"ok": True, "job_id": job.job_id(), "device": device, "shots": shots, "num_qubits": circuit.num_qubits}
    except HTTPException:
        raise
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


@app.post("/iqm/status")
def iqm_status(req: IQMStatusRequest, x_qrun_key: str | None = Header(default=None)):
    _check_key(x_qrun_key)
    token = os.environ.get("IQM_RESONANCE_TOKEN", "").strip()
    if not token:
        return {"ok": False, "error": "IQM_RESONANCE_TOKEN not configured on the service"}
    device = req.device if req.device in ALLOWED_DEVICES else DEFAULT_DEVICE
    try:
        backend = _backend(device, token)
        # Reconstruct the job handle from its id. Try the backend helper first,
        # fall back to constructing an IQMJob directly.
        job = None
        try:
            job = backend.retrieve_job(req.job_id)
        except Exception:
            from iqm.qiskit_iqm.iqm_job import IQMJob
            job = IQMJob(backend, job_id=req.job_id)

        st = job.status()
        name = getattr(st, "name", str(st)).upper()

        if name in ("DONE", "COMPLETED"):
            counts = job.result().get_counts()
            return {"ok": True, "status": "done", "counts": counts, "device": device}
        if name in ("ERROR", "FAILED", "CANCELLED"):
            return {"ok": True, "status": "failed", "counts": None, "device": device, "detail": name}
        # INITIALIZING / QUEUED / VALIDATING / RUNNING → still pending
        return {"ok": True, "status": "pending", "phase": name, "counts": None, "device": device}
    except HTTPException:
        raise
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
