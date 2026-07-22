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
    # Deliberately says nothing about credentials: this endpoint is public and
    # the repo is public, so "is the token armed?" is free reconnaissance.
    return {"ok": True, "service": "qrun-iqm", "status": "alive"}


@app.post("/iqm/run")
def iqm_run(req: IQMRunRequest, x_qrun_key: str | None = Header(default=None)):
    # Optional shared-key gate (same pattern as the transpiler).
    expected = os.environ.get("QRUN_IQM_KEY", "").strip()
    # FAIL CLOSED. The previous form was `if expected and ...`, which meant a
    # missing or mistyped QRUN_IQM_KEY made the whole service PUBLIC — and this
    # repo is public, so the endpoints are known. That mattered here more than
    # anywhere: this bridge submits jobs with the platform's IQM token, and the
    # monthly credit guard lives in QRUN's run-job.js, not here, so a direct
    # call bypasses it entirely. No key configured now means nobody gets in.
    if not expected or (x_qrun_key or "").strip() != expected:
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
    # FAIL CLOSED. The previous form was `if expected and ...`, which meant a
    # missing or mistyped QRUN_IQM_KEY made the whole service PUBLIC — and this
    # repo is public, so the endpoints are known. That mattered here more than
    # anywhere: this bridge submits jobs with the platform's IQM token, and the
    # monthly credit guard lives in QRUN's run-job.js, not here, so a direct
    # call bypasses it entirely. No key configured now means nobody gets in.
    if not expected or (x_qrun_key or "").strip() != expected:
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


# ═══════════════════════════════════════════════════════════════════════════
# Open Quantum (Quantum Rings) bridge — Rigetti + AQT real QPUs.
#
# Same service, separate code path from IQM: nothing above changes. Open Quantum
# speaks OpenQASM through its own SDK, which reads OPENQUANTUM_CLIENT_ID/SECRET
# from the environment. We return the SAME result shape as the IQM endpoints
# ({"counts": {...}}) so run-job.js and Verdict need no special casing.
#
# IonQ is intentionally NOT in ALLOWED_OQ_BACKENDS: one IonQ run (~40+ credits)
# would drain the whole free Spark balance. The IDE shows it greyed as "blocked";
# this is the second lock — even a crafted request can't spend on IonQ here.
# ═══════════════════════════════════════════════════════════════════════════

# Map QRUN's friendly device ids → Open Quantum backend_class_id.
OQ_BACKENDS = {
    "rigetti":      "rigetti:cepheus-1-108q",
    "rigetti:cepheus": "rigetti:cepheus-1-108q",
    "aqt":          "aqt:ibex-q1",
    "aqt:ibex":     "aqt:ibex-q1",
}
OQ_DEFAULT = "rigetti"          # cheapest per run (~0.6 credit)
MAX_OQ_SHOTS = 4096


class OQSubmitRequest(BaseModel):
    qasm: str = Field(max_length=MAX_QASM_CHARS)
    shots: int = 1024
    device: str = OQ_DEFAULT


class OQStatusRequest(BaseModel):
    job_id: str


# Cache the discovered organization id across requests — it doesn't change and the
# lookup costs a round-trip. Populated lazily on first submit.
_OQ_ORG_ID = None

def _oq_auth():
    # EXPLICIT client-credentials auth. The SDK's env auto-load path returns 401
    # for our key; passing ClientCredentials explicitly is the method that works
    # (verified via /oq/diag: explicit_auth -> ok, env_autoload -> 401).
    cid = os.environ.get("OPENQUANTUM_CLIENT_ID", "").strip()
    sec = os.environ.get("OPENQUANTUM_CLIENT_SECRET", "").strip()
    if not (cid and sec):
        raise HTTPException(status_code=503, detail="OPENQUANTUM_CLIENT_ID/SECRET not configured")
    from openquantum_sdk.auth import ClientCredentials, ClientCredentialsAuth
    return ClientCredentialsAuth(creds=ClientCredentials(client_id=cid, client_secret=sec))

def _oq_scheduler():
    # Imported inside the handler so a health check never fails if the SDK has an
    # import hiccup at boot.
    from openquantum_sdk.clients import SchedulerClient
    return SchedulerClient(auth=_oq_auth())

def _oq_org_id():
    # The SDK needs the organization_id on every job. Discover it once via the
    # ManagementClient (explicit auth), then cache.
    global _OQ_ORG_ID
    if _OQ_ORG_ID:
        return _OQ_ORG_ID
    from openquantum_sdk.clients import ManagementClient
    mgmt = ManagementClient(auth=_oq_auth())
    orgs = mgmt.list_user_organizations()
    _OQ_ORG_ID = orgs.organizations[0].id
    return _OQ_ORG_ID


@app.get("/oq/jobs")
def oq_jobs(x_qrun_key: str | None = Header(default=None)):
    # List recent Open Quantum jobs with their id + status, so we don't have to
    # hunt for a job_id by hand. Read-only, no credits spent.
    _check_key(x_qrun_key)
    try:
        scheduler = _oq_scheduler()
        org_id = _oq_org_id()
        try:
            page = scheduler.list_jobs(organization_id=org_id)
        except TypeError:
            page = scheduler.list_jobs()
        jobs = getattr(page, "jobs", None) or getattr(page, "items", None) or []
        out = []
        for j in jobs[:10]:
            out.append({
                "id": getattr(j, "id", None),
                "name": getattr(j, "name", None),
                "status": getattr(j, "status", None),
                "created": str(getattr(j, "created_at", "") or getattr(j, "created", "")),
            })
        return {"ok": True, "jobs": out}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


@app.get("/oq/inspect")
def oq_inspect(job_id: str, x_qrun_key: str | None = Header(default=None)):
    # Diagnostic: dump the RAW status + output of a specific job so we can see
    # exactly what Open Quantum returns (status string, result shape). Read-only,
    # no credits spent. Usage: /oq/inspect?job_id=XXXX
    _check_key(x_qrun_key)
    try:
        scheduler = _oq_scheduler()
        job = scheduler.get_job(job_id)
        out = {"job_id": job_id}
        out["raw_status"] = getattr(job, "status", None)
        out["has_output_url"] = bool(getattr(job, "output_data_url", None))
        # List the job object's attributes so we see what fields exist
        out["job_fields"] = [a for a in dir(job) if not a.startswith("_")][:40]
        # Try to download output and show its shape
        try:
            data = scheduler.download_job_output(job)
            out["output_type"] = type(data).__name__
            out["output_preview"] = str(data)[:500]
            out["counts_parsed"] = _oq_counts(data)
        except Exception as e:
            out["output_error"] = f"{type(e).__name__}: {e}"
        return out
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


@app.get("/oq/diag")
def oq_diag(x_qrun_key: str | None = Header(default=None)):
    # Free diagnostic: checks whether the Open Quantum credentials work and the
    # organization can be discovered — WITHOUT submitting a job (no credits spent).
    # Tells us exactly where a 401 comes from: missing env, bad creds, or org lookup.
    _check_key(x_qrun_key)
    have_id = bool(os.environ.get("OPENQUANTUM_CLIENT_ID", "").strip())
    have_secret = bool(os.environ.get("OPENQUANTUM_CLIENT_SECRET", "").strip())
    out = {"client_id_present": have_id, "client_secret_present": have_secret}
    if not (have_id and have_secret):
        out["ok"] = False
        out["stage"] = "env"
        out["error"] = "OPENQUANTUM_CLIENT_ID / OPENQUANTUM_CLIENT_SECRET not both set"
        return out
    # Show a masked fingerprint of the creds so we can confirm the RIGHT values
    # are loaded (length + first/last char) without ever exposing the secret.
    cid = os.environ.get("OPENQUANTUM_CLIENT_ID", "").strip()
    sec = os.environ.get("OPENQUANTUM_CLIENT_SECRET", "").strip()
    out["client_id_shape"] = f"len={len(cid)} starts={cid[:2]} ends={cid[-2:]}" if cid else "empty"
    out["client_secret_shape"] = f"len={len(sec)} starts={sec[:2]} ends={sec[-2:]}" if sec else "empty"

    # Try MULTIPLE auth methods and report which (if any) succeeds. This isolates
    # whether the problem is the auth METHOD (env auto-load vs explicit vs JSON
    # key file) or the credentials themselves.
    from openquantum_sdk.clients import ManagementClient
    attempts = {}

    def _try(label, make_mgmt):
        try:
            mgmt = make_mgmt()
            orgs = mgmt.list_user_organizations()
            names = [getattr(o, "name", "?") for o in orgs.organizations]
            attempts[label] = {"ok": True, "orgs": names}
            return names
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            body = None
            for attr in ("response", "body", "detail", "message"):
                v = getattr(e, attr, None)
                if v: body = str(v)[:200]; break
            attempts[label] = {"ok": False, "error": msg[:200], "body": body}
            return None

    # Method A: env auto-load (current approach)
    _try("env_autoload", lambda: ManagementClient())

    # Method B: explicit ClientCredentialsAuth
    def _explicit():
        from openquantum_sdk.auth import ClientCredentials, ClientCredentialsAuth
        auth = ClientCredentialsAuth(creds=ClientCredentials(client_id=cid, client_secret=sec))
        return ManagementClient(auth=auth)
    _try("explicit_auth", _explicit)

    # Method C: write a JSON key file and point OPENQUANTUM_SDK_KEY at it
    def _jsonkey():
        import json, tempfile
        p = os.path.join(tempfile.gettempdir(), "oq_sdk_key.json")
        with open(p, "w") as fh:
            json.dump({"client_id": cid, "client_secret": sec}, fh)
        os.environ["OPENQUANTUM_SDK_KEY"] = p
        return ManagementClient()
    _try("json_key_file", _jsonkey)

    out["attempts"] = attempts

    # Direct hit on the Keycloak token endpoint, bypassing the SDK entirely.
    # This shows the RAW OAuth response — the real reason (invalid_client,
    # unauthorized_client, invalid_grant…) instead of the SDK's opaque 401.
    try:
        import requests
        TOKEN_URL = "https://id.openquantum.com/realms/platform/protocol/openid-connect/token"
        # Try client_secret_post (creds in body) — the most common M2M form.
        r = requests.post(TOKEN_URL, data={
            "grant_type": "client_credentials",
            "client_id": cid,
            "client_secret": sec,
        }, headers={"Content-Type": "application/x-www-form-urlencoded"}, timeout=15)
        out["oauth_direct_status"] = r.status_code
        try:
            out["oauth_direct_body"] = r.json()
        except Exception:
            out["oauth_direct_body"] = r.text[:300]
        # Also try HTTP Basic auth (creds in Authorization header) as fallback.
        if r.status_code != 200:
            r2 = requests.post(TOKEN_URL, data={"grant_type": "client_credentials"},
                               auth=(cid, sec), timeout=15)
            out["oauth_basic_status"] = r2.status_code
            try:
                out["oauth_basic_body"] = r2.json()
            except Exception:
                out["oauth_basic_body"] = r2.text[:300]
    except Exception as e:
        out["oauth_direct_error"] = f"{type(e).__name__}: {e}"

    winner = next((k for k, v in attempts.items() if v.get("ok")), None)
    out["ok"] = bool(winner)
    out["winning_method"] = winner
    out["stage"] = "org_discovery" if winner else "auth_all_failed"
    return out


@app.post("/oq/submit")
def oq_submit(req: OQSubmitRequest, x_qrun_key: str | None = Header(default=None)):
    _check_key(x_qrun_key)
    backend_id = OQ_BACKENDS.get(req.device)
    if not backend_id:
        return {"ok": False, "error": f"device '{req.device}' not allowed (IonQ is blocked)"}
    shots = max(1, min(int(req.shots or 1024), MAX_OQ_SHOTS))
    try:
        from openquantum_sdk.models import JobPreparationCreate, JobCreate
        scheduler = _oq_scheduler()
        org_id = _oq_org_id()
        subcat = os.environ.get("OPENQUANTUM_SUBCATEGORY", "fin:port")

        # LOW-LEVEL non-blocking flow. submit_job() blocks until the job finishes
        # (it polls to completion) — on a real QPU queue that busts the HTTP
        # timeout and QRUN sees a false "error". Instead we upload → prepare →
        # create, which returns as soon as the job is QUEUED. cron-poll then
        # advances it to done, exactly like the IQM path.

        # 1) Upload the circuit
        upload_id = scheduler.upload_job_input(file_content=req.qasm.encode("utf-8"))

        # 2) Prepare (validates + prices the job)
        prep = scheduler.prepare_job(JobPreparationCreate(
            organization_id=org_id,
            backend_class_id=backend_id,
            name="QRUN job",
            upload_endpoint_id=upload_id,
            job_subcategory_id=subcat,
            shots=shots,
            configuration_data={},
        ))

        # 3) Poll preparation until the quote is ready (fast — seconds, not queue time)
        import time
        prep_result = None
        for _ in range(60):                     # up to ~60s for pricing
            prep_result = scheduler.get_preparation_result(prep.id)
            st = getattr(prep_result, "status", "")
            if st in ("Completed", "Failed"):
                break
            time.sleep(1)
        if not prep_result or getattr(prep_result, "status", "") != "Completed":
            msg = getattr(prep_result, "message", None) or "preparation did not complete"
            return {"ok": False, "error": f"prepare failed: {msg}"}

        # 4) Pick cheapest plan + priority from the quote
        quote = prep_result.quote
        cheapest_plan = min(quote, key=lambda p: p.price)
        cheapest_prio = min(cheapest_plan.queue_priorities, key=lambda q: q.price_increase)

        # 5) Create the job — this deducts credits and QUEUES it, then returns now
        job_resp = scheduler.create_job(JobCreate(
            organization_id=org_id,
            job_preparation_id=prep.id,
            execution_plan_id=cheapest_plan.execution_plan_id,
            queue_priority_id=cheapest_prio.queue_priority_id,
        ))
        job_id = getattr(job_resp, "id", None) or getattr(job_resp, "job_id", None) or str(job_resp)
        return {"ok": True, "job_id": str(job_id), "device": req.device}
    except HTTPException:
        raise
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


@app.post("/oq/status")
def oq_status(req: OQStatusRequest, x_qrun_key: str | None = Header(default=None)):
    _check_key(x_qrun_key)
    try:
        scheduler = _oq_scheduler()
        job = scheduler.get_job(req.job_id)
        raw = (getattr(job, "status", None) or "").upper()

        if raw in ("COMPLETED", "DONE", "SUCCEEDED"):
            out = scheduler.download_job_output(job)
            counts = _oq_counts(out)
            return {"ok": True, "status": "done", "counts": counts}
        if raw in ("FAILED", "ERROR", "CANCELLED", "CANCELED", "REJECTED"):
            return {"ok": True, "status": "failed", "counts": None, "detail": raw}
        # QUEUED / RUNNING / PREPARING → still pending. Surface any queue message
        # the provider attaches (position, "waiting for hardware", etc.) so the
        # UI can show real progress instead of a generic spinner.
        msg = getattr(job, "message", None)
        extra = getattr(job, "extra", None)
        return {"ok": True, "status": "pending", "phase": raw,
                "message": str(msg)[:160] if msg else None,
                "extra": str(extra)[:160] if extra else None,
                "counts": None}
    except HTTPException:
        raise
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def _oq_counts(out):
    # Normalize Open Quantum's result payload into { "0101": 512, ... }, the same
    # shape IQM returns. The SDK may hand back a counts dict directly, or a
    # results object we dig into — cover the common shapes defensively.
    if out is None:
        return None
    # already a plain dict of bitstring → count
    if isinstance(out, dict):
        if "counts" in out and isinstance(out["counts"], dict):
            return {str(k): int(v) for k, v in out["counts"].items()}
        # dict that IS the counts
        if all(isinstance(v, (int, float)) for v in out.values()) and out:
            return {str(k): int(v) for k, v in out.items()}
    # object with a .counts or .get_counts()
    c = getattr(out, "counts", None)
    if callable(getattr(out, "get_counts", None)):
        c = out.get_counts()
    if isinstance(c, dict):
        return {str(k): int(v) for k, v in c.items()}
    return None
