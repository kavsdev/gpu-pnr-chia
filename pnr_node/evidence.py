"""Evidence bundles: the machine-generated measurement is the trusted artifact, the agent's words are not.

A bundle holds
  payload   request + measured result + toolchain + repro command     (hashed)
  sha256    hash of the canonical payload (HMAC-SHA256 if PNR_EVIDENCE_KEY is set)
  narrative free text from the agent, stored beside the payload and EXCLUDED from the hash
so an edited or fabricated number is detectable, and the headline PPA never depends on model prose.
This is tamper-evidence, not a security boundary: without a secret key anyone can recompute the hash.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import tempfile
from dataclasses import asdict
from typing import Any, Dict, Optional

from .contract import EvalRequest, EvalResult

TRUST_LABELS = {"proxy": "proxy-estimate", "route": "measured-routed"}


def _canonical(payload: Dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _digest(payload: Dict[str, Any]) -> str:
    key = os.environ.get("PNR_EVIDENCE_KEY")
    if key:
        return hmac.new(key.encode(), _canonical(payload), hashlib.sha256).hexdigest()
    return hashlib.sha256(_canonical(payload)).hexdigest()


def make_bundle(req: EvalRequest, res: EvalResult, repro_cmd: str = "", narrative: Optional[str] = None) -> Dict[str, Any]:
    payload = {
        "label": TRUST_LABELS.get(res.stage, "unlabelled") if res.ok else "failed",
        "request": asdict(req),
        "result": asdict(res),
        "repro_cmd": repro_cmd,
    }
    return {"payload": payload, "sha256": _digest(payload), "hmac": bool(os.environ.get("PNR_EVIDENCE_KEY")),
            "narrative": narrative or ""}


def verify_bundle(bundle: Dict[str, Any]) -> bool:
    try:
        return hmac.compare_digest(bundle["sha256"], _digest(bundle["payload"]))
    except (KeyError, TypeError):
        return False


def atomic_write_json(path: str, obj: Any) -> None:
    """Write via a temp file in the same directory and os.replace, so readers see the old or new file, never a torn one."""
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def write_bundle(directory: str, req: EvalRequest, res: EvalResult, repro_cmd: str = "",
                 narrative: Optional[str] = None) -> str:
    path = os.path.join(directory, f"{req.design}-{req.stage}-{req.key()}.json")
    atomic_write_json(path, make_bundle(req, res, repro_cmd, narrative))
    return path
