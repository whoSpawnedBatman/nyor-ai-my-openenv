"""
OpenEnv-compatible FastAPI server for Hospital Quotation Environment.
Exposes: POST /reset  POST /step  GET /state
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional

sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from env.environment import HospitalQuotationEnv, TASKS

app = FastAPI(
    title="Hospital Quotation OpenEnv",
    description="OpenEnv-compatible hospital medicine quotation benchmark.",
    version="1.0.0",
)

# Global env instance per session (single-request server model)
_envs: Dict[str, HospitalQuotationEnv] = {}


def _get_state_dict(env: HospitalQuotationEnv) -> Dict[str, Any]:
    s = env.state
    if s is None:
        return {}
    return {
        "task": env.task,
        "order": {
            "generic_name":  s.order.generic_name if s.order else None,
            "strength":      s.order.strength if s.order else None,
            "dosage_form":   s.order.dosage_form if s.order else None,
            "quantity":      s.order.quantity if s.order else None,
            "brand_policy":  s.order.brand_policy if s.order else None,
            "target_brand":  getattr(s.order, "target_brand", None) if s.order else None,
        },
        "selected_brand":    s.selected_brand,
        "selected_supplier": s.selected_supplier,
        "calculated_price":  s.calculated_price,
        "supplier_confirmed":s.supplier_confirmed,
        "action_history":    s.action_history,
        "score":             env.score(),
    }


# ── Request / Response models ─────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task: Optional[str] = "quotation"
    session_id: Optional[str] = "default"

class StepRequest(BaseModel):
    action: str
    session_id: Optional[str] = "default"


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["health"])
def root():
    return {"status": "ok", "service": "hospital-openenv", "tasks": list(TASKS.keys())}


@app.get("/health", tags=["health"])
def health():
    return {"status": "ok"}


@app.post("/reset", tags=["openenv"])
def reset(req: Optional[ResetRequest] = None):
    """Reset (or create) the environment and return the initial observation."""
    if req is None:
        req = ResetRequest()

    task       = (req.task or "quotation") if req.task in TASKS else "quotation"
    session_id = req.session_id or "default"

    env = HospitalQuotationEnv(task=task)
    _envs[session_id] = env
    state = env.reset()

    return JSONResponse(
        content={
            "observation": _get_state_dict(env),
            "done":   False,
            "reward": 0.0,
            "score":  0.0,
            "info":   {},
        }
    )


@app.post("/step", tags=["openenv"])
def step(req: StepRequest):
    """Apply an action and return (observation, reward, done, info, score)."""
    session_id = req.session_id or "default"
    env = _envs.get(session_id)
    if env is None:
        raise HTTPException(status_code=400, detail="Environment not initialised. Call /reset first.")

    state, reward, done, info = env.step(req.action)

    return JSONResponse(
        content={
            "observation": _get_state_dict(env),
            "reward": round(reward, 4),
            "done":   done,
            "score":  round(env.score(), 4),
            "info":   info,
        }
    )


@app.get("/state", tags=["openenv"])
def get_state(session_id: str = "default"):
    """Return the current state without advancing the environment."""
    env = _envs.get(session_id)
    if env is None:
        raise HTTPException(status_code=400, detail="No active session. Call /reset first.")
    return JSONResponse(content={"observation": _get_state_dict(env), "score": env.score()})


@app.get("/tasks", tags=["openenv"])
def list_tasks():
    """Enumerate available tasks and their metadata."""
    return JSONResponse(content={"tasks": TASKS})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
