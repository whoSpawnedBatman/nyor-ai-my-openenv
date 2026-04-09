"""
OpenEnv-compatible FastAPI server for Pharma B2B Quotation Environment.
Exposes: POST /reset  POST /step  GET /state
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional, List

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from env.environment import PharmaQuotationEnv, TASKS

app = FastAPI(
    title="Nyor AI - Pharma B2B Quotation Hub",
    description="OpenEnv-compatible pharmaceutical B2B quotation benchmark.",
    version="1.0.0",
)

# Global env instance per session (single-request server model)
_envs: Dict[str, PharmaQuotationEnv] = {}


# ── Strict OpenEnv Pydantic Models ────────────────────────────────────────────

class OrderModel(BaseModel):
    generic_name: Optional[str] = None
    strength: Optional[str] = None
    dosage_form: Optional[str] = None
    quantity: Optional[int] = None
    brand_policy: Optional[str] = None
    target_brand: Optional[str] = None

class Observation(BaseModel):
    task: str
    order: OrderModel
    selected_brand: Optional[str] = None
    selected_supplier: Optional[str] = None
    calculated_price: Optional[float] = None
    supplier_confirmed: bool = False
    action_history: List[str] = Field(default_factory=list)
    score: float = 0.0

class Action(BaseModel):
    action: str
    session_id: Optional[str] = "default"

class Reward(BaseModel):
    value: float

class StepResponse(BaseModel):
    observation: Observation
    reward: float
    done: bool
    score: float
    info: Dict[str, Any]

class ResetResponse(BaseModel):
    observation: Observation
    done: bool
    reward: float
    score: float
    info: Dict[str, Any]

class StateResponse(BaseModel):
    observation: Observation
    score: float

class ResetRequest(BaseModel):
    task: Optional[str] = "quotation"
    session_id: Optional[str] = "default"


def _get_observation(env: PharmaQuotationEnv) -> Observation:
    s = env.state
    if s is None:
        raise HTTPException(status_code=400, detail="State is null.")

    order_model = OrderModel(
        generic_name=s.order.generic_name if s.order else None,
        strength=s.order.strength if s.order else None,
        dosage_form=s.order.dosage_form if s.order else None,
        quantity=s.order.quantity if s.order else None,
        brand_policy=s.order.brand_policy if s.order else None,
        target_brand=getattr(s.order, "target_brand", None) if s.order else None,
    )

    return Observation(
        task=env.task,
        order=order_model,
        selected_brand=s.selected_brand,
        selected_supplier=s.selected_supplier,
        calculated_price=s.calculated_price,
        supplier_confirmed=s.supplier_confirmed,
        action_history=s.action_history,
        score=env.score(),
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["health"])
def root():
    return {
        "status": "ok",
        "service": "pharma-b2b-openenv",
        "tasks": list(TASKS.keys()),
        "message": "Nyor AI B2B Pharma Quotation Hub is active."
    }


@app.get("/health", tags=["health"])
def health():
    return {"status": "ok"}


@app.post("/reset", response_model=ResetResponse, tags=["openenv"])
def reset(req: Optional[ResetRequest] = None):
    """Reset (or create) the environment and return the initial observation."""
    if req is None:
        req = ResetRequest()

    task       = (req.task or "quotation") if req.task in TASKS else "quotation"
    session_id = req.session_id or "default"

    env = PharmaQuotationEnv(task=task)
    _envs[session_id] = env
    env.reset()

    return ResetResponse(
        observation=_get_observation(env),
        done=False,
        reward=0.0,
        score=0.0,
        info={},
    )


@app.post("/step", response_model=StepResponse, tags=["openenv"])
def step(req: Action):
    """Apply an action and return (observation, reward, done, info, score)."""
    session_id = req.session_id or "default"
    env = _envs.get(session_id)
    if env is None:
        raise HTTPException(status_code=400, detail="Environment not initialised. Call /reset first.")

    state, raw_reward, done, info = env.step(req.action)

    return StepResponse(
        observation=_get_observation(env),
        reward=round(raw_reward, 4),
        done=done,
        score=env.score(),
        info=info,
    )


@app.get("/state", response_model=StateResponse, tags=["openenv"])
def get_state(session_id: str = "default"):
    """Return the current state without advancing the environment."""
    env = _envs.get(session_id)
    if env is None:
        raise HTTPException(status_code=400, detail="No active session. Call /reset first.")
    
    return StateResponse(
        observation=_get_observation(env),
        score=env.score()
    )


@app.get("/tasks", tags=["openenv"])
def list_tasks():
    """Enumerate available tasks and their metadata."""
    return JSONResponse(content={"tasks": TASKS})


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
