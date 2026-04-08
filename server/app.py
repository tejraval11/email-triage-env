"""
FastAPI server exposing the EmailTriageEnv as an HTTP API.
Endpoints: POST /reset, POST /step, GET /state, POST /close
"""

import os
import sys

# Ensure the server directory is in the path
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict, Optional

from email_triage_env import (
    EmailTriageEnv,
    EmailTriageAction,
    ResetResult,
    StepResult,
)

app = FastAPI(
    title="Email Triage Environment",
    description="OpenEnv-compliant RL environment for real-world email triage tasks.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single shared environment instance (stateful per session)
_env: Optional[EmailTriageEnv] = None
_current_task: str = "sort_inbox"


def get_env() -> EmailTriageEnv:
    global _env
    if _env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    return _env


# ─────────────────────────────────────
# Request/Response models
# ─────────────────────────────────────

class ResetRequest(BaseModel):
    task: Optional[str] = "sort_inbox"


class StepRequest(BaseModel):
    task: str
    payload: Dict[str, Any]


# ─────────────────────────────────────
# Endpoints
# ─────────────────────────────────────

@app.get("/", tags=["health"])
async def root():
    return {
        "name": "email_triage_env",
        "version": "1.0.0",
        "tasks": ["sort_inbox", "triage_email", "full_workflow"],
        "status": "running",
    }


@app.post("/reset", tags=["env"], response_model=Dict[str, Any])
async def reset(request: ResetRequest = ResetRequest()):
    """
    Initialize or restart the environment.
    Pass ?task=sort_inbox|triage_email|full_workflow
    """
    global _env, _current_task
    task = request.task or "sort_inbox"
    if task not in ["sort_inbox", "triage_email", "full_workflow"]:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown task '{task}'. Valid: sort_inbox, triage_email, full_workflow"
        )
    _current_task = task
    _env = EmailTriageEnv(task=task)
    result: ResetResult = _env.reset()
    return result.model_dump()


@app.post("/step", tags=["env"], response_model=Dict[str, Any])
async def step(request: StepRequest):
    """
    Take a step in the environment with the given action.
    """
    env = get_env()
    action = EmailTriageAction(task=request.task, payload=request.payload)
    try:
        result: StepResult = env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result.model_dump()


@app.get("/state", tags=["env"], response_model=Dict[str, Any])
async def state():
    """
    Get the current state of the environment.
    """
    env = get_env()
    return env.state()


@app.post("/close", tags=["env"])
async def close():
    """
    Close/finalize the current episode.
    """
    global _env
    if _env is not None:
        _env.close()
        _env = None
    return {"status": "closed"}


@app.get("/health", tags=["health"])
async def health():
    return {"status": "ok"}


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
