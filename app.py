from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from support_triage_env.environment import SupportTriageEnv
from support_triage_env.models import Action, Observation, StepResult


app = FastAPI(title="OpenEnv Support Triage", version="1.0.0")
env = SupportTriageEnv()


class ResetRequest(BaseModel):
    task_id: str | None = None


@app.get("/")
def health() -> dict:
    return {"status": "ok", "env": "support-triage-openenv"}


@app.get("/tasks")
def list_tasks() -> dict:
    return {"task_ids": env.task_ids()}


@app.post("/reset", response_model=Observation)
def reset(payload: ResetRequest | None = None) -> Observation:
    try:
        task_id = payload.task_id if payload is not None else None
        return env.reset(task_id=task_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/step", response_model=StepResult)
def step(action: Action) -> StepResult:
    return env.step(action)


@app.get("/state")
def state() -> dict:
    current = env.state()
    return current.model_dump()
