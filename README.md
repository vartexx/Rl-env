---
title: OpenEnv Support Triage Environment
sdk: docker
app_port: 7860
tags:
  - openenv
  - reinforcement-learning
  - customer-support
---

# OpenEnv Support Triage Environment

A real-world reinforcement learning environment that simulates **customer support triage operations** for SaaS products. The agent must classify incoming tickets, assign them to the right team, communicate with users, and resolve or escalate issues under SLA pressure.

This environment is designed for practical evaluation and training of agentic systems in workflows that humans do every day.

## Why this environment is useful

Support triage is a high-impact operational problem:

- Incorrect routing causes SLA breaches and customer churn.
- Abuse reports require strict escalation behavior.
- Backlog prioritization under constraints is hard for current LLM agents.

This environment captures those dynamics with deterministic grading and dense reward shaping.

## OpenEnv Interface

The environment implements the standard API:

- `reset(task_id: Optional[str]) -> Observation`
- `step(action: Action) -> StepResult`
- `state() -> EnvState`

Typed Pydantic models are implemented for:

- `Action`
- `Observation`
- `Reward`

Manifest metadata is defined in `openenv.yaml`.

## Action Space

`Action` supports the following `action_type` values:

- `classify`: set `predicted_category` and `predicted_priority`
- `assign`: set `team`
- `respond`: set `response_template`
- `resolve`: set `resolution_code`
- `escalate`: set `escalation_reason`
- `noop`: take no operation

Categories:

- `billing`
- `bug`
- `account`
- `sales`
- `abuse`

## Observation Space

Each observation includes:

- task metadata (`task_id`, `instruction`)
- step metadata (`step_count`, `max_steps`)
- queue metrics (`pending_tickets`, `resolved_tickets`, `escalated_tickets`)
- full ticket snapshots (subject, message, SLA hours left, status, agent predictions, assignment)

## Tasks and Difficulty

1. `triage_easy_password_reset` (easy)
Single account ticket. Objective: classify, assign to account ops, respond, and resolve.

2. `triage_medium_mixed_queue` (medium)
Three-ticket queue (abuse, billing, bug). Objective: correctly escalate abuse and resolve others under tighter SLA.

3. `triage_hard_backlog_optimization` (hard)
Six-ticket backlog with severe SLA and priority tradeoffs. Objective: optimize routing and outcomes while minimizing inefficient behavior.

Each task has a deterministic grader returning score in `[0.0, 1.0]`.

## Reward Design

Per-step reward provides meaningful trajectory signal:

- Positive for correct classification and priority estimation
- Positive for correct team assignment
- Positive for useful responses
- Positive for proper terminal actions (resolve non-abuse, escalate abuse)
- Penalties for invalid actions, no-op loops, wrong terminal actions, and SLA misses
- Small completion bonus when all tickets reach expected terminal state

This encourages strategy quality beyond sparse end-of-episode success.

## Baseline Inference

The baseline script is `inference.py` in repository root (as required).

It uses the OpenAI client and reads:

- `OPENAI_API_KEY`
- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN` (for deployment workflows; not required by local inference logic)

Run:

```bash
python inference.py
```

Output is JSON with per-task scores and macro average.

The script now emits evaluator-compatible structured stdout lines in strict order:

- `[START] task=<task_name> env=<benchmark> model=<model_name>`
- `[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>`
- `[END] success=<true|false> steps=<n> rewards=<r1,r2,...,rn>`

This format is intended for automated parsing during benchmark evaluation.

### Baseline Scores

Latest run (`MODEL_NAME=openai/gpt-4o-mini`) produced:

- Macro average: `0.9835`
- `triage_easy_password_reset`: `1.0000`
- `triage_medium_mixed_queue`: `0.9714`
- `triage_hard_backlog_optimization`: `0.9792`

Saved artifact: `baseline_scores.json`

## Setup

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

Health check:

```bash
curl http://localhost:7860/
```

## Docker

Build and run:

```bash
docker build -t support-triage-openenv .
docker run --rm -p 7860:7860 support-triage-openenv
```

## Hugging Face Space Deployment

1. Create a new **Docker Space**.
2. Push this repository as the Space content.
3. Add variables/secrets in Space settings:
   - `OPENAI_API_KEY`
   - `API_BASE_URL`
   - `MODEL_NAME`
   - `HF_TOKEN`
4. Ensure Space has the tag `openenv`.

The app serves endpoints:

- `GET /`
- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /tasks`

## Notes on Reproducibility

- Task definitions are deterministic.
- Graders are deterministic.
- Baseline uses fixed sampling controls (`temperature=0`, fixed `seed`).

## Quick Validation Checklist

- Docker image builds successfully
- `/reset`, `/step`, `/state` endpoints respond
- `openenv.yaml` present with model + task metadata
- `inference.py` runs and prints all three task scores

Run local pre-submission checks:

```bash
python validate_submission.py
```
