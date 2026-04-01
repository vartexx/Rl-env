from __future__ import annotations

from pathlib import Path

from support_triage_env.environment import SupportTriageEnv
from support_triage_env.graders import grade_task
from support_triage_env.models import Action


REQUIRED_FILES = [
    "openenv.yaml",
    "Dockerfile",
    "inference.py",
    "README.md",
    "app.py",
]


def check_files() -> None:
    missing = [name for name in REQUIRED_FILES if not Path(name).exists()]
    if missing:
        raise RuntimeError(f"Missing required files: {missing}")


def check_tasks_and_graders() -> None:
    env = SupportTriageEnv()
    task_ids = env.task_ids()
    if len(task_ids) < 3:
        raise RuntimeError("Need at least 3 tasks")

    for task_id in task_ids:
        env.reset(task_id=task_id)
        # Intentional noop path ensures grader is not constant and remains bounded.
        env.step(Action(action_type="noop"))
        score = grade_task(env.state())["score"]
        if not (0.0 <= score <= 1.0):
            raise RuntimeError(f"Task {task_id} score out of range: {score}")


def check_api_shape() -> None:
    env = SupportTriageEnv(task_id="triage_easy_password_reset")
    obs = env.reset(task_id="triage_easy_password_reset")
    if obs.task_id != "triage_easy_password_reset":
        raise RuntimeError("reset() did not return expected task_id")

    result = env.step(Action(action_type="noop"))
    if result.reward.value >= 0.0:
        raise RuntimeError("noop should produce a penalty for loop-avoidance shaping")


def main() -> None:
    check_files()
    check_tasks_and_graders()
    check_api_shape()
    print("Pre-submission checks passed")


if __name__ == "__main__":
    main()
