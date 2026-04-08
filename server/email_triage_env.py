"""
Core Email Triage Environment - Pydantic models and environment logic.
Implements the OpenEnv interface: reset(), step(), state(), close()
"""

from __future__ import annotations

import copy
import json
import time
import uuid
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# Pydantic Models
# ─────────────────────────────────────────────

class EmailItem(BaseModel):
    id: str
    subject: str
    sender: str
    body: str


class EmailTriageObservation(BaseModel):
    task: str
    step: int
    max_steps: int
    emails: List[EmailItem]
    instructions: str
    last_action_result: Optional[str] = None
    last_action_error: Optional[str] = None


class SortInboxAction(BaseModel):
    """Action for sort_inbox task: provide ordered list of email IDs."""
    ordered_ids: List[str] = Field(..., description="Email IDs sorted from highest to lowest priority")


class TriageEmailAction(BaseModel):
    """Action for triage_email task: assign category to each email."""
    assignments: Dict[str, str] = Field(
        ...,
        description="Map of email_id -> category (billing/support/spam/internal)"
    )


class FullWorkflowAction(BaseModel):
    """Action for full_workflow task: triage + draft a reply for each email."""
    assignments: Dict[str, str] = Field(
        ...,
        description="Map of email_id -> category"
    )
    priorities: Dict[str, str] = Field(
        ...,
        description="Map of email_id -> priority (urgent/normal/low)"
    )
    replies: Dict[str, str] = Field(
        ...,
        description="Map of email_id -> draft reply text (empty string if no reply needed)"
    )


class EmailTriageAction(BaseModel):
    """Top-level action wrapper — includes the task type and the actual action payload."""
    task: str = Field(..., description="Task name: sort_inbox | triage_email | full_workflow")
    payload: Dict[str, Any] = Field(..., description="The action payload (task-specific fields)")


class EmailTriageInfo(BaseModel):
    task: str
    step: int
    done: bool
    cumulative_reward: float
    episode_id: str
    error: Optional[str] = None


class StepResult(BaseModel):
    observation: EmailTriageObservation
    reward: float
    done: bool
    info: EmailTriageInfo


class ResetResult(BaseModel):
    observation: EmailTriageObservation
    info: EmailTriageInfo


# ─────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────

TASK_MAX_STEPS = {
    "sort_inbox": 3,
    "triage_email": 5,
    "full_workflow": 8,
}

TASK_INSTRUCTIONS = {
    "sort_inbox": (
        "You have an inbox with 5 emails. Sort them from HIGHEST to LOWEST priority "
        "(urgent > normal > low). Return an ordered list of email IDs.\n"
        "Action format: {\"task\": \"sort_inbox\", \"payload\": {\"ordered_ids\": [\"e001\", ...]}}"
    ),
    "triage_email": (
        "You have 10 emails. Assign EACH one to exactly one category: "
        "billing, support, spam, or internal. "
        "Action format: {\"task\": \"triage_email\", \"payload\": {\"assignments\": {\"e001\": \"billing\", ...}}}"
    ),
    "full_workflow": (
        "You have 5 emails. For each email: (1) assign a category (billing/support/spam/internal), "
        "(2) assign a priority (urgent/normal/low), and (3) draft a professional reply (or empty string if not needed). "
        "Action format: {\"task\": \"full_workflow\", \"payload\": {\"assignments\": {...}, \"priorities\": {...}, \"replies\": {...}}}"
    ),
}


class EmailTriageEnv:
    """
    Real-world email triage RL environment.
    Implements reset(), step(), state(), close().
    """

    def __init__(self, task: str = "sort_inbox"):
        from data.emails import (
            SORT_INBOX_EMAILS,
            TRIAGE_EMAILS,
            FULL_WORKFLOW_EMAILS,
            SORT_INBOX_GROUND_TRUTH_ORDER,
            TRIAGE_GROUND_TRUTH,
            FULL_WORKFLOW_GROUND_TRUTH,
            VALID_CATEGORIES,
            VALID_PRIORITIES,
        )

        if task not in TASK_MAX_STEPS:
            raise ValueError(f"Unknown task '{task}'. Choose from: {list(TASK_MAX_STEPS)}")

        self.task = task
        self.episode_id: str = str(uuid.uuid4())
        self.step_count: int = 0
        self.done: bool = False
        self.cumulative_reward: float = 0.0
        self.last_action_error: Optional[str] = None
        self.last_action_result: Optional[str] = None
        self.last_grade: float = 0.0
        self._previous_actions: List[str] = []

        # Load task-specific data
        if task == "sort_inbox":
            self._emails = SORT_INBOX_EMAILS
            self._ground_truth = SORT_INBOX_GROUND_TRUTH_ORDER
        elif task == "triage_email":
            self._emails = TRIAGE_EMAILS
            self._ground_truth = TRIAGE_GROUND_TRUTH
        else:
            self._emails = FULL_WORKFLOW_EMAILS
            self._ground_truth = FULL_WORKFLOW_GROUND_TRUTH

        self._valid_categories = VALID_CATEGORIES
        self._valid_priorities = VALID_PRIORITIES

    def _email_items(self) -> List[EmailItem]:
        return [
            EmailItem(id=e["id"], subject=e["subject"], sender=e["sender"], body=e["body"])
            for e in self._emails
        ]

    def _make_observation(self) -> EmailTriageObservation:
        return EmailTriageObservation(
            task=self.task,
            step=self.step_count,
            max_steps=TASK_MAX_STEPS[self.task],
            emails=self._email_items(),
            instructions=TASK_INSTRUCTIONS[self.task],
            last_action_result=self.last_action_result,
            last_action_error=self.last_action_error,
        )

    def _make_info(self) -> EmailTriageInfo:
        return EmailTriageInfo(
            task=self.task,
            step=self.step_count,
            done=self.done,
            cumulative_reward=self.cumulative_reward,
            episode_id=self.episode_id,
            error=self.last_action_error,
        )

    def reset(self) -> ResetResult:
        self.step_count = 0
        self.done = False
        self.cumulative_reward = 0.0
        self.last_action_error = None
        self.last_action_result = None
        self.last_grade = 0.0
        self._previous_actions = []
        self.episode_id = str(uuid.uuid4())
        return ResetResult(observation=self._make_observation(), info=self._make_info())

    def step(self, action: EmailTriageAction) -> StepResult:
        if self.done:
            raise RuntimeError("Episode is done. Call reset() first.")

        self.step_count += 1
        self.last_action_error = None
        self.last_action_result = None
        reward = 0.0

        # Loop detection penalty
        action_str = json.dumps(action.payload, sort_keys=True)
        if action_str in self._previous_actions:
            reward -= 0.1
            self.last_action_error = "Repeated action detected. Penalty applied."
            self.cumulative_reward += reward
            obs = self._make_observation()
            done = self.step_count >= TASK_MAX_STEPS[self.task]
            if done:
                self.done = True
            return StepResult(observation=obs, reward=reward, done=self.done, info=self._make_info())

        self._previous_actions.append(action_str)

        # Validate task matches
        if action.task != self.task:
            reward -= 0.05
            self.last_action_error = f"Action task '{action.task}' does not match env task '{self.task}'"
            self.cumulative_reward += reward
            obs = self._make_observation()
            done = self.step_count >= TASK_MAX_STEPS[self.task]
            if done:
                self.done = True
            return StepResult(observation=obs, reward=reward, done=self.done, info=self._make_info())

        # Grade the action
        grade = self.last_grade
        try:
            if self.task == "sort_inbox":
                grade, result_msg = self._grade_sort_inbox(action.payload)
            elif self.task == "triage_email":
                grade, result_msg = self._grade_triage_email(action.payload)
            else:
                grade, result_msg = self._grade_full_workflow(action.payload)
            self.last_action_result = result_msg
        except (KeyError, TypeError, ValueError) as exc:
            reward -= 0.05
            self.last_action_error = f"Malformed action payload: {exc}"

        # Strictly clamp grade between 0.01 and 0.99 (per OpenEnv Phase 2 requirements)
        grade = max(0.01, min(0.99, grade))

        # Reward is the marginal improvement in grade, plus any penalties
        grade_diff = grade - self.last_grade
        reward += grade_diff
        self.last_grade = grade

        self.cumulative_reward += reward
        done = self.step_count >= TASK_MAX_STEPS[self.task] or grade >= 0.99
        if done:
            self.done = True

        obs = self._make_observation()
        return StepResult(observation=obs, reward=round(reward, 4), done=self.done, info=self._make_info())

    def state(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "episode_id": self.episode_id,
            "step": self.step_count,
            "max_steps": TASK_MAX_STEPS[self.task],
            "done": self.done,
            "cumulative_reward": round(self.cumulative_reward, 4),
            "emails": [e["id"] for e in self._emails],
        }

    def close(self) -> None:
        self.done = True

    # ─── Graders ──────────────────────────────

    def _grade_sort_inbox(self, payload: Dict[str, Any]):
        """Score based on how well the agent sorted emails by priority."""
        ordered_ids: List[str] = payload.get("ordered_ids", [])
        ground_truth: List[str] = self._ground_truth

        if not ordered_ids:
            raise ValueError("'ordered_ids' is empty or missing")

        # Build a mapping from email_id to ground-truth rank
        gt_rank = {eid: i for i, eid in enumerate(ground_truth)}
        agent_rank = {eid: i for i, eid in enumerate(ordered_ids)}

        # Kendall tau-style: count concordant pairs
        n = len(ground_truth)
        concordant = 0
        total_pairs = 0
        for i in range(n):
            for j in range(i + 1, n):
                ei, ej = ground_truth[i], ground_truth[j]
                if ei in agent_rank and ej in agent_rank:
                    total_pairs += 1
                    if agent_rank[ei] < agent_rank[ej]:
                        concordant += 1

        score = concordant / total_pairs if total_pairs > 0 else 0.0
        msg = f"Sorting score: {score:.2f} ({concordant}/{total_pairs} concordant pairs)"
        return round(score, 4), msg

    def _grade_triage_email(self, payload: Dict[str, Any]):
        """Score based on category classification accuracy."""
        assignments: Dict[str, str] = payload.get("assignments", {})
        ground_truth: Dict[str, str] = self._ground_truth

        if not assignments:
            raise ValueError("'assignments' is empty or missing")

        # Validate categories
        invalid = [v for v in assignments.values() if v not in self._valid_categories]
        if invalid:
            raise ValueError(f"Invalid categories used: {invalid}")

        correct = sum(
            1 for eid, cat in assignments.items()
            if ground_truth.get(eid) == cat
        )
        total = len(ground_truth)
        score = correct / total
        msg = f"Triage accuracy: {score:.2f} ({correct}/{total} correct)"
        return round(score, 4), msg

    def _grade_full_workflow(self, payload: Dict[str, Any]):
        """Score = 0.4*category_accuracy + 0.3*priority_accuracy + 0.3*reply_quality."""
        assignments: Dict[str, str] = payload.get("assignments", {})
        priorities: Dict[str, str] = payload.get("priorities", {})
        replies: Dict[str, str] = payload.get("replies", {})
        ground_truth: Dict[str, Dict] = self._ground_truth

        if not assignments:
            raise ValueError("'assignments' is missing")

        # Category score
        cat_correct = sum(
            1 for eid, cat in assignments.items()
            if ground_truth.get(eid, {}).get("category") == cat
        )
        cat_score = cat_correct / len(ground_truth)

        # Priority score
        pri_correct = sum(
            1 for eid, pri in priorities.items()
            if ground_truth.get(eid, {}).get("priority") == pri
        )
        pri_score = pri_correct / len(ground_truth)

        # Reply quality score: reward non-empty replies for emails that need them
        reply_needed = {eid for eid, gt in ground_truth.items() if gt.get("requires_reply")}
        reply_score = 0.0
        if reply_needed:
            good_replies = sum(
                1 for eid in reply_needed
                if replies.get(eid, "").strip() and len(replies[eid].strip()) >= 20
            )
            reply_score = good_replies / len(reply_needed)

        total_score = 0.4 * cat_score + 0.3 * pri_score + 0.3 * reply_score
        msg = (
            f"Full workflow: category={cat_score:.2f}, priority={pri_score:.2f}, "
            f"reply_quality={reply_score:.2f} -> total={total_score:.2f}"
        )
        return round(total_score, 4), msg
