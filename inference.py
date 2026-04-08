#!/usr/bin/env python3
"""
inference.py — OpenEnv Email Triage Baseline Inference Script
=============================================================

Reads all three required environment variables:
  API_BASE_URL  : LLM API endpoint     (default: https://router.huggingface.co/v1)
  MODEL_NAME    : Model identifier     (default: Qwen/Qwen2.5-72B-Instruct)
  HF_TOKEN      : Hugging Face token   (MANDATORY — no default)

Stdout format (per spec):
  [START] task=<name> env=email_triage_env model=<model>
  [STEP]  step=<n> action=<str> reward=<x.xx> done=<bool> error=<msg|null>
  [END]   success=<bool> steps=<n> score=<x.xx> rewards=<r1,r2,...>
"""

import json
import os
import sys
import textwrap
import time
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()  # This will load variables from the .env file

# ─── Environment variables ───────────────────────────────────────────────────
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    print("ERROR: HF_TOKEN environment variable is required.", file=sys.stderr)
    sys.exit(1)

# ─── Configuration ───────────────────────────────────────────────────────────
ENV_NAME       = "email_triage_env"
ENV_BASE_URL   = os.getenv("ENV_BASE_URL", "http://localhost:7860")  # server URL
MAX_STEPS      = 5
TEMPERATURE    = 0.3
MAX_TOKENS     = 800
SUCCESS_THRESHOLD = 0.5

TASKS = ["sort_inbox", "triage_email", "full_workflow"]

# ─── Stdout logging helpers ──────────────────────────────────────────────────

def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env={ENV_NAME} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    safe_action = action.replace("\n", " ").replace("\r", "")[:200]
    error_val = error.replace("\n", " ") if error else "null"
    print(
        f"[STEP] step={step} action={safe_action} "
        f"reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ─── HTTP helpers ────────────────────────────────────────────────────────────

def http_post(path: str, body: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{ENV_BASE_URL}{path}"
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def http_get(path: str) -> Dict[str, Any]:
    url = f"{ENV_BASE_URL}{path}"
    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.loads(resp.read())


# ─── System prompts ──────────────────────────────────────────────────────────

SORT_INBOX_SYSTEM = textwrap.dedent("""\
You are an expert email assistant. Your job is to sort a set of emails by priority.
Priority levels: urgent > normal > low
Return ONLY a valid JSON object in this exact format (no explanation, no markdown):
{"task": "sort_inbox", "payload": {"ordered_ids": ["eXXX", "eYYY", ...]}}
List ALL provided email IDs from highest to lowest priority.
""").strip()

TRIAGE_SYSTEM = textwrap.dedent("""\
You are an expert email triage specialist. Categorize each email into exactly one category:
- billing: invoices, payments, subscriptions, contracts
- support: technical issues, bugs, help requests, feature requests
- spam: unsolicited promotions, scams, marketing you didn't sign up for
- internal: company announcements, HR, team communications, CI/CD notifications

Return ONLY a valid JSON object (no explanation, no markdown):
{"task": "triage_email", "payload": {"assignments": {"eXXX": "billing", "eYYY": "support", ...}}}
Assign ALL provided email IDs.
""").strip()

FULL_WORKFLOW_SYSTEM = textwrap.dedent("""\
You are a senior executive assistant. For each email, do three things:
1. Assign a category: billing / support / spam / internal
2. Assign a priority: urgent / normal / low
3. Draft a professional reply (at least 30 words) if the email requires a reply.
   Use an empty string "" if no reply is needed (e.g., for notifications or spam).

Return ONLY a valid JSON object (no explanation, no markdown):
{
  "task": "full_workflow",
  "payload": {
    "assignments": {"eXXX": "billing", ...},
    "priorities": {"eXXX": "urgent", ...},
    "replies": {"eXXX": "Dear ..., Thank you for ...", ...}
  }
}
Include ALL provided email IDs in assignments, priorities, and replies.
""").strip()

SYSTEM_PROMPTS = {
    "sort_inbox": SORT_INBOX_SYSTEM,
    "triage_email": TRIAGE_SYSTEM,
    "full_workflow": FULL_WORKFLOW_SYSTEM,
}


# ─── LLM helpers ─────────────────────────────────────────────────────────────

def build_user_prompt(obs: Dict[str, Any]) -> str:
    emails = obs.get("emails", [])
    lines = ["Here are the emails to process:\n"]
    for email in emails:
        lines.append(
            f"--- Email ID: {email['id']} ---\n"
            f"From: {email['sender']}\n"
            f"Subject: {email['subject']}\n"
            f"Body: {email['body']}\n"
        )
    lines.append(f"\nInstructions: {obs.get('instructions', '')}")
    if obs.get("last_action_error"):
        lines.append(f"\nYour previous action had an error: {obs['last_action_error']}. Please correct it.")
    if obs.get("last_action_result"):
        lines.append(f"\nPrevious result: {obs['last_action_result']}")
    return "\n".join(lines)


def get_action(client: OpenAI, task: str, obs: Dict[str, Any]) -> Tuple[Optional[Dict], str]:
    """Call the LLM and parse a JSON action. Returns (parsed_dict, raw_text)."""
    user_prompt = build_user_prompt(obs)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPTS[task]},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        raw = (completion.choices[0].message.content or "").strip()
        # Extract JSON block if wrapped in markdown
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw.strip())
        return parsed, raw.replace("\n", " ")
    except json.JSONDecodeError as e:
        return None, f"JSON_PARSE_ERROR: {e}"
    except Exception as e:
        return None, f"LLM_ERROR: {e}"


# ─── Run one task episode ────────────────────────────────────────────────────

def run_task(client: OpenAI, task: str) -> Tuple[bool, int, float, List[float]]:
    """Run one episode for the given task. Returns (success, steps, score, rewards)."""
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task, model=MODEL_NAME)

    try:
        # Reset environment
        reset_resp = http_post("/reset", {"task": task})
        obs = reset_resp.get("observation", reset_resp)
        done = reset_resp.get("info", {}).get("done", False)

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            parsed_action, raw_text = get_action(client, task, obs)

            if parsed_action is None:
                # Malformed — send a minimal fallback action to avoid crashing
                fallback_payload = _fallback_payload(task, obs)
                step_resp = http_post("/step", {"task": task, "payload": fallback_payload})
                error_str = raw_text
            else:
                step_resp = http_post("/step", {
                    "task": parsed_action.get("task", task),
                    "payload": parsed_action.get("payload", {}),
                })
                error_str = step_resp.get("observation", {}).get("last_action_error")

            reward = float(step_resp.get("reward", 0.0))
            done   = bool(step_resp.get("done", False))
            obs    = step_resp.get("observation", {})

            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=raw_text[:150],
                reward=reward,
                done=done,
                error=error_str,
            )

            if done:
                break

        # Compute score from cumulative reward
        state_resp = http_get("/state")
        cumulative = float(state_resp.get("cumulative_reward", sum(rewards)))
        # Normalize: best possible grade is 1.0
        max_possible = 1.0
        score = min(max(cumulative / max_possible, 0.01), 0.99)
        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Task '{task}' error: {exc}", file=sys.stderr)

    finally:
        try:
            http_post("/close", {})
        except Exception:
            pass

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return success, steps_taken, score, rewards


def _fallback_payload(task: str, obs: Dict[str, Any]) -> Dict[str, Any]:
    """Return a minimal valid payload if LLM output can't be parsed."""
    email_ids = [e["id"] for e in obs.get("emails", [])]
    if task == "sort_inbox":
        return {"ordered_ids": email_ids}
    elif task == "triage_email":
        return {"assignments": {eid: "support" for eid in email_ids}}
    else:
        return {
            "assignments": {eid: "support" for eid in email_ids},
            "priorities":  {eid: "normal"  for eid in email_ids},
            "replies":     {eid: "Thank you for reaching out. We will look into this and respond shortly." for eid in email_ids},
        }


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    all_scores: List[float] = []
    for task in TASKS:
        success, steps, score, rewards = run_task(client, task)
        all_scores.append(score)
        print(f"[SUMMARY] task={task} score={score:.2f} success={str(success).lower()}", flush=True)
        time.sleep(1)  # brief pause between tasks

    avg_score = sum(all_scores) / len(all_scores)
    print(f"\n[FINAL] average_score={avg_score:.2f} tasks_completed={len(all_scores)}", flush=True)


if __name__ == "__main__":
    main()
