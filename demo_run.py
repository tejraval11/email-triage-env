"""
demo_run.py - Full live demo of the Email Triage Environment
Run this to see the complete workflow across all 3 tasks.
Usage: python demo_run.py
"""

import json
import urllib.request
import urllib.error

BASE_URL = "http://localhost:7860"
SEPARATOR = "=" * 60


def post(path, body):
    url = f"{BASE_URL}{path}"
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read())


def get(path):
    with urllib.request.urlopen(f"{BASE_URL}{path}", timeout=10) as r:
        return json.loads(r.read())


def print_emails(emails):
    for e in emails:
        print(f"   [{e['id']}] From: {e['sender']}")
        print(f"          Subject: {e['subject']}")
        print()


def run_demo():
    print(SEPARATOR)
    print("  EMAIL TRIAGE ENVIRONMENT - LIVE DEMO")
    print(SEPARATOR)

    # ──────────────────────────────────────
    # Health check
    # ──────────────────────────────────────
    info = get("/")
    print(f"\nEnvironment: {info['name']} v{info['version']}")
    print(f"Available tasks: {info['tasks']}")

    # ══════════════════════════════════════
    # TASK 1: sort_inbox (EASY)
    # ══════════════════════════════════════
    print(f"\n{SEPARATOR}")
    print("  TASK 1: sort_inbox  [EASY]")
    print(SEPARATOR)

    reset = post("/reset", {"task": "sort_inbox"})
    obs = reset["observation"]
    print(f"\nStep {obs['step']}/{obs['max_steps']} | Inbox received with {len(obs['emails'])} emails:\n")
    print_emails(obs["emails"])
    print("Instructions:", obs["instructions"])

    print("\n>>> Agent Action: Sort emails from URGENT -> NORMAL -> LOW priority")
    action = {
        "task": "sort_inbox",
        "payload": {"ordered_ids": ["e003", "e001", "e008", "e005", "e010"]}
    }
    result = post("/step", action)
    print(f"    Reward: {result['reward']:.2f}")
    print(f"    Done:   {result['done']}")
    print(f"    Result: {result['observation']['last_action_result']}")
    state = get("/state")
    print(f"    Cumulative Reward: {state['cumulative_reward']:.2f}")
    post("/close", {})
    print(f"\n    FINAL SCORE: {result['reward']:.2f} / 1.00")

    # ══════════════════════════════════════
    # TASK 2: triage_email (MEDIUM)
    # ══════════════════════════════════════
    print(f"\n{SEPARATOR}")
    print("  TASK 2: triage_email  [MEDIUM]")
    print(SEPARATOR)

    reset2 = post("/reset", {"task": "triage_email"})
    obs2 = reset2["observation"]
    print(f"\nStep {obs2['step']}/{obs2['max_steps']} | Inbox received with {len(obs2['emails'])} emails:\n")
    print_emails(obs2["emails"])
    print("Instructions:", obs2["instructions"])

    # Simulate a strong (but not perfect) agent response
    print("\n>>> Agent Action: Assign categories to all 10 emails")
    assignments = {
        "e001": "billing",
        "e003": "support",
        "e006": "spam",
        "e008": "internal",
        "e010": "internal",
        "e002": "billing",
        "e004": "support",
        "e007": "spam",
        "e011": "billing",
        "e013": "support",
    }
    action2 = {"task": "triage_email", "payload": {"assignments": assignments}}
    result2 = post("/step", action2)
    print(f"    Reward: {result2['reward']:.2f}")
    print(f"    Done:   {result2['done']}")
    print(f"    Result: {result2['observation']['last_action_result']}")
    post("/close", {})
    print(f"\n    FINAL SCORE: {result2['reward']:.2f} / 1.00")

    # ══════════════════════════════════════
    # TASK 3: full_workflow (HARD)
    # ══════════════════════════════════════
    print(f"\n{SEPARATOR}")
    print("  TASK 3: full_workflow  [HARD]")
    print(SEPARATOR)

    reset3 = post("/reset", {"task": "full_workflow"})
    obs3 = reset3["observation"]
    print(f"\nStep {obs3['step']}/{obs3['max_steps']} | Inbox received with {len(obs3['emails'])} emails:\n")
    print_emails(obs3["emails"])
    print("Instructions:", obs3["instructions"])

    print("\n>>> Agent Action: Triage + Prioritize + Draft replies for all 5 emails")
    ids = [e["id"] for e in obs3["emails"]]
    action3 = {
        "task": "full_workflow",
        "payload": {
            "assignments": {
                "e001": "billing",
                "e003": "support",
                "e008": "internal",
                "e004": "support",
                "e011": "billing",
            },
            "priorities": {
                "e001": "urgent",
                "e003": "urgent",
                "e008": "normal",
                "e004": "normal",
                "e011": "urgent",
            },
            "replies": {
                "e001": "Dear Team, we sincerely apologize for the delay on Invoice #4821. Payment of $3,200 will be processed immediately today.",
                "e003": "Dear Client, we understand the severity of the production outage. Our on-call team has been immediately notified and is investigating with highest priority.",
                "e008": "Hi, confirmed - I will attend the Q2 planning meeting on Thursday at 3PM in Conference Room B with my department updates ready.",
                "e004": "Hello John, you can reset your password by visiting our login page and clicking 'Forgot Password'. A reset link will be emailed to you.",
                "e011": "Dear Procurement Team, thank you for the renewal proposal. We confirm the 3-year agreement at $120k/year. Contract will be sent for signatures by Friday.",
            },
        },
    }
    result3 = post("/step", action3)
    print(f"    Reward: {result3['reward']:.2f}")
    print(f"    Done:   {result3['done']}")
    print(f"    Result: {result3['observation']['last_action_result']}")
    post("/close", {})
    print(f"\n    FINAL SCORE: {result3['reward']:.2f} / 1.00")

    # ══════════════════════════════════════
    # PENALTY DEMO
    # ══════════════════════════════════════
    print(f"\n{SEPARATOR}")
    print("  PENALTY DEMO: Repeated action & invalid task detection")
    print(SEPARATOR)

    # Use triage_email (5 max steps) so episode doesn't end instantly
    post("/reset", {"task": "triage_email"})
    dupe_action = {
        "task": "triage_email",
        "payload": {"assignments": {"e001": "billing", "e003": "support", "e006": "spam"}}
    }
    r1 = post("/step", dupe_action)
    # Second identical action — triggers loop penalty
    r2 = post("/step", dupe_action)
    print(f"\n>>> First action (partial triage):  reward={r1['reward']:.2f}")
    print(f">>> Repeated identical action:      reward={r2['reward']:.2f}  (-0.10 penalty)")
    print(f"    Error msg: {r2['observation']['last_action_error']}")

    # Wrong task name penalty
    wrong_action = {"task": "sort_inbox", "payload": {"ordered_ids": ["e001"]}}
    r3 = post("/step", wrong_action)
    print(f"\n>>> Wrong task name in action:      reward={r3['reward']:.2f}  (-0.05 penalty)")
    print(f"    Error msg: {r3['observation']['last_action_error']}")
    post("/close", {})

    # ══════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════
    print(f"\n{SEPARATOR}")
    print("  DEMO SUMMARY")
    print(SEPARATOR)
    print(f"\n  Task 1 sort_inbox    [EASY]   -> Score: {result['reward']:.2f}")
    print(f"  Task 2 triage_email  [MEDIUM] -> Score: {result2['reward']:.2f}")
    print(f"  Task 3 full_workflow [HARD]   -> Score: {result3['reward']:.2f}")
    avg = (result['reward'] + result2['reward'] + result3['reward']) / 3
    print(f"\n  Average Score: {avg:.2f}")
    print(f"\n  Server docs:  http://localhost:7860/docs")
    print(f"  Health check: http://localhost:7860/health")
    print(f"\n{SEPARATOR}")
    print("  ALL TASKS COMPLETED SUCCESSFULLY!")
    print(SEPARATOR)


if __name__ == "__main__":
    run_demo()
