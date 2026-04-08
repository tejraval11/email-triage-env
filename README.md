# 📧 Email Triage Environment — Meta OpenEnv Hackathon

> A real-world reinforcement learning environment where an AI agent acts as an intelligent email assistant.
> Built for the [Meta OpenEnv Hackathon](https://github.com/meta-pytorch/OpenEnv).

---

## 🌍 Environment Overview

**EmailTriageEnv** simulates a corporate inbox containing billing requests, customer support tickets,
internal communications, and spam. An RL agent must triage these emails through structured
JSON actions — earning rewards for correct classification and prioritization.

This is a **real-world task** (not a game or toy problem). Email triage is a daily knowledge-worker
activity with clear, deterministic success criteria, making it ideal for RL evaluation.

| Property | Value |
|---|---|
| Environment Name | `email_triage_env` |
| Tasks | `sort_inbox` · `triage_email` · `full_workflow` |
| Reward Range | `-0.1` to `+1.0` per step |
| Protocol | HTTP REST (FastAPI) |
| Port | `7860` |

---

## 📁 Project Structure

```
META AI HACKATHON/
├── inference.py              ← Root inference script (spec-required)
├── pyproject.toml            ← Build metadata (required by openenv validate)
├── uv.lock                   ← Package lock (required by openenv validate)
├── Dockerfile                ← Moved to root for simple deployment
├── openenv.yaml              ← Moved to root for simple deployment
├── requirements.txt          ← Moved to root for simple deployment
├── .env                      ← API keys and endpoint config (ignored in git)
├── demo_run.py               ← Live demo script (run this to see it work!)
├── README.md
└── server/
    ├── app.py                ← FastAPI app with strict `main()` entrypoint
    ├── email_triage_env.py   ← Core environment (Pydantic models + graders)
    └── data/
        └── emails.py         ← 15 labeled emails + ground-truth labels
```

---

## 📥 Observation Space

Every call to `/reset` or `/step` returns an `EmailTriageObservation`:

```json
{
  "task": "triage_email",
  "step": 1,
  "max_steps": 5,
  "emails": [
    {
      "id": "e003",
      "subject": "URGENT: Production server down since 2am",
      "sender": "ops@clientco.com",
      "body": "Our production environment has been unreachable since 2AM..."
    }
  ],
  "instructions": "Assign each email to: billing / support / spam / internal",
  "last_action_result": "Triage accuracy: 0.80 (8/10 correct)",
  "last_action_error": null
}
```

---

## 📤 Action Space

All actions are JSON with a `task` field and a `payload`. Submitted via `POST /step`.

### Task 1 — `sort_inbox`
```json
{
  "task": "sort_inbox",
  "payload": {
    "ordered_ids": ["e003", "e001", "e008", "e005", "e010"]
  }
}
```

### Task 2 — `triage_email`
```json
{
  "task": "triage_email",
  "payload": {
    "assignments": {
      "e001": "billing",
      "e003": "support",
      "e006": "spam",
      "e008": "internal"
    }
  }
}
```

### Task 3 — `full_workflow`
```json
{
  "task": "full_workflow",
  "payload": {
    "assignments": { "e001": "billing", "e003": "support" },
    "priorities":  { "e001": "urgent",  "e003": "urgent" },
    "replies": {
      "e001": "We sincerely apologize for the delay. Payment will be processed today.",
      "e003": "Our on-call team has been alerted and is investigating immediately."
    }
  }
}
```

---

## 📋 Tasks

### ① `sort_inbox` — 🟢 Easy
- **Inbox**: 5 emails with mixed priorities
- **Goal**: Return email IDs sorted `urgent → normal → low`
- **Grader**: Kendall-tau concordance over priority pairs
- **Max Steps**: 3 | **Baseline Score**: ~0.70

### ② `triage_email` — 🟡 Medium
- **Inbox**: 10 emails across 4 categories
- **Goal**: Assign each email to `billing` / `support` / `spam` / `internal`
- **Grader**: Exact-match accuracy (correct / 10)
- **Max Steps**: 5 | **Baseline Score**: ~0.60

### ③ `full_workflow` — 🔴 Hard
- **Inbox**: 5 emails all requiring a response
- **Goal**: Categorize + prioritize + draft professional replies
- **Grader**: Weighted composite (40% category + 30% priority + 30% reply quality)
- **Max Steps**: 8 | **Baseline Score**: ~0.45

---

## 🏆 Reward Function

Rewards are assigned incrementally at each `step()` call:

| Situation | Reward |
|---|---|
| Perfect sort/triage | `+1.00` |
| Partial accuracy | `+0.0` to `+0.99` |
| Duplicate action (loop) | `-0.10` penalty |
| Wrong task name | `-0.05` penalty |
| Malformed JSON payload | `-0.05` penalty |

---

## 🚀 Local Setup & Usage

### Without Docker

```bash
# 1. Install dependencies
cd server
pip install -r requirements.txt

# 2. Start the server
python -m uvicorn server:app --host 0.0.0.0 --port 7860

# 3. Run the full demo (in another terminal)
cd ..
python demo_run.py
```

> **Troubleshooting Note:** If you get a `ConnectionRefusedError: [WinError 10061]` or similar when running `demo_run.py`, it means the FastAPI server isn't running. Ensure you've kept the `uvicorn` server process running in its own dedicated terminal first.

### With Docker

```bash
docker build -t email-triage-env ./server
docker run -p 7860:7860 email-triage-env
```

### API Quick Test

```bash
# Reset
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "sort_inbox"}'

# Step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"task": "sort_inbox", "payload": {"ordered_ids": ["e003","e001","e008","e005","e010"]}}'

# State
curl http://localhost:7860/state

# Interactive Swagger docs
open http://localhost:7860/docs
```

### Run Inference Script

```bash
# Required
export HF_TOKEN=hf_your_token_here

# Optional (defaults shown)
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export ENV_BASE_URL=http://localhost:7860   # or your HF Space URL

python inference.py
```

**Expected stdout format:**
```
[START] task=sort_inbox env=email_triage_env model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={"task": "sort_inbox"...} reward=1.00 done=true error=null
[END] success=true steps=1 score=0.20 rewards=1.00
[START] task=triage_email env=email_triage_env model=Qwen/Qwen2.5-72B-Instruct
...
```

---

## 📊 API Reference

| Method | Endpoint | Body | Description |
|---|---|---|---|
| `GET` | `/` | — | Environment info |
| `POST` | `/reset` | `{"task": "sort_inbox"}` | Start new episode |
| `POST` | `/step` | `{"task": "...", "payload": {...}}` | Take an action |
| `GET` | `/state` | — | Current episode state |
| `POST` | `/close` | — | End episode |
| `GET` | `/health` | — | Health check |
| `GET` | `/docs` | — | Swagger UI |

---

## ☁️ Hugging Face Spaces Deployment

### Step 1 — Create a new Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Set **SDK = Docker**
3. Set **Hardware = CPU Basic** (2 vCPU, 16 GB RAM)
4. Name it e.g. `email-triage-env`
5. Add tag: `openenv`

### Step 2 — Push the server to the Space

```bash
# Clone your Space repo
git clone https://huggingface.co/spaces/YOUR_USERNAME/email-triage-env
cd email-triage-env

# Copy all files into the Space root
cp -r "C:/Users/hardi/Downloads/META AI HACKATHON/." .

# Push
git add .
git commit -m "Add Email Triage Environment"
git push
```

> **Important**: The Space root must contain `Dockerfile`, `openenv.yaml`, `requirements.txt`, `pyproject.toml`, `uv.lock`, `inference.py`, AND the `server/` directory containing `app.py`.

### Step 3 — Wait for it to be "Running"

Monitor the Build Logs in the Space UI. When the status shows ✅ **Running**, proceed.

### Step 4 — Validate

```bash
pip install openenv-core

# From the server/ directory (has openenv.yaml)
cd "C:/Users/hardi/Downloads/META AI HACKATHON/server"
openenv validate

# Ping the live Space
curl -X POST https://YOUR_USERNAME-email-triage-env.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "sort_inbox"}'
```

### Step 5 — Run inference against the live Space

```bash
export HF_TOKEN=hf_your_token
export ENV_BASE_URL=https://YOUR_USERNAME-email-triage-env.hf.space
python inference.py
```

### Step 6 — Submit

Go to the hackathon submission page and paste your Space URL:
```
https://YOUR_USERNAME-email-triage-env.hf.space
```
The validator will:
1. Ping `POST /reset` and expect HTTP 200 ✅
2. Run `docker build` ✅
3. Run `openenv validate` ✅

---

## 📊 Baseline Performance

Evaluated with `Qwen/Qwen2.5-72B-Instruct` via HuggingFace Router:

| Task | Difficulty | Expected Score |
|---|---|---|
| `sort_inbox` | 🟢 Easy | ~0.70 |
| `triage_email` | 🟡 Medium | ~0.60 |
| `full_workflow` | 🔴 Hard | ~0.45 |
| **Average** | — | **~0.58** |

---

## 🏷 Tags

`openenv` · `email-triage` · `reinforcement-learning` · `real-world-tasks` · `meta-hackathon`
