# 🚀 Hackathon Submission Guide — Email Triage Environment

## ✅ What's Already Done (Nothing to touch)

| Item | Status |
|---|---|
| `inference.py` in project root | ✅ Done |
| FastAPI server (`server/server.py`) | ✅ Done |
| Environment logic (`server/email_triage_env.py`) | ✅ Done |
| Email dataset + ground truth (`server/data/`) | ✅ Done |
| `server/openenv.yaml` | ✅ Done |
| `server/Dockerfile` | ✅ Done |
| `server/requirements.txt` | ✅ Done |
| `README.md` | ✅ Done |
| Local server tested & verified (scores: 1.00/1.00/1.00) | ✅ Done |

---

## 🔴 What You Still Need to Do (5 Steps)

---

### STEP 1 — Create a Hugging Face Account (if you don't have one)

Go to → **https://huggingface.co/join**

---

### STEP 2 — Get Your HF Token

1. Go to → **https://huggingface.co/settings/tokens**
2. Click **New token**
3. Name it anything, select **Write** access
4. Copy the token — looks like `hf_xxxxxxxxxxxxxxxxxxxxxx`
5. Save it — you'll need it in Steps 4 and 5

---

### STEP 3 — Create a New Space

1. Go to → **https://huggingface.co/new-space**
2. Fill in:
   - **Owner**: your username
   - **Space name**: `email-triage-env`
   - **SDK**: select **Docker**
   - **Hardware**: CPU Basic (free tier)
3. Scroll down, add tag: `openenv`
4. Click **Create Space**

> Your Space URL will be: `https://huggingface.co/spaces/YOUR_USERNAME/email-triage-env`
> Your API URL will be: `https://YOUR_USERNAME-email-triage-env.hf.space`

---

### STEP 4 — Push the Environment to the Space

Open PowerShell and run these commands one by one:

```powershell
# 1. Install git-lfs if you haven't already
git lfs install

# 2. Clone your newly created Space
git clone https://huggingface.co/spaces/YOUR_USERNAME/email-triage-env
cd email-triage-env

# 3. Copy ALL files from the server/ folder into this repo root
Copy-Item "C:\Users\hardi\Downloads\META AI HACKATHON\server\*" -Destination . -Recurse -Force

# 4. Commit and push
git add .
git commit -m "Add Email Triage Environment - OpenEnv Hackathon"
git push
```

> **When prompted for credentials:**
> - Username: your HuggingFace username
> - Password: paste your HF token from Step 2

---

### STEP 5 — Wait for Build & Verify

1. Go to your Space page: `https://huggingface.co/spaces/YOUR_USERNAME/email-triage-env`
2. Click the **"App"** tab and watch the build logs
3. Wait until status shows ✅ **Running** (takes 2–5 minutes)
4. Test it is live:

```powershell
# Replace YOUR_USERNAME below
$url = "https://YOUR_USERNAME-email-triage-env.hf.space"
Invoke-WebRequest -Uri "$url/reset" -Method POST -ContentType "application/json" -Body '{"task":"sort_inbox"}'
```

You should get a JSON response with emails. If yes — your Space is live! ✅

---

### STEP 6 — Run inference.py Against Live Space

```powershell
# Set your credentials
$env:HF_TOKEN     = "hf_your_token_here"
$env:API_BASE_URL = "https://router.huggingface.co/v1"
$env:MODEL_NAME   = "Qwen/Qwen2.5-72B-Instruct"
$env:ENV_BASE_URL = "https://YOUR_USERNAME-email-triage-env.hf.space"

# Run from the project root
cd "C:\Users\hardi\Downloads\META AI HACKATHON"
python inference.py
```

You should see output like:
```
[START] task=sort_inbox env=email_triage_env model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=... reward=0.80 done=true error=null
[END] success=true steps=1 score=0.16 rewards=0.80
...
```

---

### STEP 7 — Submit

Go to the hackathon submission form and paste:
```
https://YOUR_USERNAME-email-triage-env.hf.space
```

The validator will automatically check:
- ✅ Space is live and `/reset` returns HTTP 200
- ✅ Docker build succeeds
- ✅ `openenv validate` passes

---

## ⚠️ Common Mistakes to Avoid

| Mistake | Fix |
|---|---|
| Submitting while Space is still building | Wait for ✅ Running status |
| Pushing `META AI HACKATHON/` root instead of `server/` contents | Only copy contents of `server/` to Space root |
| `inference.py` not in the hackathon project root | It's already there — don't move it |
| HF Token expired or wrong permissions | Create a new **Write** token |
| Multiple Spaces running (causes slowdowns) | Pause all other Spaces before submitting |

---

## 📂 What Goes Where (Summary)

```
HuggingFace Space repo root/       ← Push contents of server/ here
├── Dockerfile
├── server.py
├── email_triage_env.py
├── openenv.yaml
├── requirements.txt
└── data/
    ├── __init__.py
    └── emails.py

Your local machine/                ← Run inference.py from HERE
├── inference.py                   ← Must stay in root, not server/
├── demo_run.py
└── README.md
```

---

**Good luck! 🎉 Your environment scored 1.00 on all 3 tasks locally.**
