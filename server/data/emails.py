"""
Simulated email dataset with ground-truth labels for all tasks.
Each email has: id, subject, sender, body, category, priority, requires_reply, reply_tone
"""

from typing import List, Dict, Any

EMAILS: List[Dict[str, Any]] = [
    # ---- BILLING ----
    {
        "id": "e001",
        "subject": "Invoice #4821 overdue – payment required",
        "sender": "billing@acmecorp.com",
        "body": "Dear team, Invoice #4821 for $3,200 was due on March 15. Please process payment immediately to avoid service interruption.",
        "category": "billing",
        "priority": "urgent",
        "requires_reply": True,
        "reply_tone": "apologetic_and_prompt",
        "summary": "Overdue invoice needs immediate payment acknowledgment",
    },
    {
        "id": "e002",
        "subject": "Your subscription renewal is coming up",
        "sender": "noreply@saasplatform.io",
        "body": "Your annual subscription will renew on April 30 for $599. No action is needed unless you wish to cancel.",
        "category": "billing",
        "priority": "normal",
        "requires_reply": False,
        "reply_tone": None,
        "summary": "Subscription renewal notification, no action needed",
    },
    # ---- SUPPORT ----
    {
        "id": "e003",
        "subject": "URGENT: Production server down since 2am",
        "sender": "ops@clientco.com",
        "body": "Hi, our production environment has been unreachable since 2AM EST. Revenue impact is $50k/hour. Need immediate escalation.",
        "category": "support",
        "priority": "urgent",
        "requires_reply": True,
        "reply_tone": "empathetic_and_action_oriented",
        "summary": "Critical production outage requiring immediate escalation",
    },
    {
        "id": "e004",
        "subject": "How do I reset my password?",
        "sender": "john.doe@gmail.com",
        "body": "Hello, I've forgotten my password and can't log in. Can you help me reset it? Thanks.",
        "category": "support",
        "priority": "normal",
        "requires_reply": True,
        "reply_tone": "helpful_and_friendly",
        "summary": "Password reset request from user",
    },
    {
        "id": "e005",
        "subject": "Feature request: dark mode",
        "sender": "user123@example.com",
        "body": "Love the product! Would really appreciate a dark mode option. Many of my colleagues agree.",
        "category": "support",
        "priority": "low",
        "requires_reply": True,
        "reply_tone": "appreciative_and_informative",
        "summary": "Feature request for dark mode, low priority",
    },
    # ---- SPAM ----
    {
        "id": "e006",
        "subject": "You've been selected for a FREE iPhone 15!!",
        "sender": "promo@totally-legit-prizes.biz",
        "body": "Congratulations!! Click here NOW to claim your FREE iPhone 15 before it expires!!! Limited time offer!!!",
        "category": "spam",
        "priority": "low",
        "requires_reply": False,
        "reply_tone": None,
        "summary": "Promotional spam email, no action needed",
    },
    {
        "id": "e007",
        "subject": "Make $5000/day working from home",
        "sender": "richquick@moneyfast.net",
        "body": "Discover the secret method that millionaires use! Join today and start earning instantly. No experience needed.",
        "category": "spam",
        "priority": "low",
        "requires_reply": False,
        "reply_tone": None,
        "summary": "Get-rich-quick spam, no action needed",
    },
    # ---- INTERNAL ----
    {
        "id": "e008",
        "subject": "Q2 Planning Meeting – this Thursday 3PM",
        "sender": "manager@company.com",
        "body": "Hi team, reminder that our Q2 planning session is this Thursday at 3PM in Conference Room B. Please prepare your department updates.",
        "category": "internal",
        "priority": "normal",
        "requires_reply": True,
        "reply_tone": "confirmatory",
        "summary": "Internal meeting reminder, confirmation needed",
    },
    {
        "id": "e009",
        "subject": "Updated vacation policy – please read",
        "sender": "hr@company.com",
        "body": "Effective May 1, the company vacation policy will be updated. Key changes include carry-over limits and approval timelines. Full document attached.",
        "category": "internal",
        "priority": "normal",
        "requires_reply": False,
        "reply_tone": None,
        "summary": "HR policy update announcement, informational",
    },
    {
        "id": "e010",
        "subject": "Your pull request has been approved",
        "sender": "github-noreply@github.com",
        "body": "PR #234 'Fix authentication bug' has been approved by 2 reviewers and is ready to merge.",
        "category": "internal",
        "priority": "normal",
        "requires_reply": False,
        "reply_tone": None,
        "summary": "GitHub PR approval notification",
    },
    # ---- More for variety ----
    {
        "id": "e011",
        "subject": "Re: Contract renewal negotiation",
        "sender": "procurement@enterprise.com",
        "body": "Following up on our last discussion — we'd like to renew for 3 years at the negotiated rate of $120k/year. Please confirm by Friday.",
        "category": "billing",
        "priority": "urgent",
        "requires_reply": True,
        "reply_tone": "professional_and_decisive",
        "summary": "Contract renewal pending confirmation by Friday",
    },
    {
        "id": "e012",
        "subject": "System maintenance window – April 10",
        "sender": "infrastructure@company.com",
        "body": "Scheduled maintenance on April 10 from 2AM to 4AM UTC. Services will be unavailable during this window.",
        "category": "internal",
        "priority": "normal",
        "requires_reply": False,
        "reply_tone": None,
        "summary": "Scheduled maintenance notification",
    },
    {
        "id": "e013",
        "subject": "API integration broken after your update",
        "sender": "dev@partnerco.com",
        "body": "After your v2.3 release yesterday, our API integration stopped working. Getting 401 errors on all authenticated endpoints. This affects our production.",
        "category": "support",
        "priority": "urgent",
        "requires_reply": True,
        "reply_tone": "empathetic_and_action_oriented",
        "summary": "Broken API integration post-release, urgent fix required",
    },
    {
        "id": "e014",
        "subject": "Unsubscribe from our mailing list",
        "sender": "newsletter@marketingco.com",
        "body": "You're receiving this because you subscribed. Click unsubscribe at any time.",
        "category": "spam",
        "priority": "low",
        "requires_reply": False,
        "reply_tone": None,
        "summary": "Marketing newsletter, can be unsubscribed",
    },
    {
        "id": "e015",
        "subject": "Feedback on last week's webinar",
        "sender": "events@company.com",
        "body": "Thank you for attending! We'd love your feedback. The survey takes just 2 minutes.",
        "category": "internal",
        "priority": "low",
        "requires_reply": False,
        "reply_tone": None,
        "summary": "Internal survey request, low priority",
    },
]


# --- Task-specific subsets ---

SORT_INBOX_EMAILS = [EMAILS[i] for i in [2, 0, 7, 4, 9]]  # mixed priorities
SORT_INBOX_GROUND_TRUTH_ORDER = ["e003", "e001", "e008", "e005", "e010"]  # urgent>normal>low

TRIAGE_EMAILS = [EMAILS[i] for i in [0, 2, 5, 7, 9, 1, 3, 6, 10, 12]]  # 10 emails
TRIAGE_GROUND_TRUTH = {
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

FULL_WORKFLOW_EMAILS = [EMAILS[i] for i in [0, 2, 7, 3, 10]]  # 5 emails needing replies
FULL_WORKFLOW_GROUND_TRUTH = {
    "e001": {"category": "billing", "priority": "urgent", "requires_reply": True},
    "e003": {"category": "support", "priority": "urgent", "requires_reply": True},
    "e008": {"category": "internal", "priority": "normal", "requires_reply": True},
    "e004": {"category": "support", "priority": "normal", "requires_reply": True},
    "e011": {"category": "billing", "priority": "urgent", "requires_reply": True},
}

VALID_CATEGORIES = ["billing", "support", "spam", "internal"]
VALID_PRIORITIES = ["urgent", "normal", "low"]
