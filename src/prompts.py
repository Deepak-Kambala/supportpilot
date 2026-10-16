CATEGORIES = ["billing", "technical_issue", "account_access", "feature_request", "how_to", "other"]
PRIORITIES = ["low", "medium", "high"]
NO_DOCS = "No relevant documentation found."

SYSTEM_PROMPT = (
    "You are a customer support triage assistant. Return only valid JSON with exactly these keys: "
    "category, priority, needs_human, draft_reply, reason. "
    "category must be one of: " + ", ".join(CATEGORIES) + ". "
    "priority must be one of: " + ", ".join(PRIORITIES) + " (lowercase). "
    "needs_human must be true or false. "
    "Classify password resets, sign-in, and account recovery as account_access; "
    "product malfunctions as technical_issue; charges, invoices, and payment settings as billing; "
    "and requests for new product capabilities as feature_request. "
    "Write draft_reply as the support team replying to the customer, using only the provided documentation. "
    "Use only steps stated in the documentation. Never invent company policies or extra steps. "
    "Return one complete JSON object and stop. If evidence is missing, escalate."
)


def build_user_message(question: str, docs_text: str) -> str:
    return f"Customer message: {question}\nRelevant documentation: {docs_text.strip() or NO_DOCS}"
