"""Request/response schemas and the canonical enum taxonomy.

Enum values must match the problem statement EXACTLY. Any variant (case, plural,
alt spelling) is scored as a schema violation, so these frozensets are the single
source of truth that the validation layer clamps every field against.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

# --- Canonical enums (Section 7 + 5.2) -------------------------------------

CASE_TYPES = frozenset({
    "wrong_transfer",
    "payment_failed",
    "refund_request",
    "duplicate_payment",
    "merchant_settlement_delay",
    "agent_cash_in_issue",
    "phishing_or_social_engineering",
    "other",
})

DEPARTMENTS = frozenset({
    "customer_support",
    "dispute_resolution",
    "payments_ops",
    "merchant_operations",
    "agent_operations",
    "fraud_risk",
})

SEVERITIES = frozenset({"low", "medium", "high", "critical"})
VERDICTS = frozenset({"consistent", "inconsistent", "insufficient_data"})

LANGUAGES = frozenset({"en", "bn", "mixed"})
TXN_TYPES = frozenset({"transfer", "payment", "cash_in", "cash_out", "settlement", "refund"})
TXN_STATUSES = frozenset({"completed", "failed", "pending", "reversed"})

# case_type -> default department (decoded from Section 7.2 + sample cases)
CASE_TO_DEPARTMENT = {
    "wrong_transfer": "dispute_resolution",
    "payment_failed": "payments_ops",
    "refund_request": "customer_support",
    "duplicate_payment": "payments_ops",
    "merchant_settlement_delay": "merchant_operations",
    "agent_cash_in_issue": "agent_operations",
    "phishing_or_social_engineering": "fraud_risk",
    "other": "customer_support",
}

# case_type -> default severity (overridable by reasoning)
CASE_TO_SEVERITY = {
    "wrong_transfer": "high",
    "payment_failed": "high",
    "refund_request": "low",
    "duplicate_payment": "high",
    "merchant_settlement_delay": "medium",
    "agent_cash_in_issue": "high",
    "phishing_or_social_engineering": "critical",
    "other": "low",
}


# --- Request model ----------------------------------------------------------
# Deliberately lenient: optional fields are plain strings (not strict enums) so an
# unknown channel/user_type from a hidden test does not 400 the whole request.
# transaction_history is taken as raw objects and sanitised in extract.py, so one
# malformed entry never rejects the request.

class TicketRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    ticket_id: str
    complaint: str
    language: Optional[str] = None
    channel: Optional[str] = None
    user_type: Optional[str] = None
    campaign_context: Optional[str] = None
    transaction_history: Optional[list[Any]] = None
    metadata: Optional[dict[str, Any]] = None


# --- Response model (used to guarantee a well-formed body) ------------------

class AnalyzeResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    ticket_id: str
    relevant_transaction_id: Optional[str]
    evidence_verdict: str
    case_type: str
    severity: str
    department: str
    agent_summary: str
    recommended_next_action: str
    customer_reply: str
    human_review_required: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reason_codes: list[str] = []
