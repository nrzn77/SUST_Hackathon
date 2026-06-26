"""Hard validation / normalisation gate.

Applied to ANY reasoning result (LLM or rules) before reply generation. Guarantees:
  * every enum field is exactly a legal value (no variants escape)
  * relevant_transaction_id actually exists in the provided history, else null
  * verdict/relevant_id are mutually consistent
  * escalation is forced on for risky cases even if the reasoner said otherwise
"""
from __future__ import annotations

from typing import Any, Optional

from .schemas import (
    CASE_TO_DEPARTMENT,
    CASE_TO_SEVERITY,
    CASE_TYPES,
    DEPARTMENTS,
    SEVERITIES,
    VERDICTS,
)


def _clamp(value: Any, allowed: frozenset[str], default: str) -> str:
    if isinstance(value, str) and value.strip().lower() in allowed:
        return value.strip().lower()
    return default


def normalize(reasoning: dict[str, Any], valid_ids: set[str]) -> dict[str, Any]:
    case_type = _clamp(reasoning.get("case_type"), CASE_TYPES, "other")
    verdict = _clamp(reasoning.get("evidence_verdict"), VERDICTS, "insufficient_data")
    severity = _clamp(reasoning.get("severity"), SEVERITIES, CASE_TO_SEVERITY.get(case_type, "low"))

    department = reasoning.get("department")
    department = department.strip().lower() if isinstance(department, str) else ""
    if department not in DEPARTMENTS:
        department = CASE_TO_DEPARTMENT.get(case_type, "customer_support")

    # relevant_transaction_id must be a real id from the provided history
    rid = reasoning.get("relevant_transaction_id")
    if not (isinstance(rid, str) and rid in valid_ids):
        rid = None

    # consistency: cannot be "consistent" / "inconsistent" with no referenced transaction
    if rid is None and verdict in ("consistent", "inconsistent"):
        verdict = "insufficient_data"

    # escalation overrides (safety): force human review where it matters
    human_review = bool(reasoning.get("human_review_required"))
    if (
        case_type == "phishing_or_social_engineering"
        or verdict == "inconsistent"
        or (case_type in ("wrong_transfer", "duplicate_payment", "agent_cash_in_issue") and rid is not None)
    ):
        human_review = True

    confidence = reasoning.get("confidence")
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.6
    confidence = max(0.0, min(1.0, confidence))

    reason_codes = reasoning.get("reason_codes")
    if not isinstance(reason_codes, list) or not reason_codes:
        reason_codes = [case_type]
    reason_codes = [str(c) for c in reason_codes][:6]

    return {
        "relevant_transaction_id": rid,
        "evidence_verdict": verdict,
        "case_type": case_type,
        "severity": severity,
        "department": department,
        "human_review_required": human_review,
        "confidence": round(confidence, 2),
        "reason_codes": reason_codes,
        "_matched": reasoning.get("_matched"),
    }
