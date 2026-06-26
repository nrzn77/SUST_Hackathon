"""Deterministic reasoning engine.

Given a complaint + sanitised transaction history, decides the relevant
transaction, evidence verdict, case_type, severity, department, escalation and
reason codes. Used as the always-available fallback when the LLM is unavailable,
and as the candidate generator that the LLM result is cross-checked against.

Returns a plain dict with the reasoning fields (NOT the customer-facing text,
which is templated separately in reply.py).
"""
from __future__ import annotations

import re
from typing import Any, Optional

from . import extract
from .schemas import CASE_TO_DEPARTMENT, CASE_TO_SEVERITY

# --- keyword banks (English + Bangla/Banglish) ------------------------------

KW = {
    "phishing": [
        "otp", "pin", "password", "verification code", "scam", "fraud", "suspicious",
        "someone called", "claiming", "click", "link", "won", "lottery", "prize",
        "ওটিপি", "পিন", "পাসওয়ার্ড", "প্রতারণা", "সন্দেহ", "ফোন দিয়ে",
        # banglish
        "fake call", "fraud call", "vua call", "vhua", "bhua", "theke call", "call dise",
        "otp chaise", "pin chaise", "code chaise", "protarona", "frudh",
    ],
    "duplicate": [
        "twice", "two times", "double", "duplicate", "deducted twice", "charged twice",
        "two payments", "double charge", "দুইবার", "ডবল",
        # banglish
        "dui bar", "duibar", "dui baar", "dubar", "dui baar", "double kata", "duto",
    ],
    "failed": [
        "failed", "deducted", "deduction", "balance was deducted", "money was deducted",
        "but my balance", "ব্যর্থ", "কাটা হয়েছে", "টাকা কেটে",
        # banglish
        "fail", "fail hoise", "fail hoyse", "hoy nai", "hoyni", "hoini",
        "kete nise", "kete niche", "kete nilo", "kete niya", "katse", "katlo",
        "balance kete", "taka kete", "kete felse",
    ],
    "wrong": [
        "wrong number", "wrong person", "wrong recipient", "by mistake", "mistakenly",
        "reverse it", "ভুল নম্বর", "ভুল মানুষ", "ভুল করে",
        # banglish (bhul / vul = mistake)
        "bhul", "bhule", "vul", "vule", "vhul", "bul kore", "bhul kore", "vul kore",
        "wrong e", "onno number", "onno manush", "onno jaygay",
    ],
    "transfer_send": [
        "sent", "send", "transfer", "transferred", "পাঠিয়েছি", "পাঠ", "ট্রান্সফার",
        # banglish (pathানো / দেওয়া)
        "pathai", "pathay", "pathaisi", "pathaichi", "pathiyechi", "pathailam",
        "pathaiya", "pathaya", "send korsi", "send korechi", "send disi",
        "disi", "diyechi", "diychi", "dichi", "dilam", "transfer korsi",
    ],
    "cash_in": [
        "cash in", "cash-in", "cashin", "deposit", "ক্যাশ ইন", "জমা",
        # banglish
        "cash korsi", "cash in korsi", "cash korsi", "agent ke", "agent er kache",
    ],
    "settlement": [
        "settle", "settlement", "not been settled", "সেটেলমেন্ট",
        # banglish
        "settle hoy nai", "settle hoini", "settlement hoy nai",
    ],
    "refund": [
        "refund", "money back", "changed my mind", "don't want", "do not want",
        "ফেরত", "টাকা ফেরত",
        # banglish (ferot = return, back den = give back)
        "ferot", "feret", "fert", "ferat", "back den", "back chai", "back dao",
        "taka back", "taka ferot", "ferot chai", "ferot dao", "return den",
    ],
    "not_received": [
        "didn't get", "did not get", "not received", "hasn't received", "not reflected",
        "not show", "আসেনি", "পাইনি", "দেখছি না",
        # banglish
        "pai nai", "painai", "pai ni", "paini", "pai nei", "ashe nai", "asheni",
        "ashe ni", "ashena", "dhukeni", "dhuke nai", "joma hoy nai", "pelo na",
    ],
}


# Phishing fires ONLY on a third-party / scam threat signal — NOT on a bare mention
# of "pin"/"otp". This stops false positives on self-service ("I forgot my PIN") and
# on customers disclosing their own OTP.
THREAT_SIGNALS = [
    "someone called", "some one called", "called me", "got a call", "a call from",
    "claiming", "claim to be", "claims to be", "are from", "pretending", "impersonat",
    "scam", "suspicious", "fraud call", "fake call", "fake message", "fake sms",
    "asked for my", "asking for my", "ask for my", "share korte bol",
    "you won", "apni jiteche", "lottery", "prize", "click this link", "click the link",
    "reward link", "theke call", "call dise", "call diye", "call dia",
    "otp chaise", "pin chaise", "code chaise", "ফোন দিয়ে", "প্রতারণা", "সন্দেহ",
    "vua call", "bhua call", "protarona",
]

# Unauthorised / disputed charge -> route a refund to dispute_resolution, not simple support.
CONTESTED_SIGNALS = [
    "didn't authorize", "did not authorize", "never authorized", "not authorize",
    "unauthorized", "unauthorised", "didn't make", "did not make", "never made",
    "i didn't do", "i did not do", "without my permission", "without permission",
    "without my knowledge", "ami kori nai", "ami korini", "authorize kori nai",
    "fraudulent", "i dispute", "disputed", "never bought", "never purchased",
    "didn't buy", "i never", "charge i never",
]


def _has(text: str, key: str) -> bool:
    return any(kw in text for kw in KW[key])


def classify_case_type(complaint_lc: str, user_type: Optional[str], txns: list[dict]) -> str:
    """Pick case_type by primary signal. Order encodes priority (safety first)."""
    # phishing/social engineering: requires an actual scam/third-party threat
    if any(w in complaint_lc for w in THREAT_SIGNALS):
        return "phishing_or_social_engineering"
    # duplicate: explicit wording OR a detected duplicate pair
    if _has(complaint_lc, "duplicate") or _detect_duplicate_pair(txns) is not None:
        return "duplicate_payment"
    # settlement and agent cash-in are checked before the generic transfer dispute
    if _has(complaint_lc, "settlement") or (user_type == "merchant" and _has(complaint_lc, "not_received")):
        return "merchant_settlement_delay"
    if _has(complaint_lc, "cash_in") and _has(complaint_lc, "not_received"):
        return "agent_cash_in_issue"
    # wrong transfer: explicit wording OR a sent transfer the customer says went astray
    if _has(complaint_lc, "wrong") or (
        _has(complaint_lc, "transfer_send") and (_has(complaint_lc, "not_received") or _has(complaint_lc, "wrong"))
    ):
        return "wrong_transfer"
    if _has(complaint_lc, "failed"):
        return "payment_failed"
    if _has(complaint_lc, "refund"):
        return "refund_request"
    return "other"


def _detect_duplicate_pair(txns: list[dict]) -> Optional[str]:
    """Return the id of the later transaction in a duplicate pair, if one exists."""
    by_key: dict[tuple, list[dict]] = {}
    for t in txns:
        if t.get("amount") is None or t.get("status") != "completed":
            continue
        key = (t.get("amount"), t.get("counterparty"), t.get("type"))
        by_key.setdefault(key, []).append(t)
    for group in by_key.values():
        if len(group) >= 2:
            group.sort(key=lambda t: t.get("timestamp", ""))
            return group[-1]["transaction_id"]
    return None


def _established_recipient(matched: dict, txns: list[dict]) -> bool:
    """True if there are >=2 OTHER completed transfers to the same counterparty."""
    cp = matched.get("counterparty")
    if not cp:
        return False
    count = sum(
        1 for t in txns
        if t is not matched and t.get("counterparty") == cp
        and t.get("type") == "transfer" and t.get("status") == "completed"
    )
    return count >= 2


def reason(complaint: str, txns: list[dict], user_type: Optional[str]) -> dict[str, Any]:
    lc = (complaint or "").lower()
    case_type = classify_case_type(lc, user_type, txns)

    scored = extract.score_transactions(complaint, txns)
    relevant_id, ambiguous = extract.pick_relevant(scored)
    matched = next((s["txn"] for s in scored if s["txn"]["transaction_id"] == relevant_id), None)

    verdict = "consistent"
    severity = CASE_TO_SEVERITY.get(case_type, "low")
    reason_codes: list[str] = [case_type]
    confidence = 0.85

    # --- phishing / social engineering: always critical, never needs a txn ---
    if case_type == "phishing_or_social_engineering":
        verdict = "consistent" if relevant_id else "insufficient_data"
        severity = "critical"
        reason_codes = ["phishing", "credential_protection", "critical_escalation"]
        confidence = 0.95

    # --- duplicate payment: lock onto the later of the duplicate pair --------
    elif case_type == "duplicate_payment":
        dup_id = _detect_duplicate_pair(txns)
        if dup_id is not None:
            relevant_id, ambiguous, matched = dup_id, False, next(
                (t for t in txns if t["transaction_id"] == dup_id), None)
            verdict = "consistent"
            reason_codes = ["duplicate_payment", "biller_verification_required"]
            confidence = 0.92
        else:
            relevant_id, verdict, severity = None, "insufficient_data", "medium"
            reason_codes = ["duplicate_payment_claim", "needs_clarification"]
            confidence = 0.6

    # --- ambiguous / no match -> never guess --------------------------------
    elif ambiguous or relevant_id is None:
        verdict = "insufficient_data"
        relevant_id = None
        if case_type == "wrong_transfer":
            severity = "medium"
            reason_codes = ["ambiguous_match", "needs_clarification"] if ambiguous else [
                "wrong_transfer_claim", "needs_clarification"]
            confidence = 0.65
        elif case_type in ("other", "refund_request"):
            case_type = "other"
            severity = "low"
            reason_codes = ["vague_complaint", "needs_clarification"]
            confidence = 0.6
        else:
            reason_codes = [case_type, "needs_clarification"]
            confidence = 0.6

    # --- matched transaction -> assess consistency --------------------------
    else:
        if case_type == "wrong_transfer" and matched and _established_recipient(matched, txns):
            verdict = "inconsistent"
            severity = "medium"
            reason_codes = ["wrong_transfer_claim", "established_recipient_pattern", "evidence_inconsistent"]
            confidence = 0.75
        else:
            verdict = "consistent"
            confidence = 0.9
            reason_codes = _codes_for(case_type, matched)

    # Fix 3: a specific amount was claimed but nothing in history matches it
    if relevant_id is None and txns and extract.parse_amounts(complaint):
        if "claimed_record_not_found" not in reason_codes:
            reason_codes = reason_codes + ["claimed_record_not_found"]

    department = _route(case_type, user_type)
    human_review = _needs_review(case_type, verdict, relevant_id)

    # Fix 5: an unauthorised / disputed refund is a dispute, not simple support
    if case_type == "refund_request" and any(w in lc for w in CONTESTED_SIGNALS):
        department = "dispute_resolution"
        if severity == "low":
            severity = "medium"
        human_review = True
        reason_codes = ["refund_request", "contested_charge", "dispute"]

    # Fix 7: keep safety-first priority (phishing won above) but flag a co-occurring
    # financial loss so the agent sees both issues.
    if case_type == "phishing_or_social_engineering" and (
        _has(lc, "failed") or _has(lc, "wrong") or relevant_id is not None
    ):
        if "secondary_financial_issue" not in reason_codes:
            reason_codes = reason_codes + ["secondary_financial_issue"]

    return {
        "relevant_transaction_id": relevant_id,
        "evidence_verdict": verdict,
        "case_type": case_type,
        "severity": severity,
        "department": department,
        "human_review_required": human_review,
        "confidence": round(confidence, 2),
        "reason_codes": reason_codes,
        "_matched": matched,  # internal: used by reply templating, stripped before output
    }


def _codes_for(case_type: str, matched: Optional[dict]) -> list[str]:
    codes = [case_type]
    if case_type == "wrong_transfer":
        codes += ["transaction_match", "dispute_initiated"]
    elif case_type == "payment_failed":
        codes += ["potential_balance_deduction"]
    elif case_type == "agent_cash_in_issue":
        codes += ["pending_transaction", "agent_ops"] if matched and matched.get("status") == "pending" else ["agent_ops"]
    elif case_type == "merchant_settlement_delay":
        codes += ["delay", "pending"]
    elif case_type == "refund_request":
        codes += ["merchant_policy_dependent"]
    return codes


def _route(case_type: str, user_type: Optional[str]) -> str:
    return CASE_TO_DEPARTMENT.get(case_type, "customer_support")


def _needs_review(case_type: str, verdict: str, relevant_id: Optional[str]) -> bool:
    if case_type == "phishing_or_social_engineering":
        return True
    if verdict == "inconsistent":
        return True
    if case_type in ("wrong_transfer", "duplicate_payment", "agent_cash_in_issue") and relevant_id is not None:
        return True
    return False
