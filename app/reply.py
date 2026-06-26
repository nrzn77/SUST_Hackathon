"""Safe, templated customer-facing and agent-facing text + a safety scrubber.

agent_summary and recommended_next_action are ALWAYS English (agent-facing, as in
the samples). customer_reply matches the complaint language (en/mixed -> English,
bn -> Bangla). Replies are built from vetted templates so the safety penalties in
Section 8 cannot be triggered; the scrubber is a final defence-in-depth guard.
"""
from __future__ import annotations

import re
from typing import Any, Optional

# --- safe constants ---------------------------------------------------------

CRED_EN = "Please do not share your PIN or OTP with anyone."
CRED_BN = "অনুগ্রহ করে কারো সাথে আপনার পিন বা ওটিপি শেয়ার করবেন না।"
RETURN_EN = "any eligible amount will be returned through official channels"
RETURN_BN = "অফিসিয়াল চ্যানেলের মাধ্যমে যেকোনো প্রযোজ্য পরিমাণ ফেরত দেওয়া হবে"


def _amt(x: Optional[float]) -> str:
    if x is None:
        return "the reported amount"
    return str(int(x)) if float(x).is_integer() else str(x)


def _is_bn(language: str) -> bool:
    return language == "bn"


# --- public entrypoint ------------------------------------------------------

def build_text(reasoning: dict[str, Any], language: str) -> dict[str, str]:
    case_type = reasoning["case_type"]
    verdict = reasoning["evidence_verdict"]
    tid = reasoning.get("relevant_transaction_id")
    matched = reasoning.get("_matched") or {}
    amount = matched.get("amount")
    counterparty = matched.get("counterparty") or "the recipient"
    status = matched.get("status") or "pending"

    summary = _summary(case_type, verdict, tid, amount, counterparty, status)
    action = _action(case_type, verdict, tid)
    reply = _customer_reply(case_type, verdict, tid, language)

    # Fix 10: matched transaction is already reversed -> tell the agent
    if status == "reversed" and tid:
        summary += f" Note: {tid} already shows status 'reversed'."
        action = (f"Confirm whether the customer has received the reversed funds for {tid} "
                  f"before taking further action. ") + action

    # defence in depth: never emit an unsafe reply
    if not is_safe_reply(reply):
        reply = _safe_fallback(tid, language)
    return {"agent_summary": summary, "recommended_next_action": action, "customer_reply": reply}


# --- agent_summary (English) ------------------------------------------------

def _summary(case_type, verdict, tid, amount, counterparty, status) -> str:
    a = _amt(amount)
    if case_type == "wrong_transfer" and verdict == "inconsistent":
        return (f"Customer claims {tid} ({a} BDT to {counterparty}) was a wrong transfer, "
                f"but transaction history shows prior transfers to the same counterparty, "
                f"suggesting an established recipient.")
    if case_type == "wrong_transfer" and tid:
        return (f"Customer reports sending {a} BDT via {tid} to {counterparty}, "
                f"which they now believe was the wrong recipient.")
    if case_type == "wrong_transfer":
        return (f"Customer reports a {a} BDT transfer was not received. Multiple transactions "
                f"plausibly match; the correct one cannot be determined without more detail.")
    if case_type == "payment_failed":
        return (f"Customer attempted a {a} BDT payment ({tid}) which failed but reports the "
                f"balance was deducted. Requires payments operations investigation.")
    if case_type == "refund_request":
        return f"Customer requests a refund of {a} BDT for {tid} (merchant payment). Not a service failure."
    if case_type == "duplicate_payment":
        return (f"Customer reports a duplicate payment. Two identical {a} BDT payments to "
                f"{counterparty} appear in history; {tid} is likely the duplicate.")
    if case_type == "merchant_settlement_delay":
        return (f"Merchant reports settlement {tid} ({a} BDT) is delayed beyond the expected "
                f"window. Status is {status}.")
    if case_type == "agent_cash_in_issue":
        return (f"Customer reports {a} BDT cash-in via {counterparty} ({tid}) not reflected in "
                f"balance. Status is {status}.")
    if case_type == "phishing_or_social_engineering":
        return ("Customer reports an unsolicited contact claiming to be from the company and "
                "asking for credentials. Likely a social engineering attempt.")
    return ("Customer reports a vague concern without specifying transaction, amount, or issue. "
            "Insufficient detail to identify a relevant transaction.")


# --- recommended_next_action (English) --------------------------------------

def _action(case_type, verdict, tid) -> str:
    if case_type == "wrong_transfer" and verdict == "inconsistent":
        return ("Flag for human review. Verify with the customer whether this was genuinely a "
                "wrong transfer given the established transaction pattern with this recipient.")
    if case_type == "wrong_transfer" and tid:
        return f"Verify {tid} details with the customer and initiate the wrong-transfer dispute workflow per policy."
    if case_type == "wrong_transfer":
        return ("Ask the customer for the recipient's number to identify the correct transaction. "
                "Do not initiate a dispute until the transaction is confirmed.")
    if case_type == "payment_failed":
        return (f"Investigate {tid} ledger status. If the balance was deducted on a failed payment, "
                f"initiate the reversal flow within standard SLA.")
    if case_type == "refund_request":
        return ("Inform the customer that refund eligibility depends on the merchant's own policy "
                "and guide them to contact the merchant directly.")
    if case_type == "duplicate_payment":
        return (f"Verify the duplicate with payments_ops. If the biller confirms only one payment "
                f"was received, initiate reversal of {tid}.")
    if case_type == "merchant_settlement_delay":
        return ("Route to merchant_operations to verify settlement batch status. If the batch is "
                "delayed, communicate a revised ETA to the merchant.")
    if case_type == "agent_cash_in_issue":
        return (f"Investigate {tid} pending status with agent operations. Confirm settlement state "
                f"and resolve within the standard cash-in SLA.")
    if case_type == "phishing_or_social_engineering":
        return ("Escalate to fraud_risk immediately. Confirm to the customer that the company never "
                "asks for OTP. Log the reported number for fraud pattern analysis.")
    return ("Reply to the customer asking for specific details: which transaction, what amount, "
            "what went wrong, and approximate time.")


# --- customer_reply (language-matched) --------------------------------------

def _customer_reply(case_type, verdict, tid, language) -> str:
    bn = _is_bn(language)
    if case_type == "phishing_or_social_engineering":
        if bn:
            return ("কোনো তথ্য শেয়ার করার আগে যোগাযোগ করার জন্য ধন্যবাদ। আমরা কখনোই আপনার পিন, ওটিপি বা "
                    "পাসওয়ার্ড চাই না। কেউ নিজেকে আমাদের প্রতিনিধি দাবি করলেও এগুলো শেয়ার করবেন না। আমাদের "
                    "ফ্রড দলকে বিষয়টি জানানো হয়েছে।")
        return ("Thank you for reaching out before sharing any information. We never ask for your PIN, "
                "OTP, or password under any circumstances. Please do not share these with anyone, even "
                "if they claim to be from us. Our fraud team has been notified of this incident.")

    if case_type == "refund_request" and tid:
        if bn:
            return ("যোগাযোগের জন্য ধন্যবাদ। সম্পন্ন হওয়া মার্চেন্ট পেমেন্টের ফেরত মার্চেন্টের নিজস্ব নীতির উপর "
                    "নির্ভর করে। আমরা সরাসরি মার্চেন্টের সাথে যোগাযোগের পরামর্শ দিচ্ছি। সাহায্য প্রয়োজন হলে জানান। "
                    + CRED_BN)
        return ("Thank you for reaching out. Refunds for completed merchant payments depend on the "
                "merchant's own policy. We recommend contacting the merchant directly. If you need help "
                "reaching them, please reply and we will guide you. " + CRED_EN)

    if case_type == "payment_failed" and tid:
        if bn:
            return (f"আমরা লক্ষ্য করেছি যে লেনদেন {tid} এর কারণে অপ্রত্যাশিতভাবে ব্যালেন্স কেটে থাকতে পারে। আমাদের "
                    f"পেমেন্ট দল বিষয়টি যাচাই করবে এবং {RETURN_BN}। " + CRED_BN)
        return (f"We have noted that transaction {tid} may have caused an unexpected balance deduction. "
                f"Our payments team will review the case and {RETURN_EN}. " + CRED_EN)

    if case_type == "duplicate_payment" and tid:
        if bn:
            return (f"লেনদেন {tid} এর সম্ভাব্য ডবল পেমেন্ট সম্পর্কে আমরা অবগত হয়েছি। আমাদের পেমেন্ট দল বিলারের "
                    f"সাথে যাচাই করবে এবং {RETURN_BN}। " + CRED_BN)
        return (f"We have noted the possible duplicate payment for transaction {tid}. Our payments team "
                f"will verify with the biller and {RETURN_EN}. " + CRED_EN)

    if case_type == "merchant_settlement_delay" and tid:
        if bn:
            return (f"আপনার সেটেলমেন্ট {tid} সম্পর্কে আমরা অবগত হয়েছি। আমাদের মার্চেন্ট অপারেশন্স দল ব্যাচ স্ট্যাটাস "
                    f"যাচাই করে অফিসিয়াল চ্যানেলের মাধ্যমে প্রত্যাশিত সময় জানাবে।")
        return (f"We have noted your concern about settlement {tid}. Our merchant operations team will "
                f"check the batch status and update you on the expected settlement time through official channels.")

    if case_type == "agent_cash_in_issue" and tid:
        if bn:
            return (f"আপনার লেনদেন {tid} এর বিষয়ে আমরা অবগত হয়েছি। আমাদের এজেন্ট অপারেশন্স দল এটি দ্রুত যাচাই করবে "
                    f"এবং অফিসিয়াল চ্যানেলে আপনাকে জানাবে। " + CRED_BN)
        return (f"We have noted your concern about transaction {tid}. Our agent operations team will verify "
                f"it promptly and update you through official channels. " + CRED_EN)

    if case_type == "wrong_transfer" and tid:
        if bn:
            return (f"আপনার লেনদেন {tid} সম্পর্কে আমরা অবগত হয়েছি। {CRED_BN} আমাদের ডিসপিউট দল বিষয়টি যত্নসহকারে "
                    f"পর্যালোচনা করে অফিসিয়াল চ্যানেলে আপনার সাথে যোগাযোগ করবে।")
        return (f"We have received your request regarding transaction {tid}. {CRED_EN} Our dispute team "
                f"will review the case carefully and contact you through official support channels.")

    # ambiguous / insufficient / vague -> ask for clarification (no guess)
    if bn:
        return ("যোগাযোগের জন্য ধন্যবাদ। দ্রুত সাহায্যের জন্য অনুগ্রহ করে লেনদেন আইডি, পরিমাণ এবং কী সমস্যা হয়েছে "
                "তা জানান। " + CRED_BN)
    return ("Thank you for reaching out. To help you faster, please share the transaction ID, the amount "
            "involved, and a short description of what went wrong. " + CRED_EN)


def _safe_fallback(tid, language) -> str:
    if _is_bn(language):
        return ("যোগাযোগের জন্য ধন্যবাদ। আমাদের সহায়তা দল বিষয়টি পর্যালোচনা করে অফিসিয়াল চ্যানেলে আপনার সাথে "
                "যোগাযোগ করবে। " + CRED_BN)
    return ("Thank you for reaching out. Our support team will review your case and contact you through "
            "official support channels. " + CRED_EN)


# --- safety scrubber --------------------------------------------------------

_NEG = re.compile(r"(?:not|never|don'?t|do not|কখনো|না)\b", re.IGNORECASE)
_CRED_REQ = re.compile(
    r"(share|provide|send|give|enter|tell|type|confirm|need|verify|what'?s|what is)\b[^.]{0,25}\b"
    r"(pin|otp|password|one[- ]?time|card number)",
    re.IGNORECASE,
)
_REFUND_PROMISE = re.compile(
    r"\b(we|i)\s*(will|'ll|have|has)\s*(refund|reverse|return your money|pay you back|"
    r"refunded|reversed|credited your account)\b",
    re.IGNORECASE,
)
_THIRD_PARTY = re.compile(r"\bcall\s+(this number|the following number|\+?\d[\d ()-]{6,})", re.IGNORECASE)


def is_safe_reply(text: str) -> bool:
    """Return False if the reply asks for credentials, promises a refund, or redirects off-channel."""
    if _REFUND_PROMISE.search(text):
        return False
    if _THIRD_PARTY.search(text):
        return False
    for m in _CRED_REQ.finditer(text):
        window = text[max(0, m.start() - 20):m.start()]
        if not _NEG.search(window):  # an affirmative credential request
            return False
    return True
