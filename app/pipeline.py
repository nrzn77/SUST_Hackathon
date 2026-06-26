"""Orchestration: extract -> reason (LLM primary, rules fallback) -> validate -> reply."""
from __future__ import annotations

from typing import Any

from . import extract, reason_llm, reason_rules, reply, validate
from .schemas import TicketRequest


def analyze(req: TicketRequest) -> dict[str, Any]:
    complaint = req.complaint
    txns = extract.sanitize_history(req.transaction_history)
    language = extract.detect_language(complaint, req.language)
    valid_ids = {t["transaction_id"] for t in txns}

    # rule engine always runs: it is the fallback AND the hint for the LLM
    rule_result = reason_rules.reason(complaint, txns, req.user_type)

    llm_result = reason_llm.reason(complaint, txns, req.user_type, rule_hint=rule_result)
    chosen = llm_result if llm_result is not None else rule_result

    normalized = validate.normalize(chosen, valid_ids)
    # re-anchor the matched txn to the FINAL relevant id (validate may have nulled it)
    rid = normalized["relevant_transaction_id"]
    normalized["_matched"] = next((t for t in txns if t["transaction_id"] == rid), None)

    text = reply.build_text(normalized, language)

    return {
        "ticket_id": req.ticket_id,
        "relevant_transaction_id": normalized["relevant_transaction_id"],
        "evidence_verdict": normalized["evidence_verdict"],
        "case_type": normalized["case_type"],
        "severity": normalized["severity"],
        "department": normalized["department"],
        "agent_summary": text["agent_summary"],
        "recommended_next_action": text["recommended_next_action"],
        "customer_reply": text["customer_reply"],
        "human_review_required": normalized["human_review_required"],
        "confidence": normalized["confidence"],
        "reason_codes": normalized["reason_codes"],
    }
