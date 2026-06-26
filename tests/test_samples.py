"""Validate the service against the 10 public sample cases.

Functional equivalence (per the problem statement) means: same
relevant_transaction_id, same evidence_verdict, same case_type, same department,
comparable severity, and a safe customer_reply. We assert those, NOT exact text.
Run with the LLM disabled so this tests the deterministic floor.
"""
from __future__ import annotations

import json
import os
import pathlib

import pytest

os.environ.pop("GEMINI_API_KEY", None)  # force rule-engine path for a deterministic baseline
os.environ.pop("GOOGLE_API_KEY", None)

from app import pipeline, reply  # noqa: E402
from app.schemas import (  # noqa: E402
    CASE_TYPES, DEPARTMENTS, SEVERITIES, VERDICTS, TicketRequest,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
CASES = json.loads((ROOT / "sample_cases.json").read_text(encoding="utf-8"))["cases"]

_SEV_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_sample_case(case):
    exp = case["expected_output"]
    got = pipeline.analyze(TicketRequest(**case["input"]))

    assert got["ticket_id"] == exp["ticket_id"]
    assert got["relevant_transaction_id"] == exp["relevant_transaction_id"], "relevant txn"
    assert got["evidence_verdict"] == exp["evidence_verdict"], "verdict"
    assert got["case_type"] == exp["case_type"], "case_type"
    assert got["department"] == exp["department"], "department"

    # severity comparable: within one rank of expected
    assert abs(_SEV_RANK[got["severity"]] - _SEV_RANK[exp["severity"]]) <= 1, "severity off by >1"

    # schema legality
    assert got["case_type"] in CASE_TYPES
    assert got["department"] in DEPARTMENTS
    assert got["severity"] in SEVERITIES
    assert got["evidence_verdict"] in VERDICTS
    assert isinstance(got["human_review_required"], bool)
    assert 0.0 <= got["confidence"] <= 1.0

    # safety: reply must pass the scrubber
    assert reply.is_safe_reply(got["customer_reply"]), "unsafe customer_reply"
    # never literally promise a refund
    assert "we will refund" not in got["customer_reply"].lower()


def test_human_review_matches_samples():
    """Escalation flag should match the sample expectations exactly."""
    for case in CASES:
        got = pipeline.analyze(TicketRequest(**case["input"]))
        assert got["human_review_required"] == case["expected_output"]["human_review_required"], case["id"]


def test_bangla_reply_language():
    """The Bangla sample (SAMPLE-07) must get a Bangla customer_reply."""
    case = next(c for c in CASES if c["id"] == "SAMPLE-07")
    got = pipeline.analyze(TicketRequest(**case["input"]))
    assert any("ঀ" <= ch <= "৿" for ch in got["customer_reply"]), "expected Bangla reply"
