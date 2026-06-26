"""Deterministic feature extraction over a complaint + transaction history.

Pure functions, no I/O. Produces the signals consumed by both the rule engine
and the LLM adapter: normalised transactions, detected language, parsed amounts,
phone hints, and per-transaction candidate scores. Robust to messy / partial data.
"""
from __future__ import annotations

import re
from typing import Any, Optional

from .schemas import TXN_STATUSES, TXN_TYPES

# Bengali digits -> ASCII
_BN_DIGITS = {ord(c): str(i) for i, c in enumerate("০১২৩৪৫৬৭৮৯")}
_BENGALI_RANGE = re.compile(r"[ঀ-৿]")
_LATIN_RANGE = re.compile(r"[A-Za-z]")


def to_ascii_digits(text: str) -> str:
    return (text or "").translate(_BN_DIGITS)


# --- Transaction sanitisation ----------------------------------------------

def sanitize_history(raw: Optional[list[Any]]) -> list[dict[str, Any]]:
    """Coerce raw history into clean txn dicts. Bad entries are skipped, never raised."""
    out: list[dict[str, Any]] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if not isinstance(item, dict):
            continue
        tid = item.get("transaction_id")
        if tid is None or str(tid).strip() == "":
            continue  # unusable without an id
        amount = _coerce_amount(item.get("amount"))
        ttype = _norm_enum(item.get("type"), TXN_TYPES)
        status = _norm_enum(item.get("status"), TXN_STATUSES)
        out.append({
            "transaction_id": str(tid),
            "timestamp": str(item.get("timestamp") or ""),
            "type": ttype,
            "amount": amount,
            "counterparty": str(item.get("counterparty") or ""),
            "status": status,
        })
    return out


def _coerce_amount(val: Any) -> Optional[float]:
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        s = to_ascii_digits(val).replace(",", "").strip()
        m = re.search(r"\d+(\.\d+)?", s)
        if m:
            try:
                return float(m.group(0))
            except ValueError:
                return None
    return None


def _norm_enum(val: Any, allowed: frozenset[str]) -> Optional[str]:
    if val is None:
        return None
    s = str(val).strip().lower()
    return s if s in allowed else None


# --- Language detection -----------------------------------------------------

def detect_language(text: str, declared: Optional[str]) -> str:
    """Trust a valid declared language; otherwise sniff Bengali vs Latin script."""
    if declared in ("en", "bn", "mixed"):
        return declared
    bn = len(_BENGALI_RANGE.findall(text or ""))
    lat = len(_LATIN_RANGE.findall(text or ""))
    if bn == 0:
        return "en"
    if lat == 0:
        return "bn"
    # both scripts present
    return "mixed" if lat > bn * 0.3 else "bn"


# --- Amount / phone / time parsing -----------------------------------------

_PHONE_RE = re.compile(r"(?:\+?88)?0?1\d{8,9}")


def extract_phones(text: str) -> list[str]:
    """Return last-9-digit normalised phone tokens found in the text."""
    t = to_ascii_digits(text or "")
    phones = []
    for m in _PHONE_RE.findall(t):
        digits = re.sub(r"\D", "", m)
        if len(digits) >= 9:
            phones.append(digits[-9:])
    return phones


def parse_amounts(text: str) -> list[float]:
    """Parse plausible money amounts, excluding phone-number digit runs and '5k' forms."""
    t = to_ascii_digits(text or "")
    # remove phone-like runs so they are not read as amounts
    t = _PHONE_RE.sub(" ", t)
    amounts: list[float] = []
    # 5k / 5K -> 5000
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*[kK]\b", t):
        amounts.append(float(m.group(1)) * 1000)
    t_wo_k = re.sub(r"\d+(?:\.\d+)?\s*[kK]\b", " ", t)
    for m in re.finditer(r"\d{1,3}(?:,\d{3})+|\d+", t_wo_k):
        raw = m.group(0).replace(",", "")
        if len(raw) >= 7:  # too long to be a campaign-era money amount; likely an id
            continue
        try:
            val = float(raw)
        except ValueError:
            continue
        if val >= 1:
            amounts.append(val)
    # de-dup preserving order
    seen: set[float] = set()
    uniq = []
    for a in amounts:
        if a not in seen:
            seen.add(a)
            uniq.append(a)
    return uniq


_CLOCK_RE = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*([ap])\.?m\.?\b", re.IGNORECASE)
_HHMM_RE = re.compile(r"\b(\d{1,2}):(\d{2})\b")


def parse_complaint_hours(text: str) -> list[int]:
    """Extract referenced clock hours (0-23) from the complaint for weak time matching."""
    t = to_ascii_digits(text or "")
    hours: list[int] = []
    for m in _CLOCK_RE.finditer(t):
        h = int(m.group(1)) % 12
        if m.group(3).lower() == "p":
            h += 12
        hours.append(h)
    for m in _HHMM_RE.finditer(t):
        h = int(m.group(1))
        if 0 <= h <= 23:
            hours.append(h)
    return hours


def _timestamp_hour(ts: str) -> Optional[int]:
    m = re.search(r"T(\d{2}):", ts or "")
    return int(m.group(1)) if m else None


# --- Candidate scoring ------------------------------------------------------

# keyword -> txn type hints used for both scoring and case classification
TYPE_KEYWORDS = {
    "transfer": ["sent", "send", "transfer", "wrong number", "পাঠ", "ট্রান্সফার",
                 "pathai", "pathay", "pathaisi", "disi", "diyechi", "send korsi"],
    "payment": ["paid", "pay", "payment", "recharge", "bill", "bkash payment", "বিল", "পেমেন্ট",
                "রিচার্জ", "bill disi", "recharge korsi", "payment korsi"],
    "cash_in": ["cash in", "cash-in", "deposit", "agent", "ক্যাশ ইন", "এজেন্ট", "cash korsi", "joma"],
    "cash_out": ["cash out", "cash-out", "withdraw", "ক্যাশ আউট", "tulsi", "tulesi"],
    "settlement": ["settle", "settlement", "merchant", "সেটেলমেন্ট"],
    "refund": ["refund", "ফেরত", "ferot", "back den", "taka back"],
}


def score_transactions(complaint: str, txns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Score each transaction against the complaint. Higher = more likely the referent."""
    amounts = parse_amounts(complaint)
    phones = extract_phones(complaint)
    hours = parse_complaint_hours(complaint)
    lc = (complaint or "").lower()

    scored = []
    for txn in txns:
        score = 0.0
        reasons = []
        amt = txn.get("amount")
        if amt is not None and amounts:
            if any(abs(amt - a) < 0.01 for a in amounts):
                score += 3.0
                reasons.append("amount_exact")
            elif any(abs(amt - a) <= max(1.0, a * 0.05) for a in amounts):
                score += 1.0
                reasons.append("amount_near")
        # phone / counterparty match
        cp_digits = re.sub(r"\D", "", to_ascii_digits(txn.get("counterparty", "")))
        if phones and cp_digits and any(cp_digits.endswith(p) for p in phones):
            score += 2.0
            reasons.append("counterparty_match")
        # type keyword match
        ttype = txn.get("type")
        if ttype and any(kw in lc for kw in TYPE_KEYWORDS.get(ttype, [])):
            score += 1.0
            reasons.append("type_keyword")
        # time hour proximity
        th = _timestamp_hour(txn.get("timestamp", ""))
        if th is not None and hours and any(abs(th - h) <= 1 for h in hours):
            score += 1.0
            reasons.append("time_match")
        # failed/deducted signal
        if txn.get("status") == "failed" and any(
            w in lc for w in ["failed", "deduct", "কাটা", "ব্যর্থ", "fail", "kete", "katse"]
        ):
            score += 1.0
            reasons.append("status_failed")
        scored.append({"txn": txn, "score": score, "reasons": reasons})

    # recency tie-breaker (newer slightly higher)
    order = sorted(range(len(txns)), key=lambda i: txns[i].get("timestamp", ""))
    for rank, i in enumerate(order):
        scored[i]["score"] += rank * 0.01
    scored.sort(key=lambda s: s["score"], reverse=True)
    return scored


def pick_relevant(scored: list[dict[str, Any]]) -> tuple[Optional[str], bool]:
    """Choose the relevant transaction id.

    Returns (transaction_id_or_None, ambiguous). ambiguous=True means multiple
    transactions matched about equally well -> caller should treat as insufficient.
    """
    if not scored:
        return None, False
    best = scored[0]
    if best["score"] < 3.0:
        # no strong signal (need at least an exact amount or counterparty+something)
        return None, False
    if len(scored) > 1:
        second = scored[1]
        # near-tie on a meaningful score => genuinely ambiguous
        if best["score"] - second["score"] < 1.0 and second["score"] >= 3.0:
            return None, True
    return best["txn"]["transaction_id"], False
