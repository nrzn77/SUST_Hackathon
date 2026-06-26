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

# romanised-Bangla markers: if present, a text in Latin script is likely Banglish,
# so we must not "correct" a declared bn down to en.
_BANGLISH_MARKERS = (
    "pathai", "pathay", "disi", "diyechi", "korsi", "korechi", "taka", "ferot",
    "bhul", "vul", "kore", "ami", "amar", "hoise", "hoyni", "nai", "den", "bhai",
    "ekta", "kintu", "kete", "chaise", "hajar", "joma",
)


def _sniff(text: str) -> str:
    bn = len(_BENGALI_RANGE.findall(text or ""))
    lat = len(_LATIN_RANGE.findall(text or ""))
    if bn == 0:
        return "en"
    if lat == 0:
        return "bn"
    return "mixed" if lat > bn * 0.3 else "bn"


def detect_language(text: str, declared: Optional[str]) -> str:
    """Trust the declared language, but override an obvious harness mislabel.

    The script is strong evidence: if a complaint is declared 'en' yet is written
    largely in Bangla script, the customer reply should be Bangla (and vice-versa).
    Banglish (romanised Bangla) is protected so a declared 'bn' is not wrongly
    downgraded to 'en'.
    """
    text = text or ""
    bn = len(_BENGALI_RANGE.findall(text))
    lat = len(_LATIN_RANGE.findall(text))
    detected = _sniff(text)
    if declared not in ("en", "bn", "mixed"):
        return detected
    # declared English but the script is dominantly Bangla -> reply in Bangla
    if declared == "en" and bn > 0 and bn >= lat:
        return "bn"
    # declared Bangla but no Bangla script and no Banglish cues -> it is really English
    if declared == "bn" and bn == 0 and not any(m in text.lower() for m in _BANGLISH_MARKERS):
        return "en"
    return declared


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


# number words (English + Banglish) for amounts written in words ("five thousand",
# "pnach hajar", "dui lakh").
_NUM_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "hundred": 100,
    "ek": 1, "dui": 2, "tin": 3, "char": 4, "panch": 5, "pnach": 5, "pach": 5,
    "paanch": 5, "choy": 6, "chhoy": 6, "sat": 7, "saat": 7, "aat": 8, "ath": 8,
    "noy": 9, "noi": 9, "dosh": 10, "dos": 10,
}
_SCALE_WORDS = {
    "hundred": 100, "sho": 100, "shoto": 100, "thousand": 1000, "hajar": 1000,
    "hazar": 1000, "lakh": 100000, "lac": 100000, "lakhs": 100000, "million": 1000000,
}
_WORD_AMOUNT_RE = re.compile(
    r"\b(" + "|".join(sorted(_NUM_WORDS, key=len, reverse=True)) + r")\s+("
    + "|".join(sorted(_SCALE_WORDS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def _parse_word_amounts(text: str) -> list[float]:
    """Parse amounts spelled out in words, e.g. 'five thousand' -> 5000."""
    out: list[float] = []
    for m in _WORD_AMOUNT_RE.finditer(text or ""):
        n = _NUM_WORDS.get(m.group(1).lower())
        scale = _SCALE_WORDS.get(m.group(2).lower())
        if n and scale:
            out.append(float(n * scale))
    return out


# multiplier words: k / hajar / hazar / thousand -> x1000 ; lakh / lac -> x100000
_THOUSAND_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(k|hajar|hazar|hajr|thousand)\b", re.IGNORECASE)
_LAKH_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(lakh|lac|lakhs|lacs)\b", re.IGNORECASE)
_MULT_STRIP_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(k|hajar|hazar|hajr|thousand|lakh|lac|lakhs|lacs)\b", re.IGNORECASE)


def parse_amounts(text: str) -> list[float]:
    """Parse plausible money amounts.

    Handles bare digits, comma groups, '5k', and Banglish/English multiplier words
    ('5 hajar' -> 5000, '2 lakh' -> 200000). Phone-number digit runs are excluded.
    """
    t = to_ascii_digits(text or "")
    # remove phone-like runs so they are not read as amounts
    t = _PHONE_RE.sub(" ", t)
    amounts: list[float] = _parse_word_amounts(t)
    for m in _THOUSAND_RE.finditer(t):
        amounts.append(float(m.group(1)) * 1000)
    for m in _LAKH_RE.finditer(t):
        amounts.append(float(m.group(1)) * 100000)
    # strip multiplier forms so the bare-number pass does not re-read the leading digits
    t = _MULT_STRIP_RE.sub(" ", t)
    for m in re.finditer(r"\d{1,3}(?:,\d{3})+|\d+", t):
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


_TODAY_RE = re.compile(r"\b(today|aaj|aj|ajke|ajk)\b|আজ", re.IGNORECASE)
_YDAY_RE = re.compile(r"\b(yesterday|gotokal|gotkal|gotokaal|kalke|kal)\b|গতকাল|last night", re.IGNORECASE)


def resolve_day_reference(text: str, txns: list[dict[str, Any]]) -> Optional[str]:
    """Map a relative day word to a concrete date (YYYY-MM-DD) from the history.

    'today' -> the most recent date in the history; 'yesterday' -> the date before
    that. Returns None when no day word is present or no dates are available. This
    only disambiguates among same-amount transactions; it never overrides amounts.
    """
    dates = sorted({t["timestamp"][:10] for t in txns if t.get("timestamp")}, reverse=True)
    if not dates:
        return None
    if _TODAY_RE.search(text or ""):
        return dates[0]
    if _YDAY_RE.search(text or ""):
        return dates[1] if len(dates) > 1 else dates[0]
    return None


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
    target_date = resolve_day_reference(complaint, txns)
    lc = (complaint or "").lower()

    scored = []
    for txn in txns:
        score = 0.0
        reasons = []
        # strongest signal: the complaint names the transaction id directly
        tid = str(txn.get("transaction_id") or "")
        if tid and tid.lower() in lc:
            score += 6.0
            reasons.append("id_mentioned")
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
        # day-word proximity (today/yesterday) — must clearly outweigh the recency
        # tiebreak so it can actually disambiguate same-amount txns on different days
        if target_date and txn.get("timestamp", "")[:10] == target_date:
            score += 1.5
            reasons.append("day_match")
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
