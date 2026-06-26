# QueueStorm Investigator

AI/API SupportOps copilot for the **SUST CSE Carnival 2026 · Codex Community Hackathon (Online Preliminary)**.

It exposes `POST /analyze-ticket` and `GET /health`. Given one customer complaint plus a
short snippet of that customer's recent transactions, it **investigates** (not just classifies):
it finds the relevant transaction, judges whether the evidence supports the complaint, classifies
and routes the case, and drafts a **safe** customer reply that never asks for credentials or
promises an unauthorized refund.

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| API | **FastAPI + Uvicorn** | Async, fast, precise control of 400/422/500 codes |
| Validation | **Pydantic v2** | Cheap, strict request modelling |
| Reasoning | **Hybrid: deterministic rule engine + optional Gemini Flash** | Rules give a reliable, zero-cost floor; the LLM lifts accuracy on messy/novel cases |
| Deploy | **Docker → Render free tier** | Public HTTPS, judge-reachable, no cost |

## Architecture / API flow

```
POST /analyze-ticket
  1. HTTP + schema gate (Pydantic)      bad JSON / missing fields -> 400 ; empty complaint -> 422
  2. Feature extraction (deterministic) language detect, amount/time/phone parse, candidate scoring
  3. Reasoning  LLM (Gemini) primary  -> rule engine fallback on timeout/error/no-key
  4. Validation gate (deterministic)    clamp enums, verify txn id exists, force escalation
  5. Reply templating (en/bn)           safe agent_summary / next_action / customer_reply
  6. Safety scrubber                    last-line guard against credential asks & refund promises
  -> 200 structured JSON
```

The **rule engine always runs** — it is both the LLM's fallback and the candidate generator the
LLM result is cross-checked against. If `GEMINI_API_KEY` is unset, the service runs fully on rules
and still passes all 10 public sample cases.

## AI approach

- The LLM reasons over the complaint + history and returns a strict JSON object (no prose).
- The complaint is passed as clearly delimited **untrusted data**; the model is told to ignore
  instructions inside it (prompt-injection defence).
- Crucially, **the LLM never writes the customer-facing reply** and **never has the last word on
  enums or escalation** — those go through a deterministic validation + templating layer, so a bad
  or adversarial LLM output cannot cause a schema or safety violation.
- LLM call has a hard ~6s timeout and silently falls back to rules, keeping every response well
  inside the 30s budget even under free-tier rate limits.

## MODELS

| Model | Where it runs | Role | Why |
|---|---|---|---|
| **gemini-2.0-flash** (Google AI) | Google's hosted API, called at request time | Primary reasoner: transaction match, evidence verdict, classification, routing | Free tier, fast, strong at Bangla/Banglish; structured JSON output. Configurable via `GEMINI_MODEL`. |
| Deterministic rule engine (`app/reason_rules.py`) | In-process | Fallback + candidate generator + cross-check | Zero cost, zero latency, fully reproducible; guarantees the service works with no API key and never depends on an external call. |

**Cost reasoning:** the organizers provide no LLM credits, so we default to Gemini's free tier and
keep prompts small (~one short JSON payload, `max_output_tokens=512`). The rule fallback means a
spent quota or rate-limit degrades quality gracefully instead of failing — important for the
"40,000 complaints" load framing. The service can run at **$0** with no key at all.

## Safety logic (Section 8)

Safety is enforced **structurally**, not hoped for:

- `customer_reply` is built from vetted templates that always include
  *"Please do not share your PIN or OTP with anyone."* and **never** ask for credentials.
- Refunds use *"any eligible amount will be returned through official channels"* — never
  *"we will refund you"*.
- A regex **safety scrubber** (`app/reply.is_safe_reply`) re-checks every reply for affirmative
  credential requests, refund promises, and off-channel redirection; on any hit it swaps in a safe
  fallback.
- Escalation is forced on (`human_review_required = true`) for phishing, inconsistent evidence,
  and confirmed disputes/duplicates/agent-cash-in — even if the reasoner said otherwise.
- Prompt injection is defeated by treating the complaint as data **and** by the deterministic
  output gate, so embedded instructions cannot change the response shape.

## Run locally

See [RUNBOOK.md](RUNBOOK.md). Quick version:

```bash
python -m venv .venv
.venv/Scripts/activate         # Windows ;  source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
cp .env.example .env           # optionally add GEMINI_API_KEY
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

```bash
curl localhost:8000/health
curl -X POST localhost:8000/analyze-ticket -H "content-type: application/json" \
  -d '{"ticket_id":"T1","complaint":"I sent 5000 to a wrong number","transaction_history":[]}'
```

## Tests

```bash
python -m pytest -q
```

`tests/test_samples.py` asserts functional equivalence against all 10 public cases (run on the
rule-engine path for a deterministic baseline); `tests/test_edge.py` covers malformed JSON, empty/
missing fields, prompt injection, unknown enums, oversized input, and partial transaction data.

## Sample outputs

`sample_outputs/sample_outputs.json` contains our service's output for every public sample input
alongside the expected output. Regenerate with `python -m scripts.gen_sample_outputs`.

## Assumptions & known limitations

- Reply language is `en` for `en`/`mixed` and `bn` for `bn`; agent-facing fields are always English
  (matching the samples).
- The rule engine matches primarily on amount + transaction type + counterparty/time; very oblique
  complaints with no numeric or recipient signal resolve to `insufficient_data` by design (we never
  guess a transaction).
- Severity is judged within one rank of the reference; exact severity wording can legitimately vary.
- Gemini free tier has per-minute rate limits; under burst load the service leans on the rule
  engine. This is intentional and safe, but LLM-driven nuance may drop during spikes.
