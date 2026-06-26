"""FastAPI entrypoint. Exposes GET /health and POST /analyze-ticket.

HTTP status policy (Section 4.1):
  200 valid analysis
  400 malformed input (invalid JSON, missing/typed-wrong required fields)
  422 schema valid but semantically invalid (e.g. empty complaint)
  500 internal error (safe message only — never a stack trace/secret)
The service must never crash on bad input.
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from . import pipeline, reason_llm
from .schemas import TicketRequest

logger = logging.getLogger("queuestorm")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="QueueStorm Investigator", version="1.0.0")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def root():
    return {"service": "QueueStorm Investigator", "llm_enabled": reason_llm.available()}


@app.post("/analyze-ticket")
async def analyze_ticket(request: Request):
    # 1. parse JSON body
    try:
        body = await request.json()
    except Exception:
        return _error(400, "Malformed request body: invalid JSON.")
    if not isinstance(body, dict):
        return _error(400, "Malformed request body: expected a JSON object.")

    # 2. schema validation (missing/typed-wrong required fields -> 400)
    try:
        req = TicketRequest(**body)
    except ValidationError:
        return _error(400, "Invalid request: 'ticket_id' and 'complaint' are required string fields.")
    except Exception:
        return _error(400, "Invalid request payload.")

    # 3. semantic validation (422)
    if not req.complaint or not req.complaint.strip():
        return _error(422, "The 'complaint' field must not be empty.", ticket_id=req.ticket_id)
    if not req.ticket_id or not req.ticket_id.strip():
        return _error(422, "The 'ticket_id' field must not be empty.")

    # 4. analyze (never crash -> 500 with safe message)
    try:
        result = pipeline.analyze(req)
        return JSONResponse(status_code=200, content=result)
    except Exception:
        logger.exception("analyze-ticket failed for ticket_id=%s", req.ticket_id)
        return _error(500, "Internal error while analyzing the ticket.", ticket_id=req.ticket_id)


def _error(status: int, message: str, ticket_id: str | None = None) -> JSONResponse:
    body = {"error": message}
    if ticket_id is not None:
        body["ticket_id"] = ticket_id
    return JSONResponse(status_code=status, content=body)
