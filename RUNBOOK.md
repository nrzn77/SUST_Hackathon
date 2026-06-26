# RUNBOOK — QueueStorm Investigator

A stranger can copy-paste these steps to bring the service up. No guessing required.

## Option A — Local (Python 3.12)

```bash
# from the repo root
python -m venv .venv

# activate
.venv\Scripts\activate          # Windows PowerShell/CMD
# source .venv/bin/activate      # macOS / Linux

pip install -r requirements.txt

# (optional) enable Gemini reasoning; without this it runs on the rule engine
cp .env.example .env             # then put your key in GEMINI_API_KEY
# get a free key at https://aistudio.google.com/apikey

uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Verify:

```bash
curl http://localhost:8000/health
# {"status":"ok"}

curl -X POST http://localhost:8000/analyze-ticket \
  -H "content-type: application/json" \
  -d '{"ticket_id":"TKT-001","complaint":"I sent 5000 taka to a wrong number around 2pm","language":"en","transaction_history":[{"transaction_id":"TXN-9101","timestamp":"2026-04-14T14:08:22Z","type":"transfer","amount":5000,"counterparty":"+8801719876543","status":"completed"}]}'
```

## Option B — Docker

```bash
docker build -t queuestorm-investigator .

# without LLM (rule engine only):
docker run -p 8000:8000 queuestorm-investigator

# with Gemini:
docker run -p 8000:8000 -e GEMINI_API_KEY=your_key_here queuestorm-investigator
```

The container honours `$PORT` (defaults to 8000), so it works as-is on Render/Railway/Fly.

## Option C — Deploy to Render (live URL)

1. Push this repo to GitHub.
2. In Render: **New → Web Service → Build from a repository**, pick this repo.
   Render auto-detects `render.yaml` (Docker runtime, health check `/health`).
3. (Optional) In **Environment**, add `GEMINI_API_KEY`.
4. Deploy. Your base URL exposes `GET /health` and `POST /analyze-ticket`.

## Run the tests

```bash
python -m pytest -q
```

## Troubleshooting

- **Port already in use:** change `--port` (local) or the published port (Docker).
- **`google.genai` import errors:** they are caught; the service falls back to the rule engine.
- **Gemini 429 / rate limit:** expected on the free tier under load; the service auto-falls back.
