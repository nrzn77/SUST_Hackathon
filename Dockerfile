FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY sample_cases.json ./sample_cases.json

EXPOSE 8000

# Honour $PORT (Render/Railway inject it). Single worker keeps memory low; the LLM
# call is offloaded to a thread pool, so the event loop is not blocked.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
