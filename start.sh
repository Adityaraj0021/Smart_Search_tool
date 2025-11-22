#!/bin/sh
set -e

# Load .env if present (simple loader)
if [ -f /app/.env ]; then
  export $(grep -v '^#' /app/.env | xargs)
fi

# Start FastAPI backend
echo "Starting FastAPI on 0.0.0.0:8000"
uvicorn Smart_Search_tool.fastapi_backend:app --host 0.0.0.0 --port 8000 &

# Start Gradio frontend (foreground)
echo "Starting Gradio frontend"
python /app/Smart_Search_tool/app.py
