FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    OPENAI_API_KEY=""

WORKDIR /app

# Install system deps needed by some ML libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl ffmpeg libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Copy dependency file and install
COPY Smart_Search_tool/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy project files from the Smart_Search_tool directory into the container app folder
# this ensures start.sh is placed at /app/start.sh
COPY Smart_Search_tool/. /app/

# Make the start script executable
RUN chmod +x /app/start.sh

EXPOSE 8000 7861

CMD ["/app/start.sh"]
