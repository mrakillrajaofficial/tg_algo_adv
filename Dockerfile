FROM python:3.11-slim

# Install system dependencies required by some Python packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        wget \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create app directory and non-root user
WORKDIR /app
RUN useradd --create-home --shell /bin/bash appuser

# Copy dependencies first to leverage Docker cache
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application code
COPY . /app
RUN chown -R appuser:appuser /app

ENV PYTHONUNBUFFERED=1
ENV LOG_DIR=/app/logs

USER appuser

# Expose a port for healthchecks if needed (not required for background bots)
EXPOSE 8080

# Run the orchestrator. Use unbuffered python for logs to appear in Railway.
CMD ["python", "-u", "multi_agent_orchestrator.py"]
