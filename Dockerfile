FROM python:3.11-slim

# Install system dependencies required by some Python packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        wget \
        ca-certificates \
        curl \
        fonts-liberation \
        libnss3 \
        libatk1.0-0 \
        libatk-bridge2.0-0 \
        libcups2 \
        libxss1 \
        libasound2 \
        libgbm1 \
        libx11-xcb1 \
        libgtk-3-0 \
        libxcomposite1 \
        libxdamage1 \
        libxrandr2 \
        libpango-1.0-0 \
        libpangocairo-1.0-0 \
        libxrender1 \
        libxcb1 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Create app directory and non-root user
WORKDIR /app
RUN useradd --create-home --shell /bin/bash appuser

# Copy dependencies first to leverage Docker cache

# Copy requirements and install Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Install Playwright browsers into a shared path so the non-root user can use them
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright
RUN mkdir -p /ms-playwright \
    && python -m playwright install --with-deps \
    && rm -rf /root/.cache/ms-playwright/* || true

# Copy application code

# Copy application code and give ownership to appuser
COPY . /app
RUN chown -R appuser:appuser /app /ms-playwright

ENV PYTHONUNBUFFERED=1
ENV LOG_DIR=/app/logs


USER appuser

# Expose a port for healthchecks if needed (not required for background bots)
EXPOSE 8080

# Run the orchestrator. Use unbuffered python for logs to appear in Railway.
CMD ["python", "-u", "multi_agent_orchestrator.py"]
