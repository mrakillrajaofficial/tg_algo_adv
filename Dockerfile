FROM python:3.11-slim

# Install system dependencies required by some Python packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        FROM mcr.microsoft.com/playwright/python:latest

        # Use Playwright image which already includes browser binaries and necessary libs.
        WORKDIR /app

        # Copy and install Python dependencies
        COPY requirements.txt /app/requirements.txt
        RUN pip install --no-cache-dir -r /app/requirements.txt

        # Copy project and set correct ownership (Playwright image uses 'pwuser')
        COPY . /app
        RUN chown -R pwuser:pwuser /app

        ENV PYTHONUNBUFFERED=1
        ENV LOG_DIR=/app/logs

        # Run as the non-root Playwright user
        USER pwuser

        # Expose a port for healthchecks if desired
        EXPOSE 8080

        CMD ["python", "-u", "multi_agent_orchestrator.py"]
