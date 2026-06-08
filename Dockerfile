FROM mcr.microsoft.com/playwright/python:v1.60.0-jammy

# Working directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy project files
COPY . /app

# Ensure non-root pwuser owns the app directory
RUN chown -R pwuser:pwuser /app

ENV PYTHONUNBUFFERED=1
ENV LOG_DIR=/app/logs

# Run as the non-root Playwright user
USER pwuser

# Optional port for healthchecks
EXPOSE 8080

CMD ["python", "-u", "multi_agent_orchestrator.py"]
