FROM python:3.11-slim

# Prevent Python from buffering stdout/stderr and writing pyc bytecode files
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8501 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Install system utilities: curl (for health checks) and build-essential (if needed for compilation)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install pinned repository dependencies first for Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir streamlit catboost

# Create non-root user for secure container execution
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/output /app/models /app/.tmp && \
    chown -R appuser:appuser /app

# Copy application source code (excluding items specified in .dockerignore)
COPY --chown=appuser:appuser . .

USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl --fail "http://localhost:${PORT:-${STREAMLIT_SERVER_PORT:-8501}}/_stcore/health" || exit 1

CMD ["sh", "-c", "exec streamlit run app.py --server.port=${PORT:-${STREAMLIT_SERVER_PORT:-8501}} --server.address=0.0.0.0"]
