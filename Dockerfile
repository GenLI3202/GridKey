# GridKey Optimizer Service - Production Docker Image
# Using solver-optimized image for better MILP performance
FROM python:3.11-slim

# Metadata
LABEL maintainer="GridKey Team"
LABEL version="1.0"
LABEL description="BESS Optimizer Service with Renewable Integration"

# Set working directory
WORKDIR /app

# Install system dependencies for solvers AND curl for health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgmp-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for layer caching)
COPY requirements.txt requirements-api.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r requirements-api.txt

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY data/ ./data/

# Set Python path
ENV PYTHONPATH=/app

# Force HiGHS solver for production (no license required)
# CPLEX/Gurobi require licenses that won't work in containers
ENV GRIDKEY_SOLVER=highs

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run FastAPI server
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
