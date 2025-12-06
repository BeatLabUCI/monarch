# Use a slim Python image
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies first (rarely changes, good for caching)
RUN apt-get update && apt-get install -y build-essential && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy setup files for the monarch package (including LICENSE and README for metadata)
COPY setup.py pyproject.toml LICENSE README.md ./
COPY src/ ./src/

# Install build dependencies and the monarch package
RUN pip install --no-cache-dir hatchling && \
    pip install --no-cache-dir .

# Copy the rest of the application (notebooks, docs, etc.)
COPY monarch-docs/ ./monarch-docs/

# Expose the port Voilà will run on (Render sets PORT env var)
EXPOSE $PORT

# Set environment variable to bind to all interfaces
ENV JUPYTER_IP=0.0.0.0

# Enable Numba caching to speed up JIT compilation on subsequent runs
ENV NUMBA_CACHE_DIR=/app/.numba_cache
ENV NUMBA_THREADING_LAYER=workqueue
RUN mkdir -p /app/.numba_cache

# Run Voilà with Render's dynamic port
# Bind to 0.0.0.0 so Render can detect the open port
# Optimizations: pool size for handling multiple users, enable request caching
CMD sh -c "voila monarch-docs/docs/monarch_starter_interactive.ipynb \
    --port=$PORT \
    --no-browser \
    --Voila.ip=0.0.0.0 \
    --template=material \
    --pool_size=4 \
    --VoilaConfiguration.http_keep_alive_timeout=120 \
    --show_tracebacks=True"
