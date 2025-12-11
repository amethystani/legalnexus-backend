FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY config/requirements.txt config/

# Install Python dependencies
RUN pip install --no-cache-dir -r config/requirements.txt

# Copy source code
COPY src/ src/
COPY data/ data/
COPY config/ config/

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python", "src/ui/app.py"]
